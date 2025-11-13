use std::arch::x86_64::{_pext_u32, _pext_u64};

use quadrank::ranker::RankerT;
use qwt::{RankQuad, WTSupport};

use crate::bwt::BWT;

type Rank = quadrank::QuadRank;
// type Rank = qwt::RSQVector256;

pub struct FM {
    n: usize,
    rank: Rank,
    sentinel: usize,
    occ: Vec<usize>,
    prefix: usize,
    prefix_lookup: Vec<(u32, u32)>,
}

impl FM {
    pub fn size(&self) -> usize {
        self.rank.size()
            + std::mem::size_of_val(self.occ.as_slice())
            + std::mem::size_of_val(self.prefix_lookup.as_slice())
    }

    pub fn new(bwt: &BWT) -> Self {
        Self::new_with_prefix(bwt, 0)
    }
    pub fn new_with_prefix(bwt: &BWT, prefix: usize) -> Self {
        let n = bwt.bwt.len();

        let dna_bwt = bwt
            .bwt
            .iter()
            .map(|&c| b"ACTG"[c as usize])
            .collect::<Vec<_>>();
        // eprintln!("bwt: {:?}", bwt.bwt);
        // eprintln!("dna_bwt: {:?}", str::from_utf8(&dna_bwt).unwrap());

        let rank = RankerT::new(&dna_bwt);
        let mut occ = vec![0; 5];
        for &c in &bwt.bwt {
            occ[c as usize + 1] += 1;
        }
        // for sentinel
        occ[0] += 1;
        for i in 1..5 {
            occ[i] += occ[i - 1];
        }
        eprintln!("sentinel: {}", bwt.sentinel);

        let mut fm = Self {
            n,
            rank,
            sentinel: bwt.sentinel,
            occ,
            prefix: 0,
            prefix_lookup: Vec::new(),
        };

        // query every prefix of length `prefix` and store the (s, t) ranges.
        if prefix > 0 {
            eprintln!("Building prefix table for length {prefix}");
            assert!(prefix <= 8);
            let mut lookup = Vec::new();
            let mut done = false;
            for i in 0..(1 << (2 * prefix)) {
                let mask: u64 = 0x0303030303030303;
                let bases = unsafe { std::arch::x86_64::_pdep_u64(i, mask) };
                let (_, (s, t)) = fm.query_range(&bases.to_le_bytes()[0..prefix]);
                lookup.push((s, t));
            }
            eprintln!(
                "prefix lookup size: {} kB",
                std::mem::size_of_val(lookup.as_slice()) / 1024
            );
            fm.prefix_lookup = lookup;
            fm.prefix = prefix;
        }
        fm
    }

    pub fn query_range(&self, text: &[u8]) -> (usize, (u32, u32)) {
        let mut s = 0;
        let mut t = self.n + 1;
        let mut steps = 0;
        for &c in text.iter().rev() {
            steps += 1;
            let occ = self.occ[c as usize];
            let ranks_s = self
                .rank
                .count1(s as usize - (s > self.sentinel) as usize, c);
            s = occ + ranks_s as usize;
            let ranks_t = self
                .rank
                .count1(t as usize - (t > self.sentinel) as usize, c);
            t = occ + ranks_t as usize;
            if s == t {
                return (steps, (s as u32, t as u32));
            }
        }
        (steps, (s as u32, t as u32))
    }

    pub fn query(&self, text: &[u8]) -> (usize, usize) {
        let (steps, (s, t)) = self.query_range(text);
        (steps, (t - s) as usize)
    }

    #[inline(always)]
    pub fn query_batch<const B: usize>(&self, text: &[Vec<u8>; B]) -> [(usize, usize); B] {
        let mut s = [0; B];
        let mut t = [self.n + 1; B];
        let mut steps = [0; B];

        if self.prefix > 0 {
            for i in 0..B {
                let suffix = text[i].last_chunk().unwrap();
                let val = u64::from_le_bytes(*suffix);
                let index = unsafe { _pext_u64(val, 0x0303030303030303) };
                let (ss, tt) = self.prefix_lookup[index as usize];
                steps[i] += 1;
                s[i] = ss as usize;
                t[i] = tt as usize;
            }
        }

        let mut num_alive = B;
        let mut active: [u8; B] = std::array::from_fn(|i| i as u8);

        let mut text_idx = self.prefix;
        loop {
            let mut idx = 0;
            while idx < num_alive {
                let i = active[idx] as usize;

                if s[i] == t[i] || text_idx >= text[i].len() {
                    // swappop index i
                    active[idx] = active[num_alive - 1];
                    num_alive -= 1;

                    // Note: idx is not incremented here.
                    continue;
                }
                self.rank
                    .prefetch(s[i] as usize - (s[i] > self.sentinel) as usize);
                self.rank
                    .prefetch(t[i] as usize - (t[i] > self.sentinel) as usize);

                idx += 1;
            }

            if num_alive == 0 {
                break;
            }

            for idx in 0..num_alive {
                let i = active[idx] as usize;

                let c = text[i][text[i].len() - 1 - text_idx];

                steps[i] += 1;
                let occ = self.occ[c as usize];
                let ranks_s = self
                    .rank
                    .count1(s[i] as usize - (s[i] > self.sentinel) as usize, c);
                s[i] = occ + ranks_s as usize;
                let ranks_t = self
                    .rank
                    .count1(t[i] as usize - (t[i] > self.sentinel) as usize, c);
                t[i] = occ + ranks_t as usize;
            }
            text_idx += 1;
        }
        std::array::from_fn(|i| (steps[i], (t[i] - s[i]) as usize))
    }

    #[inline(always)]
    pub fn query_batch_interleaved<const B: usize>(
        &self,
        text: &[Vec<u8>; B],
    ) -> [(usize, usize); B] {
        let mut s = [0; B];
        let mut t = [self.n + 1; B];
        let mut steps = [0; B];

        let mut num_alive = B;
        let mut active: [u8; B] = std::array::from_fn(|i| i as u8);

        let mut text_idx = 0;
        while num_alive > 0 {
            let mut idx = 0;

            while idx < num_alive {
                let i = active[idx] as usize;

                let c = text[i][text[i].len() - 1 - text_idx];

                steps[i] += 1;
                let occ = self.occ[c as usize];
                let ranks_s = self
                    .rank
                    .count1(s[i] as usize - (s[i] > self.sentinel) as usize, c);
                s[i] = occ + ranks_s as usize;
                let ranks_t = self
                    .rank
                    .count1(t[i] as usize - (t[i] > self.sentinel) as usize, c);
                t[i] = occ + ranks_t as usize;

                if s[i] == t[i] || text_idx + 1 >= text[i].len() {
                    // swappop index i
                    active[idx] = active[num_alive - 1];
                    num_alive -= 1;

                    // Note: idx is not incremented here.
                    continue;
                }
                self.rank
                    .prefetch(s[i] as usize - (s[i] > self.sentinel) as usize);
                self.rank
                    .prefetch(t[i] as usize - (t[i] > self.sentinel) as usize);
                idx += 1;
            }
            text_idx += 1;
        }
        std::array::from_fn(|i| (steps[i], (t[i] - s[i]) as usize))
    }

    /// Process a total of T queries, constantly keeping B active.
    #[inline(always)]
    pub fn query_all<const B: usize, const T: usize>(
        &self,
        text: &[Vec<u8>; T],
    ) -> [(usize, usize); T] {
        let mut s = [0; T];
        let mut t = [self.n + 1; T];
        let mut steps = [0; T];

        let mut active: [usize; B] = std::array::from_fn(|i| i);
        let mut next = B;
        let mut text_idx = [0; T];

        let mut done = 0;

        while done < T {
            for idx in 0..B {
                let mut i = active[idx] as usize;
                if i >= T {
                    continue;
                }

                if s[i] == t[i] || text_idx[i] >= text[i].len() {
                    if i < T {
                        done += 1;
                        active[idx] = next;
                        i = next;
                        next += 1;
                    }
                    if i >= T {
                        continue;
                    }
                }
                self.rank
                    .prefetch(s[i] as usize - (s[i] > self.sentinel) as usize);
                self.rank
                    .prefetch(t[i] as usize - (t[i] > self.sentinel) as usize);
            }

            for idx in 0..B {
                let i = active[idx] as usize;
                if i >= T {
                    continue;
                }

                let c = text[i][text[i].len() - 1 - text_idx[i]];

                steps[i] += 1;
                let occ = self.occ[c as usize];
                let ranks_s = self
                    .rank
                    .count1(s[i] as usize - (s[i] > self.sentinel) as usize, c);
                s[i] = occ + ranks_s as usize;
                let ranks_t = self
                    .rank
                    .count1(t[i] as usize - (t[i] > self.sentinel) as usize, c);
                t[i] = occ + ranks_t as usize;
                text_idx[i] += 1;
            }
        }
        std::array::from_fn(|i| (steps[i], (t[i] - s[i]) as usize))
    }
}
