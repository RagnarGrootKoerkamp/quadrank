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
}

impl FM {
    pub fn size(&self) -> usize {
        self.rank.size() + self.occ.len() * size_of::<usize>()
    }

    pub fn new(bwt: &BWT) -> Self {
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

        Self {
            n,
            rank,
            sentinel: bwt.sentinel,
            occ,
        }
    }

    pub fn query(&self, text: &[u8]) -> (usize, usize) {
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
                return (steps, 0);
            }
        }
        (steps, (t - s) as usize)
    }

    #[inline(always)]
    pub fn query_batch<const B: usize>(&self, text: &[Vec<u8>; B]) -> [(usize, usize); B] {
        if B == 1 {
            return [self.query(&text[0]); B];
        }

        let mut s = [0; B];
        let mut t = [self.n + 1; B];
        let mut steps = [0; B];

        let mut num_alive = B;
        let mut active: [u8; B] = std::array::from_fn(|i| i as u8);

        let mut text_idx = 0;
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
        if B == 1 {
            return [self.query(&text[0]); B];
        }

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
