use std::arch::x86_64::_pext_u64;

use quadrank::quad::RankerT;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::{FmIndex, bwt::BWT};

pub struct QuadFm<Rank: RankerT> {
    n: usize,
    ranker: Rank,
    sentinel: usize,
    occ: Vec<usize>,
    prefix: usize,
    prefix_lookup: Vec<(usize, usize)>,
}

impl<Ranker: RankerT> QuadFm<Ranker> {
    fn query(&self, text: &[u8]) -> (usize, usize) {
        let mut s = 0;
        let mut t = self.n + 1;
        let mut steps = 0;
        for &c in text.iter().rev() {
            steps += 1;
            let occ = self.occ[c as usize];
            let ranks_s = self
                .ranker
                .rank1(s as usize - (s > self.sentinel) as usize, c);
            s = occ + ranks_s as usize;
            let ranks_t = self
                .ranker
                .rank1(t as usize - (t > self.sentinel) as usize, c);
            t = occ + ranks_t as usize;
            if s == t {
                return (s, t);
            }
        }
        let _ = steps;
        (s, t)
    }

    #[allow(unused)]
    #[inline(always)]
    pub fn query_batch_interleaved<const B: usize>(
        &self,
        texts: &[Vec<u8>; B],
    ) -> [(usize, usize); B] {
        let mut s = [0; B];
        let mut t = [self.n + 1; B];
        let mut steps = [0; B];

        if self.prefix > 0 {
            for i in 0..B {
                let suffix = texts[i].last_chunk().unwrap();
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

                if s[i] == t[i] || text_idx >= texts[i].len() {
                    // swappop index i
                    active[idx] = active[num_alive - 1];
                    num_alive -= 1;

                    // Note: idx is not incremented here.
                    continue;
                }
                // self.rank
                //     .prefetch(s[i] as usize - (s[i] > self.sentinel) as usize);
                // self.rank
                //     .prefetch(t[i] as usize - (t[i] > self.sentinel) as usize);

                idx += 1;
            }

            if num_alive == 0 {
                break;
            }

            for idx in 0..num_alive {
                let i = active[idx] as usize;

                let c = unsafe {
                    *texts
                        .get_unchecked(i)
                        .get_unchecked(texts[i].len() - 1 - text_idx)
                };

                steps[i] += 1;
                let occ = self.occ[c as usize];
                if false {
                    let ranks_s = self
                        .ranker
                        .rank1(s[i] as usize - (s[i] > self.sentinel) as usize, c);
                    s[i] = occ + ranks_s as usize;
                    let ranks_t = self
                        .ranker
                        .rank1(t[i] as usize - (t[i] > self.sentinel) as usize, c);
                    t[i] = occ + ranks_t as usize;
                } else {
                    let (ranks_s, ranks_t) = self.ranker.count1x2(
                        s[i] as usize - (s[i] > self.sentinel) as usize,
                        t[i] as usize - (t[i] > self.sentinel) as usize,
                        c,
                    );
                    s[i] = occ + ranks_s as usize;
                    t[i] = occ + ranks_t as usize;
                }

                if idx > 0 {
                    let i = active[idx - 1] as usize;
                    if s[i] < t[i] && text_idx + 1 < texts[i].len() {
                        let c = unsafe {
                            *texts
                                .get_unchecked(i)
                                .get_unchecked(texts[i].len() - 1 - text_idx)
                        };
                        self.ranker
                            .prefetch1(s[i] as usize - (s[i] > self.sentinel) as usize, c);
                        self.ranker
                            .prefetch1(t[i] as usize - (t[i] > self.sentinel) as usize, c);
                    }
                }
            }
            {
                let i = active[num_alive - 1] as usize;
                if s[i] < t[i] && text_idx + 1 < texts[i].len() {
                    let c = unsafe {
                        *texts
                            .get_unchecked(i)
                            .get_unchecked(texts[i].len() - 1 - text_idx)
                    };
                    self.ranker
                        .prefetch1(s[i] as usize - (s[i] > self.sentinel) as usize, c);
                    self.ranker
                        .prefetch1(t[i] as usize - (t[i] > self.sentinel) as usize, c);
                }
            }
            text_idx += 1;
        }
        std::array::from_fn(|i| (steps[i], (t[i] - s[i]) as usize))
    }

    /// Process a total of T queries, constantly keeping B active.
    #[allow(unused)]
    #[inline(always)]
    pub fn query_all<const B: usize, const T: usize>(
        &self,
        input_texts: &[Vec<u8>; T],
        mut callback: impl FnMut(usize, usize, usize),
    ) {
        let mut s = [0; B];
        let mut t = [self.n + 1; B];
        let mut steps = [0; B];
        let mut texts: [&[u8]; B] = std::array::from_fn(|i| input_texts[i].as_slice());

        let mut active = [true; B];
        let mut next = B;
        let mut text_idx = [0; T];

        let mut done = 0;

        while done < T {
            for i in 0..B {
                if !active[i] {
                    continue;
                }

                if s[i] == t[i] || text_idx[i] >= texts[i].len() {
                    callback(steps[i], s[i], t[i]);
                    done += 1;
                    if next == T {
                        active[i] = false;
                        continue;
                    }
                    steps[i] = 0;
                    s[i] = 0;
                    t[i] = self.n + 1;
                    text_idx[i] = 0;
                    texts[i] = &input_texts[next];
                    next += 1;
                }
                let c = unsafe {
                    *texts
                        .get_unchecked(i)
                        .get_unchecked(texts[i].len() - 1 - text_idx[i])
                };
                self.ranker
                    .prefetch1(s[i] as usize - (s[i] > self.sentinel) as usize, c);
                self.ranker
                    .prefetch1(t[i] as usize - (t[i] > self.sentinel) as usize, c);
            }

            for i in 0..B {
                if !active[i] {
                    continue;
                }

                let c = unsafe {
                    *texts
                        .get_unchecked(i)
                        .get_unchecked(texts[i].len() - 1 - text_idx[i])
                };

                steps[i] += 1;
                let occ = self.occ[c as usize];
                let ranks_s = self
                    .ranker
                    .rank1(s[i] as usize - (s[i] > self.sentinel) as usize, c);
                s[i] = occ + ranks_s as usize;
                let ranks_t = self
                    .ranker
                    .rank1(t[i] as usize - (t[i] > self.sentinel) as usize, c);
                t[i] = occ + ranks_t as usize;
                text_idx[i] += 1;
            }
        }
    }
}

impl<Ranker: RankerT> FmIndex for QuadFm<Ranker> {
    fn new_with_prefix(_text: &[u8], bwt: &BWT, prefix: usize) -> Self {
        let n = bwt.bwt.len();

        let ranker = Ranker::new_packed(&bwt.packed);
        let mut occ: Vec<usize> = bwt
            .bwt
            .par_iter()
            .fold_with(vec![0usize; 5], |mut occ, c| {
                occ[*c as usize + 1] += 1;
                occ
            })
            .reduce(|| vec![0; 5], |a, b| (0..5).map(|i| a[i] + b[i]).collect());
        // for sentinel
        occ[0] += 1;
        for i in 1..5 {
            occ[i] += occ[i - 1];
        }

        let mut fm = Self {
            n,
            ranker,
            sentinel: bwt.sentinel,
            occ,
            prefix: 0,
            prefix_lookup: Vec::new(),
        };

        // query every prefix of length `prefix` and store the (s, t) ranges.
        if prefix > 0 {
            assert!(prefix <= 8);
            let mut lookup = Vec::new();
            for i in 0..(1 << (2 * prefix)) {
                let mask: u64 = 0x0303030303030303;
                let bases = unsafe { std::arch::x86_64::_pdep_u64(i, mask) };
                let (s, t) = fm.query(&bases.to_le_bytes()[0..prefix]);
                lookup.push((s, t));
            }
            fm.prefix_lookup = lookup;
            fm.prefix = prefix;
        }
        fm
    }

    fn size(&self) -> usize {
        self.ranker.size()
            + std::mem::size_of_val(self.occ.as_slice())
            + std::mem::size_of_val(self.prefix_lookup.as_slice())
    }

    fn count(&self, text: &[u8]) -> usize {
        let (s, t) = self.query(text);
        t - s
    }

    const HAS_BATCH: bool = true;
    const HAS_PREFETCH: bool = true;

    #[inline(always)]
    fn count_batch<const B: usize, const PREFETCH: bool>(
        &self,
        texts: &[Vec<u8>; B],
    ) -> [usize; B] {
        let mut s = [0; B];
        let mut t = [self.n + 1; B];
        let mut steps = [0; B];

        if self.prefix > 0 {
            for i in 0..B {
                let suffix = texts[i].last_chunk().unwrap();
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

                if s[i] == t[i] || text_idx >= texts[i].len() {
                    // swappop index i
                    active[idx] = active[num_alive - 1];
                    num_alive -= 1;

                    // Note: idx is not incremented here.
                    continue;
                }
                if PREFETCH {
                    let c = unsafe {
                        *texts
                            .get_unchecked(i)
                            .get_unchecked(texts[i].len() - 1 - text_idx)
                    };
                    self.ranker
                        .prefetch1(s[i] as usize - (s[i] > self.sentinel) as usize, c);
                    self.ranker
                        .prefetch1(t[i] as usize - (t[i] > self.sentinel) as usize, c);
                }

                idx += 1;
            }

            if num_alive == 0 {
                break;
            }

            for idx in 0..num_alive {
                let i = active[idx] as usize;

                let c = unsafe {
                    *texts
                        .get_unchecked(i)
                        .get_unchecked(texts[i].len() - 1 - text_idx)
                };

                steps[i] += 1;
                let occ = self.occ[c as usize];
                let ranks_s = self
                    .ranker
                    .rank1(s[i] as usize - (s[i] > self.sentinel) as usize, c);
                s[i] = occ + ranks_s as usize;
                let ranks_t = self
                    .ranker
                    .rank1(t[i] as usize - (t[i] > self.sentinel) as usize, c);
                t[i] = occ + ranks_t as usize;
            }
            text_idx += 1;
        }
        let _ = steps;
        std::array::from_fn(|i| t[i] - s[i])
    }
}
