use quadrank::ranker::RankerT;

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
        eprintln!("occ: {occ:?}");
        for i in 1..5 {
            occ[i] += occ[i - 1];
        }
        eprintln!("occ: {occ:?}");
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
            let ranks_s = self.rank.count(s as usize - (s > self.sentinel) as usize);
            s = occ + ranks_s[c as usize] as usize;
            let ranks_t = self.rank.count(t as usize - (t > self.sentinel) as usize);
            t = occ + ranks_t[c as usize] as usize;
            if s == t {
                return (steps, 0);
            }
        }
        (steps, (t - s) as usize)
    }

    pub fn query_batch<const B: usize>(&self, text: &[Vec<u8>; B]) -> [(usize, usize); B] {
        if B == 1 {
            return [self.query(&text[0]); B];
        }

        let mut s = [0; B];
        let mut t = [self.n + 1; B];
        let mut steps = [0; B];

        let mut alive = true;
        let mut idx = 0;
        while alive {
            alive = false;

            for i in 0..B {
                if s[i] == t[i] || idx >= text[i].len() {
                    continue;
                }
                self.rank
                    .prefetch(s[i] as usize - (s[i] > self.sentinel) as usize);
                self.rank
                    .prefetch(t[i] as usize - (t[i] > self.sentinel) as usize);
            }

            for i in 0..B {
                if s[i] == t[i] || idx >= text[i].len() {
                    continue;
                }
                alive = true;

                let c = text[i][text[i].len() - 1 - idx];

                steps[i] += 1;
                let occ = self.occ[c as usize];
                let ranks_s = self
                    .rank
                    .count(s[i] as usize - (s[i] > self.sentinel) as usize);
                s[i] = occ + ranks_s[c as usize] as usize;
                let ranks_t = self
                    .rank
                    .count(t[i] as usize - (t[i] > self.sentinel) as usize);
                t[i] = occ + ranks_t[c as usize] as usize;
            }
            idx += 1;
        }
        std::array::from_fn(|i| (steps[i], (t[i] - s[i]) as usize))
    }
}
