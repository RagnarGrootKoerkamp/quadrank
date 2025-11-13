use quadrank::ranker::RankerT;

use crate::bwt::BWT;

// type Rank = quadrank::QuadRank;
type Rank = qwt::RSQVector256;

pub struct FM {
    n: usize,
    rank: Rank,
    sentinel: usize,
    occ: Vec<usize>,
}

impl FM {
    pub fn new(text: &[u8], bwt: &BWT) -> Self {
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
        // 1: correct for sentinel
        let mut s = 0;
        let mut t = self.n + 1;
        // eprintln!("s, t: {s:>5}..{t:>5}");
        let mut steps = 0;
        // eprintln!("c:   s {s:>2} t {t:>2}");
        for &c in text.iter().rev() {
            // eprintln!();
            steps += 1;
            let occ = self.occ[c as usize];
            // eprintln!("char: {c} -> occ {occ}");
            let ranks_s = self.rank.count(s as usize - (s > self.sentinel) as usize);
            // eprintln!("s {s:>2} -> ranks: {:?}", ranks_s);
            s = occ + ranks_s[c as usize] as usize;
            let ranks_t = self.rank.count(t as usize - (t > self.sentinel) as usize);
            // eprintln!("t {t:>2} -> ranks: {:?}", ranks_t);
            t = occ + ranks_t[c as usize] as usize;
            // eprintln!("s, t: {s:>5}..{t:>5}");
            // eprintln!("c: {c} s {s:>2} t {t:>2}");
            if s == t {
                return (steps, 0);
            }
        }
        (steps, (t - s) as usize)
    }
}
