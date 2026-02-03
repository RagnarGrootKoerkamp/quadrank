use std::sync::LazyLock;

use super::blocks::*;
use super::super_blocks::{ShiftPairedSB, ShiftSB, TrivialSB};
use crate::quad;

#[test]
fn quad() {
    test::<quad::Ranker<Basic128, TrivialSB>>();
    test::<quad::Ranker<Basic256, TrivialSB>>();
    test::<quad::Ranker<Basic512, TrivialSB>>();
    // test::<quad::Ranker<Basic512, SB8>>();
    // test::<quad::Ranker<Basic512, SB8>>();
    // test::<quad::Ranker<FullBlock, NoSB>>();
    // test::<quad::Ranker<FullBlockMid, NoSB>>();
    // test::<quad::Ranker<FullBlockMid, NoSB>>();
    test::<quad::Ranker<QuadBlock32_8_8_8FP, TrivialSB>>();
    test::<quad::Ranker<QuadBlock32_8_8_8FP, TrivialSB>>();
    test::<quad::Ranker<QuadBlock32_8_8_8FP, TrivialSB>>();
    test::<quad::Ranker<QuadBlock7_18_7P, TrivialSB>>();
    test::<quad::Ranker<QuadBlock24_8, TrivialSB>>();
    test::<quad::Ranker<QuadBlock24_8, TrivialSB>>();
    test::<quad::Ranker<QuadBlock64, TrivialSB>>();
    test::<quad::Ranker<QuadBlock32, TrivialSB>>();
    test::<quad::Ranker<QuadBlock16, TrivialSB>>();

    test1::<quad::Ranker<QuadBlock64, ShiftSB>>();
    test1::<quad::Ranker<QuadBlock24_8, ShiftSB>>();
    test1::<quad::Ranker<QuadBlock16, ShiftSB>>();

    test::<quad::Ranker<QuadBlock32_8_8_8FP, ShiftPairedSB>>();
    test::<quad::Ranker<QuadBlock7_18_7P, ShiftPairedSB>>();
    test::<quad::Ranker<QuadBlock24_8, ShiftPairedSB>>();
    test::<quad::Ranker<QuadBlock32, ShiftPairedSB>>();
    test::<quad::Ranker<QuadBlock16, ShiftPairedSB>>();

    #[cfg(feature = "ext")]
    {
        use crate::genedex;
        test::<qwt::RSQVector256>();
        test::<qwt::RSQVector512>();
        test::<genedex::Flat64>();
        test::<genedex::Flat512>();
        test::<genedex::Condensed64>();
        test::<genedex::Condensed512>();
    }
}

static TESTS: LazyLock<Vec<Test>> = LazyLock::new(|| tests());

fn test<R: quad::RankerT>() {
    for test in &*TESTS {
        eprintln!(
            "testing ranker {} on len {} words",
            std::any::type_name::<R>(),
            test.seq.len()
        );
        let ranker = R::new_packed(&test.seq);
        for (q, a) in &test.queries {
            let r = unsafe { ranker.rank4_unchecked(*q) };
            let r = r.map(|x| x as usize);
            assert_eq!(
                r,
                *a,
                "failed test: ranker {} seq len {}, query {}, want {:?}, got {:?}",
                std::any::type_name::<R>(),
                test.seq.len(),
                q,
                a,
                r
            );
        }
    }
}

fn test1<R: quad::RankerT>() {
    for test in &*TESTS {
        eprintln!(
            "testing ranker {} on len {} words for count1",
            std::any::type_name::<R>(),
            test.seq.len()
        );
        let ranker = R::new_packed(&test.seq);
        for (q, a) in &test.queries {
            for c in 0..4 {
                let r = unsafe { ranker.rank1_unchecked(*q, c) as usize };
                assert_eq!(
                    r,
                    a[c as usize],
                    "failed test: ranker {} seq len {}, query {}, want {}, got {}",
                    std::any::type_name::<R>(),
                    test.seq.len(),
                    q,
                    a[c as usize],
                    r
                );
            }
        }
    }
}
struct Test {
    seq: Vec<u64>,
    queries: Vec<(usize, [usize; 4])>,
}

fn tests() -> Vec<Test> {
    eprintln!("generating tests...");
    seqs()
        .into_iter()
        .map(|seq| {
            let queries = queries(&seq);
            Test { seq, queries }
        })
        .collect()
}

fn seqs() -> Vec<Vec<u64>> {
    // 01230123...
    let m = 0b1110010011100100111001001110010011100100111001001110010011100100u64;
    let mut seqs = vec![];
    for len in 1..2 {
        seqs.push(vec![0; len]);
        seqs.push(vec![m; len]);
        seqs.push(vec![rand::random::<u64>() as u64; len]);
    }
    for len in [
        // specific sizes matching an exact super block
        48usize / 8,
        56 / 8,
        14336 / 8,
        28672 / 8,
        49152 / 8,
        98304 / 8,
        3145728 / 8,
        6291456 / 8,
        // 536_870_912 / 8,
        // 805_306_368 / 8,
        // 1_073_741_824 / 8,
        // 1_610_612_736 / 8,
        // some random stuff
        rand::random_range(1000..10000),
        rand::random_range(10_000..100_000),
        rand::random_range(100_000..1_000_000),
        rand::random_range(1_000_000..10_000_000),
        // > 2^33 bits = 2^32 bp
        // rand::random_range(160_000_000..200_000_000),
    ] {
        eprintln!("generating seq of len {}", len);
        // random prefix of length 0%..25%, then 0
        let prefix = rand::random_range(0..len / 4);
        let mut seq = vec![0; len];
        for i in 0..prefix {
            seq[i] = rand::random::<u64>() as u64;
        }
        seqs.push(seq);
    }

    seqs
}

fn queries(seq: &Vec<u64>) -> Vec<(usize, [usize; 4])> {
    let n = seq.len() * u64::BITS as usize / 2;
    let mut queries = vec![];
    if n <= 1000 {
        for i in 0..=n {
            queries.push(i);
        }
    } else {
        queries.push(0);
        queries.push(n - 1);
        queries.push(n);
        for _ in 0..10000 {
            queries.push(rand::random_range(0..=n));
        }
    }
    queries.sort_unstable();
    queries.dedup();

    let mut qa = vec![];
    let mut i = 0;
    let mut a = [0, 0, 0, 0];
    for q in queries {
        while i < q {
            let word_idx = (2 * i) / u64::BITS as usize;
            let bit_idx = (2 * i) % u64::BITS as usize;
            let c = (seq[word_idx] >> bit_idx) & 3;
            a[c as usize] += 1;
            i += 1;
        }
        qa.push((q, a));
    }

    qa
}
