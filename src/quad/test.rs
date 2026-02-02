use std::sync::LazyLock;

use super::blocks::*;
use super::super_blocks::ShiftPairedSB;
use super::super_blocks::ShiftSB;
use crate::genedex;
use crate::quad;
use crate::quad::TrivialSB;
use crate::quad::count4::*;

#[test]
fn quad() {
    test::<quad::Ranker<Basic128, TrivialSB, WideSimdCount2>>();
    test::<quad::Ranker<Basic256, TrivialSB, SimdCountSlice>>();
    test::<quad::Ranker<Basic512, TrivialSB, SimdCountSlice>>();
    // test::<quad::Ranker<Basic512, SB8, U128Popcnt3>>();
    // test::<quad::Ranker<Basic512, SB8, SimdCountSlice>>();
    // test::<quad::Ranker<FullBlock, NoSB, U64PopcntSlice>>();
    // test::<quad::Ranker<FullBlockMid, NoSB, U64PopcntSlice>>();
    // test::<quad::Ranker<FullBlockMid, NoSB, WideSimdCount2>>();
    test::<quad::Ranker<QuadBlock32_8_8_8FP, TrivialSB, SimdCount8>>();
    test::<quad::Ranker<QuadBlock32_8_8_8FP, TrivialSB, SimdCount9>>();
    test::<quad::Ranker<QuadBlock32_8_8_8FP, TrivialSB, SimdCount10>>();
    test::<quad::Ranker<QuadBlock7_18_7P, TrivialSB, SimdCount10>>();
    test::<quad::Ranker<QuadBlock24_8, TrivialSB, SimdCount11B>>();
    test::<quad::Ranker<QuadBlock24_8, TrivialSB, TransposedPopcount>>();
    test::<quad::Ranker<QuadBlock64, TrivialSB, SimdCount11B>>();
    test::<quad::Ranker<QuadBlock32, TrivialSB, NoCount>>();
    test::<quad::Ranker<QuadBlock16, TrivialSB, NoCount>>();
    test::<qwt::RSQVector256>();
    test::<qwt::RSQVector512>();
    test::<genedex::Flat64>();
    test::<genedex::Flat512>();
    test::<genedex::Condensed64>();
    test::<genedex::Condensed512>();

    test1::<quad::Ranker<QuadBlock64, ShiftSB, SimdCount11B>>();
    test1::<quad::Ranker<QuadBlock24_8, ShiftSB, SimdCount11B>>();
    test1::<quad::Ranker<QuadBlock16, ShiftSB, NoCount>>();

    test::<quad::Ranker<QuadBlock32_8_8_8FP, ShiftPairedSB, SimdCount10>>();
    test::<quad::Ranker<QuadBlock7_18_7P, ShiftPairedSB, SimdCount10>>();
    test::<quad::Ranker<QuadBlock24_8, ShiftPairedSB, TransposedPopcount>>();
    test::<quad::Ranker<QuadBlock32, ShiftPairedSB, NoCount>>();
    test::<quad::Ranker<QuadBlock16, ShiftPairedSB, NoCount>>();
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
            let r = ranker.rank4(*q);
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
                let r = ranker.rank1(*q, c) as usize;
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
    seq: Vec<usize>,
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

fn seqs() -> Vec<Vec<usize>> {
    // 01230123...
    let m = 0b1110010011100100111001001110010011100100111001001110010011100100usize;
    let mut seqs = vec![];
    for len in 1..2 {
        seqs.push(vec![0; len]);
        seqs.push(vec![m; len]);
        seqs.push(vec![rand::random::<u64>() as usize; len]);
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
            seq[i] = rand::random::<u64>() as usize;
        }
        seqs.push(seq);
    }

    seqs
}

fn queries(seq: &Vec<usize>) -> Vec<(usize, [usize; 4])> {
    let n = seq.len() * usize::BITS as usize / 2;
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
            let word_idx = (2 * i) / usize::BITS as usize;
            let bit_idx = (2 * i) % usize::BITS as usize;
            let c = (seq[word_idx] >> bit_idx) & 3;
            a[c as usize] += 1;
            i += 1;
        }
        qa.push((q, a));
    }

    qa
}
