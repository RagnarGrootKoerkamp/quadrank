use std::sync::LazyLock;

use super::blocks::*;
use crate::genedex;
use crate::quad;
use crate::quad::TrivialSB;
use crate::quad::count4::*;
use crate::quad::super_blocks::{NoSB, SB8};

#[test]
fn quad() {
    test::<quad::Ranker<Plain128, TrivialSB, WideSimdCount2, false>>();
    test::<quad::Ranker<Plain256, TrivialSB, SimdCountSlice, false>>();
    test::<quad::Ranker<Plain512, TrivialSB, SimdCountSlice, false>>();
    test::<quad::Ranker<Plain512, SB8, U128Popcnt3, true>>();
    test::<quad::Ranker<Plain512, SB8, SimdCountSlice, false>>();
    test::<quad::Ranker<FullBlock, NoSB, U64PopcntSlice, false>>();
    test::<quad::Ranker<FullBlockMid, NoSB, U64PopcntSlice, false>>();
    test::<quad::Ranker<FullBlockMid, NoSB, WideSimdCount2, false>>();
    test::<quad::Ranker<QuartBlock, NoSB, SimdCount8, false>>();
    test::<quad::Ranker<QuartBlock, NoSB, SimdCount9, false>>();
    test::<quad::Ranker<QuartBlock, NoSB, SimdCount10, false>>();
    test::<quad::Ranker<PentaBlock, TrivialSB, SimdCount8, false>>();
    test::<quad::Ranker<HexaBlock, TrivialSB, WideSimdCount2, false>>();
    test::<quad::Ranker<HexaBlock2, TrivialSB, WideSimdCount2, false>>();
    test::<quad::Ranker<HexaBlockMid, TrivialSB, SimdCount8, false>>();
    test::<quad::Ranker<HexaBlockMid, TrivialSB, SimdCount9, false>>();
    test::<quad::Ranker<HexaBlockMid2, TrivialSB, SimdCount9, false>>();
    test::<quad::Ranker<HexaBlockMid2, TrivialSB, SimdCount10, false>>();
    test::<quad::Ranker<HexaBlockMid3, TrivialSB, SimdCount10, false>>();
    test::<quad::Ranker<HexaBlockMid4, TrivialSB, SimdCount10, false>>();
    test::<quad::Ranker<TriBlock, TrivialSB, SimdCount11, false>>();
    test::<quad::Ranker<TriBlock, TrivialSB, SimdCount11B, false>>();
    test::<quad::Ranker<TriBlock2, TrivialSB, SimdCount11B, false>>();
    test::<quad::Ranker<TriBlock2, TrivialSB, TransposedPopcount, false>>();
    test::<quad::Ranker<FullBlockTransposed, TrivialSB, SimdCount11B, false>>();
    test::<quad::Ranker<FullDouble32, TrivialSB, SimdCount11B, false>>();
    test::<quad::Ranker<FullDouble16, TrivialSB, SimdCount11B, false>>();
    test::<quad::Ranker<FullDouble16Inl, TrivialSB, SimdCount11B, false>>();
    test::<qwt::RSQVector256>();
    test::<qwt::RSQVector512>();
    test::<genedex::Flat64>();
    test::<genedex::Flat512>();
    test::<genedex::Condensed64>();
    test::<genedex::Condensed512>();
}

static TESTS: LazyLock<Vec<Test>> = LazyLock::new(|| tests());

fn test<R: quad::RankerT>() {
    for test in &*TESTS {
        eprintln!(
            "testing ranker {} on len {}",
            std::any::type_name::<R>(),
            test.seq.len()
        );
        let ranker = R::new_packed(&test.seq);
        for (q, a) in &test.queries {
            let r = ranker.count(*q);
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
            // let r = ranker.count1(*q, *c) as usize;
            // assert_eq!(
            //     r,
            //     *a,
            //     "failed test: ranker {} seq len {}, query {}, want {}, got {}",
            //     std::any::type_name::<R>(),
            //     test.seq.len(),
            //     q,
            //     a,
            //     r
            // );
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
    let m = 0b1110010011100100111001001110010011100100111001001110010011100100;
    let mut seqs = vec![];
    for len in 1..100 {
        seqs.push(vec![0; len]);
        seqs.push(vec![m; len]);
        seqs.push(vec![rand::random::<u64>() as usize; len]);
    }
    for len in [
        rand::random_range(1000..10000),
        rand::random_range(10_000..100_000),
        rand::random_range(100_000..1_000_000),
        rand::random_range(1_000_000..10_000_000),
        // > 2^33 bits = 2^32 bp
        rand::random_range(160_000_000..200_000_000),
    ] {
        eprintln!("generating seq of len {}", len);
        seqs.push(vec![0; len]);
        seqs.push(vec![m; len]);
        seqs.push(vec![rand::random::<u64>() as usize; len]);
    }

    seqs
}

fn queries(seq: &Vec<usize>) -> Vec<(usize, [usize; 4])> {
    let n = seq.len() * usize::BITS as usize / 2;
    let mut queries = vec![];
    if n <= 10000 {
        for i in 0..n {
            queries.push(i);
        }
    } else {
        queries.push(0);
        queries.push(n - 1);
        for _ in 0..10000 {
            queries.push(rand::random_range(0..n));
        }
    }
    queries.sort_unstable();

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
