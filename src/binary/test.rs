use std::sync::LazyLock;

use crate::binary::{self, blocks::*};
use crate::genedex;
use crate::sux;

#[test]
fn binary() {
    test::<binary::Ranker<BinaryBlock>>();
    test::<binary::Ranker<BinaryBlock2>>();
    test::<binary::Ranker<BinaryBlock3>>();
    test::<binary::Ranker<BinaryBlock4>>();
    test::<binary::Ranker<BinaryBlock5>>();
    test::<binary::Ranker<BinaryBlock6>>();
    test::<binary::Ranker<Spider>>();
    test::<genedex::Flat64>();
    test::<genedex::Flat512>();
    test::<genedex::Condensed64>();
    test::<genedex::Condensed512>();
    test::<qwt::RSNarrow>();
    test::<qwt::RSWide>();
    test::<sux::Rank9>();
    test::<sux::RankSmall1>();
    test::<sux::RankSmall2>();
    test::<sux::RankSmall3>();
    test::<sux::RankSmall4>();
    test::<sux::RankSmall5>();

    test::<succinct::Rank9<Vec<u64>>>();
    test::<succinct::JacobsonRank<Vec<u64>>>();
    test::<sucds::bit_vectors::Rank9Sel>();
    test::<rsdict::RsDict>();
    test::<bio::data_structures::rank_select::RankSelect>();
    test::<vers_vecs::RsVec>();
}

static TESTS: LazyLock<Vec<Test>> = LazyLock::new(|| tests());

fn test<R: binary::RankerT>() {
    for test in &*TESTS {
        eprintln!(
            "testing ranker {} on len {}",
            std::any::type_name::<R>(),
            test.seq.len()
        );
        let ranker = R::new_packed(&test.seq);
        for (q, a) in &test.queries {
            let r = unsafe { ranker.rank_unchecked(*q) as usize };
            assert_eq!(
                r,
                *a,
                "failed test: ranker {} seq len {}, query {}, want {}, got {}",
                std::any::type_name::<R>(),
                test.seq.len(),
                q,
                a,
                r
            );
        }
    }
}

struct Test {
    seq: Vec<usize>,
    queries: Vec<(usize, usize)>,
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
    let m = usize::MAX;
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
        // > 2^32 bits
        rand::random_range(80_000_000..100_000_000),
    ] {
        eprintln!("generating seq of len {}", len);
        seqs.push(vec![0; len]);
        seqs.push(vec![m; len]);
        seqs.push(vec![rand::random::<u64>() as usize; len]);
    }

    seqs
}

fn queries(seq: &Vec<usize>) -> Vec<(usize, usize)> {
    let n = seq.len() * usize::BITS as usize;
    let mut queries = vec![];
    if n <= 10000 {
        for i in 1..n {
            queries.push(i);
        }
    } else {
        // queries.push(0);
        queries.push(n - 1);
        for _ in 0..10000 {
            queries.push(rand::random_range(1..n));
        }
    }
    queries.sort_unstable();

    let mut qa = vec![];
    let mut i = 0;
    let mut a = 0;
    for q in queries {
        while i < q {
            let word_idx = i / usize::BITS as usize;
            let bit_idx = i % usize::BITS as usize;
            let bit = (seq[word_idx] >> bit_idx) & 1;
            a += bit as usize;
            i += 1;
        }
        qa.push((q, a));
    }

    qa
}
