use std::sync::LazyLock;

use crate::binary::{self, blocks::*};

use super::super_blocks::ShiftPairedSB;

#[allow(unused)]
enum Support {
    All,
    // can't query 0
    NoN,
    // can't query n
    NoZero,
    // can't query when answer > u32::MAX
    U32NoN,
    // seq must not be empty
    NotEmptyNoZero,
}

#[test]
fn binary() {
    use Support::*;
    test::<binary::Ranker<BinaryBlock64x2>>(All);
    test::<binary::Ranker<BinaryBlock32x2>>(All);
    test::<binary::Ranker<BinaryBlock23_9>>(All);
    test::<binary::Ranker<BinaryBlock16x2>>(All);
    test::<binary::Ranker<BinaryBlock32x2>>(All);
    test::<binary::Ranker<BinaryBlock16>>(All);
    test::<binary::Ranker<BinaryBlock16Spider>>(All);
    test::<binary::Ranker<BinaryBlock16Spider2>>(All);

    test::<binary::Ranker<BinaryBlock32x2, ShiftPairedSB>>(All);
    test::<binary::Ranker<BinaryBlock23_9, ShiftPairedSB>>(All);
    test::<binary::Ranker<BinaryBlock32, ShiftPairedSB>>(All);
    test::<binary::Ranker<BinaryBlock16x2, ShiftPairedSB>>(All);
    test::<binary::Ranker<BinaryBlock16, ShiftPairedSB>>(All);

    #[cfg(feature = "ext")]
    {
        use crate::genedex;
        use crate::sux;
        test::<genedex::Flat64>(All);
        test::<genedex::Flat512>(All);
        test::<genedex::Condensed64>(All);
        test::<genedex::Condensed512>(All);
        test::<qwt::RSNarrow>(All);
        test::<qwt::RSWide>(All);
        test::<sux::Rank9>(NoN);
        test::<sux::RankSmall0>(NoN);
        test::<sux::RankSmall1>(NoN);
        test::<sux::RankSmall2>(NoN);
        test::<sux::RankSmall3>(NoN);
        test::<sux::RankSmall4>(NoN);

        test::<succinct::Rank9<Vec<u64>>>(NoZero);
        test::<succinct::JacobsonRank<Vec<u64>>>(NotEmptyNoZero);
        test::<sucds::bit_vectors::Rank9Sel>(All);
        test::<rsdict::RsDict>(NoN);
        test::<bio::data_structures::rank_select::RankSelect>(NoZero);
        test::<vers_vecs::RsVec>(All);
        test::<bitm::RankSimple>(U32NoN);
        test::<bitm::RankSelect101111>(NoN);
    }
}

static TESTS: LazyLock<Vec<Test>> = LazyLock::new(|| tests());

fn test<R: binary::RankerT>(support: Support) {
    for test in &*TESTS {
        let n = test.seq.len() * u64::BITS as usize;
        eprintln!(
            "testing ranker {} on len {} n={n}",
            std::any::type_name::<R>(),
            test.seq.len()
        );
        if let Support::NotEmptyNoZero = support {
            if n == 0 {
                continue;
            }
        }

        let ranker = R::new_packed(&test.seq);
        for &(q, a) in &test.queries {
            match support {
                Support::NoN => {
                    if q == n {
                        continue;
                    }
                }
                Support::NoZero => {
                    if a == 0 {
                        continue;
                    }
                }
                Support::U32NoN => {
                    if a > u32::MAX as usize || q == n {
                        continue;
                    }
                }
                Support::All => {}
                Support::NotEmptyNoZero => {
                    if q == 0 {
                        continue;
                    }
                }
            }

            let r = unsafe { ranker.rank_unchecked(q) as usize };
            assert_eq!(
                r,
                a,
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
    seq: Vec<u64>,
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

fn seqs() -> Vec<Vec<u64>> {
    let mut seqs = vec![];
    for len in 0..100 {
        seqs.push(vec![0; len]);
        seqs.push(vec![u64::MAX; len]);
        seqs.push(vec![rand::random::<u64>(); len]);
    }
    for len in [
        // block and superblock sizes
        48,
        56,
        60,
        62,
        7680 / 8,
        7936 / 8,
        15360 / 8,
        15872 / 8,
        983040 / 8,
        1966080 / 8,
        // 469762048 / 8,
        // 939524096 / 8,
        // 1_006_632_960 / 8,
        // some random
        rand::random_range(1000..10000),
        rand::random_range(10_000..100_000),
        rand::random_range(100_000..1_000_000),
        rand::random_range(1_000_000..10_000_000),
        // > 2^32 bits, even with first 25% random
        // rand::random_range(90_000_000..110_000_000),
    ] {
        eprintln!("generating seq of len {}", len);
        // random prefix of length 0%..25%, then 1
        let prefix = rand::random_range(0..len / 4);
        let mut seq = vec![u64::MAX; len];
        for i in 0..prefix {
            seq[i] = rand::random::<u64>();
        }
        seqs.push(seq);
    }

    seqs
}

fn queries(seq: &Vec<u64>) -> Vec<(usize, usize)> {
    let n = seq.len() * u64::BITS as usize;
    let mut queries = vec![];
    if n <= 10000 {
        for i in 0..=n {
            queries.push(i);
        }
    } else {
        queries.push(0);
        queries.push(n - 1);
        queries.push(n);
        for _ in 0..10000 {
            queries.push(rand::random_range(1..n));
        }
    }
    queries.sort_unstable();
    queries.dedup();

    let mut qa = vec![];
    let mut i = 0;
    let mut a = 0;
    for q in queries {
        while i < q {
            let word_idx = i / u64::BITS as usize;
            let bit_idx = i % u64::BITS as usize;
            let bit = (seq[word_idx] >> bit_idx) & 1;
            a += bit as usize;
            i += 1;
        }
        qa.push((q, a));
    }

    qa
}
