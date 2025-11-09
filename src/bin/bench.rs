#![allow(incomplete_features, dead_code)]
#![feature(generic_const_exprs, coroutines, coroutine_trait, stmt_expr_attributes)]
use std::{
    array::from_fn,
    hint::black_box,
    ops::{Coroutine, CoroutineState::Complete},
    pin::pin,
};

use dna_rank::{
    DnaRank, Ranks,
    blocks::{FullBlock, HexaBlock, HexaBlock18bit, PentaBlock, QuartBlock},
    count4,
    ranker::Ranker,
};
use mem_dbg::MemSize;
use qwt::{RankQuad, RankUnsigned, SpaceUsage, WTSupport};
use sux::{bits::BitVec, traits::Rank};

fn check(pos: usize, ranks: Ranks) {
    std::hint::black_box(&ranks);
    let pos = pos as u32;
    debug_assert_eq!(
        ranks,
        [(pos + 3) / 4, (pos + 2) / 4, (pos + 1) / 4, pos / 4],
    );
}

type QS = [Vec<usize>; 5];

fn time_fn(queries: &QS, f: impl Fn(&[usize])) {
    let mut times: Vec<_> = queries
        .iter()
        .map(|queries| {
            let start = std::time::Instant::now();
            f(&queries);
            start.elapsed().as_nanos()
        })
        .collect();
    times.sort();
    let ns2 = times[2] as f64 / queries[0].len() as f64;
    eprint!(" {ns2:>4.1}",);
}

// fn time_fn(queries: &QS, f: impl Fn(&[usize]) + Sync) {
//     let start = std::time::Instant::now();
//     std::thread::scope(|scope| {
//         queries.iter().for_each(|queries| {
//             let f = &f;
//             scope.spawn(move || {
//                 f(queries);
//             });
//         });
//     });
//     let ns = start.elapsed().as_nanos() as f64 / (queries.len() * queries[0].len()) as f64;
//     eprint!(" {ns:>5.2}",);
// }

fn time(queries: &QS, f: impl Fn(usize) -> Ranks + Sync) {
    time_fn(queries, |queries| {
        for &q in queries {
            check(q, f(q));
        }
    });
}

fn time_batch<const BATCH: usize>(
    queries: &QS,
    prefetch: impl Fn(usize) + Sync,
    f: impl Fn(usize) -> Ranks + Sync,
) {
    time_fn(queries, |queries| {
        let qs = queries.as_chunks::<BATCH>().0;
        for batch in qs {
            for &q in batch {
                prefetch(q);
            }
            for &q in batch {
                check(q, f(q));
            }
        }
    })
}

fn time_stream(
    queries: &QS,
    lookahead: usize,
    prefetch: impl Fn(usize) + Sync,
    f: impl Fn(usize) -> Ranks + Sync,
) {
    time_fn(queries, |queries| {
        for (&q, &ahead) in queries.iter().zip(&queries[lookahead..]) {
            prefetch(ahead);
            check(q, f(q));
        }
    });
}

#[inline(always)]
fn time_coro2_batch<F>(queries: &QS, _lookahead: usize, f: impl Fn(usize) -> F + Sync)
where
    F: Coroutine<Return = Ranks> + Unpin,
{
    time_fn(queries, |queries| {
        for batch in queries.as_chunks::<32>().0 {
            let mut futures: [_; 32] = from_fn(|i| f(batch[i]));

            for f in &mut futures {
                pin!(f).resume(());
            }

            for (q, func) in batch.iter().zip(&mut futures) {
                let Complete(fq) = pin!(func).resume(()) else {
                    panic!()
                };
                check(*q, fq);
            }
        }
    });
}

#[inline(always)]
fn time_coro2_stream<F>(queries: &QS, _lookahead: usize, f: impl Fn(usize) -> F + Sync)
where
    F: Coroutine<Return = Ranks> + Unpin,
{
    time_fn(queries, |queries| {
        let mut funcs: [F; 32] = from_fn(|i| {
            let mut func = f(queries[i]);
            pin!(&mut func).resume(());
            func
        });

        for i in 0..queries.len() - 32 {
            // Finish the old fn.
            let func = &mut funcs[i % 32];
            let Complete(fq) = pin!(func).resume(()) else {
                panic!()
            };
            check(queries[i], fq);

            // Start a new fn.
            funcs[i % 32] = f(queries[i + 32]);
            pin!(&mut funcs[i % 32]).resume(());
        }
    });
}

#[inline(always)]
fn time_coro_batch<F>(queries: &QS, _lookahead: usize, f: impl Fn(usize) -> F + Sync)
where
    F: Coroutine<Return = Ranks> + Unpin,
{
    time_fn(queries, |queries| {
        for batch in queries.as_chunks::<B>().0 {
            let mut funcs: [_; B] = from_fn(|i| f(batch[i]));

            for (q, func) in batch.iter().zip(&mut funcs) {
                let Complete(fq) = pin!(func).resume(()) else {
                    panic!()
                };
                check(*q, fq);
            }
        }
    });
}

const B: usize = 32;

#[inline(always)]
fn time_coro_stream<F>(queries: &QS, _lookahead: usize, f: impl Fn(usize) -> F + Sync)
where
    F: Coroutine<Return = Ranks> + Unpin,
{
    time_fn(queries, |queries| {
        let mut funcs: [F; B] = from_fn(|i| f(queries[i]));

        for i in 0..queries.len() - B {
            // finish the old state
            let func = &mut funcs[i % B];
            let Complete(fq) = pin!(func).resume(()) else {
                panic!()
            };
            check(queries[i], fq);

            // new future
            funcs[i % B] = f(queries[i + B]);
        }
    });
}

#[inline(never)]
fn bench_dna_rank<const STRIDE: usize>(seq: &[u8], queries: &QS)
where
    [(); STRIDE / 4]:,
{
    eprint!("{:<20}:", format!("DnaRank<{STRIDE:>4}>"));
    let rank = DnaRank::<STRIDE>::new(&seq);

    let bits = (rank.mem_size(Default::default()) * 8) as f64 / seq.len() as f64;
    eprint!("{bits:>6.2}b |");

    // time(&queries, |p| rank.ranks_naive(p));
    // time(&queries, |p| rank.ranks_u64(p));
    // time(&queries, |p| rank.ranks_u64_prefetch(p));
    // time(&queries, |p| rank.ranks_u64_prefetch_all(p));
    time(&queries, |p| rank.ranks_u64_3(p)); // best

    // time(&queries, |p| rank.ranks_u128(p));
    // time(&queries, |p| rank.ranks_u128_3(p));
    eprintln!();
}

#[inline(never)]
fn bench_rank9(seq: &[u8], queries: &QS) {
    eprint!("{:<20}:", "rank9");

    // Cast to slice of usize.
    let bitvec = unsafe { BitVec::from_raw_parts(seq.align_to().1, seq.len() * 8) };
    let rank9 = sux::rank_sel::Rank9::new(bitvec);

    let bits = rank9.mem_size(Default::default()) as f64 / seq.len() as f64;
    eprint!("{bits:>6.2}b |");

    time(&queries, |p| [rank9.rank(p) as u32, 0, 0, 0]);
    eprintln!();
}

#[inline(never)]
fn bench_qwt(seq: &[u8], queries: &QS) {
    let seq = seq
        .iter()
        .map(|&b| match b {
            b'A' => 0u8,
            b'C' => 1,
            b'G' => 2,
            b'T' => 3,
            _ => 0,
        })
        .collect::<Vec<_>>();

    eprint!("{:<20}:", "RSQ256");
    let rsq = qwt::RSQVector256::new(&seq);
    let bits = (rsq.space_usage_byte() * 8) as f64 / seq.len() as f64;
    eprint!("{bits:>6.2}b |");
    unsafe {
        time_loop(&queries, |p| [rsq.rank_unchecked(0, p) as u32, 0, 0, 0]);

        time_stream(
            &queries,
            B,
            |p| {
                rsq.prefetch_data(p);
                rsq.prefetch_info(p)
            },
            |p| [rsq.rank_unchecked(0, p) as u32, 0, 0, 0],
        );
    }
    eprintln!();
}

#[inline(never)]
fn bench_quart<const C3: bool>(seq: &[u8], queries: &QS) {
    eprint!("{:<20}:", format!("QuartBlock {C3}"));

    let bwa_ranker = Ranker::<FullBlock>::new(&seq);
    let bits = (bwa_ranker.mem_size(Default::default()) * 8) as f64 / seq.len() as f64;
    eprint!("{bits:>6.2}b |");

    time(&queries, |p| {
        bwa_ranker.count::<count4::U64PopcntSlice, false>(p)
    });
    time_stream(
        &queries,
        B,
        |p| bwa_ranker.prefetch(p),
        |p| bwa_ranker.count::<count4::U64PopcntSlice, false>(p),
    );

    eprint!(" |");

    let ranker = Ranker::<QuartBlock>::new(&seq);
    let bits = (ranker.mem_size(Default::default()) * 8) as f64 / seq.len() as f64;
    eprint!("{bits:>6.2}b |");

    // time_fn(queries, |queries| {
    //     for &q in queries {
    //         std::hint::black_box(ranker.count1(q, q as u8 & 3));
    //     }
    // });
    // time_fn(queries, |queries| {
    //     for (&q, &ahead) in queries.iter().zip(&queries[32..]) {
    //         ranker.prefetch(ahead);
    //         std::hint::black_box(ranker.count1(q, q as u8 & 3));
    //     }
    // });

    // time(&queries, |p| ranker.count::<count4::U64Popcnt, C3>(p));
    time(&queries, |p| ranker.count::<count4::SimdCount7, false>(p));

    time_stream(
        &queries,
        B,
        |p| ranker.prefetch(p),
        |p| ranker.count::<count4::SimdCount7, false>(p),
    );
    eprint!(" |");

    time(&queries, |p| {
        let ranks = black_box(ranker.count_long::<count4::SimdCount7, false>(p));
        ranks.map(|r| r as u32)
    });

    time_stream(
        &queries,
        B,
        |p| ranker.prefetch(p),
        |p| {
            let ranks = black_box(ranker.count_long::<count4::SimdCount7, false>(p));
            ranks.map(|r| r as u32)
        },
    );

    eprint!(" |");

    let ranker = Ranker::<PentaBlock>::new(&seq);
    let bits = (ranker.mem_size(Default::default()) * 8) as f64 / seq.len() as f64;
    eprint!("{bits:>6.2}b |");

    time(&queries, |p| ranker.count::<count4::SimdCount7, false>(p));

    time_stream(
        &queries,
        B,
        |p| ranker.prefetch(p),
        |p| ranker.count::<count4::SimdCount7, false>(p),
    );

    eprint!(" |");

    let ranker = Ranker::<HexaBlock>::new(&seq);
    let bits = (ranker.mem_size(Default::default()) * 8) as f64 / seq.len() as f64;
    eprint!("{bits:>6.2}b |");

    time(&queries, |p| {
        ranker.count::<count4::WideSimdCount2, false>(p)
    });

    time_stream(
        &queries,
        B,
        |p| ranker.prefetch(p),
        |p| ranker.count::<count4::WideSimdCount2, false>(p),
    );

    eprintln!();

    // time_coro_stream(&queries, B, |p| {
    //     ranker.count_coro::<count4::U64Popcnt, C3>(p)
    // });
    // time_coro_stream(&queries, B, |p| {
    //     ranker.count_coro::<count4::ByteLookup8, C3>(p)
    // });
    // time_coro_stream(&queries, B, |p| {
    //     ranker.count_coro::<count4::SimdCount7, false>(p)
    // });
}

fn main() {
    #[cfg(debug_assertions)]
    let q = 10_000;
    #[cfg(debug_assertions)]
    let ns = [100_000];
    #[cfg(not(debug_assertions))]
    let q = 10_000_000;
    #[cfg(not(debug_assertions))]
    let ns = [100_000, 10_000_000, 1_000_000_000, 10_000_000_000usize];

    for n in ns {
        // for n in [100_000] {
        eprintln!("n = {}", n);
        let seq = b"ACTG".repeat(n / 4);
        let queries = QS::default().map(|_| {
            (0..q)
                .map(|_| rand::random_range(0..seq.len()))
                .collect::<Vec<_>>()
        });

        // bench_quart::<true>(&seq, &queries);
        bench_quart::<false>(&seq, &queries);
        // bench_best(&seq, &queries);
        // bench_rank9(&seq, &queries);
        bench_qwt(&seq, &queries);

        // bench_dna_rank::<64>(&seq, &queries);
        // bench_dna_rank::<128>(&seq, &queries);
    }
}

// TODO: 40bit support
