#![allow(incomplete_features, dead_code)]
#![feature(generic_const_exprs, coroutines, coroutine_trait, stmt_expr_attributes)]
use std::{
    any::type_name_of_val,
    array::from_fn,
    ops::{Coroutine, CoroutineState::Complete},
    pin::pin,
};

use dna_rank::{
    DnaRank, Ranks,
    blocks::{
        DumbBlock, FullBlock, HexaBlock, HexaBlock18bit, PentaBlock, PentaBlock20bit, QuartBlock,
    },
    count4::{self, CountFn, SimdCount7, WideSimdCount2},
    ranker::{BasicBlock, Ranker, SuperBlock},
    super_block::{NoSB, SB8, TrivialSB},
};
use mem_dbg::MemSize;
use qwt::{RankQuad, SpaceUsage, WTSupport};
use sux::{bits::BitVec, traits::Rank};

fn check(pos: usize, ranks: Ranks) {
    std::hint::black_box(&ranks);
    let pos = pos as u32;
    if ranks[1] == 0 {
        debug_assert_eq!(ranks, [(pos + 3) / 4, 0, 0, 0]);
    } else {
        debug_assert_eq!(
            ranks,
            [(pos + 3) / 4, (pos + 2) / 4, (pos + 1) / 4, pos / 4],
        );
    }
}

type QS = [Vec<usize>; 6];

#[derive(Clone, Copy)]
enum Threading {
    Single,
    Multi,
}

fn time_fn_median(queries: &QS, f: impl Fn(&[usize])) {
    let mut times: Vec<_> = queries
        .iter()
        .take(5)
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

fn time_fn_mt(queries: &QS, f: impl Fn(&[usize]) + Sync + Copy) {
    let start = std::time::Instant::now();
    std::thread::scope(|scope| {
        queries.iter().for_each(|queries| {
            let f = &f;
            scope.spawn(move || {
                f(queries);
            });
        });
    });
    let ns = start.elapsed().as_nanos() as f64 / (queries.len() * queries[0].len()) as f64;
    eprint!(" {ns:>5.2}",);
}

fn time_fn(queries: &QS, t: Threading, f: impl Fn(&[usize]) + Sync + Copy) {
    match t {
        Threading::Multi => time_fn_mt(queries, f),
        Threading::Single => time_fn_median(queries, f),
    }
}

fn time_latency(
    queries: &QS,
    t: Threading,
    prefetch: impl Fn(usize) + Sync,
    f: impl Fn(usize) -> Ranks + Sync + Copy,
) {
    time_fn(queries, t, |queries| {
        let mut acc = 0;
        for &q in queries {
            // Make query depend on previous result.
            let q = q ^ acc;
            prefetch(q);
            let ranks = f(q);
            acc ^= (ranks[0] & 1) as usize;
            check(q, ranks);
        }
    });
}

fn time_loop(queries: &QS, t: Threading, f: impl Fn(usize) -> Ranks + Sync + Copy) {
    time_fn(queries, t, |queries| {
        for &q in queries {
            check(q, f(q));
        }
    });
}

const BATCH: usize = 32;

fn time_batch(
    queries: &QS,
    t: Threading,
    prefetch: impl Fn(usize) + Sync,
    f: impl Fn(usize) -> Ranks + Sync,
) {
    time_fn(queries, t, |queries| {
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
    t: Threading,
    prefetch: impl Fn(usize) + Sync,
    f: impl Fn(usize) -> Ranks + Sync,
) {
    time_fn(queries, t, |queries| {
        for (&q, &ahead) in queries.iter().zip(&queries[BATCH..]) {
            prefetch(ahead);
            check(q, f(q));
        }
    });
}

fn time_trip(
    queries: &QS,
    t: Threading,
    prefetch: impl Fn(usize) + Sync + Copy,
    f: impl Fn(usize) -> Ranks + Sync + Copy,
) {
    time_latency(queries, t, prefetch, f);
    time_loop(queries, t, f);
    time_stream(queries, t, prefetch, f);
}

fn bench<BB: BasicBlock, SB: SuperBlock, CF: CountFn<{ BB::C }>, const C3: bool>(
    seq: &[u8],
    queries: &QS,
) where
    [(); BB::B]:,
    [(); SB::BB]:,
    Ranker<BB, SB>: MemSize,
{
    let name = type_name_of_val(&Ranker::<BB, SB>::count::<CF, C3>);
    let name = regex::Regex::new(r"[a-zA-Z0-9_]+::")
        .unwrap()
        .replace_all(name, |_: &regex::Captures| "".to_string());

    eprint!("{name:<60}");

    let ranker = Ranker::<BB, SB>::new(&seq);
    let bits = (ranker.mem_size(Default::default()) * 8) as f64 / seq.len() as f64;
    eprint!("{bits:>6.2}b |");

    time_trip(
        &queries,
        Threading::Single,
        |p| ranker.prefetch(p),
        |p| ranker.count::<CF, false>(p),
    );
    eprint!(" |");
    time_trip(
        &queries,
        Threading::Multi,
        |p| ranker.prefetch(p),
        |p| ranker.count::<CF, false>(p),
    );
    eprintln!();
}

#[inline(always)]
fn time_coro2_batch<F>(queries: &QS, f: impl Fn(usize) -> F + Sync)
where
    F: Coroutine<Return = Ranks> + Unpin,
{
    time_fn(queries, Threading::Single, |queries| {
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
fn time_coro2_stream<F>(queries: &QS, f: impl Fn(usize) -> F + Sync)
where
    F: Coroutine<Return = Ranks> + Unpin,
{
    time_fn(queries, Threading::Single, |queries| {
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
fn time_coro_batch<F>(queries: &QS, f: impl Fn(usize) -> F + Sync)
where
    F: Coroutine<Return = Ranks> + Unpin,
{
    time_fn(queries, Threading::Single, |queries| {
        for batch in queries.as_chunks::<BATCH>().0 {
            let mut funcs: [_; BATCH] = from_fn(|i| f(batch[i]));

            for (q, func) in batch.iter().zip(&mut funcs) {
                let Complete(fq) = pin!(func).resume(()) else {
                    panic!()
                };
                check(*q, fq);
            }
        }
    });
}

#[inline(always)]
fn time_coro_stream<F>(queries: &QS, f: impl Fn(usize) -> F + Sync)
where
    F: Coroutine<Return = Ranks> + Unpin,
{
    time_fn(queries, Threading::Single, |queries| {
        let mut funcs: [F; BATCH] = from_fn(|i| f(queries[i]));

        for i in 0..queries.len() - BATCH {
            // finish the old state
            let func = &mut funcs[i % BATCH];
            let Complete(fq) = pin!(func).resume(()) else {
                panic!()
            };
            check(queries[i], fq);

            // new future
            funcs[i % BATCH] = f(queries[i + BATCH]);
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
    time_loop(&queries, Threading::Single, |p| rank.ranks_u64_3(p)); // best

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

    time_loop(&queries, Threading::Single, |p| {
        [rank9.rank(p) as u32, 0, 0, 0]
    });
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
        time_loop(&queries, Threading::Single, |p| {
            [rsq.rank_unchecked(0, p) as u32, 0, 0, 0]
        });

        time_stream(
            &queries,
            Threading::Single,
            |p| {
                rsq.prefetch_data(p);
                rsq.prefetch_info(p)
            },
            |p| [rsq.rank_unchecked(0, p) as u32, 0, 0, 0],
        );
    }
    eprintln!();
}

fn bench_coro<const C3: bool>(seq: &[u8], queries: &QS) {
    let ranker = Ranker::<QuartBlock, NoSB>::new(&seq);
    time_coro_stream(&queries, |p| ranker.count_coro::<count4::U64Popcnt, C3>(p));
    time_coro_stream(&queries, |p| {
        ranker.count_coro::<count4::ByteLookup8, C3>(p)
    });
    time_coro_stream(&queries, |p| {
        ranker.count_coro::<count4::SimdCount7, false>(p)
    });
}

#[inline(never)]
fn bench_broken(seq: &[u8], queries: &QS) {
    bench::<PentaBlock20bit, TrivialSB, SimdCount7, false>(seq, queries);
    bench::<HexaBlock18bit, TrivialSB, WideSimdCount2, false>(seq, queries);
}

#[inline(never)]
fn bench_new(seq: &[u8], queries: &QS) {
    bench::<DumbBlock, TrivialSB, count4::U128Popcnt3, true>(seq, queries);
    bench::<DumbBlock, TrivialSB, count4::SimdCountSlice, false>(seq, queries);
    bench::<DumbBlock, SB8, count4::U128Popcnt3, true>(seq, queries);
    bench::<DumbBlock, SB8, count4::SimdCountSlice, false>(seq, queries);

    bench::<FullBlock, NoSB, count4::U64PopcntSlice, false>(seq, queries);
    bench::<QuartBlock, NoSB, count4::SimdCount7, false>(seq, queries);
    bench::<PentaBlock, TrivialSB, count4::SimdCount7, false>(seq, queries);
    bench::<HexaBlock, TrivialSB, count4::WideSimdCount2, false>(seq, queries);
}

fn main() {
    #[cfg(debug_assertions)]
    let q = 10_000;
    #[cfg(debug_assertions)]
    let ns = [100_000];
    #[cfg(not(debug_assertions))]
    let q = 10_000_000;
    #[cfg(not(debug_assertions))]
    let ns = [100_000, 1_000_000_000];

    for n in ns {
        // for n in [100_000] {
        eprintln!("n = {}", n);
        let seq = b"ACTG".repeat(n / 4);
        let queries = QS::default().map(|_| {
            (0..q)
                .map(|_| rand::random_range(0..seq.len()))
                .collect::<Vec<_>>()
        });

        bench_new(&seq, &queries);

        bench_qwt(&seq, &queries);

        // bench_rank9(&seq, &queries);

        // bench_dna_rank::<64>(&seq, &queries);
        // bench_dna_rank::<128>(&seq, &queries);
    }
}

// TODO: 40bit support
