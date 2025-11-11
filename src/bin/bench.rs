#![allow(incomplete_features, dead_code, unused)]
#![feature(generic_const_exprs, coroutines, coroutine_trait, stmt_expr_attributes)]
use std::{
    any::type_name,
    array::from_fn,
    ops::{Coroutine, CoroutineState::Complete},
    pin::pin,
};

use clap::Parser;
use dna_rank::{
    Ranks,
    blocks::{
        FullBlock, FullBlockMid, HexaBlock, HexaBlock2, HexaBlockMid, PentaBlock, Plain128,
        Plain256, Plain512, QuartBlock,
    },
    count4::{SimdCount7, SimdCountSlice, U64PopcntSlice, U128Popcnt3, WideSimdCount2},
    ranker::{Ranker, RankerT},
    super_block::{NoSB, SB8, TrivialSB},
};

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

type QS = Vec<Vec<usize>>;

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
    eprint!(" {ns2:>6.1}",);
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
    eprint!(" {ns:>6.1}",);
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

fn bench_header(threads: usize) {
    eprintln!(
        "{:<60} {:>6} | {:>6} {:>6} {:>6} | {:>6} {:>6} {:>6} |",
        "Ranker",
        "bits",
        "1t",
        "",
        "",
        format!("{threads}t",),
        "",
        ""
    );
    eprintln!(
        "{:<60} {:>6} | {:>6} {:>6} {:>6} | {:>6} {:>6} {:>6} |",
        "", "", "latncy", "loop", "stream", "latncy", "loop", "stream"
    );
}

fn bench<R: RankerT>(seq: &[u8], queries: &QS) {
    let name = type_name::<R>();
    let name = regex::Regex::new(r"[a-zA-Z0-9_]+::")
        .unwrap()
        .replace_all(name, |_: &regex::Captures| "".to_string());

    eprint!("{name:<60}");

    let ranker = R::new(&seq);
    let bits = (ranker.size() * 8) as f64 / seq.len() as f64;
    eprint!("{bits:>6.2}b |");

    for t in [Threading::Single, Threading::Multi] {
        time_trip(&queries, t, |p| ranker.prefetch(p), |p| ranker.count(p));
        eprint!(" |");
    }
    eprintln!();
}

fn bench_coro<R: RankerT>(seq: &[u8], queries: &QS) {
    let ranker = R::new(&seq);
    time_coro_stream(&queries, |p| ranker.count_coro(p));
    time_coro_stream(&queries, |p| ranker.count_coro(p));
    time_coro_stream(&queries, |p| ranker.count_coro(p));
}

#[inline(never)]
fn bench_all(seq: &[u8], queries: &QS) {
    bench_header(queries.len());
    // plain external vec
    // bench::<Ranker<Plain128, TrivialSB, WideSimdCount2, false>>(seq, queries);
    // bench::<Ranker<Plain256, TrivialSB, SimdCountSlice, false>>(seq, queries);
    // bench::<Ranker<Plain512, TrivialSB, SimdCountSlice, false>>(seq, queries);

    // // like qwt
    // bench::<Ranker<Plain512, SB8, U128Popcnt3, true>>(seq, queries);
    // bench::<Ranker<Plain512, SB8, SimdCountSlice, false>>(seq, queries);

    // fast
    bench::<Ranker<FullBlock, NoSB, U64PopcntSlice, false>>(seq, queries);
    bench::<Ranker<FullBlockMid, NoSB, U64PopcntSlice, false>>(seq, queries);
    bench::<Ranker<FullBlockMid, NoSB, WideSimdCount2, false>>(seq, queries);
    bench::<Ranker<QuartBlock, NoSB, SimdCount7, false>>(seq, queries);
    bench::<Ranker<PentaBlock, TrivialSB, SimdCount7, false>>(seq, queries);
    bench::<Ranker<HexaBlock, TrivialSB, WideSimdCount2, false>>(seq, queries);
    bench::<Ranker<HexaBlock2, TrivialSB, WideSimdCount2, false>>(seq, queries);
    bench::<Ranker<HexaBlockMid, TrivialSB, SimdCount7, false>>(seq, queries);

    // external
    // #[cfg(not(debug_assertions))]
    // bench::<sux::prelude::Rank9>(seq, queries);
    // bench::<qwt::RSQVector256>(seq, queries);

    // broken
    // bench::<Ranker<PentaBlock20bit, TrivialSB, SimdCount7, false>>(seq, queries);
    // bench::<Ranker<HexaBlock18bit, TrivialSB, WideSimdCount2, false>>(seq, queries);
}

#[derive(clap::Parser)]
struct Args {
    #[clap(short = 'j', long, default_value_t = 6)]
    threads: usize,
    #[clap(short = 'n', default_value_t = 1_000_000_000)]
    n: usize,
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

    let args = Args::parse();
    let threads = args.threads;
    // let n = args.n;

    for n in ns {
        // for n in [100_000] {
        eprintln!("n = {}", n);
        let seq = b"ACTG".repeat(n / 4);
        let queries = (0..threads.max(5))
            .map(|_| {
                (0..q)
                    .map(|_| rand::random_range(0..seq.len()))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        bench_all(&seq, &queries);
    }
}

// TODO: 40bit support
