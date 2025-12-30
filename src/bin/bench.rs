#![allow(incomplete_features, dead_code, unused)]
#![feature(generic_const_exprs, coroutines, coroutine_trait, stmt_expr_attributes)]
use std::{
    any::type_name,
    array::from_fn,
    ops::{Coroutine, CoroutineState::Complete},
    pin::pin,
};

use clap::Parser;
use quadrank::{
    binary::{
        self,
        blocks::{
            BinaryBlock, BinaryBlock2, BinaryBlock3, BinaryBlock4, BinaryBlock5, BinaryBlock6,
            Spider,
        },
    },
    genedex, prefetch_index,
    quad::Ranks,
    quad::blocks::{
        FullBlock, FullBlockMid, HexaBlock, HexaBlock2, HexaBlockMid, HexaBlockMid2, HexaBlockMid3,
        HexaBlockMid4, PentaBlock, Plain128, Plain256, Plain512, QuartBlock, TriBlock, TriBlock2,
    },
    quad::count4::{
        SimdCount7, SimdCount8, SimdCount9, SimdCount10, SimdCount11, SimdCount11B, SimdCountSlice,
        U64PopcntSlice, U128Popcnt3, WideSimdCount2,
    },
    quad::super_blocks::{NoSB, SB8, TrivialSB},
    quad::{Ranker, RankerT},
    sux::*,
};
use sux::prelude::Rank9;

type QS = Vec<Vec<usize>>;

#[derive(Clone, Copy)]
enum Threading {
    Single,
    Multi,
    Hyper,
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
    let ns = times[2] as f64 / queries[0].len() as f64;
    eprint!(" {ns:>7.2}");
    print!(",{ns:.2}");
}

fn time_fn_mt(queries: &[Vec<usize>], f: impl Fn(&[usize]) + Sync + Copy) {
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
    eprint!(" {ns:>7.2}");
    print!(",{ns:.2}");
}

fn time_fn(queries: &QS, t: Threading, f: impl Fn(&[usize]) + Sync + Copy) {
    match t {
        Threading::Single => time_fn_median(queries, f),
        Threading::Multi => time_fn_mt(&queries[0..6], f),
        Threading::Hyper => time_fn_mt(&queries[0..12], f),
    }
}

const BATCH: usize = 32;

fn time_latency(
    queries: &QS,
    t: Threading,
    prefetch: impl Fn(usize) + Sync,
    f: impl Fn(usize) -> usize + Sync + Copy,
) {
    time_fn(queries, t, |queries| {
        let mut acc = 0;
        for &q in queries {
            // Make query depend on previous result.
            let q = q ^ acc;
            prefetch(q);
            let rank = f(q);
            acc ^= (rank & 1) as usize;
        }
    });
}

fn time_loop(queries: &QS, t: Threading, f: impl Fn(usize) -> usize + Sync + Copy) {
    time_fn(queries, t, |queries| {
        for &q in queries {
            f(q);
        }
    });
}

fn time_batch(
    queries: &QS,
    t: Threading,
    prefetch: impl Fn(usize) + Sync,
    f: impl Fn(usize) -> usize + Sync,
) {
    time_fn(queries, t, |queries| {
        let qs = queries.as_chunks::<BATCH>().0;
        for batch in qs {
            for &q in batch {
                prefetch(q);
            }
            for &q in batch {
                f(q);
            }
        }
    })
}

fn time_stream(
    queries: &QS,
    t: Threading,
    prefetch: impl Fn(usize) + Sync,
    f: impl Fn(usize) -> usize + Sync,
) {
    time_fn(queries, t, |queries| {
        for i in 0..queries.len() - BATCH {
            unsafe {
                let q = *queries.get_unchecked(i);
                let ahead = *queries.get_unchecked(i + BATCH);
                // Prefetch next cacheline of queries.
                prefetch_index(queries, i + 2 * BATCH);
                prefetch(ahead);
                f(q);
            }
        }
    });
}

fn time_trip(
    queries: &QS,
    t: Threading,
    prefetch: impl Fn(usize) + Sync + Copy,
    f: impl Fn(usize) -> usize + Sync + Copy,
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
                std::hint::black_box(fq);
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
            std::hint::black_box(fq);

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
                std::hint::black_box(fq);
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
            std::hint::black_box(fq);

            // new future
            funcs[i % BATCH] = f(queries[i + BATCH]);
        }
    });
}

fn bench_header(threads: usize) {
    eprintln!(
        "{:<60} {:>11} {:>6} | {:>7} {:>7} {:>7} | {:>7} {:>7} {:>7} | {:>7} {:>7} {:>7} |",
        "Ranker",
        "n",
        "bits",
        "1t",
        "",
        "",
        format!("{}t", threads / 2),
        "",
        "",
        format!("{}t", threads),
        "",
        ""
    );
    eprintln!(
        "{:<60} {:>11} {:>6} | {:>7} {:>7} {:>7} | {:>7} {:>7} {:>7} | {:>7} {:>7} {:>7} |",
        "",
        "",
        "",
        "latncy",
        "loop",
        "stream",
        "latncy",
        "loop",
        "stream",
        "latncy",
        "loop",
        "stream"
    );
    println!(
        "ranker,sigma,n,bits,latency_1,loop_1,stream_1,latency_6,loop_6,stream_6,latency_12,loop_12,stream_12"
    );
}

fn bench<R: RankerT>(packed_seq: &[usize], queries: &QS) {
    let name = type_name::<R>();
    let name = regex::Regex::new(r"[a-zA-Z0-9_]+::")
        .unwrap()
        .replace_all(name, |_: &regex::Captures| "".to_string());

    eprint!("{name:<60}");
    let n = packed_seq.len() * 64;
    eprint!("{n:>11}");

    let ranker = R::new_packed(&packed_seq);
    let bits = (ranker.size() * 8) as f64 / (packed_seq.len() * 64) as f64;
    eprint!("{bits:>6.3}b |");
    print!("\"{name}\",4,{n},{bits:>.3}");

    for t in [Threading::Single, Threading::Multi, Threading::Hyper] {
        time_trip(
            &queries,
            t,
            |q| ranker.prefetch(q),
            |q| std::hint::black_box(unsafe { ranker.count(q) })[0] as usize,
        );
        eprint!(" |");
    }
    eprintln!();
    println!();
}

fn bench1<R: binary::RankerT>(packed_seq: &[usize], queries: &QS) {
    let name = type_name::<R>();
    let name = regex::Regex::new(r"[a-zA-Z0-9_]+::")
        .unwrap()
        .replace_all(name, |_: &regex::Captures| "".to_string());

    eprint!("{name:<60}");
    let n = packed_seq.len() * 64;
    eprint!("{n:>11}");

    let ranker = R::new_packed(&packed_seq);
    let bits = (ranker.size() * 8) as f64 / (packed_seq.len() * 64) as f64;
    eprint!("{bits:>6.3}b |");
    print!("\"{name}\",2,{n},{bits:>.3}");

    for t in [Threading::Single, Threading::Multi, Threading::Hyper] {
        time_trip(
            &queries,
            t,
            |q| ranker.prefetch(q),
            |q| std::hint::black_box(unsafe { ranker.rank_unchecked(q) as usize }),
        );
        eprint!(" |");
    }
    eprintln!();
    println!();
}

fn bench_coro<R: RankerT>(packed_seq: &[usize], queries: &QS) {
    let ranker = R::new_packed(&packed_seq);
    time_coro_stream(&queries, |p| ranker.count_coro(p));
    time_coro_stream(&queries, |p| ranker.count_coro(p));
    time_coro_stream(&queries, |p| ranker.count_coro(p));
}

#[inline(never)]
fn bench_all(seq: &[usize], queries: &QS) {
    bench_header(queries.len());
    eprintln!("QUAD");
    // plain external vec
    bench::<Ranker<Plain128, TrivialSB, WideSimdCount2, false>>(seq, queries);
    bench::<Ranker<Plain256, TrivialSB, SimdCountSlice, false>>(seq, queries);
    bench::<Ranker<Plain512, TrivialSB, SimdCountSlice, false>>(seq, queries);

    // like qwt
    bench::<Ranker<Plain512, SB8, U128Popcnt3, true>>(seq, queries);
    bench::<Ranker<Plain512, SB8, SimdCountSlice, false>>(seq, queries);

    // fast
    bench::<Ranker<FullBlock, NoSB, U64PopcntSlice, false>>(seq, queries);
    bench::<Ranker<FullBlockMid, NoSB, U64PopcntSlice, false>>(seq, queries);
    bench::<Ranker<FullBlockMid, NoSB, WideSimdCount2, false>>(seq, queries);
    bench::<Ranker<QuartBlock, NoSB, SimdCount8, false>>(seq, queries);
    bench::<Ranker<QuartBlock, NoSB, SimdCount9, false>>(seq, queries);
    bench::<Ranker<QuartBlock, NoSB, SimdCount10, false>>(seq, queries);
    bench::<Ranker<PentaBlock, TrivialSB, SimdCount8, false>>(seq, queries);
    bench::<Ranker<HexaBlock, TrivialSB, WideSimdCount2, false>>(seq, queries);
    bench::<Ranker<HexaBlock2, TrivialSB, WideSimdCount2, false>>(seq, queries);
    bench::<Ranker<HexaBlockMid, TrivialSB, SimdCount8, false>>(seq, queries);
    bench::<Ranker<HexaBlockMid, TrivialSB, SimdCount9, false>>(seq, queries);
    bench::<Ranker<HexaBlockMid2, TrivialSB, SimdCount9, false>>(seq, queries);
    bench::<Ranker<HexaBlockMid2, TrivialSB, SimdCount10, false>>(seq, queries);
    bench::<Ranker<HexaBlockMid3, TrivialSB, SimdCount10, false>>(seq, queries);
    bench::<Ranker<HexaBlockMid4, TrivialSB, SimdCount10, false>>(seq, queries);
    bench::<Ranker<TriBlock, TrivialSB, SimdCount11, false>>(seq, queries);
    bench::<Ranker<TriBlock, TrivialSB, SimdCount11B, false>>(seq, queries);
    bench::<Ranker<TriBlock2, TrivialSB, SimdCount11B, false>>(seq, queries);
    bench::<qwt::RSQVector256>(seq, queries);
    bench::<qwt::RSQVector512>(seq, queries);

    // broken
    // bench::<Ranker<PentaBlock20bit, TrivialSB, SimdCount7, false>>(seq, queries);
    // bench::<Ranker<HexaBlock18bit, TrivialSB, WideSimdCount2, false>>(seq, queries);

    eprintln!("BINARY");
    bench1::<binary::Ranker<BinaryBlock>>(seq, queries);
    bench1::<binary::Ranker<BinaryBlock2>>(seq, queries);
    bench1::<binary::Ranker<BinaryBlock3>>(seq, queries);
    bench1::<binary::Ranker<BinaryBlock4>>(seq, queries);
    bench1::<binary::Ranker<BinaryBlock5>>(seq, queries);
    bench1::<binary::Ranker<BinaryBlock6>>(seq, queries);
    // spider
    bench1::<binary::Ranker<Spider>>(seq, queries);
    // genedex
    bench1::<genedex::Flat64>(seq, queries);
    bench1::<genedex::Flat512>(seq, queries);
    bench1::<genedex::Condensed64>(seq, queries);
    bench1::<genedex::Condensed512>(seq, queries);
    // qwt
    bench1::<qwt::RSNarrow>(seq, queries);
    bench1::<qwt::RSWide>(seq, queries);
    // sux
    bench1::<Rank9>(seq, queries);
    bench1::<RankSmall1>(seq, queries);
    bench1::<RankSmall2>(seq, queries);
    bench1::<RankSmall3>(seq, queries);
    bench1::<RankSmall4>(seq, queries);
    bench1::<RankSmall5>(seq, queries);
}

#[derive(clap::Parser)]
struct Args {
    #[clap(short = 'n')]
    n: Option<usize>,
}

fn main() {
    #[cfg(debug_assertions)]
    let q = 10_000;
    #[cfg(debug_assertions)]
    let mut ns = vec![100_000];
    #[cfg(not(debug_assertions))]
    let q = 10_000_000;
    #[cfg(not(debug_assertions))]
    #[rustfmt::skip]
    let mut ns = vec![
        // 1_000_000, // L2
        // 64_000_000, // L3
        // 32_000_000_000, // RAM
        1_000_000_000,
    ];

    let args = Args::parse();
    if let Some(n) = args.n {
        ns = vec![n];
    }

    for n in ns {
        // for n in [100_000] {
        eprintln!("n = {}", n);
        // let seq = b"ACTG".repeat(n / 4);
        let seq = vec![
            // 0;
            // usize::MAX;
            0b1110010011100100111001001110010011100100111001001110010011100100;
            n.div_ceil(32)
        ];
        let queries = (0..12)
            .map(|_| (0..q).map(|_| rand::random_range(0..n)).collect::<Vec<_>>())
            .collect::<Vec<_>>();

        bench_all(&seq, &queries);
    }
}

// TODO: 40bit support
