#![allow(incomplete_features, dead_code, unused)]
#![feature(generic_const_exprs, coroutines, coroutine_trait, stmt_expr_attributes)]
use std::{
    any::type_name,
    array::from_fn,
    ops::{Coroutine, CoroutineState::Complete},
    pin::pin,
    sync::OnceLock,
};

use clap::Parser;
use prefetch_index::prefetch_index;
use quadrank::{
    binary::{
        self,
        blocks::{
            BinaryBlock16, BinaryBlock16Spider, BinaryBlock16x2, BinaryBlock23_9, BinaryBlock32,
            BinaryBlock32x2, BinaryBlock64x2,
        },
        super_blocks::ShiftSB,
    },
    genedex,
    quad::{
        LongRanks, Ranker, RankerT,
        blocks::{
            Basic128, Basic256, Basic512, QuadBlock7_18_7P, QuadBlock16, QuadBlock24_8,
            QuadBlock32, QuadBlock32_8_8_8FP, QuadBlock64,
        },
        count4::{
            NoCount, SimdCount7, SimdCount8, SimdCount9, SimdCount10, SimdCount11, SimdCount11B,
            SimdCountSlice, TransposedPopcount, U64PopcntSlice, U128Popcnt3, WideSimdCount2,
        },
        super_blocks::{NoSB, SB8, TrivialSB},
    },
    sux::*,
};
use sux::prelude::Rank9;

type QS = Vec<Vec<usize>>;

static THREADS: OnceLock<Vec<usize>> = OnceLock::new();

const REPEATS: usize = 3;

fn time_fn_mt(queries: &[Vec<usize>], f: impl Fn(&[usize]) + Sync + Copy) -> f64 {
    let start = std::time::Instant::now();
    std::thread::scope(|scope| {
        queries.iter().for_each(|queries| {
            let f = &f;
            scope.spawn(move || {
                f(queries);
            });
        });
    });
    start.elapsed().as_nanos() as f64 / (queries.len() * queries[0].len()) as f64
}

/// Take the minimum of 3 runs.
fn time_fn(queries: &QS, t: usize, f: impl Fn(&[usize]) + Sync + Copy) {
    let ns = (0..REPEATS)
        .map(|_| time_fn_mt(&queries[0..t], f))
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    eprint!(" {ns:>8.3}");
    print!(",{ns:.5}");
}

const BATCH: usize = 32;

fn time_latency(queries: &QS, t: usize, f: impl Fn(usize) -> usize + Sync + Copy) {
    time_fn(queries, t, |queries| {
        let mut acc = 0;
        for &q in queries {
            // Make query depend on previous result.
            let q = q ^ acc;
            let rank = f(q);
            acc ^= (rank & 1) as usize;
        }
    });
}

fn time_loop(queries: &QS, t: usize, f: impl Fn(usize) -> usize + Sync + Copy) {
    time_fn(queries, t, |queries| {
        for &q in queries {
            f(q);
        }
    });
}

fn time_batch(
    queries: &QS,
    t: usize,
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
    t: usize,
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
    t: usize,
    prefetch: impl Fn(usize) + Sync + Copy,
    f: impl Fn(usize) -> usize + Sync + Copy,
    stream: bool,
) {
    time_latency(queries, t, f);
    time_loop(queries, t, f);
    if stream {
        time_stream(queries, t, prefetch, f);
    } else {
        eprint!(" {:>8.3}", 0);
        print!(",{:.5}", 0);
    }
}

#[inline(always)]
fn time_coro2_batch<F>(queries: &QS, f: impl Fn(usize) -> F + Sync)
where
    F: Coroutine<Return = LongRanks> + Unpin,
{
    time_fn(queries, 1, |queries| {
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
    F: Coroutine<Return = LongRanks> + Unpin,
{
    time_fn(queries, 1, |queries| {
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
    F: Coroutine<Return = LongRanks> + Unpin,
{
    time_fn(queries, 1, |queries| {
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
    F: Coroutine<Return = LongRanks> + Unpin,
{
    time_fn(queries, 1, |queries| {
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

fn bench_header() {
    eprint!("{:<60} {:>11} {:>6} |", "Ranker", "n", "size",);
    for t in THREADS.wait() {
        eprint!(" {:>7}t {:>8} {:>8} |", t, "", "");
    }
    eprintln!();
    eprint!("{:<60} {:>11} {:>6} |", "", "", "");
    for t in THREADS.wait() {
        eprint!(" {:>8} {:>8} {:>8} |", "latncy", "loop", "stream",);
    }
    eprintln!();
    print!("ranker,sigma,n,rel_size,count4");
    for t in THREADS.wait() {
        print!(",latency_{},loop_{},stream_{}", t, t, t);
    }
    println!();
}

fn bench_one_quad<R: RankerT>(packed_seq: &[usize], queries: &QS) {
    let ranker = R::new_packed(&packed_seq);
    for count4 in [true, false] {
        let name = type_name::<R>();
        let name = regex::Regex::new(r"[a-zA-Z0-9_]+::")
            .unwrap()
            .replace_all(name, |_: &regex::Captures| "".to_string());

        eprint!("{name:<60} ");
        let n = packed_seq.len() * 64 / 2;
        eprint!("{n:>11} ");

        let rel_size = (ranker.size() * 8) as f64 / (packed_seq.len() * 64) as f64;
        eprint!("{rel_size:>5.3}x |");
        print!("\"{name}\",4,{n},{rel_size:>.3},{}", count4 as u8);

        for &t in THREADS.wait() {
            if count4 {
                time_trip(
                    &queries,
                    t,
                    |q| ranker.prefetch4(q),
                    |q| std::hint::black_box(unsafe { ranker.rank4(q) })[0] as usize,
                    true,
                );
            } else {
                time_trip(
                    &queries,
                    t,
                    |q| ranker.prefetch1(q, q as u8 & 3),
                    |q| std::hint::black_box(unsafe { ranker.rank1(q, q as u8 & 3) }),
                    true,
                );
            }
            eprint!(" |");
        }
        eprintln!();
        println!();
    }
}

fn bench_one_binary<R: binary::RankerT>(packed_seq: &[usize], queries: &QS) {
    let name = type_name::<R>();
    let name = regex::Regex::new(r"[a-zA-Z0-9_]+::")
        .unwrap()
        .replace_all(name, |_: &regex::Captures| "".to_string());

    eprint!("{name:<60} ");
    let n = packed_seq.len() * 64;
    eprint!("{n:>11} ");

    let ranker = R::new_packed(&packed_seq);
    let rel_size = (ranker.size() * 8) as f64 / (packed_seq.len() * 64) as f64;
    eprint!("{rel_size:>5.3}x |");
    print!("\"{name}\",2,{n},{rel_size:>.3},0");

    for &t in THREADS.wait() {
        time_trip(
            &queries,
            t,
            |q| ranker.prefetch(q),
            |q| std::hint::black_box(unsafe { ranker.rank_unchecked(q) as usize }),
            R::HAS_PREFETCH,
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
fn bench_quad(seq: &[usize], queries: &QS) {
    use quadrank::quad::super_blocks::ShiftSB;

    bench_header();

    bench_one_quad::<qwt::RSQVector256>(seq, queries);
    bench_one_quad::<qwt::RSQVector512>(seq, queries);

    bench_one_quad::<genedex::Flat64>(seq, queries);
    bench_one_quad::<genedex::Flat512>(seq, queries);
    bench_one_quad::<genedex::Condensed64>(seq, queries);
    bench_one_quad::<genedex::Condensed512>(seq, queries);

    bench_one_quad::<Ranker<QuadBlock64, ShiftSB, SimdCount11B>>(seq, queries);
    bench_one_quad::<Ranker<QuadBlock24_8, ShiftSB, SimdCount11B>>(seq, queries);
    bench_one_quad::<Ranker<QuadBlock16, ShiftSB, NoCount>>(seq, queries);
}

#[inline(never)]
fn bench_binary(seq: &[usize], queries: &QS) {
    bench_header();

    bench_one_binary::<qwt::RSNarrow>(seq, queries);
    bench_one_binary::<qwt::RSWide>(seq, queries);

    bench_one_binary::<genedex::Condensed64>(seq, queries);
    bench_one_binary::<genedex::Condensed512>(seq, queries);

    bench_one_binary::<bitm::RankSelect101111>(seq, queries);

    bench_one_binary::<Rank9>(seq, queries);
    bench_one_binary::<RankSmall0>(seq, queries);
    bench_one_binary::<RankSmall1>(seq, queries);
    bench_one_binary::<RankSmall2>(seq, queries);
    bench_one_binary::<RankSmall3>(seq, queries);
    bench_one_binary::<RankSmall4>(seq, queries);

    bench_one_binary::<binary::Ranker<BinaryBlock64x2>>(seq, queries);
    bench_one_binary::<binary::Ranker<BinaryBlock32x2>>(seq, queries);
    bench_one_binary::<binary::Ranker<BinaryBlock16x2>>(seq, queries);
    bench_one_binary::<binary::Ranker<BinaryBlock16>>(seq, queries);
    bench_one_binary::<binary::Ranker<BinaryBlock16Spider>>(seq, queries);
}

#[derive(clap::Parser)]
struct Args {
    /// Max number of threads
    #[clap(short = 'j')]
    threads: Option<usize>,
    #[clap(long)]
    to: Option<usize>,
    #[clap(short = 'n')]
    n: Option<usize>,
    #[clap(short = 'b')]
    binary: bool,
    #[clap(short = 'q')]
    quad: bool,
}

fn main() {
    let args = Args::parse();

    // queries per thread
    let q = 10_000_000;

    // size in bytes
    #[rustfmt::skip]
    let mut sizes = vec![
        128_000, // L2
        // 8_000_000, // L3
        4_000_000_000usize, // RAM
        // 8_000_000_000, // RAM
        // 16_000_000_000, // RAM
    ];

    let mut sizes = (13..=args.to.unwrap_or(32))
        .map(|i| 1usize << i)
        .collect::<Vec<_>>();

    THREADS.set({
        let mut ts = vec![];
        let mut t = args.threads.unwrap_or(12);
        loop {
            ts.push(t);
            if t == 1 {
                break;
            }
            t /= 2;
        }
        ts.reverse();
        ts
    });

    if let Some(n) = args.n {
        sizes = vec![n];
    }

    for size in sizes {
        eprintln!(
            "size = {} bytes = {} bits = {} bp",
            size,
            size * 8,
            size * 4
        );
        let seq = vec![
            0b1110010011100100111001001110010011100100111001001110010011100100;
            size.div_ceil(8)
        ];

        if args.binary {
            let n = size * 8;
            let queries = (0..*THREADS.wait().last().unwrap())
                .map(|_| (0..q).map(|_| rand::random_range(2..n)).collect::<Vec<_>>())
                .collect::<Vec<_>>();

            bench_binary(&seq, &queries);
        }
        if args.quad {
            let n = size * 4;
            let queries = (0..*THREADS.wait().last().unwrap())
                .map(|_| (0..q).map(|_| rand::random_range(2..n)).collect::<Vec<_>>())
                .collect::<Vec<_>>();

            bench_quad(&seq, &queries);
        }
    }
}

// TODO: 40bit support
