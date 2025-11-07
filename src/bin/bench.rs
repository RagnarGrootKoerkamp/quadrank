#![allow(incomplete_features, dead_code)]
#![feature(generic_const_exprs, coroutines, coroutine_trait, stmt_expr_attributes)]
use std::{
    array::from_fn,
    mem::MaybeUninit,
    ops::{Coroutine, CoroutineState::Complete},
    pin::{Pin, pin},
    task::Context,
};

use dna_rank::{DnaRank, Ranks, blocks::QuartBlock, count4, ranker::Ranker};
use futures::{future::join_all, stream::FuturesOrdered, task::noop_waker_ref};
use mem_dbg::MemSize;
use smol::{LocalExecutor, future::poll_once, stream::StreamExt};
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

fn time(queries: &QS, f: impl Fn(usize) -> Ranks) {
    time_fn(queries, |queries| {
        for &q in queries {
            check(q, f(q));
        }
    });
}

fn time_batch<const BATCH: usize>(
    queries: &QS,
    prefetch: impl Fn(usize),
    f: impl Fn(usize) -> Ranks,
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
    prefetch: impl Fn(usize),
    f: impl Fn(usize) -> Ranks,
) {
    time_fn(queries, |queries| {
        for (&q, &ahead) in queries.iter().zip(&queries[lookahead..]) {
            prefetch(ahead);
            check(q, f(q));
        }
    });
}

fn time_async_one_task<F>(queries: &QS, _lookahead: usize, f: impl Fn(usize) -> F)
where
    F: Future<Output = Ranks>,
{
    time_fn(queries, |queries| {
        let local_ex = LocalExecutor::new();

        smol::future::block_on(local_ex.run(async {
            let mut handles: [_; 32] = from_fn(
                #[inline(always)]
                |i| (queries[i], f(queries[i])),
            );
            for (i, &q) in queries[32..].iter().enumerate() {
                let newhandle = f(q);

                let (q, handle) = std::mem::replace(&mut handles[i % 32], (q, newhandle));
                let fq = handle.await;
                check(q, fq);
            }
            // for (q, handle) in handles {
            //     let fq = handle.await;
            //     check(q, fq);
            // }
        }));
    });
}

fn time_async_futures_ordered<F>(queries: &QS, _lookahead: usize, f: impl Fn(usize) -> F)
where
    F: Future<Output = Ranks>,
{
    time_fn(queries, |queries| {
        let local_ex = LocalExecutor::new();

        smol::future::block_on(local_ex.run(async {
            let cx = &mut Context::from_waker(noop_waker_ref());
            for batch in queries.as_chunks::<32>().0 {
                let mut futures: FuturesOrdered<_> = batch.iter().map(|&q| f(q)).collect();

                for &q in batch {
                    let fq = loop {
                        match futures.poll_next(cx) {
                            std::task::Poll::Ready(fq) => break fq.unwrap(),
                            std::task::Poll::Pending => continue,
                        }
                    };
                    check(q, fq);
                }
            }
        }));
    });
}

fn time_async_join_all_batch<F>(queries: &QS, _lookahead: usize, f: impl Fn(usize) -> F)
where
    F: Future<Output = Ranks>,
{
    time_fn(queries, |queries| {
        let local_ex = LocalExecutor::new();

        smol::future::block_on(local_ex.run(async {
            for batch in queries.as_chunks::<16>().0 {
                let futures = batch.iter().map(|&q| f(q));
                // NOTE: This needs batches of size <30 to avoid switching to a heavy implementation.
                for (&q, fq) in batch.iter().zip(join_all(futures).await) {
                    check(q, fq);
                }
            }
        }));
    });
}

/// copied from futures crate
fn iter_pin_mut<T>(slice: Pin<&mut [T]>) -> impl Iterator<Item = Pin<&mut T>> {
    // Safety: `std` _could_ make this unsound if it were to decide Pin's
    // invariants aren't required to transmit through slices. Otherwise this has
    // the same safety as a normal field pin projection.
    unsafe { slice.get_unchecked_mut() }
        .iter_mut()
        .map(|t| unsafe { Pin::new_unchecked(t) })
}
fn pin_index<T>(slice: Pin<&mut [T]>, index: usize) -> Pin<&mut T> {
    unsafe { Pin::new_unchecked(&mut slice.get_unchecked_mut()[index]) }
}

#[inline(always)]
async fn async_batches<F>(queries: &[usize], f: impl Fn(usize) -> F)
where
    F: Future<Output = Ranks>,
{
    // eprintln!("size of future: {}", std::mem::size_of::<F>());
    for batch in queries.as_chunks::<32>().0 {
        let mut futures: Pin<&mut [_; 32]> = pin!(from_fn(|i| f(batch[i])));

        // for f in iter_pin_mut(futures.as_mut()) {
        //     assert!(poll_once(f).await.is_none());
        // }
        for f in iter_pin_mut(futures.as_mut()) {
            assert!(poll_once(f).await.is_some());
        }
    }
}

#[inline(always)]
async fn async_stream<F>(queries: &[usize], f: impl Fn(usize) -> F)
where
    F: Future<Output = Ranks>,
{
    let mut futures: [MaybeUninit<(usize, F)>; 32] = from_fn(|_| MaybeUninit::uninit());
    for i in 0..32 {
        futures[i] = MaybeUninit::new((queries[i], f(queries[i])));
        // let pin = unsafe { Pin::new_unchecked(&mut futures[i].assume_init_mut().1) };
        // assert!(poll_once(pin).await.is_none());
    }

    for (i, &q) in queries.iter().enumerate() {
        // finish the old state
        {
            let (q, future) = unsafe { futures[i % 32].assume_init_mut() };
            let pin = unsafe { Pin::new_unchecked(future) };
            // let fq = poll_once(pin).await.unwrap();
            let fq = pin.await;
            check(*q, fq);
        }

        // new future
        {
            futures[i % 32] = MaybeUninit::new((q, f(q)));
            // let pin = unsafe { Pin::new_unchecked(&mut futures[i % 32].assume_init_mut().1) };
            // assert!(poll_once(pin).await.is_none());
        }
    }
}

#[inline(always)]
fn time_async_smol_batch<F>(queries: &QS, _lookahead: usize, f: impl Fn(usize) -> F)
where
    F: Future<Output = Ranks>,
{
    let f = &f;
    time_fn(queries, |queries| {
        let local_ex = LocalExecutor::new();
        smol::future::block_on(local_ex.run(async move { async_batches(queries, f).await }));
    });
}

#[inline(always)]
fn time_async_cassette_batch<F>(queries: &QS, _lookahead: usize, f: impl Fn(usize) -> F)
where
    F: Future<Output = Ranks>,
{
    let f = &f;
    time_fn(queries, |queries| {
        let future = core::pin::pin!(async { async_batches(queries, f).await });
        cassette::block_on(future);
    });
}

#[inline(always)]
fn time_async_smol_stream<F>(queries: &QS, _lookahead: usize, f: impl Fn(usize) -> F)
where
    F: Future<Output = Ranks>,
{
    let f = &f;
    time_fn(queries, |queries| {
        let local_ex = LocalExecutor::new();
        smol::future::block_on(local_ex.run(async { async_stream(queries, f).await }));
    });
}

#[inline(always)]
fn time_async_cassette_stream<F>(queries: &QS, _lookahead: usize, f: impl Fn(usize) -> F)
where
    F: Future<Output = Ranks>,
{
    let f = &f;
    time_fn(queries, |queries| {
        let future = core::pin::pin!(async { async_stream(queries, f).await });
        cassette::block_on(future);
    });
}

#[inline(always)]
fn time_coro2_batch<F>(queries: &QS, _lookahead: usize, f: impl Fn(usize) -> F)
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
fn time_coro2_stream<F>(queries: &QS, _lookahead: usize, f: impl Fn(usize) -> F)
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
fn time_coro_batch<F>(queries: &QS, _lookahead: usize, f: impl Fn(usize) -> F)
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
fn time_coro_stream<F>(queries: &QS, _lookahead: usize, f: impl Fn(usize) -> F)
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
fn bench_quart<const C3: bool>(seq: &[u8], queries: &QS) {
    eprint!("{:<20}:", format!("QuartBlock {C3}"));
    let bits = 4.0;
    eprint!("{bits:>6.2}b |");

    let ranker = Ranker::<QuartBlock>::new(&seq);

    // time_fn(queries, |queries| {
    //     for &q in queries {
    //         std::hint::black_box(ranker.count1(q, q as u8 & 3));
    //     }
    // });

    time(&queries, |p| ranker.count::<count4::U64Popcnt, C3>(p));

    eprint!(" |");

    time_stream(
        &queries,
        B,
        |p| ranker.prefetch(p),
        |p| ranker.count::<count4::U64Popcnt, C3>(p),
    );
    // time_stream(
    //     &queries,
    //     B,
    //     |p| ranker.prefetch(p),
    //     |p| ranker.count::<count4::ByteLookup8, C3>(p),
    // );
    // time_stream(
    //     &queries,
    //     B,
    //     |p| ranker.prefetch(p),
    //     |p| ranker.count::<count4::SimdCount, false>(p),
    // );

    // eprint!(" |");

    // time_coro_stream(&queries, B, |p| {
    //     ranker.count_coro::<count4::U64Popcnt, C3>(p)
    // });
    // time_coro_stream(&queries, B, |p| {
    //     ranker.count_coro::<count4::ByteLookup8, C3>(p)
    // });
    // time_coro_stream(&queries, B, |p| {
    //     ranker.count_coro::<count4::SimdCount, false>(p)
    // });
    eprintln!();
}

fn main() {
    #[cfg(debug_assertions)]
    let q = 10_000;
    #[cfg(debug_assertions)]
    let ns = [100_000];
    #[cfg(not(debug_assertions))]
    let q = 10_000_000;
    #[cfg(not(debug_assertions))]
    let ns = [100_000, 10_000_000, 1_000_000_000];

    for n in ns {
        // for n in [100_000] {
        eprintln!("n = {}", n);
        let seq = b"ACTG".repeat(n / 4);
        let queries = [(); 5].map(|_| {
            (0..q)
                .map(|_| rand::random_range(0..seq.len()))
                .collect::<Vec<_>>()
        });

        // bench_quart::<true>(&seq, &queries);
        bench_quart::<false>(&seq, &queries);
        // bench_best(&seq, &queries);
        // bench_rank9(&seq, &queries);

        // bench_dna_rank::<64>(&seq, &queries);
        // bench_dna_rank::<128>(&seq, &queries);
    }
}
