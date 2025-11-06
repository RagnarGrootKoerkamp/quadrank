#![allow(incomplete_features, dead_code)]
#![feature(generic_const_exprs)]
use std::{array::from_fn, future::poll_fn, pin::Pin, task::Context};

use cassette::Cassette;
use dna_rank::{BwaRank, BwaRank2, BwaRank3, BwaRank4, DnaRank, Ranks};
use futures::{future::join_all, stream::FuturesOrdered, task::noop_waker_ref};
use mem_dbg::MemSize;
use smol::{LocalExecutor, future::poll_once, stream::StreamExt};

fn check(pos: usize, ranks: Ranks) {
    std::hint::black_box(&ranks);
    let pos = pos as u32;
    debug_assert_eq!(
        ranks,
        [(pos + 3) / 4, (pos + 2) / 4, (pos + 1) / 4, pos / 4],
    );
}

fn time(queries: &[usize], f: impl Fn(usize) -> Ranks) {
    let start = std::time::Instant::now();
    for &q in queries {
        check(q, f(q));
    }
    let ns = start.elapsed().as_nanos() as f64 / queries.len() as f64;
    eprint!(" {ns:>5.1}",);
}

fn time_batch<const BATCH: usize>(
    queries: &[usize],
    prefetch: impl Fn(usize),
    f: impl Fn(usize) -> Ranks,
) {
    let start = std::time::Instant::now();
    let qs = queries.as_chunks::<BATCH>().0;
    for batch in qs {
        for &q in batch {
            prefetch(q);
        }
        for &q in batch {
            check(q, f(q));
        }
    }
    let q = BATCH * qs.len();
    let ns = start.elapsed().as_nanos() as f64 / q as f64;
    eprint!(" {ns:>5.1}",);
}

fn time_stream(
    queries: &[usize],
    lookahead: usize,
    prefetch: impl Fn(usize),
    f: impl Fn(usize) -> Ranks,
) {
    let start = std::time::Instant::now();
    for (&q, &ahead) in queries.iter().zip(&queries[lookahead..]) {
        prefetch(ahead);
        check(q, f(q));
    }
    let ns = start.elapsed().as_nanos() as f64 / queries.len() as f64;
    eprint!(" {ns:>5.1}",);
}

fn time_async_one_task<F>(queries: &[usize], _lookahead: usize, f: impl Fn(usize) -> F)
where
    F: Future<Output = Ranks>,
{
    let start = std::time::Instant::now();

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

    let ns = start.elapsed().as_nanos() as f64 / queries.len() as f64;
    eprint!(" {ns:>5.1}",);
}

fn time_async_futures_ordered<F>(queries: &[usize], _lookahead: usize, f: impl Fn(usize) -> F)
where
    F: Future<Output = Ranks>,
{
    let start = std::time::Instant::now();

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

    let ns = start.elapsed().as_nanos() as f64 / queries.len() as f64;
    eprint!(" {ns:>5.1}",);
}

fn time_async_join_all_batch<F>(queries: &[usize], _lookahead: usize, f: impl Fn(usize) -> F)
where
    F: Future<Output = Ranks>,
{
    let start = std::time::Instant::now();

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

    let ns = start.elapsed().as_nanos() as f64 / queries.len() as f64;
    eprint!(" {ns:>5.1}",);
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

async fn async_batches<F>(queries: &[usize], f: impl Fn(usize) -> F)
where
    F: Future<Output = Ranks>,
{
    for batch in queries.as_chunks::<32>().0 {
        let mut futures: Pin<&mut [_; 32]> = std::pin::pin!(from_fn(|i| f(batch[i])));

        for f in iter_pin_mut(futures.as_mut()) {
            assert!(poll_once(f).await.is_none());
        }
        for f in iter_pin_mut(futures.as_mut()) {
            assert!(poll_once(f).await.is_some());
        }
    }
}

fn time_async_manual_join_all_batch<F>(queries: &[usize], _lookahead: usize, f: impl Fn(usize) -> F)
where
    F: Future<Output = Ranks>,
{
    let start = std::time::Instant::now();
    let local_ex = LocalExecutor::new();
    smol::future::block_on(local_ex.run(async { async_batches(queries, f).await }));
    let ns = start.elapsed().as_nanos() as f64 / queries.len() as f64;
    eprint!(" {ns:>5.1}",);
}

fn time_async_cassette<F>(queries: &[usize], _lookahead: usize, f: impl Fn(usize) -> F)
where
    F: Future<Output = Ranks>,
{
    let start = std::time::Instant::now();
    let future = core::pin::pin!(async { async_batches(queries, f).await });
    cassette::block_on(future);
    let ns = start.elapsed().as_nanos() as f64 / queries.len() as f64;
    eprint!(" {ns:>5.1}",);
}

#[inline(never)]
fn bench_dna_rank<const STRIDE: usize>(seq: &[u8], queries: &[usize])
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
fn bench_bwa_rank(seq: &[u8], queries: &[usize]) {
    eprint!("{:<20}:", "BwaRank");
    let rank = BwaRank::new(&seq);

    let bits = (rank.mem_size(Default::default()) * 8) as f64 / seq.len() as f64;
    eprint!("{bits:>6.2}b |");

    // time(&queries, |p| rank.ranks_u64(p));
    // time(&queries, |p| rank.ranks_u64_all(p));
    time(&queries, |p| rank.ranks_u64_3(p)); // best

    // time(&queries, |p| rank.ranks_u128(p));
    // time(&queries, |p| rank.ranks_u128_3(p)); // 2nd best
    // time(&queries, |p| rank.ranks_u128_all(p));
    // time(&queries, |p| rank.ranks_bytecount(p));
    time(&queries, |p| rank.ranks_bytecount_4(p)); // original
    // time(&queries, |p| rank.ranks_bytecount_8(p));
    time(&queries, |p| rank.ranks_bytecount_16(p));
    time(&queries, |p| rank.ranks_bytecount_16_all(p)); // bad codegen?
    eprintln!();
}

#[inline(never)]
fn bench_bwa2_rank(seq: &[u8], queries: &[usize]) {
    eprint!("{:<20}:", "BwaRank2");
    let rank = BwaRank2::new(&seq);

    // let bits = (rank.mem_size(Default::default()) * 8) as f64 / seq.len() as f64;
    let bits = 4.0;
    eprint!("{bits:>6.2}b |");

    time(&queries, |p| rank.ranks_u128_3(p)); // overall fastest
    time(&queries, |p| rank.ranks_bytecount_16_all(p));
    time(&queries, |p| rank.ranks_simd_popcount(p));
    eprintln!();
}

#[inline(never)]
fn bench_bwa3_rank(seq: &[u8], queries: &[usize]) {
    eprint!("{:<20}:", "BwaRank3");
    let rank = BwaRank3::new(&seq);

    // let bits = (rank.mem_size(Default::default()) * 8) as f64 / seq.len() as f64;
    let bits = 4.0;
    eprint!("{bits:>6.2}b |");

    time(&queries, |p| rank.ranks_u128_3(p)); // overall fastest
    time(&queries, |p| rank.ranks_bytecount_16_all(p));
    time(&queries, |p| rank.ranks_simd_popcount(p));
    eprintln!();
}

#[inline(never)]
fn bench_bwa4_rank(seq: &[u8], queries: &[usize]) {
    eprint!("{:<20}:", "BwaRank4");
    let rank = BwaRank4::new(&seq);

    // let bits = (rank.mem_size(Default::default()) * 8) as f64 / seq.len() as f64;
    let bits = 4.0;
    eprint!("{bits:>6.2}b |");

    time(&queries, |p| rank.ranks_u64_popcnt(p));
    // time(&queries, |p| rank.ranks_bytecount_16_all(p));
    // time(&queries, |p| rank.ranks_simd_popcount(p));
    eprint!(" |");
    time_batch::<32>(&queries, |p| rank.prefetch(p), |p| rank.ranks_u64_popcnt(p));
    eprint!(" |");
    time_stream(
        &queries,
        32,
        |p| rank.prefetch(p),
        |p| rank.ranks_u64_popcnt(p),
    );
    // eprint!(" |");
    // time_async_one_task(
    //     &queries,
    //     32,
    //     |p| rank.ranks_u64_popcnt_async(p),
    // );
    // eprint!(" |");
    // time_async_futures_ordered(
    //     &queries,
    //     32,
    //     |p| rank.ranks_u64_popcnt_async(p),
    // );
    eprint!(" |");
    // time_async_join_all_batch(&queries, 32, |p| rank.ranks_u64_popcnt_async(p));
    // time_async_manual_join_all_batch(&queries, 32, |p| rank.ranks_u64_popcnt_async(p));
    // time_async_manual_join_all_batch(&queries, 32, |p| rank.ranks_u64_popcnt_async_nowake(p));
    // time_async_cassette(&queries, 32, |p| rank.ranks_u64_popcnt_async(p));
    time_async_cassette(&queries, 32, |p| rank.ranks_u64_popcnt_async_nowake(p));
    eprintln!();
}

fn main() {
    #[cfg(debug_assertions)]
    let q = 100_000;
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
        let queries = (0..q)
            .map(|_| rand::random_range(0..seq.len()))
            .collect::<Vec<_>>();

        bench_bwa4_rank(&seq, &queries);
        // bench_bwa3_rank(&seq, &queries);
        // bench_bwa2_rank(&seq, &queries);
        // bench_bwa_rank(&seq, &queries);
        // bench_dna_rank::<64>(&seq, &queries);
        // bench_dna_rank::<128>(&seq, &queries);
    }
}
