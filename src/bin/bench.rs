#![allow(incomplete_features, dead_code)]
#![feature(generic_const_exprs)]
use dna_rank::{BwaRank, BwaRank2, BwaRank3, BwaRank4, DnaRank, Ranks};
use mem_dbg::MemSize;

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

    time(&queries, |p| rank.ranks_u64_3(p));
    time(&queries, |p| rank.ranks_bytecount_16_all(p));
    time(&queries, |p| rank.ranks_simd_popcount(p));
    eprint!(" |");
    time_batch::<32>(&queries, |p| rank.prefetch(p), |p| rank.ranks_u64_3(p));
    time_batch::<32>(
        &queries,
        |p| rank.prefetch(p),
        |p| rank.ranks_bytecount_16_all(p),
    );
    time_batch::<32>(
        &queries,
        |p| rank.prefetch(p),
        |p| rank.ranks_simd_popcount(p),
    );
    eprint!(" |");
    time_stream(&queries, 32, |p| rank.prefetch(p), |p| rank.ranks_u64_3(p));
    time_stream(
        &queries,
        32,
        |p| rank.prefetch(p),
        |p| rank.ranks_bytecount_16_all(p),
    );
    time_stream(
        &queries,
        32,
        |p| rank.prefetch(p),
        |p| rank.ranks_simd_popcount(p),
    );
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
        bench_bwa3_rank(&seq, &queries);
        // bench_bwa2_rank(&seq, &queries);
        // bench_bwa_rank(&seq, &queries);
        // bench_dna_rank::<64>(&seq, &queries);
        // bench_dna_rank::<128>(&seq, &queries);
    }
}
