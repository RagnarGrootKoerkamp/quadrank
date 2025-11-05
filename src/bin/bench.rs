#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
use dna_rank::{DnaRank, Ranks};
use mem_dbg::MemSize;

#[inline(never)]
fn check(pos: usize, ranks: Ranks) {
    assert_eq!(pos, ranks.iter().sum());
}

fn bench_dna_rank<const STRIDE: usize>()
where
    [(); STRIDE / 4]:,
{
    let n = 1_000_000_000;
    let q = 1_000_000;
    let seq = b"ACGT".repeat(n / 4);
    let rank = DnaRank::<STRIDE>::new(&seq);

    let bits = (rank.mem_size(Default::default()) * 8) as f64 / n as f64;

    let queries = (0..q)
        .map(|_| rand::random_range(0..seq.len()))
        .collect::<Vec<_>>();

    let start = std::time::Instant::now();
    for &q in &queries {
        check(q, rank.ranks_naive(q));
    }
    let ns_naive = start.elapsed().as_nanos() as f64 / q as f64;

    let start = std::time::Instant::now();
    for &q in &queries {
        check(q, rank.ranks_u64(q));
    }
    let ns_u64 = start.elapsed().as_nanos() as f64 / q as f64;

    let start = std::time::Instant::now();
    for &q in &queries {
        check(q, rank.ranks_u64_prefetch(q));
    }
    let ns_u64_pf = start.elapsed().as_nanos() as f64 / q as f64;

    eprintln!("DnaRank<{STRIDE:>4}>:  {bits:>6.2} bit  {ns_naive:>5.1} ns  {ns_u64:>5.1} ns  {ns_u64_pf:>5.1} ns  ",);
}

fn main() {
    bench_dna_rank::<32>();
    bench_dna_rank::<64>();
    bench_dna_rank::<128>();
    bench_dna_rank::<256>();
    bench_dna_rank::<512>();
    bench_dna_rank::<1024>();
    bench_dna_rank::<2048>();
}
