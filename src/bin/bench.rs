use std::hint::black_box;

use dna_rank::DnaRank;
use mem_dbg::MemSize;

fn bench_dna_rank<const STRIDE: usize>() {
    let n = 1_000_000;
    let q = 1_000_000;
    let seq = b"ACGT".repeat(n / 4);
    let rank = DnaRank::<STRIDE>::new(&seq);

    let queries = (0..q)
        .map(|_| rand::random_range(0..seq.len()))
        .collect::<Vec<_>>();

    let start = std::time::Instant::now();
    for q in &queries {
        black_box(rank.ranks(*q));
    }
    let duration = start.elapsed();
    let ns = duration.as_nanos() as f64 / q as f64;
    let bits = (rank.mem_size(Default::default()) * 8) as f64 / n as f64;
    eprintln!("DnaRank<{STRIDE:>4}>: {ns:>5.1} ns/query    {bits:>6.2} bits/elem",);
}

fn main() {
    bench_dna_rank::<4>();
    bench_dna_rank::<8>();
    bench_dna_rank::<16>();
    bench_dna_rank::<32>();
    bench_dna_rank::<64>();
    bench_dna_rank::<128>();
    bench_dna_rank::<256>();
    bench_dna_rank::<512>();
    bench_dna_rank::<1024>();
}
