use std::hint::black_box;

use dna_rank::{DnaRank, Rank};

fn bench_dna_rank() {
    let n = 1_000_000;
    let q = 1_000_000;
    let seq = b"ACGT".repeat(n / 4);
    let rank = DnaRank::new(&seq);

    let queries = (0..q)
        .map(|_| rand::random_range(0..seq.len()))
        .collect::<Vec<_>>();

    let start = std::time::Instant::now();
    for q in &queries {
        black_box(rank.ranks(*q));
    }
    let duration = start.elapsed();
    eprintln!(
        "DnaRank: {} ns/query",
        duration.as_nanos() as f32 / q as f32
    );
}

fn main() {
    bench_dna_rank();
}
