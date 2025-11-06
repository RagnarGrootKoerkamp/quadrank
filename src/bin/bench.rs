#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
use dna_rank::{BwaRank, BwaRank2, DnaRank, Ranks};
use mem_dbg::MemSize;

fn check(_pos: usize, ranks: Ranks) {
    std::hint::black_box(&ranks);
    // assert_eq!(pos as u32, ranks.iter().sum());
}

fn time(queries: &[usize], f: impl Fn(usize) -> Ranks) {
    let start = std::time::Instant::now();
    for &q in queries {
        check(q, f(q));
    }
    let ns = start.elapsed().as_nanos() as f64 / queries.len() as f64;
    eprint!(" {ns:>5.1}",);
}

fn bench_dna_rank<const STRIDE: usize>(seq: &[u8], queries: &[usize])
where
    [(); STRIDE / 4]:,
{
    eprint!("{:<20}:", format!("DnaRank<{STRIDE:>4}>"));
    let rank = DnaRank::<STRIDE>::new(&seq);

    let bits = (rank.mem_size(Default::default()) * 8) as f64 / seq.len() as f64;
    eprint!("{bits:>6.2}b |");

    time(&queries, |p| rank.ranks_naive(p));
    time(&queries, |p| rank.ranks_u64(p));
    time(&queries, |p| rank.ranks_u64_prefetch(p));
    time(&queries, |p| rank.ranks_u64_prefetch_all(p));
    time(&queries, |p| rank.ranks_u64_3(p)); // best
    time(&queries, |p| rank.ranks_u128(p));
    time(&queries, |p| rank.ranks_u128_3(p));
    eprintln!();
}

fn bench_bwa_rank(seq: &[u8], queries: &[usize]) {
    eprint!("{:<20}:", "BwaRank");
    let rank = BwaRank::new(&seq);

    let bits = (rank.mem_size(Default::default()) * 8) as f64 / seq.len() as f64;
    eprint!("{bits:>6.2}b |");

    time(&queries, |p| rank.ranks_u64(p));
    time(&queries, |p| rank.ranks_u64_all(p));
    time(&queries, |p| rank.ranks_u64_3(p)); // best
    time(&queries, |p| rank.ranks_u128(p));
    time(&queries, |p| rank.ranks_u128_3(p)); // 2nd best
    time(&queries, |p| rank.ranks_u128_all(p));
    time(&queries, |p| rank.ranks_bytecount(p));
    time(&queries, |p| rank.ranks_bytecount_4(p)); // original
    time(&queries, |p| rank.ranks_bytecount_8(p));
    time(&queries, |p| rank.ranks_bytecount_16(p));
    time(&queries, |p| rank.ranks_bytecount_16_all(p));
    eprintln!();
}

fn bench_bwa2_rank(seq: &[u8], queries: &[usize]) {
    eprint!("{:<20}:", "BwaRank");
    let rank = BwaRank2::new(&seq);

    let bits = (rank.mem_size(Default::default()) * 8) as f64 / seq.len() as f64;
    eprint!("{bits:>6.2}b |");

    time(&queries, |p| rank.ranks_u128_3(p));
    eprintln!();
}

fn main() {
    let q = 10_000_000;
    for n in [100_000, 10_000_000, 1_000_000_000] {
        eprintln!("n = {}", n);
        let seq = b"ACGT".repeat(n / 4);
        let queries = (0..q)
            .map(|_| rand::random_range(0..seq.len()))
            .collect::<Vec<_>>();

        bench_bwa2_rank(&seq, &queries);
        bench_bwa_rank(&seq, &queries);
        bench_dna_rank::<64>(&seq, &queries);
        bench_dna_rank::<128>(&seq, &queries);
    }
}
