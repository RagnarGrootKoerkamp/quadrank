#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
use dna_rank::{DnaRank, Ranks};
use mem_dbg::MemSize;

#[inline(never)]
fn check(pos: usize, ranks: Ranks) {
    assert_eq!(pos as u32, ranks.iter().sum());
}

fn bench_dna_rank<const STRIDE: usize>()
where
    [(); STRIDE / 4]:,
{
    eprint!("DnaRank<{STRIDE:>4}>:");
    let n = 1_000_000_000;
    let q = 1_000_000;
    let seq = b"ACGT".repeat(n / 4);
    let rank = DnaRank::<STRIDE>::new(&seq);

    let bits = (rank.mem_size(Default::default()) * 8) as f64 / n as f64;
    eprint!("{bits:>6.2}b |");

    let queries = (0..q)
        .map(|_| rand::random_range(0..seq.len()))
        .collect::<Vec<_>>();

    fn time(queries: &[usize], f: impl Fn(usize) -> Ranks) {
        let start = std::time::Instant::now();
        for &q in queries {
            check(q, f(q));
        }
        let ns = start.elapsed().as_nanos() as f64 / queries.len() as f64;
        eprint!(" {ns:>5.1}",);
    }

    time(&queries, |p| rank.ranks_naive(p));
    time(&queries, |p| rank.ranks_u64(p));
    time(&queries, |p| rank.ranks_u64_prefetch(p));
    time(&queries, |p| rank.ranks_u128(p));
    eprintln!();
}

fn main() {
    bench_dna_rank::<64>();
    bench_dna_rank::<128>();
    bench_dna_rank::<256>();
    bench_dna_rank::<512>();
    bench_dna_rank::<1024>();
    bench_dna_rank::<2048>();
}
