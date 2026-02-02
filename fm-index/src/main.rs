#![allow(incomplete_features, unused)]
#![feature(generic_const_exprs, array_windows, iter_next_chunk)]

mod bwt;
#[cfg(test)]
mod caps_sa_test;
mod quad_fm;
#[cfg(test)]
mod test;

mod ext {
    mod awry;
    mod fm_crate;
    mod genedex;
}

use bwt::{BWT, DiskBWT, build_bwt_packed, pack_text, read_text};
use clap::Parser;
use quad_fm::QuadFm;
use quadrank::quad::*;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::{path::PathBuf, process::exit, sync::atomic::AtomicUsize, time::Duration};

trait FmIndex: Sized + Sync {
    /// text is values 0..4, one per byte.
    fn new_with_prefix(text: &[u8], bwt: &bwt::BWT, prefix: usize) -> Self;
    fn new(text: &[u8], bwt: &bwt::BWT) -> Self {
        Self::new_with_prefix(text, bwt, 0)
    }
    fn prep_read(_read: &mut [u8]) {}
    /// Size in bytes.
    fn size(&self) -> usize;
    /// Count the number of matches.
    fn count(&self, text: &[u8]) -> usize;
    const HAS_BATCH: bool;
    const HAS_PREFETCH: bool;
    /// Batch size B, and whether to PREFETCH or not.
    fn count_batch<const B: usize, const PREFETCH: bool>(
        &self,
        _texts: &[Vec<u8>; B],
    ) -> [usize; B] {
        unimplemented!();
    }
}

fn time2<T>(f: impl FnOnce() -> T) -> (T, Duration) {
    let start = std::time::Instant::now();
    let x = f();
    let duration = start.elapsed();
    (x, duration)
}

fn time<T>(name: &str, f: impl FnOnce() -> T) -> T {
    let (x, duration) = time2(f);
    println!("{name:<10}: {duration:5.2?}");
    x
}

#[derive(clap::Parser)]
struct Args {
    reference: PathBuf,
    reads: PathBuf,
    #[clap(long)]
    len: Option<usize>,
    #[clap(short = 'j', long)]
    threads: Option<usize>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Mode {
    Sequential,
    Batch,
    Prefetch,
}

fn bench<F: FmIndex>(text: &[u8], bwt: &bwt::BWT, reads: &Vec<Vec<u8>>, threads: &Vec<usize>) {
    let (fm, build) = time2(|| F::new_with_prefix(text, bwt, 8));

    let mut reads = reads.clone();
    for read in reads.iter_mut() {
        F::prep_read(read);
    }

    for &threads in threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                for mode in [Mode::Prefetch, Mode::Batch, Mode::Sequential] {
                    match mode {
                        Mode::Prefetch if !F::HAS_PREFETCH => continue,
                        Mode::Batch if !F::HAS_BATCH => continue,
                        _ => {}
                    }
                    let name = std::any::type_name::<F>();
                    let name = regex::Regex::new(r"[a-zA-Z0-9_]+::")
                        .unwrap()
                        .replace_all(name, |_: &regex::Captures| "".to_string());
                    let bytes = fm.size();

                    eprint!(
                        "{:<65} size: {:>4}MB = {:8.3}bits/bp build: {build:9.3?} threads: {threads} mode: {:<10} ",
                        name,
                        bytes / 1024 / 1024,
                        (8 * bytes) as f64 / bwt.bwt.len() as f64,
                        format!("{:?}", mode)
                    );
                    let total = AtomicUsize::new(0);
                    let mapped = AtomicUsize::new(0);
                    let total_matches = AtomicUsize::new(0);
                    let total_steps = AtomicUsize::new(0);
                    let start = std::time::Instant::now();
                    // const B: usize = 1024 * 16;
                    const B: usize = 32;
                    reads.as_chunks::<B>().0.par_iter().for_each(|batch| {
                        let mut m = 0;
                        let mut mp = 0;
                        match mode {
                            Mode::Sequential => {
                                for q in batch {
                                    let matches = fm.count(q);
                                    m += matches;
                                    if matches > 0 {
                                        mp += 1;
                                    }
                                }
                            }
                            Mode::Batch => {
                                for matches in fm.count_batch::<B, false>(batch) {
                                    m += matches;
                                    if matches > 0 {
                                        mp += 1;
                                    }
                                }
                            }
                            Mode::Prefetch => {
                                for matches in fm.count_batch::<B, true>(batch) {
                                    m += matches;
                                    if matches > 0 {
                                        mp += 1;
                                    }
                                }
                            }
                        }
                        let t = total.fetch_add(batch.len(), std::sync::atomic::Ordering::Relaxed);
                        let mp = mapped.fetch_add(mp, std::sync::atomic::Ordering::Relaxed);
                        let m = total_matches.fetch_add(m, std::sync::atomic::Ordering::Relaxed);
                    });

                    let duration = start.elapsed();
                    let t = total.into_inner();
                    let mp = mapped.into_inner();
                    let m = total_matches.into_inner();

                    let thrpt = t as f64 / duration.as_secs_f64();

                    eprintln!(
                        "in {duration:6.3?} {:.5} Mread/s {:>8} mapped, {:>8} matches",
                        thrpt / 1e6,
                        mp,
                        m,
                    );
                    println!(
                        "\"{name}\",{bytes},{},{threads},{mode:?},{},{thrpt},{mp},{m}",
                        build.as_secs_f64(),
                        duration.as_secs_f64()
                    );
                }
            });
    }
}

#[allow(unused)]
fn test() {
    let mut text = b"GCATACGTACGAAAAAAGCTTG".to_vec();
    pack_text(&mut text);
    println!("build bwt");
    let bwt = build_bwt_packed(&text);
    println!("build fm");
    let fm = <quad_fm::QuadFm<QuadRank16>>::new(&text, &bwt);
    println!("query");
    let query = b"TACGAA";
    let packed = query.iter().map(|&x| (x >> 1) & 3).collect::<Vec<_>>();
    let count = fm.count(&packed);
    eprintln!("matches: {count}");
    // let packed_rc = packed.iter().rev().map(|&x| x ^ 2).collect::<Vec<_>>();
    // let (steps, count) = fm.query(&packed_rc);
    // eprintln!("steps: {steps}, matches: {count}");
    exit(0);
}

fn main() {
    // test();

    let args = Args::parse();
    let text = read_text(&args.reference);
    eprintln!("len: {}", text.len());

    let bwt_path = &args.reference.with_added_extension("bwt");
    if !bwt_path.exists() {
        eprintln!("Building BWT at {}", bwt_path.display());
        bwt::bwt(&text, bwt_path);
    }
    let bwt = std::fs::read(bwt_path).unwrap();
    let bwt: BWT = bincode::decode_from_slice::<DiskBWT, _>(&bwt, bincode::config::legacy())
        .unwrap()
        .0
        .pack();

    // eprintln!("Reading queries");
    let mut reader = needletail::parse_fastx_file(&args.reads).unwrap();
    let mut reads = vec![];
    while let Some(r) = reader.next() {
        let r = r.unwrap();
        let seq = r.seq();
        let seq = if let Some(len) = args.len {
            &seq[..len]
        } else {
            &seq
        };
        // eprintln!("seq: {}", std::str::from_utf8(&seq).unwrap());
        let packed = seq.iter().map(|&x| (x >> 1) & 3).collect::<Vec<_>>();
        let packed_rc = packed.iter().rev().map(|&x| x ^ 2).collect::<Vec<_>>();
        reads.push(packed);
        reads.push(packed_rc);
        if reads.len() == 1_000_000 {
            break;
        }
    }
    eprintln!("Num reads: {}", reads.len());

    let ts = &args.threads.map(|x| vec![x]).unwrap_or(vec![1, 6, 12]);

    println!("name,bytes,build,threads,mode,time,reads_per_sec,mapped,matches",);
    bench::<QuadFm<QuadRank16>>(&text, &bwt, &reads, ts);
    bench::<QuadFm<QuadRank24_8>>(&text, &bwt, &reads, ts);
    bench::<QuadFm<QuadRank64>>(&text, &bwt, &reads, ts);
    bench::<QuadFm<quadrank::qwt::RSQVector256>>(&text, &bwt, &reads, ts);
    bench::<QuadFm<quadrank::qwt::RSQVector512>>(&text, &bwt, &reads, ts);
    bench::<QuadFm<quadrank::genedex::Flat64>>(&text, &bwt, &reads, ts);
    bench::<QuadFm<quadrank::genedex::Flat512>>(&text, &bwt, &reads, ts);
    bench::<QuadFm<quadrank::genedex::Condensed64>>(&text, &bwt, &reads, ts);
    bench::<QuadFm<quadrank::genedex::Condensed512>>(&text, &bwt, &reads, ts);
    bench::<genedex::FmIndexFlat64<i64>>(&text, &bwt, &reads, ts);
    bench::<genedex::FmIndexFlat512<i64>>(&text, &bwt, &reads, ts);
    bench::<genedex::FmIndexCondensed64<i64>>(&text, &bwt, &reads, ts);
    bench::<genedex::FmIndexCondensed512<i64>>(&text, &bwt, &reads, ts);
    bench::<awry::fm_index::FmIndex>(&text, &bwt, &reads, ts);

    // OOM for human genome; >50GB
    // bench::<fm_index::FMIndex<u8>>(&text, &bwt, &reads, ts);
    // broken?
}
