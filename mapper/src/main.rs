#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

mod bwt;
mod fm;

use clap::Parser;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::{
    path::{Path, PathBuf},
    process::exit,
    sync::atomic::AtomicUsize,
};

fn time<T>(name: &str, f: impl FnOnce() -> T) -> T {
    let start = std::time::Instant::now();
    let x = f();
    let duration = start.elapsed();
    println!("{name:<10}: {duration:5.2?}");
    x
}

#[derive(clap::Parser)]
struct Args {
    reference: PathBuf,
    reads: PathBuf,
}

fn build_bwt(mut text: Vec<u8>) -> bwt::BWT {
    // Map to 0123.
    for x in &mut text {
        *x = (*x >> 1) & 3;
    }

    // Caps-sa construction

    if text.len() > 10000 {
        // time("simple-saca", || bwt::simple_saca(&text))
        // time("small-bwt", || bwt::small_bwt(&text))
        time("caps-sa", || bwt::caps_sa(&text, false))
    } else {
        eprintln!("text: {text:?}");
        time("manual", || bwt::manual(&text))
    }
}

fn bwt(input: &Path, output: &Path) {
    let mut text = vec![];
    let mut reader = needletail::parse_fastx_file(input).unwrap();
    while let Some(record) = reader.next() {
        let record = record.unwrap();
        text.extend_from_slice(&record.seq());
    }
    let bwt = build_bwt(text);

    // write output to path.bwt:
    std::fs::write(output, serde_json::to_string(&bwt).unwrap()).unwrap();
}

fn map(bwt_path: &Path, reads_path: &Path) {
    eprintln!("Reading BWT from {}", bwt_path.display());
    let bwt = std::fs::read(bwt_path).unwrap();
    let bwt = serde_json::from_slice(&bwt).unwrap();
    eprintln!("Building FM index & rank structure");
    let fm = time("FM build", || fm::FM::new(&bwt));

    let mut reader = needletail::parse_fastx_file(reads_path).unwrap();
    let mut reads = vec![];
    while let Some(r) = reader.next() {
        let r = r.unwrap();
        let seq = r.seq();
        // eprintln!("seq: {}", std::str::from_utf8(&seq).unwrap());
        let packed = seq.iter().map(|&x| (x >> 1) & 3).collect::<Vec<_>>();
        let packed_rc = packed.iter().rev().map(|&x| x ^ 2).collect::<Vec<_>>();
        reads.push(packed);
        reads.push(packed_rc);
    }

    let total = AtomicUsize::new(0);
    let mapped = AtomicUsize::new(0);
    let total_matches = AtomicUsize::new(0);
    let total_steps = AtomicUsize::new(0);
    let start = std::time::Instant::now();
    const B: usize = 32;
    reads.as_chunks::<B>().0.par_iter().for_each(|batch| {
        let mut s = 0;
        let mut m = 0;
        let mut mp = 0;
        for (steps, matches) in fm.query_batch(batch) {
            s += steps;
            m += matches;
            if matches > 0 {
                mp += 1;
            }
        }
        let ts = total_steps.fetch_add(s, std::sync::atomic::Ordering::Relaxed);
        let t = total.fetch_add(batch.len(), std::sync::atomic::Ordering::Relaxed);
        let mp = mapped.fetch_add(mp, std::sync::atomic::Ordering::Relaxed);
        let m = total_matches.fetch_add(
            m,
            std::sync::atomic::Ordering::Relaxed,
        );

        if t % (1024 * 1024) == 0 {
            let duration = start.elapsed();
            eprint!(
                "Processed {:>8} reads ({:>8.3} steps/read, {:>8} mapped, {:>8} matches) in {:5.2?} ({:>6.2} kreads/s, {:>6.2} Mbp/s)\n",
                t,
                ts as f64 / t as f64,
                mp,
                m,
                duration,
                t as f64 / duration.as_secs_f64() / 1e3,
                ts as f64 / duration.as_secs_f64() / 1e6
            );
        }
    });

    let total = total.into_inner();
    let mapped = mapped.into_inner();
    let total_matches = total_matches.into_inner();
    let total_steps = total_steps.into_inner();

    eprintln!();
    println!("{:<15} {}", "#reads:", total);
    println!(
        "{:<15} {:.2}",
        "#steps/read:",
        total_steps as f64 / total as f64
    );
    println!("{:<15} {}", "#mapped:", mapped);
    println!("{:<15} {}", "#matches:", total_matches);
}

#[allow(unused)]
fn test() {
    let text = b"GCATACGTACGAAAAAAGCTTG";
    println!("build bwt");
    let bwt = build_bwt(text.to_vec());
    println!("build fm");
    let fm = fm::FM::new(&bwt);
    println!("query");
    let query = b"TACGAA";
    let packed = query.iter().map(|&x| (x >> 1) & 3).collect::<Vec<_>>();
    let (steps, count) = fm.query(&packed);
    eprintln!("steps: {steps}, matches: {count}");
    // let packed_rc = packed.iter().rev().map(|&x| x ^ 2).collect::<Vec<_>>();
    // let (steps, count) = fm.query(&packed_rc);
    // eprintln!("steps: {steps}, matches: {count}");
    exit(0);
}

fn main() {
    // test();

    let args = Args::parse();
    let bwt_path = &args.reference.with_added_extension("bwt");
    if !bwt_path.exists() {
        eprintln!("Building BWT at {}", bwt_path.display());
        bwt(&args.reference, bwt_path);
    }

    map(bwt_path, &args.reads);
}

#[test]
fn broken() {
    // let text = b"AGCCTTAGCTGCGACAGAATGGATCAGAAAGCTTGAAAACTTAGAGCAAAAAATTGACTATTTTGACGAGTGTCTTCTTCCAGGCATTTTCACCATCGACGCGGATCCTCCAGACGAGTTGTTTCTTGATGAACTG";
    // let query = b"TGCGACAGAATGGATCAGAAAGCTTGAAAACTTAGAGCAAAAAATTGACTATTTTGACGAGTGTCTTCTTCC";
    let text = b"AGCCTTAGCTGCGACAGAATGGATCAGAAAGCTTGAAAACTTAGAGCAAAAAATTGACTATTTTGACGAGTGTCTTCTTCCAGGCATTTTCACCATCGACGCGGATCCTCCAGACGAGTTGTTTCTTGATGAACTG";
    let query = b"GGA";
    let bwt = build_bwt(text.to_vec());
    let packed = query.iter().map(|&x| (x >> 1) & 3).collect::<Vec<_>>();
    let fm = fm::FM::new(&bwt);
    let (steps, count) = fm.query(&packed);
    eprintln!("steps: {steps}, matches: {count}");
    assert!(count > 0);
}
