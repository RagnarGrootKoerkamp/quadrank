#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

mod bwt;
mod fm;

use clap::Parser;
use std::{
    path::{Path, PathBuf},
    process::exit,
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
    let mut total = 0;
    let mut mapped = 0;
    let mut total_matches = 0;
    let mut total_steps = 0;
    let start = std::time::Instant::now();
    while let Some(record) = reader.next() {
        let record = record.unwrap();
        let seq = record.seq();
        // eprintln!("seq: {}", std::str::from_utf8(&seq).unwrap());
        let packed = seq.iter().map(|&x| (x >> 1) & 3).collect::<Vec<_>>();
        let packed_rc = packed.iter().rev().map(|&x| x ^ 2).collect::<Vec<_>>();

        for q in [packed, packed_rc] {
            let (steps, matches) = fm.query(&q);
            // eprintln!("  steps: {}, matches: {}", steps, matches);
            total += 1;
            total_steps += steps;
            total_matches += matches;
            if matches > 0 {
                mapped += 1;
            }
        }

        if total % 1024 == 0 {
            let duration = start.elapsed();
            eprint!(
                "Processed {:>8} reads ({:>8.3} steps/read, {:>8} mapped, {:>8} matches) in {:5.2?} ({:>6.2} reads/s, {:>6.2} steps/s)\r",
                total,
                total_steps as f64 / total as f64,
                mapped,
                total_matches,
                duration,
                total as f64 / duration.as_secs_f64(),
                total_steps as f64 / duration.as_secs_f64()
            );
        }
    }
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
