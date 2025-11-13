#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

mod bwt;
mod caps_sa_test;
mod fm;

use clap::Parser;
use fm_index::Text;
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
    std::fs::write(
        output,
        bincode::encode_to_vec(&bwt, bincode::config::legacy()).unwrap(),
    )
    .unwrap();
}

fn map(bwt_path: &Path, reads_path: &Path) {
    eprintln!("Reading BWT from {}", bwt_path.display());
    let bwt = std::fs::read(bwt_path).unwrap();
    let bwt = bincode::decode_from_slice(&bwt, bincode::config::legacy())
        .unwrap()
        .0;
    eprintln!("Building FM index & rank structure");
    let fm = time("FM build", || fm::FM::new(&bwt));

    eprintln!("Reading queries");
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

fn map_fm_crate(input_path: &Path, reads_path: &Path) {
    eprintln!("Reading text from {}", input_path.display());
    let mut text = vec![];
    let mut reader = needletail::parse_fastx_file(input_path).unwrap();
    while let Some(record) = reader.next() {
        let record = record.unwrap();
        text.extend_from_slice(&record.seq());
    }
    for x in &mut text {
        *x = ((*x >> 1) & 3) + 1;
    }
    text.push(0);
    eprintln!("Building FM index & rank structure");

    let fm = time("FM build", || {
        fm_index::FMIndex::<u8>::new(&fm_index::Text::with_max_character(text, 4)).unwrap()
    });

    eprintln!("Reading queries");
    let mut reader = needletail::parse_fastx_file(reads_path).unwrap();
    let mut reads = vec![];
    while let Some(r) = reader.next() {
        let r = r.unwrap();
        let seq = r.seq();
        // eprintln!("seq: {}", std::str::from_utf8(&seq).unwrap());
        let packed = seq.iter().map(|&x| ((x >> 1) & 3) + 1).collect::<Vec<_>>();
        let packed_rc = packed
            .iter()
            .rev()
            .map(|&x| ((x - 1) ^ 2) + 1)
            .collect::<Vec<_>>();
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
        let s = 0;
        let mut m = 0;
        let mut mp = 0;
        for q in batch {
            let matches = fm.search(&batch[0]).count();
            // s += steps;
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

fn map_genedex(input_path: &Path, reads_path: &Path) {
    eprintln!("Reading text from {}", input_path.display());
    let mut text = vec![];
    let mut reader = needletail::parse_fastx_file(input_path).unwrap();
    while let Some(record) = reader.next() {
        let record = record.unwrap();
        text.extend_from_slice(&record.seq());
    }
    for x in &mut text {
        *x = (*x >> 1) & 3;
    }
    text.push(0);
    eprintln!("Building FM index & rank structure on len {}", text.len());

    let fm = time("FM build", || {
        genedex::FmIndexConfig::<i32>::new()
            .suffix_array_sampling_rate(16)
            .construct_index(&[text], genedex::alphabet::u8_until(3))
    });

    eprintln!("Reading queries");
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
        let s = 0;
        let mut m = 0;
        let mut mp = 0;
        for matches in  fm.count_many(batch) {
            // s += steps;
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

    // map(bwt_path, &args.reads);
    // map_fm_crate(&args.reference, &args.reads);
    map_genedex(&args.reference, &args.reads);
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

#[test]
fn broken2() {
    // let text = b"AGCCTTAGCTGCGACAGAATGGATCAGAAAGCTTGAAAACTTAGAGCAAAAAATTGACTATTTTGACGAGTGTCTTCTTCCAGGCATTTTCACCATCGACGCGGATCCTCCAGACGAGTTGTTTCTTGATGAACTG";
    // let query = b"TGCGACAGAATGGATCAGAAAGCTTGAAAACTTAGAGCAAAAAATTGACTATTTTGACGAGTGTCTTCTTCC";
    let text = b"AGCCTTAGCTGCGACAGAATGGATCAGAAAGCTTGAAAACTTAGAGCAAAAAATTGACTATTTTGACGAGTGTCTTCTTCCAGGCATTTTCACCATCGACGCGGATCCTCCAGACGAGTTGTTTCTTGATGAACTG";
    let query = b"GGATCA";
    let mut text = text.iter().map(|&x| ((x >> 1) & 3) + 1).collect::<Vec<_>>();
    text.push(0);
    let query = query
        .iter()
        .map(|&x| ((x >> 1) & 3) + 1)
        .collect::<Vec<_>>();

    let fm = fm_index::FMIndex::new(&Text::with_max_character(text, 4)).unwrap();
    let count = fm.search(&query).count();
    eprintln!("matches: {count}");
    assert!(count > 0);
    panic!()
}

#[test]
fn fuzz_fm() {
    for _ in 0..1000 {
        let len = rand::random_range(1000..3000);
        eprintln!("Building for len {len}");
        let mut text = (0..len)
            .map(|_| rand::random_range(0..4))
            .collect::<Vec<_>>();

        let bwt = build_bwt_packed(&mut text);
        let mfm = fm::FM::new(&bwt);

        let gfm = genedex::FmIndexConfig::<i32>::new()
            .suffix_array_sampling_rate(16)
            .construct_index(&[&text], genedex::alphabet::u8_until(3));

        eprintln!("Querying");
        for _ in 0..10000 {
            let start = rand::random_range(0..len);
            let end = rand::random_range(start..=len);
            let q = &text[start..end];

            let m_cnt = mfm.query(q).1;
            let g_cnt = gfm.count(q);
            eprintln!("+ m_cnt: {}, g_cnt: {}", m_cnt, g_cnt);
            assert_eq!(m_cnt, g_cnt, "text len {}, query {:?}", len, q);
        }

        for _ in 0..10000 {
            let start = rand::random_range(0..len - 1);
            let end = rand::random_range(start + 1..=len);
            let mut q = text[start..end].to_vec();
            let ql = q.len();
            q[rand::random_range(0..ql)] = rand::random_range(0..4);

            let m_cnt = mfm.query(&q).1;
            let g_cnt = gfm.count(&q);
            eprintln!("- m_cnt: {}, g_cnt: {}", m_cnt, g_cnt);
            assert_eq!(m_cnt, g_cnt, "text len {}, query {:?}", len, q);
        }
    }
}
