use clap::Parser;
use rand::random_bool;
use rand_distr::{Distribution, Poisson};
use std::path::PathBuf;

#[derive(clap::Parser)]
struct Args {
    input: PathBuf,
    #[clap(short, default_value_t = 500_000)]
    n: usize,
    #[clap(short, default_value_t = 0.01)]
    error: f32,
    #[clap(long, default_value_t = 150)]
    len: usize,
}

fn main() {
    let args = Args::parse();

    let mut reference = vec![];
    let mut reader = needletail::parse_fastx_file(&args.input).unwrap();
    while let Some(record) = reader.next() {
        let record = record.unwrap();
        reference.extend_from_slice(&record.seq());
    }

    let rng = &mut rand::rng();
    let poisson = Poisson::new(args.len as f32 * args.error).unwrap();

    for i in 0..args.n {
        let start = rand::random_range(0..reference.len() - args.len);
        let end = start + args.len;

        let mut read = reference[start..end].to_vec();
        // number of errors is poisson
        let num_erors = poisson.sample(rng) as usize;
        for _ in 0..num_erors {
            let pos = rand::random_range(0..read.len());
            let base = read[pos];
            let mut new_base = base;
            while new_base == base {
                new_base = *b"ACGT".get(rand::random_range(0..4)).unwrap();
            }
            read[pos] = new_base;
        }

        read.make_ascii_uppercase();

        let rc = random_bool(0.5);
        if rc {
            read.reverse();
            for b in &mut read {
                *b = b"TGAC"[(*b as usize >> 1) & 3];
            }
        }

        println!(">read_{i}_at_{start}_{}", if rc { "rc" } else { "fwd" });
        println!("{}", std::str::from_utf8(&read).unwrap());
    }
}
