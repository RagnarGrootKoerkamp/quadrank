# QuadRank

This repo implements `BiRank`, `QuadRank`, and `QuadFm`, two fast rank data
structures and a simple count-only FM-index that all use batching and prefetching of queries.

BiRank and QuadRank need only a single cache-miss per query, making them
up to 2x faster than other methods in high-throughput settings.

QuadFm is up to 4x faster than genedex (https://github.com/feldroop/genedex),
which seems to be the fastest Rust-based FM-index currently.

**NOTE:** The code here is not really ready yet for consumption as a library:
- Only AVX2 is supported currently.
- The API still needs cleaning up.
- Docs still need to be written for docs.rs.
## Results

### BiRank

![Comparison plot, showing that BiRank variants are smaller and faster than others.](evals/birank.png)

### QuadRank

![Comparison plot, showing that BiRank variants are smaller and faster than others.](evals/quadrank.png)

### FM-index

Here I'm mapping simulated 150bp short reads with 1% error rate (see `examples/simulate_reads.rs`) against a 3.1 Gbp human genome.
I first build each index on the forward data (where I don't care about time/space usage),
and then count the number of matches of each fwd/rc read.
For `genedex` and `quad`, I query batches of 32 reads at a time.
I'm using 12 threads, on my 6-core i7-10750H, fixed at 3.0 GHz.

- `AWRY`: https://github.com/UM-Applied-Algorithms-Lab/AWRY
- `Genedex<>`: Genedex, with different choice of its rank structure.
- `QuadFm<>`: The FM-index in the `fm-index` directory here.
- `QuadRank*`: The rank structure we introduce.
- `qwt::RSQ{256,512}`: The rank structures of https://github.com/rossanoventurini/qwt.

![Comparison plot, showing that QuadFm is smaller and faster than others.](evals/fm.png)

### Replicating benchmarks

Run the synthetic BiRank and QuadRank benchmarks using:

``` sh
cargo run -r --example bench -F ext -- -b -q -j 12 > evals/laptop.csv
```

Run the QuadFm benchmark using:

``` sh
cd quad-fm && cargo run -r --example bench -F ext -- human-genome.fa reads.fa > ../evals/fm.csv
```

where input reads can be simulated using:

``` sh
cd quad-fm && cargo run -r --example simulate-reads -- human-genome.fa
```

Each bench should take somewhere up to half an hour. Results can be converted
into plots in `evals/plots/` by running `evals/plot.py` and `evals/plot-fm.py`.
