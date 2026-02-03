[![crates.io](https://img.shields.io/crates/v/quadrank.svg)](https://crates.io/crates/quadrank)
[![docs.rs](https://img.shields.io/docsrs/quadrank.svg?label=docs.rs)](https://docs.rs/quadrank)

# QuadRank: a High Throughput Rank

This repo implements `BiRank`, `QuadRank`, and `QuadFm`, two fast rank data
structures for binary and DNA input, and a simple count-only FM-index that all use batching and prefetching of queries.

BiRank and QuadRank need only a single cache-miss per query, making them
up to 2x faster than other methods in high-throughput settings.
QuadFm is up to 4x faster than genedex (https://github.com/feldroop/genedex),
which seems to be the fastest Rust-based FM-index currently.

**NOTE:** Currently only AVX2 is supported!

## Usage

BiRank and QuadRank are constructed from (little-endian) _packed_ data, given as
a `&[u64]` slice.
For binary input, this means the first bit is the lowest-order bit of the first
value. For QuadRank, each `u64` contains 32 2-bit values.

The main data structures are `BiRank` and `QuadRank`, which implement
`binary::RankerT` and `quad::RankerT`:

``` rust
trait binary::RankerT {
    fn new_packed(seq: &[u64]) -> Self;

    /// Number of 1 bits before pos `pos`. Requires `0 <= pos <= len`.
    unsafe fn rank_unchecked(&self, pos: usize) -> u64;

    /// Prefetch the cache lines needed to answer `rank(pos)`.
    fn prefetch(&self, pos: usize);
}

trait quad::RankerT {
    /// ASCII values `ACTG` get converted to `0123`.
    fn new_ascii_dna(seq: &[u8]) -> Self; 
    fn new_packed(seq: &[u64]) -> Self;

    /// Number of times `0 <= c < 4` occurs before pos `pos`. Requires `0 <= pos <= len`.
    unsafe fn rank1_unchecked(&self, pos: usize, c: u8) -> u64;
    /// Number of times each character occurs before pos `pos`. Requires `0 <= pos <= len`.
    unsafe fn rank4_unchecked(&self, pos: usize, c: u8) -> [u64; 4];

    /// Prefetch the cache lines needed to answer `rank1(pos, c)`.
    fn prefetch1(&self, pos: usize, c: u8);
    /// Prefetch the cache lines needed to answer `rank4(pos)`.
    fn prefetch4(&self, pos: usize);
}
```

Example:

``` rust
let packed = [u64::MAX, u64::MAX];
let rank = quadrank::BiRank::new(packed);
unsafe {
    assert_eq!(rank.rank_unchecked(0), 0);
    assert_eq!(rank.rank_unchecked(1), 1);
    assert_eq!(rank.rank_unchecked(128), 128);
}

let dna = b"ACGCGCGACTTACGCAT";
let n = dna.len(); // 17
let rank = quadrank::QuadRank::new_ascii_dna(dna);
unsafe {
    assert_eq!(rank.rank1_unchecked(0, 0), 0);
    assert_eq!(rank.rank4_unchecked(0), [0; 4]);
    assert_eq!(rank.rank1_unchecked(n, 0), 4);
    assert_eq!(rank.rank4_unchecked(n), [4, 6, 3, 4]);
}
```

## Results

### BiRank

Space-time trade-off of binary rank data structures.
On the x-axis is the space overhead compared to the
input, and on the y-ax the ns per query. Top/middle/bottom are 1/6/12 threads,
and left/middle/right are latency, throughput in a for loop, and throughput in a
loop with additional prefetching.

![Comparison plot, showing that BiRank variants are smaller and faster than others.](evals/birank.png)

### QuadRank
 
Space-time trade-off of quad rank data structures.

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
cargo run -r --example bench -F ext -- -b -q -j 12 > evals/rank-laptop.csv
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
