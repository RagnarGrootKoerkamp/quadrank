#![allow(incomplete_features)]
#![feature(
    generic_const_exprs,
    portable_simd,
    coroutines,
    coroutine_trait,
    exact_div,
    associated_const_equality
)]

use std::array::from_fn;

pub mod binary;
pub mod blocks;
pub mod count;
pub mod count4;
pub mod genedex;
pub mod qwt;
pub mod ranker;
pub mod super_block;
pub mod sux;

pub type Ranks = [u32; 4];

pub type QuartRank =
    ranker::Ranker<blocks::QuartBlock, super_block::NoSB, count4::SimdCount10, false>;
pub type HexRank =
    ranker::Ranker<blocks::HexaBlockMid4, super_block::TrivialSB, count4::SimdCount10, false>;
pub type QwtRank = ::qwt::RSQVector256;

fn add(a: Ranks, b: Ranks) -> Ranks {
    from_fn(|c| a[c] + b[c])
}

// TODO: investigate different bitpacking layouts, like QWT does

#[test]
fn test() {
    use ranker::RankerT;

    let text = b"AGCCTTAGCTGCGACAGAATGGATCAGAAAGCTTGAAAACTTAGAGCAAAAAATTGACTATTTTGACGAGTGTCTTCTTCCAGGCATTTTCACCATCGACGCGGATCCTCCAGACGAGTTGTTTCTTGATGAACTG";
    let ranker = HexRank::new(text);

    assert_eq!(ranker.count(0), [0, 0, 0, 0]);

    eprintln!();
    let base = b"ACTG".map(|c| text.iter().filter(|x| **x == c).count() as u32);
    eprintln!("want: {:?}", base);
    let cnt = ranker.count(text.len());
    eprintln!("get {:?}", cnt);
    assert_eq!(cnt, base);
    // panic!();
}
