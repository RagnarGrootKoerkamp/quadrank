//! # BiRank & QuadRank
//!
//! This crate provides data structures for `rank` queries over binary and size-4 alphabets.
//! The main entrypoints are [`BiRank`] and [`QuadRank`], and usually you won't need anything else.
//! They are aliases for instantiations of [`binary::Ranker`] and [`quad::Ranker`],
//! which implement the [`binary::RankerT`] and [`quad::RankerT`] traits respectively.
//!
//! See also the [GitHub README](https://github.com/ragnargrootkoerkamp/quadrank).
//!
//! ```
//! use quadrank::BiRank;
//! use quadrank::QuadRank;
//!
//! let packed = [0xf0f0f0f0f0f0f0f0, u64::MAX];
//! let rank = quadrank::BiRank::new(packed);
//! unsafe {
//!     assert_eq!(rank.rank_unchecked(0), 0);
//!     assert_eq!(rank.rank_unchecked(4), 0);
//!     assert_eq!(rank.rank_unchecked(8), 4);
//!     assert_eq!(rank.rank_unchecked(64), 32);
//!     assert_eq!(rank.rank_unchecked(128), 96);
//! }
//!
//! let dna = b"ACGCGCGACTTACGCAT";
//! let n = dna.len(); // 17
//! let rank = quadrank::QuadRank::new_ascii_dna(dna);
//! unsafe {
//!     assert_eq!(rank.rank1_unchecked(0, 0), 0);
//!     assert_eq!(rank.rank4_unchecked(0), [0; 4]);
//!     assert_eq!(rank.rank1_unchecked(n, 0), 4);
//!     assert_eq!(rank.rank4_unchecked(n), [4, 6, 3, 4]);
//! }
//! ```

/// Rank over binary alphabet.
pub mod binary;
/// Rank over size-4 alphabet.
pub mod quad;

/// Implementations of [`binary::RankerT`] and [`quad::RankerT`] for external crates.
///
/// Each module re-exports the relevant types.
#[cfg(feature = "ext")]
pub mod ext {
    pub mod bio;
    pub mod bitm;
    pub mod genedex;
    pub mod qwt;
    pub mod rsdict;
    pub mod succinct;
    pub mod sucds;
    pub mod sux;
    pub mod vers;
}

// Type aliases
/// Default binary rank structure. Alias for [`BiRank16`].
pub type BiRank = BiRank16;
/// Binary rank structure with 3.28% space overhead.
/// Smallest, and usually sufficiently fast.
pub type BiRank16 = binary::Ranker<binary::blocks::BinaryBlock16>;
/// Binary rank structure with 6.72% space overhead.
pub type BiRank16x2 = binary::Ranker<binary::blocks::BinaryBlock16x2>;
/// Binary rank structure with 14.3% space overhead.
pub type BiRank32x2 = binary::Ranker<binary::blocks::BinaryBlock32x2>;
/// Binary rank structure with 33.3 space overhead.
pub type BiRank64x2 = binary::Ranker<binary::blocks::BinaryBlock64x2>;

/// Default quad rank structure. Alias for [`QuadRank16`].
pub type QuadRank = QuadRank16;
/// Quad rank structure with 14.40% space overhead.
/// Smallest, and usually sufficiently fast.
pub type QuadRank16 = quad::Ranker<quad::blocks::QuadBlock16>;
/// Quad rank structure with 33% space overhead.
pub type QuadRank24_8 = quad::Ranker<quad::blocks::QuadBlock24_8>;
/// Quad rank structure with 100% space overhead.
pub type QuadRank64 = quad::Ranker<quad::blocks::QuadBlock64>;
