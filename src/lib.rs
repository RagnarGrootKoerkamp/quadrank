pub mod binary;
pub mod quad;

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
/// Main binary rank structure. Alias for [`BiRank16`].
pub type BiRank = BiRank16;
pub type BiRank16 = binary::Ranker<binary::blocks::BinaryBlock16>;
pub type BiRank16x2 = binary::Ranker<binary::blocks::BinaryBlock16x2>;
pub type BiRank32x2 = binary::Ranker<binary::blocks::BinaryBlock32x2>;
pub type BiRank64x2 = binary::Ranker<binary::blocks::BinaryBlock64x2>;

/// Main DNA rank structure. Alias for [`QuadRank16`].
pub type QuadRank = QuadRank16;
pub type QuadRank16 = quad::Ranker<quad::blocks::QuadBlock16>;
pub type QuadRank24_8 = quad::Ranker<quad::blocks::QuadBlock24_8>;
pub type QuadRank64 = quad::Ranker<quad::blocks::QuadBlock64>;
