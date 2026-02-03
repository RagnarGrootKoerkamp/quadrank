#![allow(incomplete_features)]
#![feature(
    generic_const_exprs,
    portable_simd,
    coroutines,
    coroutine_trait,
    associated_const_equality
)]

pub mod binary;
pub mod count;
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
#[cfg(feature = "ext")]
pub use ext::*;
