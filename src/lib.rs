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

pub mod blocks;
pub mod count;
pub mod count4;
pub mod qwt;
pub mod rank9;
pub mod ranker;
pub mod super_block;

pub type Ranks = [u32; 4];

fn add(a: Ranks, b: Ranks) -> Ranks {
    from_fn(|c| a[c] + b[c])
}
