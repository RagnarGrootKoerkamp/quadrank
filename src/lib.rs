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
#[cfg(test)]
pub mod test;

pub type Ranks = [u32; 4];

pub type QuartRank =
    ranker::Ranker<blocks::QuartBlock, super_block::NoSB, count4::SimdCount10, false>;
pub type HexRank =
    ranker::Ranker<blocks::HexaBlockMid4, super_block::TrivialSB, count4::SimdCount10, false>;
pub type QwtRank = ::qwt::RSQVector256;

fn add(a: Ranks, b: Ranks) -> Ranks {
    from_fn(|c| a[c] + b[c])
}

/// Prefetch the given cacheline into L1 cache.
pub fn prefetch_index<T>(s: &[T], index: usize) {
    let ptr = s.as_ptr().wrapping_add(index) as *const u64;
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
    }
    #[cfg(target_arch = "x86")]
    unsafe {
        std::arch::x86::_mm_prefetch(ptr as *const i8, std::arch::x86::_MM_HINT_T0);
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        std::arch::aarch64::_prefetch(ptr as *const i8, std::arch::aarch64::_PREFETCH_LOCALITY3);
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86", target_arch = "aarch64")))]
    {
        // Do nothing.
    }
}
