#![allow(incomplete_features)]
#![feature(generic_const_exprs, array_windows, iter_next_chunk)]

pub mod bwt;
mod quad_fm;

#[cfg(all(test, feature = "ext"))]
mod test;

#[cfg(feature = "ext")]
mod ext {
    mod awry;
    mod fm_crate;
    mod genedex;
}

pub use quad_fm::QuadFm;
// use quadrank::quad::*;
// use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
// use std::{path::PathBuf, process::exit, sync::atomic::AtomicUsize, time::Duration};

pub trait FmIndex: Sized + Sync {
    /// text is values 0..4, one per byte.
    fn new_with_prefix(text: &[u8], bwt: &bwt::BWT, prefix: usize) -> Self;
    fn new(text: &[u8], bwt: &bwt::BWT) -> Self {
        Self::new_with_prefix(text, bwt, 0)
    }
    fn prep_read(_read: &mut [u8]) {}
    /// Size in bytes.
    fn size(&self) -> usize;
    /// Count the number of matches.
    fn count(&self, text: &[u8]) -> usize;
    const HAS_BATCH: bool;
    const HAS_PREFETCH: bool;
    /// Batch size B, and whether to PREFETCH or not.
    fn count_batch<const B: usize, const PREFETCH: bool>(
        &self,
        _texts: &[Vec<u8>; B],
    ) -> [usize; B] {
        unimplemented!();
    }
}
