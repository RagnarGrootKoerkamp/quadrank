pub mod blocks;
pub mod count4;
pub mod ranker;
pub mod super_blocks;
#[cfg(test)]
pub mod test;

use packed_seq::{PackedSeqVec, SeqVec};
use std::ops::Coroutine;

pub use ranker::Ranker;
pub use super_blocks::TrivialSB;

pub type Ranks = [u32; 4];
pub type LongRanks = [u64; 4];

pub type QuadRank32_8_8_8 = Ranker<blocks::QuadBlock32_8_8_8FP, super_blocks::ShiftSB>;
pub type QuadRank7_18_7 = Ranker<blocks::QuadBlock7_18_7P, super_blocks::ShiftSB>;

pub type QuadRank64 = Ranker<blocks::QuadBlock64, super_blocks::ShiftSB>;
pub type QuadRank24_8 = Ranker<blocks::QuadBlock24_8, super_blocks::ShiftSB>;
pub type QuadRank16 = Ranker<blocks::QuadBlock16, super_blocks::ShiftSB>;

/// By default, the library works for arrays with counts up to `2^45`, corresponding to `8 TiB` of data.
/// This controls whether superblocks are used and/or prefetched.
pub const TARGET_BITS: usize = 45;

pub trait BasicBlock: Sync + Send {
    /// Character width. 1 for binary, 2 for DNA.
    const X: usize;

    /// Number of bytes per block.
    const B: usize;
    /// Number of characters per block.
    const N: usize;
    /// Bytes of the underlying count function.
    const C: usize;
    /// Bit-width of the internal global ranks.
    const W: usize;
    /// When `false`, basic data is bit-packed.
    /// When `true`, transposed layout is used with a u64 of high bits followed
    /// by a u64 of low bits.
    const TRANSPOSED: bool;

    fn new(ranks: Ranks, data: &[u8]) -> Self;
    /// Count the number of times each character occurs before position `pos`.
    fn count4(&self, pos: usize) -> Ranks;
    /// Count the number of times character `c` occurs before position `pos`.
    fn count1(&self, _pos: usize, _c: u8) -> u32 {
        unimplemented!()
    }
    /// count1 for two (usually closeby) positions in parallel.
    /// Main benefit is to interleave code, rather than actually reusing computed values.
    fn count1x2(&self, _other: &Self, _pos0: usize, _pos1: usize, _c: u8) -> (u32, u32) {
        unimplemented!()
    }
}

pub trait SuperBlock<BB: BasicBlock>: Sync + Send {
    /// Bit-width of the basic block ranks.
    const W: usize;

    /// This many low bits are shifted out.
    const SHIFT: usize;

    fn new(ranks: LongRanks, data: &[u8]) -> Self;
    fn get(&self, block_idx: usize) -> LongRanks;
    fn get1(&self, block_idx: usize, c: u8) -> usize {
        self.get(block_idx)[c as usize] as usize
    }

    /// Store a new superblock every this-many blocks.
    ///
    /// For `N` bits/block and `x` blocks, each superblock spans `N*x`
    /// characters with `N*x + N < 2^32`, and `x` fast to compute.
    /// => `x < 2^32 / N - 1`
    const BLOCKS_PER_SUPERBLOCK: usize = if BB::W == 0 {
        1
    } else if BB::W >= TARGET_BITS {
        usize::MAX
    } else {
        (((1u128 << BB::W) / BB::N as u128) as usize - 1).next_power_of_two() / 2
    };
    const BYTES_PER_SUPERBLOCK: usize = Self::BLOCKS_PER_SUPERBLOCK.saturating_mul(BB::B);
}

pub trait RankerT: Sync + Send + Sized {
    /// Construct from ASCII DNA input.
    fn new(seq: &[u8]) -> Self {
        // let mut packed_seq = seq.to_vec();
        let mut packed_seq = PackedSeqVec::from_ascii(seq).into_raw();
        // eprintln!("packed_seq: {:?}", packed_seq);
        // Add one block of padding.
        packed_seq.resize(packed_seq.len() + 1024, 0);
        let (head, data, tail) = unsafe { packed_seq.align_to::<usize>() };
        assert!(head.is_empty());
        assert!(tail.is_empty());
        Self::new_packed(data)
    }
    /// Construct from bitpacked data.
    fn new_packed(seq: &[usize]) -> Self;
    /// Prefetch the cachelines for the given position and character.
    fn prefetch1(&self, pos: usize, c: u8);
    /// Prefetch the cachelines for all characters for the given position.
    fn prefetch4(&self, pos: usize) {
        self.prefetch1(pos, 0);
    }
    fn size(&self) -> usize;
    /// Count the number of times each character occurs before position `pos`.
    fn rank4(&self, pos: usize) -> LongRanks;
    /// Count the number of times character `c` occurs before position `pos`.
    fn rank1(&self, pos: usize, c: u8) -> usize;
    #[inline(always)]
    fn count1x2(&self, pos0: usize, pos1: usize, c: u8) -> (usize, usize) {
        (self.rank1(pos0, c), self.rank1(pos1, c))
    }

    #[inline(always)]
    fn count_coro(&self, pos: usize) -> impl Coroutine<Yield = (), Return = LongRanks> + Unpin {
        self.prefetch4(pos);
        #[inline(always)]
        #[coroutine]
        move || self.rank4(pos)
    }
    #[inline(always)]
    fn count_coro2(&self, pos: usize) -> impl Coroutine<Yield = (), Return = LongRanks> + Unpin {
        #[inline(always)]
        #[coroutine]
        move || {
            self.prefetch4(pos);
            yield;
            self.rank4(pos)
        }
    }
}
