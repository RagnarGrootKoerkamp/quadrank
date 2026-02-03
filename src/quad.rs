pub mod blocks;
mod count1;
mod count4;
mod ranker;
pub mod super_blocks;
#[cfg(test)]
mod test;

use packed_seq::{PackedSeqVec, SeqVec};

pub use ranker::Ranker;

type Ranks = [u32; 4];
pub type LongRanks = [u64; 4];

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
    } else if BB::W == 64 {
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
        let (head, data, tail) = unsafe { packed_seq.align_to::<u64>() };
        assert!(head.is_empty());
        assert!(tail.is_empty());
        Self::new_packed(data)
    }
    /// Construct from bitpacked data.
    fn new_packed(seq: &[u64]) -> Self;
    /// Prefetch the cachelines for the given position and character.
    fn prefetch1(&self, pos: usize, c: u8);
    /// Prefetch the cachelines for all characters for the given position.
    fn prefetch4(&self, pos: usize) {
        self.prefetch1(pos, 0);
    }
    fn size(&self) -> usize;
    /// Count the number of times each character occurs before position `pos`.
    /// Assumes pos<len.
    unsafe fn rank4_unchecked(&self, pos: usize) -> LongRanks;
    /// Count the number of times character `c` occurs before position `pos`.
    /// Assumes pos<len.
    unsafe fn rank1_unchecked(&self, pos: usize, c: u8) -> usize;
    /// Assumes pos<len.
    #[inline(always)]
    unsafe fn count1x2_unchecked(&self, pos0: usize, pos1: usize, c: u8) -> (usize, usize) {
        unsafe { (self.rank1_unchecked(pos0, c), self.rank1_unchecked(pos1, c)) }
    }
}
