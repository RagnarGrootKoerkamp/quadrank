pub mod blocks;
mod count1;
mod count4;
mod ranker;
pub mod super_blocks;
#[cfg(test)]
mod test;

use packed_seq::{PackedSeqVec, SeqVec};

pub use ranker::QuadRank;

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
    fn get1(&self, block_idx: usize, c: u8) -> u64 {
        self.get(block_idx)[c as usize]
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

/// A data structure that can answer rank queries over an alphabet of size 4.
pub trait QuadRanker: Sync + Send + Sized {
    /// Construct from ASCII DNA input.
    ///
    /// `ACTG` is mapped to `0123` in that specific order.
    ///
    /// This calls [`packed_seq::PackedSeqVec::from_ascii`] to allocate the packed data,
    /// and then forwards to [`QuadRanker::new_packed`].
    fn new_ascii_dna(seq: &[u8]) -> Self {
        // let mut packed_seq = seq.to_vec();
        let mut packed_seq = PackedSeqVec::from_ascii(seq).into_raw();
        // Add one block of padding.
        packed_seq.resize(packed_seq.len() + 1024, 0);
        let (head, data, _tail) = unsafe { packed_seq.align_to::<u64>() };
        assert!(head.is_empty());
        // assert!(tail.is_empty());
        Self::new_packed(data)
    }

    /// Construct from bitpacked data.
    ///
    /// Data is little-endian: the two lowest-order bits of the first word encode the first character.
    fn new_packed(seq: &[u64]) -> Self;

    /// Size in bytes of the data structure.
    fn size(&self) -> usize;

    /// Compute the rank of `c` of the given position.
    ///
    /// This is the number of occurrences of `c` _before_ position `pos`.
    ///
    /// This assumes that `0 <= pos < n`, where `n` is the length in characters of the input.
    /// Some implementations, including [`QuadRank`], also support `pos` equal to `n`.
    unsafe fn rank1_unchecked(&self, pos: usize, c: u8) -> u64;

    /// Compute the rank of the given position for all 4 symbols.
    ///
    /// This assumes that `0 <= pos < n`, where `n` is the length in characters of the input.
    /// Some implementations, including [`QuadRank`], also support `pos` equal to `n`.
    unsafe fn rank4_unchecked(&self, pos: usize) -> LongRanks;

    /// Prefetch the cacheline(s) needed to compute the rank of `c` of the given position.
    fn prefetch1(&self, pos: usize, c: u8);

    /// Prefetch the cacheline(s) needed to compute all 4 ranks at the given position.
    fn prefetch4(&self, pos: usize) {
        self.prefetch1(pos, 0);
    }

    /// Assumes pos<len.
    #[doc(hidden)]
    #[inline(always)]
    unsafe fn count1x2_unchecked(&self, pos0: usize, pos1: usize, c: u8) -> (u64, u64) {
        unsafe { (self.rank1_unchecked(pos0, c), self.rank1_unchecked(pos1, c)) }
    }
}
