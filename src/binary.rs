pub mod blocks;
pub mod ranker;
pub mod super_blocks;
#[cfg(test)]
pub mod test;

pub use ranker::Ranker;
pub use super_blocks::TrivialSB;

/// By default, the library works for arrays with up to `2^43 = 1 TiB` of `1` bits.
/// This controls whether superblocks are used and/or prefetched.
pub const TARGET_BITS: usize = 43;

/// A basic block covers one cache line of bits.
pub trait BasicBlock: Send + Sync {
    /// Number of bits per basic block.
    /// We assume N is a multiple of 8.
    const N: usize;
    /// Number of bytes per basic block, equals N/8.
    const B: usize;
    /// Bit-width of the internal global ranks.
    const W: usize;
    /// `true` when the basic block returns right-inclusive counts.
    const INCLUSIVE: bool = false;

    /// Construct a new basic block with the given bits and given rank of the
    /// start of the block.
    ///
    /// `data` must have length exactly `B` bytes.
    fn new(rank: u64, data: &[u8]) -> Self;
    /// Count the number of times each character occurs before position `pos`.
    fn rank(&self, pos: usize) -> u64 {
        assert!(pos < Self::N);
        unsafe { self.rank_unchecked(pos) }
    }
    /// Unsafe version that assumes `pos < N`.
    unsafe fn rank_unchecked(&self, pos: usize) -> u64;
}

/// A super block stores ranks every X basic blocks, where X depends on the bit-width `W` of the counters inside the basic blocks.
/// A single super block stores ranks for 1 basic block.
///
/// (We don't do delta compression here for now.)
pub trait SuperBlock<BB: BasicBlock>: Sync + Send + Sized {
    /// Return the block and the stored value.
    fn new(ranks: u64, data: &[u8]) -> Self;
    /// Get the offset for basic block `idx` inside this super block.
    fn get(&self, block_idx: usize) -> u64;

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

pub trait RankerT: Sync + Sized {
    /// Construct from bitpacked data.
    fn new_packed(seq: &[usize]) -> Self;
    /// Size in bytes of the data structure.
    fn size(&self) -> usize;
    const HAS_PREFETCH: bool = false;
    /// Prefetch the cacheline for the given position.
    fn prefetch(&self, _pos: usize) {}
    /// Unsafe version that assumes `pos < len`.
    /// Pad the input to allow `pos=len`.
    unsafe fn rank_unchecked(&self, pos: usize) -> u64;
}
