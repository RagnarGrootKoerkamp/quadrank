pub mod blocks;
pub mod ranker;
pub mod super_blocks;
#[cfg(test)]
pub mod test;

pub use ranker::Ranker;
pub use super_blocks::TrivialSB;

/// A basic block covers one cache line of bits.
pub trait BasicBlock: Sync {
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
    fn new(rank: u64, data: &[u8; Self::B]) -> Self;
    /// Count the number of times each character occurs before position `pos`.
    fn rank(&self, pos: usize) -> u64 {
        assert!(pos < Self::N);
        unsafe { self.rank_unchecked(pos) }
    }
    /// Unsafe version that assumes `pos < N`.
    unsafe fn rank_unchecked(&self, pos: usize) -> u64;
}

/// A super block stores ranks every X basic blocks, where X depends on the bit-width `W` of the counters inside the basic blocks.
/// A single super block stores ranks for `BB` basic blocks, so it can delta-compress them if needed.
pub trait SuperBlock: Sync {
    /// Number of basic blocks covered by a single super block.
    const BB: usize;
    /// Bit-width of the basic block ranks.
    const W: usize;

    fn new(ranks: [u64; Self::BB]) -> Self;
    fn get(&self, idx: usize) -> u64;
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
