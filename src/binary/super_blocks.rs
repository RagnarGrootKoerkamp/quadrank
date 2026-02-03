use super::{BasicBlock, SuperBlock};

/// Do not store superblocks at all.
#[derive(mem_dbg::MemSize)]
pub struct NoSB;

impl<BB: BasicBlock> SuperBlock<BB> for NoSB {
    #[inline(always)]
    fn new(rank: u64, _data: &[u8]) -> Self {
        assert_eq!(rank, 0);
        Self
    }
    #[inline(always)]
    fn get(&self, _block_idx: usize) -> u64 {
        0
    }
}

/// Simply stores a `u64` rank.
#[derive(mem_dbg::MemSize)]
pub struct TrivialSB {
    rank: u64,
}

impl<BB: BasicBlock> SuperBlock<BB> for TrivialSB {
    #[inline(always)]
    fn new(rank: u64, _data: &[u8]) -> Self {
        Self { rank }
    }
    #[inline(always)]
    fn get(&self, _block_idx: usize) -> u64 {
        self.rank
    }
}

/// Store the high 32 bits of `x>>11`.
#[derive(mem_dbg::MemSize)]
pub struct ShiftSB {
    rank: u32,
}

const SHIFT: usize = 11;

impl<BB: BasicBlock> SuperBlock<BB> for ShiftSB {
    #[inline(always)]
    fn new(rank: u64, _data: &[u8]) -> Self {
        let rank = rank >> SHIFT;
        assert!(
            rank <= u32::MAX as u64,
            "Rank too large for HalfSB. Use TrivialSB instead."
        );
        Self {
            rank: rank.try_into().unwrap(),
        }
    }
    #[inline(always)]
    fn get(&self, _block_idx: usize) -> u64 {
        (self.rank as u64) << SHIFT
    }
}

/// Store the high 32 bits of `x>>11`.
/// This stores the offset to the middle of the superblock.
/// Blocks in the left half use `offset-2**16`, and blocks in the right half use `offset` directly.
#[derive(mem_dbg::MemSize)]
pub struct ShiftPairedSB {
    rank: u32,
}

impl<BB: BasicBlock> SuperBlock<BB> for ShiftPairedSB {
    #[inline(always)]
    fn new(rank: u64, data: &[u8]) -> Self {
        let half_count = data
            [..(<Self as SuperBlock<BB>>::BYTES_PER_SUPERBLOCK / 2).min(data.len())]
            .iter()
            .map(|&b| b.count_ones() as u64)
            .sum::<u64>();

        let rank = (rank + half_count) >> SHIFT;
        assert!(
            rank <= u32::MAX as u64,
            "Rank too large for HalfSB. Use TrivialSB instead."
        );
        Self {
            rank: rank.try_into().unwrap(),
        }
    }

    #[inline(always)]
    fn get(&self, block_idx: usize) -> u64 {
        // Unfortunately, this sub is wrapping to avoid underflows in the first few blocks.
        let mid_block = <Self as SuperBlock<BB>>::BLOCKS_PER_SUPERBLOCK / 2;
        let bits_per_half = BB::N * mid_block;
        ((self.rank as u64) << SHIFT).wrapping_sub(if block_idx < mid_block {
            bits_per_half as u64
        } else {
            0
        })
    }

    /// 2x larger than the default value.
    const BLOCKS_PER_SUPERBLOCK: usize = if BB::W == 0 {
        2
    } else if BB::W == 64 {
        panic!("ShiftPairedSB does not make sense when superblocks are not needed anyway.")
    } else {
        (((1u128 << BB::W) / BB::N as u128) as usize - 1).next_power_of_two()
    };
}
