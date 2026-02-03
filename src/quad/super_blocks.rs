use crate::quad::{SuperBlock, count4::count4_u8};

use super::{BasicBlock, LongRanks, Ranks};

#[derive(mem_dbg::MemSize)]
pub struct NoSB;

impl<BB: BasicBlock> SuperBlock<BB> for NoSB {
    const W: usize = 32;
    const SHIFT: usize = 0;
    #[inline(always)]
    fn new(_ranks: LongRanks, _data: &[u8]) -> Self {
        Self
    }
    #[inline(always)]
    fn get(&self, _block_idx: usize) -> LongRanks {
        [0; 4]
    }
}

#[repr(align(32))]
#[derive(mem_dbg::MemSize)]
pub struct TrivialSB {
    block: LongRanks,
}

impl<BB: BasicBlock> SuperBlock<BB> for TrivialSB {
    const W: usize = 0;
    const SHIFT: usize = 0;
    #[inline(always)]
    fn new(ranks: LongRanks, _data: &[u8]) -> Self {
        Self { block: ranks }
    }
    #[inline(always)]
    fn get(&self, _block_idx: usize) -> LongRanks {
        self.block
    }
}

#[repr(align(16))]
#[derive(mem_dbg::MemSize)]
pub struct ShiftSB {
    block: Ranks,
}

const SHIFT: usize = 13;

impl<BB: BasicBlock> SuperBlock<BB> for ShiftSB {
    const W: usize = 0;
    const SHIFT: usize = SHIFT;
    #[inline(always)]
    fn new(ranks: LongRanks, _data: &[u8]) -> Self {
        Self {
            block: ranks.map(|x| (x >> SHIFT) as u32),
        }
    }
    #[inline(always)]
    fn get(&self, _block_idx: usize) -> LongRanks {
        self.block.map(|x| (x as u64) << SHIFT)
    }
}

/// Store the high 32 bits of `x>>11`.
/// This stores the offset to the middle of the superblock.
/// Blocks in the left half use `offset-X`, where `X < BB::W` is the number of characters in the left half.
/// Blocks in the right half use `offset` directly.
#[derive(mem_dbg::MemSize)]
pub struct ShiftPairedSB {
    rank: Ranks,
}

impl<BB: BasicBlock> SuperBlock<BB> for ShiftPairedSB {
    const W: usize = 0;
    const SHIFT: usize = SHIFT;

    #[inline(always)]
    fn new(rank: LongRanks, data: &[u8]) -> Self {
        use crate::quad::ranker::strict_add;
        let half_count = data
            [..(<Self as SuperBlock<BB>>::BYTES_PER_SUPERBLOCK / 2).min(data.len())]
            .iter()
            .map(|&b| count4_u8(b).map(|x| x as u64))
            .fold([0u64; 4], strict_add);

        let rank = strict_add(rank, half_count).map(|x| x >> SHIFT);
        Self {
            rank: rank.map(|r| r.try_into().unwrap()),
        }
    }

    #[inline(always)]
    fn get(&self, block_idx: usize) -> LongRanks {
        let mid = <Self as SuperBlock<BB>>::BLOCKS_PER_SUPERBLOCK / 2;
        let bp_per_half = (BB::N * mid) as u64;
        let sub = if block_idx < mid { bp_per_half } else { 0 };
        // Unfortunately, this sub is wrapping to avoid underflows in the first few blocks.
        self.rank.map(|r| ((r as u64) << SHIFT).wrapping_sub(sub))
    }

    /// 2x larger than the default value.
    const BLOCKS_PER_SUPERBLOCK: usize = if BB::W <= SHIFT {
        // FIXME: Support W=0 for Shift=0
        panic!("W must be at least SHIFT")
    } else if BB::W >= super::TARGET_BITS {
        panic!("ShiftPairedSB does not make sense when superblocks are not needed anyway.")
    } else {
        (((1u128 << BB::W) / BB::N as u128) as usize - 1).next_power_of_two()
    };
}
