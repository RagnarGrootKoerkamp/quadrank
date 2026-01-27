use crate::quad::{SuperBlock, count4::count4_u8};

use super::{BasicBlock, LongRanks, Ranks};

#[derive(mem_dbg::MemSize)]
pub struct NoSB;

impl<BB: BasicBlock> SuperBlock<BB> for NoSB {
    const NBB: usize = 1;
    const W: usize = 32;
    const SHIFT: usize = 0;
    #[inline(always)]
    fn new(_ranks: [LongRanks; <Self as SuperBlock<BB>>::NBB], _data: &[u8]) -> Self {
        Self
    }
    #[inline(always)]
    fn get(&self, _idx: usize, _block_idx: usize) -> LongRanks {
        [0; 4]
    }
}

#[repr(align(32))]
#[derive(mem_dbg::MemSize)]
pub struct TrivialSB {
    block: LongRanks,
}

impl<BB: BasicBlock> SuperBlock<BB> for TrivialSB {
    const NBB: usize = 1;
    const W: usize = 0;
    const SHIFT: usize = 0;
    #[inline(always)]
    fn new(ranks: [LongRanks; <Self as SuperBlock<BB>>::NBB], _data: &[u8]) -> Self {
        Self { block: ranks[0] }
    }
    #[inline(always)]
    fn get(&self, idx: usize, _block_idx: usize) -> LongRanks {
        debug_assert!(idx == 0);
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
    const NBB: usize = 1;
    const W: usize = 0;
    const SHIFT: usize = SHIFT;
    #[inline(always)]
    fn new(ranks: [LongRanks; <Self as SuperBlock<BB>>::NBB], _data: &[u8]) -> Self {
        Self {
            block: ranks[0].map(|x| (x >> SHIFT) as u32),
        }
    }
    #[inline(always)]
    fn get(&self, idx: usize, _block_idx: usize) -> LongRanks {
        debug_assert!(idx == 0);
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
    const NBB: usize = 1;
    const W: usize = 0;
    const SHIFT: usize = SHIFT;

    #[inline(always)]
    fn new(rank: [LongRanks; <Self as SuperBlock<BB>>::NBB], data: &[u8]) -> Self {
        use crate::quad::ranker::strict_add;
        let half_count = data
            [..(<Self as SuperBlock<BB>>::BYTES_PER_SUPERBLOCK / 2).min(data.len())]
            .iter()
            .map(|&b| count4_u8(b).map(|x| x as u64))
            .fold([0u64; 4], strict_add);

        let rank = strict_add(rank[0], half_count).map(|x| x >> SHIFT);
        Self {
            rank: rank.map(|r| r.try_into().unwrap()),
        }
    }

    #[inline(always)]
    fn get(&self, idx: usize, block_idx: usize) -> LongRanks {
        debug_assert!(idx == 0);
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

/// Super block inspired by QWT.
#[repr(align(64))]
#[derive(mem_dbg::MemSize)]
pub struct SB8 {
    /// One u128 per character, encoding:
    /// - low 32 bits: super block offset shifted by 8
    /// - high 8*12 bits: counts to each block.
    blocks: [u128; 4],
}

impl<BB: BasicBlock> SuperBlock<BB> for SB8 {
    const NBB: usize = 8;
    const W: usize = 0;
    const SHIFT: usize = 0;

    #[inline(always)]
    fn new(ranks: [LongRanks; <Self as SuperBlock<BB>>::NBB], _data: &[u8]) -> Self {
        Self {
            blocks: std::array::from_fn(|c| {
                // Super block offset
                let base = ranks[0][c] >> 8 << 8;
                let mut x = (base >> 8) as u128;
                // Block counts
                for i in 0..8 {
                    let diff = ranks[i][c] - base;
                    assert!(diff < (1 << 12));
                    x |= (diff as u128) << (32 + i * 12);
                }
                x
            }),
        }
    }

    #[inline(always)]
    fn get(&self, idx: usize, _block_idx: usize) -> LongRanks {
        std::array::from_fn(|c| {
            let x = self.blocks[c];
            let base = (x as u32 as u64) << 8;
            let diff = ((x >> (32 + idx * 12)) & 0xFFF) as u64;
            base + diff
        })
    }
}
