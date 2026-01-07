use crate::quad::SuperBlock;

use super::{LongRanks, Ranks};

#[derive(mem_dbg::MemSize)]
pub struct NoSB;

impl SuperBlock for NoSB {
    const BB: usize = 1;
    const W: usize = 32;
    #[inline(always)]
    fn new(_ranks: [LongRanks; 1]) -> Self {
        Self
    }
    #[inline(always)]
    fn get(&self, _idx: usize) -> LongRanks {
        [0; 4]
    }
}

#[repr(align(32))]
#[derive(mem_dbg::MemSize)]
pub struct TrivialSB {
    block: LongRanks,
}

impl SuperBlock for TrivialSB {
    const BB: usize = 1;
    const W: usize = 0;
    #[inline(always)]
    fn new(ranks: [LongRanks; 1]) -> Self {
        Self { block: ranks[0] }
    }
    #[inline(always)]
    fn get(&self, idx: usize) -> LongRanks {
        debug_assert!(idx == 0);
        self.block
    }
}

#[repr(align(16))]
#[derive(mem_dbg::MemSize)]
pub struct HalfSB {
    block: Ranks,
}

impl SuperBlock for HalfSB {
    const BB: usize = 1;
    const W: usize = 0;
    #[inline(always)]
    fn new(ranks: [LongRanks; 1]) -> Self {
        Self {
            block: ranks[0].map(|x| (x >> 8) as u32),
        }
    }
    #[inline(always)]
    fn get(&self, idx: usize) -> LongRanks {
        debug_assert!(idx == 0);
        self.block.map(|x| (x as u64) << 8)
    }
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

impl SuperBlock for SB8 {
    const BB: usize = 8;
    const W: usize = 0;

    #[inline(always)]
    fn new(ranks: [LongRanks; 8]) -> Self {
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
    fn get(&self, idx: usize) -> LongRanks {
        std::array::from_fn(|c| {
            let x = self.blocks[c];
            let base = (x as u32 as u64) << 8;
            let diff = ((x >> (32 + idx * 12)) & 0xFFF) as u64;
            base + diff
        })
    }
}
