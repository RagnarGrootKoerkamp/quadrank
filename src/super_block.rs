use crate::{Ranks, ranker::SuperBlock};

#[derive(mem_dbg::MemSize)]
pub struct NoSB;

impl SuperBlock for NoSB {
    const BB: usize = 1;
    const W: usize = 32;
    #[inline(always)]
    fn new(_ranks: [crate::Ranks; 1]) -> Self {
        Self
    }
    #[inline(always)]
    fn get(&self, _idx: usize) -> crate::Ranks {
        [0; 4]
    }
}

#[repr(align(16))]
#[derive(mem_dbg::MemSize)]
pub struct TrivialSB {
    block: Ranks,
}

impl SuperBlock for TrivialSB {
    const BB: usize = 1;
    const W: usize = 0;
    #[inline(always)]
    fn new(ranks: [crate::Ranks; 1]) -> Self {
        Self { block: ranks[0] }
    }
    #[inline(always)]
    fn get(&self, idx: usize) -> crate::Ranks {
        debug_assert!(idx == 0);
        self.block
    }
}

/// Super block inspired by QWT.
#[repr(align(64))]
#[derive(mem_dbg::MemSize)]
pub struct SB8 {
    /// One u128 per character, encoding:
    /// - low 32 bits: super block offset
    /// - high 8*12 bits: counts to each block.
    blocks: [u128; 4],
}

impl SuperBlock for SB8 {
    const BB: usize = 8;
    const W: usize = 0;

    #[inline(always)]
    fn new(ranks: [crate::Ranks; 8]) -> Self {
        Self {
            blocks: std::array::from_fn(|c| {
                // Super block offset
                let base: u32 = ranks[0][c];
                let mut x = base as u128;
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
    fn get(&self, idx: usize) -> crate::Ranks {
        std::array::from_fn(|c| {
            let x = self.blocks[c];
            let base = x as u32;
            let diff = ((x >> (32 + idx * 12)) & 0xFFF) as u32;
            base + diff
        })
    }
}
