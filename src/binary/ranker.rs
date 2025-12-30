use crate::ranker::prefetch_index;

use super::{BasicBlock, RankerT, SuperBlock, TrivialSB};

pub struct Ranker<BB: BasicBlock, SB: SuperBlock = TrivialSB> {
    basic_blocks: Vec<BB>,
    super_blocks: Vec<SB>,
}

impl<BB: BasicBlock, SB: SuperBlock> RankerT for Ranker<BB, SB>
where
    [(); BB::B]:,
    [(); SB::BB]:,
{
    fn new_packed(seq: &[usize]) -> Self {
        let (head, seq, tail) = unsafe { seq.align_to::<u8>() };
        assert!(head.is_empty());
        assert!(tail.is_empty());
        let mut rank = 0u64;
        let mut l_rank = 0u64;

        let (chunks, tail) = seq.as_chunks::<{ BB::B }>();
        let num_chunks = chunks.len();
        let num_long_chunks = num_chunks.div_ceil(Self::LONG_STRIDE);
        let mut block_ranks = Vec::with_capacity(num_long_chunks);
        let mut blocks = Vec::with_capacity(num_chunks);
        for (i, chunk) in chunks.iter().enumerate() {
            if ((BB::W) < 32) && i % Self::LONG_STRIDE == 0 {
                l_rank += rank;
                block_ranks.push(l_rank);
                rank = 0;
            }
            blocks.push(BB::new(rank, chunk));
            for byte in chunk {
                rank += byte.count_ones() as u64;
            }
        }

        {
            let i = chunks.len();
            let mut chunk = [0; BB::B];
            chunk[..tail.len()].copy_from_slice(tail);
            if ((BB::W) < 32) && i % Self::LONG_STRIDE == 0 {
                l_rank += rank;
                block_ranks.push(l_rank);
                rank = 0;
            }
            blocks.push(BB::new(rank, &chunk));
        }

        while block_ranks.len() % SB::BB != 0 {
            block_ranks.push(l_rank);
        }

        // convert block ranks to superblocks.
        let super_blocks = block_ranks
            .as_chunks()
            .0
            .iter()
            .map(|x| SB::new(*x))
            .collect();

        Self {
            basic_blocks: blocks,
            super_blocks,
        }
    }

    /// Prefetch the cacheline for the given position.
    #[inline(always)]
    fn prefetch(&self, pos: usize) {
        let block_idx = pos / BB::N;
        prefetch_index(&self.basic_blocks, block_idx);
        // if BB::W < 32 {
        if BB::W < 24 {
            let long_pos = block_idx / Self::LONG_STRIDE;
            prefetch_index(&self.super_blocks, long_pos / SB::BB);
        }
    }

    fn size(&self) -> usize {
        self.basic_blocks.len() * size_of::<BB>() + self.super_blocks.len() * size_of::<SB>()
    }

    /// Count the number of times each character occurs before position `pos`.
    #[inline(always)]
    unsafe fn rank_unchecked(&self, pos: usize) -> u64 {
        unsafe {
            let block_idx = pos / BB::N;
            let block_pos = pos % BB::N;
            debug_assert!(block_idx < self.basic_blocks.len());
            let mut rank = self.basic_blocks.get_unchecked(block_idx).rank(block_pos);
            if (BB::W) < 32 {
                let long_pos = block_idx / Self::LONG_STRIDE;
                let long_rank = self
                    .super_blocks
                    .get_unchecked(long_pos / SB::BB)
                    .get(long_pos % SB::BB);
                rank += long_rank;
            }
            rank
        }
    }
}
impl<BB: BasicBlock, SB: SuperBlock> Ranker<BB, SB>
where
    [(); BB::B]:,
    [(); SB::BB]:,
{
    /// Store a new long block every this-many blocks.
    // Each long block should span N*x characters where N*x + N < 2^32, and x is fast to compute.
    // => x < 2^32 / N - 1
    const LONG_STRIDE: usize = if BB::W == 0 {
        1
    } else if BB::W >= 32 {
        usize::MAX
    } else {
        (((1u128 << BB::W) / BB::N as u128) as usize - 1).next_power_of_two() / 2
    };
}
