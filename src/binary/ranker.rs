use crate::prefetch_index;

use super::{BasicBlock, RankerT, SuperBlock, TrivialSB};

pub struct Ranker<BB: BasicBlock, SB: SuperBlock = TrivialSB> {
    basic_blocks: Vec<BB>,
    super_blocks: Vec<SB>,
}

const TARGET_BITS: usize = 40;

impl<BB: BasicBlock, SB: SuperBlock> RankerT for Ranker<BB, SB>
where
    [(); BB::B]:,
{
    fn new_packed(seq: &[usize]) -> Self {
        let (head, seq, tail) = unsafe { seq.align_to::<u8>() };
        assert!(head.is_empty());
        assert!(tail.is_empty());
        let mut rank = 0u64;
        let mut l_rank = 0u64;

        let chunks = seq.chunks(BB::B);
        let num_chunks = seq.len().div_ceil(BB::B);
        let num_long_chunks = num_chunks.div_ceil(Self::LONG_STRIDE);
        let mut super_blocks = Vec::with_capacity(num_long_chunks);
        let mut blocks = Vec::with_capacity(num_chunks);
        let mut array_chunk: [u8; BB::B];
        for (i, chunk) in chunks.enumerate() {
            if ((BB::W) < TARGET_BITS) && i % Self::LONG_STRIDE == 0 {
                l_rank += rank;
                // returns the actual stored rank value
                let (super_block, new_l_rank) = SB::new(l_rank);
                super_blocks.push(super_block);
                // 'leftover' rank that was not stored in the super block
                rank = l_rank - new_l_rank;
                // update l_rank to last stored value
                l_rank = new_l_rank;
            }
            let array_chunk = if chunk.len() == BB::B {
                chunk.try_into().unwrap()
            } else {
                // pad the last chunk with zeros.
                array_chunk = [0u8; BB::B];
                array_chunk[..chunk.len()].copy_from_slice(chunk);
                &array_chunk
            };
            blocks.push(BB::new(rank, array_chunk));
            for byte in chunk {
                rank += byte.count_ones() as u64;
            }
        }

        Self {
            basic_blocks: blocks,
            super_blocks,
        }
    }

    const HAS_PREFETCH: bool = true;

    /// Prefetch the cacheline for the given position.
    #[inline(always)]
    fn prefetch(&self, pos: usize) {
        let block_idx = pos / BB::N;
        prefetch_index(&self.basic_blocks, block_idx);
        if BB::W < TARGET_BITS - 12 {
            let long_pos = block_idx / Self::LONG_STRIDE;
            prefetch_index(&self.super_blocks, long_pos);
        }
    }

    fn size(&self) -> usize {
        self.basic_blocks.len() * size_of::<BB>() + self.super_blocks.len() * size_of::<SB>()
    }

    /// Count the number of times each character occurs before position `pos`.
    #[inline(always)]
    unsafe fn rank_unchecked(&self, mut pos: usize) -> u64 {
        unsafe {
            if BB::INCLUSIVE {
                if pos == 0 {
                    return 0;
                }
                pos -= 1;
            }

            let block_idx = pos / BB::N;
            let block_pos = pos % BB::N;
            debug_assert!(block_idx < self.basic_blocks.len());
            let mut rank = self.basic_blocks.get_unchecked(block_idx).rank(block_pos);
            if (BB::W) < TARGET_BITS {
                let long_pos = block_idx / Self::LONG_STRIDE;
                let long_rank = self.super_blocks.get_unchecked(long_pos).get();
                rank += long_rank;
            }
            rank
        }
    }
}

impl<BB: BasicBlock, SB: SuperBlock> Ranker<BB, SB>
where
    [(); BB::B]:,
{
    /// Store a new long block every this-many blocks.
    // Each long block should span N*x characters where N*x + N < 2^32, and x is fast to compute.
    // => x < 2^32 / N - 1
    const LONG_STRIDE: usize = if BB::W == 0 {
        1
    } else if BB::W >= TARGET_BITS {
        usize::MAX
    } else {
        (((1u128 << BB::W) / BB::N as u128) as usize - 1).next_power_of_two() / 2
    };
}
