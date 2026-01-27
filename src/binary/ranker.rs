use std::{iter::zip, mem::MaybeUninit};

use prefetch_index::prefetch_index;

use super::{BasicBlock, RankerT, SuperBlock, TARGET_BITS, super_blocks::ShiftSB};
use rayon::prelude::*;

pub struct Ranker<BB: BasicBlock, SB: SuperBlock<BB> = ShiftSB> {
    basic_blocks: Vec<BB>,
    super_blocks: Vec<SB>,
}

impl<BB: BasicBlock, SB: SuperBlock<BB>> RankerT for Ranker<BB, SB>
where
    [(); BB::B]:,
{
    fn new_packed(seq: &[usize]) -> Self {
        let (head, seq, tail) = unsafe { seq.align_to::<u8>() };
        assert!(head.is_empty());
        assert!(tail.is_empty());
        let n_blocks = seq.len().div_ceil(BB::B);

        // 1. Count ones in each superblock.
        let mut sb_offsets: Vec<u64> = seq
            .par_chunks(SB::BYTES_PER_SUPERBLOCK)
            .map(|slice| slice.iter().map(|&b| b.count_ones() as u64).sum())
            .collect();

        // 2. Accumulate to get superblock offsets.
        {
            let mut sum = 0;
            for i in 0..sb_offsets.len() {
                let cnt = sb_offsets[i];
                sb_offsets[i] = sum;
                sum += cnt;
            }
        }

        // 3. Allocate space for blocks.
        let mut blocks = vec![];
        blocks.resize_with(n_blocks, MaybeUninit::<BB>::uninit);

        let sb_chunks = seq.par_chunks(SB::BYTES_PER_SUPERBLOCK);
        let super_blocks = sb_chunks
            .zip(sb_offsets)
            .zip(blocks.par_chunks_mut(SB::BLOCKS_PER_SUPERBLOCK))
            .map(|((sb_chunk, sb_offset), blocks)| {
                let sb = SB::new(sb_offset, sb_chunk);

                let bb_chunks = sb_chunk.chunks(BB::B);
                let mut delta = 0u64;

                for (i, (block, bb_chunk)) in zip(blocks, bb_chunks).enumerate() {
                    // This must be wrapping since `get_for_block` can return negative values.
                    let remaining_delta = (sb_offset + delta).wrapping_sub(sb.get(i));

                    let mut bb_chunk_buffer = [0u8; BB::B];
                    let bb_chunk = bb_chunk.as_array().unwrap_or_else(|| {
                        bb_chunk_buffer[..bb_chunk.len()].copy_from_slice(bb_chunk);
                        bb_chunk_buffer[bb_chunk.len()..].fill(0);
                        &bb_chunk_buffer
                    });

                    block.write(BB::new(remaining_delta, bb_chunk));

                    let count = bb_chunk.iter().map(|&b| b.count_ones() as u64).sum::<u64>();
                    delta += count;
                }

                sb
            })
            .collect::<Vec<_>>();

        Self {
            basic_blocks: unsafe { std::mem::transmute::<Vec<MaybeUninit<BB>>, Vec<BB>>(blocks) },
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
            let long_pos = block_idx / SB::BLOCKS_PER_SUPERBLOCK;
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
                let long_pos = block_idx / SB::BLOCKS_PER_SUPERBLOCK;
                let long_rank = self
                    .super_blocks
                    .get_unchecked(long_pos)
                    .get(block_idx % SB::BLOCKS_PER_SUPERBLOCK);
                rank = rank.wrapping_add(long_rank);
            }
            rank
        }
    }
}
