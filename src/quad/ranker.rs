use crate::quad::{
    BasicBlock, LongRanks, RankerT, SuperBlock,
    count4::{count4_u8, count4_u64},
    super_blocks::ShiftSB,
};
use prefetch_index::prefetch_index;
use rayon::prelude::*;
use std::array::from_fn;
use std::iter::zip;
use std::mem::MaybeUninit;

pub struct Ranker<BB: BasicBlock, SB: SuperBlock<BB> = ShiftSB> {
    /// Cacheline-sized counts.
    blocks: Vec<BB>,
    /// Additional sparse counts.
    super_blocks: Vec<SB>,
}

#[inline(always)]
pub(super) fn strict_add(a: LongRanks, b: LongRanks) -> LongRanks {
    from_fn(|c| a[c].strict_add(b[c]))
}

impl<BB: BasicBlock, SB: SuperBlock<BB>> RankerT for Ranker<BB, SB> {
    fn new_packed(seq: &[u64]) -> Self {
        let seq_usize = seq;
        let (head, seq, tail) = unsafe { seq.align_to::<u8>() };
        assert!(head.is_empty());
        assert!(tail.is_empty());

        let add_block = seq.len() % BB::B == 0;
        let add_superblock = seq.len() % (SB::BYTES_PER_SUPERBLOCK) == 0;

        let n_blocks = seq.len().div_ceil(BB::B) + (add_block as usize);

        // 1. Count ones in each superblock.
        assert!(SB::BYTES_PER_SUPERBLOCK % 8 == 0);
        let mut sb_offsets: Vec<LongRanks> = seq_usize
            .par_chunks(SB::BYTES_PER_SUPERBLOCK / 8)
            .map(|slice| {
                slice
                    .iter()
                    .map(|&b| count4_u64(b as u64).map(|x| x as u64))
                    .fold([0; 4], strict_add)
            })
            .collect();

        if add_superblock {
            sb_offsets.push([0; 4]);
        }

        // 2. Accumulate to get superblock offsets.
        {
            let mut sum = [0u64; 4];
            for i in 0..sb_offsets.len() {
                let cnt = sb_offsets[i];
                sb_offsets[i] = sum;
                sum = strict_add(sum, cnt);
            }
        }

        // 3. Allocate space for blocks.
        let mut blocks = vec![];
        blocks.resize_with(n_blocks, MaybeUninit::<BB>::uninit);

        let sb_chunks = seq.par_chunks(SB::BYTES_PER_SUPERBLOCK);
        let mut super_blocks = sb_chunks
            .zip(&sb_offsets)
            .zip(blocks.par_chunks_mut(SB::BLOCKS_PER_SUPERBLOCK))
            .map(|((sb_chunk, &sb_offset), blocks)| {
                let sb = SB::new(sb_offset, sb_chunk);

                let bb_chunks = sb_chunk.chunks(BB::B);
                let num_chunks = bb_chunks.len();
                let mut delta = [0u64; 4];

                for (i, (block, bb_chunk)) in zip(blocks.iter_mut(), bb_chunks).enumerate() {
                    // This must be wrapping since `get_for_block` can return negative values.
                    let a = strict_add(sb_offset, delta);
                    let b = sb.get(i);
                    let remaining_delta = from_fn(|i| a[i].wrapping_sub(b[i]) as u32);

                    let mut bb_chunk_buffer = vec![];
                    let bb_chunk = if bb_chunk.len() == BB::B {
                        bb_chunk
                    } else {
                        bb_chunk_buffer.resize(BB::B, 0u8);
                        bb_chunk_buffer[..bb_chunk.len()].copy_from_slice(bb_chunk);
                        bb_chunk_buffer[bb_chunk.len()..].fill(0);
                        &bb_chunk_buffer
                    };

                    block.write(BB::new(remaining_delta, bb_chunk));

                    let count: LongRanks = bb_chunk
                        .iter()
                        .map(|&b| count4_u8(b).map(|x| x as u64))
                        .fold([0; 4], strict_add);
                    delta = strict_add(delta, count);
                }

                if blocks.len() > num_chunks {
                    assert_eq!(blocks.len(), num_chunks + 1);
                    let i = num_chunks;
                    let a = strict_add(sb_offset, delta);
                    let b = sb.get(i);
                    let remaining_delta = from_fn(|i| a[i].wrapping_sub(b[i]) as u32);
                    blocks[i].write(BB::new(remaining_delta, &vec![0u8; BB::B]));
                }

                sb
            })
            .collect::<Vec<_>>();

        if add_superblock {
            let sb_offset = *sb_offsets.last().unwrap();
            assert_eq!(sb_offset.iter().sum::<u64>(), seq.len() as u64 * 4);
            super_blocks.push(SB::new(sb_offset, &[]));
            let sb = super_blocks.last().unwrap();
            let a = sb_offset;
            let b = sb.get(0);
            let remaining_delta = from_fn(|i| a[i].wrapping_sub(b[i]) as u32);
            blocks
                .last_mut()
                .unwrap()
                .write(BB::new(remaining_delta, &vec![0u8; BB::B]));
        }

        Self {
            blocks: unsafe { std::mem::transmute::<Vec<MaybeUninit<BB>>, Vec<BB>>(blocks) },
            super_blocks,
        }
    }

    /// Prefetch the cacheline for the given position.
    #[inline(always)]
    fn prefetch1(&self, pos: usize, _c: u8) {
        let block_idx = pos / BB::N;
        prefetch_index(&self.blocks, block_idx);
        if BB::W < 64 {
            let long_pos = block_idx / SB::BLOCKS_PER_SUPERBLOCK;
            prefetch_index(&self.super_blocks, long_pos);
        }
    }

    fn size(&self) -> usize {
        self.blocks.len() * size_of::<BB>() + self.super_blocks.len() * size_of::<SB>()
    }

    /// Count the number of times each character occurs before position `pos`.
    #[inline(always)]
    unsafe fn rank4_unchecked(&self, pos: usize) -> LongRanks {
        // assert!(pos < self.len);
        unsafe {
            let block_idx = pos / BB::N;
            let block_pos = pos % BB::N;
            let mut ranks = self
                .blocks
                .get_unchecked(block_idx)
                .count4(block_pos)
                .map(|x| x as u64);
            if BB::W < 64 {
                let long_pos = block_idx / SB::BLOCKS_PER_SUPERBLOCK;
                let long_ranks = self
                    .super_blocks
                    .get_unchecked(long_pos)
                    .get(block_idx % SB::BLOCKS_PER_SUPERBLOCK);
                for c in 0..4 {
                    ranks[c] = ranks[c].wrapping_add(long_ranks[c]);
                }
            }
            ranks
        }
    }
    /// Count the number of times character `c` occurs before position `pos`.
    #[inline(always)]
    unsafe fn rank1_unchecked(&self, pos: usize, c: u8) -> usize {
        // assert!(pos < self.len);
        unsafe {
            let block_idx = pos / BB::N;
            let block_pos = pos % BB::N;
            let mut rank = self.blocks.get_unchecked(block_idx).count1(block_pos, c) as usize;
            if BB::W < 64 {
                let long_pos = block_idx / SB::BLOCKS_PER_SUPERBLOCK;
                let long_rank = self
                    .super_blocks
                    .get_unchecked(long_pos)
                    .get1(block_idx % SB::BLOCKS_PER_SUPERBLOCK, c);
                rank += long_rank;
            }
            rank
        }
    }
    #[inline(always)]
    unsafe fn count1x2_unchecked(&self, pos0: usize, pos1: usize, c: u8) -> (usize, usize) {
        let block_idx0 = pos0 / BB::N;
        let block_pos0 = pos0 % BB::N;
        let block_idx1 = pos1 / BB::N;
        let block_pos1 = pos1 % BB::N;
        let (rank0, rank1) =
            self.blocks[block_idx0].count1x2(&self.blocks[block_idx1], block_pos0, block_pos1, c);
        let mut rank0 = rank0 as usize;
        let mut rank1 = rank1 as usize;
        if BB::W < 64 {
            let long_pos0 = block_idx0 / SB::BLOCKS_PER_SUPERBLOCK;
            let long_pos1 = block_idx1 / SB::BLOCKS_PER_SUPERBLOCK;
            let long_ranks0 =
                self.super_blocks[long_pos0].get1(block_idx0 % SB::BLOCKS_PER_SUPERBLOCK, c);
            let long_ranks1 =
                self.super_blocks[long_pos1].get1(block_idx1 % SB::BLOCKS_PER_SUPERBLOCK, c);
            rank0 += long_ranks0;
            rank1 += long_ranks1;
        }
        (rank0, rank1)
    }
}
