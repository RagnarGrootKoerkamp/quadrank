use super::count4::CountFn;
use super::{BasicBlock, RankerT, Ranks, SuperBlock};
use crate::count::{count_u8, count_u8x8};
use crate::prefetch_index;
use std::marker::PhantomData;

pub struct Ranker<BB: BasicBlock, SB: SuperBlock, CF: CountFn<{ BB::C }>, const C3: bool> {
    /// Cacheline-sized counts.
    blocks: Vec<BB>,
    /// Additional sparse counts.
    super_blocks: Vec<SB>,
    cf: PhantomData<CF>,
}

impl<
    BB: BasicBlock,
    SB: SuperBlock,
    CF: CountFn<{ BB::C }, TRANSPOSED = { BB::TRANSPOSED }>,
    const C3: bool,
> RankerT for Ranker<BB, SB, CF, C3>
where
    [(); BB::B]:,
    [(); SB::BB]:,
{
    fn new_packed(seq: &[usize]) -> Self {
        let (head, seq, tail) = unsafe { seq.align_to::<u8>() };
        assert!(head.is_empty());
        assert!(tail.is_empty());
        let mut ranks = [0u32; 4];
        let mut l_ranks = [0u32; 4];

        let (chunks, tail) = seq.as_chunks::<{ BB::B }>();
        let num_chunks = chunks.len();
        let num_long_chunks = num_chunks.div_ceil(Self::LONG_STRIDE);
        let mut block_ranks = Vec::with_capacity(num_long_chunks);
        let mut blocks = Vec::with_capacity(num_chunks);
        for (i, chunk) in chunks.iter().enumerate() {
            if ((BB::W) < 32) && i % Self::LONG_STRIDE == 0 {
                for i in 0..4 {
                    l_ranks[i] += ranks[i];
                }
                block_ranks.push(l_ranks);
                ranks = [0; 4];
            }
            blocks.push(BB::new(ranks, chunk));

            let (words, tail) = chunk.as_chunks::<8>();
            for chunk in words {
                if BB::X == 2 {
                    for c in 0..4 {
                        ranks[c as usize] += count_u8x8(chunk, c);
                    }
                } else {
                    ranks[0] += u64::from_le_bytes(*chunk).count_ones() as u32;
                }
            }
            for &byte in tail {
                if BB::X == 2 {
                    for c in 0..4 {
                        ranks[c as usize] += count_u8(byte, c);
                    }
                } else {
                    ranks[0] += byte.count_ones() as u32;
                }
            }
        }

        {
            let i = chunks.len();
            let mut chunk = [0; BB::B];
            chunk[..tail.len()].copy_from_slice(tail);
            if ((BB::W) < 32) && i % Self::LONG_STRIDE == 0 {
                for i in 0..4 {
                    l_ranks[i] += ranks[i];
                }
                block_ranks.push(l_ranks);
                ranks = [0; 4];
            }
            blocks.push(BB::new(ranks, &chunk));
        }

        while block_ranks.len() % SB::BB != 0 {
            block_ranks.push(l_ranks);
        }

        // convert block ranks to superblocks.
        let super_blocks = block_ranks
            .as_chunks()
            .0
            .iter()
            .map(|x| SB::new(*x))
            .collect();

        Self {
            blocks,
            super_blocks,
            cf: PhantomData,
        }
    }

    /// Prefetch the cacheline for the given position.
    #[inline(always)]
    fn prefetch(&self, pos: usize) {
        let block_idx = pos / BB::N;
        prefetch_index(&self.blocks, block_idx);
        // if BB::W < 32 {
        if BB::W < 24 {
            let long_pos = block_idx / Self::LONG_STRIDE;
            prefetch_index(&self.super_blocks, long_pos / SB::BB);
        }
    }

    fn size(&self) -> usize {
        self.blocks.len() * size_of::<BB>() + self.super_blocks.len() * size_of::<SB>()
    }

    /// Count the number of times each character occurs before position `pos`.
    #[inline(always)]
    fn count(&self, pos: usize) -> Ranks {
        // assert!(pos < self.len);
        unsafe {
            let block_idx = pos / BB::N;
            let block_pos = pos % BB::N;
            let mut ranks = self
                .blocks
                .get_unchecked(block_idx)
                .count::<CF, C3>(block_pos);
            if (BB::W) < 32 {
                let long_pos = block_idx / Self::LONG_STRIDE;
                let long_ranks = self
                    .super_blocks
                    .get_unchecked(long_pos / SB::BB)
                    .get(long_pos % SB::BB);
                for c in 0..4 {
                    ranks[c] += long_ranks[c];
                }
            }
            ranks
        }
    }
    /// Count the number of times character `c` occurs before position `pos`.
    #[inline(always)]
    fn count1(&self, pos: usize, c: u8) -> usize {
        // assert!(pos < self.len);
        unsafe {
            let block_idx = pos / BB::N;
            let block_pos = pos % BB::N;
            let mut rank = self.blocks.get_unchecked(block_idx).count1(block_pos, c) as usize;
            if (BB::W) < 32 {
                let long_pos = block_idx / Self::LONG_STRIDE;
                let long_rank = self
                    .super_blocks
                    .get_unchecked(long_pos / SB::BB)
                    .get1(long_pos % SB::BB);
                rank += long_rank;
            }
            rank
        }
    }
    #[inline(always)]
    fn count1x2(&self, pos0: usize, pos1: usize, c: u8) -> (usize, usize) {
        let block_idx0 = pos0 / BB::N;
        let block_pos0 = pos0 % BB::N;
        let block_idx1 = pos1 / BB::N;
        let block_pos1 = pos1 % BB::N;
        let (rank0, rank1) =
            self.blocks[block_idx0].count1x2(&self.blocks[block_idx1], block_pos0, block_pos1, c);
        let mut rank0 = rank0 as usize;
        let mut rank1 = rank1 as usize;
        if (BB::W) < 32 {
            let long_pos0 = block_idx0 / Self::LONG_STRIDE;
            let long_pos1 = block_idx1 / Self::LONG_STRIDE;
            let long_ranks0 = self.super_blocks[long_pos0 / SB::BB].get1(long_pos0 % SB::BB);
            let long_ranks1 = self.super_blocks[long_pos1 / SB::BB].get1(long_pos1 % SB::BB);
            rank0 += long_ranks0;
            rank1 += long_ranks1;
        }
        (rank0, rank1)
    }
}
impl<
    BB: BasicBlock,
    SB: SuperBlock,
    CF: CountFn<{ BB::C }, TRANSPOSED = { BB::TRANSPOSED }>,
    const C3: bool,
> Ranker<BB, SB, CF, C3>
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
