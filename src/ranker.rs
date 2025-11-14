use packed_seq::{PackedSeqVec, SeqVec};

use crate::count::count_u8x8;
use crate::{Ranks, count4::CountFn};
// use packed_seq::{PackedSeqVec, SeqVec};
use std::marker::PhantomData;
use std::ops::Coroutine;

pub trait BasicBlock: Sync {
    /// Number of bytes per block.
    const B: usize;
    /// Number of characters per block.
    const N: usize;
    /// Bytes of the underlying count function.
    const C: usize;
    /// Bit-width of the internal global ranks.
    const W: usize;

    fn new(ranks: Ranks, data: &[u8; Self::B]) -> Self;
    /// Count the number of times each character occurs before position `pos`.
    fn count<CF: CountFn<{ Self::C }>, const C3: bool>(&self, pos: usize) -> Ranks;
    /// Count the number of times character `c` occurs before position `pos`.
    fn count1(&self, _pos: usize, _c: u8) -> u32 {
        unimplemented!()
    }
    /// count1 for two (usually closeby) positions in parallel.
    /// Main benefit is to interleave code, rather than actually reusing computed values.
    fn count1x2(&self, _other: &Self, _pos0: usize, _pos1: usize, _c: u8) -> (u32, u32) {
        unimplemented!()
    }
}

pub trait SuperBlock: Sync {
    /// Number of basic blocks.
    const BB: usize;
    /// Bit-width of the basic block ranks.
    const W: usize;

    fn new(ranks: [Ranks; Self::BB]) -> Self;
    fn get(&self, idx: usize) -> Ranks;
}

pub trait RankerT: Sync {
    fn new(seq: &[u8]) -> Self;
    /// Prefetch the cacheline for the given position.
    fn prefetch(&self, pos: usize);
    fn size(&self) -> usize;
    /// Count the number of times each character occurs before position `pos`.
    fn count(&self, pos: usize) -> Ranks;
    /// Count the number of times character `c` occurs before position `pos`.
    fn count1(&self, pos: usize, c: u8) -> u32;
    #[inline(always)]
    fn count1x2(&self, pos0: usize, pos1: usize, c: u8) -> (u32, u32) {
        (self.count1(pos0, c), self.count1(pos1, c))
    }

    #[inline(always)]
    fn count_coro(&self, pos: usize) -> impl Coroutine<Yield = (), Return = Ranks> + Unpin {
        self.prefetch(pos);
        #[inline(always)]
        #[coroutine]
        move || self.count(pos)
    }
    #[inline(always)]
    fn count_coro2(&self, pos: usize) -> impl Coroutine<Yield = (), Return = Ranks> + Unpin {
        #[inline(always)]
        #[coroutine]
        move || {
            self.prefetch(pos);
            yield;
            self.count(pos)
        }
    }
}

pub struct Ranker<BB: BasicBlock, SB: SuperBlock, CF: CountFn<{ BB::C }>, const C3: bool> {
    /// Cacheline-sized counts.
    blocks: Vec<BB>,
    /// Additional counts every 2^31 cachelines.
    super_blocks: Vec<SB>,
    cf: PhantomData<CF>,
}

impl<BB: BasicBlock, SB: SuperBlock, CF: CountFn<{ BB::C }>, const C3: bool> RankerT
    for Ranker<BB, SB, CF, C3>
where
    [(); BB::B]:,
    [(); SB::BB]:,
{
    fn new(seq: &[u8]) -> Self {
        // let mut packed_seq = seq.to_vec();
        let mut packed_seq = PackedSeqVec::from_ascii(seq).into_raw();
        // eprintln!("packed_seq: {:?}", packed_seq);
        // Add one block of padding.
        packed_seq.resize(packed_seq.len() + 2 * BB::B, 0);

        let mut ranks = [0u32; 4];
        let mut l_ranks = [0u32; 4];

        let chunks = packed_seq.as_chunks::<{ BB::B }>().0;
        let num_chunks = chunks.len();
        let num_long_chunks = num_chunks.div_ceil(Self::LONG_STRIDE);
        let mut block_ranks = Vec::with_capacity(num_long_chunks);
        let mut blocks = Vec::with_capacity(num_chunks);
        for (i, chunk) in chunks.iter().enumerate() {
            if (BB::W < 32) && i % Self::LONG_STRIDE == 0 {
                for i in 0..4 {
                    l_ranks[i] += ranks[i];
                }
                block_ranks.push(l_ranks);
                ranks = [0; 4];
            }
            blocks.push(BB::new(ranks, chunk));

            for chunk in chunk.as_chunks::<8>().0 {
                for c in 0..4 {
                    ranks[c as usize] += count_u8x8(chunk, c);
                }
            }
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
        if BB::W < 32 {
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
        let block_idx = pos / BB::N;
        let block_pos = pos % BB::N;
        let mut ranks = self.blocks[block_idx].count::<CF, C3>(block_pos);
        if (BB::W) < 32 {
            let long_pos = block_idx / Self::LONG_STRIDE;
            let long_ranks = self.super_blocks[long_pos / SB::BB].get(long_pos % SB::BB);
            for c in 0..4 {
                ranks[c] += long_ranks[c];
            }
        }
        ranks
    }
    /// Count the number of times character `c` occurs before position `pos`.
    #[inline(always)]
    fn count1(&self, pos: usize, c: u8) -> u32 {
        let block_idx = pos / BB::N;
        let block_pos = pos % BB::N;
        let mut rank = self.blocks[block_idx].count1(block_pos, c);
        if (BB::W) < 32 {
            let long_pos = block_idx / Self::LONG_STRIDE;
            let long_ranks = self.super_blocks[long_pos / SB::BB].get(long_pos % SB::BB);
            rank += long_ranks[c as usize];
        }
        rank
    }
    #[inline(always)]
    fn count1x2(&self, pos0: usize, pos1: usize, c: u8) -> (u32, u32) {
        let block_idx0 = pos0 / BB::N;
        let block_pos0 = pos0 % BB::N;
        let block_idx1 = pos1 / BB::N;
        let block_pos1 = pos1 % BB::N;
        let (mut rank0, mut rank1) =
            self.blocks[block_idx0].count1x2(&self.blocks[block_idx1], block_pos0, block_pos1, c);
        if (BB::W) < 32 {
            let long_pos0 = block_idx0 / Self::LONG_STRIDE;
            let long_pos1 = block_idx1 / Self::LONG_STRIDE;
            let long_ranks0 = self.super_blocks[long_pos0 / SB::BB].get(long_pos0 % SB::BB);
            let long_ranks1 = self.super_blocks[long_pos1 / SB::BB].get(long_pos1 % SB::BB);
            rank0 += long_ranks0[c as usize];
            rank1 += long_ranks1[c as usize];
        }
        (rank0, rank1)
    }
}
impl<BB: BasicBlock, SB: SuperBlock, CF: CountFn<{ BB::C }>, const C3: bool> Ranker<BB, SB, CF, C3>
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

/// Prefetch the given cacheline into L1 cache.
pub(crate) fn prefetch_index<T>(s: &[T], index: usize) {
    let ptr = s.as_ptr().wrapping_add(index) as *const u64;
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
    }
    #[cfg(target_arch = "x86")]
    unsafe {
        std::arch::x86::_mm_prefetch(ptr as *const i8, std::arch::x86::_MM_HINT_T0);
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        std::arch::aarch64::_prefetch(ptr as *const i8, std::arch::aarch64::_PREFETCH_LOCALITY3);
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86", target_arch = "aarch64")))]
    {
        // Do nothing.
    }
}
