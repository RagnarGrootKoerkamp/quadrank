use crate::count::count_u8x8;
use crate::{Ranks, count4::CountFn};
use packed_seq::{PackedSeqVec, SeqVec};
use std::ops::Coroutine;

pub trait Block {
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
}

#[derive(mem_dbg::MemSize)]
pub struct Ranker<B: Block> {
    /// Cacheline-sized counts.
    blocks: Vec<B>,
    /// Additional counts every 2^31 cachelines.
    long_ranks: Vec<Ranks>,
}

impl<B: Block> Ranker<B> {
    /// Store a new long block every this-many blocks.
    // Each long block should span N*x characters where N*x + N < 2^32, and x is fast to compute.
    // => x < 2^32 / N - 1
    const LONG_STRIDE: usize = if B::W == 0 {
        1
    } else {
        (((1u128 << B::W) / B::N as u128) as usize - 1).next_power_of_two() / 2
    };

    pub fn new(seq: &[u8]) -> Self
    where
        [(); B::B]:,
    {
        let mut packed_seq = PackedSeqVec::from_ascii(seq).into_raw();
        // Add one block of padding.
        packed_seq.resize(packed_seq.len() + 2 * B::B, 0);

        let mut ranks = [0u32; 4];
        let mut l_ranks = [0u64; 4];

        let chunks = packed_seq.as_chunks::<{ B::B }>().0;
        let num_chunks = chunks.len();
        let num_long_chunks = num_chunks.div_ceil(Self::LONG_STRIDE);
        let mut long_ranks = Vec::with_capacity(num_long_chunks);
        let mut blocks = Vec::with_capacity(num_chunks);
        for (i, chunk) in chunks.iter().enumerate() {
            if i % Self::LONG_STRIDE == 0 {
                for i in 0..4 {
                    l_ranks[i] += ranks[i] as u64;
                }
                long_ranks.push(l_ranks.map(|x| x as u32));
                ranks = [0; 4];
            }
            blocks.push(B::new(ranks, chunk));

            for chunk in chunk.as_chunks::<8>().0 {
                for c in 0..4 {
                    ranks[c as usize] += count_u8x8(chunk, c);
                }
            }
        }
        Self { blocks, long_ranks }
    }
    /// Prefetch the cacheline for the given position.
    #[inline(always)]
    pub fn prefetch(&self, pos: usize) {
        let block_idx = pos / B::N;
        prefetch_index(&self.blocks, block_idx);
        if B::W < 32 {
            let long_pos = block_idx / Self::LONG_STRIDE;
            prefetch_index(&self.long_ranks, long_pos);
        }
    }
    /// Count the number of times each character occurs before position `pos`.
    #[inline(always)]
    pub fn count<CF: CountFn<{ B::C }>, const C3: bool>(&self, pos: usize) -> Ranks {
        let block_idx = pos / B::N;
        let block_pos = pos % B::N;
        let mut ranks = self.blocks[block_idx].count::<CF, C3>(block_pos);
        if B::W < 32 {
            let long_pos = block_idx / Self::LONG_STRIDE;
            let long_ranks = self.long_ranks[long_pos];
            for c in 0..4 {
                ranks[c] += long_ranks[c];
            }
        }
        ranks
    }
    /// Count the number of times character `c` occurs before position `pos`.
    #[inline(always)]
    pub fn count1(&self, pos: usize, c: u8) -> u32 {
        let block_idx = pos / B::N;
        let block_pos = pos % B::N;
        self.blocks[block_idx].count1(block_pos, c)
    }

    #[inline(always)]
    pub fn count_coro<CF: CountFn<{ B::C }>, const C3: bool>(
        &self,
        pos: usize,
    ) -> impl Coroutine<Yield = (), Return = Ranks> + Unpin {
        self.prefetch(pos);
        #[inline(always)]
        #[coroutine]
        move || self.count::<CF, C3>(pos)
    }
    #[inline(always)]
    pub fn count_coro2<CF: CountFn<{ B::C }>, const C3: bool>(
        &self,
        pos: usize,
    ) -> impl Coroutine<Yield = (), Return = Ranks> + Unpin {
        #[inline(always)]
        #[coroutine]
        move || {
            self.prefetch(pos);
            yield;
            self.count::<CF, C3>(pos)
        }
    }
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
