use crate::Ranks;
use crate::count::count_u8x8;
use packed_seq::{PackedSeqVec, SeqVec};
use std::ops::Coroutine;

pub trait Block {
    /// Number of bytes per block.
    const B: usize;
    /// Number of characters per block.
    const N: usize;

    fn new(ranks: Ranks, data: &[u8; Self::B]) -> Self;
}

pub trait BlockCount<Block> {
    fn count(block: &Block, pos: usize) -> Ranks;
}

struct Ranker<B: Block> {
    blocks: Vec<B>,
}

impl<B: Block> Ranker<B> {
    fn new(seq: &[u8]) -> Self
    where
        [(); { B::B }]:,
    {
        let mut packed_seq = PackedSeqVec::from_ascii(seq).into_raw();
        packed_seq.reserve(B::B);

        let mut ranks = [0; 4];
        Self {
            blocks: packed_seq
                .as_chunks::<{ B::B }>()
                .0
                .iter()
                .map(|chunk| {
                    let start_ranks = ranks;
                    for chunk in chunk.as_chunks::<8>().0 {
                        for c in 0..4 {
                            ranks[c as usize] += count_u8x8(chunk, c);
                        }
                    }

                    B::new(start_ranks, chunk)
                })
                .collect(),
        }
    }
    fn prefetch(&self, pos: usize) {
        let block_idx = pos / B::N;
        prefetch_index(&self.blocks, block_idx);
    }
    fn count<BR: BlockCount<B>>(&self, pos: usize) -> Ranks {
        let block_idx = pos / B::N;
        let block_pos = pos % B::N;
        BR::count(&self.blocks[block_idx], block_pos)
    }
    fn count_coro<BR: BlockCount<B>>(
        &self,
        pos: usize,
    ) -> impl Coroutine<Yield = (), Return = Ranks> + Unpin {
        self.prefetch(pos);
        #[inline(always)]
        #[coroutine]
        move || self.count::<BR>(pos)
    }
    fn count_coro2<BR: BlockCount<B>>(
        &self,
        pos: usize,
    ) -> impl Coroutine<Yield = (), Return = Ranks> + Unpin {
        #[inline(always)]
        #[coroutine]
        move || {
            self.prefetch(pos);
            self.count::<BR>(pos)
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
