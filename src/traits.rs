use crate::count::count_u8x8;
use crate::{Ranks, count4::CountFn};
use packed_seq::{PackedSeqVec, SeqVec};
use std::ops::Coroutine;

pub trait Block {
    /// Number of bytes per block.
    const B: usize;
    /// Number of charactcrate::{Ranks, count4::CountFn}k.
    const N: usize;
    /// Bytes of the underlying count function.
    const C: usize;

    fn new(ranks: Ranks, data: &[u8; Self::B]) -> Self;
    fn count<CF: CountFn<{ Self::C }>, const C3: bool>(&self, pos: usize) -> Ranks;
    fn count1(&self, pos: usize, c: u8) -> u32 {
        unimplemented!()
    }
}

pub struct Ranker<B: Block> {
    blocks: Vec<B>,
}

impl<B: Block> Ranker<B> {
    pub fn new(seq: &[u8]) -> Self
    where
        [(); { B::B }]:,
    {
        let mut packed_seq = PackedSeqVec::from_ascii(seq).into_raw();
        packed_seq.resize(packed_seq.len() + B::B, 0);

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
    #[inline(always)]
    pub fn prefetch(&self, pos: usize) {
        let block_idx = pos / B::N;
        prefetch_index(&self.blocks, block_idx);
    }
    #[inline(always)]
    pub fn count<CF: CountFn<{ B::C }>, const C3: bool>(&self, pos: usize) -> Ranks {
        let block_idx = pos / B::N;
        let block_pos = pos % B::N;
        self.blocks[block_idx].count::<CF, C3>(block_pos)
    }
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

fn yield_no_wake() -> impl Future<Output = ()> {
    struct Yield(bool);
    impl Yield {
        fn new() -> Self {
            Yield(false)
        }
    }
    impl Future for Yield {
        type Output = ();

        #[inline(always)]
        fn poll(
            self: std::pin::Pin<&mut Self>,
            cx: &mut std::task::Context<'_>,
        ) -> std::task::Poll<()> {
            if self.0 {
                std::task::Poll::Ready(())
            } else {
                self.get_mut().0 = true;
                // cx.waker().wake_by_ref();
                std::task::Poll::Pending
            }
        }
    }

    Yield::new()
}
