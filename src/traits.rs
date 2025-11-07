use std::ops::Coroutine;

use packed_seq::{PackedSeqVec, SeqVec};

use crate::{Ranks, prefetch_index};

trait Block {
    /// Number of characters per block.
    const N: usize;

    fn new(data: &[u8]) -> Self;
    fn count(&self, pos: usize) -> Ranks;
}

struct Ranker<B: Block> {
    blocks: Vec<B>,
}

impl<B: Block> Ranker<B> {
    fn new(seq: &[u8]) -> Self {
        let mut packed_seq = PackedSeqVec::from_ascii(seq).into_raw();
        packed_seq.reserve(B::N / 4);
        let bytes_per_block = B::N.exact_div(4).unwrap();
        Self {
            blocks: packed_seq
                .chunks_exact(bytes_per_block)
                .map(|chunk| B::new(chunk))
                .collect(),
        }
    }
    fn prefetch(&self, pos: usize) {
        let block_idx = pos / B::N;
        prefetch_index(&self.blocks, block_idx);
    }
    fn count(&self, pos: usize) -> Ranks {
        let block_idx = pos / B::N;
        let block_pos = pos % B::N;
        self.blocks[block_idx].count(block_pos)
    }
    fn count_coro(&self, pos: usize) -> impl Coroutine<Yield = (), Return = Ranks> + Unpin {
        self.prefetch(pos);
        #[inline(always)]
        #[coroutine]
        move || self.count(pos)
    }
    fn count_coro2(&self, pos: usize) -> impl Coroutine<Yield = (), Return = Ranks> + Unpin {
        #[inline(always)]
        #[coroutine]
        move || {
            self.prefetch(pos);
            self.count(pos)
        }
    }
}
