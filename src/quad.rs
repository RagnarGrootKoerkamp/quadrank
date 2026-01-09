pub mod blocks;
pub mod count4;
pub mod ranker;
pub mod super_blocks;
#[cfg(test)]
pub mod test;

use crate::quad::count4::CountFn;
use packed_seq::{PackedSeqVec, SeqVec};
use std::array::from_fn;
use std::ops::Coroutine;

pub use ranker::Ranker;
pub use super_blocks::TrivialSB;

pub type Ranks = [u32; 4];
pub type LongRanks = [u64; 4];

pub type QuartRank = Ranker<blocks::QuartBlock, super_blocks::NoSB, count4::SimdCount10, false>;
pub type HexRank =
    Ranker<blocks::HexaBlockMid4, super_blocks::TrivialSB, count4::SimdCount10, false>;
pub type QwtRank = ::qwt::RSQVector256;

pub type FastRank =
    Ranker<blocks::FullBlockTransposed, super_blocks::HalfSB, count4::SimdCount11B, false>;
pub type MidRank = Ranker<blocks::TriBlock2, super_blocks::HalfSB, count4::SimdCount11B, false>;
pub type SmallRank =
    Ranker<blocks::FullDouble16Inl, super_blocks::HalfSB, count4::SimdCount11B, false>;

#[inline(always)]
fn add(a: Ranks, b: Ranks) -> Ranks {
    from_fn(|c| a[c] + b[c])
}

pub trait BasicBlock: Sync {
    /// Character width. 1 for binary, 2 for DNA.
    const X: usize;

    /// Number of bytes per block.
    const B: usize;
    /// Number of characters per block.
    const N: usize;
    /// Bytes of the underlying count function.
    const C: usize;
    /// Bit-width of the internal global ranks.
    const W: usize;
    /// When `false`, basic data is bit-packed.
    /// When `true`, transposed layout is used with a u64 of high bits followed
    /// by a u64 of low bits.
    const TRANSPOSED: bool;

    fn new(ranks: Ranks, data: &[u8; Self::B]) -> Self;
    /// Count the number of times each character occurs before position `pos`.
    fn count<CF: CountFn<{ Self::C }, TRANSPOSED = { Self::TRANSPOSED }>, const C3: bool>(
        &self,
        pos: usize,
    ) -> Ranks;
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

    /// This many low bits are shifted out.
    const SHIFT: usize;

    fn new(ranks: [LongRanks; Self::BB]) -> Self;
    fn get(&self, idx: usize) -> LongRanks;
    fn get1(&self, idx: usize, c: u8) -> usize {
        self.get(idx)[c as usize] as usize
    }
}

pub trait RankerT: Sync + Sized {
    /// Construct from ASCII DNA input.
    fn new(seq: &[u8]) -> Self {
        // let mut packed_seq = seq.to_vec();
        let mut packed_seq = PackedSeqVec::from_ascii(seq).into_raw();
        // eprintln!("packed_seq: {:?}", packed_seq);
        // Add one block of padding.
        packed_seq.resize(packed_seq.len() + 1024, 0);
        let (head, data, tail) = unsafe { packed_seq.align_to::<usize>() };
        assert!(head.is_empty());
        assert!(tail.is_empty());
        Self::new_packed(data)
    }
    /// Construct from bitpacked data.
    fn new_packed(seq: &[usize]) -> Self;
    /// Prefetch the cacheline for the given position.
    fn prefetch(&self, pos: usize);
    fn size(&self) -> usize;
    /// Count the number of times each character occurs before position `pos`.
    fn count(&self, pos: usize) -> LongRanks;
    /// Count the number of times character `c` occurs before position `pos`.
    fn count1(&self, pos: usize, c: u8) -> usize;
    #[inline(always)]
    fn count1x2(&self, pos0: usize, pos1: usize, c: u8) -> (usize, usize) {
        (self.count1(pos0, c), self.count1(pos1, c))
    }

    #[inline(always)]
    fn count_coro(&self, pos: usize) -> impl Coroutine<Yield = (), Return = LongRanks> + Unpin {
        self.prefetch(pos);
        #[inline(always)]
        #[coroutine]
        move || self.count(pos)
    }
    #[inline(always)]
    fn count_coro2(&self, pos: usize) -> impl Coroutine<Yield = (), Return = LongRanks> + Unpin {
        #[inline(always)]
        #[coroutine]
        move || {
            self.prefetch(pos);
            yield;
            self.count(pos)
        }
    }
}
