use qwt::{RankBin, RankQuad, WTSupport};

pub use qwt::RSNarrow;
pub use qwt::RSQVector256;
pub use qwt::RSQVector512;
pub use qwt::RSWide;

use crate::quad::{LongRanks, RankerT};

impl RankerT for RSQVector256 {
    #[inline(always)]
    fn new(seq: &[u8]) -> Self {
        let seq = seq.iter().map(|x| (x >> 1) & 3).collect::<Vec<_>>();
        RSQVector256::new(&seq)
    }

    #[inline(always)]
    fn new_packed(seq: &[u64]) -> Self {
        let bits = seq
            .iter()
            .flat_map(|word| {
                (0..usize::BITS)
                    .step_by(2)
                    .map(move |i| ((word >> i) & 3) as u8)
            })
            .collect::<Vec<u8>>();
        Self::new(&bits)
    }

    #[inline(always)]
    fn prefetch1(&self, pos: usize, _c: u8) {
        self.prefetch_data(pos);
        self.prefetch_info(pos);
    }

    #[inline(always)]
    fn size(&self) -> usize {
        mem_dbg::MemSize::mem_size(self, Default::default())
    }

    #[inline(always)]
    unsafe fn rank4_unchecked(&self, pos: usize) -> LongRanks {
        std::array::from_fn(|c| unsafe { self.rank_unchecked(c as u8, pos) as u64 })
    }

    #[inline(always)]
    unsafe fn rank1_unchecked(&self, pos: usize, c: u8) -> u64 {
        unsafe { self.rank_unchecked(c, pos) as u64 }
    }
}

impl RankerT for RSQVector512 {
    #[inline(always)]
    fn new(seq: &[u8]) -> Self {
        let seq = seq.iter().map(|x| (x >> 1) & 3).collect::<Vec<_>>();
        RSQVector512::new(&seq)
    }

    #[inline(always)]
    fn new_packed(seq: &[u64]) -> Self {
        let bits = seq
            .iter()
            .flat_map(|word| {
                (0..usize::BITS)
                    .step_by(2)
                    .map(move |i| ((word >> i) & 3) as u8)
            })
            .collect::<Vec<u8>>();
        Self::new(&bits)
    }

    #[inline(always)]
    fn prefetch1(&self, pos: usize, _c: u8) {
        self.prefetch_data(pos);
        self.prefetch_info(pos);
    }

    #[inline(always)]
    fn size(&self) -> usize {
        mem_dbg::MemSize::mem_size(self, Default::default())
    }

    #[inline(always)]
    unsafe fn rank4_unchecked(&self, pos: usize) -> LongRanks {
        std::array::from_fn(|c| unsafe { self.rank_unchecked(c as u8, pos) as u64 })
    }

    #[inline(always)]
    unsafe fn rank1_unchecked(&self, pos: usize, c: u8) -> u64 {
        unsafe { self.rank_unchecked(c, pos) as u64 }
    }
}

impl crate::binary::RankerT for qwt::RSNarrow {
    #[inline(always)]
    fn new_packed(seq: &[u64]) -> Self {
        let mut bitvec = qwt::BitVectorMut::default();
        for &x in seq {
            bitvec.append_bits(x as u64, usize::BITS as usize);
        }

        Self::new(bitvec.into())
    }

    const HAS_PREFETCH: bool = true;

    #[inline(always)]
    fn prefetch(&self, pos: usize) {
        RankBin::prefetch(self, pos);
    }

    #[inline(always)]
    fn size(&self) -> usize {
        mem_dbg::MemSize::mem_size(self, Default::default())
    }

    #[inline(always)]
    unsafe fn rank_unchecked(&self, pos: usize) -> u64 {
        unsafe { self.rank1_unchecked(pos) as u64 }
    }
}

impl crate::binary::RankerT for qwt::RSWide {
    #[inline(always)]
    fn new_packed(seq: &[u64]) -> Self {
        Self::new(qwt::BitVector::from_slice(unsafe {
            seq.align_to::<usize>().1
        }))
    }

    const HAS_PREFETCH: bool = true;

    #[inline(always)]
    fn prefetch(&self, pos: usize) {
        RankBin::prefetch(self, pos);
    }

    #[inline(always)]
    fn size(&self) -> usize {
        mem_dbg::MemSize::mem_size(self, Default::default())
    }

    #[inline(always)]
    unsafe fn rank_unchecked(&self, pos: usize) -> u64 {
        unsafe { self.rank1_unchecked(pos) as u64 }
    }
}
