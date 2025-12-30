use qwt::{RSQVector256, RankBin, RankQuad, WTSupport};

use crate::quad::{RankerT, Ranks};

impl RankerT for RSQVector256 {
    #[inline(always)]
    fn new(seq: &[u8]) -> Self {
        let seq = seq.iter().map(|x| (x >> 1) & 3).collect::<Vec<_>>();
        // eprintln!("packed seq: {:?}", &seq);
        // let seq = seq
        //     .iter()
        //     .flat_map(|x| (0..4).map(move |i| (x >> (2 * i)) & 3))
        //     .collect::<Vec<_>>();
        RSQVector256::new(&seq)
    }

    #[inline(always)]
    fn new_packed(seq: &[usize]) -> Self {
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
    fn prefetch(&self, pos: usize) {
        self.prefetch_data(pos);
        self.prefetch_info(pos);
    }

    #[inline(always)]
    fn size(&self) -> usize {
        mem_dbg::MemSize::mem_size(self, Default::default())
    }

    #[inline(always)]
    fn count(&self, pos: usize) -> Ranks {
        std::array::from_fn(|c| unsafe { self.rank_unchecked(c as u8, pos) as u32 })
    }

    #[inline(always)]
    fn count1(&self, pos: usize, c: u8) -> usize {
        unsafe { self.rank_unchecked(c, pos) as usize }
    }
}

impl crate::binary::RankerT for qwt::RSNarrow {
    #[inline(always)]
    fn new_packed(seq: &[usize]) -> Self {
        let mut bitvec = qwt::BitVectorMut::default();
        for &x in seq {
            bitvec.append_bits(x as u64, usize::BITS as usize);
        }

        Self::new(bitvec.into())
    }

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
    fn new_packed(seq: &[usize]) -> Self {
        Self::new(qwt::BitVector::from_slice(seq))
    }

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
