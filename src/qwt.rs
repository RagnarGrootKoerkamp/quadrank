use qwt::{RSQVector256, RankBin, RankQuad, WTSupport};

use crate::ranker::RankerT;

impl RankerT for RSQVector256 {
    fn new(seq: &[u8]) -> Self {
        let seq = seq.iter().map(|x| (x >> 1) & 3).collect::<Vec<_>>();
        // eprintln!("packed seq: {:?}", &seq);
        // let seq = seq
        //     .iter()
        //     .flat_map(|x| (0..4).map(move |i| (x >> (2 * i)) & 3))
        //     .collect::<Vec<_>>();
        RSQVector256::new(&seq)
    }

    fn new_packed(_seq: &[usize]) -> Self {
        unimplemented!()
    }

    #[inline(always)]
    fn prefetch(&self, pos: usize) {
        self.prefetch_data(pos);
        self.prefetch_info(pos);
    }

    fn size(&self) -> usize {
        mem_dbg::MemSize::mem_size(self, Default::default())
    }

    #[inline(always)]
    fn count(&self, pos: usize) -> crate::Ranks {
        std::array::from_fn(|c| unsafe { self.rank_unchecked(c as u8, pos) as u32 })
    }

    #[inline(always)]
    fn count1(&self, pos: usize, c: u8) -> usize {
        unsafe { self.rank_unchecked(c, pos) as usize }
    }
}

impl RankerT for qwt::RSNarrow {
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

    fn size(&self) -> usize {
        mem_dbg::MemSize::mem_size(self, Default::default())
    }

    #[inline(always)]
    fn count(&self, _pos: usize) -> crate::Ranks {
        unimplemented!()
    }

    #[inline(always)]
    fn count1(&self, pos: usize, _c: u8) -> usize {
        unsafe { self.rank1_unchecked(pos) as usize }
    }
}

impl RankerT for qwt::RSWide {
    fn new_packed(seq: &[usize]) -> Self {
        Self::new(qwt::BitVector::from_slice(seq))
    }

    #[inline(always)]
    fn prefetch(&self, pos: usize) {
        RankBin::prefetch(self, pos);
    }

    fn size(&self) -> usize {
        mem_dbg::MemSize::mem_size(self, Default::default())
    }

    #[inline(always)]
    fn count(&self, _pos: usize) -> crate::Ranks {
        unimplemented!()
    }

    #[inline(always)]
    fn count1(&self, pos: usize, _c: u8) -> usize {
        unsafe { self.rank1_unchecked(pos) as usize }
    }
}
