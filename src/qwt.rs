use packed_seq::{PackedSeqVec, SeqVec};
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
    fn count1(&self, pos: usize, c: u8) -> u32 {
        unsafe { self.rank_unchecked(c, pos) as u32 }
    }
}

impl RankerT for qwt::RSNarrow {
    fn new(seq: &[u8]) -> Self {
        let packed_seq = PackedSeqVec::from_ascii(seq).into_raw();
        let iter = packed_seq
            .as_chunks()
            .0
            .iter()
            .map(|x| u64::from_le_bytes(*x));
        let mut bitvec = qwt::BitVectorMut::default();
        for x in iter {
            bitvec.append_bits(x, 64);
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
    fn count1(&self, pos: usize, _c: u8) -> u32 {
        unsafe { self.rank1_unchecked(pos) as u32 }
    }
}

impl RankerT for qwt::RSWide {
    fn new(seq: &[u8]) -> Self {
        let packed_seq = PackedSeqVec::from_ascii(seq).into_raw();
        let iter = packed_seq
            .as_chunks()
            .0
            .iter()
            .map(|x| u64::from_le_bytes(*x));
        let mut bitvec = qwt::BitVectorMut::default();
        for x in iter {
            bitvec.append_bits(x, 64);
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
    fn count1(&self, pos: usize, _c: u8) -> u32 {
        unsafe { self.rank1_unchecked(pos) as u32 }
    }
}
