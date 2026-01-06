use crate::binary::RankerT;

impl RankerT for bio::data_structures::rank_select::RankSelect {
    fn new_packed(seq: &[usize]) -> Self {
        let seq_u8 = unsafe { seq.align_to::<u8>().1 };
        let bitvec = bv::BitVec::<u8>::from_bits(seq_u8);
        let k = 512 / 32;
        Self::new(bitvec, k)
    }

    fn size(&self) -> usize {
        self.bits().len() as usize / 8 + std::mem::size_of_val(self.superblocks_1.as_slice())
    }

    fn prefetch(&self, _pos: usize) {
        // TODO
    }

    unsafe fn rank_unchecked(&self, pos: usize) -> u64 {
        self.rank_1(pos as u64 - 1).unwrap()
    }
}
