use crate::binary::RankerT;

impl RankerT for vers_vecs::RsVec {
    fn new_packed(seq: &[usize]) -> Self {
        let bv = vers_vecs::BitVec::from_limbs_iter(seq.iter().map(|x| !*x as u64));
        Self::from_bit_vec(bv)
    }

    fn size(&self) -> usize {
        self.heap_size()
    }

    fn prefetch(&self, _pos: usize) {}

    unsafe fn rank_unchecked(&self, pos: usize) -> u64 {
        self.rank0(pos) as u64
    }
}
