use crate::binary::RankerT;
use dyn_size_of::GetSize;

impl RankerT for bitm::RankSimple {
    fn new_packed(seq: &[usize]) -> Self {
        let seq_u64 = unsafe { seq.align_to::<u64>().1 }
            .to_vec()
            .into_boxed_slice();
        Self::build(seq_u64).0
    }

    fn size(&self) -> usize {
        self.size_bytes()
    }

    unsafe fn rank_unchecked(&self, pos: usize) -> u64 {
        unsafe { bitm::Rank::rank_unchecked(self, pos) as u64 }
    }
}

impl RankerT for bitm::RankSelect101111 {
    fn new_packed(seq: &[usize]) -> Self {
        let seq_u64 = unsafe { seq.align_to::<u64>().1 }
            .to_vec()
            .into_boxed_slice();
        Self::build(seq_u64).0
    }

    fn size(&self) -> usize {
        self.size_bytes()
    }

    const HAS_PREFETCH: bool = true;

    fn prefetch(&self, pos: usize) {
        bitm::Rank::prefetch(self, pos);
    }

    unsafe fn rank_unchecked(&self, pos: usize) -> u64 {
        unsafe { bitm::Rank::rank_unchecked(self, pos) as u64 }
    }
}
