use crate::binary::RankerT;
use dyn_size_of::GetSize;

pub use bitm::RankSelect101111;
pub use bitm::RankSimple;

impl RankerT for bitm::RankSimple {
    fn new_packed(seq: &[u64]) -> Self {
        Self::build(seq.to_vec().into()).0
    }

    fn size(&self) -> usize {
        self.size_bytes()
    }

    unsafe fn rank_unchecked(&self, pos: usize) -> u64 {
        unsafe { bitm::Rank::rank_unchecked(self, pos) as u64 }
    }
}

impl RankerT for bitm::RankSelect101111 {
    fn new_packed(seq: &[u64]) -> Self {
        Self::build(seq.to_vec().into()).0
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
