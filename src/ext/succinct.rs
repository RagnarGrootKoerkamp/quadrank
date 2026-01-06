use crate::binary::RankerT;
use succinct::{BitRankSupport, SpaceUsage};

impl RankerT for succinct::Rank9<Vec<u64>> {
    fn new_packed(seq: &[usize]) -> Self {
        let seq = unsafe { seq.align_to::<u64>().1 }.to_vec();
        Self::new(seq)
    }

    fn size(&self) -> usize {
        self.total_bytes()
    }

    fn prefetch(&self, _pos: usize) {}

    unsafe fn rank_unchecked(&self, pos: usize) -> u64 {
        self.rank1(pos as u64 - 1)
    }
}

impl RankerT for succinct::JacobsonRank<Vec<u64>> {
    fn new_packed(seq: &[usize]) -> Self {
        let seq = unsafe { seq.align_to::<u64>().1 }.to_vec();
        Self::new(seq)
    }

    fn size(&self) -> usize {
        self.total_bytes()
    }

    fn prefetch(&self, _pos: usize) {}

    unsafe fn rank_unchecked(&self, pos: usize) -> u64 {
        self.rank1(pos as u64 - 1)
    }
}
