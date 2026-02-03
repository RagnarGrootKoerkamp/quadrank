use crate::binary::RankerT;
use succinct::{BitRankSupport, SpaceUsage};

pub use succinct::{JacobsonRank, Rank9};

impl RankerT for succinct::Rank9<Vec<u64>> {
    fn new_packed(seq: &[u64]) -> Self {
        Self::new(seq.to_vec())
    }

    fn size(&self) -> usize {
        self.total_bytes()
    }

    unsafe fn rank_unchecked(&self, pos: usize) -> u64 {
        self.rank1(pos as u64 - 1)
    }
}

impl RankerT for succinct::JacobsonRank<Vec<u64>> {
    fn new_packed(seq: &[u64]) -> Self {
        Self::new(seq.to_vec())
    }

    fn size(&self) -> usize {
        self.total_bytes()
    }

    unsafe fn rank_unchecked(&self, pos: usize) -> u64 {
        self.rank1(pos as u64 - 1)
    }
}
