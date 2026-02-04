use crate::binary::BiRanker;

pub use rsdict::RsDict;

impl BiRanker for rsdict::RsDict {
    fn new_packed(seq: &[u64]) -> Self {
        Self::from_blocks(seq.iter().map(|x| *x))
    }

    fn size(&self) -> usize {
        self.heap_size()
    }

    unsafe fn rank_unchecked(&self, pos: usize) -> u64 {
        self.rank(pos as u64, true)
    }
}
