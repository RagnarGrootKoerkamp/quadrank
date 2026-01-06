use crate::binary::RankerT;

impl RankerT for rsdict::RsDict {
    fn new_packed(seq: &[usize]) -> Self {
        Self::from_blocks(seq.iter().map(|x| *x as u64))
    }

    fn size(&self) -> usize {
        self.heap_size()
    }

    fn prefetch(&self, _pos: usize) {}

    unsafe fn rank_unchecked(&self, pos: usize) -> u64 {
        self.rank(pos as u64, true)
    }
}
