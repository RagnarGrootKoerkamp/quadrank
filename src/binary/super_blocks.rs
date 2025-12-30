use super::SuperBlock;

/// u64 rank for binary alphabet.
#[derive(mem_dbg::MemSize)]
pub struct TrivialSB {
    rank: u64,
}

impl SuperBlock for TrivialSB {
    const BB: usize = 1;
    const W: usize = 0;
    #[inline(always)]
    fn new(ranks: [u64; 1]) -> Self {
        Self { rank: ranks[0] }
    }
    fn get(&self, idx: usize) -> u64 {
        debug_assert_eq!(idx, 0);
        self.rank
    }
}
