use super::SuperBlock;

/// Simply stores a `u64` rank.
#[derive(mem_dbg::MemSize)]
pub struct TrivialSB {
    rank: u64,
}

impl SuperBlock for TrivialSB {
    #[inline(always)]
    fn new(rank: u64) -> (Self, u64) {
        (Self { rank }, rank)
    }
    fn get(&self) -> u64 {
        self.rank
    }
}

/// Store the high 32 bits of `x>>8`.
#[derive(mem_dbg::MemSize)]
pub struct HalfSB {
    rank: u32,
}

impl SuperBlock for HalfSB {
    #[inline(always)]
    fn new(rank: u64) -> (Self, u64) {
        let rank = rank >> 8;
        assert!(
            rank <= u32::MAX as u64,
            "Rank too large for HalfSB. Use TrivialSB instead."
        );
        (
            Self {
                rank: (rank) as u32,
            },
            rank << 8,
        )
    }
    #[inline(always)]
    fn get(&self) -> u64 {
        (self.rank as u64) << 8
    }
}
