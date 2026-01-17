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

/// Store the high 32 bits of `x>>11`.
#[derive(mem_dbg::MemSize)]
pub struct ShiftSB {
    rank: u32,
}

const SHIFT: usize = 11;

impl SuperBlock for ShiftSB {
    #[inline(always)]
    fn new(rank: u64) -> (Self, u64) {
        let rank = rank >> SHIFT;
        assert!(
            rank <= u32::MAX as u64,
            "Rank too large for HalfSB. Use TrivialSB instead."
        );
        (
            Self {
                rank: rank.try_into().unwrap(),
            },
            rank << SHIFT,
        )
    }
    #[inline(always)]
    fn get(&self) -> u64 {
        (self.rank as u64) << SHIFT
    }
}
