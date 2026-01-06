use sucds::{
    Serializable,
    bit_vectors::{Rank, Rank9Sel},
};

use crate::binary::RankerT;

impl RankerT for Rank9Sel {
    fn new_packed(seq: &[usize]) -> Self {
        Self::from_bits(
            seq.iter()
                .flat_map(|x| (0..64).map(move |i| (x >> i) & 1 == 1)),
        )
    }

    fn size(&self) -> usize {
        self.size_in_bytes()
    }

    unsafe fn rank_unchecked(&self, pos: usize) -> u64 {
        self.rank1(pos).unwrap() as u64
    }
}
