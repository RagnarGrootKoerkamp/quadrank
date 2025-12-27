use mem_dbg::MemSize;
use sux::{
    bits::BitVec,
    prelude::{Rank9, RankSmall},
    traits::Rank,
};

use crate::ranker::RankerT;

// A macro that implements RankerT for the given RankSmall parameters.
macro_rules! impl_rank_small {
    ($T: ty) => {
        impl RankerT for $T {
            fn new_packed(seq: &[usize]) -> Self {
                let bits = unsafe {
                    BitVec::from_raw_parts(seq.to_vec(), seq.len() * usize::BITS as usize)
                };
                <$T>::new(bits)
            }

            fn prefetch(&self, pos: usize) {
                sux::traits::RankUnchecked::prefetch(&self, pos);
            }

            fn size(&self) -> usize {
                self.mem_size(Default::default())
            }

            #[inline(always)]
            fn count(&self, pos: usize) -> crate::Ranks {
                [self.rank(pos) as u32, 0, 0, 0]
            }

            #[inline(always)]
            fn count1(&self, pos: usize, _c: u8) -> usize {
                self.rank(pos) as usize
            }
        }
    };
}

impl_rank_small!(Rank9);
impl_rank_small!(RankSmall<2,9>);
impl_rank_small!(RankSmall<1,9>);
impl_rank_small!(RankSmall<1,10>);
impl_rank_small!(RankSmall<1,11>);
impl_rank_small!(RankSmall<3,13>);

pub type RankSmall1 = RankSmall<1, 9>;
pub type RankSmall2 = RankSmall<2, 9>;
pub type RankSmall3 = RankSmall<1, 10>;
pub type RankSmall4 = RankSmall<1, 11>;
pub type RankSmall5 = RankSmall<3, 13>;
