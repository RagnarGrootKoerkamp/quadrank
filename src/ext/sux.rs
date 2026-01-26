use mem_dbg::MemSize;
pub use sux::prelude::Rank9;
use sux::{bits::BitVec, prelude::RankSmall, traits::RankUnchecked};

use crate::binary::RankerT;

// A macro that implements RankerT for the given RankSmall parameters.
macro_rules! impl_rank_small {
    ($T: ty) => {
        impl RankerT for $T {
            #[inline(always)]
            fn new_packed(seq: &[usize]) -> Self {
                // RankUnchecked::rank_unchecked does not work for `len`, so we add some padding.
                let bits = unsafe {
                    BitVec::from_raw_parts(seq.to_vec(), seq.len() * usize::BITS as usize)
                };
                <$T>::new(bits)
            }

            const HAS_PREFETCH: bool = true;

            #[inline(always)]
            fn prefetch(&self, pos: usize) {
                sux::traits::RankUnchecked::prefetch(&self, pos);
            }

            #[inline(always)]
            fn size(&self) -> usize {
                self.mem_size(Default::default())
            }

            #[inline(always)]
            unsafe fn rank_unchecked(&self, pos: usize) -> u64 {
                unsafe { RankUnchecked::rank_unchecked(self, pos) as u64 }
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

pub type RankSmall0 = RankSmall<2, 9>;
pub type RankSmall1 = RankSmall<1, 9>;
pub type RankSmall2 = RankSmall<1, 10>;
pub type RankSmall3 = RankSmall<1, 11>;
pub type RankSmall4 = RankSmall<3, 13>;
