use genedex::text_with_rank_support::{
    Block64, Block512, CondensedTextWithRankSupport, FlatTextWithRankSupport, TextWithRankSupport,
};

use crate::ranker::RankerT;
use mem_dbg::MemSize;

macro_rules! impl_rank {
    ($T: ty) => {
        impl RankerT for $T {
            fn new_packed(seq: &[usize]) -> Self {
                // convert bitvec to vec of 0 and 1 u8s
                let bits = seq
                    .iter()
                    .flat_map(|word| (0..usize::BITS).map(move |i| ((word >> i) & 1) as u8))
                    .collect::<Vec<u8>>();
                Self::construct(&bits, 2)
            }

            fn prefetch(&self, pos: usize) {
                TextWithRankSupport::prefetch(self, pos);
            }

            fn size(&self) -> usize {
                self.mem_size(Default::default())
            }

            #[inline(always)]
            fn count(&self, pos: usize) -> crate::Ranks {
                [unsafe { self.rank_unchecked(1, pos) } as u32, 0, 0, 0]
            }

            #[inline(always)]
            fn count1(&self, pos: usize, _c: u8) -> u32 {
                unsafe { self.rank_unchecked(1, pos) as u32 }
            }
        }
    };
}

pub type Flat64 = FlatTextWithRankSupport<u32, Block64>;
pub type Flat512 = FlatTextWithRankSupport<u32, Block512>;
pub type Condensed64 = CondensedTextWithRankSupport<u32, Block64>;
pub type Condensed512 = CondensedTextWithRankSupport<u32, Block512>;

impl_rank!(Flat64);
impl_rank!(Flat512);
impl_rank!(Condensed64);
impl_rank!(Condensed512);
