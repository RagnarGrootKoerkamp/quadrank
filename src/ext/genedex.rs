use genedex::text_with_rank_support::{
    Block64, Block512, CondensedTextWithRankSupport, FlatTextWithRankSupport, TextWithRankSupport,
};

use crate::binary;
use crate::quad::{self, LongRanks};
use mem_dbg::MemSize;

macro_rules! impl_rank {
    ($T: ty) => {
        impl binary::RankerT for $T {
            #[inline(always)]
            fn new_packed(seq: &[usize]) -> Self {
                // convert bitvec to vec of 0 and 1 u8s
                let bits = seq
                    .iter()
                    .flat_map(|word| (0..usize::BITS).map(move |i| ((word >> i) & 1) as u8))
                    .collect::<Vec<u8>>();
                Self::construct(&bits, 2)
            }

            const HAS_PREFETCH: bool = true;

            #[inline(always)]
            fn prefetch(&self, pos: usize) {
                TextWithRankSupport::prefetch(self, 1, pos);
            }

            #[inline(always)]
            fn size(&self) -> usize {
                self.mem_size(Default::default())
            }

            #[inline(always)]
            unsafe fn rank_unchecked(&self, pos: usize) -> u64 {
                unsafe { TextWithRankSupport::rank_unchecked(self, 1, pos) as u64 }
            }
        }

        impl quad::RankerT for $T {
            #[inline(always)]
            fn new_packed(seq: &[usize]) -> Self {
                // convert bitvec to vec of 0 and 1 u8s
                let bits = seq
                    .iter()
                    .flat_map(|word| {
                        (0..usize::BITS)
                            .step_by(2)
                            .map(move |i| ((word >> i) & 3) as u8)
                    })
                    .collect::<Vec<u8>>();
                Self::construct(&bits, 4)
            }

            #[inline(always)]
            fn prefetch1(&self, pos: usize, c: u8) {
                TextWithRankSupport::prefetch(self, c, pos);
            }

            #[inline(always)]
            fn prefetch4(&self, pos: usize) {
                TextWithRankSupport::prefetch_all(self, pos);
            }

            #[inline(always)]
            fn size(&self) -> usize {
                self.mem_size(Default::default())
            }

            #[inline(always)]
            unsafe fn rank4_unchecked(&self, pos: usize) -> LongRanks {
                std::array::from_fn(|c| unsafe {
                    TextWithRankSupport::rank_unchecked(self, c as u8, pos) as u64
                })
            }

            #[inline(always)]
            unsafe fn rank1_unchecked(&self, pos: usize, c: u8) -> usize {
                unsafe { TextWithRankSupport::rank_unchecked(self, c, pos) as usize }
            }
        }
    };
}

pub type Flat64 = FlatTextWithRankSupport<i64, Block64>;
pub type Flat512 = FlatTextWithRankSupport<i64, Block512>;
pub type Condensed64 = CondensedTextWithRankSupport<i64, Block64>;
pub type Condensed512 = CondensedTextWithRankSupport<i64, Block512>;

impl_rank!(Flat64);
impl_rank!(Flat512);
impl_rank!(Condensed64);
impl_rank!(Condensed512);
