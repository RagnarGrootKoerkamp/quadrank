use qwt::{RSQVector256, RankQuad, SpaceUsage, WTSupport};

use crate::ranker::RankerT;

impl RankerT for RSQVector256 {
    fn new(seq: &[u8]) -> Self {
        let seq = seq.iter().map(|x| (x >> 1) & 3).collect::<Vec<_>>();
        // eprintln!("packed seq: {:?}", &seq);
        // let seq = seq
        //     .iter()
        //     .flat_map(|x| (0..4).map(move |i| (x >> (2 * i)) & 3))
        //     .collect::<Vec<_>>();
        RSQVector256::new(&seq)
    }

    #[inline(always)]
    fn prefetch(&self, pos: usize) {
        self.prefetch_data(pos);
        self.prefetch_info(pos);
    }

    fn size(&self) -> usize {
        self.space_usage_byte()
    }

    #[inline(always)]
    fn count(&self, pos: usize) -> crate::Ranks {
        std::array::from_fn(|c| unsafe { self.rank_unchecked(c as u8, pos) as u32 })
    }

    #[inline(always)]
    fn count1(&self, pos: usize, c: u8) -> u32 {
        unsafe { self.rank_unchecked(c, pos) as u32 }
    }
}
