use genedex::text_with_rank_support::TextWithRankSupport;
use mem_dbg::MemSize;

impl<T: TextWithRankSupport<i64> + Sync> crate::FmIndex for genedex::FmIndex<i64, T> {
    fn new_with_prefix(text: &[u8], _bwt: &crate::bwt::BWT, prefix: usize) -> Self {
        let alphabet = genedex::Alphabet::from_io_symbols([0, 1, 2, 3], 0);
        genedex::FmIndexConfig::<i64, T>::new()
            .suffix_array_sampling_rate(1024)
            .lookup_table_depth(prefix)
            .construct_index(&[&text], alphabet)
    }
    fn size(&self) -> usize {
        self.mem_size(Default::default())
    }
    fn count(&self, text: &[u8]) -> usize {
        self.count(text)
    }
    const HAS_BATCH: bool = true;
    const HAS_PREFETCH: bool = false;
    fn count_batch<const B: usize, const PREFETCH: bool>(
        &self,
        texts: &[Vec<u8>; B],
    ) -> [usize; B] {
        let mut out = [0; B];
        for (i, cnt) in self.count_many(texts).take(B).enumerate() {
            out[i] = cnt;
        }
        out
    }
}
