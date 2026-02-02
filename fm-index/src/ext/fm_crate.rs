use fm_index::SearchIndex;

impl crate::FmIndex for fm_index::FMIndex<u8> {
    fn new_with_prefix(text: &[u8], _bwt: &crate::bwt::BWT, _prefix: usize) -> Self {
        // shift all up to 1..=4 and push 0
        let mut text: Vec<u8> = text.iter().map(|x| x + 1).collect();
        text.push(0);
        fm_index::FMIndex::<u8>::new(&fm_index::Text::with_max_character(&text, 4)).unwrap()
    }

    fn prep_read(read: &mut [u8]) {
        for c in read.iter_mut() {
            *c += 1;
        }
    }

    fn size(&self) -> usize {
        self.heap_size()
    }

    fn count(&self, text: &[u8]) -> usize {
        self.search(text).count()
    }

    const HAS_BATCH: bool = false;
    const HAS_PREFETCH: bool = false;
}
