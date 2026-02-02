use mem_dbg::MemSize;

impl crate::FmIndex for awry::fm_index::FmIndex {
    fn new_with_prefix(text: &[u8], _bwt: &crate::bwt::BWT, prefix: usize) -> Self {
        let temp_dir = tempfile::tempdir().unwrap();
        let input_path = temp_dir.path().join("input.txt");
        let mut file = ">\n".to_string().into_bytes();
        for &b in text {
            file.push(b"ACTG"[b as usize]);
        }
        // file.extend_from_slice(text);

        std::fs::write(&input_path, file).unwrap();
        let build_args = awry::fm_index::FmBuildArgs {
            input_file_src: input_path.to_path_buf(),
            suffix_array_output_src: None,
            suffix_array_compression_ratio: Some(1024),
            lookup_table_kmer_len: Some(prefix as u8),
            alphabet: awry::alphabet::SymbolAlphabet::Nucleotide,
            max_query_len: Some(150), // FIXME
            remove_intermediate_suffix_array_file: true,
        };
        awry::fm_index::FmIndex::new(&build_args).unwrap()
    }

    fn size(&self) -> usize {
        self.mem_size(Default::default())
    }

    fn count(&self, text: &[u8]) -> usize {
        self.count_string(unsafe { str::from_utf8_unchecked(text) }) as usize
    }

    fn prep_read(read: &mut [u8]) {
        for b in read.iter_mut() {
            *b = b"ACTG"[(*b) as usize];
        }
    }

    const HAS_BATCH: bool = false;
    const HAS_PREFETCH: bool = false;
}
