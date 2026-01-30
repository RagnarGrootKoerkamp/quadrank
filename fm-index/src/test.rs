use crate::{build_bwt_ascii, build_bwt_packed, fm};

#[test]
fn broken() {
    // let text = b"AGCCTTAGCTGCGACAGAATGGATCAGAAAGCTTGAAAACTTAGAGCAAAAAATTGACTATTTTGACGAGTGTCTTCTTCCAGGCATTTTCACCATCGACGCGGATCCTCCAGACGAGTTGTTTCTTGATGAACTG";
    // let query = b"TGCGACAGAATGGATCAGAAAGCTTGAAAACTTAGAGCAAAAAATTGACTATTTTGACGAGTGTCTTCTTCC";
    let text = b"AGCCTTAGCTGCGACAGAATGGATCAGAAAGCTTGAAAACTTAGAGCAAAAAATTGACTATTTTGACGAGTGTCTTCTTCCAGGCATTTTCACCATCGACGCGGATCCTCCAGACGAGTTGTTTCTTGATGAACTG";
    let query = b"GGA";
    let bwt = build_bwt_ascii(text.to_vec());
    let packed = query.iter().map(|&x| (x >> 1) & 3).collect::<Vec<_>>();
    let fm = <fm::FM>::new(&bwt);
    let (steps, count) = fm.count(&packed);
    eprintln!("steps: {steps}, matches: {count}");
    assert!(count > 0);
}

#[test]
fn broken2() {
    // let text = b"AGCCTTAGCTGCGACAGAATGGATCAGAAAGCTTGAAAACTTAGAGCAAAAAATTGACTATTTTGACGAGTGTCTTCTTCCAGGCATTTTCACCATCGACGCGGATCCTCCAGACGAGTTGTTTCTTGATGAACTG";
    // let query = b"TGCGACAGAATGGATCAGAAAGCTTGAAAACTTAGAGCAAAAAATTGACTATTTTGACGAGTGTCTTCTTCC";
    let text = b"AGCCTTAGCTGCGACAGAATGGATCAGAAAGCTTGAAAACTTAGAGCAAAAAATTGACTATTTTGACGAGTGTCTTCTTCCAGGCATTTTCACCATCGACGCGGATCCTCCAGACGAGTTGTTTCTTGATGAACTG";
    let query = b"GGATCA";
    let mut text = text.iter().map(|&x| ((x >> 1) & 3) + 1).collect::<Vec<_>>();
    text.push(0);
    let query = query
        .iter()
        .map(|&x| ((x >> 1) & 3) + 1)
        .collect::<Vec<_>>();

    let fm = fm_index::FMIndex::new(&fm_index::Text::with_max_character(text, 4)).unwrap();
    let count = fm.search(&query).count();
    eprintln!("matches: {count}");
    assert!(count > 0);
    panic!()
}

#[test]
fn fuzz_fm() {
    for _ in 0..1000 {
        let len = rand::random_range(1000..3000);
        eprintln!("Building for len {len}");
        let mut text = (0..len)
            .map(|_| rand::random_range(0..4))
            .collect::<Vec<_>>();

        let bwt = build_bwt_packed(&mut text);
        let mfm = <fm::FM>::new(&bwt);

        let gfm = genedex::FmIndexConfig::<i32>::new()
            .suffix_array_sampling_rate(16)
            .construct_index(&[&text], genedex::alphabet::u8_until(3));

        eprintln!("Querying");
        for _ in 0..10000 {
            let start = rand::random_range(0..len);
            let end = rand::random_range(start..=len);
            let q = &text[start..end];

            let m_cnt = mfm.count(q).1;
            let g_cnt = gfm.count(q);
            eprintln!("+ m_cnt: {}, g_cnt: {}", m_cnt, g_cnt);
            assert_eq!(m_cnt, g_cnt, "text len {}, query {:?}", len, q);
        }

        for _ in 0..10000 {
            let start = rand::random_range(0..len - 1);
            let end = rand::random_range(start + 1..=len);
            let mut q = text[start..end].to_vec();
            let ql = q.len();
            q[rand::random_range(0..ql)] = rand::random_range(0..4);

            let m_cnt = mfm.count(&q).1;
            let g_cnt = gfm.count(&q);
            eprintln!("- m_cnt: {}, g_cnt: {}", m_cnt, g_cnt);
            assert_eq!(m_cnt, g_cnt, "text len {}, query {:?}", len, q);
        }
    }
}
