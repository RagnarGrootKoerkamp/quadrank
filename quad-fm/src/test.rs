use quadrank::quad::QuadRank16;

use crate::{FmIndex, QuadFm, bwt::build_bwt_packed};

#[test]
fn fuzz_fm() {
    for _ in 0..100 {
        let len = rand::random_range(1000..3000);
        eprintln!("Building for len {len}");
        let mut text = (0..len)
            .map(|_| rand::random_range(0..4))
            .collect::<Vec<_>>();

        let bwt = build_bwt_packed(&mut text);
        let quad_fm = <QuadFm<QuadRank16>>::new(&text, &bwt);

        let genedex = genedex::FmIndexConfig::<i32>::new()
            .suffix_array_sampling_rate(16)
            .construct_index(&[&text], genedex::alphabet::u8_until(3));

        eprintln!("Querying");
        for _ in 0..10000 {
            let start = rand::random_range(0..len);
            let end = rand::random_range(start..=len);
            let q = &text[start..end];

            let m_cnt = quad_fm.count(q);
            let g_cnt = genedex.count(q);
            assert_eq!(m_cnt, g_cnt, "text len {}, query {:?}", len, q);
        }

        for _ in 0..10000 {
            let start = rand::random_range(0..len - 1);
            let end = rand::random_range(start + 1..=len);
            let mut q = text[start..end].to_vec();
            let ql = q.len();
            q[rand::random_range(0..ql)] = rand::random_range(0..4);

            let m_cnt = quad_fm.count(&q);
            let g_cnt = genedex.count(&q);
            assert_eq!(m_cnt, g_cnt, "text len {}, query {:?}", len, q);
        }
    }
}
