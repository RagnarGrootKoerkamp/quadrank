#[test]
fn fuzz_caps_sa() {
    for _ in 0..1000 {
        let len = rand::random_range(1000..3000);
        eprintln!("Building for len {len}");
        let mut text = (0..len)
            .map(|_| rand::random_range(1..=4))
            .collect::<Vec<_>>();
        text.push(0);
        test_eq(&mut text);
    }
}

fn test_eq(text: &mut Vec<u8>) {
    let mut sa1 = Vec::with_capacity(text.len());
    // caps_sa_rs::build_sa_u8(text, &mut sa1, false);

    {
        for x in text.iter_mut() {
            *x += 1;
        }
        text.push(0);
        let mut sa = Vec::with_capacity(text.len());
        caps_sa_rs::build_sa_u8(text, &mut sa, false);

        text.pop();
        for x in text.iter_mut() {
            *x -= 1;
        }

        let n = text.len();
        sa1 = sa.into_iter().filter(|&i| i != n as u32).collect();
    }

    let mut sa2: Vec<u32> = (0..text.len()).map(|x| x as u32).collect();
    sa2.sort_unstable_by_key(|&i| &text[i as usize..]);

    if sa1 != sa2 {
        assert_eq!(sa1.len(), sa2.len());
        for (i, (x, y)) in std::iter::zip(&sa1, &sa2).enumerate() {
            if x != y {
                eprintln!("mismatch at post {i}: {x} vs {y}");
            }
        }
        panic!();
    }
}
