#![allow(unused)]
use std::sync::atomic::AtomicUsize;

use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

#[derive(bincode::Encode, bincode::Decode)]
pub struct BWT {
    pub bwt: Vec<u8>,
    pub sentinel: usize,
}

fn sa_to_bwt(text: &[u8], sa: impl IntoIterator<Item = usize>) -> BWT {
    let n = text.len();
    let sa: Vec<_> = sa.into_iter().collect();
    let mut sentinel = AtomicUsize::new(0);
    let mut bwt: Vec<_> = sa
        .iter()
        .enumerate()
        .flat_map(|(idx, &i)| {
            if i == 0 {
                // +1 to correct for the insert we do below.
                sentinel.store(idx + 1, std::sync::atomic::Ordering::Relaxed);
                None
            } else {
                Some(text[i as usize - 1])
            }
        })
        .collect();
    bwt.insert(0, *text.last().unwrap());
    let sentinel = sentinel.into_inner();

    // for idx in 0..=text.len() {
    //     let i = if idx == 0 { n } else { sa[idx - 1] };
    //     let c = if idx < sentinel {
    //         bwt[idx]
    //     } else if idx == sentinel {
    //         99
    //     } else {
    //         bwt[idx - 1]
    //     };
    //     eprintln!("{idx:>2}: {c} | {i:>3} -> {:?}", &text[i..]);
    // }

    BWT { bwt, sentinel }
}

/// BWT for context 100kbp.
pub fn simple_saca(text: &[u8]) -> BWT {
    let sa = simple_saca::suffix_array::SuffixArray::<5>::new_packed::<3000000>(text, 10, 6);

    sa_to_bwt(text, sa.idxs().iter().map(|x| x.get_usize()))
}

/// Needs external memory for human genome; 2x slower.
pub fn caps_sa(text: &[u8], ext: bool) -> BWT {
    let mut sa = Vec::with_capacity(text.len());
    caps_sa_rs::build_sa_u8(text, &mut sa, ext);

    sa_to_bwt(text, sa.iter().map(|&i| i as usize))
}

/// Text must have a \0 or $ at the end.
/// Much slower, and only single-threaded.
pub fn small_bwt(text: &[u8]) -> Vec<u8> {
    let mut bwt = Vec::with_capacity(text.len());
    small_bwt::verify_terminator(text).unwrap();
    small_bwt::BwtBuilder::new(text)
        .unwrap()
        .build(&mut bwt)
        .unwrap();
    bwt
}

pub fn manual(text: &[u8]) -> BWT {
    let mut sa: Vec<usize> = (0..text.len()).collect();
    sa.sort_unstable_by_key(|&i| &text[i..]);

    sa_to_bwt(text, sa)
}
