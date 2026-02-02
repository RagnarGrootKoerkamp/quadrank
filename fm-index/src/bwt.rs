#![allow(unused)]
use std::{path::Path, sync::atomic::AtomicUsize};

use rayon::{
    iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator},
    slice::ParallelSlice,
};

use crate::time;

#[derive(bincode::Encode, bincode::Decode, PartialEq)]
pub struct DiskBWT {
    /// A vector of u8 encoded values 0/1/2/3.
    pub bwt: Vec<u8>,
    pub sentinel: usize,
}

impl DiskBWT {
    pub fn pack(self) -> BWT {
        let DiskBWT { bwt, sentinel } = self;

        let mut packed = bwt
            .par_chunks(32)
            .map(|cc| {
                let mut x = 0usize;
                for (i, c) in cc.iter().enumerate() {
                    x |= (*c as usize) << (i * 2);
                }
                x
            })
            .collect::<Vec<usize>>();
        for _ in 0..128 {
            packed.push(0);
        }
        BWT {
            bwt,
            packed,
            sentinel,
        }
    }
}

#[derive(PartialEq)]
pub struct BWT {
    /// A vector of u8 encoded values 0/1/2/3.
    pub bwt: Vec<u8>,
    /// each u8 stores 4 values.
    pub packed: Vec<usize>,
    pub sentinel: usize,
}

impl BWT {
    pub fn to_disk(&self) -> DiskBWT {
        DiskBWT {
            bwt: self.bwt.clone(),
            sentinel: self.sentinel,
        }
    }
}

fn sa_to_bwt(text: &[u8], sa: impl IntoIterator<Item = usize>) -> BWT {
    let n = text.len();
    let mut sentinel = AtomicUsize::new(0);
    let mut bwt: Vec<_> = sa
        .into_iter()
        .enumerate()
        .flat_map(|(idx, i)| {
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

    DiskBWT { bwt, sentinel }.pack()
}

pub fn libsais(text: &[u8]) -> BWT {
    let sa = libsais::suffix_array::SuffixArrayConstruction::for_text(text)
        .in_owned_buffer64()
        .multi_threaded(libsais::ThreadCount::fixed(12))
        .run()
        .unwrap();

    sa_to_bwt(text, sa.into_vec().iter().map(|&x| x as usize))
}

/// BWT for context 100kbp.
pub fn simple_saca(text: &[u8]) -> BWT {
    let sa = simple_saca::suffix_array::SuffixArray::<5>::new_packed::<3000000>(text, 10, 6);

    sa_to_bwt(text, sa.idxs().iter().map(|x| x.get_usize()))
}

/// Needs external memory for human genome; 2x slower.
/// Temporarily padds the input text with a sentinel character.
pub fn caps_sa(text: &mut Vec<u8>, ext: bool) -> BWT {
    for x in text.iter_mut() {
        *x += 1;
    }
    text.push(0);
    let mut sa = Vec::with_capacity(text.len());
    caps_sa_rs::build_sa_u8(text, &mut sa, ext);

    text.pop();
    for x in text.iter_mut() {
        *x -= 1;
    }

    // drop the sentinel index.
    let n = text.len();
    sa = sa.into_iter().filter(|&i| i as usize != n).collect();
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

pub fn build_bwt_ascii(text: &[u8]) -> BWT {
    let mut text = text.to_vec();
    pack_text(&mut text);
    build_bwt_packed(&text)
}
#[allow(unused)]
pub fn build_bwt_packed(text: &[u8]) -> BWT {
    return time("libsais", || libsais(text));
    // // return time("caps-sa", || bwt::caps_sa(text, text.len() > 800_000_000));
    // if text.len() > 1000 {
    //     // time("simple-saca", || bwt::simple_saca(&text))
    //     // time("small-bwt", || bwt::small_bwt(&text))
    //     let b1 = time("caps-sa", || bwt::caps_sa(text, false));
    //     let b2 = time("manual", || bwt::manual(&text));
    //     if b1 != b2 {
    //         eprintln!("BWT mismatch!");
    //         eprintln!("Sentinels: b1 {}, b2 {}", b1.sentinel, b2.sentinel);
    //         eprintln!("Lens: b1 {}, b2 {}", b1.bwt.len(), b2.bwt.len());
    //         for (i, (&c1, &c2)) in b1.bwt.iter().zip(b2.bwt.iter()).enumerate() {
    //             if c1 != c2 {
    //                 eprintln!("Mismatch at pos {}: b1 {}, b2 {}", i, c1, c2);
    //                 break;
    //             }
    //         }
    //         assert_eq!(b1.bwt, b2.bwt, "BWT mismatch on text len {}", text.len());
    //     }
    //     b1
    // } else {
    //     // eprintln!("text: {text:?}");
    //     time("manual", || bwt::manual(&text))
    //     // eprintln!("text len {}, using caps-sa", text.len());
    //     // time("caps-sa", || bwt::caps_sa(&text, false))
    // }
}

// text must be values in 0..4
pub fn bwt(text: &[u8], output: &Path) {
    let bwt = build_bwt_packed(text);

    // write output to path.bwt:
    std::fs::write(
        output,
        bincode::encode_to_vec(&bwt.to_disk(), bincode::config::legacy()).unwrap(),
    )
    .unwrap();
}

pub fn pack_text(text: &mut Vec<u8>) {
    for x in text {
        *x = (*x >> 1) & 3;
    }
}
pub fn read_text(input: &Path) -> Vec<u8> {
    let mut text = vec![];
    let mut reader = needletail::parse_fastx_file(input).unwrap();
    while let Some(record) = reader.next() {
        let record = record.unwrap();
        text.extend_from_slice(&record.seq());
    }
    // Map to 0123.
    for x in &mut text {
        *x = (*x >> 1) & 3;
    }

    text
}
