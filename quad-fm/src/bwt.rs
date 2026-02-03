use std::{path::Path, sync::atomic::AtomicUsize};

use rayon::{iter::ParallelIterator, slice::ParallelSlice};

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
    let sentinel = AtomicUsize::new(0);
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

pub fn build_bwt_packed(text: &[u8]) -> BWT {
    return libsais(text);
}

// text must be values in 0..4
pub fn write_bwt(text: &[u8], output: &Path) {
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
