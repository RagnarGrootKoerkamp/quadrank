#![allow(incomplete_features, unused)]
#![feature(
    generic_const_exprs,
    portable_simd,
    coroutines,
    coroutine_trait,
    stmt_expr_attributes
)]
use std::{
    arch::x86_64::{_mm256_shuffle_pd, _mm256_unpackhi_epi64, _mm256_unpacklo_epi64},
    hint::black_box,
    mem::transmute,
    ops::Coroutine,
    simd::{u8x32, u16x16, u32x8, u64x4},
};

use packed_seq::{PackedSeqVec, SeqVec};
use smol::future::yield_now;

pub type Ranks = [u32; 4];

#[derive(mem_dbg::MemSize)]
pub struct DnaRank<const STRIDE: usize> {
    n: usize,
    seq: Vec<u8>,
    counts: Vec<Ranks>,
}

impl<const STRIDE: usize> DnaRank<STRIDE> {
    pub fn new(seq: &[u8]) -> Self
    where
        [(); STRIDE / 4]:,
    {
        assert!(STRIDE % 32 == 0, "STRIDE must be a multiple of 32");
        let mut counts = Vec::with_capacity(seq.len().div_ceil(STRIDE));
        let mut ranks = [0; 4];
        counts.push(ranks);

        let n = seq.len();
        let mut seq = PackedSeqVec::from_ascii(seq).into_raw();
        seq.extend_from_slice(&vec![0; 128]); // padding

        for chunk in seq.as_chunks::<{ STRIDE / 4 }>().0 {
            for chunk in chunk.as_chunks::<8>().0 {
                for c in 0..4 {
                    ranks[c as usize] += count_u8x8(chunk, c);
                }
            }
            counts.push(ranks);
        }

        assert!(seq.len() >= n / 4 + 32);
        DnaRank { n, seq, counts }
    }

    /// Loop over packed characters.
    #[inline(always)]
    pub fn ranks_naive(&self, pos: usize) -> Ranks {
        let chunk_idx = pos / STRIDE;
        let byte_idx = chunk_idx * (STRIDE / 4);
        let mut ranks = self.counts[chunk_idx];

        for &packed in &self.seq[byte_idx..pos / 4] {
            for i in 0..4 {
                let c = (packed >> (i * 2)) & 0b11;
                ranks[c as usize] += 1;
            }
        }
        let packed = self.seq[pos / 4];
        for i in 0..pos % 4 {
            let c = (packed >> (i * 2)) & 0b11;
            ranks[c as usize] += 1;
        }

        ranks
    }

    /// Count a u64 at a time.
    #[inline(always)]
    pub fn ranks_u64(&self, pos: usize) -> Ranks {
        let chunk_idx = pos / STRIDE;
        let byte_idx = chunk_idx * (STRIDE / 4);
        let mut ranks = self.counts[chunk_idx];

        for idx in (byte_idx..pos.div_ceil(4)).step_by(8) {
            let chunk = u64::from_le_bytes(self.seq[idx..idx + 8].try_into().unwrap());
            let low_bits = (pos - idx * 4).min(32) * 2;
            let mask = if low_bits == 64 {
                u64::MAX
            } else {
                (1u64 << low_bits) - 1
            };
            let chunk = chunk & mask;
            for c in 0..4 {
                ranks[c as usize] += count_u64(chunk, c);
            }
        }
        let extra_counted = 32usize.wrapping_sub(pos) % 32;
        ranks[0] -= extra_counted as u32;

        ranks
    }

    // Prefetch the ranks, and only read them after scanning.
    #[inline(always)]
    pub fn ranks_u64_prefetch(&self, pos: usize) -> Ranks {
        let chunk_idx = pos / STRIDE;
        let byte_idx = chunk_idx * (STRIDE / 4);
        prefetch_index(&self.counts, chunk_idx);
        let mut ranks = [0; 4];

        for idx in (byte_idx..pos.div_ceil(4)).step_by(8) {
            let chunk = u64::from_le_bytes(self.seq[idx..idx + 8].try_into().unwrap());
            let low_bits = (pos - idx * 4).min(32) * 2;
            let mask = if low_bits == 64 {
                u64::MAX
            } else {
                (1u64 << low_bits) - 1
            };
            let chunk = chunk & mask;
            for c in 0..4 {
                ranks[c as usize] += count_u64(chunk, c);
            }
        }
        for c in 0..4 {
            ranks[c] += self.counts[chunk_idx][c];
        }
        let extra_counted = 32usize.wrapping_sub(pos) % 32;
        ranks[0] -= extra_counted as u32;

        ranks
    }

    // Prefetch the ranks, and only read them after scanning.
    #[inline(always)]
    pub fn ranks_u64_3(&self, pos: usize) -> Ranks {
        let chunk_idx = pos / STRIDE;
        let byte_idx = chunk_idx * (STRIDE / 4);
        prefetch_index(&self.counts, chunk_idx);
        let mut ranks = [0; 4];

        for idx in (byte_idx..pos.div_ceil(4)).step_by(8) {
            let chunk = u64::from_le_bytes(self.seq[idx..idx + 8].try_into().unwrap());
            let low_bits = (pos - idx * 4).min(32) * 2;
            let mask = if low_bits == 64 {
                u64::MAX
            } else {
                (1u64 << low_bits) - 1
            };
            let chunk = chunk & mask;
            for c in 1..4 {
                ranks[c as usize] += count_u64(chunk, c);
            }
        }
        for c in 1..4 {
            ranks[c] += self.counts[chunk_idx][c];
        }
        ranks[0] = pos as u32 - ranks[1] - ranks[2] - ranks[3];

        ranks
    }

    // Prefetch the ranks, and only read them after scanning.
    #[inline(always)]
    pub fn ranks_u64_prefetch_all(&self, pos: usize) -> Ranks {
        let chunk_idx = pos / STRIDE;
        let byte_idx = chunk_idx * (STRIDE / 4);
        prefetch_index(&self.counts, chunk_idx);
        let mut ranks = [0; 4];

        for idx in (byte_idx..(chunk_idx + 1) * (STRIDE / 4)).step_by(8) {
            let chunk = u64::from_le_bytes(self.seq[idx..idx + 8].try_into().unwrap());
            let low_bits = pos.saturating_sub(idx * 4).min(32) * 2;
            let mask = if low_bits == 64 {
                u64::MAX
            } else {
                (1u64 << low_bits) - 1
            };
            let chunk = chunk & mask;
            for c in 0..4 {
                ranks[c as usize] += count_u64(chunk, c);
            }
        }
        for c in 0..4 {
            ranks[c] += self.counts[chunk_idx][c];
        }
        let extra_counted = STRIDE - pos % STRIDE;
        ranks[0] -= extra_counted as u32;

        ranks
    }

    // Count u128 at a time.
    #[inline(always)]
    pub fn ranks_u128(&self, pos: usize) -> Ranks {
        let chunk_idx = pos / STRIDE;
        let byte_idx = chunk_idx * (STRIDE / 4);
        prefetch_index(&self.counts, chunk_idx);
        let mut ranks = [0; 4];

        for idx in (byte_idx..pos.div_ceil(4)).step_by(16) {
            let chunk = u128::from_le_bytes(self.seq[idx..idx + 16].try_into().unwrap());
            let low_bits = (pos - idx * 4).min(64) * 2;
            let mask = if low_bits == 128 {
                u128::MAX
            } else {
                (1u128 << low_bits) - 1
            };
            let chunk = chunk & mask;
            for c in 0..4 {
                ranks[c as usize] += count_u128(chunk, c);
            }
        }
        for c in 0..4 {
            ranks[c] += self.counts[chunk_idx][c];
        }
        let extra_counted = (64usize.wrapping_sub(pos)) % 64;
        ranks[0] -= extra_counted as u32;

        ranks
    }

    #[inline(always)]
    // Count u128 at a time.
    pub fn ranks_u128_3(&self, pos: usize) -> Ranks {
        let chunk_idx = pos / STRIDE;
        let byte_idx = chunk_idx * (STRIDE / 4);
        prefetch_index(&self.counts, chunk_idx);
        let mut ranks = [0; 4];

        for idx in (byte_idx..pos.div_ceil(4)).step_by(16) {
            let chunk = u128::from_le_bytes(self.seq[idx..idx + 16].try_into().unwrap());
            let low_bits = (pos - idx * 4).min(64) * 2;
            let mask = if low_bits == 128 {
                u128::MAX
            } else {
                (1u128 << low_bits) - 1
            };
            let chunk = chunk & mask;
            for c in 1..4 {
                ranks[c as usize] += count_u128(chunk, c);
            }
        }
        for c in 1..4 {
            ranks[c] += self.counts[chunk_idx][c];
        }
        ranks[0] = pos as u32 - ranks[1] - ranks[2] - ranks[3];

        ranks
    }
}

#[inline(always)]
fn count_u8x8(word: &[u8; 8], c: u8) -> u32 {
    count_u64(u64::from_le_bytes(*word), c)
}

#[inline(always)]
fn count_u8(word: u8, c: u8) -> u32 {
    // c = 00, 01, 10, 11 = cc
    // scatter = |01|01|01|...
    let scatter = 0x55u8;
    let mask = c as u8 * scatter;
    // mask = |cc|cc|cc|...

    // should be |00|00|00|... to match c.
    let tmp = word ^ mask;

    // |00| when c
    // |01| otherwise
    let union = (tmp | (tmp >> 1)) & scatter;
    4 - union.count_ones()
}

#[inline(always)]
fn count_u64(word: u64, c: u8) -> u32 {
    // c = 00, 01, 10, 11 = cc
    // scatter = |01|01|01|...
    let scatter = 0x5555555555555555u64;
    let mask = c as u64 * scatter;
    // mask = |cc|cc|cc|...

    // should be |00|00|00|... to match c.
    let tmp = word ^ mask;

    // |00| when c
    // |01| otherwise
    let union = (tmp | (tmp >> 1)) & scatter;
    32 - union.count_ones()
}

#[inline(always)]
fn count_u128(word: u128, c: u8) -> u32 {
    // c = 00, 01, 10, 11 = cc
    // scatter = |01|01|01|...
    let scatter = 0x55555555555555555555555555555555u128;
    let mask = c as u128 * scatter;
    // mask = |cc|cc|cc|...

    // should be |00|00|00|... to match c.
    let tmp = word ^ mask;

    // |00| when c
    // |01| otherwise
    let union = (tmp | (tmp >> 1)) & scatter;
    64 - union.count_ones()
}

/// Prefetch the given cacheline into L1 cache.
pub(crate) fn prefetch_index<T>(s: &[T], index: usize) {
    let ptr = s.as_ptr().wrapping_add(index) as *const u64;
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
    }
    #[cfg(target_arch = "x86")]
    unsafe {
        std::arch::x86::_mm_prefetch(ptr as *const i8, std::arch::x86::_MM_HINT_T0);
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        // TODO: Put this behind a feature flag.
        // std::arch::aarch64::_prefetch(ptr as *const i8, std::arch::aarch64::_PREFETCH_LOCALITY3);
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86", target_arch = "aarch64")))]
    {
        // Do nothing.
    }
}

/// For each 128bp, store:
/// - 4 u64 counts, for 256bits total
/// - 256 bits of packed sequence.
/// In total, exactly covers a 512bit cache line.
///
/// Based on BWA: https://github.com/lh3/bwa/blob/master/bwtindex.c#L150
#[repr(C)]
#[repr(align(64))]
#[derive(mem_dbg::MemSize)]
pub struct BwaBlock {
    // 4*64 = 256 bit counts
    ranks: [u64; 4],
    // 4*64 = 32*8 = 256 bit packed sequence
    seq: [u8; 32],
}

/// Store 4 u32 counts every 128bp = 256bits.
#[derive(mem_dbg::MemSize)]
pub struct BwaRank {
    n: usize,
    blocks: Vec<BwaBlock>,
    /// For each byte, a u32 consisting of 4 u8's containing the count of ACTG in byte.
    counts: [u32; 256],
}

impl BwaRank {
    pub fn new(seq: &[u8]) -> Self {
        let n = seq.len();
        let mut blocks = Vec::with_capacity(seq.len().div_ceil(64));
        let mut ranks = [0; 4];

        let mut seq = PackedSeqVec::from_ascii(seq).into_raw();
        seq.resize(seq.capacity(), 0);

        for chunk in seq.as_chunks::<32>().0 {
            blocks.push(BwaBlock {
                ranks: ranks.map(|x| x as u64),
                seq: *chunk,
            });
            for chunk in chunk.as_chunks::<8>().0 {
                for c in 0..4 {
                    ranks[c as usize] += count_u8x8(chunk, c);
                }
            }
        }

        BwaRank {
            n,
            blocks,
            counts: init_counts(),
        }
    }

    #[inline(always)]
    pub fn ranks_bytecount(&self, pos: usize) -> Ranks {
        let chunk_idx = pos / 128;
        prefetch_index(&self.blocks, chunk_idx);
        let chunk = &self.blocks[chunk_idx];
        let mut ranks = [0; 4];
        for c in 0..4 {
            ranks[c] += chunk.ranks[c] as u32;
        }

        let pos = pos % 128;

        let mut counts: u32 = 0;

        for idx in 0..pos.div_ceil(4) {
            let byte = chunk.seq[idx];
            let low_bits = (pos - idx * 4).min(4) * 2;
            let mask = (1u64 << low_bits) - 1;
            let byte = byte & mask as u8;
            counts += self.counts[byte as usize];
        }
        for c in 0..4 {
            ranks[c] += (counts >> (8 * c)) as u8 as u32;
        }
        let extra_counted = (4usize.wrapping_sub(pos)) % 4;
        ranks[0] -= extra_counted as u32;
        ranks
    }

    #[inline(always)]
    pub fn ranks_bytecount_4(&self, pos: usize) -> Ranks {
        let chunk_idx = pos / 128;
        prefetch_index(&self.blocks, chunk_idx);
        let chunk = &self.blocks[chunk_idx];
        let mut ranks = [0; 4];
        for c in 0..4 {
            ranks[c] += chunk.ranks[c] as u32;
        }

        let pos = pos % 128;

        let mut counts: u32 = 0;

        for idx in (0..pos.div_ceil(4)).step_by(4) {
            let chunk = u32::from_le_bytes(chunk.seq[idx..idx + 4].try_into().unwrap());
            let low_bits = (pos - idx * 4).min(16) * 2;
            let mask = (1u64 << low_bits) - 1;
            let chunk = chunk & mask as u32;
            counts += self.counts[(chunk >> 0) as u8 as usize];
            counts += self.counts[(chunk >> 8) as u8 as usize];
            counts += self.counts[(chunk >> 16) as u8 as usize];
            counts += self.counts[(chunk >> 24) as u8 as usize];
        }
        for c in 0..4 {
            ranks[c] += (counts >> (8 * c)) as u8 as u32;
        }
        let extra_counted = (16usize.wrapping_sub(pos)) % 16;
        ranks[0] -= extra_counted as u32;
        ranks
    }

    #[inline(always)]
    pub fn ranks_bytecount_8(&self, pos: usize) -> Ranks {
        let chunk_idx = pos / 128;
        prefetch_index(&self.blocks, chunk_idx);
        let chunk = &self.blocks[chunk_idx];
        let mut ranks = [0; 4];
        for c in 0..4 {
            ranks[c] += chunk.ranks[c] as u32;
        }

        let pos = pos % 128;

        let mut counts: u32 = 0;

        for idx in (0..pos.div_ceil(4)).step_by(8) {
            let chunk = u64::from_le_bytes(chunk.seq[idx..idx + 8].try_into().unwrap());
            let low_bits = (pos - idx * 4).min(32) * 2;
            let mask = if low_bits == 64 {
                u64::MAX
            } else {
                (1u64 << low_bits) - 1
            };
            let chunk = chunk & mask;
            counts += self.counts[(chunk >> 0) as u8 as usize];
            counts += self.counts[(chunk >> 8) as u8 as usize];
            counts += self.counts[(chunk >> 16) as u8 as usize];
            counts += self.counts[(chunk >> 24) as u8 as usize];
            counts += self.counts[(chunk >> 32) as u8 as usize];
            counts += self.counts[(chunk >> 40) as u8 as usize];
            counts += self.counts[(chunk >> 48) as u8 as usize];
            counts += self.counts[(chunk >> 56) as u8 as usize];
        }
        for c in 0..4 {
            ranks[c] += (counts >> (8 * c)) as u8 as u32;
        }
        let extra_counted = (32usize.wrapping_sub(pos)) % 32;
        ranks[0] -= extra_counted as u32;
        ranks
    }

    #[inline(always)]
    pub fn ranks_bytecount_16(&self, pos: usize) -> Ranks {
        let chunk_idx = pos / 128;
        prefetch_index(&self.blocks, chunk_idx);
        let chunk = &self.blocks[chunk_idx];
        let mut ranks = [0; 4];
        for c in 0..4 {
            ranks[c] += chunk.ranks[c] as u32;
        }

        let pos = pos % 128;

        let mut counts: u32 = 0;

        for idx in (0..pos.div_ceil(4)).step_by(16) {
            let chunk = u128::from_le_bytes(chunk.seq[idx..idx + 16].try_into().unwrap());
            let low_bits = (pos - idx * 4).min(64) * 2;
            let mask = if low_bits == 128 {
                u128::MAX
            } else {
                (1u128 << low_bits) - 1
            };
            let chunk = chunk & mask;
            counts += self.counts[(chunk >> 0) as u8 as usize];
            counts += self.counts[(chunk >> 8) as u8 as usize];
            counts += self.counts[(chunk >> 16) as u8 as usize];
            counts += self.counts[(chunk >> 24) as u8 as usize];
            counts += self.counts[(chunk >> 32) as u8 as usize];
            counts += self.counts[(chunk >> 40) as u8 as usize];
            counts += self.counts[(chunk >> 48) as u8 as usize];
            counts += self.counts[(chunk >> 56) as u8 as usize];
            counts += self.counts[(chunk >> 64) as u8 as usize];
            counts += self.counts[(chunk >> 72) as u8 as usize];
            counts += self.counts[(chunk >> 80) as u8 as usize];
            counts += self.counts[(chunk >> 88) as u8 as usize];
            counts += self.counts[(chunk >> 96) as u8 as usize];
            counts += self.counts[(chunk >> 104) as u8 as usize];
            counts += self.counts[(chunk >> 112) as u8 as usize];
            counts += self.counts[(chunk >> 120) as u8 as usize];
        }
        for c in 0..4 {
            ranks[c] += (counts >> (8 * c)) as u8 as u32;
        }
        let extra_counted = (64usize.wrapping_sub(pos)) % 64;
        ranks[0] -= extra_counted as u32;
        ranks
    }

    #[inline(always)]
    pub fn ranks_bytecount_16_all(&self, pos: usize) -> Ranks {
        let chunk_idx = pos / 128;
        prefetch_index(&self.blocks, chunk_idx);
        let chunk = &self.blocks[chunk_idx];
        let mut ranks = [0; 4];
        for c in 0..4 {
            ranks[c] += chunk.ranks[c] as u32;
        }

        let pos = pos % 128;

        let mut counts: u32 = 0;

        for idx in (0..32).step_by(16) {
            let chunk = u128::from_le_bytes(chunk.seq[idx..idx + 16].try_into().unwrap());
            let low_bits = pos.saturating_sub(idx * 4).min(64) * 2;
            let mask = if low_bits == 128 {
                u128::MAX
            } else {
                (1u128 << low_bits) - 1
            };
            let chunk = chunk & mask;
            let chunk = chunk;
            // Black_box to prevent SIMD (gather is slow..).
            counts += black_box(self.counts[(chunk >> 0) as u8 as usize]);
            counts += black_box(self.counts[(chunk >> 8) as u8 as usize]);
            counts += black_box(self.counts[(chunk >> 16) as u8 as usize]);
            counts += black_box(self.counts[(chunk >> 24) as u8 as usize]);
            counts += black_box(self.counts[(chunk >> 32) as u8 as usize]);
            counts += black_box(self.counts[(chunk >> 40) as u8 as usize]);
            counts += black_box(self.counts[(chunk >> 48) as u8 as usize]);
            counts += black_box(self.counts[(chunk >> 56) as u8 as usize]);
            counts += black_box(self.counts[(chunk >> 64) as u8 as usize]);
            counts += black_box(self.counts[(chunk >> 72) as u8 as usize]);
            counts += black_box(self.counts[(chunk >> 80) as u8 as usize]);
            counts += black_box(self.counts[(chunk >> 88) as u8 as usize]);
            counts += black_box(self.counts[(chunk >> 96) as u8 as usize]);
            counts += black_box(self.counts[(chunk >> 104) as u8 as usize]);
            counts += black_box(self.counts[(chunk >> 112) as u8 as usize]);
            counts += black_box(self.counts[(chunk >> 120) as u8 as usize]);
        }
        for c in 0..4 {
            ranks[c] += (counts >> (8 * c)) as u8 as u32;
        }
        let extra_counted = 128 - pos;
        ranks[0] -= extra_counted as u32;
        ranks
    }

    #[inline(always)]
    pub fn ranks_u64(&self, pos: usize) -> Ranks {
        let chunk_idx = pos / 128;
        prefetch_index(&self.blocks, chunk_idx);
        let chunk = &self.blocks[chunk_idx];
        let mut ranks = [0; 4];
        let pos = pos % 128;

        for idx in (0..pos.div_ceil(4)).step_by(8) {
            let chunk = u64::from_le_bytes(chunk.seq[idx..idx + 8].try_into().unwrap());
            let low_bits = (pos - idx * 4).min(32) * 2;
            let mask = if low_bits == 64 {
                u64::MAX
            } else {
                (1u64 << low_bits) - 1
            };
            let chunk = chunk & mask;
            for c in 0..4 {
                ranks[c as usize] += count_u64(chunk, c);
            }
        }
        for c in 0..4 {
            ranks[c] += chunk.ranks[c] as u32;
        }
        let extra_counted = (32usize.wrapping_sub(pos)) % 32;
        ranks[0] -= extra_counted as u32;
        ranks
    }

    #[inline(always)]
    pub fn ranks_u64_3(&self, pos: usize) -> Ranks {
        let chunk_idx = pos / 128;
        prefetch_index(&self.blocks, chunk_idx);
        let chunk = &self.blocks[chunk_idx];
        let mut ranks = [0; 4];
        let chunk_pos = pos % 128;

        for idx in (0..chunk_pos.div_ceil(4)).step_by(8) {
            let chunk = u64::from_le_bytes(chunk.seq[idx..idx + 8].try_into().unwrap());
            let low_bits = (chunk_pos - idx * 4).min(32) * 2;
            let mask = if low_bits == 64 {
                u64::MAX
            } else {
                (1u64 << low_bits) - 1
            };
            let chunk = chunk & mask;
            for c in 1..4 {
                ranks[c as usize] += count_u64(chunk, c);
            }
        }
        for c in 1..4 {
            ranks[c] += chunk.ranks[c] as u32;
        }
        ranks[0] = pos as u32 - ranks[1] - ranks[2] - ranks[3];
        ranks
    }

    #[inline(always)]
    pub fn ranks_u128(&self, pos: usize) -> Ranks {
        let chunk_idx = pos / 128;
        prefetch_index(&self.blocks, chunk_idx);
        let chunk = &self.blocks[chunk_idx];
        let mut ranks = [0; 4];
        let pos = pos % 128;

        for idx in (0..pos.div_ceil(4)).step_by(16) {
            let chunk = u128::from_le_bytes(chunk.seq[idx..idx + 16].try_into().unwrap());
            let low_bits = (pos - idx * 4).min(64) * 2;
            let mask = if low_bits == 128 {
                u128::MAX
            } else {
                (1u128 << low_bits) - 1
            };
            let chunk = chunk & mask;
            for c in 0..4 {
                ranks[c as usize] += count_u128(chunk, c);
            }
        }
        for c in 0..4 {
            ranks[c] += chunk.ranks[c] as u32;
        }
        let extra_counted = (64usize.wrapping_sub(pos)) % 64;
        ranks[0] -= extra_counted as u32;
        ranks
    }

    #[inline(always)]
    pub fn ranks_u128_3(&self, pos: usize) -> Ranks {
        let chunk_idx = pos / 128;
        prefetch_index(&self.blocks, chunk_idx);
        let chunk = &self.blocks[chunk_idx];
        let mut ranks = [0; 4];
        let chunk_pos = pos % 128;

        for idx in (0..chunk_pos.div_ceil(4)).step_by(16) {
            let chunk = u128::from_le_bytes(chunk.seq[idx..idx + 16].try_into().unwrap());
            let low_bits = (chunk_pos - idx * 4).min(64) * 2;
            let mask = if low_bits == 128 {
                u128::MAX
            } else {
                (1u128 << low_bits) - 1
            };
            let chunk = chunk & mask;
            for c in 1..4 {
                ranks[c as usize] += count_u128(chunk, c);
            }
        }
        for c in 1..4 {
            ranks[c] += chunk.ranks[c] as u32;
        }
        ranks[0] = pos as u32 - ranks[1] - ranks[2] - ranks[3];
        ranks
    }

    #[inline(always)]
    pub fn ranks_u64_all(&self, pos: usize) -> Ranks {
        let chunk_idx = pos / 128;
        prefetch_index(&self.blocks, chunk_idx);
        let chunk = &self.blocks[chunk_idx];
        let mut ranks = [0; 4];
        let pos = pos % 128;

        for idx in (0..32).step_by(8) {
            let chunk = u64::from_le_bytes(chunk.seq[idx..idx + 8].try_into().unwrap());
            let low_bits = pos.saturating_sub(idx * 4).min(32) * 2;
            let mask = if low_bits == 64 {
                u64::MAX
            } else {
                (1u64 << low_bits) - 1
            };
            let chunk = chunk & mask;
            for c in 0..4 {
                ranks[c as usize] += count_u64(chunk, c);
            }
        }
        for c in 0..4 {
            ranks[c] += chunk.ranks[c] as u32;
        }
        let extra_counted = 128 - pos;
        ranks[0] -= extra_counted as u32;
        ranks
    }

    #[inline(always)]
    pub fn ranks_u128_all(&self, pos: usize) -> Ranks {
        let chunk_idx = pos / 128;
        prefetch_index(&self.blocks, chunk_idx);
        let chunk = &self.blocks[chunk_idx];
        let mut ranks = [0; 4];
        let pos = pos % 128;

        for idx in (0..32).step_by(16) {
            let chunk = u128::from_le_bytes(chunk.seq[idx..idx + 16].try_into().unwrap());
            let low_bits = pos.saturating_sub(idx * 4).min(64) * 2;
            let mask = if low_bits == 128 {
                u128::MAX
            } else {
                (1u128 << low_bits) - 1
            };
            let chunk = chunk & mask;
            for c in 0..4 {
                ranks[c as usize] += count_u128(chunk, c);
            }
        }
        for c in 0..4 {
            ranks[c] += chunk.ranks[c] as u32;
        }
        let extra_counted = 128 - pos;
        ranks[0] -= extra_counted as u32;
        ranks
    }
}

fn init_counts() -> [u32; 256] {
    let mut counts = [0u32; 256];
    for b in 0..256 {
        for c in 0..4 {
            counts[b] |= (count_u8(b as u8, c) as u32) << (c * 8);
        }
    }
    counts
}

/// Each 128bit contains 16 u8 with two counts:
/// - the first element has 2 4bit counts of A and C,
/// - the second element has 2 4bit counts of G and T.
fn init_nibble_counts() -> [std::simd::u8x32; 2] {
    let mut counts = [[0u8; 16]; 2];
    for b in 0..16 {
        let bb = (b | (b << 4)) as u8;
        for c in 0..4 {
            counts[c / 2][b] += ((count_u8(bb, c as u8) / 2 as u32) as u8) << ((c & 1) * 4);
        }
    }
    counts.map(|arr| unsafe { std::mem::transmute([arr, arr]) })
}

/// For each 128bp, store:
/// - u8 offsets
/// - 3 u64 counts for c=1,2,3
/// - 256 bits of packed sequence.
/// In total, exactly covers a 512bit cache line.
///
/// This only has to count chars in 64bp=128bits, which becomes a single popcount.
#[repr(C)]
#[repr(align(64))]
#[derive(mem_dbg::MemSize)]
pub struct BwaBlock2 {
    // meta[0,1,2,3] = 0
    // meta[4,5,6,7] = count of c=0,1,2,3 in first half (64bp=128bit).
    meta: [u8; 8],
    // counts for c=1,2,3
    ranks: [u64; 3],
    // 4*64 = 32*8 = 256 bit packed sequence
    seq: [u8; 32],
}

/// Store 4 u32 counts every 128bp = 256bits.
#[derive(mem_dbg::MemSize)]
pub struct BwaRank2 {
    n: usize,
    blocks: Vec<BwaBlock2>,
    counts: [u32; 256],
}

impl BwaRank2 {
    pub fn new(seq: &[u8]) -> Self {
        let n = seq.len();
        let mut blocks = Vec::with_capacity(seq.len().div_ceil(64));
        let mut ranks = [0; 3];

        let mut seq = PackedSeqVec::from_ascii(seq).into_raw();
        seq.resize(seq.capacity(), 0);

        for chunk in seq.as_chunks::<32>().0 {
            let mut meta = [0; 8];
            // count first half.
            for chunk in &chunk.as_chunks::<8>().0[0..2] {
                for c in 0..4 {
                    meta[4 + c as usize] += count_u8x8(chunk, c) as u8;
                }
            }
            blocks.push(BwaBlock2 {
                meta,
                ranks: ranks.map(|x| x as u64),
                seq: *chunk,
            });
            for chunk in chunk.as_chunks::<8>().0 {
                for c in 1..4 {
                    ranks[c as usize - 1] += count_u8x8(chunk, c);
                }
            }
        }

        BwaRank2 {
            n,
            blocks,
            counts: init_counts(),
        }
    }

    #[inline(always)]
    pub fn ranks_u128_3(&self, pos: usize) -> Ranks {
        let chunk_idx = pos / 128;
        prefetch_index(&self.blocks, chunk_idx);
        let chunk = &self.blocks[chunk_idx];
        let mut ranks = [0; 4];
        let chunk_pos = pos % 128;

        // 0 or 1 for left or right half
        let half = chunk_pos / 64;

        // Offset of chunk and half.
        for c in 1..4 {
            ranks[c] += chunk.ranks[c - 1] as u32 + chunk.meta[4 * half + c] as u32;
        }

        // Count chosen half.
        {
            let idx = half * 16;
            let chunk = u128::from_le_bytes(chunk.seq[idx..idx + 16].try_into().unwrap());
            let low_bits = (chunk_pos - idx * 4).min(64) * 2;
            let mask = if low_bits == 128 {
                u128::MAX
            } else {
                (1u128 << low_bits) - 1
            };
            let chunk = chunk & mask;
            for c in 1..4 {
                ranks[c as usize] += count_u128(chunk, c);
            }
        }

        // Fix count for 0.
        ranks[0] = pos as u32 - ranks[1] - ranks[2] - ranks[3];

        ranks
    }

    #[inline(always)]
    pub fn ranks_bytecount_16_all(&self, pos: usize) -> Ranks {
        let chunk_idx = pos / 128;
        prefetch_index(&self.blocks, chunk_idx);
        let chunk = &self.blocks[chunk_idx];
        let mut ranks = [0; 4];
        let chunk_pos = pos % 128;

        // 0 or 1 for left or right half
        let half = chunk_pos / 64;

        // Offset of chunk and half.
        for c in 1..4 {
            ranks[c] += chunk.ranks[c - 1] as u32;
        }

        let mut counts = u32::from_le_bytes(chunk.meta.as_chunks::<4>().0[half]);

        {
            // Count the upper or lower half 128 bits.
            let idx = half * 16;
            let chunk = u128::from_le_bytes(chunk.seq[idx..idx + 16].try_into().unwrap());
            let low_bits = chunk_pos.saturating_sub(idx * 4).min(64) * 2;
            let mask = if low_bits == 128 {
                u128::MAX
            } else {
                (1u128 << low_bits) - 1
            };
            let chunk = chunk & mask;
            counts += self.counts[(chunk >> 0) as u8 as usize];
            counts += self.counts[(chunk >> 8) as u8 as usize];
            counts += self.counts[(chunk >> 16) as u8 as usize];
            counts += self.counts[(chunk >> 24) as u8 as usize];
            counts += self.counts[(chunk >> 32) as u8 as usize];
            counts += self.counts[(chunk >> 40) as u8 as usize];
            counts += self.counts[(chunk >> 48) as u8 as usize];
            counts += self.counts[(chunk >> 56) as u8 as usize];
            counts += self.counts[(chunk >> 64) as u8 as usize];
            counts += self.counts[(chunk >> 72) as u8 as usize];
            counts += self.counts[(chunk >> 80) as u8 as usize];
            counts += self.counts[(chunk >> 88) as u8 as usize];
            counts += self.counts[(chunk >> 96) as u8 as usize];
            counts += self.counts[(chunk >> 104) as u8 as usize];
            counts += self.counts[(chunk >> 112) as u8 as usize];
            counts += self.counts[(chunk >> 120) as u8 as usize];
        }
        for c in 1..4 {
            ranks[c] += (counts >> (8 * c)) as u8 as u32;
        }
        ranks[0] = pos as u32 - ranks[1] - ranks[2] - ranks[3];
        // Doesn't work here; rank[0] not stored.
        // let extra_counted = 64 - chunk_pos;
        // ranks[0] -= extra_counted as u32;

        ranks
    }

    #[inline(always)]
    pub fn ranks_simd_popcount(&self, pos: usize) -> Ranks {
        let chunk_idx = pos / 128;
        prefetch_index(&self.blocks, chunk_idx);
        let chunk = &self.blocks[chunk_idx];
        let mut ranks = [0; 4];
        let chunk_pos = pos % 128;

        // 0 or 1 for left or right half
        let half = chunk_pos / 64;

        // Offset of chunk and half.
        for c in 1..4 {
            ranks[c] += chunk.ranks[c - 1] as u32 + chunk.meta[4 * half + c] as u32;
        }

        {
            use std::mem::transmute as t;

            // Count the upper or lower half 128 bits.
            let idx = half * 16;
            let chunk = u128::from_le_bytes(chunk.seq[idx..idx + 16].try_into().unwrap());
            let low_bits = chunk_pos.saturating_sub(idx * 4).min(64) * 2;
            let mask = if low_bits == 128 {
                u128::MAX
            } else {
                (1u128 << low_bits) - 1
            };
            let chunk = chunk & mask;
            let chunk: [u64; 2] = unsafe { t(chunk) };

            // count AC in first half, GT in second half.
            let simd: u8x32 = unsafe { t([chunk[0], chunk[1], chunk[0], chunk[1]]) };
            let mask5 = u8x32::splat(0x55);
            let mask3: u64x4 = unsafe { t(u8x32::splat(0x33)) };
            let mask_f: u64x4 = unsafe { t(u8x32::splat(0x0f)) };
            // bits of the 4 chars
            // 0000 | 1010  (0, 2)
            const C02: u8x32 = u8x32::from_array(unsafe { t([[!0u8; 16], [!0xAAu8; 16]]) });
            // 0101 | 1111  (1, 3)
            const C13: u8x32 = u8x32::from_array(unsafe { t([[!0x55u8; 16], [!0xFFu8; 16]]) });

            let x1 = simd ^ C02;
            let y1 = (x1 & (x1 >> 1)) & mask5;
            let x2 = simd ^ C13;
            let y2 = (x2 & (x2 >> 1)) & mask5;

            // Go from
            // c0 c0 | c1 c1
            // c2 c2 | c3 c3
            // to: (shuffle)
            // c0 c2 | c1 c3
            // c0 c2 | c1 c3
            // where each value is a u64

            let a: u64x4 = unsafe { t(_mm256_unpacklo_epi64(t(y1), t(y2))) };
            let b: u64x4 = unsafe { t(_mm256_unpackhi_epi64(t(y1), t(y2))) };
            // Now reduce.
            let sum2 = a + b;
            let sum4 = (sum2 & mask3) + ((sum2 >> 2) & mask3);
            let sum8 = (sum4 & mask_f) + ((sum4 >> 4) & mask_f);
            let sum16 = sum8 + (sum8 >> 32);
            // Accumulate the 4 bytes in each u32 using multiplication.
            let sum64 = (unsafe { t::<_, u32x8>(sum16) } * u32x8::splat(0x0101_0101)) >> 24;
            for c in 0..4 {
                // ranks[c] += sum64[c] as u8 as u32;
                ranks[c] += sum64[2 * c] as u8 as u32;
            }
        }

        // TODO: This generates some silly SIMD.
        // If we have the absolute count for rank[0], the extra_counted correction is faster.
        ranks[0] = pos as u32 - ranks[1] - ranks[2] - ranks[3];
        // let extra_counted = 128 - pos;
        // ranks[0] -= extra_counted as u32;

        ranks
    }
}

/// v3: u32 ranks
#[repr(C)]
#[repr(align(64))]
#[derive(mem_dbg::MemSize)]
pub struct BwaBlock3 {
    // 32bit counts for the first and second half.
    ranks: [[u32; 4]; 2],
    // 4*64 = 32*8 = 256 bit packed sequence
    seq: [u8; 32],
}

/// Store 4 u32 counts every 128bp = 256bits.
// #[derive(mem_dbg::MemSize)]
pub struct BwaRank3 {
    n: usize,
    blocks: Vec<BwaBlock3>,
    counts: [u32; 256],
    masks: [[u64; 2]; 64],
}

impl BwaRank3 {
    pub fn new(seq: &[u8]) -> Self {
        let n = seq.len();
        let mut blocks = Vec::with_capacity(seq.len().div_ceil(64));
        let mut ranks = [0; 4];

        let mut seq = PackedSeqVec::from_ascii(seq).into_raw();
        seq.resize(seq.capacity(), 0);

        for chunk in seq.as_chunks::<32>().0 {
            let ranks_start = ranks;
            // count first half.
            for chunk in &chunk.as_chunks::<8>().0[0..2] {
                for c in 0..4 {
                    ranks[c as usize] += count_u8x8(chunk, c) as u32;
                }
            }
            blocks.push(BwaBlock3 {
                ranks: [ranks_start, ranks],
                seq: *chunk,
            });
            // count second half
            for chunk in &chunk.as_chunks::<8>().0[2..4] {
                for c in 0..4 {
                    ranks[c as usize] += count_u8x8(chunk, c) as u32;
                }
            }
        }

        let mut masks = [[0u64; 2]; 64];
        for i in 0..64 {
            let low_bits = i * 2;
            let mask = if low_bits == 128 {
                u128::MAX
            } else {
                (1u128 << low_bits) - 1
            };
            masks[i] = unsafe { std::mem::transmute(mask) };
        }

        BwaRank3 {
            n,
            blocks,
            counts: init_counts(),
            masks,
        }
    }

    #[inline(always)]
    pub fn ranks_u128_3(&self, pos: usize) -> Ranks {
        let chunk_idx = pos / 128;
        prefetch_index(&self.blocks, chunk_idx);
        let chunk = &self.blocks[chunk_idx];
        let chunk_pos = pos % 128;
        let half_pos = pos % 64;

        // 0 or 1 for left or right half
        let half = chunk_pos / 64;
        let mut ranks = [0; 4];

        // Count chosen half.
        {
            let idx = half * 16;
            let chunk = u128::from_le_bytes(chunk.seq[idx..idx + 16].try_into().unwrap());
            let mask: u128 = unsafe { transmute(self.masks[half_pos]) };
            let chunk = chunk & mask;
            for c in 0..4 {
                ranks[c as usize] += count_u128(chunk, c);
            }
        }

        for c in 0..4 {
            ranks[c] += chunk.ranks[half][c];
        }

        // Fix count for 0.
        let extra_counted = 64 - half_pos;
        ranks[0] -= extra_counted as u32;

        ranks
    }

    #[inline(always)]
    pub fn ranks_bytecount_16_all(&self, pos: usize) -> Ranks {
        let chunk_idx = pos / 128;
        prefetch_index(&self.blocks, chunk_idx);
        let chunk = &self.blocks[chunk_idx];
        let chunk_pos = pos % 128;
        let half_pos = pos % 64;

        // 0 or 1 for left or right half
        let half = chunk_pos / 64;
        let mut ranks = [0; 4];

        let mut counts = 0;

        {
            // Count the upper or lower half 128 bits.
            let idx = half * 16;
            let chunk = u128::from_le_bytes(chunk.seq[idx..idx + 16].try_into().unwrap());
            let mask: u128 = unsafe { transmute(self.masks[half_pos]) };
            let chunk = chunk & mask;
            counts += self.counts[(chunk >> 0) as u8 as usize];
            counts += self.counts[(chunk >> 8) as u8 as usize];
            counts += self.counts[(chunk >> 16) as u8 as usize];
            counts += self.counts[(chunk >> 24) as u8 as usize];
            counts += self.counts[(chunk >> 32) as u8 as usize];
            counts += self.counts[(chunk >> 40) as u8 as usize];
            counts += self.counts[(chunk >> 48) as u8 as usize];
            counts += self.counts[(chunk >> 56) as u8 as usize];
            counts += self.counts[(chunk >> 64) as u8 as usize];
            counts += self.counts[(chunk >> 72) as u8 as usize];
            counts += self.counts[(chunk >> 80) as u8 as usize];
            counts += self.counts[(chunk >> 88) as u8 as usize];
            counts += self.counts[(chunk >> 96) as u8 as usize];
            counts += self.counts[(chunk >> 104) as u8 as usize];
            counts += self.counts[(chunk >> 112) as u8 as usize];
            counts += self.counts[(chunk >> 120) as u8 as usize];
        }

        for c in 0..4 {
            ranks[c] += chunk.ranks[half][c];
        }
        for c in 0..4 {
            ranks[c] += (counts >> (8 * c)) as u8 as u32;
        }

        // Fix count for 0.
        let extra_counted = 64 - half_pos;
        ranks[0] -= extra_counted as u32;

        ranks
    }

    #[inline(always)]
    pub fn ranks_simd_popcount(&self, pos: usize) -> Ranks {
        let chunk_idx = pos / 128;
        prefetch_index(&self.blocks, chunk_idx);
        let chunk = &self.blocks[chunk_idx];
        let chunk_pos = pos % 128;
        let half_pos = pos % 64;

        // 0 or 1 for left or right half
        let half = chunk_pos / 64;

        let mut ranks = [0; 4];

        {
            use std::mem::transmute as t;

            // Count the upper or lower half 128 bits.
            let idx = half * 16;
            let chunk = u128::from_le_bytes(chunk.seq[idx..idx + 16].try_into().unwrap());
            let [mask0, mask1] = self.masks[half_pos];
            let mut chunk: [u64; 2] = unsafe { t(chunk) };
            chunk[0] &= mask0;
            chunk[1] &= mask1;

            // count AC in first half, GT in second half.
            let simd: u8x32 = unsafe { t([chunk[0], chunk[1], chunk[0], chunk[1]]) };
            let mask5 = u8x32::splat(0x55);
            let mask3: u64x4 = unsafe { t(u8x32::splat(0x33)) };
            let mask_f: u64x4 = unsafe { t(u8x32::splat(0x0f)) };
            // bits of the 4 chars
            // 0000 | 1010  (0, 2)
            const C02: u8x32 = u8x32::from_array(unsafe { t([[!0u8; 16], [!0xAAu8; 16]]) });
            // 0101 | 1111  (1, 3)
            const C13: u8x32 = u8x32::from_array(unsafe { t([[!0x55u8; 16], [!0xFFu8; 16]]) });

            let x1 = simd ^ C02;
            let y1 = (x1 & (x1 >> 1)) & mask5;
            let x2 = simd ^ C13;
            let y2 = (x2 & (x2 >> 1)) & mask5;

            // Go from
            // c0 c0 | c1 c1
            // c2 c2 | c3 c3
            // to: (shuffle)
            // c0 c2 | c1 c3
            // c0 c2 | c1 c3
            // where each value is a u64

            let a: u64x4 = unsafe { t(_mm256_unpacklo_epi64(t(y1), t(y2))) };
            let b: u64x4 = unsafe { t(_mm256_unpackhi_epi64(t(y1), t(y2))) };
            // Now reduce.
            let sum2 = a + b;
            let sum4 = (sum2 & mask3) + ((sum2 >> 2) & mask3);
            let sum8 = (sum4 & mask_f) + ((sum4 >> 4) & mask_f);
            let sum16 = sum8 + (sum8 >> 32);
            // Accumulate the 4 bytes in each u32 using multiplication.
            let sum64 = (unsafe { t::<_, u32x8>(sum16) } * u32x8::splat(0x0101_0101)) >> 24;
            for c in 0..4 {
                // ranks[c] += sum64[c] as u8 as u32;
                ranks[c] += sum64[2 * c] as u8 as u32;
            }
        }
        for c in 0..4 {
            ranks[c] += chunk.ranks[half][c];
        }

        // Fix count for 0.
        let extra_counted = 64 - half_pos;
        ranks[0] -= extra_counted as u32;

        ranks
    }
}

/// v4: rank for each u64 quart, instead of each u128 half
#[repr(C)]
#[repr(align(64))]
#[derive(mem_dbg::MemSize)]
pub struct BwaBlock4 {
    /// 32bit counts for the entire block
    ranks: [u32; 4],
    /// Each u32 is equivalent to [u8; 4] with counts from start to each u64 quart.
    part_ranks: [u32; 4],
    /// 4*64 = 32*8 = 256 bit packed sequence
    seq: [u8; 32],
}

/// Store 4 u32 counts every 128bp = 256bits.
// #[derive(mem_dbg::MemSize)]
pub struct BwaRank4 {
    n: usize,
    blocks: Vec<BwaBlock4>,
    counts: [u32; 256],
    masks: [u64; 64],
}

impl BwaRank4 {
    pub fn new(seq: &[u8]) -> Self {
        let n = seq.len();
        let mut blocks = Vec::with_capacity(seq.len().div_ceil(64));
        let mut ranks = [0; 4];

        let mut seq = PackedSeqVec::from_ascii(seq).into_raw();
        seq.resize(seq.capacity(), 0);

        for chunk in seq.as_chunks::<32>().0 {
            let ranks_start = ranks;
            let mut part_ranks = [0; 4];
            let mut block_ranks = [0u32; 4];
            // count each part half.
            for (i, chunk) in chunk.as_chunks::<8>().0.iter().enumerate() {
                for c in 0..4 {
                    part_ranks[c] |= block_ranks[i] << (i * 8);
                }
                for c in 0..4 {
                    let cnt = count_u8x8(chunk, c) as u32;
                    ranks[c as usize] += cnt;
                    block_ranks[c as usize] += cnt;
                }
            }
            blocks.push(BwaBlock4 {
                ranks: ranks_start,
                part_ranks,
                seq: *chunk,
            });
        }

        let mut masks = [0u64; 64];
        for i in 0..32 {
            let low_bits = i * 2;
            let mask = if low_bits == 64 {
                u64::MAX
            } else {
                (1u64 << low_bits) - 1
            };
            masks[i] = mask;
        }

        BwaRank4 {
            n,
            blocks,
            counts: init_counts(),
            masks,
        }
    }

    #[inline(always)]
    pub fn prefetch(&self, pos: usize) {
        let chunk_idx = pos / 128;
        prefetch_index(&self.blocks, chunk_idx);
    }

    #[inline(always)]
    pub fn ranks_u64_popcnt(&self, pos: usize) -> Ranks {
        let chunk_idx = pos / 128;
        prefetch_index(&self.blocks, chunk_idx);
        let chunk = &self.blocks[chunk_idx];
        let chunk_pos = pos % 128;
        let quart_pos = pos % 32;

        let quart = chunk_pos / 32;
        let mut ranks = [0; 4];

        // Count chosen quart.
        {
            let idx = quart * 8;
            let chunk = u64::from_le_bytes(chunk.seq[idx..idx + 8].try_into().unwrap());
            let mask = self.masks[quart_pos];
            let chunk = chunk & mask;
            for c in 0..4 {
                ranks[c as usize] += count_u64(chunk, c);
            }
        }

        for c in 0..4 {
            ranks[c] += chunk.ranks[c];
        }
        for c in 0..4 {
            ranks[c] += (chunk.part_ranks[c] >> (quart * 8)) & 0xff;
        }

        // Fix count for 0.
        let extra_counted = 32 - quart_pos;
        ranks[0] -= extra_counted as u32;

        ranks
    }

    #[inline(always)]
    pub fn ranks_bytecount_16_all(&self, pos: usize) -> Ranks {
        let chunk_idx = pos / 128;
        prefetch_index(&self.blocks, chunk_idx);
        let chunk = &self.blocks[chunk_idx];
        let chunk_pos = pos % 128;
        let quart_pos = pos % 32;

        let quart = chunk_pos / 32;
        let mut ranks = [0; 4];

        let mut counts = 0;

        // Count chosen quart.
        {
            let idx = quart * 8;
            let chunk = u64::from_le_bytes(chunk.seq[idx..idx + 8].try_into().unwrap());
            let mask = self.masks[quart_pos];
            let chunk = chunk & mask;
            counts += self.counts[(chunk >> 0) as u8 as usize];
            counts += self.counts[(chunk >> 8) as u8 as usize];
            counts += self.counts[(chunk >> 16) as u8 as usize];
            counts += self.counts[(chunk >> 24) as u8 as usize];
            counts += self.counts[(chunk >> 32) as u8 as usize];
            counts += self.counts[(chunk >> 40) as u8 as usize];
            counts += self.counts[(chunk >> 48) as u8 as usize];
            counts += self.counts[(chunk >> 56) as u8 as usize];
        }

        for c in 0..4 {
            ranks[c] += (counts >> (8 * c)) as u8 as u32;
        }
        for c in 0..4 {
            ranks[c] += chunk.ranks[c];
        }
        for c in 0..4 {
            ranks[c] += (chunk.part_ranks[c] >> (quart * 8)) & 0xff;
        }

        // Fix count for 0.
        let extra_counted = 32 - quart_pos;
        ranks[0] -= extra_counted as u32;

        ranks
    }

    #[inline(always)]
    pub fn ranks_simd_popcount(&self, pos: usize) -> Ranks {
        let chunk_idx = pos / 128;
        prefetch_index(&self.blocks, chunk_idx);
        let chunk = &self.blocks[chunk_idx];
        let chunk_pos = pos % 128;
        let quart_pos = pos % 32;

        let quart = chunk_pos / 32;
        let mut ranks = [0; 4];

        // Count chosen quart.
        {
            use std::mem::transmute as t;

            // Count the upper or lower half 128 bits.
            let idx = quart * 8;
            let mut chunk = u64::from_le_bytes(chunk.seq[idx..idx + 8].try_into().unwrap());
            let mask = self.masks[quart_pos];
            chunk &= mask;

            // count AC in first half, GT in second half.
            let simd = u64x4::splat(chunk);
            let mask5: u64x4 = unsafe { t(u8x32::splat(0x55)) };
            let mask3: u64x4 = unsafe { t(u8x32::splat(0x33)) };
            let mask_f: u64x4 = unsafe { t(u8x32::splat(0x0f)) };
            // bits of the 4 chars
            // 00 | 01 | 10 | 11  (0, 1, 2, 3)
            const C: u64x4 = u64x4::from_array(unsafe {
                t([[!0u8; 8], [!0x55u8; 8], [!0xAAu8; 8], [!0xFFu8; 8]])
            });

            let x = simd ^ C;
            let y = (x & (x >> 1)) & mask5;

            // Go from
            // c0 c0 | c1 c1
            // c2 c2 | c3 c3
            // to: (shuffle)
            // c0 c2 | c1 c3
            // c0 c2 | c1 c3
            // where each value is a u64

            // Now reduce.
            let sum2 = y;
            let sum4 = (sum2 & mask3) + ((sum2 >> 2) & mask3);
            let sum8 = (sum4 & mask_f) + ((sum4 >> 4) & mask_f);
            let sum16 = sum8 + (sum8 >> 32);
            // Accumulate the 4 bytes in each u32 using multiplication.
            let sum64: u32x8 = (unsafe { t::<_, u32x8>(sum16) } * u32x8::splat(0x0101_0101)) >> 24;
            for c in 0..4 {
                // ranks[c] += sum64[c] as u8 as u32;
                ranks[c] += sum64[2 * c] as u8 as u32;
            }
        }

        for c in 0..4 {
            ranks[c] += chunk.ranks[c];
        }
        for c in 0..4 {
            ranks[c] += (chunk.part_ranks[c] >> (quart * 8)) & 0xff;
        }

        // Fix count for 0.
        let extra_counted = 32 - quart_pos;
        ranks[0] -= extra_counted as u32;

        ranks
    }

    #[inline(always)]
    pub fn ranks_u64_popcnt_async(&self, pos: usize) -> impl Future<Output = Ranks> {
        #[inline(always)]
        async move {
            let chunk_idx = pos / 128;
            prefetch_index(&self.blocks, chunk_idx);

            yield_now().await;

            let chunk_idx = pos / 128;
            let chunk = &self.blocks[chunk_idx];
            let chunk_pos = pos % 128;
            let quart_pos = pos % 32;

            let quart = chunk_pos / 32;
            let mut ranks = [0; 4];

            // Count chosen quart.
            {
                let idx = quart * 8;
                let chunk = u64::from_le_bytes(chunk.seq[idx..idx + 8].try_into().unwrap());
                let mask = self.masks[quart_pos];
                let chunk = chunk & mask;
                for c in 0..4 {
                    ranks[c as usize] += count_u64(chunk, c);
                }
            }

            for c in 0..4 {
                ranks[c] += chunk.ranks[c];
            }
            for c in 0..4 {
                ranks[c] += (chunk.part_ranks[c] >> (quart * 8)) & 0xff;
            }

            // Fix count for 0.
            let extra_counted = 32 - quart_pos;
            ranks[0] -= extra_counted as u32;

            ranks
        }
    }

    #[inline(always)]
    pub fn ranks_u64_popcnt_async_nowake(&self, pos: usize) -> impl Future<Output = Ranks> {
        let chunk_idx = pos / 128;
        prefetch_index(&self.blocks, chunk_idx);
        #[inline(always)]
        async move {
            // Yield::new().await;

            // Recomputed to reduce the state across the await point.
            let chunk_idx = pos / 128;
            let chunk = &self.blocks[chunk_idx];
            let chunk_pos = pos % 128;
            let quart_pos = pos % 32;

            let quart = chunk_pos / 32;
            let mut ranks = [0; 4];

            // Count chosen quart.
            {
                let idx = quart * 8;
                let chunk = u64::from_le_bytes(chunk.seq[idx..idx + 8].try_into().unwrap());
                let mask = self.masks[quart_pos];
                let chunk = chunk & mask;
                for c in 0..4 {
                    ranks[c as usize] += count_u64(chunk, c);
                }
            }

            for c in 0..4 {
                ranks[c] += chunk.ranks[c];
            }
            for c in 0..4 {
                ranks[c] += (chunk.part_ranks[c] >> (quart * 8)) & 0xff;
            }

            // Fix count for 0.
            let extra_counted = 32 - quart_pos;
            ranks[0] -= extra_counted as u32;

            ranks
        }
    }

    #[inline(always)]
    pub fn ranks_u64_popcnt_coro(&self, pos: usize) -> impl Coroutine<Yield = (), Return = Ranks> {
        let chunk_idx = pos / 128;
        prefetch_index(&self.blocks, chunk_idx);
        #[inline(always)]
        #[coroutine]
        move || {
            // Yield::new().await;

            // Recomputed to reduce the state across the await point.
            let chunk_idx = pos / 128;
            let chunk = &self.blocks[chunk_idx];
            let chunk_pos = pos % 128;
            let quart_pos = pos % 32;

            let quart = chunk_pos / 32;
            let mut ranks = [0; 4];

            // Count chosen quart.
            {
                let idx = quart * 8;
                let chunk = u64::from_le_bytes(chunk.seq[idx..idx + 8].try_into().unwrap());
                let mask = self.masks[quart_pos];
                let chunk = chunk & mask;
                for c in 0..4 {
                    ranks[c as usize] += count_u64(chunk, c);
                }
            }

            for c in 0..4 {
                ranks[c] += chunk.ranks[c];
            }
            for c in 0..4 {
                ranks[c] += (chunk.part_ranks[c] >> (quart * 8)) & 0xff;
            }

            // Fix count for 0.
            let extra_counted = 32 - quart_pos;
            ranks[0] -= extra_counted as u32;

            ranks
        }
    }
}

struct Yield(bool);
impl Yield {
    fn new() -> Self {
        Yield(false)
    }
}
impl Future for Yield {
    type Output = ();

    #[inline(always)]
    fn poll(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        if self.0 {
            std::task::Poll::Ready(())
        } else {
            self.get_mut().0 = true;
            // cx.waker().wake_by_ref();
            std::task::Poll::Pending
        }
    }
}
