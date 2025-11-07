#![allow(incomplete_features)]
#![feature(
    generic_const_exprs,
    portable_simd,
    coroutines,
    coroutine_trait,
    exact_div
)]

pub mod blocks;
pub mod count;
pub mod count4;
pub mod ranker;

use count::*;
use packed_seq::{PackedSeqVec, SeqVec};
use ranker::prefetch_index;

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
