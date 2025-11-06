#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
use packed_seq::{PackedSeqVec, SeqVec};

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
        let extra_counted = (32 - pos) % 32;
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
        let extra_counted = (32 - pos) % 32;
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
        let extra_counted = (64 - pos) % 64;
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

        BwaRank { n, blocks }
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
        let extra_counted = (32 - pos) % 32;
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
        let extra_counted = (64 - pos) % 64;
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
    // meta[1,2,3] = 0
    // meta[5,6,7] = count of c=1,2,3 in first half (64bp=128bit).
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
                for c in 1..4 {
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

        BwaRank2 { n, blocks }
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
}
