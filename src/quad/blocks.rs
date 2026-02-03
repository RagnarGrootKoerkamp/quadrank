#![allow(non_camel_case_types)]

use std::{arch::x86_64::_mm_sign_epi32, array::from_fn};
use wide::u32x4;

use crate::{
    count::{count_u8x8, count_u64_mask, count_u64_mid_mask},
    quad::{
        BasicBlock, Ranks,
        count4::{CountFn, SimdCount10, SimdCount11B, count4_u8x8, double_mid},
    },
};

use super::count4::{
    DOUBLE_TRANSPOSED_MID_MASKS, SimdCountSlice, TRANSPOSED_MID_MASKS, WideSimdCount2,
};

#[inline(always)]
fn strict_add(a: Ranks, b: Ranks) -> Ranks {
    from_fn(|c| a[c].strict_add(b[c]))
}

#[inline(always)]
fn extra_counted<const B: usize, const T: bool, CF: CountFn<B, T>>(pos: usize) -> u32 {
    if CF::S == 0 {
        return 0;
    }
    let ans = (if CF::FIXED {
        (CF::S * 4) - pos % (CF::S * 4)
    } else {
        -(pos as isize) as usize % (CF::S * 4)
    }) as u32;
    ans
}

/// A 512 bit block that does not store any counts itself.
#[repr(align(64))]
#[derive(mem_dbg::MemSize)]
pub struct Basic512 {
    seq: [u8; 64],
}

impl BasicBlock for Basic512 {
    const X: usize = 2; // DNA
    const B: usize = 64; // Bytes of characters in block.
    const N: usize = 256; // Number of characters in block.
    const C: usize = 64; // Bytes of the underlying count function.
    const W: usize = 0;
    const TRANSPOSED: bool = false;

    fn new(_ranks: Ranks, data: &[u8]) -> Self {
        Basic512 {
            seq: *data.as_array().unwrap(),
        }
    }

    #[inline(always)]
    fn count4(&self, pos: usize) -> Ranks {
        type CF = SimdCountSlice;
        let mut ranks = CF::count(&self.seq, pos);
        ranks[0] -= extra_counted::<64, false, CF>(pos);
        ranks
    }
}

/// A 256 bit block that does not store any counts itself.
#[repr(align(32))]
#[derive(mem_dbg::MemSize)]
pub struct Basic256 {
    seq: [u8; 32],
}

impl BasicBlock for Basic256 {
    const X: usize = 2; // DNA
    const B: usize = 32; // Bytes of characters in block.
    const N: usize = 128; // Number of characters in block.
    const C: usize = 32; // Bytes of the underlying count function.
    const W: usize = 0;
    const TRANSPOSED: bool = false;

    fn new(_ranks: Ranks, data: &[u8]) -> Self {
        let data: &[u8; _] = data.as_array().unwrap();
        Self { seq: *data }
    }

    #[inline(always)]
    fn count4(&self, pos: usize) -> Ranks {
        type CF = SimdCountSlice;
        let mut ranks = CF::count(&self.seq, pos);
        ranks[0] -= extra_counted::<32, false, CF>(pos);
        ranks
    }
}

/// A 256 bit block that does not store any counts itself.
#[repr(align(16))]
#[derive(mem_dbg::MemSize)]
pub struct Basic128 {
    seq: [u8; 16],
}

impl BasicBlock for Basic128 {
    const X: usize = 2; // DNA
    const B: usize = 16; // Bytes of characters in block.
    const N: usize = 64; // Number of characters in block.
    const C: usize = 16; // Bytes of the underlying count function.
    const W: usize = 0;
    const TRANSPOSED: bool = false;

    fn new(_ranks: Ranks, data: &[u8]) -> Self {
        let data: &[u8; _] = data.as_array().unwrap();
        Self { seq: *data }
    }

    #[inline(always)]
    fn count4(&self, pos: usize) -> Ranks {
        type CF = WideSimdCount2;
        let mut ranks = CF::count(&self.seq, pos);
        ranks[0] -= extra_counted::<_, false, CF>(pos);
        ranks
    }
}

/// u32 global offset to start and 1/2.
#[repr(align(64))]
#[derive(mem_dbg::MemSize)]
pub struct QuadBlock32x2P {
    // 32bit counts for the first and second half.
    ranks: [[u32; 4]; 2],
    // 4*64 = 32*8 = 256 bit packed sequence
    seq: [u8; 32],
}

impl BasicBlock for QuadBlock32x2P {
    const X: usize = 2; // DNA
    const B: usize = 32;
    const N: usize = 128;
    const C: usize = 16;
    const W: usize = 32;
    const TRANSPOSED: bool = false;

    fn new(ranks: Ranks, data: &[u8]) -> Self {
        let data: &[u8; Self::B] = data.as_array().unwrap();
        let mut half_ranks = ranks;
        // count first half.
        for chunk in &data.as_chunks::<8>().0[0..2] {
            for c in 0..4 {
                half_ranks[c as usize] =
                    half_ranks[c as usize].strict_add(count_u8x8(chunk, c) as u32);
            }
        }
        QuadBlock32x2P {
            ranks: [ranks, half_ranks],
            seq: *data,
        }
    }

    #[inline(always)]
    fn count4(&self, pos: usize) -> Ranks {
        type CF = SimdCount10;
        let mut ranks = [0; 4];

        // 0 or 1 for left or right half
        let half = pos / 64;
        let half_pos = pos % 64;

        let idx = half * 16;
        let inner_counts = CF::count(&self.seq[idx..idx + 16].try_into().unwrap(), half_pos);

        ranks[0] -= extra_counted::<_, _, CF>(pos);

        for c in 0..4 {
            ranks[c] += inner_counts[c];
        }
        for c in 0..4 {
            ranks[c] += self.ranks[half][c];
        }

        ranks
    }
}

/// u32 global offset and 4 8bit deltas to each 64 bit part.
#[repr(align(64))]
#[derive(mem_dbg::MemSize)]
pub struct QuadBlock32_8_8_8FP {
    /// 32bit counts for the entire block
    ranks: [u32; 4],
    /// Each u32 is equivalent to [u8; 4] with counts from start to each u64 quart.
    part_ranks: [u32; 4],
    seq: [u8; 32],
}

impl BasicBlock for QuadBlock32_8_8_8FP {
    const X: usize = 2; // DNA
    const B: usize = 32;
    const N: usize = 128;
    const C: usize = 8;
    const W: usize = 32;
    const TRANSPOSED: bool = false;

    fn new(ranks: Ranks, data: &[u8]) -> Self {
        let data: &[u8; Self::B] = data.as_array().unwrap();
        let mut part_ranks = [0; 4];
        let mut block_ranks = [0u32; 4];
        // count each part half.
        for (i, chunk) in data.as_chunks::<8>().0.iter().enumerate() {
            for c in 0..4 {
                part_ranks[c] |= block_ranks[c] << (i * 8);
            }
            for c in 0..4 {
                let cnt = count_u8x8(chunk, c) as u32;
                block_ranks[c as usize] += cnt;
            }
        }
        QuadBlock32_8_8_8FP {
            ranks,
            part_ranks,
            seq: *data,
        }
    }

    #[inline(always)]
    fn count4(&self, pos: usize) -> Ranks {
        type CF = SimdCount10;
        let mut ranks = [0; 4];

        let quart = pos / 32;
        let quart_pos = pos % 32;

        let idx = quart * 8;
        let chunk = &self.seq[idx..idx + 8].try_into().unwrap();
        let inner_counts = CF::count(chunk, quart_pos);
        for c in 0..4 {
            ranks[c] += inner_counts[c];
        }

        for c in 0..4 {
            ranks[c] += self.ranks[c];
        }

        for c in 0..4 {
            ranks[c] += (self.part_ranks[c] >> (quart * 8)) & 0xff;
        }

        ranks
    }

    #[inline(always)]
    fn count1(&self, pos: usize, c: u8) -> u32 {
        let mut rank = 0;
        let quart = pos / 32;
        let quart_pos = pos % 32;
        let idx = quart * 8;
        let chunk = u64::from_le_bytes(self.seq[idx..idx + 8].try_into().unwrap());
        let inner_count = count_u64_mask(chunk, c, quart_pos);
        rank += inner_count;
        rank += (self.part_ranks[c as usize] >> (quart * 8)) & 0xff;
        rank += self.ranks[c as usize];

        rank
    }
    #[inline(always)]
    fn count1x2(&self, other: &Self, pos0: usize, pos1: usize, c: u8) -> (u32, u32) {
        let mut rank0 = 0;
        let mut rank1 = 0;
        let quart0 = pos0 / 32;
        let quart_pos0 = pos0 % 32;
        let quart1 = pos1 / 32;
        let quart_pos1 = pos1 % 32;
        let idx0 = quart0 * 8;
        let idx1 = quart1 * 8;
        let chunk0 = u64::from_le_bytes(self.seq[idx0..idx0 + 8].try_into().unwrap());
        let chunk1 = u64::from_le_bytes(other.seq[idx1..idx1 + 8].try_into().unwrap());
        let inner_count0 = count_u64_mask(chunk0, c, quart_pos0);
        let inner_count1 = count_u64_mask(chunk1, c, quart_pos1);
        rank0 += inner_count0;
        rank1 += inner_count1;
        rank0 += (self.part_ranks[c as usize] >> (quart0 * 8)) & 0xff;
        rank1 += (other.part_ranks[c as usize] >> (quart1 * 8)) & 0xff;
        rank0 += self.ranks[c as usize];
        rank1 += other.ranks[c as usize];
        (rank0, rank1)
    }
}

/// Store four 18 bit offsets to 3/6th, and two 7 bit deltas from there to to 1/6th and 5/6th.
/// Then count the previous or next 32bp.
#[repr(align(64))]
#[derive(mem_dbg::MemSize)]
pub struct QuadBlock7_18_7P {
    /// high half: 18bit counts for the global offset to position 32 (middle of first u128).
    /// low half: 2x 7bit counts for the first half of 2nd and 3rd 128 parts.
    ranks: [u32; 4],
    // u128x3 = u8x48 = 384 bit packed sequence
    seq: [u8; 48],
}

impl BasicBlock for QuadBlock7_18_7P {
    const X: usize = 2; // DNA
    const B: usize = 48;
    const N: usize = 192;
    const C: usize = 8;
    const W: usize = 18;
    const TRANSPOSED: bool = false;

    fn new(mut ranks: Ranks, data: &[u8]) -> Self {
        let data: &[u8; Self::B] = data.as_array().unwrap();
        // Counts before each u64 block.
        let mut bs = [[0u32; 4]; 6];
        // count each part half.
        for (i, chunk) in data.as_chunks::<8>().0.iter().enumerate() {
            bs[i] = strict_add(bs[i], count4_u8x8(*chunk));
        }
        // global ranks are to the middle
        ranks = strict_add(strict_add(ranks, bs[0]), strict_add(bs[1], bs[2]));
        let p1 = strict_add(bs[1], bs[2]);
        let p2 = strict_add(bs[3], bs[4]);
        let part_ranks: Ranks = from_fn(|c| (p1[c] << 7) | p2[c]);
        Self {
            ranks: from_fn(|c| (ranks[c] << 14) | part_ranks[c]),
            seq: *data,
        }
    }

    #[inline(always)]
    fn count4(&self, pos: usize) -> Ranks {
        type CF = SimdCount10;
        let mut ranks = u32x4::splat(0);

        let hex = pos / 32;

        let idx = hex * 8;

        let inner_counts = CF::count_mid(&self.seq[idx..idx + 8].try_into().unwrap(), pos % 64);

        use std::mem::transmute as t;
        let sign = (pos as u32 % 64).wrapping_sub(32);
        ranks += unsafe { t::<_, u32x4>(_mm_sign_epi32(t(inner_counts), t(u32x4::splat(sign)))) };

        let self_ranks = u32x4::new(self.ranks);
        ranks += self_ranks >> 14;

        let shuffle = u32x4::splat(0x000077u32);
        let shift = (shuffle >> (4 * hex) as u32) & u32x4::splat(7);

        let parts = self_ranks & u32x4::splat(0x3fff);
        let sign2 = (hex / 2).wrapping_sub(1);
        ranks += unsafe {
            t::<_, u32x4>(_mm_sign_epi32(
                t((parts >> shift) & u32x4::splat(0x7f)),
                t(u32x4::splat(sign2 as u32)),
            ))
        };
        ranks.to_array()
    }

    #[inline(always)]
    fn count1(&self, pos: usize, c: u8) -> u32 {
        let hex = pos / 32;

        let idx = hex * 8;

        let self_ranks = self.ranks[c as usize];
        let mut rank = self_ranks >> 14;

        let word = u64::from_le_bytes(self.seq[idx..idx + 8].try_into().unwrap());
        let inner = count_u64_mid_mask(word, c, pos % 64);

        rank += if (pos & 32) > 0 {
            inner
        } else {
            inner.wrapping_neg()
        };

        let shuffle = 0x000707u64;
        let shift = (shuffle >> (8 * hex)) & 7;
        let parts = self_ranks & 0x3fff;
        let sign2 = (hex / 2).wrapping_sub(1);
        rank = rank.wrapping_add((((parts) >> shift) & 0x7f).wrapping_mul(sign2 as u32));
        rank
    }

    #[inline(always)]
    fn count1x2(&self, other: &Self, pos0: usize, pos1: usize, c: u8) -> (u32, u32) {
        let hex0 = pos0 / 32;
        let hex1 = pos1 / 32;

        let idx0 = hex0 * 8;
        let idx1 = hex1 * 8;

        let word0 = u64::from_le_bytes(self.seq[idx0..idx0 + 8].try_into().unwrap());
        let inner0 = count_u64_mid_mask(word0, c, pos0 % 64);
        let word1 = u64::from_le_bytes(other.seq[idx1..idx1 + 8].try_into().unwrap());
        let inner1 = count_u64_mid_mask(word1, c, pos1 % 64);

        let mut rank0 = if (pos0 & 32) > 0 {
            inner0
        } else {
            inner0.wrapping_neg()
        };
        let mut rank1 = if (pos1 & 32) > 0 {
            inner1
        } else {
            inner1.wrapping_neg()
        };

        let self_ranks0 = self.ranks[c as usize];
        let self_ranks1 = other.ranks[c as usize];

        rank0 = rank0.wrapping_add(self_ranks0 >> 14);
        rank1 = rank1.wrapping_add(self_ranks1 >> 14);

        let shuffle = 0x000077u32;
        let shift0 = (shuffle >> (4 * hex0) as u32) & 7;
        let shift1 = (shuffle >> (4 * hex1) as u32) & 7;
        let parts0 = self_ranks0 & 0x3fff;
        let parts1 = self_ranks1 & 0x3fff;
        let sign20 = (hex0 / 2).wrapping_sub(1);
        let sign21 = (hex1 / 2).wrapping_sub(1);
        rank0 = rank0.wrapping_add((((parts0) >> shift0) & 0x7f).wrapping_mul(sign20 as u32));
        rank1 = rank1.wrapping_add((((parts1) >> shift1) & 0x7f).wrapping_mul(sign21 as u32));
        (rank0, rank1)
    }
}

fn transpose_bits(data: &[u8; 16]) -> [u64; 2] {
    let mut out = [0u64; 2];
    for i in 0..16 {
        let byte = data[i];
        for b in 0..4 {
            let l = (byte >> (2 * b)) & 1;
            let h = (byte >> (2 * b + 1)) & 1;
            out[0] |= (l as u64) << (4 * i + b);
            out[1] |= (h as u64) << (4 * i + b);
        }
    }
    out
}
fn negate_and_transpose_bits(data: &[u8; 16]) -> [u64; 2] {
    let [l, h] = transpose_bits(data);
    [!l, !h]
}

/// Store four 24 bit offsets to 1/3th, and an 8 bit deltas to the end.
/// Then count the previous or next 64bp.
#[repr(align(64))]
#[derive(mem_dbg::MemSize)]
pub struct QuadBlock24_8 {
    /// 24 high bits: offset to end
    /// 8 low bits: delta to end of first trip
    ranks: [u32; 4],
    // u128x3 = u8x48 = 384 bit packed sequence
    seq: [[u8; 16]; 3],
}

impl BasicBlock for QuadBlock24_8 {
    const X: usize = 2; // DNA
    const B: usize = 48;
    const N: usize = 192;
    const C: usize = 16;
    // TODO: Make this 25
    const W: usize = 24;
    const TRANSPOSED: bool = true;

    fn new(mut ranks: Ranks, data: &[u8]) -> Self {
        let data: &[u8; Self::B] = data.as_array().unwrap();
        // Counts before each u64 block.
        let mut bs = [[0u32; 4]; 6];
        let mut sum = [0u32; 4];
        // count each part half.
        for (i, chunk) in data.as_chunks::<8>().0.iter().enumerate() {
            bs[i] = strict_add(bs[i], count4_u8x8(*chunk));
            sum = strict_add(sum, bs[i]);
        }
        // global ranks are to block
        ranks = strict_add(ranks, sum);
        let part_rank = strict_add(strict_add(bs[2], bs[3]), strict_add(bs[4], bs[5]));
        Self {
            ranks: from_fn(|c| ranks[c].strict_shl(8).strict_add(part_rank[c])),
            seq: from_fn(|i| unsafe {
                std::mem::transmute(transpose_bits(
                    &data[i * 16..i * 16 + 16].try_into().unwrap(),
                ))
            }),
        }
    }

    #[inline(always)]
    fn count4(&self, pos: usize) -> Ranks {
        type CF = SimdCount11B;
        let mut ranks = u32x4::splat(0);

        let tri = pos / 64;

        let inner_counts = CF::count_mid(&self.seq[tri], pos % 128);

        use std::mem::transmute as t;
        let sign = (pos as u32 % 128).wrapping_sub(64);
        ranks += unsafe { t::<_, u32x4>(_mm_sign_epi32(t(inner_counts), t(u32x4::splat(sign)))) };

        let self_ranks = u32x4::new(self.ranks);
        ranks += self_ranks >> 8;

        // for tri=0 and tri=1, shift down by 0
        // for tri=2, shift down by 8
        let shift = (tri as u32 / 2) * 8;
        let parts = self_ranks & u32x4::splat(0x00ff);
        ranks -= parts >> u32x4::splat(shift);
        ranks.to_array()
    }

    #[inline(always)]
    fn count1(&self, pos: usize, c: u8) -> u32 {
        let tri = pos / 64;
        let [l, h]: [u64; 2] = unsafe { std::mem::transmute(self.seq[tri]) };
        let mask = TRANSPOSED_MID_MASKS[pos % 128];
        let c2 = !(c as u64);
        let l = l ^ (c2 & 1).wrapping_neg();
        let h = h ^ ((c2 >> 1) & 1).wrapping_neg();
        let cnt = (l & h & mask).count_ones();

        let mut rank = self.ranks[c as usize] >> 8;
        let shift = (tri as u32 / 2) * 8;
        let part = self.ranks[c as usize] & 0x00ff;
        rank -= part >> shift;

        if pos % 128 < 64 {
            rank - cnt
        } else {
            rank + cnt
        }
    }
}

/// Store four 64 bit offsets to the middle, then count 64 bp.
///
/// Like BWA-MEM.
#[repr(align(64))]
#[derive(mem_dbg::MemSize)]
pub struct QuadBlock64 {
    // 4*64 = 256 bit counts
    // FIXME: Update to actual u64 values
    ranks: [[u32; 4]; 2],
    // 2x transposed packed sequence
    seq: [[u8; 16]; 2],
}

impl BasicBlock for QuadBlock64 {
    const X: usize = 2; // DNA
    const B: usize = 32;
    const N: usize = 128;
    const C: usize = 16;
    const W: usize = 32;
    const TRANSPOSED: bool = true;

    fn new(mut ranks: Ranks, data: &[u8]) -> Self {
        let data: &[u8; Self::B] = data.as_array().unwrap();
        // Counts before each u64 block.
        let mut bs = [[0u32; 4]; 4];
        let mut sum = [0u32; 4];
        // count each part half.
        for (i, chunk) in data.as_chunks::<8>().0.iter().enumerate() {
            bs[i] = strict_add(bs[i], count4_u8x8(*chunk));
            sum = strict_add(sum, bs[i]);
        }
        // global ranks are to block
        ranks = strict_add(ranks, strict_add(bs[0], bs[1]));
        Self {
            ranks: [ranks; 2],
            seq: from_fn(|i| unsafe {
                std::mem::transmute(transpose_bits(
                    &data[i * 16..i * 16 + 16].try_into().unwrap(),
                ))
            }),
        }
    }

    #[inline(always)]
    fn count4(&self, pos: usize) -> Ranks {
        type CF = SimdCount11B;
        let mut ranks = u32x4::splat(0);

        let half = pos / 64;

        let inner_counts = CF::count_mid(&self.seq[half], pos % 128);

        use std::mem::transmute as t;
        let sign = (pos as u32).wrapping_sub(64);
        ranks += unsafe { t::<_, u32x4>(_mm_sign_epi32(t(inner_counts), t(u32x4::splat(sign)))) };

        let self_ranks = u32x4::new(self.ranks[0]);
        ranks += self_ranks;

        // for tri=0 and tri=1, shift down by 0
        // for tri=2, shift down by 8
        ranks.to_array()
    }

    #[inline(always)]
    fn count1(&self, pos: usize, c: u8) -> u32 {
        let half = pos / 64;
        let mask = TRANSPOSED_MID_MASKS[pos];
        let [l, h]: [u64; 2] = unsafe { std::mem::transmute(self.seq[half]) };
        let c2 = !(c as u64);
        let l = l ^ (c2 & 1).wrapping_neg();
        let h = h ^ ((c2 >> 1) & 1).wrapping_neg();
        let inner_count = (l & h & mask).count_ones();
        let rank = self.ranks[0][c as usize];
        if pos < 64 {
            rank - inner_count
        } else {
            rank + inner_count
        }
    }
}

/// Store four 32 bit offsets to the middle, then count 128 bp.
#[repr(align(64))]
#[derive(mem_dbg::MemSize)]
pub struct QuadBlock32 {
    // First [u32; 4]: ranks
    // Then [[u64;2];3] of (high, low) transposed pairs
    seq: [[u8; 16]; 4],
}

impl BasicBlock for QuadBlock32 {
    const X: usize = 2; // DNA
    const B: usize = 48;
    const N: usize = 192;
    const C: usize = 0;
    const W: usize = 32;
    const TRANSPOSED: bool = true;

    fn new(mut ranks: Ranks, data: &[u8]) -> Self {
        let data: &[u8; Self::B] = data.as_array().unwrap();
        // Counts before each u64 block.
        let mut bs = [[0u32; 4]; 6];
        // count each part half.
        for (i, chunk) in data.as_chunks::<8>().0.iter().enumerate() {
            bs[i] = strict_add(bs[i], count4_u8x8(*chunk));
        }
        // global ranks are to block
        unsafe {
            let mut seq = [[0u64; 2]; 4];
            ranks = strict_add(ranks, strict_add(bs[0], bs[1]));
            seq[0] = std::mem::transmute(ranks);
            for i in 0..3 {
                seq[i + 1] = std::mem::transmute(negate_and_transpose_bits(
                    &data[i * 16..i * 16 + 16].try_into().unwrap(),
                ))
            }
            Self {
                seq: std::mem::transmute(seq),
            }
        }
    }

    #[inline(always)]
    fn count4(&self, mut pos: usize) -> Ranks {
        let mut ranks = u32x4::splat(0);

        // correct for 128bits of ranks
        pos += 64;

        let half = pos / 128;

        let inner_counts = double_mid(&self.seq[2 * half..2 * half + 2].try_into().unwrap(), pos);

        use std::mem::transmute as t;
        let sign = (pos as u32).wrapping_sub(128);
        ranks += unsafe { t::<_, u32x4>(_mm_sign_epi32(t(inner_counts), t(u32x4::splat(sign)))) };

        let self_ranks = u32x4::new(unsafe { t(self.seq[0]) });
        ranks += self_ranks;

        ranks.to_array()
    }
}

/// Store four 16 bit offsets to the middle, then count 128 bp.
///
/// The offsets are interleaved with the transposed sequence data:
/// The `[u16; 8]` is `[A, C, high, high, G, T, low, low]`.
#[repr(align(64))]
#[derive(mem_dbg::MemSize)]
pub struct QuadBlock16 {
    seq: [[u8; 16]; 4],
}

impl BasicBlock for QuadBlock16 {
    const X: usize = 2; // DNA
    const B: usize = 56;
    const N: usize = 224;
    const C: usize = 0;
    const W: usize = 16;
    const TRANSPOSED: bool = true;

    fn new(mut ranks: Ranks, data: &[u8]) -> Self {
        let data: &[u8; Self::B] = data.as_array().unwrap();
        // FIXME
        // Counts before each u64 block.
        // count each part half.
        for chunk in data[0..24].as_chunks::<8>().0 {
            ranks = strict_add(ranks, count4_u8x8(*chunk));
        }
        // global ranks are to block
        unsafe {
            let mut seq = [[0u16; 8]; 4];

            let [low, high] = negate_and_transpose_bits(&data[0..16].try_into().unwrap());

            seq[0] = [
                ranks[0].try_into().unwrap(),
                ranks[1].try_into().unwrap(),
                low as u16,
                (low >> 16) as u16,
                ranks[2].try_into().unwrap(),
                ranks[3].try_into().unwrap(),
                high as u16,
                (high >> 16) as u16,
            ];
            for i in 0..3 {
                seq[i + 1] = std::mem::transmute(negate_and_transpose_bits(
                    &data[8 + i * 16..8 + i * 16 + 16].try_into().unwrap(),
                ))
            }
            Self {
                seq: std::mem::transmute(seq),
            }
        }
    }

    #[inline(always)]
    fn count4(&self, mut pos: usize) -> Ranks {
        let mut ranks = u32x4::splat(0);

        // correct for 128bits of ranks
        pos += 32;

        let half = pos / 128;
        // FIXME: Avoid convertion from u64x4 to u32x4 and then back.
        let inner_counts = double_mid(&self.seq[2 * half..2 * half + 2].try_into().unwrap(), pos);

        use std::mem::transmute as t;
        let sign = (pos as u32).wrapping_sub(128);
        // FIXME: Is there a u64x4 equivalent of this?
        ranks += unsafe { t::<_, u32x4>(_mm_sign_epi32(t(inner_counts), t(u32x4::splat(sign)))) };

        let u16s: &[u16; 8] = unsafe { t(&self.seq[0]) };
        // This becomes a single simd shuffle.
        let self_ranks = u32x4::new([
            u16s[0] as u32,
            u16s[1] as u32,
            u16s[4] as u32,
            u16s[5] as u32,
        ]);
        ranks += self_ranks;

        ranks.to_array()
    }

    #[inline(always)]
    fn count1(&self, mut pos: usize, c: u8) -> u32 {
        pos += 32;
        let half = pos / 128;
        let data: &[[u8; 16]; 2] = (self.seq[2 * half..2 * half + 2]).try_into().unwrap();
        let masks = DOUBLE_TRANSPOSED_MID_MASKS[pos];
        let mut cnt = 0;

        for i in 0..2 {
            let l = u64::from_le_bytes(data[i][0..8].try_into().unwrap());
            let h = u64::from_le_bytes(data[i][8..16].try_into().unwrap());
            let mask = masks[i];
            // chunk &= mask;

            let l = l ^ (c as u64 & 1).wrapping_neg();
            let h = h ^ ((c as u64) >> 1).wrapping_neg();
            cnt += (l & h & mask).count_ones();
        }

        let seq_u16 = unsafe { &*(self.seq[0].as_ptr() as *const [u16; 8]) };
        let idx = c + (c & 2);
        let rank = seq_u16[idx as usize] as u32;
        if pos < 128 { rank - cnt } else { rank + cnt }
    }
}
