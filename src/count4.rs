//! Various methods for counting the number of characters equal to 0,1,2,3.

use std::{
    arch::x86_64::{
        _mm256_sad_epu8, _mm256_shuffle_epi8, _mm256_unpackhi_epi64, _mm256_unpacklo_epi64,
    },
    simd::{u8x32, u32x8, u64x4},
};

use crate::{
    Ranks,
    count::{count_u8, count_u64, count_u128},
};

pub fn count4_u8x8(data: [u8; 8]) -> Ranks {
    SimdCount7::count(&data, 32)
}

pub trait CountFn<const B: usize>: Sync {
    /// The number of bytes processed at a time.
    /// Used to compute the overshoot.
    const S: usize;
    /// Fixed: always the entire B-byte input is processed.
    /// Non-fixed (variable): Only process first pos/B chunks.
    const FIXED: bool;
    /// Function that can count on B bytes of data.
    fn count(data: &[u8; B], pos: usize) -> Ranks;
    /// Count characters from position `pos` to the end.
    fn count_right(_data: &[u8; B], _pos: usize) -> Ranks {
        unimplemented!()
    }
    fn count_mid(_data: &[u8; B], _pos: usize) -> Ranks {
        unimplemented!()
    }
}

pub struct Naive;
impl<const B: usize> CountFn<B> for Naive {
    const S: usize = 1;
    const FIXED: bool = false;
    #[inline(always)]
    fn count(data: &[u8; B], pos: usize) -> Ranks {
        let mut counts = [0u32; 4];
        for &byte in &data[0..pos / 4] {
            for i in 0..4 {
                let c = (byte >> (i * 2)) & 0b11;
                counts[c as usize] += 1;
            }
        }
        {
            let byte = data[pos / 4];
            for i in 0..(pos % 4) {
                let c = (byte >> (i * 2)) & 0b11;
                counts[c as usize] += 1;
            }
        }
        counts
    }
}

pub struct U64PopcntSlice;
impl<const B: usize> CountFn<B> for U64PopcntSlice {
    const S: usize = 8;
    const FIXED: bool = false;
    #[inline(always)]
    fn count(data: &[u8; B], pos: usize) -> Ranks {
        let mut ranks = [0; 4];
        for idx in (0..pos.div_ceil(4)).step_by(8) {
            let chunk = u64::from_le_bytes(data[idx..idx + 8].try_into().unwrap());
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
        ranks
    }
    #[inline(always)]
    fn count_right(data: &[u8; B], pos: usize) -> Ranks {
        let mut ranks = [0; 4];
        for idx in (8 * (pos / 32)..B).step_by(8).rev() {
            let chunk = u64::from_le_bytes(data[idx..idx + 8].try_into().unwrap());
            let low_bits = pos.saturating_sub(idx * 4) * 2;
            let mask = if low_bits == 64 {
                0
            } else {
                !((1u64 << low_bits) - 1)
            };
            let chunk = chunk & mask;
            for c in 0..4 {
                ranks[c as usize] += count_u64(chunk, c);
            }
        }
        ranks
    }
}

pub struct U64Popcnt;
impl CountFn<8> for U64Popcnt {
    const S: usize = 8;
    const FIXED: bool = true;
    #[inline(always)]
    fn count(data: &[u8; 8], pos: usize) -> Ranks {
        let mut ranks = [0; 4];
        {
            let chunk = u64::from_le_bytes((*data).try_into().unwrap());
            let mask = MASKS[pos];
            let chunk = chunk & mask;

            let scatter = 0x5555555555555555u64;
            let mask = u64x4::from_array(std::array::from_fn(|c| c as u64 * scatter));
            let chunk = u64x4::splat(chunk);
            let tmp = chunk ^ mask;
            let union = (tmp | (tmp >> 1)) & u64x4::splat(scatter);

            for c in 0..4 {
                ranks[c as usize] += 32 - union.as_array()[c].count_ones();
            }
        }

        ranks
    }
}

pub struct U64Popcnt3;
impl<const B: usize> CountFn<B> for U64Popcnt3 {
    const S: usize = 8;
    const FIXED: bool = false;
    fn count(data: &[u8; B], pos: usize) -> Ranks {
        let mut ranks = [0; 4];
        for idx in (0..pos.div_ceil(4)).step_by(8) {
            let chunk = u64::from_le_bytes(data[idx..idx + 8].try_into().unwrap());
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
        ranks
    }
}

pub struct U128Popcnt;
impl<const B: usize> CountFn<B> for U128Popcnt {
    const S: usize = 16;
    const FIXED: bool = false;
    #[inline(always)]
    fn count(data: &[u8; B], pos: usize) -> Ranks {
        let mut ranks = [0; 4];
        for idx in (0..pos.div_ceil(4)).step_by(16) {
            let chunk = u128::from_le_bytes(data[idx..idx + 16].try_into().unwrap());
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
        ranks
    }
}

pub struct U128Popcnt3;
impl<const B: usize> CountFn<B> for U128Popcnt3 {
    const S: usize = 16;
    const FIXED: bool = false;
    #[inline(always)]
    fn count(data: &[u8; B], pos: usize) -> Ranks {
        let mut ranks = [0; 4];
        for idx in (0..pos.div_ceil(4)).step_by(16) {
            let chunk = u128::from_le_bytes(data[idx..idx + 16].try_into().unwrap());
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
        ranks
    }
}

pub const BYTE_COUNTS: [u32; 256] = {
    let mut counts = [0u32; 256];
    let mut b = 0;
    while b < 256 {
        counts[b] |= (count_u8(b as u8, 0) as u32) << (0 * 8);
        counts[b] |= (count_u8(b as u8, 1) as u32) << (1 * 8);
        counts[b] |= (count_u8(b as u8, 2) as u32) << (2 * 8);
        counts[b] |= (count_u8(b as u8, 3) as u32) << (3 * 8);
        b += 1;
    }
    counts
};

pub struct ByteLookup;
impl<const B: usize> CountFn<B> for ByteLookup {
    const S: usize = 1;
    const FIXED: bool = false;
    #[inline(always)]
    fn count(data: &[u8; B], pos: usize) -> Ranks {
        let mut counts: u32 = 0;

        for idx in 0..pos.div_ceil(4) {
            let byte = data[idx];
            let low_bits = (pos - idx * 4).min(4) * 2;
            let mask = (1u64 << low_bits) - 1;
            let byte = byte & mask as u8;
            counts += BYTE_COUNTS[byte as usize];
        }

        std::array::from_fn(|i| (counts >> (8 * i)) & 0xff)
    }
}

pub struct ByteLookup4;
impl<const B: usize> CountFn<B> for ByteLookup4 {
    const S: usize = 4;
    const FIXED: bool = false;
    #[inline(always)]
    fn count(data: &[u8; B], pos: usize) -> Ranks {
        let mut counts: u32 = 0;

        for idx in (0..pos.div_ceil(4)).step_by(4) {
            let chunk = u32::from_le_bytes(data[idx..idx + 4].try_into().unwrap());
            let low_bits = (pos - idx * 4).min(16) * 2;
            let mask = (1u64 << low_bits) - 1;
            let chunk = chunk & mask as u32;
            counts += BYTE_COUNTS[(chunk >> 0) as u8 as usize];
            counts += BYTE_COUNTS[(chunk >> 8) as u8 as usize];
            counts += BYTE_COUNTS[(chunk >> 16) as u8 as usize];
            counts += BYTE_COUNTS[(chunk >> 24) as u8 as usize];
        }

        std::array::from_fn(|i| (counts >> (8 * i)) & 0xff)
    }
}

pub struct ByteLookup8Slice;
impl<const B: usize> CountFn<B> for ByteLookup8Slice {
    const S: usize = 8;
    const FIXED: bool = false;
    #[inline(always)]
    fn count(data: &[u8; B], pos: usize) -> Ranks {
        let mut counts: u32 = 0;

        for idx in (0..pos.div_ceil(4)).step_by(8) {
            let chunk = u64::from_le_bytes(data[idx..idx + 8].try_into().unwrap());
            let low_bits = (pos - idx * 4).min(32) * 2;
            let mask = if low_bits == 64 {
                u64::MAX
            } else {
                (1u64 << low_bits) - 1
            };
            let chunk = chunk & mask;
            counts += BYTE_COUNTS[(chunk >> 0) as u8 as usize];
            counts += BYTE_COUNTS[(chunk >> 8) as u8 as usize];
            counts += BYTE_COUNTS[(chunk >> 16) as u8 as usize];
            counts += BYTE_COUNTS[(chunk >> 24) as u8 as usize];
            counts += BYTE_COUNTS[(chunk >> 32) as u8 as usize];
            counts += BYTE_COUNTS[(chunk >> 40) as u8 as usize];
            counts += BYTE_COUNTS[(chunk >> 48) as u8 as usize];
            counts += BYTE_COUNTS[(chunk >> 56) as u8 as usize];
        }

        std::array::from_fn(|i| (counts >> (8 * i)) & 0xff)
    }
}

pub struct ByteLookup8;
impl CountFn<8> for ByteLookup8 {
    const S: usize = 8;
    const FIXED: bool = true;
    #[inline(always)]
    fn count(data: &[u8; 8], pos: usize) -> Ranks {
        let mut counts: u32 = 0;

        {
            let chunk = u64::from_le_bytes((*data).try_into().unwrap());
            let low_bits = pos * 2;
            let mask = if low_bits == 64 {
                u64::MAX
            } else {
                (1u64 << low_bits) - 1
            };
            let chunk = chunk & mask;
            counts += BYTE_COUNTS[(chunk >> 0) as u8 as usize];
            counts += BYTE_COUNTS[(chunk >> 8) as u8 as usize];
            counts += BYTE_COUNTS[(chunk >> 16) as u8 as usize];
            counts += BYTE_COUNTS[(chunk >> 24) as u8 as usize];
            counts += BYTE_COUNTS[(chunk >> 32) as u8 as usize];
            counts += BYTE_COUNTS[(chunk >> 40) as u8 as usize];
            counts += BYTE_COUNTS[(chunk >> 48) as u8 as usize];
            counts += BYTE_COUNTS[(chunk >> 56) as u8 as usize];
        }

        std::array::from_fn(|i| (counts >> (8 * i)) & 0xff)
    }
}

pub struct ByteLookup16;
impl<const B: usize> CountFn<B> for ByteLookup16 {
    const S: usize = 16;
    const FIXED: bool = false;
    #[inline(always)]
    fn count(data: &[u8; B], pos: usize) -> Ranks {
        let mut counts: u32 = 0;

        for idx in (0..pos.div_ceil(4)).step_by(16) {
            let chunk = u128::from_le_bytes(data[idx..idx + 16].try_into().unwrap());
            let low_bits = (pos - idx * 4).min(64) * 2;
            let mask = if low_bits == 128 {
                u128::MAX
            } else {
                (1u128 << low_bits) - 1
            };
            let chunk = chunk & mask;
            counts += BYTE_COUNTS[(chunk >> 0) as u8 as usize];
            counts += BYTE_COUNTS[(chunk >> 8) as u8 as usize];
            counts += BYTE_COUNTS[(chunk >> 16) as u8 as usize];
            counts += BYTE_COUNTS[(chunk >> 24) as u8 as usize];
            counts += BYTE_COUNTS[(chunk >> 32) as u8 as usize];
            counts += BYTE_COUNTS[(chunk >> 40) as u8 as usize];
            counts += BYTE_COUNTS[(chunk >> 48) as u8 as usize];
            counts += BYTE_COUNTS[(chunk >> 56) as u8 as usize];
            counts += BYTE_COUNTS[(chunk >> 64) as u8 as usize];
            counts += BYTE_COUNTS[(chunk >> 72) as u8 as usize];
            counts += BYTE_COUNTS[(chunk >> 80) as u8 as usize];
            counts += BYTE_COUNTS[(chunk >> 88) as u8 as usize];
            counts += BYTE_COUNTS[(chunk >> 96) as u8 as usize];
            counts += BYTE_COUNTS[(chunk >> 104) as u8 as usize];
            counts += BYTE_COUNTS[(chunk >> 112) as u8 as usize];
            counts += BYTE_COUNTS[(chunk >> 120) as u8 as usize];
        }

        std::array::from_fn(|i| (counts >> (8 * i)) & 0xff)
    }
}

pub struct ByteLookup16x2;
impl CountFn<32> for ByteLookup16x2 {
    const S: usize = 32;
    const FIXED: bool = false;
    #[inline(always)]
    fn count(data: &[u8; 32], pos: usize) -> Ranks {
        let mut counts: u32 = 0;

        for idx in (0..pos.div_ceil(4)).step_by(16) {
            let chunk = u128::from_le_bytes(data[idx..idx + 16].try_into().unwrap());
            let low_bits = (pos - idx * 4).min(64) * 2;
            let mask = if low_bits == 128 {
                u128::MAX
            } else {
                (1u128 << low_bits) - 1
            };
            let chunk = chunk & mask;
            counts += BYTE_COUNTS[(chunk >> 0) as u8 as usize];
            counts += BYTE_COUNTS[(chunk >> 8) as u8 as usize];
            counts += BYTE_COUNTS[(chunk >> 16) as u8 as usize];
            counts += BYTE_COUNTS[(chunk >> 24) as u8 as usize];
            counts += BYTE_COUNTS[(chunk >> 32) as u8 as usize];
            counts += BYTE_COUNTS[(chunk >> 40) as u8 as usize];
            counts += BYTE_COUNTS[(chunk >> 48) as u8 as usize];
            counts += BYTE_COUNTS[(chunk >> 56) as u8 as usize];
            counts += BYTE_COUNTS[(chunk >> 64) as u8 as usize];
            counts += BYTE_COUNTS[(chunk >> 72) as u8 as usize];
            counts += BYTE_COUNTS[(chunk >> 80) as u8 as usize];
            counts += BYTE_COUNTS[(chunk >> 88) as u8 as usize];
            counts += BYTE_COUNTS[(chunk >> 96) as u8 as usize];
            counts += BYTE_COUNTS[(chunk >> 104) as u8 as usize];
            counts += BYTE_COUNTS[(chunk >> 112) as u8 as usize];
            counts += BYTE_COUNTS[(chunk >> 120) as u8 as usize];
        }

        std::array::from_fn(|i| (counts >> (8 * i)) & 0xff)
    }
}

/// Wide, because it counts 128 at a time.
pub struct WideSimdCount;
impl CountFn<16> for WideSimdCount {
    const S: usize = 16;
    const FIXED: bool = true;
    #[inline(always)]
    fn count(data: &[u8; 16], pos: usize) -> Ranks {
        let mut ranks = [0; 4];
        {
            use std::mem::transmute as t;

            let chunk = u128::from_le_bytes(*data);
            let low_bits = pos * 2;
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
        ranks
    }
}

pub static MASKS: [u64; 64] = {
    let mut masks = [0u64; 64];
    let mut i = 0;
    while i < 32 {
        let low_bits = i * 2;
        let mask = if low_bits == 64 {
            u64::MAX
        } else {
            (1u64 << low_bits) - 1
        };
        masks[i] = mask;
        masks[i + 32] = !mask;
        i += 1;
    }
    masks
};
pub static MID_MASKS: [u64; 64] = {
    let mut masks = [0u64; 64];
    let mut i = 0;
    while i < 32 {
        let low_bits = i * 2;
        let mask = if low_bits == 64 {
            u64::MAX
        } else {
            (1u64 << low_bits) - 1
        };
        masks[i] = !mask;
        masks[i + 32] = mask;
        i += 1;
    }
    masks
};

pub static SIMD_MASKS: [u64x4; 64] = {
    let mut masks = [u64x4::splat(0); 64];
    let mut i = 0;
    while i < 32 {
        let low_bits = i * 2;
        let mask = if low_bits == 64 {
            u64::MAX
        } else {
            (1u64 << low_bits) - 1
        };
        masks[i] = u64x4::splat(mask);
        masks[i + 32] = u64x4::splat(!mask);
        i += 1;
    }
    masks
};
pub static SIMD_MID_MASKS: [u64x4; 64] = {
    let mut masks = [u64x4::splat(0); 64];
    let mut i = 0;
    while i < 32 {
        let low_bits = i * 2;
        let mask = if low_bits == 64 {
            u64::MAX
        } else {
            (1u64 << low_bits) - 1
        };
        masks[i] = u64x4::splat(!mask);
        masks[i + 32] = u64x4::splat(mask);
        i += 1;
    }
    masks
};

/// First half: mask out everything >= pos
/// Second half: mask out everything < pos
pub static WIDE_MASKS: [u128; 128] = {
    let mut masks = [0u128; 128];
    let mut i = 0;
    while i < 64 {
        let low_bits = i * 2;
        let mask = if low_bits == 128 {
            u128::MAX
        } else {
            (1u128 << low_bits) - 1
        };
        masks[i] = mask;
        masks[i + 64] = !mask;
        i += 1;
    }
    masks
};

impl CountFn<8> for WideSimdCount {
    const S: usize = 8;
    const FIXED: bool = true;
    #[inline(always)]
    fn count(data: &[u8; 8], pos: usize) -> Ranks {
        let mut ranks = [0; 4];
        {
            use std::mem::transmute as t;

            // Count one u64 quarter of bits.
            let mut chunk = u64::from_le_bytes((*data).try_into().unwrap());
            let mask = MASKS[pos];
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

            // Now reduce.
            let sum1 = y;
            let sum2 = (sum1 & mask3) + ((sum1 >> 2) & mask3);
            let sum4 = (sum2 & mask_f) + ((sum2 >> 4) & mask_f);
            let sum8 = sum4 + (sum4 >> 32);
            // Accumulate the 4 bytes in each u32 using multiplication.
            let sum32: u32x8 = (unsafe { t::<_, u32x8>(sum8) } * u32x8::splat(0x0101_0101)) >> 24;
            for c in 0..4 {
                // ranks[c] += sum64[c] as u8 as u32;
                ranks[c] += sum32[2 * c] as u8 as u32;
            }
        }

        ranks
    }
}

pub struct SimdCount2;
impl CountFn<8> for SimdCount2 {
    const S: usize = 8;
    const FIXED: bool = true;
    #[inline(always)]
    fn count(data: &[u8; 8], pos: usize) -> Ranks {
        let mut ranks = [0; 4];
        {
            use std::mem::transmute as t;

            // Count one u64 quarter of bits.
            let mut chunk = u64::from_le_bytes((*data).try_into().unwrap());
            let mask = MASKS[pos];
            chunk &= mask;

            // count AC in first half, GT in second half.
            let simd = u64x4::splat(chunk);
            let zero = u8x32::splat(0);
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

            // Now reduce.
            let sum1 = y;
            let sum2 = (sum1 & mask3) + ((sum1 >> 2) & mask3);
            let sum4 = (sum2 & mask_f) + ((sum2 >> 4) & mask_f);
            // Accumulate the 8 bytes in each u64 and write them to the low 16 bits.
            let sum32: u64x4 = unsafe { t(_mm256_sad_epu8(t(sum4), t(zero))) };
            for c in 0..4 {
                ranks[c] += sum32[c] as u32;
            }
        }

        ranks
    }
}

pub struct SimdCount3;
impl CountFn<8> for SimdCount3 {
    const S: usize = 8;
    const FIXED: bool = true;
    #[inline(always)]
    fn count(data: &[u8; 8], pos: usize) -> Ranks {
        let mut ranks = [0; 4];
        {
            use std::mem::transmute as t;

            // Count one u64 quarter of bits.
            let mut chunk = u64::from_le_bytes((*data).try_into().unwrap());
            let mask = MASKS[pos];
            chunk &= mask;

            // count AC in first half, GT in second half.
            let simd = u64x4::splat(chunk);
            let zero = u8x32::splat(0);
            let mask5: u64x4 = unsafe { t(u8x32::splat(0x55)) };
            // bits of the 4 chars
            // 00 | 01 | 10 | 11  (0, 1, 2, 3)
            const C: u64x4 = u64x4::from_array(unsafe {
                t([[!0u8; 8], [!0x55u8; 8], [!0xAAu8; 8], [!0xFFu8; 8]])
            });

            let x = simd ^ C;
            let y = (x & (x >> 1)) & mask5;

            let byte_counts = u8x32::from_array([
                // popcount(0..16) (in fact, only 0b0000, 0b0001, 0b0100, 0b0101 matter)
                0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, //
                0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
            ]);

            // Now reduce.
            let lo = y & mask5;
            let hi = (y >> 4) & mask5;
            let popcnt1: u8x32 = unsafe { t(_mm256_shuffle_epi8(t(byte_counts), t(lo))) };
            let popcnt2: u8x32 = unsafe { t(_mm256_shuffle_epi8(t(byte_counts), t(hi))) };
            let sum4 = popcnt1 + popcnt2;
            // Accumulate the 8 bytes in each u64 and write them to the low 16 bits.
            let sum32: u64x4 = unsafe { t(_mm256_sad_epu8(t(sum4), t(zero))) };
            for c in 0..4 {
                ranks[c] += sum32[c] as u32;
            }
        }

        ranks
    }
}

pub struct SimdCount4;
impl CountFn<8> for SimdCount4 {
    const S: usize = 8;
    const FIXED: bool = true;
    #[inline(always)]
    fn count(data: &[u8; 8], pos: usize) -> Ranks {
        let mut ranks = [0; 4];
        {
            use std::mem::transmute as t;

            // Count one u64 quarter of bits.
            let mut chunk = u64::from_le_bytes((*data).try_into().unwrap());
            let mask = MASKS[pos];
            chunk &= mask;

            // count AC in first half, GT in second half.
            let simd = u64x4::splat(chunk);
            let zero = u8x32::splat(0);
            let mask5: u64x4 = unsafe { t(u8x32::splat(0x55)) };
            // bits of the 4 chars
            // 00 | 01 | 10 | 11  (0, 1, 2, 3)
            const C: u64x4 = u64x4::from_array(unsafe {
                t([[!0u8; 8], [!0x55u8; 8], [!0xAAu8; 8], [!0xFFu8; 8]])
            });

            let x = simd ^ C;
            let y = (x & (x >> 1)) & mask5;

            let byte_counts = u8x32::from_array([
                // popcount(0..16) (in fact, only 0b0000, 0b0001, 0b0100, 0b0101 matter)
                0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, //
                0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
            ]);

            // Now reduce.
            // no need for mask_f here.
            // Those are needed to get rid of possible 1 high bits that mask the value to 0,
            // but we already know those aren't 0 anyway in our case.
            let lo = y;
            let hi = y >> 4;
            let popcnt1: u8x32 = unsafe { t(_mm256_shuffle_epi8(t(byte_counts), t(lo))) };
            let popcnt2: u8x32 = unsafe { t(_mm256_shuffle_epi8(t(byte_counts), t(hi))) };
            let sum4 = popcnt1 + popcnt2;
            // Accumulate the 8 bytes in each u64 and write them to the low 16 bits.
            let sum32: u64x4 = unsafe { t(_mm256_sad_epu8(t(sum4), t(zero))) };
            for c in 0..4 {
                ranks[c] += sum32[c] as u32;
            }
        }

        ranks
    }
}

pub struct SimdCount5;
impl CountFn<8> for SimdCount5 {
    const S: usize = 8;
    const FIXED: bool = true;
    #[inline(always)]
    fn count(data: &[u8; 8], pos: usize) -> Ranks {
        let mut ranks = [0; 4];
        {
            use std::mem::transmute as t;

            // Count one u64 quarter of bits.
            let mut chunk = u64::from_le_bytes((*data).try_into().unwrap());
            let mask = MASKS[pos];
            chunk &= mask;

            // count AC in first half, GT in second half.
            let simd = u64x4::splat(chunk);
            let zero = u8x32::splat(0);
            let mask5: u64x4 = unsafe { t(u8x32::splat(0x55)) };
            let mask_f: u64x4 = unsafe { t(u8x32::splat(0x0f)) };
            // bits of the 4 chars
            // 00 | 01 | 10 | 11  (0, 1, 2, 3)
            const C: u64x4 = u64x4::from_array(unsafe {
                t([[!0u8; 8], [!0x55u8; 8], [!0xAAu8; 8], [!0xFFu8; 8]])
            });

            let x = simd ^ C;
            let y = (x & (x >> 1)) & mask5;

            let byte_counts = u8x32::from_array([
                // popcount(0..16) (in fact, only 0b0000, 0b0001, 0b0100, 0b0101 matter)
                0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, //
                0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
            ]);

            // Now reduce.
            let y: u8x32 = unsafe { t(y) };
            // no need for mask_f here.
            // Those are needed to get rid of possible 1 high bits that mask the value to 0,
            // but we already know those aren't 0 anyway in our case.
            let mix = (y | (y >> 3)) & unsafe { t::<_, u8x32>(mask_f) };
            let sum4: u8x32 = unsafe { t(_mm256_shuffle_epi8(t(byte_counts), t(mix))) };
            // Accumulate the 8 bytes in each u64 and write them to the low 16 bits.
            let sum32: u64x4 = unsafe { t(_mm256_sad_epu8(t(sum4), t(zero))) };
            for c in 0..4 {
                ranks[c] += sum32[c] as u32;
            }
        }

        ranks
    }
}

// New: Drop the &mask5 since high bits are 0 anyway.
pub struct SimdCount6;
impl CountFn<8> for SimdCount6 {
    const S: usize = 8;
    const FIXED: bool = true;
    #[inline(always)]
    fn count(data: &[u8; 8], pos: usize) -> Ranks {
        let mut ranks = [0; 4];
        {
            use std::mem::transmute as t;

            // Count one u64 quarter of bits.
            let mut chunk = u64::from_le_bytes((*data).try_into().unwrap());
            let mask = MASKS[pos];
            chunk &= mask;

            // count AC in first half, GT in second half.
            let simd = u64x4::splat(chunk);
            let zero = u8x32::splat(0);
            let mask_f: u64x4 = unsafe { t(u8x32::splat(0x0f)) };
            // bits of the 4 chars
            // 00 | 01 | 10 | 11  (0, 1, 2, 3)
            const C: u64x4 = u64x4::from_array(unsafe {
                t([[!0u8; 8], [!0x55u8; 8], [!0xAAu8; 8], [!0xFFu8; 8]])
            });

            let x = simd ^ C;
            let y = x & (x >> 1);

            let byte_counts = u8x32::from_array([
                // +1 for 0001
                // +1 for 0100
                0, 1, 0, 1, 1, 2, 1, 2, 0, 1, 0, 1, 1, 2, 1, 2, //
                0, 1, 0, 1, 1, 2, 1, 2, 0, 1, 0, 1, 1, 2, 1, 2,
            ]);

            // Now reduce.
            // no need for mask_f here.
            // Those are needed to get rid of possible 1 high bits that mask the value to 0,
            // but we already know those aren't 0 anyway in our case.
            let lo = y & mask_f;
            let hi = (y >> 4) & mask_f;
            let popcnt1: u8x32 = unsafe { t(_mm256_shuffle_epi8(t(byte_counts), t(lo))) };
            let popcnt2: u8x32 = unsafe { t(_mm256_shuffle_epi8(t(byte_counts), t(hi))) };
            let sum4 = popcnt1 + popcnt2;
            // Accumulate the 8 bytes in each u64 and write them to the low 16 bits.
            let sum32: u64x4 = unsafe { t(_mm256_sad_epu8(t(sum4), t(zero))) };
            for c in 0..4 {
                ranks[c] += sum32[c] as u32;
            }
        }

        ranks
    }
}

// New: drop the x&(x>>1) and instead lookup nibbles directly
pub struct SimdCount7;
impl CountFn<8> for SimdCount7 {
    const S: usize = 8;
    const FIXED: bool = true;
    #[inline(always)]
    fn count(data: &[u8; 8], pos: usize) -> Ranks {
        let mut ranks = [0; 4];
        {
            use std::mem::transmute as t;

            // Count one u64 quarter of bits.
            let mut chunk = u64::from_le_bytes((*data).try_into().unwrap());
            let mask = if pos == 32 {
                u64::MAX
            } else {
                (1u64 << (2 * pos)) - 1
            };
            chunk &= mask;

            // count AC in first half, GT in second half.
            let simd = u64x4::splat(chunk);
            let zero = u8x32::splat(0);
            let mask_f: u64x4 = unsafe { t(u8x32::splat(0x0f)) };
            // bits of the 4 chars
            // 00 | 01 | 10 | 11  (0, 1, 2, 3)
            const C: u64x4 = u64x4::from_array(unsafe {
                t([[!0u8; 8], [!0x55u8; 8], [!0xAAu8; 8], [!0xFFu8; 8]])
            });

            let x = simd ^ C;
            let y = x;

            let byte_counts = u8x32::from_array([
                // +1 for 11 in the low half
                // +1 for 11 in the high half
                0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 2, //
                0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 2,
            ]);

            // Now reduce.
            // no need for mask_f here.
            // Those are needed to get rid of possible 1 high bits that mask the value to 0,
            // but we already know those aren't 0 anyway in our case.
            let lo = y & mask_f;
            let hi = (y >> 4) & mask_f;
            let popcnt1: u8x32 = unsafe { t(_mm256_shuffle_epi8(t(byte_counts), t(lo))) };
            let popcnt2: u8x32 = unsafe { t(_mm256_shuffle_epi8(t(byte_counts), t(hi))) };
            let sum4 = popcnt1 + popcnt2;
            // Accumulate the 8 bytes in each u64 and write them to the low 16 bits.
            let sum32: u64x4 = unsafe { t(_mm256_sad_epu8(t(sum4), t(zero))) };
            for c in 0..4 {
                ranks[c] += sum32[c] as u32;
            }
        }

        ranks
    }
    #[inline(always)]
    fn count_right(data: &[u8; 8], pos: usize) -> Ranks {
        let mut ranks = [0; 4];
        {
            use std::mem::transmute as t;

            // Count one u64 quarter of bits.
            let mut chunk = u64::from_le_bytes((*data).try_into().unwrap());
            let mask = if pos == 32 {
                0
            } else {
                !((1u64 << (2 * pos)) - 1)
            };
            chunk &= mask;

            // count AC in first half, GT in second half.
            let simd = u64x4::splat(chunk);
            let zero = u8x32::splat(0);
            let mask_f: u64x4 = unsafe { t(u8x32::splat(0x0f)) };
            // bits of the 4 chars
            // 00 | 01 | 10 | 11  (0, 1, 2, 3)
            const C: u64x4 = u64x4::from_array(unsafe {
                t([[!0u8; 8], [!0x55u8; 8], [!0xAAu8; 8], [!0xFFu8; 8]])
            });

            let x = simd ^ C;
            let y = x;

            let byte_counts = u8x32::from_array([
                // +1 for 11 in the low half
                // +1 for 11 in the high half
                0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 2, //
                0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 2,
            ]);

            // Now reduce.
            // no need for mask_f here.
            // Those are needed to get rid of possible 1 high bits that mask the value to 0,
            // but we already know those aren't 0 anyway in our case.
            let lo = y & mask_f;
            let hi = (y >> 4) & mask_f;
            let popcnt1: u8x32 = unsafe { t(_mm256_shuffle_epi8(t(byte_counts), t(lo))) };
            let popcnt2: u8x32 = unsafe { t(_mm256_shuffle_epi8(t(byte_counts), t(hi))) };
            let sum4 = popcnt1 + popcnt2;
            // Accumulate the 8 bytes in each u64 and write them to the low 16 bits.
            let sum32: u64x4 = unsafe { t(_mm256_sad_epu8(t(sum4), t(zero))) };
            for c in 0..4 {
                ranks[c] += sum32[c] as u32;
            }
        }

        ranks
    }
    /// Pos can twice the size here.
    /// If first half, count top elements, otherwise count bottom elements.
    #[inline(always)]
    fn count_mid(data: &[u8; 8], pos: usize) -> Ranks {
        let mut ranks = [0; 4];
        {
            use std::mem::transmute as t;

            // Count one u64 quarter of bits.
            let mut chunk = u64::from_le_bytes((*data).try_into().unwrap());
            let mask = MID_MASKS[pos];
            chunk &= mask;

            // count AC in first half, GT in second half.
            let simd = u64x4::splat(chunk);
            let zero = u8x32::splat(0);
            let mask_f: u64x4 = unsafe { t(u8x32::splat(0x0f)) };
            // bits of the 4 chars
            // 00 | 01 | 10 | 11  (0, 1, 2, 3)
            const C: u64x4 = u64x4::from_array(unsafe {
                t([[!0u8; 8], [!0x55u8; 8], [!0xAAu8; 8], [!0xFFu8; 8]])
            });

            let x = simd ^ C;
            let y = x;

            let byte_counts = u8x32::from_array([
                // +1 for 11 in the low half
                // +1 for 11 in the high half
                0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 2, //
                0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 2,
            ]);

            // Now reduce.
            // no need for mask_f here.
            // Those are needed to get rid of possible 1 high bits that mask the value to 0,
            // but we already know those aren't 0 anyway in our case.
            let lo = y & mask_f;
            let hi = (y >> 4) & mask_f;
            let popcnt1: u8x32 = unsafe { t(_mm256_shuffle_epi8(t(byte_counts), t(lo))) };
            let popcnt2: u8x32 = unsafe { t(_mm256_shuffle_epi8(t(byte_counts), t(hi))) };
            let sum4 = popcnt1 + popcnt2;
            // Accumulate the 8 bytes in each u64 and write them to the low 16 bits.
            let sum32: u64x4 = unsafe { t(_mm256_sad_epu8(t(sum4), t(zero))) };
            for c in 0..4 {
                ranks[c] += sum32[c] as u32;
            }
        }

        ranks
    }
}

// New: Mask in SIMD, to avoid overcounting
pub struct SimdCount8;
impl CountFn<8> for SimdCount8 {
    /// 0: exact
    const S: usize = 0;
    const FIXED: bool = true;
    #[inline(always)]
    fn count(data: &[u8; 8], pos: usize) -> Ranks {
        let mut ranks = [0; 4];
        {
            use std::mem::transmute as t;

            let chunk = u64::from_le_bytes((*data).try_into().unwrap());
            let mask = if pos == 32 {
                u64::MAX
            } else {
                (1u64 << (2 * pos)) - 1
            };
            // chunk &= mask;

            // count AC in first half, GT in second half.
            let simd = u64x4::splat(chunk);
            let zero = u8x32::splat(0);
            let mask_f: u64x4 = unsafe { t(u8x32::splat(0x0f)) };
            // bits of the 4 chars
            // 00 | 01 | 10 | 11  (0, 1, 2, 3)
            const C: u64x4 = u64x4::from_array(unsafe {
                t([[!0u8; 8], [!0x55u8; 8], [!0xAAu8; 8], [!0xFFu8; 8]])
            });

            let x = (simd ^ C) & u64x4::splat(mask);
            let y = x;

            let byte_counts = u8x32::from_array([
                // +1 for 11 in the low half
                // +1 for 11 in the high half
                0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 2, //
                0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 2,
            ]);

            // Now reduce.
            // no need for mask_f here.
            // Those are needed to get rid of possible 1 high bits that mask the value to 0,
            // but we already know those aren't 0 anyway in our case.
            let lo = y & mask_f;
            let hi = (y >> 4) & mask_f;
            let popcnt1: u8x32 = unsafe { t(_mm256_shuffle_epi8(t(byte_counts), t(lo))) };
            let popcnt2: u8x32 = unsafe { t(_mm256_shuffle_epi8(t(byte_counts), t(hi))) };
            let sum4 = popcnt1 + popcnt2;
            // Accumulate the 8 bytes in each u64 and write them to the low 16 bits.
            let sum32: u64x4 = unsafe { t(_mm256_sad_epu8(t(sum4), t(zero))) };
            for c in 0..4 {
                ranks[c] += sum32[c] as u32;
            }
        }

        ranks
    }
    #[inline(always)]
    fn count_right(data: &[u8; 8], pos: usize) -> Ranks {
        let mut ranks = [0; 4];
        {
            use std::mem::transmute as t;

            // Count one u64 quarter of bits.
            let chunk = u64::from_le_bytes((*data).try_into().unwrap());
            let mask = if pos == 32 {
                0
            } else {
                !((1u64 << (2 * pos)) - 1)
            };
            // chunk &= mask;

            // count AC in first half, GT in second half.
            let simd = u64x4::splat(chunk);
            let zero = u8x32::splat(0);
            let mask_f: u64x4 = unsafe { t(u8x32::splat(0x0f)) };
            // bits of the 4 chars
            // 00 | 01 | 10 | 11  (0, 1, 2, 3)
            const C: u64x4 = u64x4::from_array(unsafe {
                t([[!0u8; 8], [!0x55u8; 8], [!0xAAu8; 8], [!0xFFu8; 8]])
            });

            let x = (simd ^ C) & u64x4::splat(mask);
            let y = x;

            let byte_counts = u8x32::from_array([
                // +1 for 11 in the low half
                // +1 for 11 in the high half
                0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 2, //
                0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 2,
            ]);

            // Now reduce.
            // no need for mask_f here.
            // Those are needed to get rid of possible 1 high bits that mask the value to 0,
            // but we already know those aren't 0 anyway in our case.
            let lo = y & mask_f;
            let hi = (y >> 4) & mask_f;
            let popcnt1: u8x32 = unsafe { t(_mm256_shuffle_epi8(t(byte_counts), t(lo))) };
            let popcnt2: u8x32 = unsafe { t(_mm256_shuffle_epi8(t(byte_counts), t(hi))) };
            let sum4 = popcnt1 + popcnt2;
            // Accumulate the 8 bytes in each u64 and write them to the low 16 bits.
            let sum32: u64x4 = unsafe { t(_mm256_sad_epu8(t(sum4), t(zero))) };
            for c in 0..4 {
                ranks[c] += sum32[c] as u32;
            }
        }

        ranks
    }
    /// Pos can twice the size here.
    /// If first half, count top elements, otherwise count bottom elements.
    #[inline(always)]
    fn count_mid(data: &[u8; 8], pos: usize) -> Ranks {
        let mut ranks = [0; 4];
        {
            use std::mem::transmute as t;

            // Count one u64 quarter of bits.
            let chunk = u64::from_le_bytes((*data).try_into().unwrap());
            let mask = MID_MASKS[pos];
            // chunk &= mask;

            // count AC in first half, GT in second half.
            let simd = u64x4::splat(chunk);
            let zero = u8x32::splat(0);
            let mask_f: u64x4 = unsafe { t(u8x32::splat(0x0f)) };
            // bits of the 4 chars
            // 00 | 01 | 10 | 11  (0, 1, 2, 3)
            const C: u64x4 = u64x4::from_array(unsafe {
                t([[!0u8; 8], [!0x55u8; 8], [!0xAAu8; 8], [!0xFFu8; 8]])
            });

            let x = (simd ^ C) & u64x4::splat(mask);
            let y = x;

            let byte_counts = u8x32::from_array([
                // +1 for 11 in the low half
                // +1 for 11 in the high half
                0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 2, //
                0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 2,
            ]);

            // Now reduce.
            // no need for mask_f here.
            // Those are needed to get rid of possible 1 high bits that mask the value to 0,
            // but we already know those aren't 0 anyway in our case.
            let lo = y & mask_f;
            let hi = (y >> 4) & mask_f;
            let popcnt1: u8x32 = unsafe { t(_mm256_shuffle_epi8(t(byte_counts), t(lo))) };
            let popcnt2: u8x32 = unsafe { t(_mm256_shuffle_epi8(t(byte_counts), t(hi))) };
            let sum4 = popcnt1 + popcnt2;
            // Accumulate the 8 bytes in each u64 and write them to the low 16 bits.
            let sum32: u64x4 = unsafe { t(_mm256_sad_epu8(t(sum4), t(zero))) };
            for c in 0..4 {
                ranks[c] += sum32[c] as u32;
            }
        }

        ranks
    }
}

// New: Masked is looked up in MASKS.
pub struct SimdCount9;
impl CountFn<8> for SimdCount9 {
    /// 0: exact
    const S: usize = 0;
    const FIXED: bool = true;
    #[inline(always)]
    fn count(data: &[u8; 8], pos: usize) -> Ranks {
        let mut ranks = [0; 4];
        {
            use std::mem::transmute as t;

            let chunk = u64::from_le_bytes((*data).try_into().unwrap());
            let mask = MASKS[pos];
            // chunk &= mask;

            // count AC in first half, GT in second half.
            let simd = u64x4::splat(chunk);
            let zero = u8x32::splat(0);
            let mask_f: u64x4 = unsafe { t(u8x32::splat(0x0f)) };
            // bits of the 4 chars
            // 00 | 01 | 10 | 11  (0, 1, 2, 3)
            const C: u64x4 = u64x4::from_array(unsafe {
                t([[!0u8; 8], [!0x55u8; 8], [!0xAAu8; 8], [!0xFFu8; 8]])
            });

            let x = (simd ^ C) & u64x4::splat(mask);
            let y = x;

            let byte_counts = u8x32::from_array([
                // +1 for 11 in the low half
                // +1 for 11 in the high half
                0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 2, //
                0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 2,
            ]);

            // Now reduce.
            // no need for mask_f here.
            // Those are needed to get rid of possible 1 high bits that mask the value to 0,
            // but we already know those aren't 0 anyway in our case.
            let lo = y & mask_f;
            let hi = (y >> 4) & mask_f;
            let popcnt1: u8x32 = unsafe { t(_mm256_shuffle_epi8(t(byte_counts), t(lo))) };
            let popcnt2: u8x32 = unsafe { t(_mm256_shuffle_epi8(t(byte_counts), t(hi))) };
            let sum4 = popcnt1 + popcnt2;
            // Accumulate the 8 bytes in each u64 and write them to the low 16 bits.
            let sum32: u64x4 = unsafe { t(_mm256_sad_epu8(t(sum4), t(zero))) };
            for c in 0..4 {
                ranks[c] += sum32[c] as u32;
            }
        }

        ranks
    }
    #[inline(always)]
    fn count_right(data: &[u8; 8], pos: usize) -> Ranks {
        let mut ranks = [0; 4];
        {
            use std::mem::transmute as t;

            // Count one u64 quarter of bits.
            let chunk = u64::from_le_bytes((*data).try_into().unwrap());
            let mask = !MASKS[pos];
            // chunk &= mask;

            // count AC in first half, GT in second half.
            let simd = u64x4::splat(chunk);
            let zero = u8x32::splat(0);
            let mask_f: u64x4 = unsafe { t(u8x32::splat(0x0f)) };
            // bits of the 4 chars
            // 00 | 01 | 10 | 11  (0, 1, 2, 3)
            const C: u64x4 = u64x4::from_array(unsafe {
                t([[!0u8; 8], [!0x55u8; 8], [!0xAAu8; 8], [!0xFFu8; 8]])
            });

            let x = (simd ^ C) & u64x4::splat(mask);
            let y = x;

            let byte_counts = u8x32::from_array([
                // +1 for 11 in the low half
                // +1 for 11 in the high half
                0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 2, //
                0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 2,
            ]);

            // Now reduce.
            // no need for mask_f here.
            // Those are needed to get rid of possible 1 high bits that mask the value to 0,
            // but we already know those aren't 0 anyway in our case.
            let lo = y & mask_f;
            let hi = (y >> 4) & mask_f;
            let popcnt1: u8x32 = unsafe { t(_mm256_shuffle_epi8(t(byte_counts), t(lo))) };
            let popcnt2: u8x32 = unsafe { t(_mm256_shuffle_epi8(t(byte_counts), t(hi))) };
            let sum4 = popcnt1 + popcnt2;
            // Accumulate the 8 bytes in each u64 and write them to the low 16 bits.
            let sum32: u64x4 = unsafe { t(_mm256_sad_epu8(t(sum4), t(zero))) };
            for c in 0..4 {
                ranks[c] += sum32[c] as u32;
            }
        }

        ranks
    }
    /// Pos can twice the size here.
    /// If first half, count top elements, otherwise count bottom elements.
    #[inline(always)]
    fn count_mid(data: &[u8; 8], pos: usize) -> Ranks {
        let mut ranks = [0; 4];
        {
            use std::mem::transmute as t;

            // Count one u64 quarter of bits.
            let chunk = u64::from_le_bytes((*data).try_into().unwrap());
            let mask = MID_MASKS[pos];
            // chunk &= mask;

            // count AC in first half, GT in second half.
            let simd = u64x4::splat(chunk);
            let zero = u8x32::splat(0);
            let mask_f: u64x4 = unsafe { t(u8x32::splat(0x0f)) };
            // bits of the 4 chars
            // 00 | 01 | 10 | 11  (0, 1, 2, 3)
            const C: u64x4 = u64x4::from_array(unsafe {
                t([[!0u8; 8], [!0x55u8; 8], [!0xAAu8; 8], [!0xFFu8; 8]])
            });

            let x = (simd ^ C) & u64x4::splat(mask);
            let y = x;

            let byte_counts = u8x32::from_array([
                // +1 for 11 in the low half
                // +1 for 11 in the high half
                0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 2, //
                0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 2,
            ]);

            // Now reduce.
            // no need for mask_f here.
            // Those are needed to get rid of possible 1 high bits that mask the value to 0,
            // but we already know those aren't 0 anyway in our case.
            let lo = y & mask_f;
            let hi = (y >> 4) & mask_f;
            let popcnt1: u8x32 = unsafe { t(_mm256_shuffle_epi8(t(byte_counts), t(lo))) };
            let popcnt2: u8x32 = unsafe { t(_mm256_shuffle_epi8(t(byte_counts), t(hi))) };
            let sum4 = popcnt1 + popcnt2;
            // Accumulate the 8 bytes in each u64 and write them to the low 16 bits.
            let sum32: u64x4 = unsafe { t(_mm256_sad_epu8(t(sum4), t(zero))) };
            for c in 0..4 {
                ranks[c] += sum32[c] as u32;
            }
        }

        ranks
    }
}

/// Wide variant of SimdCount3.
/// This interleaves masks for the two u64 halves.
pub struct WideSimdCount2;
impl CountFn<16> for WideSimdCount2 {
    const S: usize = 16;
    const FIXED: bool = true;
    #[inline(always)]
    fn count(data: &[u8; 16], pos: usize) -> Ranks {
        let mut ranks = [0; 4];
        {
            use std::mem::transmute as t;

            // Count one u64 quarter of bits.
            let mut chunk = u128::from_le_bytes((*data).try_into().unwrap());
            let mask = WIDE_MASKS[pos];
            chunk &= mask;
            let [chunk0, chunk1]: [u64; 2] = unsafe { t(chunk) };

            // count AC in first half, GT in second half.
            let simd0 = u64x4::splat(chunk0);
            let simd1 = u64x4::splat(chunk1);
            let zero = u8x32::splat(0);
            let mask5: u64x4 = unsafe { t(u8x32::splat(0x55)) };
            let mask_a: u64x4 = unsafe { t(u8x32::splat(0xaa)) };
            let mask_f: u64x4 = unsafe { t(u8x32::splat(0x0f)) };
            // bits of the 4 chars
            // 00 | 01 | 10 | 11  (0, 1, 2, 3)
            const C: u64x4 = u64x4::from_array(unsafe {
                t([[!0u8; 8], [!0x55u8; 8], [!0xAAu8; 8], [!0xFFu8; 8]])
            });

            let x0 = simd0 ^ C;
            let y0 = (x0 & (x0 >> 1)) & mask5;
            // New: make the second mask in the high position of each pair.
            let x1 = simd1 ^ C;
            let y1 = (x1 & (x1 << 1)) & mask_a;
            // New: interleave the two masks.
            let y = y0 | y1;

            let byte_counts = u8x32::from_array([
                // popcount(0..16)
                0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, //
                0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
            ]);

            // Now reduce.
            let y: u64x4 = unsafe { t(y) };
            // Note: we do need &mask_f here, since high bits could be set.
            let lo = y & mask_f;
            let hi = (y >> 4) & mask_f;
            let popcnt1: u8x32 = unsafe { t(_mm256_shuffle_epi8(t(byte_counts), t(lo))) };
            let popcnt2: u8x32 = unsafe { t(_mm256_shuffle_epi8(t(byte_counts), t(hi))) };
            let sum4 = popcnt1 + popcnt2;
            // Accumulate the 8 bytes in each u64 and write them to the low 16 bits.
            let sum32: u64x4 = unsafe { t(_mm256_sad_epu8(t(sum4), t(zero))) };
            for c in 0..4 {
                ranks[c] += sum32[c] as u32;
            }
        }

        ranks
    }
    #[inline(always)]
    fn count_right(data: &[u8; 16], pos: usize) -> Ranks {
        let mut ranks = [0; 4];
        {
            use std::mem::transmute as t;

            // Count one u64 quarter of bits.
            let mut chunk = u128::from_le_bytes((*data).try_into().unwrap());
            let mask = WIDE_MASKS[64 + pos];
            chunk &= mask;
            let [chunk0, chunk1]: [u64; 2] = unsafe { t(chunk) };

            // count AC in first half, GT in second half.
            let simd0 = u64x4::splat(chunk0);
            let simd1 = u64x4::splat(chunk1);
            let zero = u8x32::splat(0);
            let mask5: u64x4 = unsafe { t(u8x32::splat(0x55)) };
            let mask_a: u64x4 = unsafe { t(u8x32::splat(0xaa)) };
            let mask_f: u64x4 = unsafe { t(u8x32::splat(0x0f)) };
            // bits of the 4 chars
            // 00 | 01 | 10 | 11  (0, 1, 2, 3)
            const C: u64x4 = u64x4::from_array(unsafe {
                t([[!0u8; 8], [!0x55u8; 8], [!0xAAu8; 8], [!0xFFu8; 8]])
            });

            let x0 = simd0 ^ C;
            let y0 = (x0 & (x0 >> 1)) & mask5;
            // New: make the second mask in the high position of each pair.
            let x1 = simd1 ^ C;
            let y1 = (x1 & (x1 << 1)) & mask_a;
            // New: interleave the two masks.
            let y = y0 | y1;

            let byte_counts = u8x32::from_array([
                // popcount(0..16)
                0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, //
                0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
            ]);

            // Now reduce.
            let y: u64x4 = unsafe { t(y) };
            // Note: we do need &mask_f here, since high bits could be set.
            let lo = y & mask_f;
            let hi = (y >> 4) & mask_f;
            let popcnt1: u8x32 = unsafe { t(_mm256_shuffle_epi8(t(byte_counts), t(lo))) };
            let popcnt2: u8x32 = unsafe { t(_mm256_shuffle_epi8(t(byte_counts), t(hi))) };
            let sum4 = popcnt1 + popcnt2;
            // Accumulate the 8 bytes in each u64 and write them to the low 16 bits.
            let sum32: u64x4 = unsafe { t(_mm256_sad_epu8(t(sum4), t(zero))) };
            for c in 0..4 {
                ranks[c] += sum32[c] as u32;
            }
        }

        ranks
    }
}

/// Counts a slice a u128 at a time.
pub struct SimdCountSlice;
impl<const B: usize> CountFn<B> for SimdCountSlice {
    const S: usize = 16;
    const FIXED: bool = false;
    #[inline(always)]
    fn count(data: &[u8; B], pos: usize) -> Ranks {
        assert!(B % 16 == 0);
        let mut ranks = [0; 4];
        for idx in (0..pos.div_ceil(4)).step_by(16) {
            let chunk = &data[idx..idx + 16];
            let chunk_ranks = <WideSimdCount as CountFn<16>>::count(
                chunk.try_into().unwrap(),
                (pos - idx * 4).min(64),
            );
            for c in 0..4 {
                ranks[c] += chunk_ranks[c];
            }
        }
        ranks
    }
}
