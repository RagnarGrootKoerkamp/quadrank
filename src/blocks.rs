#![allow(non_camel_case_types)]

use std::{arch::x86_64::_mm_sign_epi32, array::from_fn, hint::assert_unchecked, simd::u32x4};

use crate::{
    Ranks, add,
    count::{count_u8x8, count_u8x16, count_u64_mask, count_u64_mid_mask},
    count4::{BINARY_MID_MASKS, BINARY_MID_MASKS256, CountFn, WideSimdCount2, count4_u8x8},
    ranker::BasicBlock,
};

#[inline(always)]
fn extra_counted<const B: usize, C: CountFn<B>>(pos: usize) -> u32 {
    if C::S == 0 {
        return 0;
    }
    let ans = (if C::FIXED {
        (C::S * 4) - pos % (C::S * 4)
    } else {
        -(pos as isize) as usize % (C::S * 4)
    }) as u32;
    ans
}

/// A full 512bit cacheline block that does not store any counts itself.
#[repr(align(64))]
#[derive(mem_dbg::MemSize)]
pub struct Plain512 {
    seq: [u8; 64],
}

impl BasicBlock for Plain512 {
    const X: usize = 2; // DNA
    const B: usize = 64; // Bytes of characters in block.
    const N: usize = 256; // Number of characters in block.
    const C: usize = 64; // Bytes of the underlying count function.
    const W: usize = 0;
    const TRANSPOSED: bool = false;

    fn new(_ranks: Ranks, data: &[u8; Self::B]) -> Self {
        Plain512 { seq: *data }
    }

    #[inline(always)]
    fn count<C: CountFn<{ Self::C }, TRANSPOSED = { Self::TRANSPOSED }>, const C3: bool>(
        &self,
        pos: usize,
    ) -> Ranks {
        let mut ranks = C::count(&self.seq, pos);
        if C3 {
            ranks[0] = pos as u32 - ranks[1] - ranks[2] - ranks[3];
        } else {
            ranks[0] -= extra_counted::<_, C>(pos);
        }
        ranks
    }
}

#[repr(align(32))]
#[derive(mem_dbg::MemSize)]
pub struct Plain256 {
    seq: [u8; 32],
}

impl BasicBlock for Plain256 {
    const X: usize = 2; // DNA
    const B: usize = 32; // Bytes of characters in block.
    const N: usize = 128; // Number of characters in block.
    const C: usize = 32; // Bytes of the underlying count function.
    const W: usize = 0;
    const TRANSPOSED: bool = false;

    fn new(_ranks: Ranks, data: &[u8; Self::B]) -> Self {
        Self { seq: *data }
    }

    #[inline(always)]
    fn count<C: CountFn<{ Self::C }>, const C3: bool>(&self, pos: usize) -> Ranks {
        let mut ranks = C::count(&self.seq, pos);
        if C3 {
            ranks[0] = pos as u32 - ranks[1] - ranks[2] - ranks[3];
        } else {
            ranks[0] -= extra_counted::<_, C>(pos);
        }
        ranks
    }
}

#[repr(align(16))]
#[derive(mem_dbg::MemSize)]
pub struct Plain128 {
    seq: [u8; 16],
}

impl BasicBlock for Plain128 {
    const X: usize = 2; // DNA
    const B: usize = 16; // Bytes of characters in block.
    const N: usize = 64; // Number of characters in block.
    const C: usize = 16; // Bytes of the underlying count function.
    const W: usize = 0;
    const TRANSPOSED: bool = false;

    fn new(_ranks: Ranks, data: &[u8; Self::B]) -> Self {
        Self { seq: *data }
    }

    #[inline(always)]
    fn count<C: CountFn<{ Self::C }>, const C3: bool>(&self, pos: usize) -> Ranks {
        let mut ranks = C::count(&self.seq, pos);
        if C3 {
            ranks[0] = pos as u32 - ranks[1] - ranks[2] - ranks[3];
        } else {
            ranks[0] -= extra_counted::<_, C>(pos);
        }
        ranks
    }
}

/// For each 128bp, store:
/// - 4 u64 counts, for 256bits total
/// - 256 bits of packed sequence.
/// In total, exactly covers a 512bit cache line.
///
/// Based on BWA: https://github.com/lh3/bwa/blob/master/bwtindex.c#L150
#[repr(align(64))]
#[derive(mem_dbg::MemSize)]
pub struct FullBlock {
    // 4*64 = 256 bit counts
    ranks: [u64; 4],
    // 4*64 = 32*8 = 256 bit packed sequence
    seq: [u8; 32],
}

impl BasicBlock for FullBlock {
    const X: usize = 2; // DNA
    const B: usize = 32; // Bytes of characters in block.
    const N: usize = 128; // Number of characters in block.
    const C: usize = 32; // Bytes of the underlying count function.
    const W: usize = 64;
    const TRANSPOSED: bool = false;

    fn new(ranks: Ranks, data: &[u8; Self::B]) -> Self {
        FullBlock {
            ranks: ranks.map(|x| x as u64),
            seq: *data,
        }
    }

    #[inline(always)]
    fn count<C: CountFn<{ Self::C }>, const C3: bool>(&self, pos: usize) -> Ranks {
        let mut ranks = [0; 4];
        for c in 0..4 {
            ranks[c] += self.ranks[c] as u32;
        }

        let inner_counts = C::count(&self.seq, pos);
        for c in 0..4 {
            ranks[c] += inner_counts[c];
        }

        if C3 {
            ranks[0] = pos as u32 - ranks[1] - ranks[2] - ranks[3];
        } else {
            ranks[0] -= extra_counted::<_, C>(pos);
        }
        ranks
    }
}

/// For each 128bp, store:
/// - 4 u64 counts *to the middle, after 64bp*, for 256bits total
/// - 256 bits of packed sequence.
/// In total, exactly covers a 512bit cache line.
///
/// Based on BWA: https://github.com/lh3/bwa/blob/master/bwtindex.c#L150
#[repr(align(64))]
#[derive(mem_dbg::MemSize)]
pub struct FullBlockMid {
    // 4*64 = 256 bit counts
    ranks: [u64; 4],
    // 4*64 = 32*8 = 256 bit packed sequence
    seq: [u8; 32],
}

impl BasicBlock for FullBlockMid {
    const X: usize = 2; // DNA
    const B: usize = 32; // Bytes of characters in block.
    const N: usize = 128; // Number of characters in block.
    const C: usize = 16; // Bytes of the underlying count function.
    const W: usize = 64;
    const TRANSPOSED: bool = false;

    fn new(mut ranks: Ranks, data: &[u8; Self::B]) -> Self {
        for chunk in &data.as_chunks::<8>().0[0..2] {
            for c in 0..4 {
                ranks[c as usize] += count_u8x8(chunk, c) as u32;
            }
        }
        let sum = ranks[0] + ranks[1] + ranks[2] + ranks[3];
        assert!(sum as usize % Self::N == Self::N / 2);
        Self {
            ranks: ranks.map(|x| x as u64),
            seq: *data,
        }
    }

    #[inline(always)]
    fn count<C: CountFn<{ Self::C }>, const C3: bool>(&self, pos: usize) -> Ranks {
        let mut ranks = [0; 4];
        for c in 0..4 {
            ranks[c] += self.ranks[c] as u32;
        }

        if pos < 64 {
            let inner_counts = C::count_right(&self.seq[0..16].try_into().unwrap(), pos);
            if !C3 {
                // ranks[0] += pos as u32 % 64;
                let e = extra_counted::<_, C>(pos);
                let f = (C::S as u32 * 4 - e) % (C::S as u32 * 4);
                ranks[0] += f;
            }
            for c in 0..4 {
                ranks[c] -= inner_counts[c];
            }
        } else {
            let inner_counts = C::count(&self.seq[16..32].try_into().unwrap(), pos % 64);
            for c in 0..4 {
                ranks[c] += inner_counts[c];
            }
            if !C3 {
                ranks[0] -= extra_counted::<_, C>(pos);
            }
        }

        if C3 {
            ranks[0] = pos as u32 - ranks[1] - ranks[2] - ranks[3];
        }
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
#[repr(align(64))]
#[derive(mem_dbg::MemSize)]
pub struct HalfBlock {
    // meta[0,1,2,3] = 0
    // meta[4,5,6,7] = count of c=0,1,2,3 in first half (64bp=128bit).
    meta: [u8; 8],
    // counts for c=1,2,3
    ranks: [u64; 3],
    // 4*64 = 32*8 = 256 bit packed sequence
    seq: [u8; 32],
}

impl BasicBlock for HalfBlock {
    const X: usize = 2; // DNA
    const B: usize = 32;
    const N: usize = 128;
    const C: usize = 16;
    const W: usize = 32;
    const TRANSPOSED: bool = false;

    fn new(ranks: Ranks, data: &[u8; Self::B]) -> Self {
        let mut meta = [0; 8];
        // count first half.
        for chunk in &data.as_chunks::<8>().0[0..2] {
            for c in 0..4 {
                meta[4 + c as usize] += count_u8x8(chunk, c) as u8;
            }
        }
        HalfBlock {
            meta,
            ranks: [ranks[1] as u64, ranks[2] as u64, ranks[3] as u64],
            seq: *data,
        }
    }

    #[inline(always)]
    fn count<C: CountFn<16>, const C3: bool>(&self, pos: usize) -> Ranks {
        let mut ranks = [0; 4];

        // 0 or 1 for left or right half
        let half = pos / 64;
        let half_pos = pos % 64;

        // Offset of chunk and half.
        for c in 0..4 {
            ranks[c] += self.ranks[c - 1] as u32 + self.meta[4 * half + c] as u32;
        }

        let idx = half * 16;
        let inner_counts = C::count(&self.seq[idx..idx + 16].try_into().unwrap(), half_pos);
        for c in 0..4 {
            ranks[c] += inner_counts[c];
        }

        assert!(C3);
        ranks[0] = pos as u32 - ranks[1] - ranks[2] - ranks[3];
        ranks
    }
}

/// For each 128bp, store:
/// - 4 u64 offsets to the start of the block
/// - 4 u64 offsets to halfway the block
/// - 256 bits of packed sequence.
/// In total, exactly covers a 512bit cache line.
///
/// This only has to count chars in 64bp=128bits, which becomes a single popcount.
#[repr(align(64))]
#[derive(mem_dbg::MemSize)]
pub struct HalfBlock2 {
    // 32bit counts for the first and second half.
    ranks: [[u32; 4]; 2],
    // 4*64 = 32*8 = 256 bit packed sequence
    seq: [u8; 32],
}

impl BasicBlock for HalfBlock2 {
    const X: usize = 2; // DNA
    const B: usize = 32;
    const N: usize = 128;
    const C: usize = 16;
    const W: usize = 32;
    const TRANSPOSED: bool = false;

    fn new(ranks: Ranks, data: &[u8; Self::B]) -> Self {
        let mut half_ranks = ranks;
        // count first half.
        for chunk in &data.as_chunks::<8>().0[0..2] {
            for c in 0..4 {
                half_ranks[c as usize] += count_u8x8(chunk, c) as u32;
            }
        }
        HalfBlock2 {
            ranks: [ranks, half_ranks],
            seq: *data,
        }
    }

    #[inline(always)]
    fn count<C: CountFn<16>, const C3: bool>(&self, pos: usize) -> Ranks {
        let mut ranks = [0; 4];

        // 0 or 1 for left or right half
        let half = pos / 64;
        let half_pos = pos % 64;

        let idx = half * 16;
        let inner_counts = C::count(&self.seq[idx..idx + 16].try_into().unwrap(), half_pos);

        if C3 {
            ranks[0] = half_pos as u32 - ranks[1] - ranks[2] - ranks[3];
        } else {
            ranks[0] -= extra_counted::<_, C>(pos);
        }

        for c in 0..4 {
            ranks[c] += inner_counts[c];
        }
        for c in 0..4 {
            ranks[c] += self.ranks[half][c];
        }

        ranks
    }
}

/// u32 global ranks, and 8bit ranks for each u64 quart.
#[repr(align(64))]
#[derive(mem_dbg::MemSize)]
pub struct QuartBlock {
    /// 32bit counts for the entire block
    ranks: [u32; 4],
    /// Each u32 is equivalent to [u8; 4] with counts from start to each u64 quart.
    part_ranks: [u32; 4],
    // 4*64 = 32*8 = 256 bit packed sequence
    seq: [u8; 32],
}

impl BasicBlock for QuartBlock {
    const X: usize = 2; // DNA
    const B: usize = 32;
    const N: usize = 128;
    const C: usize = 8;
    const W: usize = 32;
    const TRANSPOSED: bool = false;

    fn new(ranks: Ranks, data: &[u8; Self::B]) -> Self {
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
        QuartBlock {
            ranks,
            part_ranks,
            seq: *data,
        }
    }

    #[inline(always)]
    fn count<C: CountFn<8>, const C3: bool>(&self, pos: usize) -> Ranks {
        let mut ranks = [0; 4];

        let quart = pos / 32;
        let quart_pos = pos % 32;

        let idx = quart * 8;
        let chunk = &self.seq[idx..idx + 8].try_into().unwrap();
        let inner_counts = C::count(chunk, quart_pos);
        for c in 0..4 {
            ranks[c] += inner_counts[c];
        }

        // if C3 {
        //     ranks[0] = quart_pos as u32 - ranks[1] - ranks[2] - ranks[3];
        // } else {
        //     ranks[0] -= extra_counted::<_, C>(pos);
        // }

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

/// u16 global ranks, and 8bit ranks for 4 of the 5 u64 parts.
#[repr(align(64))]
#[derive(mem_dbg::MemSize)]
pub struct PentaBlock {
    /// 16bit counts for the global offset
    ranks: [u16; 4],
    /// Each u32 is equivalent to [u8; 4] with counts from start to 4 of 5 u64 parts.
    part_ranks: [u32; 4],
    // u64x5 = u8x40 = 320 bit packed sequence
    seq: [u8; 40],
}

impl BasicBlock for PentaBlock {
    const X: usize = 2; // DNA
    const B: usize = 40;
    const N: usize = 160;
    const C: usize = 8;
    const W: usize = 16;
    const TRANSPOSED: bool = false;

    fn new(ranks: Ranks, data: &[u8; Self::B]) -> Self {
        let mut part_ranks = [0u32; 4];
        let mut block_ranks = [0u32; 4];
        // count each part half.
        for (i, chunk) in data.as_chunks::<8>().0.iter().enumerate() {
            for c in 0..4 {
                block_ranks[c as usize] += count_u8x8(chunk, c);
            }
            for c in 0..4 {
                if i < 4 {
                    part_ranks[c] |= block_ranks[c].unbounded_shl(24 - 8 * i as u32);
                }
            }
        }
        PentaBlock {
            ranks: ranks.map(|x| x as u16),
            part_ranks,
            seq: *data,
        }
    }

    #[inline(always)]
    fn count<C: CountFn<8>, const C3: bool>(&self, pos: usize) -> Ranks {
        let mut ranks = [0; 4];

        let pent = pos / 32;
        let pent_pos = pos % 32;

        let idx = pent * 8;
        let inner_counts = C::count(&self.seq[idx..idx + 8].try_into().unwrap(), pent_pos);
        for c in 0..4 {
            ranks[c] += inner_counts[c];
        }

        if C3 {
            ranks[0] = pent_pos as u32 - ranks[1] - ranks[2] - ranks[3];
        } else {
            ranks[0] -= extra_counted::<_, C>(pos);
        }

        for c in 0..4 {
            ranks[c] += self.ranks[c] as u32;
        }
        for c in 0..4 {
            ranks[c] += (self.part_ranks[c].unbounded_shr(32 - 8 * pent as u32)) & 0xff;
        }

        ranks
    }
}

/// As PentaBlock, but part_ranks are 'inlined' with ranks
#[repr(align(64))]
#[derive(mem_dbg::MemSize)]
pub struct PentaBlock20bit {
    /// for each char, a packed u48:
    /// - 20bit count for the global offset (in the low bits)
    /// - u28 = [u7;2] cumulative counts for first 4 of 5 u64 parts.
    ranks: [u8; 24],
    // u64x5 = u8x40 = 320 bit packed sequence
    seq: [u8; 40],
}

impl BasicBlock for PentaBlock20bit {
    const X: usize = 2; // DNA
    const B: usize = 40;
    const N: usize = 160;
    const C: usize = 8;
    const W: usize = 20;
    const TRANSPOSED: bool = false;

    fn new(ranks: Ranks, data: &[u8; Self::B]) -> Self {
        let mut part_ranks = [0u32; 4];
        let mut block_ranks = [0u32; 4];
        // count each part half.
        for (i, chunk) in data.as_chunks::<8>().0.iter().enumerate() {
            for c in 0..4 {
                block_ranks[c as usize] += count_u8x8(chunk, c);
            }
            for c in 0..4 {
                if i < 4 {
                    assert!(
                        block_ranks[c] < 128,
                        "part rank overflow c={c} i={i} {} data = {data:?}",
                        block_ranks[c]
                    );
                    part_ranks[c] |= block_ranks[c].unbounded_shl(32 - 7 * (i + 1) as u32);
                }
            }
        }
        let mut packed_ranks = [0u8; 24];
        for c in 0..4 {
            let value = (ranks[c] as u64 & ((1 << 20) - 1)) + ((part_ranks[c] as u64) << 16);
            let bytes = value.to_le_bytes();
            let bytes = &bytes[0..6];
            packed_ranks[c * 6..c * 6 + 6].copy_from_slice(bytes);
        }

        PentaBlock20bit {
            ranks: packed_ranks,
            seq: *data,
        }
    }

    #[inline(always)]
    fn count<C: CountFn<8>, const C3: bool>(&self, pos: usize) -> Ranks {
        let mut ranks = [0; 4];

        let pent = pos / 32;
        let pent_pos = pos % 32;

        let idx = pent * 8;
        let inner_counts = C::count(&self.seq[idx..idx + 8].try_into().unwrap(), pent_pos);
        for c in 0..4 {
            ranks[c] += inner_counts[c];
        }

        if C3 {
            ranks[0] = pent_pos as u32 - ranks[1] - ranks[2] - ranks[3];
        } else {
            ranks[0] -= extra_counted::<_, C>(pos);
        }

        for c in 0..4 {
            let val = u64::from_le_bytes(self.ranks[c * 6..c * 6 + 8].try_into().unwrap());
            ranks[c] += ((val >> 16) & ((1 << 20) - 1)) as u32;
            ranks[c] += (val.unbounded_shr(64 - 7 * pent as u32)) as u32 & 0x7f;
        }

        ranks
    }
}

/// u16 global ranks, and a u16 encoding offsets for 2 of the 3 u128 parts.
#[repr(align(64))]
#[derive(mem_dbg::MemSize)]
pub struct HexaBlock {
    /// 16bit counts for the global offset
    ranks: [u16; 4],
    /// Each u16 is equivalent to [u8; 2] with counts from start to 2 of 3 128 parts.
    part_ranks: [u16; 4],
    // u128x3 = u8x48 = 384 bit packed sequence
    seq: [u8; 48],
}

impl BasicBlock for HexaBlock {
    const X: usize = 2; // DNA
    const B: usize = 48;
    const N: usize = 192;
    const C: usize = 16;
    const W: usize = 16;
    const TRANSPOSED: bool = false;

    fn new(ranks: Ranks, data: &[u8; Self::B]) -> Self {
        let mut part_ranks = [0u16; 4];
        let mut block_ranks = [0u16; 4];
        // count each part half.
        for (i, chunk) in data.as_chunks::<16>().0.iter().enumerate() {
            for c in 0..4 {
                block_ranks[c as usize] += count_u8x16(chunk, c) as u16;
            }
            for c in 0..4 {
                if i < 2 {
                    part_ranks[c] |= block_ranks[c].unbounded_shl(8 - 8 * i as u32);
                }
            }
        }
        HexaBlock {
            ranks: ranks.map(|x| x as u16),
            part_ranks,
            seq: *data,
        }
    }

    #[inline(always)]
    fn count<C: CountFn<16>, const C3: bool>(&self, pos: usize) -> Ranks {
        let mut ranks = [0; 4];

        let hex = pos / 64;
        let hex_pos = pos % 64;

        let idx = hex * 16;
        let inner_counts = C::count(&self.seq[idx..idx + 16].try_into().unwrap(), hex_pos);
        for c in 0..4 {
            ranks[c] += inner_counts[c];
        }

        if C3 {
            ranks[0] = hex_pos as u32 - ranks[1] - ranks[2] - ranks[3];
        } else {
            ranks[0] -= extra_counted::<_, C>(pos);
        }

        for c in 0..4 {
            ranks[c] += self.ranks[c] as u32;
        }
        for c in 0..4 {
            ranks[c] += (self.part_ranks[c].unbounded_shr(16 - 8 * hex as u32)) as u32 & 0xff;
        }

        ranks
    }
}

/// u16 global ranks, and a u16 encoding offsets for 2 of the 3 u128 parts.
#[repr(align(64))]
#[derive(mem_dbg::MemSize)]
pub struct HexaBlock2 {
    /// high half: 16bit counts for the global offset
    /// low half: 2x u8 offset
    ranks: [u32; 4],
    // u128x3 = u8x48 = 384 bit packed sequence
    seq: [u8; 48],
}

impl BasicBlock for HexaBlock2 {
    const X: usize = 2; // DNA
    const B: usize = 48;
    const N: usize = 192;
    const C: usize = 16;
    const W: usize = 16;
    const TRANSPOSED: bool = false;

    fn new(ranks: Ranks, data: &[u8; Self::B]) -> Self {
        let mut part_ranks = [0u16; 4];
        let mut block_ranks = [0u16; 4];
        // count each part half.
        for (i, chunk) in data.as_chunks::<16>().0.iter().enumerate() {
            for c in 0..4 {
                block_ranks[c as usize] += count_u8x16(chunk, c) as u16;
            }
            for c in 0..4 {
                if i < 2 {
                    part_ranks[c] |= block_ranks[c].unbounded_shl(8 - 8 * i as u32);
                }
            }
        }
        HexaBlock2 {
            ranks: from_fn(|c| (ranks[c] << 16) | part_ranks[c] as u32),
            seq: *data,
        }
    }

    #[inline(always)]
    fn count<C: CountFn<16>, const C3: bool>(&self, pos: usize) -> Ranks {
        let mut ranks = [0; 4];

        let hex = pos / 64;
        let hex_pos = pos % 64;

        let idx = hex * 16;
        let inner_counts = C::count(&self.seq[idx..idx + 16].try_into().unwrap(), hex_pos);
        for c in 0..4 {
            ranks[c] += inner_counts[c];
        }

        if C3 {
            ranks[0] = hex_pos as u32 - ranks[1] - ranks[2] - ranks[3];
        } else {
            ranks[0] -= extra_counted::<_, C>(pos);
        }

        for c in 0..4 {
            ranks[c] += self.ranks[c] >> 16;
        }
        for c in 0..4 {
            ranks[c] += (self.ranks[c] & 0xffff) >> (16 - 8 * hex as u32) & 0xff;
            // ranks[c] += (self.part_ranks[c].unbounded_shr(16 - 8 * hex as u32)) as u32 & 0xff;
            // ranks[c] += (self.part_ranks[c].unbounded_shr(16 - 8 * (hex / 2) as u32)) as u32 & 0xff;
        }

        ranks
    }

    #[inline(always)]
    fn count1(&self, pos: usize, c: u8) -> u32 {
        self.count::<WideSimdCount2, false>(pos)[c as usize]
    }
}

/// As HexaBlock, but part_ranks are 'inlined' with ranks
#[repr(align(64))]
#[derive(mem_dbg::MemSize)]
pub struct HexaBlock18bit {
    /// for each char:
    /// - 18bit count for the global offset (in the low bits)
    /// - u14 = [u7;2] cumulative counts for first 2 of 3 u128 parts.
    ranks: [u32; 4],
    // u128x3 = u8x48 = 384 bit packed sequence
    seq: [u8; 48],
}

impl BasicBlock for HexaBlock18bit {
    const X: usize = 2; // DNA
    const B: usize = 48;
    const N: usize = 192;
    const C: usize = 16;
    const W: usize = 18;
    const TRANSPOSED: bool = false;

    fn new(ranks: Ranks, data: &[u8; Self::B]) -> Self {
        for x in ranks {
            assert!(x + (Self::N as u32) < (1 << 18));
        }
        let mut part_ranks = [0u32; 4];
        let mut block_ranks = [0u32; 4];
        // count each part half.
        for (i, chunk) in data.as_chunks::<16>().0.iter().enumerate() {
            for c in 0..4 {
                block_ranks[c as usize] += count_u8x16(chunk, c);
            }
            for c in 0..4 {
                if i < 2 {
                    assert!(block_ranks[c] < 128);
                    part_ranks[c] |= block_ranks[c].unbounded_shl(16 - 7 * (i + 1) as u32);
                }
            }
        }
        HexaBlock18bit {
            ranks: std::array::from_fn(|c| {
                (ranks[c] & ((1 << 18) - 1)) + ((part_ranks[c] as u32) << 16)
            }),
            seq: *data,
        }
    }

    #[inline(always)]
    fn count<C: CountFn<16>, const C3: bool>(&self, pos: usize) -> Ranks {
        let mut ranks = [0; 4];

        let hex = pos / 64;
        let hex_pos = pos % 64;

        let idx = hex * 16;
        let inner_counts = C::count(&self.seq[idx..idx + 16].try_into().unwrap(), hex_pos);
        for c in 0..4 {
            ranks[c] += inner_counts[c];
        }

        if C3 {
            ranks[0] = hex_pos as u32 - ranks[1] - ranks[2] - ranks[3];
        } else {
            ranks[0] -= extra_counted::<_, C>(pos);
        }

        for c in 0..4 {
            // 16+ to move from the high to the low 16 bits.
            ranks[c] += self.ranks[c].unbounded_shr(16 + 16 - 7 * hex as u32) & 0x7f;
        }
        for c in 0..4 {
            ranks[c] += self.ranks[c] & ((1 << 18) - 1);
        }

        ranks
    }
}

/// u16 global ranks, and a u16 encoding offsets for 2 of the 3 u128 parts.
#[repr(align(64))]
#[derive(mem_dbg::MemSize)]
pub struct HexaBlockMid {
    /// high half: 16bit counts for the global offset to position 32 (middle of first u128).
    /// low half: 2x 8bit counts for the first half of 2nd and 3rd 128 parts.
    ranks: [u32; 4],
    // u128x3 = u8x48 = 384 bit packed sequence
    seq: [u8; 48],
}

impl BasicBlock for HexaBlockMid {
    const X: usize = 2; // DNA
    const B: usize = 48;
    const N: usize = 192;
    const C: usize = 8;
    const W: usize = 16;
    const TRANSPOSED: bool = false;

    fn new(mut ranks: Ranks, data: &[u8; Self::B]) -> Self {
        // Counts before each u64 block.
        let mut bs = [[0u32; 4]; 6];
        // count each part half.
        for (i, chunk) in data.as_chunks::<8>().0.iter().enumerate() {
            bs[i] = add(bs[i], count4_u8x8(*chunk));
        }
        // global ranks include the first u64.
        ranks = add(ranks, bs[0]);
        let p1 = add(bs[1], bs[2]);
        let p2 = add(add(bs[1], bs[2]), add(bs[3], bs[4]));
        let part_ranks: Ranks = from_fn(|c| (p1[c] << 8) | p2[c]);
        Self {
            ranks: from_fn(|c| (ranks[c] << 16) | part_ranks[c]),
            seq: *data,
        }
    }

    #[inline(always)]
    fn count<C: CountFn<8>, const C3: bool>(&self, pos: usize) -> Ranks {
        let mut ranks = [0; 4];

        let hex = pos / 32;
        let hex_pos = pos % 32;

        let idx = hex * 8;

        for c in 0..4 {
            ranks[c] += self.ranks[c] >> 16;
        }

        let inner_counts = C::count_mid(&self.seq[idx..idx + 8].try_into().unwrap(), pos % 64);
        if (pos & 32) == 0 {
            for c in 0..4 {
                ranks[c] = ranks[c].wrapping_sub(inner_counts[c]);
            }
        } else {
            for c in 0..4 {
                ranks[c] += inner_counts[c];
            }
        }

        if C3 {
            ranks[0] = hex_pos as u32 - ranks[1] - ranks[2] - ranks[3];
        } else {
            if C::S != 0 {
                ranks[0] = ranks[0].wrapping_add(((pos as u32 + 32) % 64).wrapping_sub(32));
            }
        }

        for c in 0..4 {
            ranks[c] = ranks[c]
                .wrapping_add(((self.ranks[c] & 0xffff) >> (16 - 8 * (hex / 2) as u32)) & 0xff);
        }
        ranks

        // NOTE: Above could be a lookup:
        // let mut ranks = u32x4::from_array(ranks);
        // let self_ranks = u32x4::from_array(self.ranks);
        // ranks += (self_ranks >> 16) + ((self_ranks & u32x4::splat(0xffff)) >> SHUFFLE[hex])
        //     & u32x4::splat(0xff);
        // ranks.to_array()
    }
}

// New: avoid the if statement to add or subtract.
#[repr(align(64))]
#[derive(mem_dbg::MemSize)]
pub struct HexaBlockMid2 {
    /// high half: 16bit counts for the global offset to position 32 (middle of first u128).
    /// low half: 2x 8bit counts for the first half of 2nd and 3rd 128 parts.
    ranks: [u32; 4],
    // u128x3 = u8x48 = 384 bit packed sequence
    seq: [u8; 48],
}

impl BasicBlock for HexaBlockMid2 {
    const X: usize = 2; // DNA
    const B: usize = 48;
    const N: usize = 192;
    const C: usize = 8;
    const W: usize = 16;
    const TRANSPOSED: bool = false;

    fn new(mut ranks: Ranks, data: &[u8; Self::B]) -> Self {
        // Counts before each u64 block.
        let mut bs = [[0u32; 4]; 6];
        // count each part half.
        for (i, chunk) in data.as_chunks::<8>().0.iter().enumerate() {
            bs[i] = add(bs[i], count4_u8x8(*chunk));
        }
        // global ranks include the first u64.
        ranks = add(ranks, bs[0]);
        let p1 = add(bs[1], bs[2]);
        let p2 = add(add(bs[1], bs[2]), add(bs[3], bs[4]));
        let part_ranks: Ranks = from_fn(|c| (p1[c] << 8) | p2[c]);
        Self {
            ranks: from_fn(|c| (ranks[c] << 16) | part_ranks[c]),
            seq: *data,
        }
    }

    #[inline(always)]
    fn count<C: CountFn<8>, const C3: bool>(&self, pos: usize) -> Ranks {
        assert!(!C3);
        assert!(C::S == 0);
        let mut ranks = u32x4::splat(0);

        let hex = pos / 32;

        let idx = hex * 8;

        let inner_counts = C::count_mid(&self.seq[idx..idx + 8].try_into().unwrap(), pos % 64);

        use std::mem::transmute as t;
        let sign = (pos as u32 % 64).wrapping_sub(32);
        ranks += unsafe { t::<_, u32x4>(_mm_sign_epi32(t(inner_counts), t(u32x4::splat(sign)))) };

        let self_ranks = u32x4::from_array(self.ranks);
        static SHUFFLE: [u32; 8] = [16, 16, 8, 8, 0, 0, 0, 0];
        ranks += u32x4::from_array(self.ranks) >> 16;
        ranks += ((self_ranks & u32x4::splat(0xffff)) >> SHUFFLE[hex]) & u32x4::splat(0xff);
        ranks.to_array()
    }
}

// New: 18bit value towards the **middle** of the entire block
// Then 7bit delta (up to 64) to left and right quarter
// New: shuffle lookup table via shift.
// TODO: Investigate if 6bit delta is enough if we block 1 position.
#[repr(align(64))]
#[derive(mem_dbg::MemSize)]
pub struct HexaBlockMid3 {
    /// high half: 18bit counts for the global offset to position 32 (middle of first u128).
    /// low half: 2x 7bit counts for the first half of 2nd and 3rd 128 parts.
    ranks: [u32; 4],
    // u128x3 = u8x48 = 384 bit packed sequence
    seq: [u8; 48],
}

impl BasicBlock for HexaBlockMid3 {
    const X: usize = 2; // DNA
    const B: usize = 48;
    const N: usize = 192;
    const C: usize = 8;
    const W: usize = 18;
    const TRANSPOSED: bool = false;

    fn new(mut ranks: Ranks, data: &[u8; Self::B]) -> Self {
        // Counts before each u64 block.
        let mut bs = [[0u32; 4]; 6];
        // count each part half.
        for (i, chunk) in data.as_chunks::<8>().0.iter().enumerate() {
            bs[i] = add(bs[i], count4_u8x8(*chunk));
        }
        // global ranks are to the middle
        ranks = add(add(ranks, bs[0]), add(bs[1], bs[2]));
        let p1 = add(bs[1], bs[2]);
        let p2 = add(bs[3], bs[4]);
        let part_ranks: Ranks = from_fn(|c| (p1[c] << 7) | p2[c]);
        Self {
            ranks: from_fn(|c| (ranks[c] << 14) | part_ranks[c]),
            seq: *data,
        }
    }

    #[inline(always)]
    fn count<C: CountFn<8>, const C3: bool>(&self, pos: usize) -> Ranks {
        assert!(!C3);
        assert!(C::S == 0);
        let mut ranks = u32x4::splat(0);

        let hex = pos / 32;

        let idx = hex * 8;

        let inner_counts = C::count_mid(&self.seq[idx..idx + 8].try_into().unwrap(), pos % 64);

        use std::mem::transmute as t;
        let sign = (pos as u32 % 64).wrapping_sub(32);
        ranks += unsafe { t::<_, u32x4>(_mm_sign_epi32(t(inner_counts), t(u32x4::splat(sign)))) };

        let self_ranks = u32x4::from_array(self.ranks);
        ranks += self_ranks >> 14;

        let shuffle = 0x000707u64;
        let parts = self_ranks & u32x4::splat(0x3fff);
        let sign2 = (hex / 2).wrapping_sub(1);
        let shift = ((shuffle >> idx) & 7) as u32;
        ranks += unsafe {
            t::<_, u32x4>(_mm_sign_epi32(
                t((parts >> shift) & u32x4::splat(0x7f)),
                t(u32x4::splat(sign2 as u32)),
            ))
        };
        ranks.to_array()
    }
}

// New: shuffle lookup table via SIMD shift
// TODO: Investigate if 6bit delta is enough if we block 1 position.
#[repr(align(64))]
#[derive(mem_dbg::MemSize)]
pub struct HexaBlockMid4 {
    /// high half: 18bit counts for the global offset to position 32 (middle of first u128).
    /// low half: 2x 7bit counts for the first half of 2nd and 3rd 128 parts.
    ranks: [u32; 4],
    // u128x3 = u8x48 = 384 bit packed sequence
    seq: [u8; 48],
}

impl BasicBlock for HexaBlockMid4 {
    const X: usize = 2; // DNA
    const B: usize = 48;
    const N: usize = 192;
    const C: usize = 8;
    const W: usize = 18;
    const TRANSPOSED: bool = false;

    fn new(mut ranks: Ranks, data: &[u8; Self::B]) -> Self {
        // Counts before each u64 block.
        let mut bs = [[0u32; 4]; 6];
        // count each part half.
        for (i, chunk) in data.as_chunks::<8>().0.iter().enumerate() {
            bs[i] = add(bs[i], count4_u8x8(*chunk));
        }
        // global ranks are to the middle
        ranks = add(add(ranks, bs[0]), add(bs[1], bs[2]));
        let p1 = add(bs[1], bs[2]);
        let p2 = add(bs[3], bs[4]);
        let part_ranks: Ranks = from_fn(|c| (p1[c] << 7) | p2[c]);
        Self {
            ranks: from_fn(|c| (ranks[c] << 14) | part_ranks[c]),
            seq: *data,
        }
    }

    #[inline(always)]
    fn count<C: CountFn<8>, const C3: bool>(&self, pos: usize) -> Ranks {
        assert!(!C3);
        assert!(C::S == 0);
        let mut ranks = u32x4::splat(0);

        let hex = pos / 32;

        let idx = hex * 8;

        let inner_counts = C::count_mid(&self.seq[idx..idx + 8].try_into().unwrap(), pos % 64);

        use std::mem::transmute as t;
        let sign = (pos as u32 % 64).wrapping_sub(32);
        ranks += unsafe { t::<_, u32x4>(_mm_sign_epi32(t(inner_counts), t(u32x4::splat(sign)))) };

        let self_ranks = u32x4::from_array(self.ranks);
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

/// New: Use transposed 16byte count that processes 128 bits at once.
/// Layout: 128bit counts, then 3x a 128bit block.
/// New: store offset to start of 2nd block and delta to start of 3rd block.
#[repr(align(64))]
#[derive(mem_dbg::MemSize)]
pub struct TriBlock {
    /// high 32-8=24 bits: global counts to start of 2nd block.
    /// low 8 bits: counts from start of 2nd block to start of 3rd block.
    /// low half: 2x 8bit counts for the first half of 2nd and 3rd 128 parts.
    /// TODO: 25+7 bits?
    ranks: [u32; 4],
    // u128x3 = u8x48 = 384 bit packed sequence
    seq: [[u8; 16]; 3],
}

impl BasicBlock for TriBlock {
    const X: usize = 2; // DNA
    const B: usize = 48;
    const N: usize = 192;
    const C: usize = 16;
    // TODO: Make this 25
    const W: usize = 24;
    const TRANSPOSED: bool = true;

    fn new(mut ranks: Ranks, data: &[u8; Self::B]) -> Self {
        // Counts before each u64 block.
        let mut bs = [[0u32; 4]; 6];
        // count each part half.
        for (i, chunk) in data.as_chunks::<8>().0.iter().enumerate() {
            bs[i] = add(bs[i], count4_u8x8(*chunk));
        }
        // global ranks are to block
        ranks = add(add(ranks, bs[0]), bs[1]);
        let part_rank = add(add(bs[2], bs[3]), add(bs[4], bs[5]));
        Self {
            ranks: from_fn(|c| (ranks[c] << 8) | part_rank[c]),
            seq: from_fn(|i| unsafe {
                std::mem::transmute(transpose_bits(
                    &data[i * 16..i * 16 + 16].try_into().unwrap(),
                ))
            }),
        }
    }

    #[inline(always)]
    fn count<C: CountFn<16>, const C3: bool>(&self, pos: usize) -> Ranks {
        assert!(!C3);
        assert!(C::S == 0);
        let mut ranks = u32x4::splat(0);

        let tri = pos / 64;

        let inner_counts = C::count_mid(&self.seq[tri].try_into().unwrap(), pos % 128);

        use std::mem::transmute as t;
        let sign = (pos as u32 % 128).wrapping_sub(64);
        ranks += unsafe { t::<_, u32x4>(_mm_sign_epi32(t(inner_counts), t(u32x4::splat(sign)))) };

        let self_ranks = u32x4::from_array(self.ranks);
        ranks += self_ranks >> 8;

        // for tri=0 and tri=1, shift down by 8
        // for tri=2, shift down by 0
        let shift = 8 - (tri as u32 / 2) * 8;
        let parts = self_ranks & u32x4::splat(0x00ff);
        ranks += parts >> u32x4::splat(shift);
        ranks.to_array()
    }
}

/// New: main offset to end, delta to end of first trip
/// That way, the shift-down-by-8 comes out nicer.
#[repr(align(64))]
#[derive(mem_dbg::MemSize)]
pub struct TriBlock2 {
    /// 24 high bits: offset to end
    /// 8 low bits: delta to end of first trip
    ranks: [u32; 4],
    // u128x3 = u8x48 = 384 bit packed sequence
    seq: [[u8; 16]; 3],
}

impl BasicBlock for TriBlock2 {
    const X: usize = 2; // DNA
    const B: usize = 48;
    const N: usize = 192;
    const C: usize = 16;
    // TODO: Make this 25
    const W: usize = 24;
    const TRANSPOSED: bool = true;

    fn new(mut ranks: Ranks, data: &[u8; Self::B]) -> Self {
        // Counts before each u64 block.
        let mut bs = [[0u32; 4]; 6];
        let mut sum = [0u32; 4];
        // count each part half.
        for (i, chunk) in data.as_chunks::<8>().0.iter().enumerate() {
            bs[i] = add(bs[i], count4_u8x8(*chunk));
            sum = add(sum, bs[i]);
        }
        // global ranks are to block
        ranks = add(ranks, sum);
        let part_rank = add(add(bs[2], bs[3]), add(bs[4], bs[5]));
        Self {
            ranks: from_fn(|c| (ranks[c] << 8) | part_rank[c]),
            seq: from_fn(|i| unsafe {
                std::mem::transmute(transpose_bits(
                    &data[i * 16..i * 16 + 16].try_into().unwrap(),
                ))
            }),
        }
    }

    #[inline(always)]
    fn count<C: CountFn<16>, const C3: bool>(&self, pos: usize) -> Ranks {
        assert!(!C3);
        assert!(C::S == 0);
        let mut ranks = u32x4::splat(0);

        let tri = pos / 64;

        let inner_counts = C::count_mid(&self.seq[tri], pos % 128);

        use std::mem::transmute as t;
        let sign = (pos as u32 % 128).wrapping_sub(64);
        ranks += unsafe { t::<_, u32x4>(_mm_sign_epi32(t(inner_counts), t(u32x4::splat(sign)))) };

        let self_ranks = u32x4::from_array(self.ranks);
        ranks += self_ranks >> 8;

        // for tri=0 and tri=1, shift down by 0
        // for tri=2, shift down by 8
        let shift = (tri as u32 / 2) * 8;
        let parts = self_ranks & u32x4::splat(0x00ff);
        ranks += parts >> u32x4::splat(shift);
        ranks.to_array()
    }
}

/// Like TriBlock, but for binary counts.
#[repr(align(64))]
#[derive(mem_dbg::MemSize)]
pub struct BinaryBlock {
    /// offset to end of 1st and 3rd 128bit block.
    /// 8 low bits: delta to end of first trip
    ranks: [u64; 2],
    // u128x3 = u8x48 = 384 bit packed sequence
    seq: [[u64; 2]; 3],
}

impl BasicBlock for BinaryBlock {
    const X: usize = 1; // Binary
    const B: usize = 48;
    const N: usize = 384;
    const C: usize = 16;
    const W: usize = 64;
    const TRANSPOSED: bool = true;

    fn new(ranks: Ranks, data: &[u8; Self::B]) -> Self {
        // Counts in each u64 block.
        let mut bs = [0u64; 6];
        let mut sum = 0u64;
        // count each part half.
        for (i, chunk) in data.as_chunks::<8>().0.iter().enumerate() {
            bs[i] = u64::from_le_bytes(*chunk).count_ones() as u64;
            sum += bs[i];
        }
        // partial ranks after 1 and 3 blocks
        let ranks = [ranks[0] as u64 + bs[0] + bs[1], ranks[0] as u64 + sum];
        Self {
            ranks,
            seq: unsafe { std::mem::transmute(*data) },
        }
    }

    fn count<CF: CountFn<{ Self::C }>, const C3: bool>(&self, _pos: usize) -> Ranks {
        unimplemented!()
    }

    #[inline(always)]
    fn count1(&self, pos: usize, _c: u8) -> u32 {
        let tri = pos / 128;
        let pos = pos % 256;

        let [mask_l, mask_h] = BINARY_MID_MASKS[pos];
        let [l, h] = self.seq[tri];
        let inner_count = (l & mask_l).count_ones() + (h & mask_h).count_ones();
        if pos < 128 {
            self.ranks[tri / 2] as u32 - inner_count
        } else {
            self.ranks[tri / 2] as u32 + inner_count
        }
    }
}

/// Store two 32bit ranks at the end, and put in some more bits.
#[repr(align(64))]
#[derive(mem_dbg::MemSize)]
pub struct BinaryBlock2 {
    // u128x3 + u64 = u8x56 = 448 bit packed sequence
    seq: [u8; 56],
    /// offset to end of 1st and 3rd 128bit block.
    /// 8 low bits: delta to end of first trip
    ranks: [u32; 2],
}

impl BasicBlock for BinaryBlock2 {
    const X: usize = 1; // Binary
    const B: usize = 56;
    const N: usize = 448;
    const C: usize = 16;
    const W: usize = 32;
    const TRANSPOSED: bool = true;

    fn new(ranks: Ranks, data: &[u8; Self::B]) -> Self {
        // Counts in each u64 block.
        let mut bs = [0; 8];
        // count each part half.
        for (i, chunk) in data.as_chunks::<8>().0.iter().enumerate() {
            bs[i] = u64::from_le_bytes(*chunk).count_ones();
        }
        // partial ranks after 1 and 3 blocks
        let ranks = [
            ranks[0] + bs[0] + bs[1],
            ranks[0] + bs[0] + bs[1] + bs[2] + bs[3] + bs[4] + bs[5],
        ];
        Self {
            ranks,
            seq: unsafe { std::mem::transmute(*data) },
        }
    }

    fn count<CF: CountFn<{ Self::C }>, const C3: bool>(&self, _pos: usize) -> Ranks {
        unimplemented!()
    }

    #[inline(always)]
    fn count1(&self, pos: usize, _c: u8) -> u32 {
        unsafe {
            let quad = pos / 128;
            let pos = pos % 256;

            let [mask_l, mask_h] = BINARY_MID_MASKS[pos];
            // NOTE: This *will* go out-of-bounds, but it's ok because 'ranks' is used as padding.
            let l = u64::from_le_bytes(
                self.seq
                    .get_unchecked(quad * 16..quad * 16 + 8)
                    .try_into()
                    .unwrap(),
            );
            // This reads outside `self.seq` and into `self.ranks`.
            let h = (self.seq.as_ptr().add(quad * 16 + 8) as *const u64).read();
            let inner_count = (l & mask_l).count_ones() + (h & mask_h).count_ones();
            if pos < 128 {
                self.ranks[quad / 2] as u32 - inner_count
            } else {
                self.ranks[quad / 2] as u32 + inner_count
            }
        }
    }
}

/// Store 31bit ranks and 9bit delta at the end.
/// #positions: 512-31-9 = 472
#[repr(align(64))]
#[derive(mem_dbg::MemSize)]
pub struct BinaryBlock3 {
    /// last 5 bytes are global rank info
    /// low 9 bits: delta from 1/4 to 3/4
    /// high 31 bits global offset
    seq: [u8; 64],
}

impl BasicBlock for BinaryBlock3 {
    const X: usize = 1; // Binary
    const B: usize = 59;
    const N: usize = 472;
    const C: usize = 16;
    const W: usize = 32;
    const TRANSPOSED: bool = true;

    fn new(ranks: Ranks, data: &[u8; Self::B]) -> Self {
        // Counts in each u64 block.
        let mut bs = [0; 8];
        // count each part half.
        for (i, chunk) in data.as_chunks::<8>().0.iter().enumerate() {
            bs[i] = u64::from_le_bytes(*chunk).count_ones();
        }
        // partial ranks after 1 and 3 blocks
        let delta = bs[2] + bs[3] + bs[4] + bs[5];
        let ranks = ranks[0] + bs[0] + bs[1] + delta;

        let mut seq = [0u8; 64];
        for i in 0..59 {
            seq[i] = data[i];
        }
        let rank_bytes = ((ranks as u64) << 9) | (delta as u64);
        seq[59..64].copy_from_slice(&rank_bytes.to_le_bytes()[0..5]);
        Self { seq }
    }

    fn count<CF: CountFn<{ Self::C }>, const C3: bool>(&self, _pos: usize) -> Ranks {
        unimplemented!()
    }

    #[inline(always)]
    fn count1(&self, pos: usize, _c: u8) -> u32 {
        unsafe {
            let quad = pos / 128;
            let pos = pos % 256;

            let [mask_l, mask_h] = BINARY_MID_MASKS[pos];
            // NOTE: This *will* go out-of-bounds, but it's ok because 'ranks' is used as padding.
            let l = u64::from_le_bytes(
                self.seq
                    .get_unchecked(quad * 16..quad * 16 + 8)
                    .try_into()
                    .unwrap(),
            );
            let h = u64::from_le_bytes(
                self.seq
                    .get_unchecked(quad * 16 + 8..quad * 16 + 16)
                    .try_into()
                    .unwrap(),
            );
            let inner_count = (l & mask_l).count_ones() + (h & mask_h).count_ones();
            let meta = u64::from_le_bytes(self.seq.get_unchecked(56..64).try_into().unwrap()) >> 24;
            let rank = (meta >> 9) as u32;
            let delta = meta as u32 & 0x1ff;
            let delta = delta >> ((quad / 2) * 16);

            if pos < 128 {
                rank - delta - inner_count
            } else {
                rank - delta + inner_count
            }
        }
    }
}

/// Store two 16 bit global ranks at the end.
/// to 1/4 and 3/4
#[repr(align(64))]
#[derive(mem_dbg::MemSize)]
pub struct BinaryBlock4 {
    seq: [u8; 60],
    ranks: [u16; 2],
}

impl BasicBlock for BinaryBlock4 {
    const X: usize = 1; // Binary
    const B: usize = 60;
    const N: usize = 480;
    const C: usize = 16;
    const W: usize = 16;
    const TRANSPOSED: bool = true;

    fn new(ranks: Ranks, data: &[u8; Self::B]) -> Self {
        // Counts in each u64 block.
        let mut bs = [0; 8];
        // count each part half.
        for (i, chunk) in data.as_chunks::<8>().0.iter().enumerate() {
            bs[i] = u64::from_le_bytes(*chunk).count_ones();
        }
        // partial ranks after 1 and 3 blocks
        let rank = ranks[0] + bs[0] + bs[1];
        let delta = bs[2] + bs[3] + bs[4] + bs[5];
        let ranks = [rank as u16, rank as u16 + delta as u16];
        Self { seq: *data, ranks }
    }

    fn count<CF: CountFn<{ Self::C }>, const C3: bool>(&self, _pos: usize) -> Ranks {
        unimplemented!()
    }

    #[inline(always)]
    fn count1(&self, pos: usize, _c: u8) -> u32 {
        unsafe {
            let quad = pos / 128;
            let pos = pos % 256;

            let [mask_l, mask_h] = BINARY_MID_MASKS[pos];
            // NOTE: This *will* go out-of-bounds, but it's ok because 'ranks' is used as padding.
            let l = u64::from_le_bytes(
                self.seq
                    .get_unchecked(quad * 16..quad * 16 + 8)
                    .try_into()
                    .unwrap(),
            );
            // This reads outside `self.seq` and into `self.ranks`.
            let h = (self.seq.as_ptr().add(quad * 16 + 8) as *const u64).read();
            let inner_count = (l & mask_l).count_ones() + (h & mask_h).count_ones();
            let rank = *self.ranks.get_unchecked(quad / 2) as u32;
            if pos < 128 {
                rank - inner_count
            } else {
                rank + inner_count
            }
        }
    }
}

/// Only store a count to the middle and then count 256 bits.
#[repr(align(64))]
#[derive(mem_dbg::MemSize)]
pub struct BinaryBlock5 {
    seq: [u8; 60],
    rank: u32,
}

impl BasicBlock for BinaryBlock5 {
    const X: usize = 1; // Binary
    const B: usize = 60;
    const N: usize = 480;
    const C: usize = 16;
    const W: usize = 32;
    const TRANSPOSED: bool = true;

    fn new(ranks: Ranks, data: &[u8; Self::B]) -> Self {
        // Counts in each u64 block.
        let mut bs = [0; 8];
        // count each part half.
        for (i, chunk) in data.as_chunks::<8>().0.iter().enumerate() {
            bs[i] = u64::from_le_bytes(*chunk).count_ones();
        }
        // partial ranks after 1 and 3 blocks
        let rank = ranks[0] + bs[0] + bs[1] + bs[2] + bs[3];
        Self { seq: *data, rank }
    }

    fn count<CF: CountFn<{ Self::C }>, const C3: bool>(&self, _pos: usize) -> Ranks {
        unimplemented!()
    }

    #[inline(always)]
    fn count1(&self, pos: usize, _c: u8) -> u32 {
        let half = pos / 256;
        let pos = pos;

        let [m0, m1, m2, m3] = BINARY_MID_MASKS256[pos];
        // NOTE: This *will* go out-of-bounds, but it's ok because 'ranks' is used as padding.
        let [v0, v1, v2, v3]: [u64; 4] =
            unsafe { self.seq.as_ptr().cast::<[u64; 4]>().add(half).read() };
        let inner_count = (v0 & m0).count_ones()
            + (v1 & m1).count_ones()
            + (v2 & m2).count_ones()
            + (v3 & m3).count_ones();
        if pos < 256 {
            self.rank - inner_count
        } else {
            self.rank + inner_count
        }
    }
}

/// Only store a count to the middle and then count 256 bits.
#[repr(align(64))]
#[repr(C)]
#[derive(mem_dbg::MemSize)]
pub struct BinaryBlock6 {
    seq: [u8; 62],
    rank: u16,
}

impl BasicBlock for BinaryBlock6 {
    const X: usize = 1; // Binary
    const B: usize = 62;
    const N: usize = 496;
    const C: usize = 16;
    const W: usize = 16;
    const TRANSPOSED: bool = true;

    fn new(ranks: Ranks, data: &[u8; Self::B]) -> Self {
        // Counts in each u64 block.
        let mut bs = [0; 8];
        // count each part half.
        for (i, chunk) in data.as_chunks::<8>().0.iter().enumerate() {
            bs[i] = u64::from_le_bytes(*chunk).count_ones();
        }
        // partial ranks after 1 and 3 blocks
        let rank = (ranks[0] + bs[0] + bs[1] + bs[2] + bs[3]) as u16;
        Self { seq: *data, rank }
    }

    fn count<CF: CountFn<{ Self::C }>, const C3: bool>(&self, _pos: usize) -> Ranks {
        unimplemented!()
    }

    #[inline(always)]
    fn count1(&self, pos: usize, _c: u8) -> u32 {
        let half = pos / 256;
        let pos = pos;

        let [m0, m1, m2, m3] = BINARY_MID_MASKS256[pos];
        // NOTE: This *will* go out-of-bounds, but it's ok because 'ranks' is used as padding.
        let [v0, v1, v2, v3]: [u64; 4] =
            unsafe { self.seq.as_ptr().cast::<[u64; 4]>().add(half).read() };
        let inner_count = (v0 & m0).count_ones()
            + (v1 & m1).count_ones()
            + (v2 & m2).count_ones()
            + (v3 & m3).count_ones();
        if pos < 256 {
            self.rank as u32 - inner_count
        } else {
            self.rank as u32 + inner_count
        }
    }
}

/// Like BinarBlock6, but count to the start and linear scan rank in block.
#[repr(align(64))]
#[repr(C)]
#[derive(mem_dbg::MemSize)]
pub struct Spider {
    // First 2 bytes are rank offset.
    // The rest bits.
    seq: [u8; 64],
}

impl BasicBlock for Spider {
    const X: usize = 1; // Binary
    const B: usize = 62;
    const N: usize = 496;
    const C: usize = 16;
    const W: usize = 16;
    const TRANSPOSED: bool = true;

    fn new(ranks: Ranks, data: &[u8; Self::B]) -> Self {
        let mut seq = [0; 64];
        seq[0] = (ranks[0] & 0xff) as u8;
        seq[1] = ((ranks[0] >> 8) & 0xff) as u8;
        seq[2..].copy_from_slice(&data[0..62]);
        Self { seq }
    }

    fn count<CF: CountFn<{ Self::C }>, const C3: bool>(&self, _pos: usize) -> Ranks {
        unimplemented!()
    }

    #[inline(always)]
    fn count1(&self, pos: usize, _c: u8) -> u32 {
        unsafe { assert_unchecked(pos < 512) };
        let words = self.seq.as_ptr().cast::<u64>();
        let last_uint = pos / 64;
        let mut pop_val = 0;
        let final_x;

        const BIT_MASK: u64 = 0xFFFFFFFFFFFF0000;

        unsafe {
            if last_uint > 0 {
                pop_val += (words.read() & BIT_MASK).count_ones();
                for i in 1..last_uint {
                    pop_val += words.add(i).read().count_ones();
                }

                final_x = words.add(last_uint).read() >> (63 - (pos % 64));
            } else {
                final_x = words.add(last_uint).read() >> (63 - pos);
            }
            pop_val + final_x.count_ones()
        }
    }
}
