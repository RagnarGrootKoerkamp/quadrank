#![allow(non_camel_case_types)]

use std::array::from_fn;

use crate::{
    Ranks, add,
    count::{count_u8x8, count_u8x16, count_u64},
    count4::{CountFn, MASKS, count4_u8x8},
    ranker::BasicBlock,
};

#[inline(always)]
fn extra_counted<const B: usize, C: CountFn<B>>(pos: usize) -> u32 {
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
    const B: usize = 64; // Bytes of characters in block.
    const N: usize = 256; // Number of characters in block.
    const C: usize = 64; // Bytes of the underlying count function.
    const W: usize = 0;

    fn new(_ranks: Ranks, data: &[u8; Self::B]) -> Self {
        Plain512 { seq: *data }
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

#[repr(align(32))]
#[derive(mem_dbg::MemSize)]
pub struct Plain256 {
    seq: [u8; 32],
}

impl BasicBlock for Plain256 {
    const B: usize = 32; // Bytes of characters in block.
    const N: usize = 128; // Number of characters in block.
    const C: usize = 32; // Bytes of the underlying count function.
    const W: usize = 0;

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
    const B: usize = 16; // Bytes of characters in block.
    const N: usize = 64; // Number of characters in block.
    const C: usize = 16; // Bytes of the underlying count function.
    const W: usize = 0;

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
    const B: usize = 32; // Bytes of characters in block.
    const N: usize = 128; // Number of characters in block.
    const C: usize = 32; // Bytes of the underlying count function.
    const W: usize = 64;

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
    const B: usize = 32; // Bytes of characters in block.
    const N: usize = 128; // Number of characters in block.
    const C: usize = 16; // Bytes of the underlying count function.
    const W: usize = 64;

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
    const B: usize = 32;
    const N: usize = 128;
    const C: usize = 16;
    const W: usize = 32;

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
    const B: usize = 32;
    const N: usize = 128;
    const C: usize = 16;
    const W: usize = 32;

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
    const B: usize = 32;
    const N: usize = 128;
    const C: usize = 8;
    const W: usize = 32;

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
        let inner_counts = C::count(&self.seq[idx..idx + 8].try_into().unwrap(), quart_pos);
        for c in 0..4 {
            ranks[c] += inner_counts[c];
        }

        if C3 {
            ranks[0] = quart_pos as u32 - ranks[1] - ranks[2] - ranks[3];
        } else {
            ranks[0] -= extra_counted::<_, C>(pos);
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
        let mut chunk = u64::from_le_bytes(self.seq[idx..idx + 8].try_into().unwrap());
        let mask = MASKS[quart_pos];
        chunk &= mask;
        let inner_count = count_u64(chunk, c);
        rank += inner_count;
        rank += (self.part_ranks[c as usize] >> (quart * 8)) & 0xff;
        rank += self.ranks[c as usize];

        rank
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
    const B: usize = 40;
    const N: usize = 160;
    const C: usize = 8;
    const W: usize = 16;

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
    const B: usize = 40;
    const N: usize = 160;
    const C: usize = 8;
    const W: usize = 20;

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
    /// 16it counts for the global offset
    ranks: [u16; 4],
    /// Each u16 is equivalent to [u8; 2] with counts from start to 2 of 3 128 parts.
    part_ranks: [u16; 4],
    // u128x3 = u8x48 = 384 bit packed sequence
    seq: [u8; 48],
}

impl BasicBlock for HexaBlock {
    const B: usize = 48;
    const N: usize = 192;
    const C: usize = 16;
    const W: usize = 16;

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
    const B: usize = 48;
    const N: usize = 192;
    const C: usize = 16;
    const W: usize = 18;

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
    /// 16it counts for the global offset to position 32 (middle of first u128).
    ranks: [u16; 4],
    /// Each u16 is equivalent to [u8; 2] with deltas to middle of 2nd and 3rd 128 parts.
    part_ranks: [u16; 4],
    // u128x3 = u8x48 = 384 bit packed sequence
    seq: [u8; 48],
}

impl BasicBlock for HexaBlockMid {
    const B: usize = 48;
    const N: usize = 192;
    const C: usize = 8;
    const W: usize = 16;

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
        let part_ranks = from_fn(|c| ((p1[c] << 8) | p2[c]) as u16);
        Self {
            ranks: ranks.map(|x| x as u16),
            part_ranks,
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
            ranks[c] += self.ranks[c] as u32;
        }

        if (pos & 32) == 0 {
            let inner_counts = C::count_right(&self.seq[idx..idx + 8].try_into().unwrap(), hex_pos);
            if !C3 {
                let e = extra_counted::<_, C>(pos);
                let f = (C::S as u32 * 4 - e) % (C::S as u32 * 4);
                ranks[0] += f;
            }
            for c in 0..4 {
                ranks[c] -= inner_counts[c];
            }
        } else {
            let inner_counts = C::count(&self.seq[idx..idx + 8].try_into().unwrap(), hex_pos);
            for c in 0..4 {
                ranks[c] += inner_counts[c];
            }
            if !C3 {
                ranks[0] -= extra_counted::<_, C>(pos);
            }
        }

        if C3 {
            ranks[0] = hex_pos as u32 - ranks[1] - ranks[2] - ranks[3];
        }

        for c in 0..4 {
            ranks[c] += (self.part_ranks[c].unbounded_shr(16 - 8 * (hex / 2) as u32)) as u32 & 0xff;
        }

        ranks
    }
}
