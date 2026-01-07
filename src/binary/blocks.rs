use std::hint::assert_unchecked;

use super::BasicBlock;

/// Like TriBlock, but for binary counts.
#[repr(align(64))]
#[repr(C)]
#[derive(mem_dbg::MemSize)]
pub struct BinaryBlock1 {
    /// offset to end of 1st and 3rd 128bit block.
    /// 8 low bits: delta to end of first trip
    ranks: [u64; 2],
    // u128x3 = u8x48 = 384 bit packed sequence
    seq: [[u64; 2]; 3],
}

impl BasicBlock for BinaryBlock1 {
    const B: usize = 48;
    const N: usize = 384;
    const W: usize = 64;

    fn new(rank: u64, data: &[u8; Self::B]) -> Self {
        // Counts in each u64 block.
        let mut bs = [0u64; 6];
        let mut sum = 0u64;
        // count each part half.
        for (i, chunk) in data.as_chunks::<8>().0.iter().enumerate() {
            bs[i] = u64::from_le_bytes(*chunk).count_ones() as u64;
            sum += bs[i];
        }
        // partial ranks after 1 and 3 blocks
        let ranks = [rank as u64 + bs[0] + bs[1], rank as u64 + sum];
        Self {
            ranks,
            seq: unsafe { std::mem::transmute(*data) },
        }
    }

    #[inline(always)]
    unsafe fn rank_unchecked(&self, pos: usize) -> u64 {
        let tri = pos / 128;
        let pos = pos % 256;

        let [mask_l, mask_h] = BINARY_MID_MASKS[pos];
        let [l, h] = self.seq[tri];
        let inner_count = ((l & mask_l).count_ones() + (h & mask_h).count_ones()) as u64;
        if pos < 128 {
            self.ranks[tri / 2] as u64 - inner_count
        } else {
            self.ranks[tri / 2] as u64 + inner_count
        }
    }
}

/// Store two 32bit ranks at the end, and put in some more bits.
#[repr(align(64))]
#[repr(C)]
#[derive(mem_dbg::MemSize)]
pub struct BinaryBlock2 {
    // u128x3 + u64 = u8x56 = 448 bit packed sequence
    seq: [u8; 56],
    /// offset to end of 1st and 3rd 128bit block.
    ranks: [u32; 2],
}

impl BasicBlock for BinaryBlock2 {
    const B: usize = 56;
    const N: usize = 448;
    const W: usize = 32;

    fn new(rank: u64, data: &[u8; Self::B]) -> Self {
        let rank = rank as u32;
        // Counts in each u64 block.
        let mut bs = [0; 8];
        // count each part half.
        for (i, chunk) in data.as_chunks::<8>().0.iter().enumerate() {
            bs[i] = u64::from_le_bytes(*chunk).count_ones();
        }
        // partial ranks after 1 and 3 blocks
        let ranks = [
            rank + bs[0] + bs[1],
            rank + bs[0] + bs[1] + bs[2] + bs[3] + bs[4] + bs[5],
        ];
        Self {
            ranks,
            seq: unsafe { std::mem::transmute(*data) },
        }
    }

    #[inline(always)]
    unsafe fn rank_unchecked(&self, pos: usize) -> u64 {
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
            let inner_count = ((l & mask_l).count_ones() + (h & mask_h).count_ones()) as u64;
            if pos < 128 {
                self.ranks[quad / 2] as u64 - inner_count
            } else {
                self.ranks[quad / 2] as u64 + inner_count
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
    const B: usize = 59;
    const N: usize = 472;
    const W: usize = 31;

    fn new(rank: u64, data: &[u8; Self::B]) -> Self {
        // Counts in each u64 block.
        let mut bs = [0; 8];
        // count each part half.
        for (i, chunk) in data.as_chunks::<8>().0.iter().enumerate() {
            bs[i] = u64::from_le_bytes(*chunk).count_ones() as u64;
        }
        // partial ranks after 1 and 3 blocks
        let delta = bs[2] + bs[3] + bs[4] + bs[5];
        let ranks = rank + bs[0] + bs[1] + delta;

        let mut seq = [0u8; 64];
        for i in 0..59 {
            seq[i] = data[i];
        }
        let rank_bytes = ((ranks as u64) << 9) | (delta as u64);
        seq[59..64].copy_from_slice(&rank_bytes.to_le_bytes()[0..5]);
        Self { seq }
    }

    #[inline(always)]
    unsafe fn rank_unchecked(&self, pos: usize) -> u64 {
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
            let inner_count = ((l & mask_l).count_ones() + (h & mask_h).count_ones()) as u64;
            let meta = u64::from_le_bytes(self.seq.get_unchecked(56..64).try_into().unwrap()) >> 24;
            let rank = (meta >> 9) as u64;
            let delta = meta as u32 & 0x1ff;
            let delta = (delta >> ((quad / 2) * 16)) as u64;

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
#[repr(C)]
#[derive(mem_dbg::MemSize)]
pub struct BinaryBlock4 {
    seq: [u8; 60],
    ranks: [u16; 2],
}

impl BasicBlock for BinaryBlock4 {
    const B: usize = 60;
    const N: usize = 480;
    const W: usize = 16;

    fn new(rank: u64, data: &[u8; Self::B]) -> Self {
        // Counts in each u64 block.
        let mut bs = [0; 8];
        // count each part half.
        for (i, chunk) in data.as_chunks::<8>().0.iter().enumerate() {
            bs[i] = u64::from_le_bytes(*chunk).count_ones() as u64;
        }
        // partial ranks after 1 and 3 blocks
        let rank = rank + bs[0] + bs[1];
        let delta = bs[2] + bs[3] + bs[4] + bs[5];
        let ranks = [rank as u16, rank as u16 + delta as u16];
        Self { seq: *data, ranks }
    }

    #[inline(always)]
    unsafe fn rank_unchecked(&self, pos: usize) -> u64 {
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
            let inner_count = ((l & mask_l).count_ones() + (h & mask_h).count_ones()) as u64;
            let rank = *self.ranks.get_unchecked(quad / 2) as u64;
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
#[repr(C)]
#[derive(mem_dbg::MemSize)]
pub struct BinaryBlock5 {
    seq: [u8; 60],
    rank: u32,
}

impl BasicBlock for BinaryBlock5 {
    const B: usize = 60;
    const N: usize = 480;
    const W: usize = 32;

    fn new(rank: u64, data: &[u8; Self::B]) -> Self {
        // Counts in each u64 block.
        let mut bs = [0; 8];
        // count each part half.
        for (i, chunk) in data.as_chunks::<8>().0.iter().enumerate() {
            bs[i] = u64::from_le_bytes(*chunk).count_ones() as u64;
        }
        // partial ranks after 1 and 3 blocks
        let rank = rank + bs[0] + bs[1] + bs[2] + bs[3];
        Self {
            seq: *data,
            rank: rank as u32,
        }
    }

    #[inline(always)]
    unsafe fn rank_unchecked(&self, pos: usize) -> u64 {
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
            (self.rank - inner_count).into()
        } else {
            (self.rank + inner_count).into()
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
    const B: usize = 62;
    const N: usize = 496;
    const W: usize = 16;

    fn new(rank: u64, data: &[u8; Self::B]) -> Self {
        // Counts in each u64 block.
        let mut bs = [0; 8];
        // count each part half.
        for (i, chunk) in data.as_chunks::<8>().0.iter().enumerate() {
            bs[i] = u64::from_le_bytes(*chunk).count_ones() as u64;
        }
        // partial ranks after 1 and 3 blocks
        let rank = (rank + bs[0] + bs[1] + bs[2] + bs[3]) as u16;
        Self { seq: *data, rank }
    }

    #[inline(always)]
    unsafe fn rank_unchecked(&self, pos: usize) -> u64 {
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
            self.rank as u64 - inner_count as u64
        } else {
            self.rank as u64 + inner_count as u64
        }
    }
}

/// Like BinaryBlock6, but count to the start and linear scan rank in block.
#[repr(align(64))]
#[repr(C)]
#[derive(mem_dbg::MemSize)]
pub struct Spider {
    // First 2 bytes are rank offset.
    // The rest bits.
    seq: [u8; 64],
}

impl BasicBlock for Spider {
    const B: usize = 62;
    const N: usize = 496;
    const W: usize = 16;
    const INCLUSIVE: bool = true;

    fn new(rank: u64, data: &[u8; Self::B]) -> Self {
        let mut seq = [0; 64];
        seq[0] = (rank & 0xff) as u8;
        seq[1] = ((rank >> 8) & 0xff) as u8;
        seq[2..].copy_from_slice(&data[0..62]);
        Self { seq }
    }

    /// `pos` is in [0, 512) and right-inclusive here.
    #[inline(always)]
    unsafe fn rank_unchecked(&self, pos: usize) -> u64 {
        // Pad for the first 16 bits.
        let pos = pos + 16;
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

                // FIXME: Why does the original have >> here.
                final_x = words.add(last_uint).read() << (63 - (pos % 64));
            } else {
                final_x = (words.read() & BIT_MASK) << (63 - pos);
            }
            // correct for inclusive position
            let rank = words.cast::<u16>().read() as u64;
            let inner_count = pop_val + final_x.count_ones();
            // FIXME: Add inclusive vs exclusive generic.
            rank + inner_count as u64
        }
    }
}

pub static BINARY_MID_MASKS: [[u64; 2]; 256] = {
    let mut masks = [[0u64; 2]; 256];
    let mut i = 0;
    while i < 128 {
        let low_bits = i;
        let mask = if low_bits == 128 {
            u128::MAX
        } else {
            (1u128 << low_bits) - 1
        };
        unsafe {
            masks[i] = std::mem::transmute(!mask);
            masks[i + 128] = std::mem::transmute(mask);
        }
        i += 1;
    }
    masks
};

pub static BINARY_MID_MASKS256: [[u64; 4]; 512] = {
    let mut masks = [[0u64; 4]; 512];
    let mut i = 0;
    while i < 256 {
        let lo_mask = if i < 128 { (1u128 << i) - 1 } else { u128::MAX };
        let hi_mask = if i <= 128 {
            0
        } else {
            (1u128 << (i - 128)) - 1
        };
        unsafe {
            masks[i] = std::mem::transmute([!lo_mask, !hi_mask]);
            masks[i + 256] = std::mem::transmute([lo_mask, hi_mask]);
        }
        i += 1;
    }
    masks
};
