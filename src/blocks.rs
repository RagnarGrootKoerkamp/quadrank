#![allow(non_camel_case_types)]

use crate::{Ranks, count::count_u8x8, count4::CountFn, traits::Block};

/// For each 128bp, store:
/// - 4 u64 counts, for 256bits total
/// - 256 bits of packed sequence.
/// In total, exactly covers a 512bit cache line.
///
/// Based on BWA: https://github.com/lh3/bwa/blob/master/bwtindex.c#L150
#[repr(C)]
#[repr(align(64))]
#[derive(mem_dbg::MemSize)]
pub struct FullBlock {
    // 4*64 = 256 bit counts
    ranks: [u64; 4],
    // 4*64 = 32*8 = 256 bit packed sequence
    seq: [u8; 32],
}

impl Block for FullBlock {
    const B: usize = 32;
    const N: usize = 128;

    fn new(ranks: Ranks, data: &[u8; Self::B]) -> Self {
        FullBlock {
            ranks: ranks.map(|x| x as u64),
            seq: *data,
        }
    }
}

impl FullBlock {
    fn count_4<C: CountFn<32>>(&self, pos: usize) -> Ranks {
        let mut ranks = [0; 4];
        for c in 0..4 {
            ranks[c] += self.ranks[c] as u32;
        }

        let inner_counts = C::count(&self.seq, pos);
        for c in 0..4 {
            ranks[c] += inner_counts[c];
        }

        let extra_counted = (4usize.wrapping_sub(pos)) % 4;
        ranks[0] -= extra_counted as u32;
        ranks
    }

    fn count_3<C: CountFn<32>>(&self, pos: usize) -> Ranks {
        let mut ranks = [0; 4];
        for c in 0..4 {
            ranks[c] += self.ranks[c] as u32;
        }
        let inner_counts = C::count(&self.seq, pos);
        for c in 0..4 {
            ranks[c] += inner_counts[c];
        }

        ranks[0] = pos as u32 - ranks[1] - ranks[2] - ranks[3];
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
pub struct HalfBlock {
    // meta[0,1,2,3] = 0
    // meta[4,5,6,7] = count of c=0,1,2,3 in first half (64bp=128bit).
    meta: [u8; 8],
    // counts for c=1,2,3
    ranks: [u64; 3],
    // 4*64 = 32*8 = 256 bit packed sequence
    seq: [u8; 32],
}

impl Block for HalfBlock {
    const B: usize = 32;
    const N: usize = 128;

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
}

impl HalfBlock {
    fn count_3<C: CountFn<16>>(&self, pos: usize) -> Ranks {
        let mut ranks = [0; 4];
        let chunk_pos = pos % 128;

        // 0 or 1 for left or right half
        let half = chunk_pos / 64;

        // Offset of chunk and half.
        for c in 0..4 {
            ranks[c] += self.ranks[c - 1] as u32 + self.meta[4 * half + c] as u32;
        }

        let idx = half * 16;
        let inner_counts = C::count(&self.seq[idx..idx + 16].try_into().unwrap(), pos);
        for c in 0..4 {
            ranks[c] += inner_counts[c];
        }

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
#[repr(C)]
#[repr(align(64))]
#[derive(mem_dbg::MemSize)]
pub struct HalfBlock2 {
    // 32bit counts for the first and second half.
    ranks: [[u32; 4]; 2],
    // 4*64 = 32*8 = 256 bit packed sequence
    seq: [u8; 32],
}

impl Block for HalfBlock2 {
    const B: usize = 32;
    const N: usize = 128;

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
}

impl HalfBlock2 {
    fn count_4<C: CountFn<16>>(&self, pos: usize) -> Ranks {
        let mut ranks = [0; 4];
        let chunk_pos = pos % 128;

        // 0 or 1 for left or right half
        let half = chunk_pos / 64;

        let idx = half * 16;
        let inner_counts = C::count(&self.seq[idx..idx + 16].try_into().unwrap(), pos);
        for c in 0..4 {
            ranks[c] += inner_counts[c];
        }
        for c in 0..4 {
            ranks[c] += self.ranks[half][c];
        }

        let extra_counted = (4usize.wrapping_sub(pos)) % 4;
        ranks[0] -= extra_counted as u32;
        ranks
    }

    fn count_3<C: CountFn<16>>(&self, pos: usize) -> Ranks {
        let mut ranks = [0; 4];
        let chunk_pos = pos % 128;

        // 0 or 1 for left or right half
        let half = chunk_pos / 64;

        let idx = half * 16;
        let inner_counts = C::count(&self.seq[idx..idx + 16].try_into().unwrap(), pos);
        for c in 0..4 {
            ranks[c] += inner_counts[c];
        }
        for c in 0..4 {
            ranks[c] += self.ranks[half][c];
        }

        ranks[0] = pos as u32 - ranks[1] - ranks[2] - ranks[3];
        ranks
    }
}

/// u32 global ranks, and 8bit ranks for each u64 quart.
#[repr(C)]
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

impl Block for QuartBlock {
    const B: usize = 32;
    const N: usize = 128;

    fn new(ranks: Ranks, data: &[u8; Self::B]) -> Self {
        let mut part_ranks = [0; 4];
        let mut block_ranks = [0u32; 4];
        // count each part half.
        for (i, chunk) in data.as_chunks::<8>().0.iter().enumerate() {
            for c in 0..4 {
                part_ranks[c] |= block_ranks[i] << (i * 8);
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
}

impl QuartBlock {
    fn count_4<C: CountFn<8>>(&self, pos: usize) -> Ranks {
        let mut ranks = [0; 4];
        let chunk_pos = pos % 128;

        // 0 or 1 for left or right half
        let quart = pos / 32;
        let quart_pos = pos % 32;

        let idx = quart * 8;
        let inner_counts = C::count(&self.seq[idx..idx + 8].try_into().unwrap(), pos);
        for c in 0..4 {
            ranks[c] += inner_counts[c];
        }
        for c in 0..4 {
            ranks[c] += (self.part_ranks[c] >> (quart * 8)) & 0xff;
        }

        let extra_counted = (4usize.wrapping_sub(pos)) % 4;
        ranks[0] -= extra_counted as u32;
        ranks
    }

    fn count_3<C: CountFn<8>>(&self, pos: usize) -> Ranks {
        let mut ranks = [0; 4];
        let chunk_pos = pos % 128;

        // 0 or 1 for left or right half
        let quart = pos / 32;
        let quart_pos = pos % 32;

        let idx = quart * 8;
        let inner_counts = C::count(&self.seq[idx..idx + 8].try_into().unwrap(), pos);
        for c in 0..4 {
            ranks[c] += inner_counts[c];
        }
        for c in 0..4 {
            ranks[c] += (self.part_ranks[c] >> (quart * 8)) & 0xff;
        }

        ranks[0] = pos as u32 - ranks[1] - ranks[2] - ranks[3];
        ranks
    }
}
