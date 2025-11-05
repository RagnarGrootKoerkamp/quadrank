use packed_seq::{PackedSeqVec, SeqVec};

pub type Ranks = [usize; 4];

#[derive(mem_dbg::MemSize)]
pub struct DnaRank<const STRIDE: usize> {
    n: usize,
    seq: Vec<u8>,
    counts: Vec<Ranks>,
}

impl<const STRIDE: usize> DnaRank<STRIDE> {
    pub fn new(seq: &[u8]) -> Self {
        assert!(STRIDE % 4 == 0, "STRIDE must be a multiple of 4");
        let mut counts = Vec::with_capacity(seq.len().div_ceil(STRIDE));
        let mut ranks = [0; 4];
        counts.push(ranks);
        for chunk in seq.as_chunks::<STRIDE>().0 {
            for &c in chunk {
                ranks[packed_seq::pack_char(c) as usize] += 1;
            }
            counts.push(ranks);
        }

        let n = seq.len();
        let mut seq = PackedSeqVec::from_ascii(seq).into_raw();
        seq.resize(seq.capacity(), 0);
        assert!(seq.len() >= n / 4 + 32);
        DnaRank { n, seq, counts }
    }

    /// Loop over packed characters.
    pub fn ranks_naive(&self, pos: usize) -> [usize; 4] {
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
    pub fn ranks(&self, pos: usize) -> [usize; 4] {
        let chunk_idx = pos / STRIDE;
        let byte_idx = chunk_idx * (STRIDE / 4);
        let mut ranks = self.counts[chunk_idx];

        for idx in (byte_idx..pos.div_ceil(4)).step_by(32) {
            let chunk = u64::from_le_bytes(self.seq[idx..idx + 8].try_into().unwrap());
            let low_bits = (pos - idx).max(32) * 2;
            let chunk = chunk & ((1u64 << low_bits) - 1);
            for c in 0..4 {
                ranks[c as usize] += count_u64(chunk, c);
            }
        }

        ranks
    }
}

fn count_u64(word: u64, c: u8) -> usize {
    assert!(c < 4);
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
    32 - union.count_ones() as usize
}
