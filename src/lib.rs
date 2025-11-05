use packed_seq::{PackedSeqVec, Seq, SeqVec};

pub type Ranks = [usize; 4];

pub trait Rank {
    fn new(seq: &[u8]) -> Self;
    fn rank(&self, pos: usize, c: u8) -> usize {
        self.ranks(pos)[packed_seq::pack_char(c) as usize]
    }
    fn ranks(&self, pos: usize) -> Ranks;
    fn interval_ranks(&self, pos1: usize, pos2: usize) -> (Ranks, Ranks) {
        (self.ranks(pos1), self.ranks(pos2))
    }
}

const STRIDE: usize = 256;

pub struct DnaRank {
    seq: PackedSeqVec,
    counts: Vec<Ranks>,
}

impl Rank for DnaRank {
    fn new(seq: &[u8]) -> Self {
        let mut counts = Vec::with_capacity(seq.len().div_ceil(STRIDE));
        let mut ranks = [0; 4];
        counts.push(ranks);
        for chunk in seq.as_chunks::<STRIDE>().0 {
            for &c in chunk {
                ranks[packed_seq::pack_char(c) as usize] += 1;
            }
            counts.push(ranks);
        }

        DnaRank {
            seq: PackedSeqVec::from_ascii(seq),
            counts,
        }
    }

    fn ranks(&self, pos: usize) -> [usize; 4] {
        let idx = pos / STRIDE;
        let mut ranks = self.counts[idx];
        for c in self.seq.slice(idx * STRIDE..pos).iter_bp() {
            ranks[c as usize] += 1;
        }
        ranks
    }
}
