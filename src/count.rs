//! Various methods for counting the number of characters equal to c.

#[inline(always)]
pub fn count_u8x8(word: &[u8; 8], c: u8) -> u32 {
    count_u64(u64::from_le_bytes(*word), c)
}

#[inline(always)]
pub fn count_u8(word: u8, c: u8) -> u32 {
    // c = 00, 01, 10, 11 = cc
    // scatter = |01|01|01|...
    let scatter = 0x55u8;
    let mask = c as u8 * scatter;
    // mask = |cc|cc|cc|...

    // should be |00|00|00|... to match c.
    let tmp = word ^ mask;

    // |00| when c
    // |01| otherwise
    let union = (tmp | (tmp >> 1)) & scatter;
    4 - union.count_ones()
}

#[inline(always)]
pub fn count_u64(word: u64, c: u8) -> u32 {
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
pub fn count_u128(word: u128, c: u8) -> u32 {
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
