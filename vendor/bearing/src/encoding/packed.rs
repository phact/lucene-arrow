// SPDX-License-Identifier: Apache-2.0

//! Packed integer encoding for compact storage of fixed-width values.
//!
//! Provides bit-packing functions for sequences of integers at a fixed
//! bits-per-value (BPV). Supports both LSB-first (little-endian) and
//! MSB-first (big-endian) bit ordering.

use std::io;
use std::io::Write;

/// Supported bits-per-value for DirectWriter encoding.
pub const SUPPORTED_BITS_PER_VALUE: &[u32] = &[1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64];

/// Token byte layout: bits 1-7 = bitsPerValue, bit 0 = min-is-zero flag.
pub const BPV_SHIFT: u32 = 1;

/// Flag indicating the block minimum is zero.
pub const MIN_VALUE_EQUALS_0: u8 = 1;

/// Returns the number of bits required to represent values up to and including `max_value`,
/// interpreted as unsigned, rounded up to the nearest supported BPV.
pub fn unsigned_bits_required(max_value: i64) -> u32 {
    let raw = if max_value == 0 {
        1
    } else {
        64 - (max_value as u64).leading_zeros()
    };
    *SUPPORTED_BITS_PER_VALUE
        .iter()
        .find(|&&s| s >= raw)
        .expect("raw bits exceeds max supported BPV")
}

/// Returns the number of bits required to represent a value.
pub fn bits_required(value: u64) -> u32 {
    if value == 0 {
        0
    } else {
        64 - value.leading_zeros()
    }
}

/// Returns the raw number of bits required to represent `value`, treating it as unsigned.
///
/// Unlike [`unsigned_bits_required`], this does NOT round up to a supported BPV.
/// Returns `max(1, 64 - leading_zeros)`.
pub fn packed_bits_required(value: i64) -> u32 {
    if value == 0 {
        1
    } else {
        64 - (value as u64).leading_zeros()
    }
}

/// Returns the maximum value representable with the given number of bits.
pub fn packed_max_value(bits_per_value: u32) -> i64 {
    if bits_per_value == 64 {
        i64::MAX
    } else {
        !(!0i64 << bits_per_value)
    }
}

/// Packs `count` values MSB-first into bytes.
///
/// Returns exactly `ceil(count * bits_per_value / 8)` bytes.
pub fn pack_msb(values: &[i64], count: usize, bits_per_value: u32) -> Vec<u8> {
    let total_bytes = (count as u64 * bits_per_value as u64).div_ceil(8) as usize;
    let mut blocks = vec![0u8; total_bytes];
    let mut blocks_offset = 0;
    let mut next_block: u8 = 0;
    let mut bits_left: u32 = 8;
    let bpv = bits_per_value;

    for &value in values.iter().take(count) {
        let v = value as u64;
        if bpv < bits_left {
            next_block |= (v << (bits_left - bpv)) as u8;
            bits_left -= bpv;
        } else {
            let mut bits = bpv - bits_left;
            blocks[blocks_offset] = next_block | (v >> bits) as u8;
            blocks_offset += 1;
            while bits >= 8 {
                bits -= 8;
                blocks[blocks_offset] = (v >> bits) as u8;
                blocks_offset += 1;
            }
            bits_left = 8 - bits;
            next_block = ((v & ((1u64 << bits) - 1)) << bits_left) as u8;
        }
    }

    if bits_left < 8 && blocks_offset < total_bytes {
        blocks[blocks_offset] = next_block;
    }

    blocks
}

/// Writes packed integers MSB-first (big-endian bit packing).
///
/// Values are packed contiguously with high-order bits first, producing
/// `ceil(count * bpv / 8)` bytes.
pub fn packed_ints_write(
    output: &mut dyn Write,
    values: &[i64],
    bits_per_value: u32,
) -> io::Result<()> {
    let packed = pack_msb(values, values.len(), bits_per_value);
    output.write_all(&packed)
}

/// Writes a variable-length long that handles negative values via unsigned right shift.
///
/// NOT the same as varint::write_vlong — this variant caps at 9 bytes.
pub fn write_block_packed_vlong(output: &mut dyn Write, value: i64) -> io::Result<()> {
    let mut i = value as u64;
    let mut k = 0;
    while (i & !0x7F) != 0 && k < 8 {
        output.write_all(&[(i & 0x7F | 0x80) as u8])?;
        i >>= 7;
        k += 1;
    }
    output.write_all(&[i as u8])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bits_required() {
        assert_eq!(bits_required(0), 0);
        assert_eq!(bits_required(1), 1);
        assert_eq!(bits_required(2), 2);
        assert_eq!(bits_required(3), 2);
        assert_eq!(bits_required(255), 8);
        assert_eq!(bits_required(256), 9);
    }

    #[test]
    fn test_unsigned_bits_required() {
        assert_eq!(unsigned_bits_required(0), 1);
        assert_eq!(unsigned_bits_required(1), 1);
        assert_eq!(unsigned_bits_required(2), 2);
        assert_eq!(unsigned_bits_required(3), 2);
        assert_eq!(unsigned_bits_required(4), 4);
        assert_eq!(unsigned_bits_required(15), 4);
        assert_eq!(unsigned_bits_required(16), 8);
        assert_eq!(unsigned_bits_required(255), 8);
        assert_eq!(unsigned_bits_required(256), 12);
        assert_eq!(unsigned_bits_required(4095), 12);
        assert_eq!(unsigned_bits_required(4096), 16);
        assert_eq!(unsigned_bits_required(i64::MAX), 64);
        assert_eq!(unsigned_bits_required(-1), 64);
    }

    #[test]
    fn test_packed_bits_required() {
        assert_eq!(packed_bits_required(0), 1);
        assert_eq!(packed_bits_required(1), 1);
        assert_eq!(packed_bits_required(2), 2);
        assert_eq!(packed_bits_required(3), 2);
        assert_eq!(packed_bits_required(255), 8);
        assert_eq!(packed_bits_required(256), 9);
        assert_eq!(packed_bits_required(i64::MAX), 63);
        assert_eq!(packed_bits_required(-1), 64);
    }

    #[test]
    fn test_packed_max_value() {
        assert_eq!(packed_max_value(1), 1);
        assert_eq!(packed_max_value(8), 255);
        assert_eq!(packed_max_value(16), 65535);
        assert_eq!(packed_max_value(63), i64::MAX);
        assert_eq!(packed_max_value(64), i64::MAX);
    }

    #[test]
    fn test_pack_msb_4bit() {
        let values = [0xA_i64, 0xB];
        let result = pack_msb(&values, 2, 4);
        assert_eq!(result, vec![0xAB]);
    }

    #[test]
    fn test_pack_msb_8bit() {
        let values = [0x12_i64, 0x34, 0xAB];
        let result = pack_msb(&values, 3, 8);
        assert_eq!(result, vec![0x12, 0x34, 0xAB]);
    }

    #[test]
    fn test_pack_msb_1bit() {
        let values = [1_i64, 0, 1, 0, 1, 1, 0, 0];
        let result = pack_msb(&values, 8, 1);
        assert_eq!(result, vec![0xAC]);
    }

    #[test]
    fn test_pack_msb_5bit_spanning() {
        let values = [31_i64, 21, 1];
        let result = pack_msb(&values, 3, 5);
        assert_len_eq_x!(&result, 2);
        assert_eq!(result, vec![0xFD, 0x42]);
    }

    #[test]
    fn test_pack_msb_64bit() {
        let values = [0x0102030405060708_i64];
        let result = pack_msb(&values, 1, 64);
        assert_eq!(result, vec![0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]);
    }

    #[test]
    fn test_packed_ints_write() {
        let mut out = Vec::new();
        let values = [0xA_i64, 0xB];
        packed_ints_write(&mut out, &values, 4).unwrap();
        assert_eq!(out, &[0xAB]);
    }

    #[test]
    fn test_write_block_packed_vlong_small() {
        let mut out = Vec::new();
        write_block_packed_vlong(&mut out, 42).unwrap();
        assert_eq!(out, &[42]);
    }

    #[test]
    fn test_write_block_packed_vlong_large() {
        let mut out = Vec::new();
        write_block_packed_vlong(&mut out, 128).unwrap();
        assert_eq!(out, &[0x80, 0x01]);
    }

    #[test]
    fn test_write_block_packed_vlong_negative() {
        let mut out = Vec::new();
        write_block_packed_vlong(&mut out, -1).unwrap();
        assert_len_eq_x!(&out, 9);
        for &b in &out[..8] {
            assert_eq!(b, 0xFF);
        }
        assert_eq!(out[8], 0xFF);
    }

    // Cross-validated against Java BlockPackedWriter output

    #[test]
    fn test_packed_ints_write_java_4bpv() {
        let mut out = Vec::new();
        let values: Vec<i64> = (0..10).collect();
        packed_ints_write(&mut out, &values, 4).unwrap();
        #[rustfmt::skip]
        let expected: &[u8] = &[0x01, 0x23, 0x45, 0x67, 0x89];
        assert_eq!(out, expected);
    }

    #[test]
    fn test_packed_ints_write_java_5bpv() {
        let mut out = Vec::new();
        let values = [31_i64, 21, 1, 0, 15, 7, 3, 30];
        packed_ints_write(&mut out, &values, 5).unwrap();
        #[rustfmt::skip]
        let expected: &[u8] = &[0xFD, 0x42, 0x07, 0x9C, 0x7E];
        assert_eq!(out, expected);
    }

    #[test]
    fn test_packed_ints_write_java_8bpv() {
        let mut out = Vec::new();
        let values = [0x12_i64, 0x34, 0xAB];
        packed_ints_write(&mut out, &values, 8).unwrap();
        #[rustfmt::skip]
        let expected: &[u8] = &[0x12, 0x34, 0xAB];
        assert_eq!(out, expected);
    }
}
