// SPDX-License-Identifier: Apache-2.0

//! Compact float encoding for small positive integers.
//!
//! Encodes positive integers into a single byte using a 4-bit mantissa and
//! 3-bit exponent format. Small values (below [`NUM_FREE_VALUES`]) are stored
//! directly; larger values use the float-like encoding.
//!
//! Used by Lucene for encoding document field lengths into norm values.

/// `longToInt4(i32::MAX)` — maximum value representable in the 4-bit float format.
const MAX_INT4: u32 = 231;

/// Number of values encoded directly (identity mapping) before switching to
/// the float-like encoding.
pub const NUM_FREE_VALUES: u32 = 255 - MAX_INT4; // 24

/// Encodes a positive long into a 4-bit mantissa + 3-bit exponent format.
///
/// Preserves ordering: if `a < b`, then `long_to_int4(a) <= long_to_int4(b)`.
pub fn long_to_int4(i: i64) -> i32 {
    assert!(i >= 0);
    let num_bits = 64 - (i as u64).leading_zeros();
    if num_bits < 4 {
        // subnormal value
        i as i32
    } else {
        // normal value
        let shift = num_bits - 4;
        let mut encoded = (i as u64 >> shift) as i32;
        // clear the most significant bit, which is implicit
        encoded &= 0x07;
        // encode the shift, adding 1 because 0 is reserved for subnormal values
        encoded |= (shift as i32 + 1) << 3;
        encoded
    }
}

/// Decodes a `long_to_int4`-encoded value back to the original long.
pub const fn int4_to_long(i: u32) -> i64 {
    let bits = (i & 0x07) as i64;
    let shift = (i >> 3) as i32 - 1;
    if shift == -1 {
        bits
    } else {
        (bits | 0x08) << shift
    }
}

/// Encodes an integer to a byte using the SmallFloat `intToByte4` format.
///
/// Values below [`NUM_FREE_VALUES`] are stored directly. Larger values use
/// [`long_to_int4`] with an offset.
pub fn int_to_byte4(i: i32) -> u8 {
    if i < 0 {
        return 0;
    }
    if (i as u32) < NUM_FREE_VALUES {
        i as u8
    } else {
        (NUM_FREE_VALUES + long_to_int4(i as i64 - NUM_FREE_VALUES as i64) as u32) as u8
    }
}

/// Decodes a byte encoded with [`int_to_byte4`] back to the original integer.
pub const fn byte4_to_int(b: u8) -> i32 {
    let i = b as u32;
    if i < NUM_FREE_VALUES {
        i as i32
    } else {
        (NUM_FREE_VALUES as i64 + int4_to_long(i - NUM_FREE_VALUES)) as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assertables::*;

    #[test]
    fn test_roundtrip_small_values() {
        for i in 0..24i32 {
            let encoded = int_to_byte4(i);
            let decoded = byte4_to_int(encoded);
            assert_eq!(decoded, i, "roundtrip failed for {i}");
        }
    }

    #[test]
    fn test_roundtrip_small_exact() {
        // Values below NUM_FREE_VALUES round-trip exactly
        assert_eq!(byte4_to_int(int_to_byte4(0)), 0);
        assert_eq!(byte4_to_int(int_to_byte4(1)), 1);
        assert_eq!(byte4_to_int(int_to_byte4(10)), 10);
        assert_eq!(byte4_to_int(int_to_byte4(23)), 23);
    }

    #[test]
    fn test_roundtrip_lossy_larger_values() {
        // Larger values are lossy but decoded value <= original
        for v in [100, 1000, 10000] {
            let decoded = byte4_to_int(int_to_byte4(v));
            assert_le!(decoded, v, "decoded should not exceed original for {v}");
            assert_gt!(decoded, 0);
        }
    }

    #[test]
    fn test_encoding_preserves_ordering() {
        let mut prev = 0u8;
        for i in 0..10000i32 {
            let encoded = int_to_byte4(i);
            assert_ge!(encoded, prev, "ordering violated at {i}");
            prev = encoded;
        }
    }

    #[test]
    fn test_negative_encodes_to_zero() {
        assert_eq!(int_to_byte4(-1), 0);
        assert_eq!(int_to_byte4(-100), 0);
    }

    #[test]
    fn test_byte4_to_int_full_range() {
        // All 256 byte values should decode without panic
        for b in 0..=255u8 {
            let decoded = byte4_to_int(b);
            assert_ge!(decoded, 0);
        }
    }

    #[test]
    fn test_byte4_to_int_monotonic() {
        let mut prev = 0i32;
        for b in 0..=255u8 {
            let decoded = byte4_to_int(b);
            assert_ge!(decoded, prev, "not monotonic at byte {b}");
            prev = decoded;
        }
    }
}
