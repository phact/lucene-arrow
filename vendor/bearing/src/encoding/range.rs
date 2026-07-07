// SPDX-License-Identifier: Apache-2.0

//! Range encoding for multi-dimensional numeric ranges.
//!
//! Encodes min/max pairs for 1-4 dimensions as sortable byte arrays.
//! Used by `IntRange`, `LongRange`, `FloatRange`, and `DoubleRange` fields.

use std::fmt;

use crate::encoding::sortable_bytes::{from_double, from_float, from_int, from_long};

/// Encodes a range as sortable bytes: `[min1..minN, max1..maxN]`.
///
/// Each min/max value is converted to sortable bytes via `encode_fn`, then
/// packed into a single byte array with mins first, then maxs.
fn encode_range<T: PartialOrd + fmt::Display + Copy>(
    mins: &[T],
    maxs: &[T],
    elem_size: usize,
    encode_fn: fn(T) -> Vec<u8>,
) -> Vec<u8> {
    validate_range_dims(mins.len(), maxs.len());
    for i in 0..mins.len() {
        assert!(
            mins[i] <= maxs[i],
            "min[{i}] ({}) > max[{i}] ({})",
            mins[i],
            maxs[i]
        );
    }
    let dims = mins.len();
    let mut bytes = vec![0u8; dims * 2 * elem_size];
    for (i, &min) in mins.iter().enumerate() {
        bytes[i * elem_size..(i + 1) * elem_size].copy_from_slice(&encode_fn(min));
    }
    for (i, &max) in maxs.iter().enumerate() {
        let offset = (dims + i) * elem_size;
        bytes[offset..offset + elem_size].copy_from_slice(&encode_fn(max));
    }
    bytes
}

/// Encodes an integer range as sortable bytes: `[min1..minN, max1..maxN]`.
///
/// Each dimension's min and max are encoded as 4-byte sortable values.
/// Panics if `mins` and `maxs` differ in length, exceed 4 dimensions,
/// or any `min[i] > max[i]`.
pub fn encode_int(mins: &[i32], maxs: &[i32]) -> Vec<u8> {
    encode_range(mins, maxs, 4, |v| from_int(v).to_vec())
}

/// Encodes a long range as sortable bytes: `[min1..minN, max1..maxN]`.
///
/// Each dimension's min and max are encoded as 8-byte sortable values.
pub fn encode_long(mins: &[i64], maxs: &[i64]) -> Vec<u8> {
    encode_range(mins, maxs, 8, |v| from_long(v).to_vec())
}

/// Encodes a float range as sortable bytes: `[min1..minN, max1..maxN]`.
///
/// Each dimension's min and max are converted to sortable ints, then encoded
/// as 4-byte sortable values.
pub fn encode_float(mins: &[f32], maxs: &[f32]) -> Vec<u8> {
    encode_range(mins, maxs, 4, |v| from_float(v).to_vec())
}

/// Encodes a double range as sortable bytes: `[min1..minN, max1..maxN]`.
///
/// Each dimension's min and max are converted to sortable longs, then encoded
/// as 8-byte sortable values.
pub fn encode_double(mins: &[f64], maxs: &[f64]) -> Vec<u8> {
    encode_range(mins, maxs, 8, |v| from_double(v).to_vec())
}

fn validate_range_dims(min_len: usize, max_len: usize) {
    assert_eq!(min_len, max_len, "mins and maxs must have equal length");
    assert!(
        (1..=4).contains(&min_len),
        "dimensions must be 1-4, got {min_len}"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_int_single_dim() {
        let bytes = encode_int(&[10], &[20]);
        assert_len_eq_x!(&bytes, 8); // 1 dim * 2 * 4 bytes
        assert_eq!(&bytes[0..4], &from_int(10));
        assert_eq!(&bytes[4..8], &from_int(20));
    }

    #[test]
    fn test_encode_int_multi_dim() {
        let bytes = encode_int(&[1, 2], &[10, 20]);
        assert_len_eq_x!(&bytes, 16); // 2 dims * 2 * 4 bytes
        assert_eq!(&bytes[0..4], &from_int(1));
        assert_eq!(&bytes[4..8], &from_int(2));
        assert_eq!(&bytes[8..12], &from_int(10));
        assert_eq!(&bytes[12..16], &from_int(20));
    }

    #[test]
    fn test_encode_long_single_dim() {
        let bytes = encode_long(&[100], &[200]);
        assert_len_eq_x!(&bytes, 16); // 1 dim * 2 * 8 bytes
        assert_eq!(&bytes[0..8], &from_long(100));
        assert_eq!(&bytes[8..16], &from_long(200));
    }

    #[test]
    fn test_encode_float_single_dim() {
        let bytes = encode_float(&[1.0], &[2.0]);
        assert_len_eq_x!(&bytes, 8);
        assert_eq!(&bytes[0..4], &from_float(1.0));
        assert_eq!(&bytes[4..8], &from_float(2.0));
    }

    #[test]
    fn test_encode_double_single_dim() {
        let bytes = encode_double(&[1.0], &[2.0]);
        assert_len_eq_x!(&bytes, 16);
        assert_eq!(&bytes[0..8], &from_double(1.0));
        assert_eq!(&bytes[8..16], &from_double(2.0));
    }

    #[test]
    #[should_panic(expected = "equal length")]
    fn test_encode_int_mismatched_dims() {
        encode_int(&[1, 2], &[10]);
    }

    #[test]
    #[should_panic(expected = "dimensions must be 1-4")]
    fn test_encode_int_zero_dims() {
        encode_int(&[], &[]);
    }

    #[test]
    #[should_panic(expected = "dimensions must be 1-4")]
    fn test_encode_int_five_dims() {
        encode_int(&[1, 2, 3, 4, 5], &[10, 20, 30, 40, 50]);
    }

    #[test]
    #[should_panic(expected = "min[0]")]
    fn test_encode_int_min_gt_max() {
        encode_int(&[20], &[10]);
    }

    #[test]
    fn test_encode_int_equal_min_max() {
        let bytes = encode_int(&[5], &[5]);
        assert_eq!(&bytes[0..4], &bytes[4..8]);
    }

    #[test]
    fn test_encode_long_multi_dim() {
        let bytes = encode_long(&[1, 2, 3], &[10, 20, 30]);
        assert_len_eq_x!(&bytes, 48); // 3 dims * 2 * 8 bytes
    }

    #[test]
    fn test_encode_float_multi_dim() {
        let bytes = encode_float(&[0.5, 1.5], &[2.5, 3.5]);
        assert_len_eq_x!(&bytes, 16); // 2 dims * 2 * 4 bytes
    }

    #[test]
    fn test_encode_double_four_dims() {
        let bytes = encode_double(&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0]);
        assert_len_eq_x!(&bytes, 64); // 4 dims * 2 * 8 bytes
    }
}
