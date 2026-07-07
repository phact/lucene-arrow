// SPDX-License-Identifier: Apache-2.0
//! Lowercase ASCII compression for blocktree suffix data.
//!
//! This is a Lucene-specific compression algorithm — there is no standard name,
//! RFC, or external library that implements it. Ported directly from
//! `org.apache.lucene.util.compress.LowercaseAsciiCompression`.
//!
//! Compresses byte arrays that mostly contain characters in the `[0x1F,0x3F)` or
//! `[0x5F,0x7F)` ranges (digits, lowercase letters, '.', '-', '_'). Each byte is
//! reduced to a 6-bit value, then 4 bytes are packed into 3. Non-compressible bytes
//! are recorded as delta-encoded exceptions. Compression fails if the input is too
//! short (<8 bytes) or has too many exceptions (>len/32).

use std::io;
use std::io::BufRead;
use std::io::Cursor;
use std::io::Read;

use crate::encoding::varint;

/// Returns whether a byte value is compressible (maps to a 6-bit value).
fn is_compressible(b: u8) -> bool {
    let high3 = (b.wrapping_add(1)) & !0x1F;
    high3 == 0x20 || high3 == 0x60
}

/// Compress `input[..len]` using lowercase ASCII packing.
///
/// Returns `Some(compressed_bytes)` if compression succeeds (output < input length),
/// or `None` if the input is too short or has too many exceptions.
pub fn compress(input: &[u8], len: usize) -> Option<Vec<u8>> {
    if len < 8 {
        return None;
    }

    // 1. Count exceptions — fail if too many.
    let max_exceptions = len >> 5;
    let mut previous_exception_index: usize = 0;
    let mut num_exceptions: usize = 0;
    for (i, &byte) in input[..len].iter().enumerate() {
        if !is_compressible(byte) {
            while i - previous_exception_index > 0xFF {
                num_exceptions += 1;
                previous_exception_index += 0xFF;
            }
            num_exceptions += 1;
            if num_exceptions > max_exceptions {
                return None;
            }
            previous_exception_index = i;
        }
    }

    // 2. Shift all bytes to the [0,0x40) range (6 bits).
    let compressed_len = len - (len >> 2);
    let mut tmp = vec![0u8; len];
    for i in 0..len {
        let b = input[i].wrapping_add(1);
        tmp[i] = (b & 0x1F) | ((b & 0x40) >> 1);
    }

    // 3. Pack 4 bytes into 3 — scatter high bits of the tail into the head.
    let mut o = 0;
    for i in compressed_len..len {
        tmp[o] |= (tmp[i] & 0x30) << 2; // bits 4-5
        o += 1;
    }
    for i in compressed_len..len {
        tmp[o] |= (tmp[i] & 0x0C) << 4; // bits 2-3
        o += 1;
    }
    for i in compressed_len..len {
        tmp[o] |= (tmp[i] & 0x03) << 6; // bits 0-1
        o += 1;
    }

    // Build output: packed bytes, then vint exception count, then exception pairs
    let mut out = Vec::with_capacity(compressed_len + 1 + num_exceptions * 2);
    out.extend_from_slice(&tmp[..compressed_len]);

    // Write exception count as VInt
    varint::write_vint(&mut out, num_exceptions as i32).unwrap();

    if num_exceptions > 0 {
        previous_exception_index = 0;
        let mut num_exceptions2 = 0;
        for i in 0..len {
            let b = input[i];
            if !is_compressible(b) {
                while i - previous_exception_index > 0xFF {
                    out.push(0xFF);
                    previous_exception_index += 0xFF;
                    out.push(input[previous_exception_index]);
                    num_exceptions2 += 1;
                }
                out.push((i - previous_exception_index) as u8);
                previous_exception_index = i;
                out.push(b);
                num_exceptions2 += 1;
            }
        }
        assert_eq!(num_exceptions, num_exceptions2);
    }

    if out.len() < len { Some(out) } else { None }
}

/// Decompress lowercase ASCII data from a cursor.
///
/// `len` is the original uncompressed length. Reads compressed bytes from the
/// cursor and restores the original byte values. Cursor state is undefined on
/// error; the caller is expected to abort the surrounding read.
pub fn decompress_from_cursor(cursor: &mut Cursor<&[u8]>, len: usize) -> io::Result<Vec<u8>> {
    let saved = len >> 2;
    let compressed_len = len - saved;

    // 1. Read the packed bytes
    let mut out = vec![0u8; len];
    cursor.read_exact(&mut out[..compressed_len])?;

    // 2. Restore the leading 2 bits of each packed byte
    for i in 0..saved {
        out[compressed_len + i] = ((out[i] & 0xC0) >> 2)
            | ((out[saved + i] & 0xC0) >> 4)
            | ((out[(saved << 1) + i] & 0xC0) >> 6);
    }

    // 3. Move back to the original range
    for b in &mut out[..len] {
        *b = ((*b & 0x1F) | 0x20 | ((*b & 0x20) << 1)).wrapping_sub(1);
    }

    // 4. Restore exceptions
    let num_exceptions = varint::read_vint_cursor(cursor)? as usize;
    let exception_bytes = 2 * num_exceptions;
    {
        let buf = cursor.fill_buf()?;
        if buf.len() < exception_bytes {
            return Err(io::Error::from(io::ErrorKind::UnexpectedEof));
        }
        let mut i = 0usize;
        for k in 0..num_exceptions {
            i += buf[k * 2] as usize;
            out[i] = buf[k * 2 + 1];
        }
    }
    cursor.consume(exception_bytes);

    Ok(out)
}

/// Decompress data produced by [`compress`]. `len` is the original uncompressed length.
#[cfg(test)]
fn decompress(compressed: &[u8], len: usize) -> Vec<u8> {
    let saved = len >> 2;
    let compressed_len = len - saved;

    let mut out = vec![0u8; len];

    // 1. Copy packed bytes
    out[..compressed_len].copy_from_slice(&compressed[..compressed_len]);

    // 2. Restore the leading 2 bits of each packed byte into whole bytes
    for i in 0..saved {
        out[compressed_len + i] = ((out[i] & 0xC0) >> 2)
            | ((out[saved + i] & 0xC0) >> 4)
            | ((out[(saved << 1) + i] & 0xC0) >> 6);
    }

    // 3. Move back to original range
    for b in out.iter_mut().take(len) {
        let v = *b;
        *b = ((v & 0x1F) | 0x20 | ((v & 0x20) << 1)).wrapping_sub(1);
    }

    // 4. Restore exceptions
    let mut pos = compressed_len;
    let mut cursor = &compressed[pos..];
    let num_exceptions = varint::read_vint(&mut cursor).unwrap();
    pos += (compressed.len() - pos) - cursor.len();

    let mut i: usize = 0;
    for _ in 0..num_exceptions {
        i += compressed[pos] as usize;
        pos += 1;
        out[i] = compressed[pos];
        pos += 1;
    }

    out
}

#[cfg(test)]
mod tests {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    use super::*;

    fn round_trip(input: &[u8]) -> bool {
        round_trip_len(input, input.len())
    }

    fn round_trip_len(input: &[u8], len: usize) -> bool {
        if let Some(compressed) = compress(input, len) {
            assert_lt!(compressed.len(), len);
            let decompressed = decompress(&compressed, len);
            assert_eq!(&decompressed[..], &input[..len]);
            true
        } else {
            false
        }
    }

    #[test]
    fn test_simple() {
        assert!(!round_trip(b""));
        assert!(!round_trip(b"ab1"));
        assert!(!round_trip(b"ab1cdef"));
        assert!(round_trip(b"ab1cdefg"));
        assert!(!round_trip(b"ab1cdEfg")); // too many exceptions
        assert!(round_trip(b"ab1cdefg"));
        // 1 exception, but enough chars to be worth encoding
        assert!(round_trip(b"ab1.dEfg427hiogchio:'nwm un!94twxz"));
    }

    #[test]
    fn test_not_really_simple() {
        let input = b"cion1cion_desarrollociones_oraclecionesnaturacionesnatura2tedppsa-integrationdemotiontion cloud gen2tion instance - dev1tion instance - testtion-devbtion-instancetion-prdtion-promerication-qation064533tion535217tion697401tion761348tion892818tion_matrationcauto_simmonsintgic_testtioncloudprodictioncloudservicetiongateway10tioninstance-jtsundatamartprd??o";
        round_trip(input);
    }

    #[test]
    fn test_far_away_exceptions() {
        let mut s = String::from("01W");
        for _ in 0..300 {
            s.push('a');
        }
        s.push_str("W.");
        assert!(round_trip(s.as_bytes()));
    }

    #[test]
    fn test_all_compressible() {
        // All lowercase letters — should always compress
        let input = b"abcdefghijklmnopqrstuvwxyz0123456789.-_";
        assert!(round_trip(input));
    }

    #[test]
    fn test_random_compressible_ascii() {
        for iter in 0..100 {
            let len = 8 + (iter * 7) % 200;
            let mut bytes = vec![0u8; len];
            for (i, byte) in bytes.iter_mut().enumerate().take(len) {
                let mut hasher = DefaultHasher::new();
                (iter, i).hash(&mut hasher);
                let seed = hasher.finish();
                // Generate compressible byte: map to [0x1F,0x3F) or [0x5F,0x7F)
                let b = (seed % 32) as u8;
                let v = b | 0x20 | ((b & 0x20) << 1);
                *byte = v.wrapping_sub(1);
            }
            assert!(round_trip(&bytes), "failed at iter {}", iter);
        }
    }

    #[test]
    fn test_random_compressible_with_exceptions() {
        for iter in 0..100 {
            let len = 64 + (iter * 13) % 500;
            let max_exceptions = len >> 5;
            let mut exceptions = 0;
            let mut bytes = vec![0u8; len];
            for (i, byte) in bytes.iter_mut().enumerate().take(len) {
                let mut hasher = DefaultHasher::new();
                (iter, i, "exc").hash(&mut hasher);
                let seed = hasher.finish();
                if exceptions < max_exceptions && seed.is_multiple_of(50) {
                    *byte = (seed % 256) as u8;
                    exceptions += 1;
                } else {
                    let b = (seed % 32) as u8;
                    let v = b | 0x20 | ((b & 0x20) << 1);
                    *byte = v.wrapping_sub(1);
                }
            }
            assert!(round_trip(&bytes), "failed at iter {}", iter);
        }
    }

    #[test]
    fn test_all_uppercase_fails() {
        // All uppercase — too many exceptions
        assert!(!round_trip(b"ABCDEFGH"));
    }

    #[test]
    fn test_mixed_case_long_enough() {
        // Mostly lowercase with a few uppercase — should compress if enough chars
        let mut input = vec![0u8; 256];
        for (i, byte) in input.iter_mut().enumerate().take(256) {
            *byte = b'a' + (i % 26) as u8;
        }
        // Sprinkle a few uppercase (exceptions)
        input[50] = b'A';
        input[100] = b'B';
        input[150] = b'C';
        assert!(round_trip(&input));
    }
}
