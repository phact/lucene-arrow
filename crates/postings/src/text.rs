// SPDX-License-Identifier: Apache-2.0

//! Text analysis for the write path (P9): "standard-lite" tokenizer —
//! Unicode-alphanumeric runs, lowercased, length-capped. Deterministic
//! and documented so the query side can match it exactly (BM25 parity
//! requires index-time and query-time analysis to agree; SPEC §10 risk
//! register). Markdown needs no dedicated stripping for BM25: its syntax
//! is punctuation and dissolves under alphanumeric-run tokenization.
//!
//! `int_to_byte4` is Lucene's `SmallFloat` norm encoding (ported —
//! Bearing's copy is pub(crate)); BM25 stores the encoded field length
//! per doc as the norm.

pub const MAX_TOKEN_LEN: usize = 255;

/// Tokenize into lowercase alphanumeric runs. Yields borrowed slices when
/// already lowercase (common case) — collected as owned for simplicity.
pub fn tokenize(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        if ch.is_alphanumeric() {
            for l in ch.to_lowercase() {
                cur.push(l);
            }
        } else if !cur.is_empty() {
            if cur.len() <= MAX_TOKEN_LEN {
                out.push(std::mem::take(&mut cur));
            } else {
                cur.clear();
            }
        }
    }
    if !cur.is_empty() && cur.len() <= MAX_TOKEN_LEN {
        out.push(cur);
    }
    out
}

// --- SmallFloat (Lucene org.apache.lucene.util.SmallFloat) ---

const MAX_INT4: u32 = 231; // longToInt4(Integer.MAX_VALUE)
pub const NUM_FREE_VALUES: u32 = 255 - MAX_INT4; // 24

/// 4-bit-mantissa/3-bit-exponent encoding; order-preserving.
pub fn long_to_int4(i: i64) -> i32 {
    debug_assert!(i >= 0);
    let num_bits = 64 - (i as u64).leading_zeros();
    if num_bits < 4 {
        i as i32
    } else {
        let shift = num_bits - 4;
        let mut encoded = (i as u64 >> shift) as i32;
        encoded &= 0x07;
        encoded |= (shift as i32 + 1) << 3;
        encoded
    }
}

/// Lucene `SmallFloat.intToByte4`: the BM25 norm byte for a field length.
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

/// Inverse of [`long_to_int4`].
pub const fn int4_to_long(i: u32) -> i64 {
    let bits = (i & 0x07) as i64;
    let shift = (i >> 3) as i32 - 1;
    if shift == -1 { bits } else { (bits | 0x08) << shift }
}

/// Inverse of [`int_to_byte4`] (used by BM25 scoring: norm byte → length).
pub fn byte4_to_int(b: u8) -> i32 {
    let i = b as u32;
    if i < NUM_FREE_VALUES {
        i as i32
    } else {
        (int4_to_long(i - NUM_FREE_VALUES) + NUM_FREE_VALUES as i64) as i32
    }
}

/// Zero-allocation tokenizer: same tokens as [`tokenize`], yielded via a
/// reusable scratch buffer (no per-token String). The hot path for
/// [`crate::build::build_parallel`]; identity with `tokenize` is gated
/// in tests.
pub fn for_each_token(text: &str, mut f: impl FnMut(&str)) {
    let mut scratch = String::with_capacity(64);
    let bytes = text.as_bytes();
    let mut i = 0usize;
    let flush = |scratch: &mut String, f: &mut dyn FnMut(&str)| {
        if !scratch.is_empty() {
            if scratch.len() <= MAX_TOKEN_LEN {
                f(scratch);
            }
            scratch.clear();
        }
    };
    while i < bytes.len() {
        let b = bytes[i];
        if b < 0x80 {
            // ASCII fast path — byte-level classify + lowercase.
            if b.is_ascii_alphanumeric() {
                scratch.push(b.to_ascii_lowercase() as char);
            } else {
                flush(&mut scratch, &mut f);
            }
            i += 1;
        } else {
            // Non-ASCII: decode one char, full Unicode rules.
            let ch = text[i..].chars().next().unwrap();
            if ch.is_alphanumeric() {
                for l in ch.to_lowercase() {
                    scratch.push(l);
                }
            } else {
                flush(&mut scratch, &mut f);
            }
            i += ch.len_utf8();
        }
    }
    flush(&mut scratch, &mut f);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_float_roundtrips_and_orders() {
        // Free values are identity.
        for i in 0..24 {
            assert_eq!(int_to_byte4(i) as i32, i);
        }
        // Order-preserving and roundtrip-consistent (byte4_to_int is the
        // canonical decoded length Lucene uses in BM25).
        let mut prev = -1i32;
        for i in [0, 1, 23, 24, 25, 100, 255, 1000, 65535, i32::MAX] {
            let b = int_to_byte4(i) as i32;
            assert!(b >= prev, "ordering broke at {i}");
            prev = b;
            let back = byte4_to_int(int_to_byte4(i));
            assert!(back <= i, "decode must not exceed input ({i} -> {back})");
        }
        assert_eq!(int_to_byte4(i32::MAX), 255);
    }

    #[test]
    fn for_each_token_matches_tokenize() {
        for text in [
            "# Hello, *World*! `code_x` [link](http://a.b/c)",
            "Ünïcode ΣΙΓΜΑ 42x ss ß İ",
            "",
            "one",
        ] {
            let mut got = Vec::new();
            for_each_token(text, |t| got.push(t.to_string()));
            assert_eq!(got, tokenize(text), "{text:?}");
        }
    }

    #[test]
    fn tokenizer_is_standard_lite() {
        assert_eq!(
            tokenize("# Hello, *World*! `code_x` [link](http://a.b/c)"),
            ["hello", "world", "code", "x", "link", "http", "a", "b", "c"]
        );
        assert_eq!(tokenize("Ünïcode ΣΙΓΜΑ 42x"), ["ünïcode", "σιγμα", "42x"]);
    }
}
