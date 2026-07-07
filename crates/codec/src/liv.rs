// SPDX-License-Identifier: Apache-2.0

//! Live docs (`.liv`, Lucene90LiveDocsFormat): index header (suffix =
//! delete generation in base 36) + `ceil(maxDoc/64)` LE longs (FixedBitSet,
//! **1 = live**) + footer. Never skip these: aggregates over tombstones are
//! silent corruption (SPEC §7.5).

use lucene_arrow_core::cursor::{Cursor, read_index_header, verify_footer};
use lucene_arrow_core::{Error, Result};

pub const CODEC_NAME: &str = "Lucene90LiveDocs";
pub const VERSION: i32 = 0;
pub const EXTENSION: &str = "liv";

/// Java `Long.toString(gen, 36)` for non-negative generations.
pub fn base36(mut v: i64) -> String {
    debug_assert!(v >= 0);
    if v == 0 {
        return "0".to_string();
    }
    let digits = b"0123456789abcdefghijklmnopqrstuvwxyz";
    let mut out = Vec::new();
    while v > 0 {
        out.push(digits[(v % 36) as usize]);
        v /= 36;
    }
    out.reverse();
    String::from_utf8(out).expect("ascii")
}

/// `.liv` file name for a segment + delete generation.
pub fn file_name(segment: &str, del_gen: i64) -> String {
    format!("{segment}_{}.{EXTENSION}", base36(del_gen))
}

/// Parse a `.liv` file into live-doc words (LSB-first, 1 = live), verifying
/// the segment id, generation suffix, and that the tombstone count matches
/// `del_count` (the same cross-check Java's reader does).
pub fn read_live_docs(
    bytes: &[u8],
    max_doc: u32,
    segment_id: &[u8; 16],
    del_gen: i64,
    del_count: i32,
) -> Result<Vec<u64>> {
    let header = read_index_header(bytes, CODEC_NAME, VERSION, VERSION)?;
    if header.segment_id != *segment_id {
        return Err(Error::corrupt(".liv segment id mismatch"));
    }
    if header.suffix != base36(del_gen) {
        return Err(Error::corrupt(format!(
            ".liv generation suffix {:?} != expected {:?}",
            header.suffix,
            base36(del_gen)
        )));
    }
    verify_footer(bytes)?;

    let num_words = (max_doc as usize).div_ceil(64);
    let mut c = Cursor::at(bytes, header.length);
    let mut words = Vec::with_capacity(num_words);
    for _ in 0..num_words {
        words.push(c.le_i64()? as u64);
    }

    let live: u32 = words
        .iter()
        .enumerate()
        .map(|(w, &word)| {
            // Mask tail bits beyond max_doc in the last word.
            let valid_bits = (max_doc as usize - w * 64).min(64);
            let mask = if valid_bits == 64 { u64::MAX } else { (1u64 << valid_bits) - 1 };
            (word & mask).count_ones()
        })
        .sum();
    let deleted = max_doc - live;
    if deleted as i32 != del_count {
        return Err(Error::corrupt(format!(
            ".liv says {deleted} deleted docs, segments_N says {del_count}"
        )));
    }
    Ok(words)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base36_matches_java() {
        assert_eq!(base36(0), "0");
        assert_eq!(base36(1), "1");
        assert_eq!(base36(35), "z");
        assert_eq!(base36(36), "10");
        assert_eq!(base36(46655), "zzz");
    }
}
