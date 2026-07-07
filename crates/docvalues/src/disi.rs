// SPDX-License-Identifier: Apache-2.0

//! IndexedDISI: which documents carry a value (SPEC §7.1 sparse validity).
//!
//! 65536-doc blocks, each ALL / DENSE (bitmap + optional rank table) /
//! SPARSE (doc-id shorts), followed by a NO_MORE_DOCS sentinel block and a
//! jump table. Write side mirrors Bearing's `indexed_disi.rs`
//! byte-for-byte; read side is a bulk decoder straight into an Arrow-style
//! validity bitmap (no dense intermediate, SPEC §11.3).

use lucene_arrow_core::cursor::Cursor;
use lucene_arrow_core::{Error, Result};

pub const BLOCK_SIZE: u32 = 65536;
const DENSE_BLOCK_LONGS: usize = BLOCK_SIZE as usize / 64; // 1024
const MAX_ARRAY_LENGTH: u32 = (1 << 12) - 1; // 4095: SPARSE if ≤, DENSE if >
const NO_MORE_DOCS: i32 = i32::MAX;
pub const DEFAULT_DENSE_RANK_POWER: u8 = 9;

/// Serialize the set `doc_ids` (sorted, unique) as IndexedDISI blocks +
/// sentinel + jump table, appended to `out`. Returns the jump-table entry
/// count for the `.dvm` metadata.
pub fn write_bit_set(doc_ids: &[u32], out: &mut Vec<u8>) -> Result<i16> {
    write_bit_set_with_rank_power(doc_ids, out, DEFAULT_DENSE_RANK_POWER as i8)
}

fn write_bit_set_with_rank_power(doc_ids: &[u32], out: &mut Vec<u8>, dense_rank_power: i8) -> Result<i16> {
    if !(7..=15).contains(&dense_rank_power) && dense_rank_power != -1 {
        return Err(Error::invalid(format!("denseRankPower must be 7-15 or -1, got {dense_rank_power}")));
    }
    let origo = out.len();
    let mut total_cardinality: u32 = 0;
    let mut jumps: Vec<(u32, u32)> = Vec::new(); // (cumulative index, offset) per block
    let mut last_block: i32 = 0;

    let mut i = 0;
    while i < doc_ids.len() {
        let block = (doc_ids[i] >> 16) as i32;
        let mut buffer = [0i64; DENSE_BLOCK_LONGS];
        while i < doc_ids.len() && (doc_ids[i] >> 16) as i32 == block {
            let within = doc_ids[i] & 0xFFFF;
            buffer[(within >> 6) as usize] |= 1i64 << (within & 63);
            i += 1;
        }
        let cardinality: u32 = buffer.iter().map(|w| w.count_ones()).sum();

        let offset = (out.len() - origo) as u32;
        add_jumps(&mut jumps, offset, total_cardinality, last_block, block + 1);
        last_block = block + 1;

        flush_block(block, &buffer, cardinality, dense_rank_power, out);
        total_cardinality += cardinality;
    }

    // NO_MORE_DOCS sentinel block (bit 65535 of block 32767).
    let offset = (out.len() - origo) as u32;
    add_jumps(&mut jumps, offset, total_cardinality, last_block, last_block + 1);
    let mut sentinel = [0i64; DENSE_BLOCK_LONGS];
    let within = NO_MORE_DOCS & 0xFFFF;
    sentinel[(within >> 6) as usize] |= 1i64 << (within & 63);
    flush_block(NO_MORE_DOCS >> 16, &sentinel, 1, dense_rank_power, out);

    // Jump table.
    let mut count = last_block + 1;
    if count == 2 {
        count = 0; // single real block: table not worth storing
    }
    for &(index, offset) in &jumps[..count as usize] {
        out.extend_from_slice(&(index as i32).to_le_bytes());
        out.extend_from_slice(&(offset as i32).to_le_bytes());
    }
    Ok(count as i16)
}

fn flush_block(block: i32, buffer: &[i64; DENSE_BLOCK_LONGS], cardinality: u32, dense_rank_power: i8, out: &mut Vec<u8>) {
    out.extend_from_slice(&(block as i16).to_le_bytes());
    out.extend_from_slice(&((cardinality - 1) as i16).to_le_bytes());

    if cardinality > MAX_ARRAY_LENGTH {
        if cardinality != BLOCK_SIZE {
            // DENSE
            if dense_rank_power != -1 {
                out.extend_from_slice(&create_rank(buffer, dense_rank_power));
            }
            for &word in buffer.iter() {
                out.extend_from_slice(&word.to_le_bytes());
            }
        }
        // ALL: header only
    } else {
        // SPARSE: each set bit's low 16 bits
        for (word_idx, &word) in buffer.iter().enumerate() {
            let mut bits = word as u64;
            while bits != 0 {
                let bit = bits.trailing_zeros();
                let doc_in_block = word_idx as u32 * 64 + bit;
                out.extend_from_slice(&(doc_in_block as i16).to_le_bytes());
                bits &= bits - 1;
            }
        }
    }
}

/// Rank table for a DENSE block: one 2-byte big-endian entry per
/// 2^dense_rank_power bits.
fn create_rank(buffer: &[i64; DENSE_BLOCK_LONGS], dense_rank_power: i8) -> Vec<u8> {
    let longs_per_rank = 1usize << (dense_rank_power - 6);
    let rank_mark = longs_per_rank - 1;
    let rank_index_shift = (dense_rank_power - 7) as usize;
    let mut rank = vec![0u8; DENSE_BLOCK_LONGS >> rank_index_shift];
    let mut bit_count: u32 = 0;
    for (word, &w) in buffer.iter().enumerate() {
        if word & rank_mark == 0 {
            rank[word >> rank_index_shift] = (bit_count >> 8) as u8;
            rank[(word >> rank_index_shift) + 1] = (bit_count & 0xFF) as u8;
        }
        bit_count += w.count_ones();
    }
    rank
}

fn add_jumps(jumps: &mut Vec<(u32, u32)>, offset: u32, index: u32, start_block: i32, end_block: i32) {
    if jumps.len() < end_block as usize {
        jumps.resize(end_block as usize, (0, 0));
    }
    for b in start_block..end_block {
        jumps[b as usize] = (index, offset);
    }
}

/// Byte length of the rank table a DENSE block carries.
pub fn dense_rank_bytes(dense_rank_power: u8) -> usize {
    if dense_rank_power == 0xFF { 0 } else { DENSE_BLOCK_LONGS >> (dense_rank_power - 7) }
}

/// Bulk-decode a DISI region into an Arrow validity bitmap.
///
/// `bytes` is the region at `[docsWithFieldOffset, +docsWithFieldLength)`
/// (blocks + sentinel + jump table; the tail is ignored — jump tables are
/// for random access, we scan). Returns LSB-first 64-bit validity words
/// covering `[0, max_doc)`; exactly `num_values` bits are set.
pub fn decode(bytes: &[u8], num_values: u64, max_doc: u32, dense_rank_power: u8) -> Result<Vec<u64>> {
    let mut bitmap = vec![0u64; (max_doc as usize).div_ceil(64)];
    let mut seen: u64 = 0;
    let mut c = Cursor::new(bytes);
    let rank_bytes = dense_rank_bytes(dense_rank_power);

    while seen < num_values {
        let block = c.le_i16()? as u16 as u32;
        let cardinality = (c.le_i16()? as u16 as u32) + 1;
        let base = block * BLOCK_SIZE;
        if base >= max_doc {
            return Err(Error::corrupt(format!("DISI block {block} beyond max_doc {max_doc}")));
        }
        if cardinality == BLOCK_SIZE {
            // ALL
            for d in 0..BLOCK_SIZE {
                let doc = base + d;
                bitmap[(doc / 64) as usize] |= 1u64 << (doc % 64);
            }
        } else if cardinality > MAX_ARRAY_LENGTH {
            // DENSE: rank table (skipped), then always 1024 LE longs (the
            // block covers 65536 docs even when max_doc ends mid-block),
            // word-aligned to the bitmap because base is a multiple of 65536.
            c.skip(rank_bytes)?;
            let word_base = (base / 64) as usize;
            for w in 0..DENSE_BLOCK_LONGS {
                let word = c.le_i64()? as u64;
                if word_base + w < bitmap.len() {
                    bitmap[word_base + w] = word;
                } else if word != 0 {
                    return Err(Error::corrupt(format!(
                        "DISI DENSE block {block} sets bits beyond max_doc {max_doc}"
                    )));
                }
            }
        } else {
            // SPARSE
            for _ in 0..cardinality {
                let within = c.le_i16()? as u16 as u32;
                let doc = base + within;
                if doc >= max_doc {
                    return Err(Error::corrupt(format!("DISI doc {doc} beyond max_doc {max_doc}")));
                }
                bitmap[(doc / 64) as usize] |= 1u64 << (doc % 64);
            }
        }
        seen += cardinality as u64;
    }
    if seen != num_values {
        return Err(Error::corrupt(format!("DISI cardinality {seen} != expected {num_values}")));
    }
    Ok(bitmap)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(doc_ids: &[u32], max_doc: u32) {
        let mut out = Vec::new();
        write_bit_set(doc_ids, &mut out).unwrap();
        let bitmap = decode(&out, doc_ids.len() as u64, max_doc, DEFAULT_DENSE_RANK_POWER).unwrap();
        let mut decoded = Vec::new();
        for (w, &word) in bitmap.iter().enumerate() {
            let mut bits = word;
            while bits != 0 {
                decoded.push(w as u32 * 64 + bits.trailing_zeros());
                bits &= bits - 1;
            }
        }
        assert_eq!(decoded, doc_ids);
    }

    #[test]
    fn sparse_block() {
        round_trip(&[3, 700, 4095, 65535], 70000);
    }

    #[test]
    fn dense_block() {
        let docs: Vec<u32> = (0..20000).map(|i| i * 3).collect(); // card 20000 in block 0
        round_trip(&docs, 65536);
    }

    #[test]
    fn all_block_plus_sparse_block() {
        let mut docs: Vec<u32> = (0..65536).collect(); // ALL block 0
        docs.extend([65536 + 10, 65536 + 99]); // SPARSE block 1
        round_trip(&docs, 140000);
    }

    #[test]
    fn skips_empty_blocks() {
        round_trip(&[5, 262144 + 7], 262200); // blocks 0 and 4; 1-3 empty
    }
}
