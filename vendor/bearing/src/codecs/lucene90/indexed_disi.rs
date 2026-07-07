// SPDX-License-Identifier: Apache-2.0
//! IndexedDISI bitset encoding for sparse document presence.
//!
//! Encodes sorted document IDs into 65536-doc blocks using one of three strategies:
//! - **ALL**: block has all 65536 docs set (no data after header)
//! - **DENSE**: block has 4096–65535 docs set (rank table + bitmap)
//! - **SPARSE**: block has 1–4095 docs set (array of 16-bit offsets)
//!
//! Used by doc values and norms to support sparse fields (present in some but not all documents).

use std::io;

use crate::store::{IndexInput, IndexOutput};

/// Number of doc IDs per block.
const BLOCK_SIZE: i32 = 65536;

/// Number of `i64` words in a dense block bitmap.
const DENSE_BLOCK_LONGS: usize = BLOCK_SIZE as usize / 64; // 1024

/// Maximum cardinality for SPARSE encoding; above this threshold, DENSE is used.
const MAX_ARRAY_LENGTH: i32 = (1 << 12) - 1; // 4095

/// Sentinel value indicating no more documents.
const NO_MORE_DOCS: i32 = i32::MAX;

/// Default rank power for DENSE blocks: rank entry every 2^9 = 512 doc IDs.
pub const DEFAULT_DENSE_RANK_POWER: i8 = 9;

/// Writes an IndexedDISI bitset for the given sorted doc IDs.
///
/// Returns the jump table entry count (stored in metadata).
pub fn write_bit_set(doc_ids: &[i32], max_doc: i32, out: &mut dyn IndexOutput) -> io::Result<i16> {
    write_bit_set_with_rank_power(doc_ids, max_doc, out, DEFAULT_DENSE_RANK_POWER)
}

/// Writes an IndexedDISI bitset with a specified dense rank power.
///
/// `dense_rank_power` controls rank granularity for DENSE blocks (valid: 7–15, or -1 to disable).
/// Returns the jump table entry count.
fn write_bit_set_with_rank_power(
    doc_ids: &[i32],
    _max_doc: i32,
    out: &mut dyn IndexOutput,
    dense_rank_power: i8,
) -> io::Result<i16> {
    let origo = out.file_pointer();

    if !(7..=15).contains(&dense_rank_power) && dense_rank_power != -1 {
        return Err(io::Error::other(format!(
            "Acceptable values for denseRankPower are 7-15. The provided power was {}",
            dense_rank_power
        )));
    }

    let mut total_cardinality: i32 = 0;
    let mut jumps: Vec<(i32, i32)> = Vec::new(); // (index, offset) per block
    let mut last_block: i32 = 0;

    // Process doc IDs block by block
    let mut i = 0;
    while i < doc_ids.len() {
        let doc = doc_ids[i];
        let block = doc >> 16;

        // Fill buffer with all docs in this block
        let mut buffer = [0i64; DENSE_BLOCK_LONGS];
        while i < doc_ids.len() && (doc_ids[i] >> 16) == block {
            let within_block = doc_ids[i] & 0xFFFF;
            buffer[(within_block >> 6) as usize] |= 1i64 << (within_block & 63);
            i += 1;
        }

        let block_cardinality = buffer.iter().map(|w| w.count_ones() as i32).sum::<i32>();

        // Add jump entries for blocks from last_block to block+1
        let offset = (out.file_pointer() - origo) as i32;
        add_jumps(&mut jumps, offset, total_cardinality, last_block, block + 1);
        last_block = block + 1;

        // Flush block
        flush(block, &buffer, block_cardinality, dense_rank_power, out)?;

        total_cardinality += block_cardinality;
    }

    // NO_MORE_DOCS sentinel block
    let offset = (out.file_pointer() - origo) as i32;
    add_jumps(
        &mut jumps,
        offset,
        total_cardinality,
        last_block,
        last_block + 1,
    );

    let mut sentinel_buffer = [0i64; DENSE_BLOCK_LONGS];
    let nmd_within = NO_MORE_DOCS & 0xFFFF; // 65535
    sentinel_buffer[(nmd_within >> 6) as usize] |= 1i64 << (nmd_within & 63);
    flush(
        NO_MORE_DOCS >> 16, // 32767
        &sentinel_buffer,
        1,
        dense_rank_power,
        out,
    )?;

    // Flush jump table
    flush_block_jumps(&jumps, last_block + 1, out)
}

/// Writes a single block to the output.
fn flush(
    block: i32,
    buffer: &[i64; DENSE_BLOCK_LONGS],
    cardinality: i32,
    dense_rank_power: i8,
    out: &mut dyn IndexOutput,
) -> io::Result<()> {
    debug_assert!((0..BLOCK_SIZE).contains(&block));
    debug_assert!(cardinality > 0 && cardinality <= BLOCK_SIZE);

    // Block header
    out.write_le_short(block as i16)?;
    out.write_le_short((cardinality - 1) as i16)?;

    if cardinality > MAX_ARRAY_LENGTH {
        if cardinality != BLOCK_SIZE {
            // DENSE block
            if dense_rank_power != -1 {
                let rank = create_rank(buffer, dense_rank_power);
                out.write_all(&rank)?;
            }
            for &word in buffer.iter() {
                out.write_le_long(word)?;
            }
        }
        // ALL block: no data needed
    } else {
        // SPARSE block: write each set bit's position as a short
        for (word_idx, &word) in buffer.iter().enumerate() {
            let mut bits = word as u64;
            while bits != 0 {
                let bit = bits.trailing_zeros() as i32;
                let doc_in_block = (word_idx as i32) * 64 + bit;
                out.write_le_short(doc_in_block as i16)?;
                bits &= bits - 1; // clear lowest set bit
            }
        }
    }

    Ok(())
}

/// Creates a rank table for a DENSE block.
///
/// One rank entry per `2^dense_rank_power` bits, each entry is 2 bytes (big-endian within entry).
fn create_rank(buffer: &[i64; DENSE_BLOCK_LONGS], dense_rank_power: i8) -> Vec<u8> {
    let longs_per_rank = 1usize << (dense_rank_power - 6);
    let rank_mark = longs_per_rank - 1;
    let rank_index_shift = (dense_rank_power - 7) as usize;
    let rank_len = DENSE_BLOCK_LONGS >> rank_index_shift;
    let mut rank = vec![0u8; rank_len];

    let mut bit_count: i32 = 0;
    for word in 0..DENSE_BLOCK_LONGS {
        if (word & rank_mark) == 0 {
            // Big-endian within the 2-byte rank entry
            rank[word >> rank_index_shift] = (bit_count >> 8) as u8;
            rank[(word >> rank_index_shift) + 1] = (bit_count & 0xFF) as u8;
        }
        bit_count += buffer[word].count_ones() as i32;
    }

    rank
}

/// Adds jump table entries for block range [start_block, end_block).
fn add_jumps(
    jumps: &mut Vec<(i32, i32)>,
    offset: i32,
    index: i32,
    start_block: i32,
    end_block: i32,
) {
    if jumps.len() < end_block as usize {
        jumps.resize(end_block as usize, (0, 0));
    }
    for b in start_block..end_block {
        jumps[b as usize] = (index, offset);
    }
}

/// Flushes the jump table and returns the entry count.
fn flush_block_jumps(
    jumps: &[(i32, i32)],
    block_count: i32,
    out: &mut dyn IndexOutput,
) -> io::Result<i16> {
    let mut count = block_count;
    if count == 2 {
        // Single real block + NO_MORE_DOCS: not worth storing jump table
        count = 0;
    }

    for &(index, offset) in &jumps[..count as usize] {
        out.write_le_int(index)?; // cumulative doc count
        out.write_le_int(offset)?; // byte offset from origo
    }

    Ok(count as i16)
}

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------

/// Block encoding method for the current block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Method {
    Sparse,
    Dense,
    All,
}

/// Disk-based document ID set iterator with ordinal tracking.
///
/// Returns documents in sorted order and tracks the ordinal index (position
/// among all set documents). Used by norms and doc values readers to map
/// doc IDs to sparse value ordinals.
pub struct IndexedDISI<'a> {
    slice: IndexInput<'a>,
    jump_table: Option<IndexInput<'a>>,
    jump_table_entry_count: i32,
    dense_rank_power: i8,
    dense_rank_table: Option<Vec<u8>>,

    // Block state
    block: i32,
    block_end: usize,
    dense_bitmap_offset: usize,
    next_block_index: i32,
    method: Method,
    doc: i32,
    index: i32,

    // SPARSE state
    exists: bool,
    next_exist_doc_in_block: i32,

    // DENSE state
    word: i64,
    word_index: i32,
    number_of_ones: i32,
    dense_origo_index: i32,

    // ALL state
    gap: i32,

    cost: i64,
}

impl<'a> IndexedDISI<'a> {
    /// Creates an `IndexedDISI` by slicing block data and jump table from a parent input.
    pub(crate) fn new(
        data: &IndexInput<'a>,
        offset: usize,
        length: usize,
        jump_table_entry_count: i16,
        dense_rank_power: u8,
        cost: i64,
    ) -> io::Result<Self> {
        let block_slice = create_block_slice(data, offset, length, jump_table_entry_count as i32)?;
        let jump_table = create_jump_table(data, offset, length, jump_table_entry_count as i32)?;
        Self::from_parts(
            block_slice,
            jump_table,
            jump_table_entry_count as i32,
            dense_rank_power as i8,
            cost,
        )
    }

    /// Creates an `IndexedDISI` from pre-sliced block data and jump table.
    fn from_parts(
        slice: IndexInput<'a>,
        jump_table: Option<IndexInput<'a>>,
        jump_table_entry_count: i32,
        dense_rank_power: i8,
        cost: i64,
    ) -> io::Result<Self> {
        if !(7..=15).contains(&dense_rank_power) && dense_rank_power != -1 {
            return Err(io::Error::other(format!(
                "Acceptable values for denseRankPower are 7-15. Got {dense_rank_power}"
            )));
        }

        let rank_index_shift = dense_rank_power - 7;
        let dense_rank_table = if dense_rank_power == -1 {
            None
        } else {
            Some(vec![0u8; DENSE_BLOCK_LONGS >> rank_index_shift as usize])
        };

        Ok(Self {
            slice,
            jump_table,
            jump_table_entry_count,
            dense_rank_power,
            dense_rank_table,
            block: -1,
            block_end: 0,
            dense_bitmap_offset: 0,
            next_block_index: -1,
            method: Method::Sparse,
            doc: -1,
            index: -1,
            exists: false,
            next_exist_doc_in_block: -1,
            word: 0,
            word_index: -1,
            number_of_ones: 0,
            dense_origo_index: 0,
            gap: 0,
            cost,
        })
    }

    /// Returns the current document ID, or -1 if not positioned.
    pub fn doc_id(&self) -> i32 {
        self.doc
    }

    /// Returns the ordinal index of the current document among all set documents.
    pub fn index(&self) -> i32 {
        self.index
    }

    /// Returns the estimated number of documents in this set.
    pub fn cost(&self) -> i64 {
        self.cost
    }

    /// Advances to the first document at or after `target`.
    ///
    /// Returns `Ok(Some(doc))` if found, `Ok(None)` if no more documents.
    pub fn advance(&mut self, target: i32) -> io::Result<Option<i32>> {
        let target_block = target & 0xFFFF_0000u32 as i32;
        if self.block < target_block {
            self.advance_block(target_block)?;
        }
        if self.block == target_block {
            if self.advance_within_block(target)? {
                return Ok(Some(self.doc));
            }
            self.read_block_header()?;
        }
        let found = self.advance_within_block(self.block)?;
        if !found {
            self.doc = NO_MORE_DOCS;
            return Ok(None);
        }
        if self.doc == NO_MORE_DOCS {
            return Ok(None);
        }
        Ok(Some(self.doc))
    }

    /// Checks whether `target` is in the set and positions the index accordingly.
    ///
    /// After calling, `index()` returns the ordinal if the doc exists.
    pub fn advance_exact(&mut self, target: i32) -> io::Result<bool> {
        let target_block = target & 0xFFFF_0000u32 as i32;
        if self.block < target_block {
            self.advance_block(target_block)?;
        }
        let found = self.block == target_block && self.advance_exact_within_block(target)?;
        self.doc = target;
        Ok(found)
    }

    /// Advances to the next document.
    pub fn next_doc(&mut self) -> io::Result<Option<i32>> {
        self.advance(self.doc + 1)
    }

    fn advance_block(&mut self, target_block: i32) -> io::Result<()> {
        let block_index = target_block >> 16;

        // Use jump table if destination is 2+ blocks ahead
        if let Some(ref mut jt) = self.jump_table
            && block_index >= (self.block >> 16) + 2
        {
            let in_range = if block_index < self.jump_table_entry_count {
                block_index
            } else {
                self.jump_table_entry_count - 1
            };
            let jt_offset = in_range as usize * 8;
            jt.seek(jt_offset)?;
            let index = jt.read_le_int()?;
            let offset = jt.read_le_int()?;
            self.next_block_index = index - 1;
            self.slice.seek(offset as usize)?;
            self.read_block_header()?;
            return Ok(());
        }

        // Fallback: iterate blocks forward
        loop {
            self.slice.seek(self.block_end)?;
            self.read_block_header()?;
            if self.block >= target_block {
                break;
            }
        }
        Ok(())
    }

    fn read_block_header(&mut self) -> io::Result<()> {
        self.block = (self.slice.read_le_short()? as u16 as i32) << 16;
        let num_values = 1 + self.slice.read_le_short()? as u16 as i32;
        self.index = self.next_block_index;
        self.next_block_index = self.index + num_values;

        if num_values <= MAX_ARRAY_LENGTH {
            // SPARSE
            self.method = Method::Sparse;
            self.block_end = self.slice.position() + (num_values as usize * 2);
            self.next_exist_doc_in_block = -1;
        } else if num_values == BLOCK_SIZE {
            // ALL
            self.method = Method::All;
            self.block_end = self.slice.position();
            self.gap = self.block - self.index - 1;
        } else {
            // DENSE
            self.method = Method::Dense;
            let rank_table_len = self.dense_rank_table.as_ref().map_or(0, |t| t.len());
            self.dense_bitmap_offset = self.slice.position() + rank_table_len;
            self.block_end = self.dense_bitmap_offset + (1 << 13); // 8192 bytes

            // Load rank table
            if let Some(ref mut table) = self.dense_rank_table {
                self.slice.read_bytes(table)?;
            }
            self.word_index = -1;
            self.number_of_ones = self.index + 1;
            self.dense_origo_index = self.number_of_ones;
        }
        Ok(())
    }

    fn advance_within_block(&mut self, target: i32) -> io::Result<bool> {
        match self.method {
            Method::Sparse => self.advance_within_sparse(target),
            Method::Dense => self.advance_within_dense(target),
            Method::All => {
                self.advance_within_all(target);
                Ok(true)
            }
        }
    }

    fn advance_exact_within_block(&mut self, target: i32) -> io::Result<bool> {
        match self.method {
            Method::Sparse => self.advance_exact_within_sparse(target),
            Method::Dense => self.advance_exact_within_dense(target),
            Method::All => {
                self.advance_exact_within_all(target);
                Ok(true)
            }
        }
    }

    // --- SPARSE block methods ---

    fn advance_within_sparse(&mut self, target: i32) -> io::Result<bool> {
        let target_in_block = target & 0xFFFF;
        while self.index < self.next_block_index {
            let doc = self.slice.read_le_short()? as u16 as i32;
            self.index += 1;
            if doc >= target_in_block {
                self.doc = self.block | doc;
                self.exists = true;
                self.next_exist_doc_in_block = doc;
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn advance_exact_within_sparse(&mut self, target: i32) -> io::Result<bool> {
        let target_in_block = target & 0xFFFF;
        if self.next_exist_doc_in_block > target_in_block {
            return Ok(false);
        }
        if target == self.doc {
            return Ok(self.exists);
        }
        while self.index < self.next_block_index {
            let doc = self.slice.read_le_short()? as u16 as i32;
            self.index += 1;
            if doc >= target_in_block {
                self.next_exist_doc_in_block = doc;
                if doc != target_in_block {
                    self.index -= 1;
                    let pos = self.slice.position() - 2;
                    self.slice.seek(pos)?;
                    break;
                }
                self.exists = true;
                return Ok(true);
            }
        }
        self.exists = false;
        Ok(false)
    }

    // --- DENSE block methods ---

    fn advance_within_dense(&mut self, target: i32) -> io::Result<bool> {
        let target_in_block = target & 0xFFFF;
        let target_word_index = target_in_block >> 6;

        // Rank skip if far enough ahead
        if self.dense_rank_power != -1
            && target_word_index - self.word_index >= (1 << (self.dense_rank_power - 6))
        {
            self.rank_skip(target_in_block)?;
        }

        // Read words up to target
        for _ in (self.word_index + 1)..=target_word_index {
            self.word = self.slice.read_le_long()?;
            self.number_of_ones += self.word.count_ones() as i32;
        }
        self.word_index = target_word_index;

        // Check if target bit or any after it in this word are set
        let shift = target & 63;
        let left_bits = ((self.word as u64) >> shift) as i64;
        if left_bits != 0 {
            self.doc = target + (left_bits as u64).trailing_zeros() as i32;
            self.index = self.number_of_ones - left_bits.count_ones() as i32;
            return Ok(true);
        }

        // Scan forward through remaining words
        while self.word_index + 1 < 1024 {
            self.word_index += 1;
            self.word = self.slice.read_le_long()?;
            if self.word != 0 {
                self.index = self.number_of_ones;
                self.number_of_ones += self.word.count_ones() as i32;
                self.doc = self.block
                    | (self.word_index << 6)
                    | (self.word as u64).trailing_zeros() as i32;
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn advance_exact_within_dense(&mut self, target: i32) -> io::Result<bool> {
        let target_in_block = target & 0xFFFF;
        let target_word_index = target_in_block >> 6;

        // Rank skip if far enough ahead
        if self.dense_rank_power != -1
            && target_word_index - self.word_index >= (1 << (self.dense_rank_power - 6))
        {
            self.rank_skip(target_in_block)?;
        }

        for _ in (self.word_index + 1)..=target_word_index {
            self.word = self.slice.read_le_long()?;
            self.number_of_ones += self.word.count_ones() as i32;
        }
        self.word_index = target_word_index;

        let shift = target & 63;
        let left_bits = ((self.word as u64) >> shift) as i64;
        self.index = self.number_of_ones - left_bits.count_ones() as i32;
        Ok((left_bits & 1) != 0)
    }

    fn rank_skip(&mut self, target_in_block: i32) -> io::Result<()> {
        let table = self.dense_rank_table.as_ref().unwrap();
        let rank_index = target_in_block >> self.dense_rank_power;
        let rank = ((table[(rank_index << 1) as usize] as u16) << 8
            | table[((rank_index << 1) + 1) as usize] as u16) as i32;

        let rank_aligned_word_index = (rank_index << self.dense_rank_power) >> 6;
        self.slice
            .seek(self.dense_bitmap_offset + rank_aligned_word_index as usize * 8)?;
        let rank_word = self.slice.read_le_long()?;
        let dense_noo = rank + rank_word.count_ones() as i32;

        self.word_index = rank_aligned_word_index;
        self.word = rank_word;
        self.number_of_ones = self.dense_origo_index + dense_noo;
        Ok(())
    }

    // --- ALL block methods ---

    fn advance_within_all(&mut self, target: i32) {
        self.doc = target;
        self.index = target - self.gap;
    }

    fn advance_exact_within_all(&mut self, target: i32) {
        self.index = target - self.gap;
    }
}

/// Slices the block data portion (without jump table) from a parent input.
pub(crate) fn create_block_slice<'a>(
    data: &IndexInput<'a>,
    offset: usize,
    length: usize,
    jump_table_entry_count: i32,
) -> io::Result<IndexInput<'a>> {
    let jump_table_bytes = if jump_table_entry_count < 0 {
        0
    } else {
        jump_table_entry_count as usize * 8
    };
    data.view("disi-blocks", offset, length - jump_table_bytes)
}

/// Slices the jump table portion from a parent input, or `None` if no jump table.
pub(crate) fn create_jump_table<'a>(
    data: &IndexInput<'a>,
    offset: usize,
    length: usize,
    jump_table_entry_count: i32,
) -> io::Result<Option<IndexInput<'a>>> {
    if jump_table_entry_count <= 0 {
        Ok(None)
    } else {
        let jump_table_bytes = jump_table_entry_count as usize * 8;
        let jt_offset = offset + length - jump_table_bytes;
        let slice = data.view("disi-jumptable", jt_offset, jump_table_bytes)?;
        Ok(Some(slice))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use crate::store::DataOutput;
    use crate::store::memory::MemoryIndexOutput;
    use assertables::*;

    /// Helper to write a bitset and return the raw bytes and jump table entry count.
    fn write_and_get_bytes(doc_ids: &[i32], max_doc: i32) -> (Vec<u8>, i16) {
        let mut out = MemoryIndexOutput::new("test".to_string());
        let entry_count = write_bit_set(doc_ids, max_doc, &mut out).unwrap();
        (out.bytes().to_vec(), entry_count)
    }

    #[test]
    fn test_empty_input() {
        // No documents set — should still write NO_MORE_DOCS sentinel block
        let (bytes, entry_count) = write_and_get_bytes(&[], 10);

        // Entry count: 0+1=1 block (just the sentinel), blockCount=1 → written as-is
        // Actually: last_block stays 0, so blockCount = 0+1 = 1
        // blockCount=1 is not 2, so it writes 1 jump entry
        assert_eq!(entry_count, 1);

        // Sentinel block: blockID=32767 (NO_MORE_DOCS >>> 16), cardinality-1=0
        assert_eq!(&bytes[0..2], &32767i16.to_le_bytes());
        assert_eq!(&bytes[2..4], &0i16.to_le_bytes());

        // SPARSE: single entry, doc 65535 within block
        assert_eq!(&bytes[4..6], &(-1i16).to_le_bytes()); // 65535 as i16 = -1

        // Jump table: 1 entry (index=0, offset=0)
        assert_eq!(&bytes[6..10], &0i32.to_le_bytes());
        assert_eq!(&bytes[10..14], &0i32.to_le_bytes());
    }

    #[test]
    fn test_single_doc() {
        let (bytes, entry_count) = write_and_get_bytes(&[42], 100);

        // blockCount = 0+1+1 = 2 (block 0 + sentinel) → optimized to 0
        assert_eq!(entry_count, 0);

        // Block 0: blockID=0, cardinality-1=0
        assert_eq!(&bytes[0..2], &0i16.to_le_bytes());
        assert_eq!(&bytes[2..4], &0i16.to_le_bytes());

        // SPARSE: single entry, doc 42
        assert_eq!(&bytes[4..6], &42i16.to_le_bytes());

        // Sentinel block follows
        assert_eq!(&bytes[6..8], &32767i16.to_le_bytes());
    }

    #[test]
    fn test_sparse_block() {
        // 3 docs in block 0 — well under 4096 threshold
        let doc_ids = vec![10, 100, 1000];
        let (bytes, _) = write_and_get_bytes(&doc_ids, 2000);

        // Block header
        assert_eq!(&bytes[0..2], &0i16.to_le_bytes()); // blockID=0
        assert_eq!(&bytes[2..4], &2i16.to_le_bytes()); // cardinality-1=2

        // SPARSE entries
        assert_eq!(&bytes[4..6], &10i16.to_le_bytes());
        assert_eq!(&bytes[6..8], &100i16.to_le_bytes());
        assert_eq!(&bytes[8..10], &1000i16.to_le_bytes());
    }

    #[test]
    fn test_all_block() {
        // All 65536 docs set — ALL encoding, no data bytes after header
        let doc_ids: Vec<i32> = (0..65536).collect();
        let (bytes, _) = write_and_get_bytes(&doc_ids, 65536);

        // Block header
        assert_eq!(&bytes[0..2], &0i16.to_le_bytes()); // blockID=0
        assert_eq!(&bytes[2..4], &(-1i16).to_le_bytes()); // cardinality-1=65535

        // Next should be sentinel block header (no data for ALL block)
        assert_eq!(&bytes[4..6], &32767i16.to_le_bytes()); // sentinel blockID
    }

    #[test]
    fn test_dense_block() {
        // 5000 docs set — above 4095 threshold, DENSE encoding
        let doc_ids: Vec<i32> = (0..5000).collect();
        let (bytes, _) = write_and_get_bytes(&doc_ids, 65536);

        // Block header
        assert_eq!(&bytes[0..2], &0i16.to_le_bytes()); // blockID=0
        assert_eq!(&bytes[2..4], &4999i16.to_le_bytes()); // cardinality-1=4999

        // Rank table: 128 entries of 2 bytes = 256 bytes
        // First rank entry should be 0 (no bits before position 0)
        assert_eq!(bytes[4], 0);
        assert_eq!(bytes[5], 0);

        // After rank table (256 bytes) comes bitmap (1024 longs = 8192 bytes)
        let bitmap_start = 4 + 256;
        // First long should have bits 0-63 all set
        let first_long =
            i64::from_le_bytes(bytes[bitmap_start..bitmap_start + 8].try_into().unwrap());
        assert_eq!(first_long, -1i64); // all 64 bits set

        // Total block size: 4 (header) + 256 (rank) + 8192 (bitmap) = 8452
        let sentinel_start = 8452;
        assert_eq!(
            &bytes[sentinel_start..sentinel_start + 2],
            &32767i16.to_le_bytes()
        );
    }

    #[test]
    fn test_dense_block_rank_values() {
        // Verify rank table cumulative popcount correctness
        // Set docs 0-511 (first rank entry = 0) and 512-1023 (second rank entry = 512)
        // Plus enough more to be DENSE (need > 4095)
        let doc_ids: Vec<i32> = (0..4100).collect();
        let (bytes, _) = write_and_get_bytes(&doc_ids, 65536);

        // Rank table starts at offset 4 (after header)
        // Entry 0: 0 bits before position 0
        assert_eq!(bytes[4], 0); // high byte
        assert_eq!(bytes[5], 0); // low byte

        // Entry 1: 512 bits in positions 0-511 (8 longs × 64 bits)
        let rank1 = ((bytes[6] as u16) << 8) | bytes[7] as u16;
        assert_eq!(rank1, 512);

        // Entry 2: 1024 bits
        let rank2 = ((bytes[8] as u16) << 8) | bytes[9] as u16;
        assert_eq!(rank2, 1024);
    }

    #[test]
    fn test_multi_block_with_gap() {
        // Docs in block 0 and block 2 (gap at block 1)
        let mut doc_ids: Vec<i32> = vec![0, 1, 2];
        doc_ids.extend(131072..131075); // block 2: docs 131072-131074
        let (bytes, entry_count) = write_and_get_bytes(&doc_ids, 200000);

        // Should have jump entries for blocks 0,1,2 + sentinel = 4
        // blockCount = 3+1 = 4 (last_block after block 2 = 3, +1 for sentinel)
        // Actually: after block 0, last_block=1. After block 2, last_block=3.
        // So blockCount = 3+1 = 4
        assert_eq!(entry_count, 4);

        // Block 0: SPARSE with 3 docs
        assert_eq!(&bytes[0..2], &0i16.to_le_bytes());
        assert_eq!(&bytes[2..4], &2i16.to_le_bytes()); // cardinality-1=2
    }

    #[test]
    fn test_sparse_dense_boundary() {
        // Exactly MAX_ARRAY_LENGTH (4095) docs — should be SPARSE
        let doc_ids: Vec<i32> = (0..4095).collect();
        let (bytes, _) = write_and_get_bytes(&doc_ids, 65536);
        let cardinality_minus_1 = i16::from_le_bytes(bytes[2..4].try_into().unwrap());
        assert_eq!(cardinality_minus_1, 4094);
        // Block data: 4095 shorts = 8190 bytes
        // Sentinel starts at 4 + 8190 = 8194
        assert_eq!(&bytes[8194..8196], &32767i16.to_le_bytes());

        // Exactly MAX_ARRAY_LENGTH + 1 (4096) docs — should be DENSE
        let doc_ids: Vec<i32> = (0..4096).collect();
        let (bytes, _) = write_and_get_bytes(&doc_ids, 65536);
        let cardinality_minus_1 = i16::from_le_bytes(bytes[2..4].try_into().unwrap());
        assert_eq!(cardinality_minus_1, 4095);
        // DENSE: 4 (header) + 256 (rank) + 8192 (bitmap) = 8452
        assert_eq!(&bytes[8452..8454], &32767i16.to_le_bytes());
    }

    #[test]
    fn test_jump_table_structure() {
        // Multiple blocks to verify jump table entries
        // Block 0: 3 docs, Block 1: 2 docs
        let mut doc_ids = vec![10, 20, 30];
        doc_ids.extend([65536 + 5, 65536 + 10]); // block 1
        let (bytes, entry_count) = write_and_get_bytes(&doc_ids, 200000);

        // blockCount = 2+1 = 3 (blocks 0,1 + sentinel)
        assert_eq!(entry_count, 3);

        // Find the jump table at the end
        // Block 0: 4 (header) + 6 (3 shorts) = 10 bytes
        // Block 1: 4 (header) + 4 (2 shorts) = 8 bytes
        // Sentinel: 4 (header) + 2 (1 short) = 6 bytes
        // Total blocks = 24 bytes
        // Jump table: 3 entries × 8 bytes = 24 bytes
        // Total = 48 bytes
        assert_len_eq_x!(&bytes, 48);

        // Jump entry 0: index=0, offset=0
        assert_eq!(i32::from_le_bytes(bytes[24..28].try_into().unwrap()), 0);
        assert_eq!(i32::from_le_bytes(bytes[28..32].try_into().unwrap()), 0);

        // Jump entry 1: index=3 (3 docs in block 0), offset=10
        assert_eq!(i32::from_le_bytes(bytes[32..36].try_into().unwrap()), 3);
        assert_eq!(i32::from_le_bytes(bytes[36..40].try_into().unwrap()), 10);

        // Jump entry 2 (sentinel): index=5 (3+2 docs), offset=18
        assert_eq!(i32::from_le_bytes(bytes[40..44].try_into().unwrap()), 5);
        assert_eq!(i32::from_le_bytes(bytes[44..48].try_into().unwrap()), 18);
    }

    #[test]
    fn test_single_block_jump_table_optimization() {
        // Single real block + sentinel = blockCount 2 → optimized to 0 entries
        let doc_ids = vec![0, 1, 2];
        let (_, entry_count) = write_and_get_bytes(&doc_ids, 100);
        assert_eq!(entry_count, 0);
    }

    #[test]
    fn test_docs_at_block_boundary() {
        // Doc at last position of block 0 and first position of block 1
        let doc_ids = vec![65535, 65536];
        let (bytes, _) = write_and_get_bytes(&doc_ids, 200000);

        // Block 0: blockID=0, cardinality=1
        assert_eq!(&bytes[0..2], &0i16.to_le_bytes());
        assert_eq!(&bytes[2..4], &0i16.to_le_bytes()); // cardinality-1=0

        // SPARSE: doc 65535 within block
        assert_eq!(&bytes[4..6], &(-1i16).to_le_bytes()); // 65535 as i16

        // Block 1: blockID=1, cardinality=1
        assert_eq!(&bytes[6..8], &1i16.to_le_bytes());
        assert_eq!(&bytes[8..10], &0i16.to_le_bytes()); // cardinality-1=0

        // SPARSE: doc 0 within block
        assert_eq!(&bytes[10..12], &0i16.to_le_bytes());
    }

    #[test]
    fn test_dense_bitmap_correctness() {
        // 4096 contiguous docs — DENSE encoding
        let all_docs: Vec<i32> = (0..4096).collect();
        let (bytes, _) = write_and_get_bytes(&all_docs, 65536);

        // DENSE block: header(4) + rank(256) + bitmap(8192)
        let bitmap_start = 4 + 256;

        // First long: bits 0-63 all set
        let word0 = i64::from_le_bytes(bytes[bitmap_start..bitmap_start + 8].try_into().unwrap());
        assert_eq!(word0, -1i64); // all 64 bits set

        // Long at index 64 (word 64 = docs 4096-4159): should be all zeros
        let word64_offset = bitmap_start + 64 * 8;
        let word64 =
            i64::from_le_bytes(bytes[word64_offset..word64_offset + 8].try_into().unwrap());
        assert_eq!(word64, 0);
    }

    // -----------------------------------------------------------------------
    // Reader round-trip tests
    // Ported from org.apache.lucene.codecs.lucene90.TestIndexedDISI
    // -----------------------------------------------------------------------

    /// Opens an `IndexedDISI` over `bytes`. Caller must keep `bytes` alive.
    fn disi_over<'a>(bytes: &'a [u8], entry_count: i16, cost: i64) -> IndexedDISI<'a> {
        let input = IndexInput::unnamed(bytes);
        IndexedDISI::new(
            &input,
            0,
            bytes.len(),
            entry_count,
            DEFAULT_DENSE_RANK_POWER as u8,
            cost,
        )
        .unwrap()
    }

    /// Verify next_doc() returns exactly the expected doc IDs in order.
    fn assert_next_doc_sequence(disi: &mut IndexedDISI<'_>, expected: &[i32]) {
        for (ordinal, &expected_doc) in expected.iter().enumerate() {
            let doc = disi.next_doc().unwrap();
            assert_eq!(doc, Some(expected_doc), "ordinal {ordinal}");
            assert_eq!(disi.index(), ordinal as i32, "index at ordinal {ordinal}");
        }
        assert_none!(disi.next_doc().unwrap());
    }

    /// Verify advance() finds each expected doc, with index tracking.
    fn assert_advance_sequence(disi: &mut IndexedDISI<'_>, expected: &[i32]) {
        for (ordinal, &expected_doc) in expected.iter().enumerate() {
            let doc = disi.advance(expected_doc).unwrap();
            assert_eq!(doc, Some(expected_doc), "advance to {expected_doc}");
            assert_eq!(disi.index(), ordinal as i32, "index at {expected_doc}");
        }
    }

    /// Verify advance_exact() returns true for set docs and false for unset docs.
    fn assert_advance_exact(doc_ids: &[i32], max_doc: i32) {
        let (bytes, entry_count) = write_and_get_bytes(doc_ids, max_doc);
        let mut disi = disi_over(&bytes, entry_count, doc_ids.len() as i64);
        let doc_set: HashSet<i32> = doc_ids.iter().copied().collect();

        // Forward-only check: advance_exact can only go forward
        let check_limit = max_doc.min(200_000);
        let mut ordinal = 0;
        for target in 0..check_limit {
            let result = disi.advance_exact(target).unwrap();
            if doc_set.contains(&target) {
                assert!(result, "advance_exact({target}) should be true");
                assert_eq!(disi.index(), ordinal, "index at doc {target}");
                ordinal += 1;
            } else {
                assert!(!result, "advance_exact({target}) should be false");
            }
        }
    }

    #[test]
    fn test_read_empty() {
        // Ported from TestIndexedDISI.testEmpty
        let docs: [i32; 0] = [];
        let (bytes, entry_count) = write_and_get_bytes(&docs, 10);
        let mut disi = disi_over(&bytes, entry_count, docs.len() as i64);
        assert_none!(disi.next_doc().unwrap());
    }

    #[test]
    fn test_read_one_doc() {
        // Ported from TestIndexedDISI.testOneDoc
        let docs = [42];
        let (bytes, entry_count) = write_and_get_bytes(&docs, 100);
        let mut disi = disi_over(&bytes, entry_count, docs.len() as i64);
        assert_next_doc_sequence(&mut disi, &docs);
    }

    #[test]
    fn test_read_two_docs() {
        // Ported from TestIndexedDISI.testTwoDocs
        let docs = [42, 12345];
        let (bytes, entry_count) = write_and_get_bytes(&docs, 20000);
        let mut disi = disi_over(&bytes, entry_count, docs.len() as i64);
        assert_next_doc_sequence(&mut disi, &docs);
    }

    #[test]
    fn test_read_all_docs() {
        // Ported from TestIndexedDISI.testAllDocs
        let docs: Vec<i32> = (0..65536).collect();
        let (bytes, entry_count) = write_and_get_bytes(&docs, 65536);
        let mut disi = disi_over(&bytes, entry_count, docs.len() as i64);

        // Verify first few and last
        assert_eq!(disi.next_doc().unwrap(), Some(0));
        assert_eq!(disi.index(), 0);
        assert_eq!(disi.advance(65535).unwrap(), Some(65535));
        assert_eq!(disi.index(), 65535);
        assert_none!(disi.next_doc().unwrap());
    }

    #[test]
    fn test_read_sparse_next_doc() {
        // SPARSE block: few docs, iterate with next_doc
        let docs = vec![10, 100, 1000, 5000, 60000];
        let (bytes, entry_count) = write_and_get_bytes(&docs, 65536);
        let mut disi = disi_over(&bytes, entry_count, docs.len() as i64);
        assert_next_doc_sequence(&mut disi, &docs);
    }

    #[test]
    fn test_read_sparse_advance() {
        // SPARSE block: advance to specific targets
        let docs = vec![10, 100, 1000, 5000, 60000];
        let (bytes, entry_count) = write_and_get_bytes(&docs, 65536);
        let mut disi = disi_over(&bytes, entry_count, docs.len() as i64);
        assert_advance_sequence(&mut disi, &docs);
    }

    #[test]
    fn test_read_sparse_advance_exact() {
        // SPARSE block: advance_exact for presence/absence
        let docs = vec![10, 100, 1000, 5000];
        assert_advance_exact(&docs, 6000);
    }

    #[test]
    fn test_read_dense_next_doc() {
        // DENSE block: 5000 contiguous docs
        let docs: Vec<i32> = (0..5000).collect();
        let (bytes, entry_count) = write_and_get_bytes(&docs, 65536);
        let mut disi = disi_over(&bytes, entry_count, docs.len() as i64);

        // Spot check first, middle, and end
        assert_eq!(disi.next_doc().unwrap(), Some(0));
        assert_eq!(disi.index(), 0);
        assert_eq!(disi.advance(2500).unwrap(), Some(2500));
        assert_eq!(disi.index(), 2500);
        assert_eq!(disi.advance(4999).unwrap(), Some(4999));
        assert_eq!(disi.index(), 4999);
        assert_none!(disi.next_doc().unwrap());
    }

    #[test]
    fn test_read_dense_advance_exact() {
        // DENSE block: verify advance_exact with gaps
        // Every 3rd doc set (like Java's DENSE test pattern)
        let docs: Vec<i32> = (0..65536).filter(|d| d % 3 == 0).collect();
        assert_gt!(docs.len(), MAX_ARRAY_LENGTH as usize); // confirm DENSE
        assert_advance_exact(&docs, 65536);
    }

    #[test]
    fn test_read_all_block_next_doc() {
        // ALL block: every doc in range
        let docs: Vec<i32> = (0..65536).collect();
        let (bytes, entry_count) = write_and_get_bytes(&docs, 65536);
        let mut disi = disi_over(&bytes, entry_count, docs.len() as i64);

        assert_eq!(disi.next_doc().unwrap(), Some(0));
        assert_eq!(disi.index(), 0);
        assert_eq!(disi.advance(32768).unwrap(), Some(32768));
        assert_eq!(disi.index(), 32768);
        assert_eq!(disi.advance(65535).unwrap(), Some(65535));
        assert_eq!(disi.index(), 65535);
        assert_none!(disi.next_doc().unwrap());
    }

    #[test]
    fn test_read_all_block_advance_exact() {
        // ALL block: every doc should exist
        let docs: Vec<i32> = (0..65536).collect();
        let (bytes, entry_count) = write_and_get_bytes(&docs, 65536);
        let mut disi = disi_over(&bytes, entry_count, docs.len() as i64);
        for target in [0, 100, 32768, 65535] {
            assert!(disi.advance_exact(target).unwrap(), "doc {target}");
            assert_eq!(disi.index(), target);
        }
    }

    #[test]
    fn test_read_multi_block_sparse() {
        // Multiple SPARSE blocks across block boundaries
        let mut docs = vec![10, 100, 1000]; // block 0
        docs.extend([65536 + 5, 65536 + 200]); // block 1
        docs.extend([131072 + 42]); // block 2
        let (bytes, entry_count) = write_and_get_bytes(&docs, 200000);
        let mut disi = disi_over(&bytes, entry_count, docs.len() as i64);
        assert_next_doc_sequence(&mut disi, &docs);
    }

    #[test]
    fn test_read_multi_block_advance() {
        // Advance across block boundaries
        let mut docs = vec![10, 100]; // block 0
        docs.extend([65536 + 5, 65536 + 200]); // block 1
        docs.extend([131072 + 42]); // block 2
        let (bytes, entry_count) = write_and_get_bytes(&docs, 200000);
        let mut disi = disi_over(&bytes, entry_count, docs.len() as i64);

        // Skip block 0 entirely
        assert_eq!(disi.advance(65536).unwrap(), Some(65536 + 5));
        assert_eq!(disi.index(), 2);

        // Skip to block 2
        assert_eq!(disi.advance(131072).unwrap(), Some(131072 + 42));
        assert_eq!(disi.index(), 4);
    }

    #[test]
    fn test_read_multi_block_advance_exact() {
        // advance_exact across mixed blocks
        let mut docs = vec![10, 100]; // block 0 sparse
        docs.extend([65536 + 5]); // block 1 sparse
        let (bytes, entry_count) = write_and_get_bytes(&docs, 200000);
        let mut disi = disi_over(&bytes, entry_count, docs.len() as i64);

        assert!(disi.advance_exact(10).unwrap());
        assert_eq!(disi.index(), 0);
        assert!(!disi.advance_exact(50).unwrap());
        assert!(disi.advance_exact(100).unwrap());
        assert_eq!(disi.index(), 1);
        assert!(disi.advance_exact(65536 + 5).unwrap());
        assert_eq!(disi.index(), 2);
    }

    #[test]
    fn test_read_mixed_block_types() {
        // Ported from TestIndexedDISI.testRandomBlocks pattern
        // Block 0: SPARSE (few docs)
        // Block 1: ALL (65536 docs)
        // Block 2: DENSE (every 3rd doc)
        let mut docs: Vec<i32> = vec![0, 100, 4000]; // block 0: 3 docs = SPARSE
        docs.extend(65536..131072); // block 1: all 65536 = ALL
        let dense_docs: Vec<i32> = (131072..196608).filter(|d| d % 3 == 0).collect();
        docs.extend(&dense_docs); // block 2: ~21845 docs = DENSE

        let (bytes, entry_count) = write_and_get_bytes(&docs, 200000);
        let mut disi = disi_over(&bytes, entry_count, docs.len() as i64);

        // SPARSE block
        assert_eq!(disi.next_doc().unwrap(), Some(0));
        assert_eq!(disi.index(), 0);
        assert_eq!(disi.advance(100).unwrap(), Some(100));
        assert_eq!(disi.index(), 1);

        // ALL block
        assert_eq!(disi.advance(65536).unwrap(), Some(65536));
        assert_eq!(disi.index(), 3);
        assert_eq!(disi.advance(100000).unwrap(), Some(100000));
        assert_eq!(disi.index(), 3 + (100000 - 65536));

        // DENSE block — 131072 % 3 == 2, so first set doc is 131073
        assert_eq!(disi.advance(131072).unwrap(), Some(131073));
        // Index = 3 (sparse) + 65536 (all) + 0 (first in dense)
        assert_eq!(disi.index(), 3 + 65536);

        assert_none!(disi.advance(200000).unwrap(), "past all blocks");
    }

    #[test]
    fn test_read_advance_past_end() {
        let docs = vec![10, 20, 30];
        let (bytes, entry_count) = write_and_get_bytes(&docs, 100);
        let mut disi = disi_over(&bytes, entry_count, docs.len() as i64);
        assert_none!(disi.advance(50).unwrap());
    }

    #[test]
    fn test_read_half_full() {
        // Ported from TestIndexedDISI.testHalfFull
        // Every other doc — results in DENSE encoding
        let docs: Vec<i32> = (0..65536).filter(|d| d % 2 == 0).collect();
        assert_eq!(docs.len(), 32768);
        let (bytes, entry_count) = write_and_get_bytes(&docs, 65536);
        let mut disi = disi_over(&bytes, entry_count, docs.len() as i64);

        // Iterate through all and verify indices
        let mut count = 0;
        while let Some(doc) = disi.next_doc().unwrap() {
            assert_eq!(doc, count * 2);
            assert_eq!(disi.index(), count);
            count += 1;
        }
        assert_eq!(count, 32768);
    }

    #[test]
    fn test_read_one_doc_missing() {
        // Ported from TestIndexedDISI.testOneDocMissing
        // All docs except doc 500 — DENSE block
        let docs: Vec<i32> = (0..65536).filter(|&d| d != 500).collect();
        assert_eq!(docs.len(), 65535);
        let (bytes, entry_count) = write_and_get_bytes(&docs, 65536);
        let mut disi = disi_over(&bytes, entry_count, docs.len() as i64);

        // Doc 499 should exist at index 499
        assert!(disi.advance_exact(499).unwrap());
        assert_eq!(disi.index(), 499);

        // Doc 500 should not exist
        assert!(!disi.advance_exact(500).unwrap());

        // Doc 501 should exist at index 500 (shifted down by 1)
        assert!(disi.advance_exact(501).unwrap());
        assert_eq!(disi.index(), 500);
    }

    #[test]
    fn test_read_sparse_dense_boundary() {
        // Ported from TestIndexedDISI.testSparseDenseBoundary
        // Exactly 4095 docs (MAX_ARRAY_LENGTH) → SPARSE
        let docs: Vec<i32> = (0..4095).collect();
        let (bytes, entry_count) = write_and_get_bytes(&docs, 65536);
        let mut disi = disi_over(&bytes, entry_count, docs.len() as i64);
        assert_next_doc_sequence(&mut disi, &docs);

        // Exactly 4096 docs → DENSE
        let docs: Vec<i32> = (0..4096).collect();
        let (bytes, entry_count) = write_and_get_bytes(&docs, 65536);
        let mut disi = disi_over(&bytes, entry_count, docs.len() as i64);
        assert_next_doc_sequence(&mut disi, &docs);
    }

    #[test]
    fn test_read_block_boundary_docs() {
        // Docs at block boundaries
        let docs = vec![65535, 65536];
        let (bytes, entry_count) = write_and_get_bytes(&docs, 200000);
        let mut disi = disi_over(&bytes, entry_count, docs.len() as i64);
        assert_next_doc_sequence(&mut disi, &docs);
    }

    #[test]
    fn test_read_dense_rank_skip() {
        // Dense block with advance that should trigger rank-based skipping
        // Every other doc = 32768 docs, DENSE
        let docs: Vec<i32> = (0..65536).filter(|d| d % 2 == 0).collect();
        let (bytes, entry_count) = write_and_get_bytes(&docs, 65536);
        let mut disi = disi_over(&bytes, entry_count, docs.len() as i64);

        // Advance far enough to trigger rank skip (>= 2^(9-6) = 8 words = 512 docs)
        assert_eq!(disi.advance(0).unwrap(), Some(0));
        assert_eq!(disi.index(), 0);
        // Jump well past the rank threshold
        assert_eq!(disi.advance(2000).unwrap(), Some(2000));
        assert_eq!(disi.index(), 1000);
        assert_eq!(disi.advance(60000).unwrap(), Some(60000));
        assert_eq!(disi.index(), 30000);
    }

    #[test]
    fn test_illegal_dense_rank_power() {
        // Ported from TestIndexedDISI.testIllegalDenseRankPower
        let (bytes, _) = write_and_get_bytes(&[0], 10);
        let input = IndexInput::unnamed(&bytes);

        for bad_power in [0u8, 1, 6, 16] {
            let result = IndexedDISI::new(&input, 0, bytes.len(), 0, bad_power, 1);
            assert!(result.is_err(), "power {bad_power} should be rejected");
        }

        // Valid powers should succeed
        for good_power in [7u8, 9, 15] {
            let result = IndexedDISI::new(&input, 0, bytes.len(), 0, good_power, 1);
            assert!(result.is_ok(), "power {good_power} should be accepted");
        }
    }

    #[test]
    fn test_read_position_not_zero() {
        // Ported from TestIndexedDISI.testPositionNotZero
        // Write some prefix bytes, then the DISI data at a non-zero offset
        let doc_ids = vec![10, 50, 1000];
        let mut out = MemoryIndexOutput::new("test".to_string());
        // Write 37 garbage bytes first
        for i in 0..37u8 {
            out.write_byte(i).unwrap();
        }
        let offset = out.file_pointer() as usize;
        let entry_count = write_bit_set(&doc_ids, 2000, &mut out).unwrap();
        let length = out.file_pointer() as usize - offset;

        let bytes = out.bytes().to_vec();
        let input = IndexInput::unnamed(&bytes);
        let mut disi = IndexedDISI::new(
            &input,
            offset,
            length,
            entry_count,
            DEFAULT_DENSE_RANK_POWER as u8,
            doc_ids.len() as i64,
        )
        .unwrap();
        assert_next_doc_sequence(&mut disi, &doc_ids);
    }
}
