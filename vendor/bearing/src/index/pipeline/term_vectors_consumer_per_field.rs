// SPDX-License-Identifier: Apache-2.0

//! Per-field term vector processing.
//!
//! `TermVectorsConsumerPerField` is the secondary per-field writer in the
//! TermsHash chain. It is attached to `FreqProxTermsWriterPerField` via the
//! `nextPerField` link and processes the same terms for term vector storage.
//!
//! Stream 0: positions + payloads
//! Stream 1: offsets

use std::fmt;
use std::io;
use std::mem;

use crate::codecs::lucene90::term_vectors::{CompressingTermVectorsWriter, TermVectorsWriter};
use crate::document::{IndexOptions, TermOffset};
use crate::index::pipeline::terms_hash::{
    BYTES_PER_POSTING, ParallelPostingsArray, TermsHash, TermsHashPerField, TermsHashPerFieldTrait,
    oversize,
};
use crate::util::byte_block_pool::{ByteBlockPool, ByteSliceReader, FIRST_LEVEL_SIZE};
use crate::util::bytes_ref_hash::BytesRefHash;

/// Per-field term vector writer.
///
/// Implements `TermsHashPerFieldTrait`, using `add_by_pool_offset` for term
/// deduplication — the term bytes are already interned in the primary
/// (FreqProx) `BytesRefHash`.
///
/// Pools are owned externally in a `TermsHash` stored on
/// `TermVectorsConsumer`.
pub(crate) struct TermVectorsConsumerPerField {
    /// Base term hash functionality (BytesRefHash, stream cursors).
    pub(crate) base: TermsHashPerField,

    /// Per-term TV metadata.
    pub(crate) postings_array: TermVectorsPostingsArray,

    /// Whether this field is storing term vectors for the current document.
    pub(crate) do_vectors: bool,
    /// Whether to store positions in term vectors.
    pub(crate) do_vector_positions: bool,
    /// Whether to store offsets in term vectors.
    pub(crate) do_vector_offsets: bool,
    /// Whether to store payloads in term vectors.
    pub(crate) do_vector_payloads: bool,
    /// Whether we actually saw any payloads for this field.
    pub(crate) has_payloads: bool,
    // Current token state — set by FreqProx's add() before calling add_by_text_start.
    pub(crate) current_position: i32,
    pub(crate) current_offset: TermOffset,
}

impl TermVectorsConsumerPerField {
    /// Creates a new `TermVectorsConsumerPerField`.
    ///
    /// Always uses 2 streams (positions + offsets), matching Java.
    pub(crate) fn new(field_name: String) -> Self {
        let stream_count = 2;
        let base = TermsHashPerField::new(stream_count, field_name, IndexOptions::DocsAndFreqs);

        Self {
            base,
            postings_array: TermVectorsPostingsArray::new(2),
            do_vectors: false,
            do_vector_positions: false,
            do_vector_offsets: false,
            do_vector_payloads: false,
            has_payloads: false,
            current_position: 0,
            current_offset: TermOffset::default(),
        }
    }

    /// Called after all tokens for this field in the current document.
    #[expect(dead_code)]
    pub(crate) fn finish(&self) {
        // No-op; finishDocument is driven externally.
    }

    /// Returns true if this field has term vector data to flush.
    pub(crate) fn has_data(&self) -> bool {
        self.do_vectors && self.num_terms() > 0
    }

    /// Writes this field's term vector data to the codec writer, reading
    /// term text from the shared term byte pool and position/offset data from
    /// the TV pools via `ByteSliceReader`.
    ///
    /// `term_byte_pool` is the shared pool from the accumulator.
    /// `tv_terms_hash` provides the TV int/byte pools for position/offset streams.
    pub(crate) fn finish_document(
        &mut self,
        field_number: u32,
        term_byte_pool: &ByteBlockPool,
        tv_terms_hash: &TermsHash,
        writer: &mut CompressingTermVectorsWriter,
    ) -> io::Result<()> {
        if !self.do_vectors {
            return Ok(());
        }
        self.do_vectors = false;

        let num_terms = self.num_terms();

        self.base.sort_terms(term_byte_pool);
        let sorted_ids = self.base.sorted_term_ids();

        writer.start_field(
            field_number,
            num_terms as i32,
            self.do_vector_positions,
            self.do_vector_offsets,
            self.has_payloads,
        );

        for &sorted_id in &sorted_ids[..num_terms] {
            let term_id = sorted_id as usize;
            let freq = self.postings_array.freqs[term_id];

            let text_start = self.postings_array.base.text_starts[term_id] as usize;
            let term_bytes = BytesRefHash::read_bytes_at_pool(term_byte_pool, text_start);

            writer.start_term(term_bytes, freq);

            if self.do_vector_positions || self.do_vector_offsets {
                let mut pos_reader = if self.do_vector_positions {
                    let (start, end) = self.get_stream_range(&tv_terms_hash.int_pool, term_id, 0);
                    Some(ByteSliceReader::new(&tv_terms_hash.byte_pool, start, end))
                } else {
                    None
                };
                let mut off_reader = if self.do_vector_offsets {
                    let (start, end) = self.get_stream_range(&tv_terms_hash.int_pool, term_id, 1);
                    Some(ByteSliceReader::new(&tv_terms_hash.byte_pool, start, end))
                } else {
                    None
                };
                writer.add_prox(freq, pos_reader.as_mut(), off_reader.as_mut());
            }
            writer.finish_term();
        }
        writer.finish_field();

        self.reset();
        Ok(())
    }

    /// Writes this field's term vector data to the codec writer when the
    /// TV per-field owns its own term bytes (used when terms were added
    /// via `add()` rather than `add_by_text_start()`).
    ///
    /// `tv_terms_hash` provides both the byte pool (for term text AND
    /// position/offset streams) and the int pool (for stream addresses).
    #[cfg(test)]
    pub(crate) fn finish_document_self_owned(
        &mut self,
        field_number: u32,
        tv_terms_hash: &TermsHash,
        writer: &mut CompressingTermVectorsWriter,
    ) -> io::Result<()> {
        if !self.do_vectors {
            return Ok(());
        }
        self.do_vectors = false;

        let num_terms = self.num_terms();

        self.base.sort_terms(&tv_terms_hash.byte_pool);
        let sorted_ids = self.base.sorted_term_ids();

        writer.start_field(
            field_number,
            num_terms as i32,
            self.do_vector_positions,
            self.do_vector_offsets,
            self.has_payloads,
        );

        for &sorted_id in &sorted_ids[..num_terms] {
            let term_id = sorted_id as usize;
            let freq = self.postings_array.freqs[term_id];

            let term_bytes = self.base.bytes_hash.get(&tv_terms_hash.byte_pool, term_id);

            writer.start_term(term_bytes, freq);

            if self.do_vector_positions || self.do_vector_offsets {
                let mut pos_reader = if self.do_vector_positions {
                    let (start, end) = self.get_stream_range(&tv_terms_hash.int_pool, term_id, 0);
                    Some(ByteSliceReader::new(&tv_terms_hash.byte_pool, start, end))
                } else {
                    None
                };
                let mut off_reader = if self.do_vector_offsets {
                    let (start, end) = self.get_stream_range(&tv_terms_hash.int_pool, term_id, 1);
                    Some(ByteSliceReader::new(&tv_terms_hash.byte_pool, start, end))
                } else {
                    None
                };
                writer.add_prox(freq, pos_reader.as_mut(), off_reader.as_mut());
            }
            writer.finish_term();
        }
        writer.finish_field();

        self.reset();
        Ok(())
    }

    /// Returns the number of unique terms in the TV hash.
    pub(crate) fn num_terms(&self) -> usize {
        self.base.bytes_hash.size()
    }

    /// Clears the term hash and resets state for reuse between documents.
    pub(crate) fn reset(&mut self) {
        self.base.reset();
    }

    /// Returns the stream range `(start, end)` for the given term and stream.
    pub(crate) fn get_stream_range(
        &self,
        int_pool: &[i32],
        term_id: usize,
        stream: usize,
    ) -> (usize, usize) {
        assert!(stream < self.base.stream_count);
        let address_offset = self.postings_array.base.address_offset[term_id] as usize;
        let end = int_pool[address_offset + stream] as usize;
        let start =
            self.postings_array.base.byte_starts[term_id] as usize + stream * FIRST_LEVEL_SIZE;
        (start, end)
    }

    /// Write position and/or offset data for a term occurrence.
    fn write_prox(&mut self, terms_hash: &mut TermsHash, term_id: usize) {
        if self.do_vector_offsets {
            let last_offset = self.postings_array.last_offsets[term_id];
            let start_offset = self.current_offset.start as i32;
            let offset_length = self.current_offset.length;
            self.base
                .write_v_int(terms_hash, 1, start_offset - last_offset);
            self.base.write_v_int(terms_hash, 1, offset_length as i32);
            self.postings_array.last_offsets[term_id] = start_offset + offset_length as i32;
        }

        if self.do_vector_positions {
            let pos = self.current_position - self.postings_array.last_positions[term_id];
            self.base.write_v_int(terms_hash, 0, pos << 1);
            self.postings_array.last_positions[term_id] = self.current_position;
        }
    }
}

impl fmt::Debug for TermVectorsConsumerPerField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TermVectorsConsumerPerField")
            .field("field_name", &self.base.field_name())
            .field("num_terms", &self.base.bytes_hash.size())
            .field("do_vectors", &self.do_vectors)
            .field("do_vector_positions", &self.do_vector_positions)
            .field("do_vector_offsets", &self.do_vector_offsets)
            .finish()
    }
}

impl TermsHashPerFieldTrait for TermVectorsConsumerPerField {
    fn base(&self) -> &TermsHashPerField {
        &self.base
    }

    fn base_mut(&mut self) -> &mut TermsHashPerField {
        &mut self.base
    }

    fn postings_array_base(&self) -> &ParallelPostingsArray {
        &self.postings_array.base
    }

    fn postings_array_base_mut(&mut self) -> &mut ParallelPostingsArray {
        &mut self.postings_array.base
    }

    fn ensure_postings_capacity(&mut self, term_id: usize) {
        while term_id >= self.postings_array.size() {
            let grown = self.postings_array.grow();
            self.postings_array = grown;
        }
    }

    fn new_term(&mut self, terms_hash: &mut TermsHash, term_id: usize, _doc_id: i32) {
        let postings = &mut self.postings_array;

        postings.freqs[term_id] = 1;
        postings.last_offsets[term_id] = 0;
        postings.last_positions[term_id] = 0;

        self.write_prox(terms_hash, term_id);
    }

    fn add_term(&mut self, terms_hash: &mut TermsHash, term_id: usize, _doc_id: i32) {
        self.postings_array.freqs[term_id] += 1;

        self.write_prox(terms_hash, term_id);
    }
}

// ---------------------------------------------------------------------------
// TermVectorsPostingsArray
// ---------------------------------------------------------------------------

/// Per-term posting metadata for term vectors.
///
/// Simpler than `FreqProxPostingsArray` — no document tracking since
/// term vectors are per-document. Tracks frequency, last position, and
/// last offset within a single document.
#[derive(Debug)]
pub(crate) struct TermVectorsPostingsArray {
    /// Base arrays (text starts, address offsets, byte starts).
    pub base: ParallelPostingsArray,
    /// How many times each term occurred in the current document.
    pub freqs: Vec<i32>,
    /// Last offset we saw for each term.
    pub last_offsets: Vec<i32>,
    /// Last position where each term occurred.
    pub last_positions: Vec<i32>,
}

impl TermVectorsPostingsArray {
    /// Creates a new array with the given initial capacity.
    pub(crate) fn new(size: usize) -> Self {
        Self {
            base: ParallelPostingsArray::new(size),
            freqs: vec![0; size],
            last_offsets: vec![0; size],
            last_positions: vec![0; size],
        }
    }

    /// Returns the current capacity.
    pub(crate) fn size(&self) -> usize {
        self.base.size()
    }

    /// Returns bytes per posting (base + 3 int fields).
    pub(crate) fn bytes_per_posting(&self) -> usize {
        BYTES_PER_POSTING + 3 * mem::size_of::<i32>()
    }

    /// Grows the arrays to accommodate at least one more entry.
    pub(crate) fn grow(&self) -> Self {
        let new_size = oversize(self.size() + 1, self.bytes_per_posting());
        let mut new_array = Self::new(new_size);
        self.copy_to(&mut new_array, self.size());
        new_array
    }

    /// Copies `num_to_copy` elements from `self` into `to_array`.
    pub(crate) fn copy_to(&self, to_array: &mut TermVectorsPostingsArray, num_to_copy: usize) {
        self.base.copy_to(&mut to_array.base, num_to_copy);
        to_array.freqs[..num_to_copy].copy_from_slice(&self.freqs[..num_to_copy]);
        to_array.last_offsets[..num_to_copy].copy_from_slice(&self.last_offsets[..num_to_copy]);
        to_array.last_positions[..num_to_copy].copy_from_slice(&self.last_positions[..num_to_copy]);
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::codecs::lucene90::term_vectors::{CompressingTermVectorsWriter, TermVectorsWriter};
    use crate::store::MemoryDirectory;
    use crate::util::byte_block_pool::ByteBlockPool;
    use assertables::*;

    fn new_term_pool() -> ByteBlockPool {
        ByteBlockPool::new(32 * 1024)
    }

    /// Helper to read a VInt from a byte slice reader.
    fn read_vint(reader: &mut ByteSliceReader<'_>) -> i32 {
        reader.read_vint().unwrap()
    }

    #[test]
    fn test_tv_new_term_positions() {
        let mut tv_th = TermsHash::new();
        let mut tv = TermVectorsConsumerPerField::new("body".to_string());
        tv.do_vectors = true;
        tv.do_vector_positions = true;

        tv.current_position = 0;
        tv.current_offset = TermOffset {
            start: 0,
            length: 5,
        };
        TermsHashPerFieldTrait::add_by_text_start(&mut tv, &mut tv_th, 100, 0);

        assert_eq!(tv.num_terms(), 1);
        assert_eq!(tv.postings_array.freqs[0], 1);
        assert_eq!(tv.postings_array.last_positions[0], 0);

        // Read position stream (stream 0)
        let (start, end) = tv.get_stream_range(&tv_th.int_pool, 0, 0);
        let mut reader = ByteSliceReader::new(&tv_th.byte_pool, start, end);
        let pos_code = read_vint(&mut reader);
        assert_eq!(pos_code, 0);
    }

    #[test]
    fn test_tv_add_term_positions() {
        let mut tv_th = TermsHash::new();
        let mut tv = TermVectorsConsumerPerField::new("body".to_string());
        tv.do_vectors = true;
        tv.do_vector_positions = true;

        // Same term at positions 0 and 3
        tv.current_position = 0;
        tv.current_offset = TermOffset {
            start: 0,
            length: 5,
        };
        TermsHashPerFieldTrait::add_by_text_start(&mut tv, &mut tv_th, 100, 0);

        tv.current_position = 3;
        tv.current_offset = TermOffset {
            start: 18,
            length: 5,
        };
        TermsHashPerFieldTrait::add_by_text_start(&mut tv, &mut tv_th, 100, 0);

        assert_eq!(tv.num_terms(), 1);
        assert_eq!(tv.postings_array.freqs[0], 2);

        let (start, end) = tv.get_stream_range(&tv_th.int_pool, 0, 0);
        let mut reader = ByteSliceReader::new(&tv_th.byte_pool, start, end);

        let pos0 = read_vint(&mut reader);
        assert_eq!(pos0, 0);
        let pos1 = read_vint(&mut reader);
        assert_eq!(pos1, 6); // delta 3 << 1
    }

    #[test]
    fn test_tv_offsets() {
        let mut tv_th = TermsHash::new();
        let mut tv = TermVectorsConsumerPerField::new("body".to_string());
        tv.do_vectors = true;
        tv.do_vector_offsets = true;

        tv.current_position = 0;
        tv.current_offset = TermOffset {
            start: 0,
            length: 5,
        };
        TermsHashPerFieldTrait::add_by_text_start(&mut tv, &mut tv_th, 100, 0);

        tv.current_position = 1;
        tv.current_offset = TermOffset {
            start: 10,
            length: 5,
        };
        TermsHashPerFieldTrait::add_by_text_start(&mut tv, &mut tv_th, 100, 0);

        assert_eq!(tv.num_terms(), 1);
        assert_eq!(tv.postings_array.freqs[0], 2);

        // Read offset stream (stream 1)
        let (start, end) = tv.get_stream_range(&tv_th.int_pool, 0, 1);
        let mut reader = ByteSliceReader::new(&tv_th.byte_pool, start, end);

        let start_delta_0 = read_vint(&mut reader);
        assert_eq!(start_delta_0, 0);
        let length_0 = read_vint(&mut reader);
        assert_eq!(length_0, 5);

        let start_delta_1 = read_vint(&mut reader);
        assert_eq!(start_delta_1, 5); // 10 - 5
        let length_1 = read_vint(&mut reader);
        assert_eq!(length_1, 5);
    }

    #[test]
    fn test_tv_multiple_terms() {
        let mut tv_th = TermsHash::new();
        let mut tv = TermVectorsConsumerPerField::new("body".to_string());
        tv.do_vectors = true;
        tv.do_vector_positions = true;

        tv.current_position = 0;
        tv.current_offset = TermOffset {
            start: 0,
            length: 5,
        };
        TermsHashPerFieldTrait::add_by_text_start(&mut tv, &mut tv_th, 100, 0);

        tv.current_position = 1;
        tv.current_offset = TermOffset {
            start: 6,
            length: 5,
        };
        TermsHashPerFieldTrait::add_by_text_start(&mut tv, &mut tv_th, 200, 0);

        tv.current_position = 2;
        tv.current_offset = TermOffset {
            start: 12,
            length: 5,
        };
        TermsHashPerFieldTrait::add_by_text_start(&mut tv, &mut tv_th, 100, 0);

        assert_eq!(tv.num_terms(), 2);
        assert_eq!(tv.postings_array.freqs[0], 2);
        assert_eq!(tv.postings_array.freqs[1], 1);
    }

    #[test]
    fn test_tv_reset_clears_between_docs() {
        let mut tv_th = TermsHash::new();
        let mut tv = TermVectorsConsumerPerField::new("body".to_string());
        tv.do_vectors = true;
        tv.do_vector_positions = true;

        tv.current_position = 0;
        tv.current_offset = TermOffset {
            start: 0,
            length: 5,
        };
        TermsHashPerFieldTrait::add_by_text_start(&mut tv, &mut tv_th, 100, 0);
        assert_eq!(tv.num_terms(), 1);

        tv.reset();
        assert_eq!(tv.num_terms(), 0);

        tv.current_position = 0;
        tv.current_offset = TermOffset {
            start: 0,
            length: 5,
        };
        TermsHashPerFieldTrait::add_by_text_start(&mut tv, &mut tv_th, 200, 1);
        assert_eq!(tv.num_terms(), 1);
    }

    #[test]
    fn test_tv_postings_array_grow() {
        let arr = TermVectorsPostingsArray::new(2);
        assert_eq!(arr.size(), 2);
        let grown = arr.grow();
        assert_gt!(grown.size(), 2);
    }

    #[test]
    fn test_debug_format() {
        let tv = TermVectorsConsumerPerField::new("body".to_string());
        let debug = format!("{tv:?}");
        assert_contains!(debug, "TermVectorsConsumerPerField");
        assert_contains!(debug, "body");
    }

    #[test]
    fn test_has_data_false_when_no_vectors() {
        let tv = TermVectorsConsumerPerField::new("body".to_string());
        assert!(!tv.has_data());
    }

    #[test]
    fn test_has_data_true_when_terms_added() {
        let mut tv_th = TermsHash::new();
        let mut tv = TermVectorsConsumerPerField::new("body".to_string());
        tv.do_vectors = true;
        tv.do_vector_positions = true;

        tv.current_position = 0;
        tv.current_offset = TermOffset {
            start: 0,
            length: 5,
        };
        TermsHashPerFieldTrait::add_by_text_start(&mut tv, &mut tv_th, 100, 0);
        assert!(tv.has_data());
    }

    #[test]
    fn test_ensure_postings_capacity_grows() {
        let mut tv = TermVectorsConsumerPerField::new("body".to_string());
        assert_eq!(tv.postings_array.size(), 2);
        tv.ensure_postings_capacity(10);
        assert_ge!(tv.postings_array.size(), 11);
    }

    #[test]
    fn test_finish_document_self_owned() {
        let dir = MemoryDirectory::create();
        let segment_id = [0u8; 16];
        let mut writer = CompressingTermVectorsWriter::new(&dir, "_0", "", &segment_id).unwrap();

        let mut term_pool = new_term_pool();
        let mut tv_th = TermsHash::new();
        let mut tv = TermVectorsConsumerPerField::new("body".to_string());
        tv.do_vectors = true;
        tv.do_vector_positions = true;
        tv.do_vector_offsets = true;

        // Add two terms
        tv.current_position = 0;
        tv.current_offset = TermOffset {
            start: 0,
            length: 5,
        };
        TermsHashPerFieldTrait::add(&mut tv, &mut term_pool, &mut tv_th, b"hello", 0);

        tv.current_position = 1;
        tv.current_offset = TermOffset {
            start: 6,
            length: 5,
        };
        TermsHashPerFieldTrait::add(&mut tv, &mut term_pool, &mut tv_th, b"world", 0);

        // Flush via finish_document_self_owned
        writer.start_document(1);
        tv.finish_document_self_owned(0, &tv_th, &mut writer)
            .unwrap();
        writer.finish_document().unwrap();

        // After finish, state should be reset
        assert_eq!(tv.num_terms(), 0);
        assert!(!tv.do_vectors);
    }

    #[test]
    fn test_positions_and_offsets_combined() {
        let mut tv_th = TermsHash::new();
        let mut tv = TermVectorsConsumerPerField::new("body".to_string());
        tv.do_vectors = true;
        tv.do_vector_positions = true;
        tv.do_vector_offsets = true;

        // Term at position 0 with offsets [0, 5)
        tv.current_position = 0;
        tv.current_offset = TermOffset {
            start: 0,
            length: 5,
        };
        TermsHashPerFieldTrait::add_by_text_start(&mut tv, &mut tv_th, 100, 0);

        // Same term at position 2 with offsets [10, 15)
        tv.current_position = 2;
        tv.current_offset = TermOffset {
            start: 10,
            length: 5,
        };
        TermsHashPerFieldTrait::add_by_text_start(&mut tv, &mut tv_th, 100, 0);

        assert_eq!(tv.postings_array.freqs[0], 2);

        // Read position stream (stream 0)
        let (start, end) = tv.get_stream_range(&tv_th.int_pool, 0, 0);
        let mut reader = ByteSliceReader::new(&tv_th.byte_pool, start, end);
        let pos0 = read_vint(&mut reader);
        assert_eq!(pos0, 0); // position 0 << 1
        let pos1 = read_vint(&mut reader);
        assert_eq!(pos1, 4); // delta 2 << 1

        // Read offset stream (stream 1)
        let (start, end) = tv.get_stream_range(&tv_th.int_pool, 0, 1);
        let mut reader = ByteSliceReader::new(&tv_th.byte_pool, start, end);
        let off0_start = read_vint(&mut reader);
        assert_eq!(off0_start, 0);
        let off0_len = read_vint(&mut reader);
        assert_eq!(off0_len, 5);
        let off1_start = read_vint(&mut reader);
        assert_eq!(off1_start, 5); // delta: 10 - 5
        let off1_len = read_vint(&mut reader);
        assert_eq!(off1_len, 5);
    }
}
