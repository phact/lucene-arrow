// SPDX-License-Identifier: Apache-2.0

//! [`FieldConsumer`] that accumulates per-document term vector data and
//! writes `.tvd`/`.tvx`/`.tvm` files at flush time.
//!
//! Uses pool-based `TermVectorsConsumerPerField` instances to accumulate
//! per-document term vector data, then streams it to the codec writer
//! at `finish_document` time. Pools are reset per-document.

use std::collections::HashMap;
use std::fmt;
use std::io;
use std::mem;

use crate::analysis::Token;
use crate::codecs::lucene90::term_vectors::{CompressingTermVectorsWriter, TermVectorsWriter};
use crate::index::field::Field;
use crate::index::pipeline::consumer::{FieldConsumer, TokenInterest};
use crate::index::pipeline::segment_accumulator::SegmentAccumulator;
use crate::index::pipeline::segment_context::SegmentContext;
use crate::index::pipeline::term_vectors_consumer_per_field::TermVectorsConsumerPerField;
use crate::index::pipeline::terms_hash::{TermsHash, TermsHashPerFieldTrait};

/// Accumulates per-document term vector data during indexing and writes
/// `.tvd`/`.tvx`/`.tvm` files at flush time.
///
/// Owns a per-document `TermsHash` (TV pools) that is reset after each
/// document is written to the codec writer.
pub struct TermVectorsConsumer {
    /// Lazy-initialized codec writer.
    writer: Option<CompressingTermVectorsWriter>,
    /// Shared TV pools, reset per-document.
    tv_terms_hash: TermsHash,
    /// Per-field TV consumers, keyed by field_id.
    per_field: HashMap<u32, TvPerFieldState>,
    /// Fields that have TV data in the current document.
    active_field_ids: Vec<u32>,
    /// Whether the current document has any term vector fields.
    has_vectors: bool,
    /// Number of documents processed (for gap-filling non-TV docs).
    num_docs: i32,
    /// Last doc ID written to the TV writer.
    tv_last_doc_id: i32,
    /// Current doc ID being processed.
    current_doc_id: i32,
}

/// Per-field state for term vector processing.
struct TvPerFieldState {
    tv_pf: TermVectorsConsumerPerField,
    field_number: u32,
    /// Accumulated absolute position for the current field in the current document.
    current_position: i32,
}

impl mem_dbg::MemSize for TermVectorsConsumer {
    fn mem_size_rec(
        &self,
        flags: mem_dbg::SizeFlags,
        _refs: &mut mem_dbg::HashMap<usize, usize>,
    ) -> usize {
        let writer_size = self.writer.as_ref().map(|w| w.mem_size(flags)).unwrap_or(0);
        let tv_hash_size = self.tv_terms_hash.mem_size(flags);
        mem::size_of::<Self>() + writer_size + tv_hash_size
    }
}

impl fmt::Debug for TermVectorsConsumer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TermVectorsConsumer")
            .field("field_count", &self.per_field.len())
            .field("has_vectors", &self.has_vectors)
            .finish()
    }
}

impl Default for TermVectorsConsumer {
    fn default() -> Self {
        Self::new()
    }
}

impl TermVectorsConsumer {
    /// Creates a new consumer with no accumulated data.
    pub fn new() -> Self {
        Self {
            writer: None,
            tv_terms_hash: TermsHash::new(),
            per_field: HashMap::new(),
            active_field_ids: Vec::new(),
            has_vectors: false,
            num_docs: 0,
            tv_last_doc_id: 0,
            current_doc_id: 0,
        }
    }
}

impl FieldConsumer for TermVectorsConsumer {
    fn start_document(&mut self, doc_id: i32) -> io::Result<()> {
        self.current_doc_id = doc_id;
        self.has_vectors = false;
        self.active_field_ids.clear();
        Ok(())
    }

    fn start_field(
        &mut self,
        field_id: u32,
        field: &Field,
        _accumulator: &mut SegmentAccumulator,
    ) -> io::Result<TokenInterest> {
        let Some(opts) = field.field_type().term_vector_options() else {
            return Ok(TokenInterest::NoTokens);
        };

        self.has_vectors = true;

        let state = self
            .per_field
            .entry(field_id)
            .or_insert_with(|| TvPerFieldState {
                tv_pf: TermVectorsConsumerPerField::new(field.name().to_string()),
                field_number: field_id,
                current_position: -1,
            });

        // Reset position tracking for this field in this document
        state.current_position = -1;

        // Configure for this document's field
        state.tv_pf.do_vectors = true;
        state.tv_pf.do_vector_positions = opts.has_positions();
        state.tv_pf.do_vector_offsets = opts.has_offsets();
        state.tv_pf.do_vector_payloads = opts.has_payloads();

        self.active_field_ids.push(field_id);

        Ok(TokenInterest::WantsTokens)
    }

    fn add_token(
        &mut self,
        field_id: u32,
        _field: &Field,
        token: &Token<'_>,
        accumulator: &mut SegmentAccumulator,
    ) -> io::Result<()> {
        let state = self
            .per_field
            .get_mut(&field_id)
            .expect("add_token called without active TV field");

        // Accumulate absolute position from position_increment
        state.current_position += token.position_increment;

        state.tv_pf.current_position = state.current_position;
        state.tv_pf.current_offset = token.offset;

        // Read the text_start hint set by PostingsConsumer for this token
        let text_start = accumulator.take_text_start_hint();

        // Intern by pool offset — no byte copying, references shared term byte pool
        TermsHashPerFieldTrait::add_by_text_start(
            &mut state.tv_pf,
            &mut self.tv_terms_hash,
            text_start,
            self.current_doc_id,
        );

        Ok(())
    }

    fn finish_field(
        &mut self,
        _field_id: u32,
        _field: &Field,
        _accumulator: &mut SegmentAccumulator,
    ) -> io::Result<()> {
        // Data is in the pools — nothing to do here
        Ok(())
    }

    fn finish_document(
        &mut self,
        doc_id: i32,
        accumulator: &mut SegmentAccumulator,
        context: &SegmentContext,
    ) -> io::Result<()> {
        self.num_docs += 1;

        if !self.has_vectors {
            return Ok(());
        }

        // Lazy-create the writer
        if self.writer.is_none() {
            self.writer = Some(CompressingTermVectorsWriter::new(
                &*context.directory,
                &context.segment_name,
                "",
                &context.segment_id,
            )?);
        }

        let writer = self.writer.as_mut().unwrap();

        // Fill gaps for docs without vectors
        while self.tv_last_doc_id < doc_id {
            writer.start_document(0);
            writer.finish_document()?;
            self.tv_last_doc_id += 1;
        }

        // Count active fields
        let num_vector_fields = self.active_field_ids.len() as i32;
        writer.start_document(num_vector_fields);

        // Sort active fields by field number for deterministic output
        self.active_field_ids.sort();

        let term_byte_pool = accumulator.term_byte_pool();
        for &field_id in &self.active_field_ids.clone() {
            let state = self.per_field.get_mut(&field_id).unwrap();
            if state.tv_pf.has_data() {
                state.tv_pf.finish_document(
                    state.field_number,
                    term_byte_pool,
                    &self.tv_terms_hash,
                    writer,
                )?;
            }
        }

        writer.finish_document()?;
        self.tv_last_doc_id = doc_id + 1;

        // Reset TV pools for next document
        self.tv_terms_hash.reset();
        for state in self.per_field.values_mut() {
            state.tv_pf.reset();
        }

        Ok(())
    }

    fn flush(
        &mut self,
        context: &SegmentContext,
        accumulator: &SegmentAccumulator,
    ) -> io::Result<Vec<String>> {
        let Some(mut writer) = self.writer.take() else {
            return Ok(Vec::new());
        };

        writer.finish(accumulator.doc_count())?;
        Ok(CompressingTermVectorsWriter::file_names(
            &context.segment_name,
            "",
        ))
    }
}

#[cfg(test)]
mod tests {

    use assertables::*;

    use super::*;
    use crate::document::TermOffset;
    use crate::index::field::{TermVectorOptions, text};
    use crate::store::MemoryDirectory;

    use crate::util::bytes_ref_hash::BytesRefHash;

    fn test_context() -> SegmentContext {
        SegmentContext {
            directory: MemoryDirectory::create(),
            segment_name: "_0".to_string(),
            segment_id: [0u8; 16],
        }
    }

    /// Simulates what PostingsConsumer does: intern term bytes and set the hint.
    fn set_hint_for_token(accum: &mut SegmentAccumulator, hash: &mut BytesRefHash, text: &[u8]) {
        let term_id = hash.add(accum.term_byte_pool_mut(), text);
        let id = if term_id >= 0 {
            term_id as usize
        } else {
            ((-term_id) - 1) as usize
        };
        let text_start = hash.byte_start(id);
        accum.set_text_start_hint(text_start);
    }

    #[test]
    fn test_no_tv_fields_returns_no_tokens() {
        let mut consumer = TermVectorsConsumer::new();
        let mut accum = SegmentAccumulator::new();
        let field = text("contents").value("hello world");
        let interest = consumer.start_field(0, &field, &mut accum).unwrap();
        assert_eq!(interest, TokenInterest::NoTokens);
    }

    #[test]
    fn test_tv_field_returns_wants_tokens() {
        let mut consumer = TermVectorsConsumer::new();
        let mut accum = SegmentAccumulator::new();
        let field = text("contents")
            .with_term_vectors(TermVectorOptions::PositionsAndOffsets)
            .value("hello world");
        let interest = consumer.start_field(0, &field, &mut accum).unwrap();
        assert_eq!(interest, TokenInterest::WantsTokens);
    }

    #[test]
    fn test_doc_without_tv_not_written() {
        let context = test_context();
        let mut consumer = TermVectorsConsumer::new();
        let mut accum = SegmentAccumulator::new();

        consumer.start_document(0).unwrap();
        let field = text("contents").value("hello");
        consumer.start_field(0, &field, &mut accum).unwrap();
        consumer.finish_field(0, &field, &mut accum).unwrap();
        consumer.finish_document(0, &mut accum, &context).unwrap();

        // No writer created
        assert!(consumer.writer.is_none());
    }

    #[test]
    fn mem_size_empty_is_small() {
        use mem_dbg::{MemSize, SizeFlags};
        let consumer = TermVectorsConsumer::new();
        assert_lt!(consumer.mem_size(SizeFlags::CAPACITY), 68_000);
    }

    #[test]
    fn default_creates_new() {
        let consumer = TermVectorsConsumer::default();
        assert!(consumer.writer.is_none());
    }

    #[test]
    fn debug_format() {
        let consumer = TermVectorsConsumer::new();
        let debug = format!("{consumer:?}");
        assert_contains!(debug, "TermVectorsConsumer");
        assert_contains!(debug, "has_vectors");
    }

    #[test]
    fn full_pipeline_add_token_finish_flush() {
        let context = test_context();
        let mut consumer = TermVectorsConsumer::new();
        let mut accum = SegmentAccumulator::new();
        let mut hash = BytesRefHash::new(4);

        let field = text("contents")
            .with_term_vectors(TermVectorOptions::PositionsAndOffsets)
            .value("hello world");

        // Doc 0 with term vectors
        consumer.start_document(0).unwrap();
        consumer.start_field(0, &field, &mut accum).unwrap();

        let token1 = Token {
            text: "hello",
            position_increment: 1,
            offset: TermOffset {
                start: 0,
                length: 5,
            },
        };
        set_hint_for_token(&mut accum, &mut hash, b"hello");
        consumer.add_token(0, &field, &token1, &mut accum).unwrap();

        let token2 = Token {
            text: "world",
            position_increment: 1,
            offset: TermOffset {
                start: 6,
                length: 5,
            },
        };
        set_hint_for_token(&mut accum, &mut hash, b"world");
        consumer.add_token(0, &field, &token2, &mut accum).unwrap();

        consumer.finish_field(0, &field, &mut accum).unwrap();
        consumer.finish_document(0, &mut accum, &context).unwrap();
        accum.increment_doc_count();

        // Writer should have been lazily created
        assert!(consumer.writer.is_some());

        // Flush
        let files = consumer.flush(&context, &accum).unwrap();
        assert!(!files.is_empty());

        // Verify files exist in the directory
        let guard = &*context.directory;
        for name in &files {
            let data = guard.read_file(name).unwrap();
            assert_not_empty!(data);
        }
    }

    #[test]
    fn flush_fills_gap_docs() {
        let context = test_context();
        let mut consumer = TermVectorsConsumer::new();
        let mut accum = SegmentAccumulator::new();
        let mut hash = BytesRefHash::new(4);

        let field = text("contents")
            .with_term_vectors(TermVectorOptions::PositionsAndOffsets)
            .value("hello");

        // Doc 0: no term vectors
        consumer.start_document(0).unwrap();
        let plain = text("contents").value("no tv");
        consumer.start_field(0, &plain, &mut accum).unwrap();
        consumer.finish_field(0, &plain, &mut accum).unwrap();
        consumer.finish_document(0, &mut accum, &context).unwrap();
        accum.increment_doc_count();

        // Doc 1: has term vectors — will trigger gap fill for doc 0
        consumer.start_document(1).unwrap();
        consumer.start_field(0, &field, &mut accum).unwrap();
        let token = Token {
            text: "hello",
            position_increment: 1,
            offset: TermOffset {
                start: 0,
                length: 5,
            },
        };
        set_hint_for_token(&mut accum, &mut hash, b"hello");
        consumer.add_token(0, &field, &token, &mut accum).unwrap();
        consumer.finish_field(0, &field, &mut accum).unwrap();
        consumer.finish_document(1, &mut accum, &context).unwrap();
        accum.increment_doc_count();

        let files = consumer.flush(&context, &accum).unwrap();
        assert!(!files.is_empty());
    }

    #[test]
    fn flush_no_tv_returns_empty() {
        let context = test_context();
        let mut consumer = TermVectorsConsumer::new();
        let accum = SegmentAccumulator::new();
        let files = consumer.flush(&context, &accum).unwrap();
        assert_is_empty!(&files);
    }

    #[test]
    fn multiple_fields_sorted_by_number() {
        let context = test_context();
        let mut consumer = TermVectorsConsumer::new();
        let mut accum = SegmentAccumulator::new();
        let mut hash = BytesRefHash::new(4);

        let field_a = text("zzz")
            .with_term_vectors(TermVectorOptions::PositionsAndOffsets)
            .value("alpha");
        let field_b = text("aaa")
            .with_term_vectors(TermVectorOptions::PositionsAndOffsets)
            .value("beta");

        consumer.start_document(0).unwrap();

        // Field 5 first, then field 2 — should be sorted by field number
        consumer.start_field(5, &field_a, &mut accum).unwrap();
        let t1 = Token {
            text: "alpha",
            position_increment: 1,
            offset: TermOffset {
                start: 0,
                length: 5,
            },
        };
        set_hint_for_token(&mut accum, &mut hash, b"alpha");
        consumer.add_token(5, &field_a, &t1, &mut accum).unwrap();
        consumer.finish_field(5, &field_a, &mut accum).unwrap();

        consumer.start_field(2, &field_b, &mut accum).unwrap();
        let t2 = Token {
            text: "beta",
            position_increment: 1,
            offset: TermOffset {
                start: 0,
                length: 4,
            },
        };
        set_hint_for_token(&mut accum, &mut hash, b"beta");
        consumer.add_token(2, &field_b, &t2, &mut accum).unwrap();
        consumer.finish_field(2, &field_b, &mut accum).unwrap();

        consumer.finish_document(0, &mut accum, &context).unwrap();
        accum.increment_doc_count();

        let files = consumer.flush(&context, &accum).unwrap();
        assert!(!files.is_empty());
    }
}
