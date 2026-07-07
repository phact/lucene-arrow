// SPDX-License-Identifier: Apache-2.0

//! [`FieldConsumer`] that accumulates terms and postings, then flushes via
//! the block tree terms dictionary and postings codecs.

use std::collections::HashMap;
use std::fmt;
use std::io;
use std::mem;

use crate::analysis::Token;
use crate::codecs::competitive_impact::BufferedNormsLookup;
use crate::codecs::lucene103::blocktree_writer::{BlockTreeTermsWriter, BufferedFieldTerms};
use crate::document::{IndexOptions, TermOffset};
use crate::index::field::{Field, InvertableValue};
use crate::index::pipeline::consumer::{FieldConsumer, TokenInterest};
use crate::index::pipeline::segment_accumulator::SegmentAccumulator;
use crate::index::pipeline::segment_context::SegmentContext;
use crate::index::pipeline::terms_hash::{FreqProxTermsWriterPerField, TermsHash};

/// Accumulates postings (terms, frequencies, positions) for indexed fields
/// and writes them at flush time via the block tree + postings codecs.
///
/// Owns a shared `TermsHash` (int pool + byte pool) that all per-field
/// `FreqProxTermsWriterPerField` instances write into.
pub struct PostingsConsumer {
    /// field_id → per-field postings state
    per_field: HashMap<u32, PerFieldState>,
    /// Shared pools for all per-field writers.
    terms_hash: TermsHash,
    current_doc_id: i32,
    current_position: i32,
}

impl mem_dbg::MemSize for PostingsConsumer {
    fn mem_size_rec(
        &self,
        flags: mem_dbg::SizeFlags,
        refs: &mut mem_dbg::HashMap<usize, usize>,
    ) -> usize {
        // Estimate per-field overhead
        let per_field_size: usize = self
            .per_field
            .values()
            .map(|s| s.mem_size_rec(flags, refs))
            .sum();
        per_field_size + self.terms_hash.mem_size_rec(flags, refs) + mem::size_of::<Self>()
    }
}

/// Per-field state tracked by PostingsConsumer.
struct PerFieldState {
    writer: FreqProxTermsWriterPerField,
    field_name: String,
    field_number: u32,
    index_options: IndexOptions,
}

impl mem_dbg::MemSize for PerFieldState {
    fn mem_size_rec(
        &self,
        flags: mem_dbg::SizeFlags,
        refs: &mut mem_dbg::HashMap<usize, usize>,
    ) -> usize {
        mem::size_of::<Self>() + self.field_name.len() + self.writer.mem_size_rec(flags, refs)
    }
}

impl fmt::Debug for PostingsConsumer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PostingsConsumer")
            .field("field_count", &self.per_field.len())
            .finish()
    }
}

impl PostingsConsumer {
    /// Creates a new consumer with empty pools.
    pub fn new() -> Self {
        Self {
            per_field: HashMap::new(),
            terms_hash: TermsHash::new(),
            current_doc_id: 0,
            current_position: 0,
        }
    }
}

impl Default for PostingsConsumer {
    fn default() -> Self {
        Self::new()
    }
}

impl FieldConsumer for PostingsConsumer {
    fn start_document(&mut self, doc_id: i32) -> io::Result<()> {
        self.current_doc_id = doc_id;
        Ok(())
    }

    fn start_field(
        &mut self,
        field_id: u32,
        field: &Field,
        accumulator: &mut SegmentAccumulator,
    ) -> io::Result<TokenInterest> {
        let opts = field.field_type().index_options();
        if opts == IndexOptions::None {
            return Ok(TokenInterest::NoTokens);
        }

        self.current_position = 0;

        let state = self
            .per_field
            .entry(field_id)
            .or_insert_with(|| PerFieldState {
                writer: FreqProxTermsWriterPerField::new(field.name().to_string(), opts),
                field_name: field.name().to_string(),
                field_number: field_id,
                index_options: opts,
            });

        // Non-tokenized indexed fields: record the exact value as a single term
        if let Some(InvertableValue::ExactMatch(value)) = field.field_type().invertable() {
            state.writer.current_position = 0;
            state.writer.current_offset = TermOffset::default();
            state.writer.add(
                accumulator.term_byte_pool_mut(),
                &mut self.terms_hash,
                value.as_bytes(),
                self.current_doc_id,
            )?;
            return Ok(TokenInterest::NoTokens);
        }

        // Feature fields: single term with explicit freq encoding.
        if let Some(InvertableValue::Feature(term_name, value)) = field.field_type().invertable() {
            let freq = (f32::to_bits(*value) >> 15) as i32;
            // For feature fields, we add the term then manually set the freq.
            // The FreqProx encoding handles this: after add(), the pending
            // doc has freq=1. We need to override it.
            state.writer.current_position = 0;
            state.writer.current_offset = TermOffset::default();
            let tid = state.writer.add(
                accumulator.term_byte_pool_mut(),
                &mut self.terms_hash,
                term_name.as_bytes(),
                self.current_doc_id,
            )?;
            // Override the term frequency for feature fields
            if let Some(ref mut freqs) = state.writer.postings_array.term_freqs {
                freqs[tid] = freq;
            }
            return Ok(TokenInterest::NoTokens);
        }

        Ok(TokenInterest::WantsTokens)
    }

    fn add_token(
        &mut self,
        field_id: u32,
        _field: &Field,
        token: &Token<'_>,
        accumulator: &mut SegmentAccumulator,
    ) -> io::Result<()> {
        self.current_position += token.position_increment;
        let position = self.current_position - 1;

        let state = self
            .per_field
            .get_mut(&field_id)
            .expect("start_field must precede add_token");

        state.writer.current_position = position;
        state.writer.current_offset = token.offset;
        let term_id = state.writer.add(
            accumulator.term_byte_pool_mut(),
            &mut self.terms_hash,
            token.text.as_bytes(),
            self.current_doc_id,
        )?;

        // Set hint for term vectors consumer
        let text_start = state.writer.postings_array.base.text_starts[term_id];
        accumulator.set_text_start_hint(text_start);

        Ok(())
    }

    fn finish_field(
        &mut self,
        _field_id: u32,
        _field: &Field,
        accumulator: &mut SegmentAccumulator,
    ) -> io::Result<()> {
        accumulator.clear_text_start_hint();
        Ok(())
    }

    fn finish_document(
        &mut self,
        _doc_id: i32,
        _accumulator: &mut SegmentAccumulator,
        _context: &SegmentContext,
    ) -> io::Result<()> {
        Ok(())
    }

    fn flush(
        &mut self,
        context: &SegmentContext,
        accumulator: &SegmentAccumulator,
    ) -> io::Result<Vec<String>> {
        if self.per_field.is_empty() {
            return Ok(vec![]);
        }

        // Flush pending docs for all fields
        for state in self.per_field.values_mut() {
            state.writer.flush_pending_docs(&mut self.terms_hash);
        }

        // Sort fields by field number for deterministic output
        let mut field_ids: Vec<u32> = self.per_field.keys().copied().collect();
        field_ids.sort();

        // Sort terms for each field (term bytes are in the accumulator's pool)
        let term_byte_pool = accumulator.term_byte_pool();
        for state in self.per_field.values_mut() {
            if state.writer.num_terms() > 0 {
                state.writer.sort_terms(term_byte_pool);
            }
        }

        // Determine the maximum IndexOptions across all fields
        let max_index_options = self
            .per_field
            .values()
            .map(|s| s.index_options)
            .max()
            .unwrap_or(IndexOptions::Docs);

        // Suffix must match PerFieldPostingsFormat.suffix written in .fnm attributes
        let per_field_suffix = "Lucene103_0";

        let mut writer = BlockTreeTermsWriter::new(
            &*context.directory,
            &context.segment_name,
            per_field_suffix,
            &context.segment_id,
            max_index_options,
        )?;

        let norms_data = accumulator.norms();

        for &field_id in &field_ids {
            let state = self.per_field.get(&field_id).unwrap();

            if state.writer.num_terms() == 0 {
                continue;
            }

            let norms = if let Some(field_norms) = norms_data.get(&field_id) {
                BufferedNormsLookup::new(&field_norms.values, &field_norms.docs)
            } else {
                BufferedNormsLookup::no_norms()
            };

            let field_terms = BufferedFieldTerms::new(
                &state.writer,
                term_byte_pool,
                &self.terms_hash,
                &state.field_name,
                state.field_number,
            );

            writer.write_field(&field_terms, &norms)?;
        }

        writer.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::field::{feature, stored, string, text};
    use crate::store::MemoryDirectory;
    use assertables::*;

    fn test_context() -> SegmentContext {
        SegmentContext {
            directory: MemoryDirectory::create(),
            segment_name: "_0".to_string(),
            segment_id: [0u8; 16],
        }
    }

    fn make_token(text: &str) -> Token<'_> {
        Token {
            text,
            offset: TermOffset {
                start: 0,
                length: text.len() as u16,
            },
            position_increment: 1,
        }
    }

    /// Process a single document with one text field through the consumer.
    fn process_doc(
        consumer: &mut PostingsConsumer,
        doc_id: i32,
        field_id: u32,
        field: &Field,
        tokens: &[&str],
        acc: &mut SegmentAccumulator,
    ) {
        let context = test_context();
        consumer.start_document(doc_id).unwrap();
        consumer.start_field(field_id, field, acc).unwrap();
        for &t in tokens {
            let token = make_token(t);
            consumer.add_token(field_id, field, &token, acc).unwrap();
        }
        consumer.finish_field(field_id, field, acc).unwrap();
        consumer.finish_document(doc_id, acc, &context).unwrap();
    }

    #[test]
    fn single_doc_produces_postings_files() {
        let context = test_context();
        let mut consumer = PostingsConsumer::new();
        let mut acc = SegmentAccumulator::new();
        let field = text("body").stored().value("ignored");

        process_doc(&mut consumer, 0, 0, &field, &["hello", "world"], &mut acc);

        let names = consumer.flush(&context, &acc).unwrap();

        assert_len_eq_x!(&names, 6);
        assert!(names.contains(&"_0_Lucene103_0.tim".to_string()));
        assert!(names.contains(&"_0_Lucene103_0.tip".to_string()));
        assert!(names.contains(&"_0_Lucene103_0.tmd".to_string()));
        assert!(names.contains(&"_0_Lucene103_0.doc".to_string()));
        assert!(names.contains(&"_0_Lucene103_0.pos".to_string()));
        assert!(names.contains(&"_0_Lucene103_0.psm".to_string()));
    }

    #[test]
    fn multiple_docs_produce_postings() {
        let context = test_context();
        let mut consumer = PostingsConsumer::new();
        let mut acc = SegmentAccumulator::new();
        let field = text("body").stored().value("ignored");

        process_doc(&mut consumer, 0, 0, &field, &["hello", "world"], &mut acc);
        process_doc(&mut consumer, 1, 0, &field, &["hello", "rust"], &mut acc);
        process_doc(&mut consumer, 2, 0, &field, &["world", "rust"], &mut acc);

        let names = consumer.flush(&context, &acc).unwrap();
        assert_ge!(names.len(), 5);
    }

    #[test]
    fn stored_only_field_ignored() {
        let context = test_context();
        let mut consumer = PostingsConsumer::new();
        let mut acc = SegmentAccumulator::new();
        let field = stored("title").string("hello");

        consumer.start_document(0).unwrap();
        let interest = consumer.start_field(0, &field, &mut acc).unwrap();
        assert_eq!(interest, TokenInterest::NoTokens);
        consumer.finish_field(0, &field, &mut acc).unwrap();
        consumer.finish_document(0, &mut acc, &context).unwrap();

        let names = consumer.flush(&context, &acc).unwrap();
        assert_is_empty!(&names);
    }

    #[test]
    fn multiple_fields_produce_postings() {
        let context = test_context();
        let mut consumer = PostingsConsumer::new();
        let mut acc = SegmentAccumulator::new();
        let title = text("title").stored().value("ignored");
        let body = text("body").stored().value("ignored");

        consumer.start_document(0).unwrap();

        consumer.start_field(0, &title, &mut acc).unwrap();
        let token = make_token("hello");
        consumer.add_token(0, &title, &token, &mut acc).unwrap();
        consumer.finish_field(0, &title, &mut acc).unwrap();

        consumer.start_field(1, &body, &mut acc).unwrap();
        let token = make_token("world");
        consumer.add_token(1, &body, &token, &mut acc).unwrap();
        consumer.finish_field(1, &body, &mut acc).unwrap();

        consumer.finish_document(0, &mut acc, &context).unwrap();

        let names = consumer.flush(&context, &acc).unwrap();
        assert_ge!(names.len(), 5);
    }

    #[test]
    fn no_tokenized_fields_produces_no_files() {
        let context = test_context();
        let mut consumer = PostingsConsumer::new();
        let acc = SegmentAccumulator::new();

        let names = consumer.flush(&context, &acc).unwrap();
        assert_is_empty!(&names);
    }

    #[test]
    fn indexed_field_returns_no_tokens() {
        let context = test_context();
        let mut consumer = PostingsConsumer::new();
        let mut acc = SegmentAccumulator::new();
        let field = string("title").stored().value("hello");

        consumer.start_document(0).unwrap();
        let interest = consumer.start_field(0, &field, &mut acc).unwrap();
        assert_eq!(interest, TokenInterest::NoTokens);
        consumer.finish_field(0, &field, &mut acc).unwrap();
        consumer.finish_document(0, &mut acc, &context).unwrap();
    }

    #[test]
    fn indexed_field_produces_postings_without_positions() {
        let context = test_context();
        let mut consumer = PostingsConsumer::new();
        let mut acc = SegmentAccumulator::new();

        for doc_id in 0..3 {
            let field = string("title").stored().value(format!("doc_{doc_id}"));
            consumer.start_document(doc_id).unwrap();
            consumer.start_field(0, &field, &mut acc).unwrap();
            consumer.finish_field(0, &field, &mut acc).unwrap();
            consumer
                .finish_document(doc_id, &mut acc, &context)
                .unwrap();
        }

        let names = consumer.flush(&context, &acc).unwrap();

        // Should produce terms files but NO positions file
        assert!(names.iter().any(|n| n.ends_with(".tim")));
        assert!(names.iter().any(|n| n.ends_with(".doc")));
        assert!(!names.iter().any(|n| n.ends_with(".pos")));
    }

    #[test]
    fn mixed_indexed_and_tokenized_fields() {
        let context = test_context();
        let mut consumer = PostingsConsumer::new();
        let mut acc = SegmentAccumulator::new();

        let title = string("title").stored().value("hello");
        let body = text("body").stored().value("ignored");

        consumer.start_document(0).unwrap();

        // StringField — handled in start_field, returns NoTokens
        let interest = consumer.start_field(0, &title, &mut acc).unwrap();
        assert_eq!(interest, TokenInterest::NoTokens);
        consumer.finish_field(0, &title, &mut acc).unwrap();

        // TextField — returns WantsTokens
        let interest = consumer.start_field(1, &body, &mut acc).unwrap();
        assert_eq!(interest, TokenInterest::WantsTokens);
        let token = make_token("world");
        consumer.add_token(1, &body, &token, &mut acc).unwrap();
        consumer.finish_field(1, &body, &mut acc).unwrap();

        consumer.finish_document(0, &mut acc, &context).unwrap();

        let names = consumer.flush(&context, &acc).unwrap();

        // Both fields produce terms
        assert!(names.iter().any(|n| n.ends_with(".tim")));
        assert!(names.iter().any(|n| n.ends_with(".doc")));
        // Positions file exists because body field has positions
        assert!(names.iter().any(|n| n.ends_with(".pos")));
    }

    #[test]
    fn feature_field_returns_no_tokens() {
        let context = test_context();
        let mut consumer = PostingsConsumer::new();
        let mut acc = SegmentAccumulator::new();
        let field = feature("pagerank").value("score", 0.95);

        consumer.start_document(0).unwrap();
        let interest = consumer.start_field(0, &field, &mut acc).unwrap();
        assert_eq!(interest, TokenInterest::NoTokens);
        consumer.finish_field(0, &field, &mut acc).unwrap();
        consumer.finish_document(0, &mut acc, &context).unwrap();
    }

    #[test]
    fn feature_field_produces_postings_without_positions() {
        let context = test_context();
        let mut consumer = PostingsConsumer::new();
        let mut acc = SegmentAccumulator::new();

        for doc_id in 0..3 {
            let field = feature("pagerank").value("score", 0.95);
            consumer.start_document(doc_id).unwrap();
            consumer.start_field(0, &field, &mut acc).unwrap();
            consumer.finish_field(0, &field, &mut acc).unwrap();
            consumer
                .finish_document(doc_id, &mut acc, &context)
                .unwrap();
        }

        let names = consumer.flush(&context, &acc).unwrap();

        // Should produce terms files but NO positions file
        assert!(names.iter().any(|n| n.ends_with(".tim")));
        assert!(names.iter().any(|n| n.ends_with(".doc")));
        assert!(!names.iter().any(|n| n.ends_with(".pos")));
    }

    #[test]
    fn mem_size_baseline() {
        use mem_dbg::{MemSize, SizeFlags};
        let consumer = PostingsConsumer::new();
        // Should have some baseline size from the TermsHash pools
        assert_gt!(consumer.mem_size(SizeFlags::CAPACITY), 0);
    }

    #[test]
    fn mem_size_grows_with_documents() {
        use mem_dbg::{MemSize, SizeFlags};
        let mut consumer = PostingsConsumer::new();
        let mut acc = SegmentAccumulator::new();
        let field = text("body").stored().value("ignored");

        let baseline = consumer.mem_size(SizeFlags::CAPACITY);

        for doc_id in 0..50 {
            process_doc(
                &mut consumer,
                doc_id,
                0,
                &field,
                &["hello", "world", "foo", "bar"],
                &mut acc,
            );
        }

        let after = consumer.mem_size(SizeFlags::CAPACITY);
        assert_gt!(after, baseline);
    }
}
