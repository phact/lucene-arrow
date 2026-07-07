// SPDX-License-Identifier: Apache-2.0

//! [`FieldConsumer`] that computes norms from token counts and writes `.nvm`, `.nvd`.

use std::io;

use crate::analysis::Token;
use crate::codecs::lucene90::norms;
use crate::codecs::lucene90::norms_producer::BufferedNormsProducer;
use crate::document::{DocValuesType, IndexOptions};
use crate::index::FieldInfo;
use crate::index::field::Field;
use crate::index::field_infos::PointDimensionConfig;
use crate::index::pipeline::consumer::{FieldConsumer, TokenInterest};
use crate::index::pipeline::segment_accumulator::SegmentAccumulator;
use crate::index::pipeline::segment_context::SegmentContext;
use crate::util::small_float;

/// Computes and writes per-field norms from token counts.
///
/// For each tokenized field that has norms enabled, counts tokens via
/// `add_token` and computes a SmallFloat-encoded norm in `finish_field`.
/// Norm values are stored in the [`SegmentAccumulator`] so that other
/// consumers (e.g., postings) can access them at flush time.
/// At flush time, reads norms from the accumulator and writes `.nvm`
/// and `.nvd` via the norms codec.
#[derive(Debug, Default, mem_dbg::MemSize)]
#[mem_size(flat)]
pub struct NormsConsumer {
    current_token_count: i32,
    current_has_norms: bool,
    current_doc_id: i32,
}

impl NormsConsumer {
    /// Creates a new consumer.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Computes the BM25 norm value for a field.
///
/// Encodes the field length (token count) as a single byte using
/// SmallFloat, then sign-extends through i8 to match Java's widening.
fn compute_norm(field_length: i32) -> i64 {
    small_float::int_to_byte4(field_length) as i8 as i64
}

impl FieldConsumer for NormsConsumer {
    fn start_document(&mut self, doc_id: i32) -> io::Result<()> {
        self.current_doc_id = doc_id;
        Ok(())
    }

    fn start_field(
        &mut self,
        _field_id: u32,
        field: &Field,
        _accumulator: &mut SegmentAccumulator,
    ) -> io::Result<TokenInterest> {
        self.current_has_norms = field.field_type().has_norms();
        self.current_token_count = 0;

        if self.current_has_norms {
            Ok(TokenInterest::WantsTokens)
        } else {
            Ok(TokenInterest::NoTokens)
        }
    }

    fn add_token(
        &mut self,
        _field_id: u32,
        _field: &Field,
        token: &Token<'_>,
        _accumulator: &mut SegmentAccumulator,
    ) -> io::Result<()> {
        self.current_token_count += token.position_increment;
        Ok(())
    }

    fn finish_field(
        &mut self,
        field_id: u32,
        field: &Field,
        accumulator: &mut SegmentAccumulator,
    ) -> io::Result<()> {
        if self.current_has_norms && self.current_token_count > 0 {
            let norm = compute_norm(self.current_token_count);
            accumulator.record_norm(field_id, field.name(), self.current_doc_id, norm);
        }
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
        let norms_data = accumulator.norms();
        if norms_data.is_empty() {
            return Ok(vec![]);
        }

        let producer = BufferedNormsProducer::new(norms_data);

        // Build FieldInfo objects for each norms field, sorted by field number
        let mut field_infos: Vec<FieldInfo> = norms_data
            .iter()
            .map(|(&field_number, data)| {
                FieldInfo::new(
                    data.field_name.clone(),
                    field_number,
                    false,
                    false, // omit_norms = false → has_norms() = true
                    IndexOptions::DocsAndFreqsAndPositions,
                    DocValuesType::None,
                    PointDimensionConfig::default(),
                )
            })
            .collect();
        field_infos.sort_by_key(|f| f.number());
        let field_info_refs: Vec<&FieldInfo> = field_infos.iter().collect();

        norms::write(
            &*context.directory,
            &context.segment_name,
            "",
            &context.segment_id,
            &field_info_refs,
            &producer,
            accumulator.doc_count(),
        )
    }
}

#[cfg(test)]
mod tests {
    use std::mem;

    use assertables::*;

    use super::*;
    use crate::document::TermOffset;
    use crate::index::field::{stored, text};
    use crate::store::MemoryDirectory;

    fn test_context() -> SegmentContext {
        SegmentContext {
            directory: MemoryDirectory::create(),
            segment_name: "_0".to_string(),
            segment_id: [0u8; 16],
        }
    }

    /// Simulates the worker calling start_field → add_token (N times) → finish_field
    /// for a tokenized field with the given token count.
    fn process_tokenized_field(
        consumer: &mut NormsConsumer,
        field_id: u32,
        field: &Field,
        token_count: i32,
        acc: &mut SegmentAccumulator,
    ) {
        consumer.start_field(field_id, field, acc).unwrap();
        let mut buf = String::new();
        for _ in 0..token_count {
            buf.clear();
            buf.push_str("token");
            let token = Token {
                text: &buf,
                offset: TermOffset {
                    start: 0,
                    length: 5,
                },
                position_increment: 1,
            };
            consumer.add_token(field_id, field, &token, acc).unwrap();
        }
        consumer.finish_field(field_id, field, acc).unwrap();
    }

    #[test]
    fn computes_norms_from_token_count() {
        let context = test_context();
        let mut consumer = NormsConsumer::new();
        let mut acc = SegmentAccumulator::new();
        let field = text("body").stored().value("ignored");

        // Doc 0: 3 tokens, Doc 1: 10 tokens, Doc 2: 1 token
        for (doc_id, count) in [(0, 3), (1, 10), (2, 1)] {
            consumer.start_document(doc_id).unwrap();
            process_tokenized_field(&mut consumer, 0, &field, count, &mut acc);
            consumer
                .finish_document(doc_id, &mut acc, &context)
                .unwrap();
            acc.increment_doc_count();
        }

        let context = test_context();
        let names = consumer.flush(&context, &acc).unwrap();
        assert_len_eq_x!(&names, 2);
        assert_eq!(names[0], "_0.nvm");
        assert_eq!(names[1], "_0.nvd");
    }

    #[test]
    fn non_tokenized_produces_no_files() {
        let context = test_context();
        let mut consumer = NormsConsumer::new();
        let mut acc = SegmentAccumulator::new();
        let field = stored("title").string("ignored");

        consumer.start_document(0).unwrap();
        consumer.start_field(0, &field, &mut acc).unwrap();
        consumer.finish_field(0, &field, &mut acc).unwrap();
        consumer.finish_document(0, &mut acc, &context).unwrap();
        acc.increment_doc_count();

        let context = test_context();
        let names = consumer.flush(&context, &acc).unwrap();
        assert_is_empty!(&names);
    }

    #[test]
    fn zero_tokens_produces_no_norm_for_that_doc() {
        let context = test_context();
        let mut consumer = NormsConsumer::new();
        let mut acc = SegmentAccumulator::new();
        let field = text("body").stored().value("ignored");

        // Doc 0: 3 tokens (gets norm), Doc 1: 0 tokens (no norm)
        consumer.start_document(0).unwrap();
        process_tokenized_field(&mut consumer, 0, &field, 3, &mut acc);
        consumer.finish_document(0, &mut acc, &context).unwrap();
        acc.increment_doc_count();

        consumer.start_document(1).unwrap();
        process_tokenized_field(&mut consumer, 0, &field, 0, &mut acc);
        consumer.finish_document(1, &mut acc, &context).unwrap();
        acc.increment_doc_count();

        let context = test_context();
        let names = consumer.flush(&context, &acc).unwrap();
        // Should still write files (1 doc has norms)
        assert_len_eq_x!(&names, 2);
    }

    #[test]
    fn norms_stored_in_accumulator() {
        let context = test_context();
        let mut consumer = NormsConsumer::new();
        let mut acc = SegmentAccumulator::new();
        let field = text("body").stored().value("ignored");

        consumer.start_document(0).unwrap();
        process_tokenized_field(&mut consumer, 0, &field, 5, &mut acc);
        consumer.finish_document(0, &mut acc, &context).unwrap();
        acc.increment_doc_count();

        consumer.start_document(1).unwrap();
        process_tokenized_field(&mut consumer, 0, &field, 3, &mut acc);
        consumer.finish_document(1, &mut acc, &context).unwrap();
        acc.increment_doc_count();

        let norms = acc.norms();
        assert_len_eq_x!(norms, 1); // one field
        let field_norms = &norms[&0];
        assert_eq!(field_norms.field_name, "body");
        assert_eq!(field_norms.docs, vec![0, 1]);
        assert_len_eq_x!(&field_norms.values, 2);
    }

    #[test]
    fn compute_norm_matches_expected_values() {
        // SmallFloat encoding: small values are identity, larger are lossy
        assert_eq!(compute_norm(1), 1);
        assert_eq!(compute_norm(2), 2);
        assert_eq!(compute_norm(3), 3);

        // Larger values get compressed — verify sign extension
        let norm_100 = compute_norm(100);
        assert_ne!(norm_100, 100); // lossy
        assert_gt!(norm_100, 0); // positive field length → positive norm
    }

    #[test]
    fn mem_size_is_struct_size() {
        use mem_dbg::{MemSize, SizeFlags};
        let consumer = NormsConsumer::new();
        assert_eq!(
            consumer.mem_size(SizeFlags::CAPACITY),
            mem::size_of::<NormsConsumer>()
        );
    }
}
