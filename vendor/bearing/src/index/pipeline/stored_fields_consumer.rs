// SPDX-License-Identifier: Apache-2.0

//! [`FieldConsumer`] that streams stored field data to codec files per-document.
//!
//! Instead of buffering `StoredDoc` structs in memory, this consumer writes
//! stored fields to a `Lucene90StoredFieldsWriter` incrementally as
//! documents are indexed. Only a single document's fields are held in
//! memory at a time.

use std::fmt;
use std::io;

use crate::analysis::Token;
use crate::codecs::lucene90::stored_fields::{Lucene90StoredFieldsWriter, StoredFieldsWriter};
use crate::document::StoredValue;
use crate::index::field::Field;
use crate::index::pipeline::consumer::{FieldConsumer, TokenInterest};
use crate::index::pipeline::segment_accumulator::SegmentAccumulator;
use crate::index::pipeline::segment_context::SegmentContext;

/// Streams stored field values to codec files per-document.
///
/// The codec writer is lazily created on the first document with stored
/// fields. Gap documents (docs with no stored fields between real docs)
/// are filled at flush time to maintain document ID alignment.
pub struct StoredFieldsConsumer {
    writer: Option<Lucene90StoredFieldsWriter>,
    current_doc_fields: Vec<(u32, StoredValue)>,
    last_doc: i32,
}

impl fmt::Debug for StoredFieldsConsumer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StoredFieldsConsumer")
            .field("has_writer", &self.writer.is_some())
            .field("current_doc_fields", &self.current_doc_fields.len())
            .field("last_doc", &self.last_doc)
            .finish()
    }
}

impl mem_dbg::MemSize for StoredFieldsConsumer {
    fn mem_size_rec(
        &self,
        flags: mem_dbg::SizeFlags,
        refs: &mut mem_dbg::HashMap<usize, usize>,
    ) -> usize {
        self.current_doc_fields.mem_size_rec(flags, refs)
    }
}

impl Default for StoredFieldsConsumer {
    fn default() -> Self {
        Self::new()
    }
}

impl StoredFieldsConsumer {
    /// Creates a new consumer.
    pub fn new() -> Self {
        Self {
            writer: None,
            current_doc_fields: Vec::new(),
            last_doc: -1,
        }
    }

    /// Ensures the writer is created, lazily opening codec files.
    fn ensure_writer(&mut self, context: &SegmentContext) -> io::Result<()> {
        if self.writer.is_none() {
            self.writer = Some(Lucene90StoredFieldsWriter::new(
                &*context.directory,
                &context.segment_name,
                "",
                &context.segment_id,
            )?);
        }
        Ok(())
    }

    /// Fills gap documents between `last_doc` and `doc_id`.
    fn fill_gaps(&mut self, doc_id: i32) -> io::Result<()> {
        let writer = self.writer.as_mut().unwrap();
        while self.last_doc + 1 < doc_id {
            self.last_doc += 1;
            writer.start_document()?;
            writer.finish_document()?;
        }
        Ok(())
    }
}

impl FieldConsumer for StoredFieldsConsumer {
    fn start_document(&mut self, _doc_id: i32) -> io::Result<()> {
        self.current_doc_fields.clear();
        Ok(())
    }

    fn start_field(
        &mut self,
        field_id: u32,
        field: &Field,
        _accumulator: &mut SegmentAccumulator,
    ) -> io::Result<TokenInterest> {
        if let Some(sv) = field.field_type().stored() {
            self.current_doc_fields.push((field_id, sv.clone()));
        }
        Ok(TokenInterest::NoTokens)
    }

    fn add_token(
        &mut self,
        _field_id: u32,
        _field: &Field,
        _token: &Token<'_>,
        _accumulator: &mut SegmentAccumulator,
    ) -> io::Result<()> {
        Ok(())
    }

    fn finish_field(
        &mut self,
        _field_id: u32,
        _field: &Field,
        _accumulator: &mut SegmentAccumulator,
    ) -> io::Result<()> {
        Ok(())
    }

    fn finish_document(
        &mut self,
        doc_id: i32,
        _accumulator: &mut SegmentAccumulator,
        context: &SegmentContext,
    ) -> io::Result<()> {
        self.ensure_writer(context)?;
        self.fill_gaps(doc_id)?;
        self.last_doc = doc_id;

        let writer = self.writer.as_mut().unwrap();
        writer.start_document()?;
        for (field_number, value) in &self.current_doc_fields {
            writer.write_field(*field_number, value)?;
        }
        writer.finish_document()?;
        self.current_doc_fields.clear();
        Ok(())
    }

    fn flush(
        &mut self,
        context: &SegmentContext,
        accumulator: &SegmentAccumulator,
    ) -> io::Result<Vec<String>> {
        let num_docs = accumulator.doc_count();
        if num_docs == 0 {
            return Ok(vec![]);
        }

        self.ensure_writer(context)?;

        // Fill trailing gap documents
        while self.last_doc < num_docs - 1 {
            self.last_doc += 1;
            let writer = self.writer.as_mut().unwrap();
            writer.start_document()?;
            writer.finish_document()?;
        }

        let mut writer = self.writer.take().unwrap();
        writer.finish(num_docs)?;

        Ok(Lucene90StoredFieldsWriter::file_names(
            &context.segment_name,
            "",
        ))
    }
}

#[cfg(test)]
mod tests {

    use assertables::*;

    use super::*;
    use crate::index::field::{stored, text};
    use crate::store::MemoryDirectory;

    fn test_context() -> SegmentContext {
        SegmentContext {
            directory: MemoryDirectory::create(),
            segment_name: "_0".to_string(),
            segment_id: [0u8; 16],
        }
    }

    #[test]
    fn flush_produces_three_files() {
        let context = test_context();
        let mut consumer = StoredFieldsConsumer::new();
        let mut acc = SegmentAccumulator::new();

        consumer.start_document(0).unwrap();
        let field = stored("title").string("hello");
        consumer.start_field(0, &field, &mut acc).unwrap();
        consumer.finish_field(0, &field, &mut acc).unwrap();
        consumer.finish_document(0, &mut acc, &context).unwrap();
        acc.increment_doc_count();

        let names = consumer.flush(&context, &acc).unwrap();
        assert_len_eq_x!(&names, 3);
        assert_eq!(names[0], "_0.fdt");
        assert_eq!(names[1], "_0.fdx");
        assert_eq!(names[2], "_0.fdm");
    }

    #[test]
    fn non_stored_field_streams_empty_doc() {
        let context = test_context();
        let mut consumer = StoredFieldsConsumer::new();
        let mut acc = SegmentAccumulator::new();

        consumer.start_document(0).unwrap();
        let field = text("not_stored").value("invisible");
        consumer.start_field(0, &field, &mut acc).unwrap();
        consumer.finish_field(0, &field, &mut acc).unwrap();
        consumer.finish_document(0, &mut acc, &context).unwrap();
        acc.increment_doc_count();

        let names = consumer.flush(&context, &acc).unwrap();
        assert_len_eq_x!(&names, 3);
    }

    #[test]
    fn multiple_docs_multiple_fields() {
        let context = test_context();
        let mut consumer = StoredFieldsConsumer::new();
        let mut acc = SegmentAccumulator::new();

        for doc_id in 0..3 {
            consumer.start_document(doc_id).unwrap();

            let f1 = stored("title").string(format!("title {doc_id}"));
            consumer.start_field(0, &f1, &mut acc).unwrap();
            consumer.finish_field(0, &f1, &mut acc).unwrap();

            let f2 = stored("body").string(format!("body {doc_id}"));
            consumer.start_field(1, &f2, &mut acc).unwrap();
            consumer.finish_field(1, &f2, &mut acc).unwrap();

            consumer
                .finish_document(doc_id, &mut acc, &context)
                .unwrap();
            acc.increment_doc_count();
        }

        let names = consumer.flush(&context, &acc).unwrap();
        assert_len_eq_x!(&names, 3);

        // Verify files have content
        let guard = &*context.directory;
        for name in &names {
            let data = guard.read_file(name).unwrap();
            assert_not_empty!(data);
        }
    }

    #[test]
    fn flush_no_docs_returns_empty() {
        let context = test_context();
        let mut consumer = StoredFieldsConsumer::new();
        let acc = SegmentAccumulator::new();

        let names = consumer.flush(&context, &acc).unwrap();
        assert_is_empty!(&names);
    }

    #[test]
    fn mem_size_empty_is_small() {
        use mem_dbg::{MemSize, SizeFlags};
        let consumer = StoredFieldsConsumer::new();
        assert_lt!(consumer.mem_size(SizeFlags::CAPACITY), 200);
    }

    #[test]
    fn mem_size_bounded_after_streaming() {
        use mem_dbg::{MemSize, SizeFlags};
        let context = test_context();
        let mut consumer = StoredFieldsConsumer::new();
        let mut acc = SegmentAccumulator::new();

        for doc_id in 0..10 {
            consumer.start_document(doc_id).unwrap();
            let field = stored("title").string(format!("doc {doc_id}"));
            consumer.start_field(0, &field, &mut acc).unwrap();
            consumer.finish_field(0, &field, &mut acc).unwrap();
            consumer
                .finish_document(doc_id, &mut acc, &context)
                .unwrap();
            acc.increment_doc_count();
        }

        // Streaming consumer doesn't accumulate — MemSize should be small
        // (only current_doc_fields which was cleared after each doc)
        assert_lt!(consumer.mem_size(SizeFlags::CAPACITY), 200);
    }

    #[test]
    fn debug_format() {
        let consumer = StoredFieldsConsumer::new();
        let debug = format!("{consumer:?}");
        assert_contains!(debug, "StoredFieldsConsumer");
        assert_contains!(debug, "has_writer");
        assert_contains!(debug, "last_doc");
    }

    #[test]
    fn default_creates_new() {
        let consumer = StoredFieldsConsumer::default();
        assert_eq!(consumer.last_doc, -1);
    }

    #[test]
    fn fill_gaps_writes_empty_docs() {
        let context = test_context();
        let mut consumer = StoredFieldsConsumer::new();
        let mut acc = SegmentAccumulator::new();

        // Skip doc 0, write doc 2 — should fill gaps for docs 0 and 1
        consumer.start_document(2).unwrap();
        let field = stored("title").string("doc 2");
        consumer.start_field(0, &field, &mut acc).unwrap();
        consumer.finish_field(0, &field, &mut acc).unwrap();
        consumer.finish_document(2, &mut acc, &context).unwrap();
        acc.increment_doc_count();
        acc.increment_doc_count();
        acc.increment_doc_count();

        let names = consumer.flush(&context, &acc).unwrap();
        assert_len_eq_x!(&names, 3);
    }
}
