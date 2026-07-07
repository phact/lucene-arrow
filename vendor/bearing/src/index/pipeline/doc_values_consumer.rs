// SPDX-License-Identifier: Apache-2.0

//! [`FieldConsumer`] that buffers doc values data and writes `.dvm`, `.dvd`.

use std::collections::HashMap;
use std::fmt;
use std::io;

use crate::analysis::Token;
use crate::codecs::lucene90::doc_values::{
    self, BinaryDocValue, DocValuesAccumulator, DocValuesFieldData, NumericDocValue,
    SortedDocValue, SortedNumericDocValue, SortedSetDocValue,
};
use crate::codecs::lucene90::doc_values_producer::BufferedDocValuesProducer;
use crate::document::DocValuesType;
use crate::document::IndexOptions;
use crate::index::FieldInfo;
use crate::index::field::{DocValue, Field};
use crate::index::field_infos::PointDimensionConfig;
use crate::index::pipeline::consumer::{FieldConsumer, TokenInterest};
use crate::index::pipeline::segment_accumulator::SegmentAccumulator;
use crate::index::pipeline::segment_context::SegmentContext;

/// Per-field state for accumulating doc values during indexing.
#[derive(mem_dbg::MemSize)]
struct PerFieldState {
    field_name: String,
    accumulator: DocValuesAccumulator,
}

impl fmt::Debug for PerFieldState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PerFieldState")
            .field("field_name", &self.field_name)
            .finish_non_exhaustive()
    }
}

/// Buffers doc values per field and flushes them as Lucene90 doc values files.
#[derive(Debug, Default, mem_dbg::MemSize)]
pub struct DocValuesConsumer {
    /// field_id → per-field accumulation state
    fields: HashMap<u32, PerFieldState>,
    current_doc_id: i32,
}

impl DocValuesConsumer {
    /// Creates a new consumer.
    pub fn new() -> Self {
        Self {
            fields: HashMap::new(),
            current_doc_id: 0,
        }
    }
}

impl FieldConsumer for DocValuesConsumer {
    fn start_document(&mut self, doc_id: i32) -> io::Result<()> {
        self.current_doc_id = doc_id;
        Ok(())
    }

    fn start_field(
        &mut self,
        field_id: u32,
        field: &Field,
        _accumulator: &mut SegmentAccumulator,
    ) -> io::Result<TokenInterest> {
        let dv_type = field.field_type().doc_values_type();
        if dv_type == DocValuesType::None {
            return Ok(TokenInterest::NoTokens);
        }

        let doc_id = self.current_doc_id;

        let state = self.fields.entry(field_id).or_insert_with(|| {
            let accumulator = match dv_type {
                DocValuesType::Numeric => DocValuesAccumulator::Numeric(Vec::new()),
                DocValuesType::Binary => DocValuesAccumulator::Binary(Vec::new()),
                DocValuesType::Sorted => DocValuesAccumulator::Sorted(Vec::new()),
                DocValuesType::SortedSet => DocValuesAccumulator::SortedSet(Vec::new()),
                DocValuesType::SortedNumeric => DocValuesAccumulator::SortedNumeric(Vec::new()),
                DocValuesType::None => unreachable!(),
            };
            PerFieldState {
                field_name: field.name().to_string(),
                accumulator,
            }
        });

        // TODO: These clones could be eliminated by making FieldConsumer methods
        // take `&mut Field`, allowing us to take ownership of DocValue data via
        // Option::take instead of cloning. Only this consumer reads doc_value().
        if let Some(dv) = field.field_type().doc_value() {
            match (&mut state.accumulator, dv) {
                (DocValuesAccumulator::Numeric(vals), DocValue::Numeric(v)) => {
                    vals.push(NumericDocValue { doc_id, value: *v });
                }
                (DocValuesAccumulator::Binary(vals), DocValue::Binary(b)) => {
                    vals.push(BinaryDocValue {
                        doc_id,
                        value: b.clone(),
                    });
                }
                (DocValuesAccumulator::Sorted(vals), DocValue::Sorted(b)) => {
                    vals.push(SortedDocValue {
                        doc_id,
                        value: b.clone(),
                    });
                }
                (DocValuesAccumulator::SortedSet(vals), DocValue::SortedSet(terms)) => {
                    vals.push(SortedSetDocValue {
                        doc_id,
                        values: terms.clone(),
                    });
                }
                (DocValuesAccumulator::SortedNumeric(vals), DocValue::SortedNumeric(numbers)) => {
                    vals.push(SortedNumericDocValue {
                        doc_id,
                        values: numbers.clone(),
                    });
                }
                _ => {
                    return Err(io::Error::other(format!(
                        "doc values type mismatch for field '{}'",
                        field.name()
                    )));
                }
            }
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
        _doc_id: i32,
        _accumulator: &mut SegmentAccumulator,
        _context: &SegmentContext,
    ) -> io::Result<()> {
        Ok(())
    }

    fn flush(
        &mut self,
        context: &SegmentContext,
        _accumulator: &SegmentAccumulator,
    ) -> io::Result<Vec<String>> {
        if self.fields.is_empty() {
            return Ok(vec![]);
        }

        let mut fields: Vec<DocValuesFieldData> = self
            .fields
            .drain()
            .map(|(field_id, state)| {
                let dv_type = match &state.accumulator {
                    DocValuesAccumulator::Numeric(_) => DocValuesType::Numeric,
                    DocValuesAccumulator::Binary(_) => DocValuesType::Binary,
                    DocValuesAccumulator::Sorted(_) => DocValuesType::Sorted,
                    DocValuesAccumulator::SortedSet(_) => DocValuesType::SortedSet,
                    DocValuesAccumulator::SortedNumeric(_) => DocValuesType::SortedNumeric,
                };
                DocValuesFieldData {
                    name: state.field_name,
                    number: field_id,
                    doc_values_type: dv_type,
                    doc_values: state.accumulator,
                }
            })
            .collect();
        fields.sort_by_key(|f| f.number);

        let producer = BufferedDocValuesProducer::new(&fields);

        // Build FieldInfo objects for each doc values field, sorted by field number
        let mut field_infos: Vec<FieldInfo> = fields
            .iter()
            .map(|f| {
                FieldInfo::new(
                    f.name.clone(),
                    f.number,
                    false,
                    true, // omit_norms
                    IndexOptions::None,
                    f.doc_values_type,
                    PointDimensionConfig::default(),
                )
            })
            .collect();
        field_infos.sort_by_key(|f| f.number());
        let field_info_refs: Vec<&FieldInfo> = field_infos.iter().collect();

        let num_docs = self.current_doc_id + 1;

        // Per-field doc values use suffix "Lucene90_0" matching Java's PerFieldDocValuesFormat
        let segment_suffix = "Lucene90_0";
        doc_values::write(
            &*context.directory,
            &context.segment_name,
            segment_suffix,
            &context.segment_id,
            &field_info_refs,
            &producer,
            num_docs,
        )
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::index::field::{
        binary_dv, numeric_dv, sorted_dv, sorted_numeric_dv, sorted_set_dv, stored,
    };
    use crate::store::MemoryDirectory;

    fn test_context() -> SegmentContext {
        SegmentContext {
            directory: MemoryDirectory::create(),
            segment_name: "_0".to_string(),
            segment_id: [0u8; 16],
        }
    }

    #[test]
    fn no_dv_fields_produces_no_files() {
        let context = test_context();
        let mut consumer = DocValuesConsumer::new();
        let mut acc = SegmentAccumulator::new();

        consumer.start_document(0).unwrap();
        let field = stored("title").string("hello");
        consumer.start_field(0, &field, &mut acc).unwrap();
        consumer.finish_field(0, &field, &mut acc).unwrap();
        consumer.finish_document(0, &mut acc, &context).unwrap();

        let names = consumer.flush(&context, &acc).unwrap();
        assert!(names.is_empty());
    }

    #[test]
    fn numeric_dv_produces_two_files() {
        let context = test_context();
        let mut consumer = DocValuesConsumer::new();
        let mut acc = SegmentAccumulator::new();

        consumer.start_document(0).unwrap();
        let field = numeric_dv("count").value(42);
        consumer.start_field(0, &field, &mut acc).unwrap();
        consumer.finish_field(0, &field, &mut acc).unwrap();
        consumer.finish_document(0, &mut acc, &context).unwrap();

        let names = consumer.flush(&context, &acc).unwrap();
        assert_eq!(names.len(), 2);
        assert!(names[0].ends_with(".dvm"));
        assert!(names[1].ends_with(".dvd"));
    }

    #[test]
    fn binary_dv_produces_two_files() {
        let context = test_context();
        let mut consumer = DocValuesConsumer::new();
        let mut acc = SegmentAccumulator::new();

        consumer.start_document(0).unwrap();
        let field = binary_dv("payload").value(vec![1, 2, 3]);
        consumer.start_field(0, &field, &mut acc).unwrap();
        consumer.finish_field(0, &field, &mut acc).unwrap();
        consumer.finish_document(0, &mut acc, &context).unwrap();

        let names = consumer.flush(&context, &acc).unwrap();
        assert_eq!(names.len(), 2);
    }

    #[test]
    fn sorted_dv_produces_two_files() {
        let context = test_context();
        let mut consumer = DocValuesConsumer::new();
        let mut acc = SegmentAccumulator::new();

        consumer.start_document(0).unwrap();
        let field = sorted_dv("category").value(b"sports".to_vec());
        consumer.start_field(0, &field, &mut acc).unwrap();
        consumer.finish_field(0, &field, &mut acc).unwrap();
        consumer.finish_document(0, &mut acc, &context).unwrap();

        let names = consumer.flush(&context, &acc).unwrap();
        assert_eq!(names.len(), 2);
    }

    #[test]
    fn sorted_set_dv_produces_two_files() {
        let context = test_context();
        let mut consumer = DocValuesConsumer::new();
        let mut acc = SegmentAccumulator::new();

        consumer.start_document(0).unwrap();
        let field = sorted_set_dv("tags").value(vec![b"a".to_vec(), b"b".to_vec()]);
        consumer.start_field(0, &field, &mut acc).unwrap();
        consumer.finish_field(0, &field, &mut acc).unwrap();
        consumer.finish_document(0, &mut acc, &context).unwrap();

        let names = consumer.flush(&context, &acc).unwrap();
        assert_eq!(names.len(), 2);
    }

    #[test]
    fn sorted_numeric_dv_produces_two_files() {
        let context = test_context();
        let mut consumer = DocValuesConsumer::new();
        let mut acc = SegmentAccumulator::new();

        consumer.start_document(0).unwrap();
        let field = sorted_numeric_dv("sizes").value(vec![10, 20, 30]);
        consumer.start_field(0, &field, &mut acc).unwrap();
        consumer.finish_field(0, &field, &mut acc).unwrap();
        consumer.finish_document(0, &mut acc, &context).unwrap();

        let names = consumer.flush(&context, &acc).unwrap();
        assert_eq!(names.len(), 2);
    }

    #[test]
    fn multiple_docs_accumulates() {
        let context = test_context();
        let mut consumer = DocValuesConsumer::new();
        let mut acc = SegmentAccumulator::new();

        for doc_id in 0..5 {
            consumer.start_document(doc_id).unwrap();
            let field = numeric_dv("count").value(doc_id as i64 * 10);
            consumer.start_field(0, &field, &mut acc).unwrap();
            consumer.finish_field(0, &field, &mut acc).unwrap();
            consumer
                .finish_document(doc_id, &mut acc, &context)
                .unwrap();
        }

        let names = consumer.flush(&context, &acc).unwrap();
        assert_eq!(names.len(), 2);

        let guard = &*context.directory;
        for name in &names {
            let data = guard.read_file(name).unwrap();
            assert!(!data.is_empty());
        }
    }

    #[test]
    fn multiple_dv_fields() {
        let context = test_context();
        let mut consumer = DocValuesConsumer::new();
        let mut acc = SegmentAccumulator::new();

        consumer.start_document(0).unwrap();

        let f1 = numeric_dv("count").value(42);
        consumer.start_field(0, &f1, &mut acc).unwrap();
        consumer.finish_field(0, &f1, &mut acc).unwrap();

        let f2 = binary_dv("payload").value(vec![1, 2, 3]);
        consumer.start_field(1, &f2, &mut acc).unwrap();
        consumer.finish_field(1, &f2, &mut acc).unwrap();

        consumer.finish_document(0, &mut acc, &context).unwrap();

        let names = consumer.flush(&context, &acc).unwrap();
        assert_eq!(names.len(), 2);
    }

    #[test]
    fn non_dv_field_is_ignored() {
        let context = test_context();
        let mut consumer = DocValuesConsumer::new();
        let mut acc = SegmentAccumulator::new();

        consumer.start_document(0).unwrap();
        let field = stored("title").string("hello");
        let interest = consumer.start_field(0, &field, &mut acc).unwrap();
        assert_eq!(interest, TokenInterest::NoTokens);
        consumer.finish_field(0, &field, &mut acc).unwrap();
        consumer.finish_document(0, &mut acc, &context).unwrap();

        let names = consumer.flush(&context, &acc).unwrap();
        assert!(names.is_empty());
    }

    #[test]
    fn file_names_use_per_field_suffix() {
        let context = test_context();
        let mut consumer = DocValuesConsumer::new();
        let mut acc = SegmentAccumulator::new();

        consumer.start_document(0).unwrap();
        let field = numeric_dv("count").value(42);
        consumer.start_field(0, &field, &mut acc).unwrap();
        consumer.finish_field(0, &field, &mut acc).unwrap();
        consumer.finish_document(0, &mut acc, &context).unwrap();

        let names = consumer.flush(&context, &acc).unwrap();
        assert_eq!(names[0], "_0_Lucene90_0.dvm");
        assert_eq!(names[1], "_0_Lucene90_0.dvd");
    }

    #[test]
    fn mem_size_empty_is_small() {
        use mem_dbg::{MemSize, SizeFlags};
        let consumer = DocValuesConsumer::new();
        assert_lt!(consumer.mem_size(SizeFlags::CAPACITY), 200);
    }

    #[test]
    fn mem_size_grows_with_dv_fields() {
        use mem_dbg::{MemSize, SizeFlags};
        let context = test_context();
        let mut consumer = DocValuesConsumer::new();
        let mut acc = SegmentAccumulator::new();

        for doc_id in 0..10 {
            consumer.start_document(doc_id).unwrap();
            let field = numeric_dv("count").value(doc_id as i64);
            consumer.start_field(0, &field, &mut acc).unwrap();
            consumer.finish_field(0, &field, &mut acc).unwrap();
            consumer
                .finish_document(doc_id, &mut acc, &context)
                .unwrap();
        }

        assert_gt!(consumer.mem_size(SizeFlags::CAPACITY), 0);
    }
}
