// SPDX-License-Identifier: Apache-2.0

//! [`FieldConsumer`] that buffers point values and writes `.kdd`, `.kdi`, `.kdm`.

use std::collections::HashMap;
use std::fmt;
use std::io;

use crate::analysis::Token;
use crate::codecs::lucene90::points::{self, PointsFieldData};
use crate::codecs::lucene90::points_reader::BufferedPointsProducer;
use crate::document::{DocValuesType, IndexOptions};
use crate::index::FieldInfo;
use crate::index::field::{Field, PointsValue};
use crate::index::field_infos::PointDimensionConfig;
use crate::index::pipeline::consumer::{FieldConsumer, TokenInterest};
use crate::index::pipeline::segment_accumulator::SegmentAccumulator;
use crate::index::pipeline::segment_context::SegmentContext;

/// Per-field state for accumulating point values during indexing.
#[derive(mem_dbg::MemSize)]
struct PerFieldState {
    field_name: String,
    field_number: u32,
    dimension_count: u32,
    index_dimension_count: u32,
    bytes_per_dim: u32,
    points: Vec<(i32, Vec<u8>)>,
}

impl fmt::Debug for PerFieldState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PerFieldState")
            .field("field_name", &self.field_name)
            .field("point_count", &self.points.len())
            .finish()
    }
}

/// Buffers point values per field and flushes them as BKD tree files.
#[derive(Debug, Default, mem_dbg::MemSize)]
pub struct PointsConsumer {
    /// field_id → per-field accumulation state
    fields: HashMap<u32, PerFieldState>,
    current_doc_id: i32,
}

impl PointsConsumer {
    /// Creates a new consumer.
    pub fn new() -> Self {
        Self {
            fields: HashMap::new(),
            current_doc_id: 0,
        }
    }
}

/// Extracts dimension metadata from a `PointsValue`.
fn dimensions(pv: &PointsValue) -> (u32, u32, u32) {
    match pv {
        PointsValue::Single {
            bytes_per_dim,
            encoded,
        } => {
            let dims = (encoded.len() / bytes_per_dim) as u32;
            (dims, dims, *bytes_per_dim as u32)
        }
        PointsValue::Range {
            dims,
            bytes_per_dim,
            ..
        } => {
            let d = (*dims * 2) as u32;
            (d, d, *bytes_per_dim as u32)
        }
    }
}

impl FieldConsumer for PointsConsumer {
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
        let pv = match field.field_type().points() {
            Some(pv) => pv,
            None => return Ok(TokenInterest::NoTokens),
        };

        let (dim_count, idx_dim_count, bpd) = dimensions(pv);
        let encoded = match pv {
            PointsValue::Single { encoded, .. } => encoded.clone(),
            PointsValue::Range { encoded, .. } => encoded.clone(),
        };

        let state = self
            .fields
            .entry(field_id)
            .or_insert_with(|| PerFieldState {
                field_name: field.name().to_string(),
                field_number: field_id,
                dimension_count: dim_count,
                index_dimension_count: idx_dim_count,
                bytes_per_dim: bpd,
                points: Vec::new(),
            });

        state.points.push((self.current_doc_id, encoded));

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

        let mut fields_data: Vec<PointsFieldData> = self
            .fields
            .values()
            .filter(|s| !s.points.is_empty())
            .map(|s| PointsFieldData {
                field_name: s.field_name.clone(),
                field_number: s.field_number,
                dimension_count: s.dimension_count,
                index_dimension_count: s.index_dimension_count,
                bytes_per_dim: s.bytes_per_dim,
                points: s.points.clone(),
            })
            .collect();

        if fields_data.is_empty() {
            return Ok(vec![]);
        }

        // Sort by field number for deterministic output
        fields_data.sort_by_key(|f| f.field_number);

        let producer = BufferedPointsProducer::new(&fields_data);

        let mut field_infos: Vec<FieldInfo> = fields_data
            .iter()
            .map(|f| {
                FieldInfo::new(
                    f.field_name.clone(),
                    f.field_number,
                    false,
                    true,
                    IndexOptions::None,
                    DocValuesType::None,
                    PointDimensionConfig {
                        dimension_count: f.dimension_count,
                        index_dimension_count: f.index_dimension_count,
                        num_bytes: f.bytes_per_dim,
                    },
                )
            })
            .collect();
        field_infos.sort_by_key(|f| f.number());
        let field_info_refs: Vec<&FieldInfo> = field_infos.iter().collect();

        points::write(
            &*context.directory,
            &context.segment_name,
            "",
            &context.segment_id,
            &field_info_refs,
            &producer,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::field::{int_field, lat_lon, long_field};
    use crate::store::MemoryDirectory;
    use assertables::*;

    fn test_context() -> SegmentContext {
        SegmentContext {
            directory: MemoryDirectory::create(),
            segment_name: "_0".to_string(),
            segment_id: [0u8; 16],
        }
    }

    #[test]
    fn no_point_fields_produces_no_files() {
        let context = test_context();
        let mut consumer = PointsConsumer::new();
        let acc = SegmentAccumulator::new();

        let names = consumer.flush(&context, &acc).unwrap();
        assert_is_empty!(&names);
    }

    #[test]
    fn int_field_produces_point_files() {
        let context = test_context();
        let mut consumer = PointsConsumer::new();
        let mut acc = SegmentAccumulator::new();

        for doc_id in 0..3 {
            let field = int_field("count").value(doc_id * 10);
            consumer.start_document(doc_id).unwrap();
            consumer.start_field(0, &field, &mut acc).unwrap();
            consumer.finish_field(0, &field, &mut acc).unwrap();
            consumer
                .finish_document(doc_id, &mut acc, &context)
                .unwrap();
        }

        let names = consumer.flush(&context, &acc).unwrap();
        assert_eq!(names.len(), 3);
        assert!(names.iter().any(|n| n.ends_with(".kdd")));
        assert!(names.iter().any(|n| n.ends_with(".kdi")));
        assert!(names.iter().any(|n| n.ends_with(".kdm")));
    }

    #[test]
    fn lat_lon_field_produces_point_files() {
        let context = test_context();
        let mut consumer = PointsConsumer::new();
        let mut acc = SegmentAccumulator::new();

        let field = lat_lon("location").value(40.7128, -74.0060);
        consumer.start_document(0).unwrap();
        consumer.start_field(0, &field, &mut acc).unwrap();
        consumer.finish_field(0, &field, &mut acc).unwrap();
        consumer.finish_document(0, &mut acc, &context).unwrap();

        let names = consumer.flush(&context, &acc).unwrap();
        assert_eq!(names.len(), 3);
    }

    #[test]
    fn dimensions_single_1d() {
        let pv = PointsValue::Single {
            bytes_per_dim: 4,
            encoded: vec![0; 4],
        };
        assert_eq!(dimensions(&pv), (1, 1, 4));
    }

    #[test]
    fn dimensions_single_2d() {
        let pv = PointsValue::Single {
            bytes_per_dim: 4,
            encoded: vec![0; 8],
        };
        assert_eq!(dimensions(&pv), (2, 2, 4));
    }

    #[test]
    fn dimensions_range() {
        let pv = PointsValue::Range {
            dims: 2,
            bytes_per_dim: 8,
            encoded: vec![0; 32],
        };
        assert_eq!(dimensions(&pv), (4, 4, 8));
    }

    #[test]
    fn multiple_point_fields() {
        let context = test_context();
        let mut consumer = PointsConsumer::new();
        let mut acc = SegmentAccumulator::new();

        consumer.start_document(0).unwrap();

        let f1 = int_field("count").value(42);
        consumer.start_field(0, &f1, &mut acc).unwrap();
        consumer.finish_field(0, &f1, &mut acc).unwrap();

        let f2 = long_field("timestamp").value(1234567890);
        consumer.start_field(1, &f2, &mut acc).unwrap();
        consumer.finish_field(1, &f2, &mut acc).unwrap();

        consumer.finish_document(0, &mut acc, &context).unwrap();

        let names = consumer.flush(&context, &acc).unwrap();
        // One set of .kdd/.kdi/.kdm for all point fields in the segment
        assert_eq!(names.len(), 3);
    }

    #[test]
    fn mem_size_empty_is_small() {
        use mem_dbg::{MemSize, SizeFlags};
        let consumer = PointsConsumer::new();
        assert_lt!(consumer.mem_size(SizeFlags::CAPACITY), 200);
    }

    #[test]
    fn mem_size_grows_with_point_fields() {
        use mem_dbg::{MemSize, SizeFlags};
        let context = test_context();
        let mut consumer = PointsConsumer::new();
        let mut acc = SegmentAccumulator::new();

        for doc_id in 0..10 {
            consumer.start_document(doc_id).unwrap();
            let field = int_field("size").value(doc_id * 100);
            consumer.start_field(0, &field, &mut acc).unwrap();
            consumer.finish_field(0, &field, &mut acc).unwrap();
            consumer
                .finish_document(doc_id, &mut acc, &context)
                .unwrap();
        }

        assert_gt!(consumer.mem_size(SizeFlags::CAPACITY), 0);
    }
}
