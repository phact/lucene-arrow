// SPDX-License-Identifier: Apache-2.0

//! `.vemf` metadata → vector [`DecodePlan`]s (`Raw` blocks — DMA, no
//! kernel). Parse order per `Lucene99FlatVectorsReader.readFields` /
//! `OrdToDocDISIReaderConfiguration.fromStoredMeta` (Lucene 10.3.2).
//!
//! [`DecodePlan`]: lucene_arrow_core::plan::DecodePlan

use std::sync::Arc;

use arrow_schema::{DataType, Field};

use crate::{Similarity, VectorEncoding, consts};
use lucene_arrow_core::cursor::{Cursor, read_index_header, verify_footer};
use lucene_arrow_core::plan::{BlockDecode, Coverage, DecodePlan, DisiPlan, FieldId, PLAN_VERSION};
use lucene_arrow_core::{Error, Result};

/// What the planner needs per vector field (from the `.fnm` inventory).
#[derive(Debug, Clone)]
pub struct VecField {
    pub number: i32,
    pub name: String,
}

/// A planned vector column: the core plan (one `Raw` block over the packed
/// vectors, ord order) plus the vector-specific facts the frame layer puts
/// in Arrow field metadata (SPEC §7.2).
#[derive(Debug)]
pub struct VectorPlan {
    pub plan: DecodePlan,
    pub encoding: VectorEncoding,
    pub similarity: Similarity,
    pub dim: u32,
    /// Vectors stored (== docs with the field).
    pub count: u64,
}

/// Parse a whole `.vemf` file into vector plans. The `.vec` data file is
/// not touched — `Raw` blocks are pure byte ranges.
pub fn plan_vectors(
    vemf: &[u8],
    fields: &[VecField],
    max_doc: u32,
    vec_file: &str,
) -> Result<Vec<VectorPlan>> {
    let header =
        read_index_header(vemf, consts::META_CODEC, consts::VERSION, consts::VERSION)?;
    verify_footer(vemf)?;

    let mut c = Cursor::at(vemf, header.length);
    let mut plans = Vec::new();

    loop {
        let field_number = c.le_i32()?;
        if field_number == -1 {
            break;
        }
        let field = fields.iter().find(|f| f.number == field_number).ok_or_else(|| {
            Error::corrupt(format!(".vemf references unknown field number {field_number}"))
        })?;

        let encoding = VectorEncoding::from_ordinal(c.le_i32()?)
            .ok_or_else(|| Error::corrupt("invalid vector encoding ordinal"))?;
        let similarity = Similarity::from_ordinal(c.le_i32()?)
            .ok_or_else(|| Error::corrupt("invalid vector similarity ordinal"))?;
        let data_offset = c.vlong()? as u64;
        let data_length = c.vlong()? as u64;
        let dim = c.vint()? as u32;
        let count = c.le_i32()?;
        if count < 0 {
            return Err(Error::corrupt("negative vector count"));
        }
        let count = count as u64;

        // OrdToDocDISIReaderConfiguration.fromStoredMeta
        let docs_with_field_offset = c.le_i64()?;
        let docs_with_field_length = c.le_i64()?;
        let jump_table_entry_count = c.le_i16()?;
        let dense_rank_power = c.u8()?;
        if docs_with_field_offset > -1 {
            // Sparse: ord→doc DirectMonotonic addresses. Redundant for full
            // scans (the DISI bitmap already orders docs by ord), so skip.
            c.skip(8)?; // addressesOffset
            let block_shift = c.vint()? as u32;
            let blocks = count.div_ceil(1u64 << block_shift);
            c.skip(blocks as usize * lucene_arrow_docvalues::monotonic::META_BYTES_PER_BLOCK)?;
            c.skip(8)?; // addressesLength
        }

        let expected_len = count * dim as u64 * encoding.width() as u64;
        if data_length != expected_len {
            return Err(Error::corrupt(format!(
                "vector data length {data_length} != count {count} × dim {dim} × width {}",
                encoding.width()
            )));
        }

        let coverage = match docs_with_field_offset {
            -2 => Coverage::Empty { num_docs: max_doc },
            -1 => Coverage::Dense { num_docs: max_doc },
            off if off >= 0 => Coverage::Sparse {
                num_docs: max_doc,
                disi: DisiPlan {
                    offset: off as u64,
                    len: docs_with_field_length as u64,
                    jump_table_entry_count,
                    dense_rank_power,
                    num_values: count,
                },
            },
            off => return Err(Error::corrupt(format!("invalid docsWithFieldOffset {off}"))),
        };

        let child = match encoding {
            VectorEncoding::Float32 => DataType::Float32,
            VectorEncoding::Byte => DataType::Int8,
        };
        let arrow_type = DataType::FixedSizeList(
            Arc::new(Field::new("item", child, false)),
            dim as i32,
        );

        plans.push(VectorPlan {
            plan: DecodePlan {
                plan_version: PLAN_VERSION,
                column: FieldId::new(field.number, field.name.clone()),
                file: vec_file.to_string(),
                arrow_type,
                blocks: if count == 0 {
                    Vec::new()
                } else {
                    vec![BlockDecode::Raw { offset: data_offset, len: data_length }]
                },
                coverage,
                num_values: count,
            },
            encoding,
            similarity,
            dim,
            count,
        });
    }

    Ok(plans)
}
