// SPDX-License-Identifier: Apache-2.0

//! `.dvm` metadata → [`DecodePlan`]s (the CPU plan stage, SPEC §5, §6).
//!
//! Walks every field entry in a `.dvm` file (parse order per Java
//! `Lucene90DocValuesProducer.readFields`, cross-checked against Bearing's
//! producer). NUMERIC fields — and SORTED_NUMERIC fields that degrade to
//! single-valued NUMERIC (SPEC §7.2) — become plans; other types are
//! structurally skipped and reported, so a `.dvm` containing them still
//! parses cleanly. P3 extends this walker to SORTED/SORTED_SET.

use arrow_schema::DataType;

use crate::consts;
use crate::monotonic;
use crate::terms::{TERMS_DICT_BLOCK_SIZE, TermsDictPlan};
use lucene_arrow_core::cursor::{Cursor, read_index_header, verify_footer};
use lucene_arrow_core::plan::{BlockDecode, Coverage, DecodePlan, DisiPlan, FieldId, PLAN_VERSION};
use lucene_arrow_core::{Error, Result};

/// Doc-values type of a field, as recorded in `.fnm`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DvKind {
    Numeric,
    Binary,
    Sorted,
    SortedNumeric,
    SortedSet,
}

/// What the planner needs to know about each doc-values field (from the
/// codec crate's `.fnm` inventory).
#[derive(Debug, Clone)]
pub struct DvField {
    pub number: i32,
    pub name: String,
    pub kind: DvKind,
    /// Lucene 10 doc-values skip index present (`DocValuesSkipIndexType !=
    /// NONE`). Extra metadata precedes the entry when set.
    pub has_skip_index: bool,
}

/// A DirectMonotonic address sequence in `.dvd` (List offsets, ord
/// streams): `count` monotone values.
#[derive(Debug, Clone)]
pub struct AddressesPlan {
    pub offset: u64,
    pub len: u64,
    pub block_shift: u32,
    pub meta: Vec<monotonic::MetaBlock>,
    pub count: u64,
}

impl AddressesPlan {
    /// Bulk-decode the whole sequence from full `.dvd` bytes.
    pub fn decode(&self, dvd: &[u8]) -> Result<Vec<i64>> {
        let end = self.offset + self.len;
        if end as usize > dvd.len() {
            return Err(Error::corrupt("addresses region beyond data file"));
        }
        monotonic::decode(&self.meta, &dvd[self.offset as usize..], self.count, self.block_shift)
    }
}

/// A SORTED / SORTED_SET column: ordinal stream + terms dictionary
/// (SPEC §7.2 → `Dictionary<Int32, Utf8>` / `List<Dictionary<…>>`).
#[derive(Debug)]
pub struct SortedPlan {
    /// Ordinal decode plan. Single-valued: one ord per doc-with-value.
    /// Multi-valued: the concatenated ord stream (`addresses` slices it).
    pub ords: DecodePlan,
    /// `Some` iff multi-valued: `num_docs_with_field + 1` offsets into the
    /// ord stream.
    pub addresses: Option<AddressesPlan>,
    pub terms: TermsDictPlan,
    /// True when declared SORTED_SET in `.fnm` (even if single-valued).
    pub set: bool,
}

/// A multi-valued SORTED_NUMERIC column → `List<Int64>` (SPEC §7.2).
#[derive(Debug)]
pub struct MultiNumericPlan {
    /// Concatenated value stream (coverage says which docs have entries).
    pub values: DecodePlan,
    /// `num_docs_with_field + 1` offsets into the value stream.
    pub addresses: AddressesPlan,
}

/// A BINARY column → `Binary` (SPEC §7.2).
#[derive(Debug)]
pub struct BinaryPlan {
    pub column: FieldId,
    /// Concatenated value bytes in `.dvd`.
    pub data_offset: u64,
    pub data_len: u64,
    pub coverage: Coverage,
    pub num_docs_with_field: u32,
    pub min_length: u32,
    pub max_length: u32,
    /// `None` ⇒ fixed-length values (`min_length` bytes each); otherwise
    /// `num_docs_with_field + 1` start offsets relative to `data_offset`.
    pub addresses: Option<AddressesPlan>,
}

/// Planner output: plans by shape, plus what was skipped and why.
#[derive(Debug)]
pub struct DocValuesPlans {
    /// NUMERIC + single-valued SORTED_NUMERIC (degraded per SPEC §7.2).
    pub plans: Vec<DecodePlan>,
    pub sorted: Vec<SortedPlan>,
    pub multi_numeric: Vec<MultiNumericPlan>,
    pub binary: Vec<BinaryPlan>,
    pub skipped: Vec<(FieldId, &'static str)>,
}

/// Parse a whole `.dvm` file (header and footer included) into decode
/// plans. `dvd` is needed only to walk multi-block value headers; for
/// single-block fields it is untouched (metadata-only planning).
pub fn plan_doc_values(
    dvm: &[u8],
    dvd: &[u8],
    fields: &[DvField],
    max_doc: u32,
    dvd_file: &str,
) -> Result<DocValuesPlans> {
    let header = read_index_header(dvm, consts::META_CODEC, consts::VERSION, consts::VERSION)?;
    verify_footer(dvm)?;

    let mut c = Cursor::at(dvm, header.length);
    let mut plans = Vec::new();
    let mut sorted = Vec::new();
    let mut multi_numeric = Vec::new();
    let mut binary = Vec::new();
    let skipped = Vec::new(); // every DV type now plans; kept for API stability

    loop {
        let field_number = c.le_i32()?;
        if field_number == -1 {
            break;
        }
        let field = fields.iter().find(|f| f.number == field_number).ok_or_else(|| {
            Error::corrupt(format!(".dvm references unknown field number {field_number}"))
        })?;
        let field_id = FieldId::new(field.number, field.name.clone());
        let type_byte = c.u8()?;

        // Skip-index metadata precedes type-specific metadata.
        if field.has_skip_index {
            c.skip(8 + 8 + 8 + 8 + 4 + 4)?;
        }

        match type_byte {
            consts::TYPE_NUMERIC => {
                if field.kind != DvKind::Numeric {
                    return Err(Error::corrupt(format!(
                        "field {} is {:?} in .fnm but NUMERIC in .dvm",
                        field.name, field.kind
                    )));
                }
                plans.push(numeric_entry_to_plan(&mut c, dvd, field_id, max_doc, dvd_file, false)?);
            }
            consts::TYPE_SORTED_NUMERIC => {
                let entry = read_numeric_entry(&mut c)?;
                let num_docs_with_field = c.le_i32()? as i64;
                if entry.num_values > num_docs_with_field {
                    let addresses = read_addresses(&mut c, num_docs_with_field as u64 + 1)?;
                    let mut values =
                        numeric_entry_into_plan(entry, dvd, field_id, max_doc, dvd_file, true)?;
                    fix_multi_coverage(&mut values, num_docs_with_field as u64);
                    multi_numeric.push(MultiNumericPlan { values, addresses });
                } else {
                    // Single-valued: identical payload to NUMERIC (SPEC §7.2
                    // "degrades to NUMERIC with lucene.multi=false").
                    plans.push(numeric_entry_into_plan(entry, dvd, field_id, max_doc, dvd_file, true)?);
                }
            }
            consts::TYPE_BINARY => {
                binary.push(read_binary_entry(&mut c, field_id, max_doc)?);
            }
            consts::TYPE_SORTED => {
                let entry = read_numeric_entry(&mut c)?; // ordinals
                let ords = numeric_entry_into_plan(entry, dvd, field_id, max_doc, dvd_file, false)?;
                let terms = read_terms_dict(&mut c)?;
                sorted.push(SortedPlan { ords, addresses: None, terms, set: false });
            }
            consts::TYPE_SORTED_SET => {
                let multi = c.u8()?;
                let entry = read_numeric_entry(&mut c)?;
                let (addresses, num_docs_with_field) = if multi != 0 {
                    let ndwf = c.le_i32()? as i64;
                    (Some(read_addresses(&mut c, ndwf as u64 + 1)?), Some(ndwf as u64))
                } else {
                    (None, None)
                };
                let mut ords = numeric_entry_into_plan(entry, dvd, field_id, max_doc, dvd_file, false)?;
                if let Some(ndwf) = num_docs_with_field {
                    fix_multi_coverage(&mut ords, ndwf);
                }
                let terms = read_terms_dict(&mut c)?;
                sorted.push(SortedPlan { ords, addresses, terms, set: true });
            }
            other => {
                return Err(Error::corrupt(format!("invalid doc values type byte {other}")));
            }
        }
    }

    Ok(DocValuesPlans { plans, sorted, multi_numeric, binary, skipped })
}

/// Multi-valued columns share the numeric entry layout, but the DISI set
/// tracks *docs with the field*, not values — patch the coverage cost so
/// bulk DISI decoding stops at the right cardinality.
fn fix_multi_coverage(plan: &mut DecodePlan, num_docs_with_field: u64) {
    if let Coverage::Sparse { disi, .. } = &mut plan.coverage {
        disi.num_values = num_docs_with_field;
    }
}

/// Everything the NUMERIC `.dvm` entry records (Java `NumericEntry`).
#[derive(Debug)]
struct NumericEntry {
    docs_with_field_offset: i64,
    docs_with_field_length: i64,
    jump_table_entry_count: i16,
    dense_rank_power: u8,
    num_values: i64,
    table: Option<Vec<i64>>,
    bits_per_value: u8,
    min: i64,
    gcd: i64,
    values_offset: i64,
    values_length: i64,
    #[allow(dead_code)] // random-access jump table; the planner walks blocks sequentially
    jump_table_offset: i64,
}

fn read_numeric_entry(c: &mut Cursor<'_>) -> Result<NumericEntry> {
    let docs_with_field_offset = c.le_i64()?;
    let docs_with_field_length = c.le_i64()?;
    let jump_table_entry_count = c.le_i16()?;
    let dense_rank_power = c.u8()?;
    let num_values = c.le_i64()?;
    let table_size = c.le_i32()?;
    let table = if table_size > 0 {
        if table_size > 256 {
            return Err(Error::corrupt(format!("numeric table size {table_size} > 256")));
        }
        let mut t = Vec::with_capacity(table_size as usize);
        for _ in 0..table_size {
            t.push(c.le_i64()?);
        }
        Some(t)
    } else if table_size == -1 || table_size == consts::MULTI_BLOCK_TABLE_SIZE {
        None
    } else {
        return Err(Error::corrupt(format!("invalid numeric table size {table_size}")));
    };
    let bits_per_value = c.u8()?;
    let min = c.le_i64()?;
    let gcd = c.le_i64()?;
    let values_offset = c.le_i64()?;
    let values_length = c.le_i64()?;
    let jump_table_offset = c.le_i64()?;
    Ok(NumericEntry {
        docs_with_field_offset,
        docs_with_field_length,
        jump_table_entry_count,
        dense_rank_power,
        num_values,
        table,
        bits_per_value,
        min,
        gcd,
        values_offset,
        values_length,
        jump_table_offset,
    })
}

fn numeric_entry_to_plan(
    c: &mut Cursor<'_>,
    dvd: &[u8],
    field_id: FieldId,
    max_doc: u32,
    dvd_file: &str,
    single_valued_multi: bool,
) -> Result<DecodePlan> {
    let entry = read_numeric_entry(c)?;
    numeric_entry_into_plan(entry, dvd, field_id, max_doc, dvd_file, single_valued_multi)
}

fn numeric_entry_into_plan(
    entry: NumericEntry,
    dvd: &[u8],
    field_id: FieldId,
    max_doc: u32,
    dvd_file: &str,
    _single_valued_multi: bool,
) -> Result<DecodePlan> {
    let num_values = entry.num_values as u64;

    let coverage = match entry.docs_with_field_offset {
        -2 => Coverage::Empty { num_docs: max_doc },
        -1 => Coverage::Dense { num_docs: max_doc },
        off if off >= 0 => Coverage::Sparse {
            num_docs: max_doc,
            disi: DisiPlan {
                offset: off as u64,
                len: entry.docs_with_field_length as u64,
                jump_table_entry_count: entry.jump_table_entry_count,
                dense_rank_power: entry.dense_rank_power,
                num_values,
            },
        },
        off => return Err(Error::corrupt(format!("invalid docsWithFieldOffset {off}"))),
    };

    let mut blocks = Vec::new();
    if num_values > 0 {
        match entry.bits_per_value {
            0 => {
                // Constant: every value is `min`.
                blocks.push(BlockDecode::DeltaPacked {
                    offset: entry.values_offset as u64,
                    len: 0,
                    bit_width: 0,
                    base: entry.min,
                    values: num_values,
                });
            }
            consts::MULTI_BLOCK_BPV => {
                // Multi-block: walk 16384-value block headers in .dvd.
                let mut d = Cursor::at(dvd, entry.values_offset as usize);
                let mut remaining = num_values;
                while remaining > 0 {
                    let n = remaining.min(consts::NUMERIC_BLOCK_SIZE as u64);
                    let bpv = d.u8()?;
                    if bpv == 0 {
                        let value = d.le_i64()?;
                        blocks.push(BlockDecode::DeltaPacked {
                            offset: d.pos() as u64,
                            len: 0,
                            bit_width: 0,
                            base: value,
                            values: n,
                        });
                    } else {
                        let block_min = d.le_i64()?;
                        let buffer_size = d.le_i32()?;
                        if buffer_size < 0 {
                            return Err(Error::corrupt("negative block buffer size"));
                        }
                        blocks.push(BlockDecode::GcdPacked {
                            offset: d.pos() as u64,
                            len: buffer_size as u64,
                            bit_width: bpv,
                            base: block_min,
                            gcd: entry.gcd,
                            values: n,
                        });
                        d.skip(buffer_size as usize)?;
                    }
                    remaining -= n;
                }
            }
            bpv => {
                let offset = entry.values_offset as u64;
                let len = entry.values_length as u64;
                if let Some(table) = entry.table {
                    blocks.push(BlockDecode::Table {
                        offset,
                        len,
                        bit_width: bpv,
                        table,
                        values: num_values,
                    });
                } else if entry.gcd != 1 {
                    blocks.push(BlockDecode::GcdPacked {
                        offset,
                        len,
                        bit_width: bpv,
                        base: entry.min,
                        gcd: entry.gcd,
                        values: num_values,
                    });
                } else if entry.min != 0 {
                    blocks.push(BlockDecode::DeltaPacked {
                        offset,
                        len,
                        bit_width: bpv,
                        base: entry.min,
                        values: num_values,
                    });
                } else {
                    blocks.push(BlockDecode::Direct { offset, len, bit_width: bpv, values: num_values });
                }
            }
        }
    }

    Ok(DecodePlan {
        plan_version: PLAN_VERSION,
        column: field_id,
        file: dvd_file.to_string(),
        arrow_type: DataType::Int64,
        blocks,
        coverage,
        num_values,
    })
}

// --- Structural skippers (keep the cursor in sync past unplanned types) ----



fn read_binary_entry(c: &mut Cursor<'_>, column: FieldId, max_doc: u32) -> Result<BinaryPlan> {
    let data_offset = c.le_i64()?;
    let data_len = c.le_i64()?;
    let docs_with_field_offset = c.le_i64()?;
    let docs_with_field_length = c.le_i64()?;
    let jump_table_entry_count = c.le_i16()?;
    let dense_rank_power = c.u8()?;
    let num_docs_with_field = c.le_i32()?;
    let min_length = c.le_i32()?;
    let max_length = c.le_i32()?;
    if data_offset < 0 || data_len < 0 || num_docs_with_field < 0 || min_length < 0 || max_length < min_length {
        return Err(Error::corrupt("invalid BINARY entry"));
    }
    let addresses = if max_length > min_length {
        Some(read_addresses(c, num_docs_with_field as u64 + 1)?)
    } else {
        None
    };
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
                num_values: num_docs_with_field as u64,
            },
        },
        off => return Err(Error::corrupt(format!("invalid docsWithFieldOffset {off}"))),
    };
    Ok(BinaryPlan {
        column,
        data_offset: data_offset as u64,
        data_len: data_len as u64,
        coverage,
        num_docs_with_field: num_docs_with_field as u32,
        min_length: min_length as u32,
        max_length: max_length as u32,
        addresses,
    })
}

fn skip_direct_monotonic_meta(c: &mut Cursor<'_>, num_values: i64, block_shift: u32) -> Result<()> {
    let block_size = 1u64 << block_shift;
    let num_blocks = (num_values.max(0) as u64).div_ceil(block_size);
    c.skip(num_blocks as usize * monotonic::META_BYTES_PER_BLOCK)
}


/// Capturing form of `skip_direct_monotonic_addresses`: layout is
/// `i64 offset, vint blockShift, DM meta blocks, i64 length`.
fn read_addresses(c: &mut Cursor<'_>, count: u64) -> Result<AddressesPlan> {
    let offset = c.le_i64()?;
    let block_shift = c.vint()? as u32;
    let meta = monotonic::read_meta(c, count, block_shift)?;
    let len = c.le_i64()?;
    if offset < 0 || len < 0 {
        return Err(Error::corrupt("negative addresses offset/length"));
    }
    Ok(AddressesPlan { offset: offset as u64, len: len as u64, block_shift, meta, count })
}

/// Capturing form of the terms-dict metadata walk (order per
/// `add_terms_dict`): the DirectMonotonic sequence has one value per
/// 64-term block. (Bearing's skip derives the count with the *DM* shift —
/// harmless below ~4.2M terms, wrong above; this uses the real block size.)
fn read_terms_dict(c: &mut Cursor<'_>) -> Result<TermsDictPlan> {
    let num_terms = c.vlong()?;
    if num_terms < 0 {
        return Err(Error::corrupt("negative term count"));
    }
    let num_blocks = (num_terms as u64).div_ceil(TERMS_DICT_BLOCK_SIZE);
    let address_block_shift = c.le_i32()? as u32;
    let address_meta = monotonic::read_meta(c, num_blocks, address_block_shift)?;
    let max_term_length = c.le_i32()?;
    let max_block_length = c.le_i32()?;
    let terms_offset = c.le_i64()?;
    let terms_len = c.le_i64()?;
    let addresses_offset = c.le_i64()?;
    let addresses_len = c.le_i64()?;
    if terms_offset < 0 || terms_len < 0 || addresses_offset < 0 || addresses_len < 0 || max_term_length < 0 {
        return Err(Error::corrupt("negative terms dict field"));
    }

    // Reverse index (binary-search support): parse-skip, unused for bulk
    // materialization. One DM value per 1024 terms, plus a final entry.
    let reverse_shift = c.le_i32()? as u32;
    let num_reverse_blocks = (num_terms as u64).div_ceil(1u64 << reverse_shift) as i64;
    skip_direct_monotonic_meta(c, num_reverse_blocks + 1, consts::DIRECT_MONOTONIC_BLOCK_SHIFT)?;
    c.skip(8 + 8 + 8 + 8)?; // reverse index off/len, reverse addresses off/len

    Ok(TermsDictPlan {
        num_terms: num_terms as u64,
        address_meta,
        address_block_shift,
        max_term_length: max_term_length as u32,
        max_block_length: max_block_length as u32,
        terms_offset: terms_offset as u64,
        terms_len: terms_len as u64,
        addresses_offset: addresses_offset as u64,
        addresses_len: addresses_len as u64,
    })
}
