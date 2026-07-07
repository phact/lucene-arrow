// SPDX-License-Identifier: Apache-2.0
//! Doc values writer for numeric, binary, sorted, and sorted-set doc values.

use std::collections::{BTreeSet, HashMap};
use std::io;
use std::mem;

use log::debug;

use crate::codecs::lucene90::doc_values_producer::DocValuesProducer;
use crate::codecs::lucene90::indexed_disi;
use crate::codecs::packed_writers::{DirectMonotonicWriter, DirectWriter};
use crate::codecs::{codec_footers, codec_headers};
use crate::document::DocValuesType;
use crate::encoding::lz4::{self, FastHashTable};
use crate::encoding::packed::unsigned_bits_required;
use crate::encoding::write_encoding::WriteEncoding;
use crate::index::FieldInfo;
use crate::index::index_file_names;
use crate::search::doc_id_set_iterator::NO_MORE_DOCS;
use crate::store::memory::MemoryIndexOutput;
use crate::store::{DataOutput, Directory, IndexOutput, VecOutput};
use crate::util::string_helper;

/// A single numeric doc value entry.
#[derive(mem_dbg::MemSize)]
#[mem_size(flat)]
pub(crate) struct NumericDocValue {
    pub doc_id: i32,
    pub value: i64,
}

/// A single binary doc value entry.
#[derive(mem_dbg::MemSize)]
pub(crate) struct BinaryDocValue {
    pub doc_id: i32,
    pub value: Vec<u8>,
}

/// A single sorted doc value entry.
#[derive(mem_dbg::MemSize)]
pub(crate) struct SortedDocValue {
    pub doc_id: i32,
    pub value: Vec<u8>,
}

/// A single sorted-numeric doc value entry.
#[derive(mem_dbg::MemSize)]
pub(crate) struct SortedNumericDocValue {
    pub doc_id: i32,
    pub values: Vec<i64>,
}

/// A single sorted-set doc value entry.
#[derive(mem_dbg::MemSize)]
pub(crate) struct SortedSetDocValue {
    pub doc_id: i32,
    pub values: Vec<Vec<u8>>,
}

/// Per-field doc values accumulation state.
#[derive(mem_dbg::MemSize)]
pub(crate) enum DocValuesAccumulator {
    Numeric(Vec<NumericDocValue>),
    Binary(Vec<BinaryDocValue>),
    Sorted(Vec<SortedDocValue>),
    SortedNumeric(Vec<SortedNumericDocValue>),
    SortedSet(Vec<SortedSetDocValue>),
}

/// Per-field data passed to the doc values writer.
pub(crate) struct DocValuesFieldData {
    pub name: String,
    pub number: u32,
    pub doc_values_type: DocValuesType,
    pub doc_values: DocValuesAccumulator,
}

// File extensions
pub(crate) const DATA_EXTENSION: &str = "dvd";
pub(crate) const META_EXTENSION: &str = "dvm";

// Codec names and version
pub(crate) const DATA_CODEC: &str = "Lucene90DocValuesData";
pub(crate) const META_CODEC: &str = "Lucene90DocValuesMetadata";
pub(crate) const VERSION: i32 = 0;

// Doc values type bytes
pub(crate) const NUMERIC: u8 = 0;
pub(crate) const BINARY: u8 = 1;
pub(crate) const SORTED: u8 = 2;
pub(crate) const SORTED_SET: u8 = 3;
pub(crate) const SORTED_NUMERIC: u8 = 4;

// Terms dictionary constants
const TERMS_DICT_BLOCK_LZ4_SHIFT: usize = 6;
const TERMS_DICT_BLOCK_LZ4_SIZE: usize = 1 << TERMS_DICT_BLOCK_LZ4_SHIFT;
const TERMS_DICT_BLOCK_LZ4_MASK: usize = TERMS_DICT_BLOCK_LZ4_SIZE - 1;
const TERMS_DICT_REVERSE_INDEX_SHIFT: i32 = 10;
const TERMS_DICT_REVERSE_INDEX_SIZE: usize = 1 << TERMS_DICT_REVERSE_INDEX_SHIFT;
const TERMS_DICT_REVERSE_INDEX_MASK: usize = TERMS_DICT_REVERSE_INDEX_SIZE - 1;
pub(crate) const DIRECT_MONOTONIC_BLOCK_SHIFT: u32 = 16;

/// Writes doc values files (.dvm, .dvd) for a segment using a [`DocValuesProducer`].
///
/// `field_infos` must be sorted by field number.
pub(crate) fn write(
    directory: &dyn Directory,
    segment_name: &str,
    segment_suffix: &str,
    segment_id: &[u8; 16],
    field_infos: &[&FieldInfo],
    producer: &dyn DocValuesProducer,
    num_docs: i32,
) -> io::Result<Vec<String>> {
    let dvm_name =
        index_file_names::segment_file_name(segment_name, segment_suffix, META_EXTENSION);
    let dvd_name =
        index_file_names::segment_file_name(segment_name, segment_suffix, DATA_EXTENSION);

    let mut meta = directory.create_output(&dvm_name)?;
    let mut data = directory.create_output(&dvd_name)?;

    codec_headers::write_index_header(&mut *meta, META_CODEC, VERSION, segment_id, segment_suffix)?;
    codec_headers::write_index_header(&mut *data, DATA_CODEC, VERSION, segment_id, segment_suffix)?;

    for &field_info in field_infos {
        meta.write_le_int(field_info.number() as i32)?;

        match field_info.doc_values_type() {
            DocValuesType::Numeric => {
                meta.write_byte(NUMERIC)?;
                let vals = collect_numeric(producer, field_info)?;
                debug!(
                    "doc_values: field={:?} (#{}) -> NUMERIC, {} docs",
                    field_info.name(),
                    field_info.number(),
                    vals.len()
                );
                add_numeric_field(&mut *meta, &mut *data, &vals, num_docs)?;
            }
            DocValuesType::Binary => {
                meta.write_byte(BINARY)?;
                let vals = collect_binary(producer, field_info)?;
                debug!(
                    "doc_values: field={:?} (#{}) -> BINARY, {} docs",
                    field_info.name(),
                    field_info.number(),
                    vals.len()
                );
                add_binary_field(&mut *meta, &mut *data, &vals, num_docs)?;
            }
            DocValuesType::Sorted => {
                meta.write_byte(SORTED)?;
                debug!(
                    "doc_values: field={:?} (#{}) -> SORTED",
                    field_info.name(),
                    field_info.number(),
                );
                add_sorted_field(&mut *meta, &mut *data, producer, field_info, num_docs)?;
            }
            DocValuesType::SortedNumeric => {
                meta.write_byte(SORTED_NUMERIC)?;
                let vals = collect_sorted_numeric(producer, field_info)?;
                debug!(
                    "doc_values: field={:?} (#{}) -> SORTED_NUMERIC, {} docs",
                    field_info.name(),
                    field_info.number(),
                    vals.len()
                );
                add_sorted_numeric_field(&mut *meta, &mut *data, &vals, num_docs)?;
            }
            DocValuesType::SortedSet => {
                meta.write_byte(SORTED_SET)?;
                debug!(
                    "doc_values: field={:?} (#{}) -> SORTED_SET",
                    field_info.name(),
                    field_info.number(),
                );
                add_sorted_set_field(&mut *meta, &mut *data, producer, field_info, num_docs)?;
            }
            DocValuesType::None => continue,
        }
    }

    // EOF marker
    meta.write_le_int(-1)?;

    // Write footers
    codec_footers::write_footer(&mut *meta)?;
    codec_footers::write_footer(&mut *data)?;

    Ok(vec![dvm_name, dvd_name])
}

/// Collects numeric doc values from the producer into a Vec.
fn collect_numeric(
    producer: &dyn DocValuesProducer,
    field_info: &FieldInfo,
) -> io::Result<Vec<NumericDocValue>> {
    let mut iter = match producer.get_numeric(field_info)? {
        Some(iter) => iter,
        None => return Ok(vec![]),
    };
    let mut vals = Vec::new();
    loop {
        let doc = iter.next_doc()?;
        if doc == NO_MORE_DOCS {
            break;
        }
        vals.push(NumericDocValue {
            doc_id: doc,
            value: iter.long_value()?,
        });
    }
    Ok(vals)
}

/// Collects binary doc values from the producer into a Vec.
fn collect_binary(
    producer: &dyn DocValuesProducer,
    field_info: &FieldInfo,
) -> io::Result<Vec<BinaryDocValue>> {
    let mut iter = match producer.get_binary(field_info)? {
        Some(iter) => iter,
        None => return Ok(vec![]),
    };
    let mut vals = Vec::new();
    loop {
        let doc = iter.next_doc()?;
        if doc == NO_MORE_DOCS {
            break;
        }
        vals.push(BinaryDocValue {
            doc_id: doc,
            value: iter.binary_value()?.to_vec(),
        });
    }
    Ok(vals)
}

/// Collects sorted-numeric doc values from the producer into a Vec.
fn collect_sorted_numeric(
    producer: &dyn DocValuesProducer,
    field_info: &FieldInfo,
) -> io::Result<Vec<SortedNumericDocValue>> {
    let mut iter = match producer.get_sorted_numeric(field_info)? {
        Some(iter) => iter,
        None => return Ok(vec![]),
    };
    let mut vals = Vec::new();
    loop {
        let doc = iter.next_doc()?;
        if doc == NO_MORE_DOCS {
            break;
        }
        let count = iter.doc_value_count();
        let mut values = Vec::with_capacity(count as usize);
        for _ in 0..count {
            values.push(iter.next_value()?);
        }
        vals.push(SortedNumericDocValue {
            doc_id: doc,
            values,
        });
    }
    Ok(vals)
}

/// Adds a NUMERIC field (single value per doc, no numDocsWithField).
fn add_numeric_field(
    meta: &mut dyn DataOutput,
    data: &mut dyn IndexOutput,
    vals: &[NumericDocValue],
    num_docs: i32,
) -> io::Result<()> {
    // No skip index for MVP

    let doc_ids: Vec<i32> = vals.iter().map(|entry| entry.doc_id).collect();
    let all_values: Vec<i64> = vals.iter().map(|entry| entry.value).collect();

    // Unlike SORTED_NUMERIC, NUMERIC does NOT write numDocsWithField after writeValues
    write_values(
        meta,
        data,
        &all_values,
        &doc_ids,
        vals.len() as i32,
        num_docs,
        false,
    )?;

    Ok(())
}

/// Adds a BINARY field.
fn add_binary_field(
    mut meta: &mut dyn IndexOutput,
    data: &mut dyn IndexOutput,
    vals: &[BinaryDocValue],
    num_docs: i32,
) -> io::Result<()> {
    // Write data offset
    let start = data.file_pointer() as i64;
    meta.write_le_long(start)?;

    // Write concatenated binary values to data
    let mut min_length = i32::MAX;
    let mut max_length = 0i32;
    for entry in vals {
        let len = entry.value.len() as i32;
        min_length = min_length.min(len);
        max_length = max_length.max(len);
        data.write_all(&entry.value)?;
    }

    let num_docs_with_field = vals.len() as i32;
    if num_docs_with_field == 0 {
        min_length = 0;
    }

    // Write data length
    meta.write_le_long(data.file_pointer() as i64 - start)?;

    // Docs-with-field indicator
    if num_docs_with_field == 0 {
        meta.write_le_long(-2)?; // EMPTY
        meta.write_le_long(0)?;
        meta.write_le_short(-1)?;
        meta.write_byte(0xFF)?;
    } else if num_docs_with_field == num_docs {
        meta.write_le_long(-1)?; // ALL
        meta.write_le_long(0)?;
        meta.write_le_short(-1)?;
        meta.write_byte(0xFF)?;
    } else {
        // SPARSE — write IndexedDISI bitset
        let doc_ids: Vec<i32> = vals.iter().map(|entry| entry.doc_id).collect();
        let disi_offset = data.file_pointer() as i64;
        meta.write_le_long(disi_offset)?;
        let jump_table_entry_count = indexed_disi::write_bit_set(&doc_ids, num_docs, &mut *data)?;
        meta.write_le_long(data.file_pointer() as i64 - disi_offset)?;
        meta.write_le_short(jump_table_entry_count)?;
        meta.write_byte(indexed_disi::DEFAULT_DENSE_RANK_POWER as u8)?;
    }

    meta.write_le_int(num_docs_with_field)?;
    meta.write_le_int(min_length)?;
    meta.write_le_int(max_length)?;

    // Variable-length encoding: write addresses
    if max_length > min_length {
        let addresses_start = data.file_pointer() as i64;
        meta.write_le_long(addresses_start)?;
        meta.write_vint(DIRECT_MONOTONIC_BLOCK_SHIFT as i32)?;

        let mut address_buffer = MemoryIndexOutput::new("temp_binary_addr".to_string());
        let mut dm_writer = DirectMonotonicWriter::new(DIRECT_MONOTONIC_BLOCK_SHIFT);

        let mut cumulative: i64 = 0;
        for entry in vals {
            dm_writer.add(cumulative);
            cumulative += entry.value.len() as i64;
        }
        dm_writer.add(cumulative); // final entry for total length

        dm_writer.finish(meta, &mut address_buffer)?;
        data.write_all(address_buffer.bytes())?;

        meta.write_le_long(data.file_pointer() as i64 - addresses_start)?;
    }

    Ok(())
}

/// Adds a SORTED field (single ordinal per doc + terms dictionary).
///
/// Reads ordinals and term lookups from the [`DocValuesProducer`].
fn add_sorted_field(
    meta: &mut dyn IndexOutput,
    data: &mut dyn IndexOutput,
    producer: &dyn DocValuesProducer,
    field_info: &FieldInfo,
    num_docs: i32,
) -> io::Result<()> {
    // No skip index for MVP

    let mut sorted_dv = match producer.get_sorted(field_info)? {
        Some(dv) => dv,
        None => {
            // Empty: write zero-length values block + empty terms dict
            write_values(meta, data, &[], &[], 0, num_docs, true)?;
            add_terms_dict(meta, data, &[])?;
            return Ok(());
        }
    };

    // Build terms dictionary from ordinal lookups
    let value_count = sorted_dv.value_count();
    let mut sorted_terms: Vec<Vec<u8>> = Vec::with_capacity(value_count as usize);
    for ord in 0..value_count {
        sorted_terms.push(sorted_dv.lookup_ord(ord)?.to_vec());
    }

    // Collect per-doc ordinals
    let mut doc_ids = Vec::new();
    let mut ordinals = Vec::new();
    loop {
        let doc = sorted_dv.next_doc()?;
        if doc == NO_MORE_DOCS {
            break;
        }
        doc_ids.push(doc);
        ordinals.push(sorted_dv.ord_value()? as i64);
    }

    // writeValues for ordinals (ords=true) — no multiValued byte, no numDocsWithField
    write_values(
        meta,
        data,
        &ordinals,
        &doc_ids,
        doc_ids.len() as i32,
        num_docs,
        true,
    )?;

    // Write terms dictionary
    let term_refs: Vec<&[u8]> = sorted_terms.iter().map(|t| t.as_slice()).collect();
    add_terms_dict(meta, data, &term_refs)?;

    Ok(())
}

/// Adds a SORTED_NUMERIC field.
fn add_sorted_numeric_field(
    mut meta: &mut dyn IndexOutput,
    data: &mut dyn IndexOutput,
    vals: &[SortedNumericDocValue],
    num_docs: i32,
) -> io::Result<()> {
    // No skip index for MVP

    // Sort values within each doc (Java's SortedNumericDocValues contract)
    let mut sorted_vals: Vec<SortedNumericDocValue> = vals
        .iter()
        .map(|entry| SortedNumericDocValue {
            doc_id: entry.doc_id,
            values: entry.values.clone(),
        })
        .collect();
    for entry in sorted_vals.iter_mut() {
        entry.values.sort();
    }

    // Collect all values in doc order (flattened)
    let doc_ids: Vec<i32> = sorted_vals.iter().map(|entry| entry.doc_id).collect();
    let all_values: Vec<i64> = sorted_vals
        .iter()
        .flat_map(|entry| entry.values.iter().copied())
        .collect();

    let num_docs_with_value = sorted_vals.len() as i32;
    let (num_docs_with_field, num_values) = write_values(
        meta,
        data,
        &all_values,
        &doc_ids,
        num_docs_with_value,
        num_docs,
        false,
    )?;

    meta.write_le_int(num_docs_with_field)?;

    // If multi-valued (numValues > numDocsWithField), write address table
    if num_values > num_docs_with_field as i64 {
        let addresses_start = data.file_pointer() as i64;
        meta.write_le_long(addresses_start)?;
        meta.write_vint(DIRECT_MONOTONIC_BLOCK_SHIFT as i32)?;

        let mut address_buffer = MemoryIndexOutput::new("temp_sn_addr".to_string());
        let mut dm_writer = DirectMonotonicWriter::new(DIRECT_MONOTONIC_BLOCK_SHIFT);

        let mut cumulative: i64 = 0;
        for entry in &sorted_vals {
            dm_writer.add(cumulative);
            cumulative += entry.values.len() as i64;
        }
        dm_writer.add(cumulative);

        dm_writer.finish(meta, &mut address_buffer)?;
        data.write_all(address_buffer.bytes())?;

        meta.write_le_long(data.file_pointer() as i64 - addresses_start)?;
    }

    Ok(())
}

/// Adds a SORTED_SET field.
///
/// Reads ordinals and term lookups from the [`DocValuesProducer`].
/// Single-valued: uses the SORTED reader path (writeByte(0), writeValues, termDict).
/// Multi-valued: uses the SORTED_NUMERIC reader path (writeByte(1), writeValues,
/// numDocsWithField, address table, termDict).
fn add_sorted_set_field(
    mut meta: &mut dyn IndexOutput,
    data: &mut dyn IndexOutput,
    producer: &dyn DocValuesProducer,
    field_info: &FieldInfo,
    num_docs: i32,
) -> io::Result<()> {
    // First pass: detect single-valued and collect per-doc ordinals
    let mut sorted_set_dv = match producer.get_sorted_set(field_info)? {
        Some(dv) => dv,
        None => {
            // Empty: single-valued path with zero entries
            meta.write_byte(0)?;
            write_values(meta, data, &[], &[], 0, num_docs, true)?;
            add_terms_dict(meta, data, &[])?;
            return Ok(());
        }
    };

    // Build terms dictionary from ordinal lookups
    let value_count = sorted_set_dv.value_count();
    let mut sorted_terms: Vec<Vec<u8>> = Vec::with_capacity(value_count as usize);
    for ord in 0..value_count {
        sorted_terms.push(sorted_set_dv.lookup_ord(ord)?.to_vec());
    }
    let term_refs: Vec<&[u8]> = sorted_terms.iter().map(|t| t.as_slice()).collect();

    // Collect per-doc ordinal lists
    let mut is_single_valued = true;
    let mut ord_vals: Vec<(i32, Vec<i64>)> = Vec::new();
    loop {
        let doc = sorted_set_dv.next_doc()?;
        if doc == NO_MORE_DOCS {
            break;
        }
        let count = sorted_set_dv.doc_value_count();
        if count > 1 {
            is_single_valued = false;
        }
        let mut ords = Vec::with_capacity(count as usize);
        for _ in 0..count {
            ords.push(sorted_set_dv.next_ord()?);
        }
        ord_vals.push((doc, ords));
    }

    if is_single_valued {
        // Single-valued path: writeByte(0), writeValues (ords), termDict
        // Java's readSorted() does NOT read numDocsWithField.
        meta.write_byte(0)?;

        let doc_ids: Vec<i32> = ord_vals.iter().map(|(doc_id, _)| *doc_id).collect();
        let ordinals: Vec<i64> = ord_vals.iter().map(|(_, ords)| ords[0]).collect();

        write_values(
            meta,
            data,
            &ordinals,
            &doc_ids,
            doc_ids.len() as i32,
            num_docs,
            true,
        )?;
        add_terms_dict(meta, data, &term_refs)?;
    } else {
        // Multi-valued path: doAddSortedNumericField(ords=true) writes multiValued
        // byte (1), writeValues, numDocsWithField, address table. Then termDict.
        meta.write_byte(1)?;

        // Flatten ordinals
        let doc_ids: Vec<i32> = ord_vals.iter().map(|(doc_id, _)| *doc_id).collect();
        let all_ordinals: Vec<i64> = ord_vals
            .iter()
            .flat_map(|(_doc_id, ords)| ords.iter().copied())
            .collect();

        let num_docs_with_value = ord_vals.len() as i32;
        let (num_docs_with_field, num_values) = write_values(
            meta,
            data,
            &all_ordinals,
            &doc_ids,
            num_docs_with_value,
            num_docs,
            true,
        )?;

        meta.write_le_int(num_docs_with_field)?;

        // If multi-valued, write address table
        if num_values > num_docs_with_field as i64 {
            let addresses_start = data.file_pointer() as i64;
            meta.write_le_long(addresses_start)?;
            meta.write_vint(DIRECT_MONOTONIC_BLOCK_SHIFT as i32)?;

            let mut address_buffer = MemoryIndexOutput::new("temp_ss_addr".to_string());
            let mut dm_writer = DirectMonotonicWriter::new(DIRECT_MONOTONIC_BLOCK_SHIFT);

            let mut cumulative: i64 = 0;
            for (_doc_id, ords) in &ord_vals {
                dm_writer.add(cumulative);
                cumulative += ords.len() as i64;
            }
            dm_writer.add(cumulative);

            dm_writer.finish(meta, &mut address_buffer)?;
            data.write_all(address_buffer.bytes())?;

            meta.write_le_long(data.file_pointer() as i64 - addresses_start)?;
        }

        add_terms_dict(meta, data, &term_refs)?;
    }

    Ok(())
}

/// Core numeric encoding for doc values.
/// Returns (numDocsWithField, numValues).
fn write_values(
    meta: &mut dyn DataOutput,
    data: &mut dyn IndexOutput,
    all_values: &[i64],
    doc_ids: &[i32],
    num_docs_with_value: i32,
    max_doc: i32,
    ords: bool,
) -> io::Result<(i32, i64)> {
    let num_values = all_values.len() as i64;

    // Docs-with-field indicator
    if num_docs_with_value == 0 {
        // EMPTY
        meta.write_le_long(-2)?;
        meta.write_le_long(0)?;
        meta.write_le_short(-1)?;
        meta.write_byte(0xFF)?; // denseRankPower = -1 as byte
    } else if num_docs_with_value == max_doc {
        // ALL
        meta.write_le_long(-1)?;
        meta.write_le_long(0)?;
        meta.write_le_short(-1)?;
        meta.write_byte(0xFF)?;
    } else {
        // SPARSE — write IndexedDISI bitset
        let offset = data.file_pointer() as i64;
        meta.write_le_long(offset)?;
        let jump_table_entry_count = indexed_disi::write_bit_set(doc_ids, max_doc, &mut *data)?;
        meta.write_le_long(data.file_pointer() as i64 - offset)?;
        meta.write_le_short(jump_table_entry_count)?;
        meta.write_byte(indexed_disi::DEFAULT_DENSE_RANK_POWER as u8)?;
    }

    meta.write_le_long(num_values)?;

    if num_values == 0 {
        // No values: write minimal metadata
        meta.write_le_int(-1)?; // tablesize
        meta.write_byte(0)?; // numBitsPerValue
        meta.write_le_long(0)?; // min
        meta.write_le_long(0)?; // gcd
        meta.write_le_long(data.file_pointer() as i64)?; // valueOffset
        meta.write_le_long(0)?; // valuesLength
        meta.write_le_long(-1)?; // jumpTableOffset
        return Ok((0, 0));
    }

    // Compute min, max, GCD, unique values
    let first_value = all_values[0];
    let mut min = all_values[0];
    let mut max = all_values[0];
    let mut gcd: i64 = 0;
    let mut unique_values: Option<BTreeSet<i64>> = if ords { None } else { Some(BTreeSet::new()) };

    for &v in all_values {
        min = min.min(v);
        max = max.max(v);

        if gcd != 1 {
            if !(i64::MIN / 2..=i64::MAX / 2).contains(&v) {
                gcd = 1;
            } else {
                gcd = gcd_compute(gcd, v - first_value);
            }
        }

        if let Some(ref mut set) = unique_values {
            set.insert(v);
            if set.len() > 256 {
                unique_values = None;
            }
        }
    }

    // Ordinals validation
    if ords && num_values > 0 {
        assert!(min == 0, "min value for ordinals should be 0, got {}", min);
        if max != 0 {
            assert!(gcd == 1, "GCD on ordinals should be 1, got {}", gcd);
        }
    }

    let num_bits_per_value: u32;
    let mut encode_table: Option<HashMap<i64, i64>> = None;

    if min >= max {
        // Constant value
        num_bits_per_value = 0;
        meta.write_le_int(-1)?; // tablesize
    } else if let Some(ref uv) = unique_values {
        if uv.len() > 1 {
            let table_bpv = unsigned_bits_required(uv.len() as i64 - 1);
            let delta_bpv = unsigned_bits_required((max - min) / gcd);
            if table_bpv < delta_bpv {
                // Table encoding is more compact
                num_bits_per_value = table_bpv;
                let sorted_unique: Vec<i64> = uv.iter().copied().collect();
                meta.write_le_int(sorted_unique.len() as i32)?; // tablesize
                for &v in &sorted_unique {
                    meta.write_le_long(v)?;
                }
                let mut enc = HashMap::new();
                for (i, &v) in sorted_unique.iter().enumerate() {
                    enc.insert(v, i as i64);
                }
                encode_table = Some(enc);
                min = 0;
                gcd = 1;
            } else {
                // Delta encoding
                num_bits_per_value = delta_bpv;
                meta.write_le_int(-1)?; // tablesize
                // Optimization: if gcd==1 && min>0 && bits(max)==bits(max-min) → set min=0
                if gcd == 1
                    && min > 0
                    && unsigned_bits_required(max) == unsigned_bits_required(max - min)
                {
                    min = 0;
                }
            }
        } else {
            // Single unique value but min < max shouldn't happen, treat as delta
            num_bits_per_value = unsigned_bits_required((max - min) / gcd);
            meta.write_le_int(-1)?;
        }
    } else {
        // No table (too many unique values or ords mode)
        num_bits_per_value = unsigned_bits_required((max - min) / gcd);
        // No blocks for MVP (doBlocks=false)
        meta.write_le_int(-1)?; // tablesize
        // Optimization: if gcd==1 && min>0 && bits(max)==bits(max-min) → set min=0
        if gcd == 1 && min > 0 && unsigned_bits_required(max) == unsigned_bits_required(max - min) {
            min = 0;
        }
    }

    meta.write_byte(num_bits_per_value as u8)?;
    meta.write_le_long(min)?;
    meta.write_le_long(gcd)?;
    let start_offset = data.file_pointer() as i64;
    meta.write_le_long(start_offset)?; // valueOffset

    // Write data
    if num_bits_per_value > 0 {
        let mut writer = DirectWriter::new(num_bits_per_value);
        if let Some(ref enc) = encode_table {
            for &v in all_values {
                writer.add(enc[&v]);
            }
        } else {
            for &v in all_values {
                writer.add((v - min) / gcd);
            }
        }
        writer.finish(data)?;
    }

    let values_length = data.file_pointer() as i64 - start_offset;
    meta.write_le_long(values_length)?; // valuesLength
    meta.write_le_long(-1)?; // jumpTableOffset (always -1 for single-block)

    Ok((num_docs_with_value, num_values))
}

/// GCD computation matching Java's MathUtil.gcd
fn gcd_compute(a: i64, b: i64) -> i64 {
    // Use absolute values (wrapping_neg handles i64::MIN correctly)
    let mut a = if a < 0 { a.wrapping_neg() } else { a };
    let mut b = if b < 0 { b.wrapping_neg() } else { b };
    if a == 0 {
        return b;
    }
    if b == 0 {
        return a;
    }
    // Binary GCD algorithm
    let shift = (a | b).trailing_zeros();
    a >>= a.trailing_zeros();
    loop {
        b >>= b.trailing_zeros();
        if a == b {
            break;
        }
        if (a as u64) > (b as u64) || a == i64::MIN {
            mem::swap(&mut a, &mut b);
        }
        if a == 1 {
            break;
        }
        b -= a;
    }
    a << shift
}

/// Writes the terms dictionary for SORTED/SORTED_SET doc values.
fn add_terms_dict(
    mut meta: &mut dyn IndexOutput,
    mut data: &mut dyn IndexOutput,
    sorted_terms: &[&[u8]],
) -> io::Result<()> {
    let size = sorted_terms.len() as i64;
    meta.write_vlong(size)?;

    let block_mask = TERMS_DICT_BLOCK_LZ4_MASK;

    meta.write_le_int(DIRECT_MONOTONIC_BLOCK_SHIFT as i32)?;

    // Create DirectMonotonicWriter for block addresses
    let mut address_buffer = MemoryIndexOutput::new("temp_addr".to_string());
    let mut dm_writer = DirectMonotonicWriter::new(DIRECT_MONOTONIC_BLOCK_SHIFT);

    let mut previous: &[u8] = &[];
    let mut max_length: i32 = 0;
    let mut max_block_length: i32 = 0;
    let start = data.file_pointer() as i64;

    // Buffer for suffix data within a block (terms dict buffer)
    let mut suffix_buffer: Vec<u8> = Vec::new();
    // We need to track the first term of the block as the LZ4 dictionary
    let mut dict_bytes: Vec<u8> = Vec::new();
    // Reusable hash table across blocks, matching Java's FastCompressionHashTable reuse
    let mut lz4_ht = FastHashTable::new();

    for (ord, term) in sorted_terms.iter().enumerate() {
        if (ord & block_mask) == 0 {
            if ord != 0 {
                // Flush the previous block
                let uncompressed_length =
                    compress_and_write_terms_block(data, &dict_bytes, &suffix_buffer, &mut lz4_ht)?;
                max_block_length = max_block_length.max(uncompressed_length as i32);
                suffix_buffer.clear();
            }

            dm_writer.add((data.file_pointer() as i64) - start);

            // Write first term of block directly to data
            data.write_vint(term.len() as i32)?;
            data.write_all(term)?;

            // Save first term as dictionary for LZ4 compression
            dict_bytes = term.to_vec();
        } else {
            // Prefix-compress subsequent terms
            let prefix_length = string_helper::bytes_difference(previous, term);
            let suffix_length = term.len() - prefix_length;
            assert!(suffix_length > 0, "duplicate terms in sorted set");

            // Pack prefix/suffix into a byte
            let byte = (prefix_length.min(15) as u8)
                | ((suffix_length.saturating_sub(1).min(15) as u8) << 4);
            suffix_buffer.push(byte);

            if prefix_length >= 15 {
                VecOutput(&mut suffix_buffer).write_vint((prefix_length - 15) as i32)?;
            }
            if suffix_length >= 16 {
                VecOutput(&mut suffix_buffer).write_vint((suffix_length - 16) as i32)?;
            }

            suffix_buffer.extend_from_slice(&term[prefix_length..]);
        }

        max_length = max_length.max(term.len() as i32);
        previous = term;
    }

    // Flush last block if there's suffix data
    if !suffix_buffer.is_empty() {
        let uncompressed_length =
            compress_and_write_terms_block(data, &dict_bytes, &suffix_buffer, &mut lz4_ht)?;
        max_block_length = max_block_length.max(uncompressed_length as i32);
    }

    // Finish DirectMonotonicWriter: write to address_buffer, metadata to meta
    // Add final entry if we had blocks
    dm_writer.finish(meta, &mut address_buffer)?;

    meta.write_le_int(max_length)?;
    meta.write_le_int(max_block_length)?;
    meta.write_le_long(start)?;
    meta.write_le_long(data.file_pointer() as i64 - start)?;

    // Write address buffer to data
    let addr_start = data.file_pointer() as i64;
    data.write_all(address_buffer.bytes())?;
    meta.write_le_long(addr_start)?;
    meta.write_le_long(data.file_pointer() as i64 - addr_start)?;

    // Write reverse terms index
    write_terms_index(meta, data, sorted_terms)?;

    Ok(())
}

/// Compress suffix buffer with dict_bytes as LZ4 dictionary and write to data.
/// Returns the uncompressed length (of suffix data only).
fn compress_and_write_terms_block(
    mut data: &mut dyn DataOutput,
    dict_bytes: &[u8],
    suffix_buffer: &[u8],
    ht: &mut FastHashTable,
) -> io::Result<usize> {
    let uncompressed_length = suffix_buffer.len();
    data.write_vint(uncompressed_length as i32)?;

    // Build combined buffer: dict + suffix
    let mut combined = Vec::with_capacity(dict_bytes.len() + suffix_buffer.len());
    combined.extend_from_slice(dict_bytes);
    combined.extend_from_slice(suffix_buffer);

    let compressed = lz4::compress_with_dictionary_reuse(&combined, dict_bytes.len(), ht);
    data.write_all(&compressed)?;

    Ok(uncompressed_length)
}

/// Writes the reverse terms index for SORTED/SORTED_SET.
fn write_terms_index(
    meta: &mut dyn IndexOutput,
    data: &mut dyn IndexOutput,
    sorted_terms: &[&[u8]],
) -> io::Result<()> {
    meta.write_le_int(TERMS_DICT_REVERSE_INDEX_SHIFT)?;

    let start = data.file_pointer() as i64;

    let mut address_buffer = MemoryIndexOutput::new("temp_reverse_addr".to_string());
    let mut dm_writer = DirectMonotonicWriter::new(DIRECT_MONOTONIC_BLOCK_SHIFT);

    let mut previous: Option<&[u8]> = None;
    let mut offset: i64 = 0;

    for (ord, term) in sorted_terms.iter().enumerate() {
        if (ord & TERMS_DICT_REVERSE_INDEX_MASK) == 0 {
            dm_writer.add(offset);
            let sort_key_len = if ord == 0 {
                0 // no previous term: no bytes to write
            } else {
                string_helper::sort_key_length(previous.unwrap(), term)
            };
            offset += sort_key_len as i64;
            data.write_all(&term[..sort_key_len])?;
        }
        // Track previous for the term just before the next boundary
        if (ord & TERMS_DICT_REVERSE_INDEX_MASK) == TERMS_DICT_REVERSE_INDEX_MASK {
            previous = Some(term);
        }
    }

    // Final entry
    dm_writer.add(offset);
    dm_writer.finish(meta, &mut address_buffer)?;

    meta.write_le_long(start)?;
    meta.write_le_long(data.file_pointer() as i64 - start)?;

    let addr_start = data.file_pointer() as i64;
    data.write_all(address_buffer.bytes())?;
    meta.write_le_long(addr_start)?;
    meta.write_le_long(data.file_pointer() as i64 - addr_start)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codecs::codec_footers::FOOTER_LENGTH;
    use crate::codecs::codec_headers::index_header_length;
    use crate::codecs::lucene90::doc_values_producer::BufferedDocValuesProducer;
    use crate::document::{DocValuesType, IndexOptions};
    use crate::index::FieldInfo;
    use crate::index::field_infos::PointDimensionConfig;
    use crate::store::{MemoryDirectory, SharedDirectory};
    use crate::test_util::TestDataReader;
    use assertables::{assert_ge, assert_gt};

    fn bytes(s: &str) -> Vec<u8> {
        s.as_bytes().to_vec()
    }

    fn make_test_directory() -> SharedDirectory {
        MemoryDirectory::create()
    }

    /// Test helper: wraps fields in a BufferedDocValuesProducer and calls write().
    fn test_write(
        directory: &dyn Directory,
        segment_name: &str,
        segment_suffix: &str,
        segment_id: &[u8; 16],
        fields: &[DocValuesFieldData],
        num_docs: i32,
    ) -> io::Result<Vec<String>> {
        let producer = BufferedDocValuesProducer::new(fields);
        let field_infos: Vec<FieldInfo> = fields
            .iter()
            .map(|f| {
                FieldInfo::new(
                    f.name.clone(),
                    f.number,
                    false,
                    true,
                    IndexOptions::None,
                    f.doc_values_type,
                    PointDimensionConfig::default(),
                )
            })
            .collect();
        let fi_refs: Vec<&FieldInfo> = field_infos.iter().collect();
        write(
            directory,
            segment_name,
            segment_suffix,
            segment_id,
            &fi_refs,
            &producer,
            num_docs,
        )
    }

    fn nv(doc_id: i32, value: i64) -> NumericDocValue {
        NumericDocValue { doc_id, value }
    }

    fn bv(doc_id: i32, value: Vec<u8>) -> BinaryDocValue {
        BinaryDocValue { doc_id, value }
    }

    fn sv(doc_id: i32, value: Vec<u8>) -> SortedDocValue {
        SortedDocValue { doc_id, value }
    }

    fn snv(doc_id: i32, values: Vec<i64>) -> SortedNumericDocValue {
        SortedNumericDocValue { doc_id, values }
    }

    fn ssv(doc_id: i32, values: Vec<Vec<u8>>) -> SortedSetDocValue {
        SortedSetDocValue { doc_id, values }
    }

    fn make_field_data_numeric(
        name: &str,
        number: u32,
        values: Vec<NumericDocValue>,
    ) -> DocValuesFieldData {
        DocValuesFieldData {
            name: name.to_string(),
            number,
            doc_values_type: DocValuesType::Numeric,
            doc_values: DocValuesAccumulator::Numeric(values),
        }
    }

    fn make_field_data_binary(
        name: &str,
        number: u32,
        values: Vec<BinaryDocValue>,
    ) -> DocValuesFieldData {
        DocValuesFieldData {
            name: name.to_string(),
            number,
            doc_values_type: DocValuesType::Binary,
            doc_values: DocValuesAccumulator::Binary(values),
        }
    }

    fn make_field_data_sorted(
        name: &str,
        number: u32,
        values: Vec<SortedDocValue>,
    ) -> DocValuesFieldData {
        DocValuesFieldData {
            name: name.to_string(),
            number,
            doc_values_type: DocValuesType::Sorted,
            doc_values: DocValuesAccumulator::Sorted(values),
        }
    }

    fn make_field_data_sorted_numeric(
        name: &str,
        number: u32,
        values: Vec<SortedNumericDocValue>,
    ) -> DocValuesFieldData {
        DocValuesFieldData {
            name: name.to_string(),
            number,
            doc_values_type: DocValuesType::SortedNumeric,
            doc_values: DocValuesAccumulator::SortedNumeric(values),
        }
    }

    fn make_field_data_sorted_set(
        name: &str,
        number: u32,
        values: Vec<SortedSetDocValue>,
    ) -> DocValuesFieldData {
        DocValuesFieldData {
            name: name.to_string(),
            number,
            doc_values_type: DocValuesType::SortedSet,
            doc_values: DocValuesAccumulator::SortedSet(values),
        }
    }

    #[test]
    fn test_gcd_compute() {
        assert_eq!(gcd_compute(0, 5), 5);
        assert_eq!(gcd_compute(5, 0), 5);
        assert_eq!(gcd_compute(12, 8), 4);
        assert_eq!(gcd_compute(0, 0), 0);
        assert_eq!(gcd_compute(7, 7), 7);
        assert_eq!(gcd_compute(100, 75), 25);
        assert_eq!(gcd_compute(-12, 8), 4);
    }

    #[test]
    fn test_vec_output_vint() {
        let mut buf = Vec::new();
        VecOutput(&mut buf).write_vint(0).unwrap();
        assert_eq!(buf, vec![0]);

        let mut buf = Vec::new();
        VecOutput(&mut buf).write_vint(127).unwrap();
        assert_eq!(buf, vec![0x7F]);

        let mut buf = Vec::new();
        VecOutput(&mut buf).write_vint(128).unwrap();
        assert_eq!(buf, vec![0x80, 0x01]);
    }

    // Ported from org.apache.lucene.codecs.lucene90.TestLucene90DocValuesFormat
    #[test]
    fn test_sorted_numeric_constant() {
        // 3 docs all with value 42 → bpv=0, min=42
        let fields = vec![make_field_data_sorted_numeric(
            "modified",
            1,
            vec![snv(0, vec![42]), snv(1, vec![42]), snv(2, vec![42])],
        )];

        let segment_id = [0u8; 16];
        let directory = make_test_directory();
        let result = test_write(&directory, "_0", "Lucene90_0", &segment_id, &fields, 3).unwrap();

        assert_len_eq_x!(&result, 2);
        assert_eq!(result[0], "_0_Lucene90_0.dvm");
        assert_eq!(result[1], "_0_Lucene90_0.dvd");

        let dvm = directory.read_file(&result[0]).unwrap();

        // Verify codec magic
        assert_eq!(&dvm[0..4], &[0x3f, 0xd7, 0x6c, 0x17]);

        // Parse after header
        let meta_header_len = index_header_length(META_CODEC, "Lucene90_0");
        let entry = &dvm[meta_header_len..];

        // field_number = 1
        assert_eq!(&entry[0..4], &1i32.to_le_bytes());
        // type byte = SORTED_NUMERIC (4)
        assert_eq!(entry[4], SORTED_NUMERIC);

        // docs_with_field_offset = -1 (ALL)
        assert_eq!(&entry[5..13], &(-1i64).to_le_bytes());
        // docs_with_field_length = 0
        assert_eq!(&entry[13..21], &0i64.to_le_bytes());
        // jump_table_entry_count = -1
        assert_eq!(&entry[21..23], &(-1i16).to_le_bytes());
        // dense_rank_power = 0xFF
        assert_eq!(entry[23], 0xFF);

        // numValues = 3
        assert_eq!(&entry[24..32], &3i64.to_le_bytes());

        // tablesize = -1
        assert_eq!(&entry[32..36], &(-1i32).to_le_bytes());
        // numBitsPerValue = 0 (constant)
        assert_eq!(entry[36], 0);
        // min = 42
        assert_eq!(&entry[37..45], &42i64.to_le_bytes());

        // Verify footer present
        let footer_start = dvm.len() - FOOTER_LENGTH;
        assert_eq!(
            &dvm[footer_start..footer_start + 4],
            &[0xc0, 0x28, 0x93, 0xe8]
        );
    }

    #[test]
    fn test_sorted_numeric_different() {
        // 3 docs with distinct timestamps
        let fields = vec![make_field_data_sorted_numeric(
            "modified",
            1,
            vec![snv(0, vec![1000]), snv(1, vec![2000]), snv(2, vec![3000])],
        )];

        let segment_id = [0u8; 16];
        let directory = make_test_directory();
        let result = test_write(&directory, "_0", "Lucene90_0", &segment_id, &fields, 3).unwrap();

        let dvm = directory.read_file(&result[0]).unwrap();
        let dvd = directory.read_file(&result[1]).unwrap();

        // Should succeed and produce valid output
        assert_not_empty!(dvm);
        assert_not_empty!(dvd);

        // Parse metadata
        let meta_header_len = index_header_length(META_CODEC, "Lucene90_0");
        let entry = &dvm[meta_header_len..];

        // field_number = 1
        assert_eq!(&entry[0..4], &1i32.to_le_bytes());
        // type byte = SORTED_NUMERIC
        assert_eq!(entry[4], SORTED_NUMERIC);

        // numValues = 3
        assert_eq!(&entry[24..32], &3i64.to_le_bytes());

        // Values have GCD=1000, min=1000, so encoded as (val-1000)/1000 = 0,1,2
        // GCD of (0, 1000-1000, 2000-1000, 3000-1000) = gcd(0, 0, 1000, 2000) = 1000
        // bpv = unsignedBitsRequired((3000-1000)/1000) = unsignedBitsRequired(2) = 2
        let gcd_offset = 37 + 8; // min is at 37, gcd at 37+8=45
        let gcd_val = i64::from_le_bytes(entry[gcd_offset..gcd_offset + 8].try_into().unwrap());
        assert_eq!(gcd_val, 1000);
    }

    #[test]
    fn test_sorted_set_single_valued() {
        // 3 docs with distinct path values
        let fields = vec![make_field_data_sorted_set(
            "path",
            0,
            vec![
                ssv(0, vec![bytes("/a.txt")]),
                ssv(1, vec![bytes("/b.txt")]),
                ssv(2, vec![bytes("/c.txt")]),
            ],
        )];

        let segment_id = [0u8; 16];
        let directory = make_test_directory();
        let result = test_write(&directory, "_0", "Lucene90_0", &segment_id, &fields, 3).unwrap();

        let dvm = directory.read_file(&result[0]).unwrap();
        let dvd = directory.read_file(&result[1]).unwrap();

        assert_not_empty!(dvm);
        assert_not_empty!(dvd);

        // Verify codec magic
        assert_eq!(&dvm[0..4], &[0x3f, 0xd7, 0x6c, 0x17]);
        assert_eq!(&dvd[0..4], &[0x3f, 0xd7, 0x6c, 0x17]);

        // Parse metadata
        let meta_header_len = index_header_length(META_CODEC, "Lucene90_0");
        let entry = &dvm[meta_header_len..];

        // field_number = 0
        assert_eq!(&entry[0..4], &0i32.to_le_bytes());
        // type byte = SORTED_SET (3)
        assert_eq!(entry[4], SORTED_SET);
        // multiValued = 0 (single-valued)
        assert_eq!(entry[5], 0);

        // Footer present
        let footer_start = dvm.len() - FOOTER_LENGTH;
        assert_eq!(
            &dvm[footer_start..footer_start + 4],
            &[0xc0, 0x28, 0x93, 0xe8]
        );
    }

    #[test]
    fn test_header_footer_eof() {
        // Test that the EOF marker (-1 as i32) appears before the footer
        let fields = vec![make_field_data_sorted_numeric(
            "modified",
            0,
            vec![snv(0, vec![1])],
        )];

        let segment_id = [0u8; 16];
        let directory = make_test_directory();
        let result = test_write(&directory, "_0", "", &segment_id, &fields, 1).unwrap();

        let dvm = directory.read_file(&result[0]).unwrap();

        // EOF marker is 4 bytes before footer
        let eof_pos = dvm.len() - FOOTER_LENGTH - 4;
        assert_eq!(&dvm[eof_pos..eof_pos + 4], &(-1i32).to_le_bytes());
    }

    #[test]
    fn test_terms_dict_prefix_compression_byte() {
        // Verify the prefix/suffix byte encoding
        // prefix=3, suffix_len=5: byte = min(3,15) | (min(4,15) << 4) = 3 | (4<<4) = 3|64 = 67
        let prefix_len = 3usize;
        let suffix_len = 5usize;
        let byte = (prefix_len.min(15) as u8) | ((suffix_len.saturating_sub(1).min(15) as u8) << 4);
        assert_eq!(byte, 0x43); // 3 | (4 << 4) = 0x43

        // Large prefix: prefix=20 (clamped to 15), suffix=1 → 0x0F, then VInt(5) follows
        let byte2 = 15u8;
        assert_eq!(byte2, 0x0F);
    }

    #[test]
    fn test_two_fields_combined() {
        // Test writing both field types together (like the real indexer)
        let fields = vec![
            make_field_data_sorted_set(
                "path",
                0,
                vec![
                    ssv(0, vec![bytes("/a.txt")]),
                    ssv(1, vec![bytes("/b.txt")]),
                    ssv(2, vec![bytes("/c.txt")]),
                ],
            ),
            make_field_data_sorted_numeric(
                "modified",
                1,
                vec![snv(0, vec![1000]), snv(1, vec![2000]), snv(2, vec![3000])],
            ),
        ];

        let segment_id = [0u8; 16];
        let directory = make_test_directory();
        let result = test_write(&directory, "_0", "Lucene90_0", &segment_id, &fields, 3).unwrap();

        assert_len_eq_x!(&result, 2);
        assert_eq!(result[0], "_0_Lucene90_0.dvm");
        assert_eq!(result[1], "_0_Lucene90_0.dvd");

        let dvm = directory.read_file(&result[0]).unwrap();

        // Parse: after header, first field (path=0, SORTED_SET)
        let meta_header_len = index_header_length(META_CODEC, "Lucene90_0");
        let entry = &dvm[meta_header_len..];
        assert_eq!(&entry[0..4], &0i32.to_le_bytes()); // field 0
        assert_eq!(entry[4], SORTED_SET);

        // Second field should appear later (modified=1, SORTED_NUMERIC)
        // Find the EOF marker
        let eof_pos = dvm.len() - FOOTER_LENGTH - 4;
        assert_eq!(&dvm[eof_pos..eof_pos + 4], &(-1i32).to_le_bytes());
    }

    /// Helper to read through .dvm metadata like Java's Lucene90DocValuesProducer.readFields().
    /// Wraps `TestDataReader` with domain-specific read methods for doc values metadata.
    struct DvmReader<'a>(TestDataReader<'a>);

    impl<'a> DvmReader<'a> {
        fn new(data: &'a [u8], start: usize) -> Self {
            Self(TestDataReader::new(data, start))
        }

        /// Read a numeric entry (matches Java readNumeric).
        /// Returns num_values for multi-valued detection.
        fn read_numeric(&mut self) -> i64 {
            let _docs_with_field_offset = self.0.read_le_long();
            let _docs_with_field_length = self.0.read_le_long();
            let _jump_table_entry_count = self.0.read_le_short();
            let _dense_rank_power = self.0.read_byte();
            let num_values = self.0.read_le_long();
            let table_size = self.0.read_le_int();
            if table_size >= 0 {
                for _ in 0..table_size {
                    let _table_val = self.0.read_le_long();
                }
            }
            let _bits_per_value = self.0.read_byte();
            let _min_value = self.0.read_le_long();
            let _gcd = self.0.read_le_long();
            let _values_offset = self.0.read_le_long();
            let _values_length = self.0.read_le_long();
            let _value_jump_table_offset = self.0.read_le_long();
            num_values
        }

        /// Read DirectMonotonicReader.loadMeta (matches Java)
        fn read_dm_meta(&mut self, num_values: i64, _block_shift: i32) {
            let num_blocks = (num_values + (1 << 16) - 1) / (1 << 16);
            for _ in 0..num_blocks {
                let _min = self.0.read_le_long();
                let _avg_inc = self.0.read_le_int(); // Float.intBitsToFloat
                let _offset = self.0.read_le_long();
                let _bits_required = self.0.read_byte();
            }
        }

        /// Read term dict entry (matches Java readTermDict)
        fn read_term_dict(&mut self) {
            let terms_dict_size = self.0.read_vlong();
            let block_shift = self.0.read_le_int();
            let addresses_size = (terms_dict_size + (1i64 << TERMS_DICT_BLOCK_LZ4_SHIFT) - 1)
                >> TERMS_DICT_BLOCK_LZ4_SHIFT;
            self.read_dm_meta(addresses_size, block_shift);
            let _max_term_length = self.0.read_le_int();
            let _max_block_length = self.0.read_le_int();
            let _terms_data_offset = self.0.read_le_long();
            let _terms_data_length = self.0.read_le_long();
            let _terms_addresses_offset = self.0.read_le_long();
            let _terms_addresses_length = self.0.read_le_long();
            let terms_dict_index_shift = self.0.read_le_int();
            let index_size =
                (terms_dict_size + (1i64 << terms_dict_index_shift) - 1) >> terms_dict_index_shift;
            self.read_dm_meta(1 + index_size, block_shift);
            let _terms_index_offset = self.0.read_le_long();
            let _terms_index_length = self.0.read_le_long();
            let _terms_index_addresses_offset = self.0.read_le_long();
            let _terms_index_addresses_length = self.0.read_le_long();
        }

        /// Read sorted entry (matches Java readSorted — no multiValued byte)
        fn read_sorted(&mut self) {
            let _ = self.read_numeric();
            self.read_term_dict();
        }

        /// Read binary entry (matches Java readBinary)
        fn read_binary(&mut self) {
            let _data_offset = self.0.read_le_long();
            let _data_length = self.0.read_le_long();
            let _docs_with_field_offset = self.0.read_le_long();
            let _docs_with_field_length = self.0.read_le_long();
            let _jump_table_entry_count = self.0.read_le_short();
            let _dense_rank_power = self.0.read_byte();
            let num_docs_with_field = self.0.read_le_int();
            let min_length = self.0.read_le_int();
            let max_length = self.0.read_le_int();
            if max_length > min_length {
                let _addresses_offset = self.0.read_le_long();
                let block_shift = self.0.read_vint();
                let num_addresses = num_docs_with_field as i64 + 1;
                self.read_dm_meta(num_addresses, block_shift);
                let _addresses_length = self.0.read_le_long();
            }
        }

        /// Read sorted set entry (matches Java readSortedSet)
        fn read_sorted_set(&mut self) {
            let multi_valued = self.0.read_byte();
            if multi_valued == 0 {
                // Single-valued: readSorted path (readNumeric + readTermDict)
                let _ = self.read_numeric();
                self.read_term_dict();
            } else {
                // Multi-valued: readSortedNumeric path + readTermDict
                self.read_sorted_numeric();
                self.read_term_dict();
            }
        }

        /// Read sorted numeric entry (matches Java readSortedNumeric)
        fn read_sorted_numeric(&mut self) {
            let num_values = self.read_numeric();
            let num_docs_with_field = self.0.read_le_int();
            // If multi-valued, read address table
            if num_values > num_docs_with_field as i64 {
                let _addresses_offset = self.0.read_le_long();
                let block_shift = self.0.read_vint();
                let num_addresses = num_docs_with_field as i64 + 1;
                self.read_dm_meta(num_addresses, block_shift);
                let _addresses_length = self.0.read_le_long();
            }
        }
    }

    #[test]
    fn test_two_fields_dvm_parseable_like_java() {
        // Regression test: verifies that SORTED_SET does NOT write an extra
        // numDocsWithField int, which would shift all subsequent field reads.
        let fields = vec![
            make_field_data_sorted_set(
                "path",
                0,
                vec![
                    ssv(0, vec![bytes("/a.txt")]),
                    ssv(1, vec![bytes("/b.txt")]),
                    ssv(2, vec![bytes("/c.txt")]),
                ],
            ),
            make_field_data_sorted_numeric(
                "modified",
                1,
                vec![snv(0, vec![1000]), snv(1, vec![2000]), snv(2, vec![3000])],
            ),
        ];

        let segment_id = [0u8; 16];
        let directory = make_test_directory();
        let result = test_write(&directory, "_0", "Lucene90_0", &segment_id, &fields, 3).unwrap();
        let dvm = directory.read_file(&result[0]).unwrap();

        // Parse the .dvm like Java's Lucene90DocValuesProducer.readFields()
        let meta_header_len = index_header_length(META_CODEC, "Lucene90_0");
        let mut reader = DvmReader::new(&dvm, meta_header_len);

        // First field: path (#0, SORTED_SET)
        let field0 = reader.0.read_le_int();
        assert_eq!(field0, 0, "first field number should be 0");
        let type0 = reader.0.read_byte();
        assert_eq!(type0, SORTED_SET, "first field type should be SORTED_SET");
        reader.read_sorted_set();

        // Second field: modified (#1, SORTED_NUMERIC)
        let field1 = reader.0.read_le_int();
        assert_eq!(
            field1, 1,
            "second field number should be 1 (got {field1} — extra bytes written for SORTED_SET?)"
        );
        let type1 = reader.0.read_byte();
        assert_eq!(
            type1, SORTED_NUMERIC,
            "second field type should be SORTED_NUMERIC"
        );
        reader.read_sorted_numeric();

        // EOF marker
        let eof = reader.0.read_le_int();
        assert_eq!(eof, -1, "expected EOF marker (-1)");
    }

    #[test]
    fn test_sorted_numeric_with_gcd() {
        // Values with GCD: 100, 200, 300 → GCD=100 (from first_value diff)
        let fields = vec![make_field_data_sorted_numeric(
            "field",
            0,
            vec![snv(0, vec![100]), snv(1, vec![200]), snv(2, vec![300])],
        )];

        let segment_id = [0u8; 16];
        let directory = make_test_directory();
        let result = test_write(&directory, "_0", "", &segment_id, &fields, 3).unwrap();

        // Should complete without error
        assert_len_eq_x!(&result, 2);
    }

    #[test]
    fn test_sorted_set_identical_paths() {
        // All docs have the same path → ordinals are all 0 → constant
        let fields = vec![make_field_data_sorted_set(
            "path",
            0,
            vec![
                ssv(0, vec![bytes("/same.txt")]),
                ssv(1, vec![bytes("/same.txt")]),
                ssv(2, vec![bytes("/same.txt")]),
            ],
        )];

        let segment_id = [0u8; 16];
        let directory = make_test_directory();
        let result = test_write(&directory, "_0", "", &segment_id, &fields, 3).unwrap();
        assert_len_eq_x!(&result, 2);
    }

    #[test]
    fn test_numeric_constant() {
        let fields = vec![make_field_data_numeric(
            "count",
            0,
            vec![nv(0, 42), nv(1, 42), nv(2, 42)],
        )];

        let segment_id = [0u8; 16];
        let directory = make_test_directory();
        let result = test_write(&directory, "_0", "Lucene90_0", &segment_id, &fields, 3).unwrap();

        let dvm = directory.read_file(&result[0]).unwrap();
        let meta_header_len = index_header_length(META_CODEC, "Lucene90_0");
        let entry = &dvm[meta_header_len..];

        // field_number = 0
        assert_eq!(&entry[0..4], &0i32.to_le_bytes());
        // type byte = NUMERIC (0)
        assert_eq!(entry[4], NUMERIC);
        // docs_with_field_offset = -1 (ALL)
        assert_eq!(&entry[5..13], &(-1i64).to_le_bytes());
        // numValues = 3
        assert_eq!(&entry[24..32], &3i64.to_le_bytes());
        // tablesize = -1
        assert_eq!(&entry[32..36], &(-1i32).to_le_bytes());
        // bpv = 0 (constant)
        assert_eq!(entry[36], 0);
        // min = 42
        assert_eq!(&entry[37..45], &42i64.to_le_bytes());
    }

    #[test]
    fn test_numeric_different_values() {
        let fields = vec![make_field_data_numeric(
            "score",
            0,
            vec![nv(0, 10), nv(1, 20), nv(2, 30)],
        )];

        let segment_id = [0u8; 16];
        let directory = make_test_directory();
        let result = test_write(&directory, "_0", "", &segment_id, &fields, 3).unwrap();

        let dvm = directory.read_file(&result[0]).unwrap();
        let dvd = directory.read_file(&result[1]).unwrap();
        assert_not_empty!(dvm);
        assert_not_empty!(dvd);
    }

    #[test]
    fn test_numeric_no_num_docs_with_field() {
        // NUMERIC must NOT write numDocsWithField — verify by parsing the metadata
        let fields = vec![make_field_data_numeric(
            "count",
            0,
            vec![nv(0, 100), nv(1, 200), nv(2, 300)],
        )];

        let segment_id = [0u8; 16];
        let directory = make_test_directory();
        let result = test_write(&directory, "_0", "Lucene90_0", &segment_id, &fields, 3).unwrap();

        let dvm = directory.read_file(&result[0]).unwrap();
        let meta_header_len = index_header_length(META_CODEC, "Lucene90_0");
        let mut reader = DvmReader::new(&dvm, meta_header_len);

        let field0 = reader.0.read_le_int();
        assert_eq!(field0, 0);
        let type0 = reader.0.read_byte();
        assert_eq!(type0, NUMERIC);
        // NUMERIC: read_numeric only (no numDocsWithField)
        reader.read_numeric();

        // Next should be EOF marker
        let eof = reader.0.read_le_int();
        assert_eq!(
            eof, -1,
            "expected EOF marker — NUMERIC should not write numDocsWithField"
        );
    }

    #[test]
    fn test_sorted_field() {
        let fields = vec![make_field_data_sorted(
            "category",
            0,
            vec![
                sv(0, bytes("alpha")),
                sv(1, bytes("beta")),
                sv(2, bytes("alpha")),
            ],
        )];

        let segment_id = [0u8; 16];
        let directory = make_test_directory();
        let result = test_write(&directory, "_0", "Lucene90_0", &segment_id, &fields, 3).unwrap();

        let dvm = directory.read_file(&result[0]).unwrap();
        let meta_header_len = index_header_length(META_CODEC, "Lucene90_0");
        let entry = &dvm[meta_header_len..];

        // field_number = 0
        assert_eq!(&entry[0..4], &0i32.to_le_bytes());
        // type byte = SORTED (2) — no multiValued byte
        assert_eq!(entry[4], SORTED);
    }

    #[test]
    fn test_sorted_parseable() {
        // Verify SORTED metadata is parseable (no multiValued byte, unlike SORTED_SET)
        let fields = vec![make_field_data_sorted(
            "category",
            0,
            vec![sv(0, bytes("x")), sv(1, bytes("y")), sv(2, bytes("z"))],
        )];

        let segment_id = [0u8; 16];
        let directory = make_test_directory();
        let result = test_write(&directory, "_0", "Lucene90_0", &segment_id, &fields, 3).unwrap();

        let dvm = directory.read_file(&result[0]).unwrap();
        let meta_header_len = index_header_length(META_CODEC, "Lucene90_0");
        let mut reader = DvmReader::new(&dvm, meta_header_len);

        let field0 = reader.0.read_le_int();
        assert_eq!(field0, 0);
        let type0 = reader.0.read_byte();
        assert_eq!(type0, SORTED);
        reader.read_sorted();

        let eof = reader.0.read_le_int();
        assert_eq!(eof, -1, "expected EOF marker");
    }

    #[test]
    fn test_binary_fixed_length() {
        let fields = vec![make_field_data_binary(
            "hash",
            0,
            vec![
                bv(0, vec![0xAA, 0xBB, 0xCC, 0xDD]),
                bv(1, vec![0x11, 0x22, 0x33, 0x44]),
                bv(2, vec![0xFF, 0xEE, 0xDD, 0xCC]),
            ],
        )];

        let segment_id = [0u8; 16];
        let directory = make_test_directory();
        let result = test_write(&directory, "_0", "Lucene90_0", &segment_id, &fields, 3).unwrap();

        let dvm = directory.read_file(&result[0]).unwrap();
        let meta_header_len = index_header_length(META_CODEC, "Lucene90_0");
        let entry = &dvm[meta_header_len..];

        assert_eq!(&entry[0..4], &0i32.to_le_bytes());
        assert_eq!(entry[4], BINARY);
    }

    #[test]
    fn test_binary_parseable() {
        let fields = vec![make_field_data_binary(
            "data",
            0,
            vec![
                bv(0, vec![1, 2, 3]),
                bv(1, vec![4, 5, 6]),
                bv(2, vec![7, 8, 9]),
            ],
        )];

        let segment_id = [0u8; 16];
        let directory = make_test_directory();
        let result = test_write(&directory, "_0", "Lucene90_0", &segment_id, &fields, 3).unwrap();

        let dvm = directory.read_file(&result[0]).unwrap();
        let meta_header_len = index_header_length(META_CODEC, "Lucene90_0");
        let mut reader = DvmReader::new(&dvm, meta_header_len);

        let field0 = reader.0.read_le_int();
        assert_eq!(field0, 0);
        let type0 = reader.0.read_byte();
        assert_eq!(type0, BINARY);
        reader.read_binary();

        let eof = reader.0.read_le_int();
        assert_eq!(eof, -1, "expected EOF marker");
    }

    #[test]
    fn test_binary_variable_length() {
        let fields = vec![make_field_data_binary(
            "payload",
            0,
            vec![bv(0, vec![1]), bv(1, vec![2, 3, 4]), bv(2, vec![5, 6])],
        )];

        let segment_id = [0u8; 16];
        let directory = make_test_directory();
        let result = test_write(&directory, "_0", "Lucene90_0", &segment_id, &fields, 3).unwrap();

        let dvm = directory.read_file(&result[0]).unwrap();
        let meta_header_len = index_header_length(META_CODEC, "Lucene90_0");
        let mut reader = DvmReader::new(&dvm, meta_header_len);

        let field0 = reader.0.read_le_int();
        assert_eq!(field0, 0);
        let type0 = reader.0.read_byte();
        assert_eq!(type0, BINARY);
        reader.read_binary();

        let eof = reader.0.read_le_int();
        assert_eq!(
            eof, -1,
            "expected EOF marker — variable-length binary should be parseable"
        );
    }

    #[test]
    fn test_all_dv_types_combined() {
        // Write all 5 doc values types in one segment and verify parseability
        let fields = vec![
            make_field_data_numeric("num", 0, vec![nv(0, 10), nv(1, 20), nv(2, 30)]),
            make_field_data_binary(
                "bin",
                1,
                vec![bv(0, vec![0xAA]), bv(1, vec![0xBB]), bv(2, vec![0xCC])],
            ),
            make_field_data_sorted(
                "sort",
                2,
                vec![sv(0, bytes("a")), sv(1, bytes("b")), sv(2, bytes("c"))],
            ),
            make_field_data_sorted_set(
                "sortset",
                3,
                vec![
                    ssv(0, vec![bytes("x")]),
                    ssv(1, vec![bytes("y")]),
                    ssv(2, vec![bytes("z")]),
                ],
            ),
            make_field_data_sorted_numeric(
                "sortnum",
                4,
                vec![snv(0, vec![100]), snv(1, vec![200]), snv(2, vec![300])],
            ),
        ];

        let segment_id = [0u8; 16];
        let directory = make_test_directory();
        let result = test_write(&directory, "_0", "Lucene90_0", &segment_id, &fields, 3).unwrap();

        let dvm = directory.read_file(&result[0]).unwrap();
        let meta_header_len = index_header_length(META_CODEC, "Lucene90_0");
        let mut reader = DvmReader::new(&dvm, meta_header_len);

        // Field 0: NUMERIC
        assert_eq!(reader.0.read_le_int(), 0);
        assert_eq!(reader.0.read_byte(), NUMERIC);
        reader.read_numeric();

        // Field 1: BINARY
        assert_eq!(reader.0.read_le_int(), 1);
        assert_eq!(reader.0.read_byte(), BINARY);
        reader.read_binary();

        // Field 2: SORTED
        assert_eq!(reader.0.read_le_int(), 2);
        assert_eq!(reader.0.read_byte(), SORTED);
        reader.read_sorted();

        // Field 3: SORTED_SET
        assert_eq!(reader.0.read_le_int(), 3);
        assert_eq!(reader.0.read_byte(), SORTED_SET);
        reader.read_sorted_set();

        // Field 4: SORTED_NUMERIC
        assert_eq!(reader.0.read_le_int(), 4);
        assert_eq!(reader.0.read_byte(), SORTED_NUMERIC);
        reader.read_sorted_numeric();

        // EOF
        assert_eq!(reader.0.read_le_int(), -1);
    }

    #[test]
    fn test_sorted_numeric_multi_valued() {
        // 3 docs with varying value counts: 1, 2, 3 values
        let fields = vec![make_field_data_sorted_numeric(
            "tags",
            0,
            vec![
                snv(0, vec![100]),
                snv(1, vec![300, 200]), // unsorted — should be sorted by codec
                snv(2, vec![600, 400, 500]), // unsorted — should be sorted by codec
            ],
        )];

        let segment_id = [0u8; 16];
        let directory = make_test_directory();
        let result = test_write(&directory, "_0", "Lucene90_0", &segment_id, &fields, 3).unwrap();
        assert_len_eq_x!(&result, 2);

        // Verify MetaReader can parse the full metadata
        let dvm = directory.read_file(&result[0]).unwrap();
        let meta_header_len = index_header_length(META_CODEC, "Lucene90_0");
        let mut reader = DvmReader::new(&dvm, meta_header_len);

        assert_eq!(reader.0.read_le_int(), 0);
        assert_eq!(reader.0.read_byte(), SORTED_NUMERIC);
        reader.read_sorted_numeric();

        // EOF
        assert_eq!(reader.0.read_le_int(), -1);
    }

    #[test]
    fn test_sorted_numeric_multi_valued_values_sorted() {
        // Verify that values are sorted within each doc by checking the flattened
        // data output: for doc1=[300,200] and doc2=[600,400,500], the flattened
        // values should be [100, 200, 300, 400, 500, 600].
        let fields = vec![make_field_data_sorted_numeric(
            "nums",
            0,
            vec![
                snv(0, vec![100]),
                snv(1, vec![300, 200]),
                snv(2, vec![600, 400, 500]),
            ],
        )];

        let segment_id = [0u8; 16];
        let directory = make_test_directory();
        let result = test_write(&directory, "_0", "Lucene90_0", &segment_id, &fields, 3).unwrap();

        // The .dvm metadata should have numValues=6 (total across all docs)
        let dvm = directory.read_file(&result[0]).unwrap();
        let meta_header_len = index_header_length(META_CODEC, "Lucene90_0");
        let entry = &dvm[meta_header_len..];

        // Skip field_number (4) + type byte (1) + docs_with_field (8+8+2+1) = 24 bytes
        // numValues is at offset 24
        let num_values = i64::from_le_bytes(entry[24..32].try_into().unwrap());
        assert_eq!(num_values, 6, "total values across all docs should be 6");
    }

    #[test]
    fn test_two_fields_multi_valued_sorted_numeric_dvm_parseable() {
        // Two-field test: multi-valued SORTED_SET (single-valued) + multi-valued SORTED_NUMERIC
        let fields = vec![
            make_field_data_sorted_set(
                "path",
                0,
                vec![
                    ssv(0, vec![bytes("/a.txt")]),
                    ssv(1, vec![bytes("/b.txt")]),
                    ssv(2, vec![bytes("/c.txt")]),
                ],
            ),
            make_field_data_sorted_numeric(
                "counts",
                1,
                vec![
                    snv(0, vec![10, 20]),
                    snv(1, vec![30]),
                    snv(2, vec![40, 50, 60]),
                ],
            ),
        ];

        let segment_id = [0u8; 16];
        let directory = make_test_directory();
        let result = test_write(&directory, "_0", "Lucene90_0", &segment_id, &fields, 3).unwrap();

        let dvm = directory.read_file(&result[0]).unwrap();
        let meta_header_len = index_header_length(META_CODEC, "Lucene90_0");
        let mut reader = DvmReader::new(&dvm, meta_header_len);

        // Field 0: SORTED_SET (single-valued)
        assert_eq!(reader.0.read_le_int(), 0);
        assert_eq!(reader.0.read_byte(), SORTED_SET);
        reader.read_sorted_set();

        // Field 1: SORTED_NUMERIC (multi-valued)
        assert_eq!(reader.0.read_le_int(), 1);
        assert_eq!(reader.0.read_byte(), SORTED_NUMERIC);
        reader.read_sorted_numeric();

        // EOF
        assert_eq!(reader.0.read_le_int(), -1);
    }

    #[test]
    fn test_sorted_set_multi_valued() {
        // 3 docs with 1, 2, 3 values respectively
        let fields = vec![make_field_data_sorted_set(
            "tags",
            0,
            vec![
                ssv(0, vec![bytes("alpha")]),
                ssv(1, vec![bytes("gamma"), bytes("beta")]),
                ssv(2, vec![bytes("delta"), bytes("alpha"), bytes("gamma")]),
            ],
        )];

        let segment_id = [0u8; 16];
        let directory = make_test_directory();
        let result = test_write(&directory, "_0", "Lucene90_0", &segment_id, &fields, 3).unwrap();
        assert_len_eq_x!(&result, 2);

        // Verify MetaReader can parse the full metadata
        let dvm = directory.read_file(&result[0]).unwrap();
        let meta_header_len = index_header_length(META_CODEC, "Lucene90_0");
        let mut reader = DvmReader::new(&dvm, meta_header_len);

        assert_eq!(reader.0.read_le_int(), 0);
        assert_eq!(reader.0.read_byte(), SORTED_SET);
        reader.read_sorted_set();

        // EOF
        assert_eq!(reader.0.read_le_int(), -1);
    }

    #[test]
    fn test_sorted_set_multi_valued_dedup() {
        // Doc with duplicate terms should produce unique ordinals
        let fields = vec![make_field_data_sorted_set(
            "tags",
            0,
            vec![
                ssv(0, vec![bytes("alpha"), bytes("alpha"), bytes("beta")]),
                ssv(1, vec![bytes("beta"), bytes("beta")]),
            ],
        )];

        let segment_id = [0u8; 16];
        let directory = make_test_directory();
        let result = test_write(&directory, "_0", "Lucene90_0", &segment_id, &fields, 2).unwrap();
        assert_len_eq_x!(&result, 2);

        // Verify MetaReader can parse the full metadata
        let dvm = directory.read_file(&result[0]).unwrap();
        let meta_header_len = index_header_length(META_CODEC, "Lucene90_0");
        let mut reader = DvmReader::new(&dvm, meta_header_len);

        assert_eq!(reader.0.read_le_int(), 0);
        assert_eq!(reader.0.read_byte(), SORTED_SET);
        reader.read_sorted_set();

        // EOF
        assert_eq!(reader.0.read_le_int(), -1);
    }

    #[test]
    fn test_two_fields_multi_valued_sorted_set_and_sorted_numeric() {
        // Two multi-valued fields: SORTED_SET + SORTED_NUMERIC
        let fields = vec![
            make_field_data_sorted_set(
                "tags",
                0,
                vec![
                    ssv(0, vec![bytes("a"), bytes("b")]),
                    ssv(1, vec![bytes("c")]),
                    ssv(2, vec![bytes("a"), bytes("c")]),
                ],
            ),
            make_field_data_sorted_numeric(
                "nums",
                1,
                vec![
                    snv(0, vec![10, 20]),
                    snv(1, vec![30]),
                    snv(2, vec![40, 50, 60]),
                ],
            ),
        ];

        let segment_id = [0u8; 16];
        let directory = make_test_directory();
        let result = test_write(&directory, "_0", "Lucene90_0", &segment_id, &fields, 3).unwrap();

        let dvm = directory.read_file(&result[0]).unwrap();
        let meta_header_len = index_header_length(META_CODEC, "Lucene90_0");
        let mut reader = DvmReader::new(&dvm, meta_header_len);

        // Field 0: SORTED_SET (multi-valued)
        assert_eq!(reader.0.read_le_int(), 0);
        assert_eq!(reader.0.read_byte(), SORTED_SET);
        reader.read_sorted_set();

        // Field 1: SORTED_NUMERIC (multi-valued)
        assert_eq!(reader.0.read_le_int(), 1);
        assert_eq!(reader.0.read_byte(), SORTED_NUMERIC);
        reader.read_sorted_numeric();

        // EOF
        assert_eq!(reader.0.read_le_int(), -1);
    }

    #[test]
    fn test_sparse_numeric_field() {
        // 3 docs with values out of 10 total — SPARSE pattern
        let fields = vec![make_field_data_numeric(
            "score",
            0,
            vec![nv(1, 100), nv(5, 200), nv(8, 300)],
        )];

        let segment_id = [0u8; 16];
        let dir = make_test_directory();
        let names = test_write(&dir, "_0", "", &segment_id, &fields, 10).unwrap();

        let dvm = dir.read_file(&names[0]).unwrap();
        let dvd = dir.read_file(&names[1]).unwrap();

        let meta_header_len = index_header_length(META_CODEC, "");
        let mut reader = TestDataReader::new(&dvm[meta_header_len..], 0);

        // field_number = 0, type = NUMERIC
        assert_eq!(reader.read_le_int(), 0);
        assert_eq!(reader.read_byte(), NUMERIC);

        // docsWithFieldOffset >= 0 means SPARSE (IndexedDISI data in .dvd)
        let docs_with_field_offset = reader.read_le_long();
        assert_ge!(docs_with_field_offset, 0);

        // docsWithFieldLength > 0
        let docs_with_field_length = reader.read_le_long();
        assert_gt!(docs_with_field_length, 0);

        // jumpTableEntryCount (>= 0 means real count from IndexedDISI)
        let jump_table_entry_count = reader.read_le_short();
        assert_ge!(jump_table_entry_count, 0);

        // denseRankPower = 9 (DEFAULT_DENSE_RANK_POWER)
        let dense_rank_power = reader.read_byte();
        assert_eq!(dense_rank_power, 9);

        // Verify IndexedDISI data was written to .dvd at the specified offset
        let data_header_len = index_header_length(DATA_CODEC, "");
        let disi_start = docs_with_field_offset as usize;
        assert_ge!(disi_start, data_header_len);
        assert_eq!(
            docs_with_field_length as usize,
            (disi_start + docs_with_field_length as usize) - disi_start
        );

        // Verify the IndexedDISI block header: blockID=0, cardinality-1=2 (3 docs)
        let block_id = i16::from_le_bytes(dvd[disi_start..disi_start + 2].try_into().unwrap());
        assert_eq!(block_id, 0);
        let card_minus_1 =
            i16::from_le_bytes(dvd[disi_start + 2..disi_start + 4].try_into().unwrap());
        assert_eq!(card_minus_1, 2);
    }

    #[test]
    fn test_sparse_binary_field() {
        // 2 docs with values out of 5 total — SPARSE pattern
        let fields = vec![make_field_data_binary(
            "tag",
            0,
            vec![bv(1, b"hello".to_vec()), bv(3, b"world".to_vec())],
        )];

        let segment_id = [0u8; 16];
        let dir = make_test_directory();
        let names = test_write(&dir, "_0", "", &segment_id, &fields, 5).unwrap();

        let dvm = dir.read_file(&names[0]).unwrap();

        let meta_header_len = index_header_length(META_CODEC, "");
        let mut reader = TestDataReader::new(&dvm[meta_header_len..], 0);

        // field_number = 0, type = BINARY
        assert_eq!(reader.read_le_int(), 0);
        assert_eq!(reader.read_byte(), BINARY);

        // dataOffset, dataLength
        let _data_offset = reader.read_le_long();
        let _data_length = reader.read_le_long();

        // docsWithFieldOffset >= 0 means SPARSE
        let docs_with_field_offset = reader.read_le_long();
        assert_ge!(docs_with_field_offset, 0);

        let docs_with_field_length = reader.read_le_long();
        assert_gt!(docs_with_field_length, 0);

        let jump_table_entry_count = reader.read_le_short();
        assert_ge!(jump_table_entry_count, 0);

        let dense_rank_power = reader.read_byte();
        assert_eq!(dense_rank_power, 9);

        // numDocsWithField = 2
        assert_eq!(reader.read_le_int(), 2);
    }

    #[test]
    fn test_sparse_sorted_field() {
        // 2 docs with values out of 5 total — SPARSE sorted
        let fields = vec![make_field_data_sorted(
            "category",
            0,
            vec![sv(0, b"alpha".to_vec()), sv(3, b"beta".to_vec())],
        )];

        let segment_id = [0u8; 16];
        let dir = make_test_directory();
        let names = test_write(&dir, "_0", "", &segment_id, &fields, 5).unwrap();

        let dvm = dir.read_file(&names[0]).unwrap();

        let meta_header_len = index_header_length(META_CODEC, "");
        let mut reader = TestDataReader::new(&dvm[meta_header_len..], 0);

        // field_number = 0, type = SORTED
        assert_eq!(reader.read_le_int(), 0);
        assert_eq!(reader.read_byte(), SORTED);

        // docsWithFieldOffset >= 0 means SPARSE
        let docs_with_field_offset = reader.read_le_long();
        assert_ge!(docs_with_field_offset, 0);

        let docs_with_field_length = reader.read_le_long();
        assert_gt!(docs_with_field_length, 0);

        let _jump_table_entry_count = reader.read_le_short();

        let dense_rank_power = reader.read_byte();
        assert_eq!(dense_rank_power, 9);
    }

    #[test]
    fn test_sparse_sorted_numeric_field() {
        // 2 docs with values out of 5 total — SPARSE sorted numeric
        let fields = vec![make_field_data_sorted_numeric(
            "counts",
            0,
            vec![snv(1, vec![10, 20]), snv(4, vec![30])],
        )];

        let segment_id = [0u8; 16];
        let dir = make_test_directory();
        let names = test_write(&dir, "_0", "", &segment_id, &fields, 5).unwrap();

        let dvm = dir.read_file(&names[0]).unwrap();

        let meta_header_len = index_header_length(META_CODEC, "");
        let mut reader = TestDataReader::new(&dvm[meta_header_len..], 0);

        // field_number = 0, type = SORTED_NUMERIC
        assert_eq!(reader.read_le_int(), 0);
        assert_eq!(reader.read_byte(), SORTED_NUMERIC);

        // docsWithFieldOffset >= 0 means SPARSE
        let docs_with_field_offset = reader.read_le_long();
        assert_ge!(docs_with_field_offset, 0);

        let docs_with_field_length = reader.read_le_long();
        assert_gt!(docs_with_field_length, 0);

        let _jump_table_entry_count = reader.read_le_short();

        let dense_rank_power = reader.read_byte();
        assert_eq!(dense_rank_power, 9);
    }
}
