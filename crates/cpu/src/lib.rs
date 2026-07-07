// SPDX-License-Identifier: Apache-2.0

//! CPU reference executors (SPEC crates/cpu).
//!
//! The correctness reference and portability fallback (SPEC §3.5): consumes
//! [`DecodePlan`]s and data-file bytes, produces Arrow arrays. The GPU
//! executor must be bit-identical to this on every input (SPEC §12.3).
//!
//! [`DecodePlan`]: lucene_arrow_core::plan::DecodePlan

use std::sync::Arc;

use arrow_array::{
    ArrayRef, BinaryArray, DictionaryArray, FixedSizeListArray, Float32Array, Float64Array,
    Int8Array, Int32Array, Int64Array, ListArray, StringArray,
};
use arrow_buffer::{NullBuffer, OffsetBuffer, ScalarBuffer};
use arrow_schema::{DataType, Field};

use lucene_arrow_core::plan::{BlockDecode, Coverage, DecodePlan};
use lucene_arrow_core::{Error, Result};
use lucene_arrow_docvalues::read::{BinaryPlan, MultiNumericPlan, SortedPlan};
use lucene_arrow_docvalues::{direct, disi, terms};

/// Decode a numeric doc-values plan against the full data-file bytes
/// (`.dvd`). Output length is the plan's doc count; sparse coverage becomes
/// Arrow validity ("field absent for this doc", SPEC §7.1 — never
/// deletion).
pub fn decode_numeric(plan: &DecodePlan, dvd: &[u8]) -> Result<ArrayRef> {
    let mut values: Vec<i64> = Vec::with_capacity(plan.num_values as usize);
    for block in &plan.blocks {
        decode_block(block, dvd, &mut values)?;
    }
    if values.len() as u64 != plan.num_values {
        return Err(Error::corrupt(format!(
            "plan promised {} values, blocks yielded {}",
            plan.num_values,
            values.len()
        )));
    }

    let (num_docs, validity) = match &plan.coverage {
        Coverage::Dense { num_docs } => {
            if values.len() != *num_docs as usize {
                return Err(Error::corrupt(format!(
                    "dense column: {} values for {num_docs} docs",
                    values.len()
                )));
            }
            (*num_docs, None)
        }
        Coverage::Empty { num_docs } => {
            values = vec![0i64; *num_docs as usize];
            (*num_docs, Some(vec![0u64; (*num_docs as usize).div_ceil(64)]))
        }
        Coverage::Sparse { num_docs, disi: d } => {
            let end = d
                .offset
                .checked_add(d.len)
                .ok_or_else(|| Error::corrupt("DISI range overflow"))?;
            if end as usize > dvd.len() {
                return Err(Error::corrupt("DISI range beyond data file"));
            }
            let region = &dvd[d.offset as usize..end as usize];
            let bitmap = disi::decode(region, d.num_values, *num_docs, d.dense_rank_power)?;
            // Scatter compact values to their doc positions (null slots 0).
            let mut scattered = vec![0i64; *num_docs as usize];
            let mut v = 0usize;
            for (w, &word) in bitmap.iter().enumerate() {
                let mut bits = word;
                while bits != 0 {
                    let doc = w * 64 + bits.trailing_zeros() as usize;
                    scattered[doc] = values[v];
                    v += 1;
                    bits &= bits - 1;
                }
            }
            if v != values.len() {
                return Err(Error::corrupt("DISI cardinality != decoded value count"));
            }
            values = scattered;
            (*num_docs, Some(bitmap))
        }
    };

    let nulls = validity.map(|words| {
        NullBuffer::new(arrow_buffer::BooleanBuffer::new(
            arrow_buffer::Buffer::from_vec(words),
            0,
            num_docs as usize,
        ))
    });

    match plan.arrow_type {
        DataType::Int64 => Ok(Arc::new(Int64Array::new(values.into(), nulls)) as ArrayRef),
        DataType::Float64 => {
            let floats: Vec<f64> = values.iter().map(|&v| f64::from_bits(v as u64)).collect();
            Ok(Arc::new(Float64Array::new(floats.into(), nulls)) as ArrayRef)
        }
        ref other => Err(Error::unsupported(format!("numeric decode into {other}"))),
    }
}

/// Decode just the packed value stream of a plan (no coverage/scatter):
/// the building block for multi-valued shapes where positions come from
/// an addresses sequence rather than docids.
pub fn decode_values(plan: &DecodePlan, dvd: &[u8]) -> Result<Vec<i64>> {
    let mut values = Vec::with_capacity(plan.num_values as usize);
    for block in &plan.blocks {
        decode_block(block, dvd, &mut values)?;
    }
    if values.len() as u64 != plan.num_values {
        return Err(Error::corrupt("blocks yielded wrong value count"));
    }
    Ok(values)
}

/// Coverage → (per-doc validity, docs-with-value count), plus a closure
/// input for offset building. Returns the DISI bitmap when sparse.
fn coverage_bitmap(coverage: &Coverage, dvd: &[u8]) -> Result<(u32, Option<Vec<u64>>)> {
    match coverage {
        Coverage::Dense { num_docs } => Ok((*num_docs, None)),
        Coverage::Empty { num_docs } => {
            Ok((*num_docs, Some(vec![0u64; (*num_docs as usize).div_ceil(64)])))
        }
        Coverage::Sparse { num_docs, disi: d } => {
            let end = d.offset + d.len;
            if end as usize > dvd.len() {
                return Err(Error::corrupt("DISI range beyond data file"));
            }
            let region = &dvd[d.offset as usize..end as usize];
            let bitmap = disi::decode(region, d.num_values, *num_docs, d.dense_rank_power)?;
            Ok((*num_docs, Some(bitmap)))
        }
    }
}

fn bitmap_to_nulls(bitmap: Option<Vec<u64>>, num_docs: u32) -> Option<NullBuffer> {
    bitmap.map(|words| {
        NullBuffer::new(arrow_buffer::BooleanBuffer::new(
            arrow_buffer::Buffer::from_vec(words),
            0,
            num_docs as usize,
        ))
    })
}

/// Per-doc List offsets from a per-doc-with-value addresses sequence.
fn doc_offsets(
    num_docs: u32,
    bitmap: Option<&Vec<u64>>,
    addresses: &[i64],
) -> Result<OffsetBuffer<i32>> {
    let mut offsets = Vec::with_capacity(num_docs as usize + 1);
    offsets.push(0i32);
    match bitmap {
        None => {
            if addresses.len() != num_docs as usize + 1 {
                return Err(Error::corrupt("dense addresses count != num_docs + 1"));
            }
            for &a in &addresses[1..] {
                offsets.push(i32::try_from(a).map_err(|_| Error::corrupt("offset > i32::MAX"))?);
            }
        }
        Some(words) => {
            let mut rank = 0usize; // docs-with-value seen so far
            for d in 0..num_docs as usize {
                let valid = words[d / 64] >> (d % 64) & 1 == 1;
                if valid {
                    rank += 1;
                }
                let end = addresses.get(rank).copied().unwrap_or(0);
                offsets.push(i32::try_from(end).map_err(|_| Error::corrupt("offset > i32::MAX"))?);
            }
        }
    }
    Ok(OffsetBuffer::new(ScalarBuffer::from(offsets)))
}

/// Terms dictionary → Arrow values array: `Utf8` when every term is valid
/// UTF-8 (the common keyword case), `Binary` otherwise (SPEC §7.2).
fn terms_to_values(dict: &terms::TermsDict) -> ArrayRef {
    let offsets = OffsetBuffer::new(ScalarBuffer::from(dict.offsets.clone()));
    let bytes = arrow_buffer::Buffer::from_slice_ref(&dict.bytes);
    if std::str::from_utf8(&dict.bytes).is_ok() {
        // Safety: offsets are monotone within bytes; whole buffer is UTF-8
        // and every term boundary was produced by the byte-sorted writer.
        Arc::new(unsafe { StringArray::new_unchecked(offsets, bytes, None) }) as ArrayRef
    } else {
        Arc::new(BinaryArray::new(offsets, bytes, None)) as ArrayRef
    }
}

/// Decode a SORTED / single-valued SORTED_SET / multi-valued SORTED_SET
/// column with per-segment dictionaries (`lucene.dict = segment`,
/// SPEC §7.3): `Dictionary<Int32, Utf8|Binary>` or `List<Dictionary<…>>`.
pub fn decode_sorted(plan: &SortedPlan, dvd: &[u8]) -> Result<ArrayRef> {
    let dict = terms::materialize(&plan.terms, dvd)?;
    let values = terms_to_values(&dict);
    let (num_docs, bitmap) = coverage_bitmap(&plan.ords.coverage, dvd)?;

    match &plan.addresses {
        None => {
            // One ord per doc-with-value: scatter to doc slots.
            let ords = decode_values(&plan.ords, dvd)?;
            let mut keys = vec![0i32; num_docs as usize];
            match &bitmap {
                None => {
                    if ords.len() != num_docs as usize {
                        return Err(Error::corrupt("dense ords != num_docs"));
                    }
                    for (d, &o) in ords.iter().enumerate() {
                        keys[d] = i32::try_from(o).map_err(|_| Error::corrupt("ord > i32"))?;
                    }
                }
                Some(words) => {
                    let mut v = 0usize;
                    for (w, &word) in words.iter().enumerate() {
                        let mut bits = word;
                        while bits != 0 {
                            let doc = w * 64 + bits.trailing_zeros() as usize;
                            keys[doc] = i32::try_from(ords[v])
                                .map_err(|_| Error::corrupt("ord > i32"))?;
                            v += 1;
                            bits &= bits - 1;
                        }
                    }
                }
            }
            let keys = Int32Array::new(keys.into(), bitmap_to_nulls(bitmap, num_docs));
            Ok(Arc::new(DictionaryArray::try_new(keys, values)?) as ArrayRef)
        }
        Some(addresses) => {
            // Concatenated ord stream + per-doc offsets → List<Dictionary>.
            let ords = decode_values(&plan.ords, dvd)?;
            let addr = addresses.decode(dvd)?;
            let offsets = doc_offsets(num_docs, bitmap.as_ref(), &addr)?;
            let mut keys = Vec::with_capacity(ords.len());
            for &o in &ords {
                keys.push(i32::try_from(o).map_err(|_| Error::corrupt("ord > i32"))?);
            }
            let child = DictionaryArray::try_new(Int32Array::from(keys), values)?;
            let field = Arc::new(Field::new(
                "item",
                arrow_array::Array::data_type(&child).clone(),
                false,
            ));
            Ok(Arc::new(ListArray::new(
                field,
                offsets,
                Arc::new(child),
                bitmap_to_nulls(bitmap, num_docs),
            )) as ArrayRef)
        }
    }
}

/// Decode a multi-valued SORTED_NUMERIC column → `List<Int64>` (SPEC §7.2;
/// values within a doc come back in Lucene's sorted order).
pub fn decode_multi_numeric(plan: &MultiNumericPlan, dvd: &[u8]) -> Result<ArrayRef> {
    let values = decode_values(&plan.values, dvd)?;
    let addr = plan.addresses.decode(dvd)?;
    let (num_docs, bitmap) = coverage_bitmap(&plan.values.coverage, dvd)?;
    let offsets = doc_offsets(num_docs, bitmap.as_ref(), &addr)?;
    let child = Int64Array::from(values);
    let field = Arc::new(Field::new("item", DataType::Int64, false));
    Ok(Arc::new(ListArray::new(
        field,
        offsets,
        Arc::new(child),
        bitmap_to_nulls(bitmap, num_docs),
    )) as ArrayRef)
}

/// Decode a BINARY column into a doc-aligned `Binary` array
/// (SPEC §7.2): values concatenated in doc order; fixed-length when
/// `min == max`, else per-value DirectMonotonic start offsets.
pub fn decode_binary(plan: &BinaryPlan, dvd: &[u8]) -> Result<ArrayRef> {
    let end = plan.data_offset + plan.data_len;
    if end as usize > dvd.len() {
        return Err(Error::corrupt("binary data region beyond file"));
    }
    let region = &dvd[plan.data_offset as usize..end as usize];
    let (num_docs, bitmap) = coverage_bitmap(&plan.coverage, dvd)?;
    let n = plan.num_docs_with_field as usize;

    // Per-value start offsets (n + 1 entries) into `region`.
    let value_offsets: Vec<i64> = match &plan.addresses {
        None => (0..=n as i64).map(|i| i * plan.min_length as i64).collect(),
        Some(addr) => addr.decode(dvd)?,
    };
    if value_offsets.len() != n + 1 {
        return Err(Error::corrupt("binary addresses count mismatch"));
    }

    // Doc-aligned offsets: docs without a value get an empty slice + null.
    let mut offsets = Vec::with_capacity(num_docs as usize + 1);
    offsets.push(0i32);
    match &bitmap {
        None => {
            if n != num_docs as usize {
                return Err(Error::corrupt("dense binary column count mismatch"));
            }
            for &o in &value_offsets[1..] {
                offsets.push(i32::try_from(o).map_err(|_| Error::corrupt("binary offset > i32"))?);
            }
        }
        Some(words) => {
            let mut rank = 0usize;
            for d in 0..num_docs as usize {
                if words[d / 64] >> (d % 64) & 1 == 1 {
                    rank += 1;
                }
                let o = value_offsets[rank];
                offsets.push(i32::try_from(o).map_err(|_| Error::corrupt("binary offset > i32"))?);
            }
        }
    }

    let values = arrow_buffer::Buffer::from_slice_ref(region);
    let array = BinaryArray::try_new(
        OffsetBuffer::new(ScalarBuffer::from(offsets)),
        values,
        bitmap_to_nulls(bitmap, num_docs),
    )?;
    Ok(Arc::new(array) as ArrayRef)
}

/// Decode a flat-vector plan (`Raw` block) into a doc-aligned
/// `FixedSizeList` array: row `i` = doc `i`, null = "no vector for this
/// doc" (SPEC §7.2). Vectors are stored ord-ordered; sparse coverage
/// scatters them to doc slots via the DISI bitmap.
pub fn decode_flat_vectors(plan: &DecodePlan, vec_data: &[u8]) -> Result<ArrayRef> {
    let DataType::FixedSizeList(child_field, dim) = &plan.arrow_type else {
        return Err(Error::invalid("vector plan must have FixedSizeList type"));
    };
    let dim = *dim as usize;
    let width = match child_field.data_type() {
        DataType::Float32 => 4,
        DataType::Int8 => 1,
        other => return Err(Error::unsupported(format!("vector child type {other}"))),
    };

    let payload: &[u8] = match plan.blocks.as_slice() {
        [] => &[],
        [BlockDecode::Raw { offset, len }] => {
            let end = offset.checked_add(*len).ok_or_else(|| Error::corrupt("raw range overflow"))?;
            if end as usize > vec_data.len() {
                return Err(Error::corrupt("vector data range beyond file"));
            }
            &vec_data[*offset as usize..end as usize]
        }
        _ => return Err(Error::invalid("vector plan must be a single Raw block")),
    };
    let count = plan.num_values as usize;
    if payload.len() != count * dim * width {
        return Err(Error::corrupt("vector payload size mismatch"));
    }

    // (row index in output, validity) per coverage.
    let (num_rows, nulls, doc_of_ord): (usize, Option<NullBuffer>, Option<Vec<usize>>) =
        match &plan.coverage {
            Coverage::Dense { num_docs } => {
                if count != *num_docs as usize {
                    return Err(Error::corrupt("dense vector column count mismatch"));
                }
                (*num_docs as usize, None, None)
            }
            Coverage::Empty { num_docs } => {
                let words = vec![0u64; (*num_docs as usize).div_ceil(64)];
                let nulls = NullBuffer::new(arrow_buffer::BooleanBuffer::new(
                    arrow_buffer::Buffer::from_vec(words),
                    0,
                    *num_docs as usize,
                ));
                (*num_docs as usize, Some(nulls), Some(Vec::new()))
            }
            Coverage::Sparse { num_docs, disi: d } => {
                let end = d.offset + d.len;
                if end as usize > vec_data.len() {
                    return Err(Error::corrupt("DISI range beyond data file"));
                }
                let region = &vec_data[d.offset as usize..end as usize];
                let bitmap = disi::decode(region, d.num_values, *num_docs, d.dense_rank_power)?;
                let mut docs = Vec::with_capacity(count);
                for (w, &word) in bitmap.iter().enumerate() {
                    let mut bits = word;
                    while bits != 0 {
                        docs.push(w * 64 + bits.trailing_zeros() as usize);
                        bits &= bits - 1;
                    }
                }
                let nulls = NullBuffer::new(arrow_buffer::BooleanBuffer::new(
                    arrow_buffer::Buffer::from_vec(bitmap),
                    0,
                    *num_docs as usize,
                ));
                (*num_docs as usize, Some(nulls), Some(docs))
            }
        };

    // Child values buffer, doc-aligned (zeros in null slots).
    let mut child_bytes = vec![0u8; num_rows * dim * width];
    match &doc_of_ord {
        None => child_bytes.copy_from_slice(payload),
        Some(docs) => {
            for (ord, &doc) in docs.iter().enumerate() {
                let src = ord * dim * width;
                let dst = doc * dim * width;
                child_bytes[dst..dst + dim * width].copy_from_slice(&payload[src..src + dim * width]);
            }
        }
    }

    let child: ArrayRef = match child_field.data_type() {
        DataType::Float32 => {
            let floats: Vec<f32> = child_bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().expect("4 bytes")))
                .collect();
            Arc::new(Float32Array::from(floats))
        }
        DataType::Int8 => {
            let ints: Vec<i8> = child_bytes.iter().map(|&b| b as i8).collect();
            Arc::new(Int8Array::from(ints))
        }
        _ => unreachable!("checked above"),
    };

    Ok(Arc::new(FixedSizeListArray::new(child_field.clone(), dim as i32, child, nulls)) as ArrayRef)
}

/// One block: unpack + fused arithmetic epilogue (the CPU mirror of the
/// SPEC §11.3 kernel).
fn decode_block(block: &BlockDecode, dvd: &[u8], out: &mut Vec<i64>) -> Result<()> {
    let payload = |offset: u64, len: u64| -> Result<&[u8]> {
        let end = offset.checked_add(len).ok_or_else(|| Error::corrupt("block range overflow"))?;
        if end as usize > dvd.len() {
            return Err(Error::corrupt(format!("block [{offset}, {end}) beyond data file")));
        }
        Ok(&dvd[offset as usize..end as usize])
    };

    // Single pass, no intermediate buffer: unpack feeds the arithmetic
    // epilogue through an inlined closure (SPEC §11.3, fused).
    match block {
        BlockDecode::Direct { offset, len, bit_width, values } => {
            direct::for_each_unpacked(payload(*offset, *len)?, *bit_width, *values as usize, |x| {
                out.push(x as i64)
            })?;
        }
        BlockDecode::DeltaPacked { bit_width: 0, base, values, .. } => {
            out.extend(std::iter::repeat_n(*base, *values as usize));
        }
        BlockDecode::DeltaPacked { offset, len, bit_width, base, values } => {
            let base = *base;
            direct::for_each_unpacked(payload(*offset, *len)?, *bit_width, *values as usize, |x| {
                out.push(base.wrapping_add(x as i64))
            })?;
        }
        BlockDecode::GcdPacked { offset, len, bit_width, base, gcd, values } => {
            let (base, gcd) = (*base, *gcd);
            direct::for_each_unpacked(payload(*offset, *len)?, *bit_width, *values as usize, |x| {
                out.push(base.wrapping_add(gcd.wrapping_mul(x as i64)))
            })?;
        }
        BlockDecode::Table { offset, len, bit_width, table, values } => {
            let mut bad = None;
            direct::for_each_unpacked(payload(*offset, *len)?, *bit_width, *values as usize, |x| {
                match table.get(x as usize) {
                    Some(&v) => out.push(v),
                    None => bad = Some(x),
                }
            })?;
            if let Some(x) = bad {
                return Err(Error::corrupt(format!("table index {x} out of range")));
            }
        }
        BlockDecode::Monotonic { .. } | BlockDecode::Ordinals { .. } | BlockDecode::Raw { .. } => {
            return Err(Error::unsupported(
                "block kind not used by numeric doc values (lands with P2/P3)",
            ));
        }
    }
    Ok(())
}
