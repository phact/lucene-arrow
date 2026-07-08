// SPDX-License-Identifier: Apache-2.0

//! NUMERIC doc values encoding (SPEC §10, §11.7 CPU reference).
//!
//! Faithful port of Bearing's `write_values`/`add_numeric_field` (itself a
//! port of Java `Lucene90DocValuesConsumer`), so output is byte-identical
//! to what Bearing writes and cross-validated against Java Lucene. Like
//! Bearing, the encoder always emits single-block mode (valid, readable by
//! Java; multi-block is a size optimization Java sometimes picks — read
//! side handles it, write side may learn it later; encode-policy placement
//! is decision register #6).

use std::collections::BTreeSet;

use crate::direct;
use crate::disi;
use lucene_arrow_core::{Error, Result};

/// Field statistics driving the §10.2/§11.7 encoding policy. The policy
/// itself is host-side and shared by every executor (decision register
/// #6): same stats → same plan → same bytes.
#[derive(Debug, Clone, PartialEq)]
pub struct NumericStats {
    pub min: i64,
    pub max: i64,
    /// GCD of deltas from the first value (1 when heterogeneous or any
    /// value is outside `i64::MIN/2..=i64::MAX/2`, matching Java).
    pub gcd: i64,
    /// Sorted distinct values, present only when ≤ 256 of them.
    pub table: Option<Vec<i64>>,
}

/// Stats + pack executor for numeric doc values. Implementations must be
/// bit-identical: the CPU one is the reference; the GPU one lives in
/// `lucene-arrow-gpu` and is differential-gated against it.
pub trait NumericEncoder {
    fn stats(&self, values: &[i64]) -> Result<NumericStats>;
    /// Pack `(v - base) / gcd` (or the table index of `v` when `table` is
    /// given) at `bpv`, appending DirectWriter padding.
    fn pack(
        &self,
        values: &[i64],
        bpv: u8,
        base: i64,
        gcd: i64,
        table: Option<&[i64]>,
    ) -> Result<Vec<u8>>;

    /// [`stats`](Self::stats) over a logically-concatenated chunk list —
    /// the zero-copy ingest lane hands Arrow batch slices straight through
    /// instead of materializing one host Vec. MUST equal `stats(concat)`.
    fn stats_chunks(&self, chunks: &[&[i64]]) -> Result<NumericStats> {
        self.stats(&chunks.concat())
    }

    /// [`pack`](Self::pack) over the same chunk list. MUST equal
    /// `pack(concat, ..)` byte-for-byte.
    fn pack_chunks(
        &self,
        chunks: &[&[i64]],
        bpv: u8,
        base: i64,
        gcd: i64,
        table: Option<&[i64]>,
    ) -> Result<Vec<u8>> {
        self.pack(&chunks.concat(), bpv, base, gcd, table)
    }
}

/// The reference executor (SPEC §3.5).
#[derive(Debug, Default, Clone, Copy)]
pub struct CpuEncoder;

impl NumericEncoder for CpuEncoder {
    fn stats(&self, values: &[i64]) -> Result<NumericStats> {
        let first_value = values[0];
        let mut min = first_value;
        let mut max = first_value;
        let mut gcd: i64 = 0;
        let mut unique: Option<BTreeSet<i64>> = Some(BTreeSet::new());
        for &v in values {
            min = min.min(v);
            max = max.max(v);
            if gcd != 1 {
                if !(i64::MIN / 2..=i64::MAX / 2).contains(&v) {
                    gcd = 1;
                } else {
                    gcd = gcd_compute(gcd, v.wrapping_sub(first_value));
                }
            }
            if let Some(set) = unique.as_mut() {
                set.insert(v);
                if set.len() > 256 {
                    unique = None;
                }
            }
        }
        Ok(NumericStats { min, max, gcd, table: unique.map(|s| s.into_iter().collect()) })
    }

    fn pack(
        &self,
        values: &[i64],
        bpv: u8,
        base: i64,
        gcd: i64,
        table: Option<&[i64]>,
    ) -> Result<Vec<u8>> {
        let encoded: Vec<i64> = match table {
            Some(t) => values
                .iter()
                .map(|v| {
                    t.binary_search(v)
                        .map(|i| i as i64)
                        .map_err(|_| Error::invalid("value missing from table"))
                })
                .collect::<Result<_>>()?,
            None => values.iter().map(|&v| v.wrapping_sub(base).wrapping_div(gcd)).collect(),
        };
        let mut out = Vec::new();
        direct::pack(&encoded, bpv, &mut out);
        Ok(out)
    }

    fn stats_chunks(&self, chunks: &[&[i64]]) -> Result<NumericStats> {
        // Same fold as `stats`, streamed across chunks (no concat).
        let first_value = chunks[0][0];
        let mut min = first_value;
        let mut max = first_value;
        let mut gcd: i64 = 0;
        let mut unique: Option<BTreeSet<i64>> = Some(BTreeSet::new());
        for &v in chunks.iter().copied().flatten() {
            min = min.min(v);
            max = max.max(v);
            if gcd != 1 {
                if !(i64::MIN / 2..=i64::MAX / 2).contains(&v) {
                    gcd = 1;
                } else {
                    gcd = gcd_compute(gcd, v.wrapping_sub(first_value));
                }
            }
            if let Some(set) = unique.as_mut() {
                set.insert(v);
                if set.len() > 256 {
                    unique = None;
                }
            }
        }
        Ok(NumericStats { min, max, gcd, table: unique.map(|s| s.into_iter().collect()) })
    }

    fn pack_chunks(
        &self,
        chunks: &[&[i64]],
        bpv: u8,
        base: i64,
        gcd: i64,
        table: Option<&[i64]>,
    ) -> Result<Vec<u8>> {
        // One transform pass over the chunks into the pre-pack buffer —
        // the single alloc `pack` would have done anyway.
        let n: usize = chunks.iter().map(|c| c.len()).sum();
        let mut encoded: Vec<i64> = Vec::with_capacity(n);
        match table {
            Some(t) => {
                for &v in chunks.iter().copied().flatten() {
                    let i = t
                        .binary_search(&v)
                        .map_err(|_| Error::invalid("value missing from table"))?;
                    encoded.push(i as i64);
                }
            }
            None => {
                for &v in chunks.iter().copied().flatten() {
                    encoded.push(v.wrapping_sub(base).wrapping_div(gcd));
                }
            }
        }
        let mut out = Vec::new();
        direct::pack(&encoded, bpv, &mut out);
        Ok(out)
    }
}

/// One encoded field: `meta` is the `.dvm` entry (field number + type byte
/// included); `data` is the `.dvd` payload. All offsets inside `meta` are
/// absolute `.dvd` file offsets, computed from `dvd_base` (the `.dvd`
/// length before this field's payload is appended).
pub struct EncodedField {
    pub meta: Vec<u8>,
    pub data: Vec<u8>,
}

/// Encode one NUMERIC doc-values field.
///
/// `docs` are the (sorted, unique) docids carrying a value; `values[i]`
/// belongs to `docs[i]`. `max_doc` is the segment's doc count.
pub fn encode_numeric_field(
    field_number: i32,
    docs: &[u32],
    values: &[i64],
    max_doc: u32,
    dvd_base: u64,
) -> Result<EncodedField> {
    encode_numeric_field_with(&CpuEncoder, field_number, docs, values, max_doc, dvd_base)
}

/// Value stream for [`write_values`]: one flat slice, or the zero-copy
/// lane's list of Arrow batch slices (logically concatenated).
pub enum ValuesIn<'a> {
    Flat(&'a [i64]),
    Chunks(&'a [&'a [i64]]),
}

impl ValuesIn<'_> {
    fn len(&self) -> usize {
        match self {
            ValuesIn::Flat(v) => v.len(),
            ValuesIn::Chunks(c) => c.iter().map(|c| c.len()).sum(),
        }
    }
    fn stats(&self, e: &dyn NumericEncoder) -> Result<NumericStats> {
        match self {
            ValuesIn::Flat(v) => e.stats(v),
            ValuesIn::Chunks(c) => e.stats_chunks(c),
        }
    }
    fn pack(
        &self,
        e: &dyn NumericEncoder,
        bpv: u8,
        base: i64,
        gcd: i64,
        table: Option<&[i64]>,
    ) -> Result<Vec<u8>> {
        match self {
            ValuesIn::Flat(v) => e.pack(v, bpv, base, gcd, table),
            ValuesIn::Chunks(c) => e.pack_chunks(c, bpv, base, gcd, table),
        }
    }
}

/// Encode one **dense** NUMERIC field straight from Arrow batch slices —
/// the zero-copy ingest lane. Every doc has a value (`Σ chunks == max_doc`),
/// so no docs array is ever materialized. Byte-identical to
/// [`encode_numeric_field_with`] on the concatenation.
pub fn encode_numeric_field_dense_chunks(
    encoder: &dyn NumericEncoder,
    field_number: i32,
    chunks: &[&[i64]],
    max_doc: u32,
    dvd_base: u64,
) -> Result<EncodedField> {
    let n: usize = chunks.iter().map(|c| c.len()).sum();
    if n != max_doc as usize {
        return Err(Error::invalid(format!("dense chunks: {n} values != max_doc {max_doc}")));
    }
    let mut meta = Vec::new();
    let mut data = Vec::new();
    meta.extend_from_slice(&field_number.to_le_bytes());
    meta.push(crate::consts::TYPE_NUMERIC);
    write_values(&mut meta, &mut data, &ValuesIn::Chunks(chunks), &[], max_doc, dvd_base, encoder)?;
    Ok(EncodedField { meta, data })
}

/// [`encode_numeric_field`] with an explicit executor (CPU reference or
/// the GPU encoder from `lucene-arrow-gpu`). Same plan policy either way
/// → same bytes (differential-gated).
pub fn encode_numeric_field_with(
    encoder: &dyn NumericEncoder,
    field_number: i32,
    docs: &[u32],
    values: &[i64],
    max_doc: u32,
    dvd_base: u64,
) -> Result<EncodedField> {
    if docs.len() != values.len() {
        return Err(Error::invalid("docs and values length mismatch"));
    }
    if docs.windows(2).any(|w| w[0] >= w[1]) {
        return Err(Error::invalid("docs must be sorted and unique"));
    }
    if let Some(&last) = docs.last()
        && last >= max_doc
    {
        return Err(Error::invalid(format!("doc {last} >= max_doc {max_doc}")));
    }

    let mut meta = Vec::new();
    let mut data = Vec::new();
    meta.extend_from_slice(&field_number.to_le_bytes());
    meta.push(crate::consts::TYPE_NUMERIC);
    write_values(&mut meta, &mut data, &ValuesIn::Flat(values), docs, max_doc, dvd_base, encoder)?;
    Ok(EncodedField { meta, data })
}

/// The `writeValues` body shared by NUMERIC (and later SORTED_NUMERIC).
/// For [`ValuesIn::Chunks`] callers, `docs` is empty and the field is
/// dense (`vals.len() == max_doc`); flat callers pass real docids.
fn write_values(
    meta: &mut Vec<u8>,
    data: &mut Vec<u8>,
    vals: &ValuesIn<'_>,
    docs: &[u32],
    max_doc: u32,
    dvd_base: u64,
    encoder: &dyn NumericEncoder,
) -> Result<()> {
    let num_values = vals.len() as i64;
    let num_docs_with_value =
        if docs.is_empty() && num_values == max_doc as i64 { max_doc } else { docs.len() as u32 };

    // Docs-with-field indicator (IndexedDISI metadata).
    if num_docs_with_value == 0 {
        meta.extend_from_slice(&(-2i64).to_le_bytes());
        meta.extend_from_slice(&0i64.to_le_bytes());
        meta.extend_from_slice(&(-1i16).to_le_bytes());
        meta.push(0xFF); // denseRankPower = -1
    } else if num_docs_with_value == max_doc {
        meta.extend_from_slice(&(-1i64).to_le_bytes());
        meta.extend_from_slice(&0i64.to_le_bytes());
        meta.extend_from_slice(&(-1i16).to_le_bytes());
        meta.push(0xFF);
    } else {
        let offset = dvd_base + data.len() as u64;
        meta.extend_from_slice(&(offset as i64).to_le_bytes());
        let before = data.len();
        let jump_table_entry_count = disi::write_bit_set(docs, data)?;
        meta.extend_from_slice(&((data.len() - before) as i64).to_le_bytes());
        meta.extend_from_slice(&jump_table_entry_count.to_le_bytes());
        meta.push(disi::DEFAULT_DENSE_RANK_POWER);
    }

    meta.extend_from_slice(&num_values.to_le_bytes());

    if num_values == 0 {
        meta.extend_from_slice(&(-1i32).to_le_bytes()); // tableSize
        meta.push(0); // numBitsPerValue
        meta.extend_from_slice(&0i64.to_le_bytes()); // min
        meta.extend_from_slice(&0i64.to_le_bytes()); // gcd
        meta.extend_from_slice(&((dvd_base + data.len() as u64) as i64).to_le_bytes());
        meta.extend_from_slice(&0i64.to_le_bytes()); // valuesLength
        meta.extend_from_slice(&(-1i64).to_le_bytes()); // jumpTableOffset
        return Ok(());
    }

    // Stats: min, max, gcd, unique values (≤256) — executor-computed.
    let stats = vals.stats(encoder)?;
    let (mut min, max) = (stats.min, stats.max);
    let mut gcd = stats.gcd;
    let unique = stats.table;

    let num_bits_per_value: u8;
    let mut encode_table: Option<Vec<i64>> = None;

    if min >= max {
        num_bits_per_value = 0;
        meta.extend_from_slice(&(-1i32).to_le_bytes());
    } else if let Some(uv) = unique.as_ref().filter(|uv| uv.len() > 1) {
        let table_bpv = direct::unsigned_bits_required(uv.len() as i64 - 1);
        let delta_bpv = direct::unsigned_bits_required(max.wrapping_sub(min).wrapping_div(gcd));
        if table_bpv < delta_bpv {
            num_bits_per_value = table_bpv;
            meta.extend_from_slice(&(uv.len() as i32).to_le_bytes());
            for v in uv {
                meta.extend_from_slice(&v.to_le_bytes());
            }
            encode_table = Some(uv.clone());
            min = 0;
            gcd = 1;
        } else {
            num_bits_per_value = delta_bpv;
            meta.extend_from_slice(&(-1i32).to_le_bytes());
            if gcd == 1 && min > 0 && direct::unsigned_bits_required(max) == direct::unsigned_bits_required(max.wrapping_sub(min)) {
                min = 0;
            }
        }
    } else {
        num_bits_per_value = direct::unsigned_bits_required(max.wrapping_sub(min).wrapping_div(gcd));
        meta.extend_from_slice(&(-1i32).to_le_bytes());
        if gcd == 1 && min > 0 && direct::unsigned_bits_required(max) == direct::unsigned_bits_required(max.wrapping_sub(min)) {
            min = 0;
        }
    }

    meta.push(num_bits_per_value);
    meta.extend_from_slice(&min.to_le_bytes());
    meta.extend_from_slice(&gcd.to_le_bytes());
    let start_offset = dvd_base + data.len() as u64;
    meta.extend_from_slice(&(start_offset as i64).to_le_bytes());

    if num_bits_per_value > 0 {
        let payload = vals.pack(encoder, num_bits_per_value, min, gcd, encode_table.as_deref())?;
        data.extend_from_slice(&payload);
    }

    let values_length = dvd_base + data.len() as u64 - start_offset;
    meta.extend_from_slice(&(values_length as i64).to_le_bytes());
    meta.extend_from_slice(&(-1i64).to_le_bytes()); // jumpTableOffset: single block
    Ok(())
}

/// Java `MathUtil.gcd` (binary GCD over absolute values).
fn gcd_compute(a: i64, b: i64) -> i64 {
    let mut a = if a < 0 { a.wrapping_neg() } else { a };
    let mut b = if b < 0 { b.wrapping_neg() } else { b };
    if a == 0 {
        return b;
    }
    if b == 0 {
        return a;
    }
    let shift = (a | b).trailing_zeros();
    a >>= a.trailing_zeros();
    loop {
        b >>= b.trailing_zeros();
        if a == b {
            break;
        }
        if (a as u64) > (b as u64) || a == i64::MIN {
            std::mem::swap(&mut a, &mut b);
        }
        if a == 1 {
            break;
        }
        b -= a;
    }
    a << shift
}


// --- SORTED / SORTED_SET write path (SPEC §10.3) -------------------------

/// Sort-key length helper (Java `StringHelper.sortKeyLength`): shared
/// prefix length + 1, clamped to the shorter term's length + 1... exactly:
/// bytes needed so `key` sorts `b` after `a`.
fn sort_key_length(a: &[u8], b: &[u8]) -> usize {
    let common = a.iter().zip(b).take_while(|(x, y)| x == y).count();
    (common + 1).min(b.len())
}

/// Byte-difference position (Java `StringHelper.bytesDifference`): length
/// of the common prefix of consecutive sorted terms.
fn bytes_difference(prev: &[u8], term: &[u8]) -> usize {
    prev.iter().zip(term).take_while(|(x, y)| x == y).count()
}

/// Port of Bearing's `add_terms_dict` (byte-identical: same LZ4, same
/// DirectMonotonic): 64-term blocks, first term raw, suffix stream
/// LZ4-compressed with the block's first term as dictionary.
fn add_terms_dict(
    meta: &mut Vec<u8>,
    data: &mut Vec<u8>,
    sorted_terms: &[&[u8]],
    dvd_base: u64,
) -> Result<()> {
    use lucene_arrow_core::cursor::{write_vint, write_vlong};

    write_vlong(meta, sorted_terms.len() as u64);
    meta.extend_from_slice(&(crate::consts::DIRECT_MONOTONIC_BLOCK_SHIFT as i32).to_le_bytes());

    let block_mask = (crate::terms::TERMS_DICT_BLOCK_SIZE - 1) as usize;
    let mut dm_values: Vec<i64> = Vec::new();
    let mut previous: &[u8] = &[];
    let mut max_length: i32 = 0;
    let mut max_block_length: i32 = 0;
    let start = dvd_base + data.len() as u64;

    let mut suffix_buffer: Vec<u8> = Vec::new();
    let mut dict_bytes: Vec<u8> = Vec::new();
    let mut lz4_ht = bearing::encoding::lz4::FastHashTable::new();

    let flush_block = |data: &mut Vec<u8>,
                           dict: &[u8],
                           suffix: &[u8],
                           ht: &mut bearing::encoding::lz4::FastHashTable|
     -> i32 {
        write_vint(data, suffix.len() as u32);
        let mut combined = Vec::with_capacity(dict.len() + suffix.len());
        combined.extend_from_slice(dict);
        combined.extend_from_slice(suffix);
        let compressed =
            bearing::encoding::lz4::compress_with_dictionary_reuse(&combined, dict.len(), ht);
        data.extend_from_slice(&compressed);
        suffix.len() as i32
    };

    for (ord, term) in sorted_terms.iter().enumerate() {
        if ord & block_mask == 0 {
            if ord != 0 {
                let l = flush_block(data, &dict_bytes, &suffix_buffer, &mut lz4_ht);
                max_block_length = max_block_length.max(l);
                suffix_buffer.clear();
            }
            dm_values.push((dvd_base + data.len() as u64 - start) as i64);
            write_vint(data, term.len() as u32);
            data.extend_from_slice(term);
            dict_bytes = term.to_vec();
        } else {
            let prefix_len = bytes_difference(previous, term);
            let suffix_len = term.len() - prefix_len;
            if suffix_len == 0 {
                return Err(Error::invalid("duplicate terms in sorted dictionary"));
            }
            suffix_buffer.push(
                (prefix_len.min(15) as u8) | ((suffix_len.saturating_sub(1).min(15) as u8) << 4),
            );
            if prefix_len >= 15 {
                write_vint(&mut suffix_buffer, (prefix_len - 15) as u32);
            }
            if suffix_len >= 16 {
                write_vint(&mut suffix_buffer, (suffix_len - 16) as u32);
            }
            suffix_buffer.extend_from_slice(&term[prefix_len..]);
        }
        max_length = max_length.max(term.len() as i32);
        previous = term;
    }
    if !suffix_buffer.is_empty() {
        let l = flush_block(data, &dict_bytes, &suffix_buffer, &mut lz4_ht);
        max_block_length = max_block_length.max(l);
    }

    // Block addresses: DM meta → .dvm, packed data → a side buffer that
    // lands in .dvd *after* the terms bytes (matching the Java order).
    let mut address_buffer: Vec<u8> = Vec::new();
    crate::monotonic::write(
        &dm_values,
        crate::consts::DIRECT_MONOTONIC_BLOCK_SHIFT,
        meta,
        &mut address_buffer,
    )?;
    meta.extend_from_slice(&max_length.to_le_bytes());
    meta.extend_from_slice(&max_block_length.to_le_bytes());
    meta.extend_from_slice(&(start as i64).to_le_bytes());
    meta.extend_from_slice(&((dvd_base + data.len() as u64 - start) as i64).to_le_bytes());
    let addr_start = dvd_base + data.len() as u64;
    data.extend_from_slice(&address_buffer);
    meta.extend_from_slice(&(addr_start as i64).to_le_bytes());
    meta.extend_from_slice(&((dvd_base + data.len() as u64 - addr_start) as i64).to_le_bytes());

    write_terms_index(meta, data, sorted_terms, dvd_base)
}

/// Port of Bearing's `write_terms_index` (reverse index for term lookup).
fn write_terms_index(
    meta: &mut Vec<u8>,
    data: &mut Vec<u8>,
    sorted_terms: &[&[u8]],
    dvd_base: u64,
) -> Result<()> {
    const REVERSE_SHIFT: i32 = 10;
    const REVERSE_MASK: usize = (1 << REVERSE_SHIFT) - 1;
    meta.extend_from_slice(&REVERSE_SHIFT.to_le_bytes());

    let start = dvd_base + data.len() as u64;
    let mut dm_values: Vec<i64> = Vec::new();
    let mut previous: Option<&[u8]> = None;
    let mut offset: i64 = 0;
    for (ord, term) in sorted_terms.iter().enumerate() {
        if ord & REVERSE_MASK == 0 {
            dm_values.push(offset);
            let key_len = if ord == 0 {
                0
            } else {
                sort_key_length(previous.expect("prior boundary"), term)
            };
            offset += key_len as i64;
            data.extend_from_slice(&term[..key_len]);
        }
        if ord & REVERSE_MASK == REVERSE_MASK {
            previous = Some(term);
        }
    }
    dm_values.push(offset);

    let mut address_buffer: Vec<u8> = Vec::new();
    crate::monotonic::write(
        &dm_values,
        crate::consts::DIRECT_MONOTONIC_BLOCK_SHIFT,
        meta,
        &mut address_buffer,
    )?;
    meta.extend_from_slice(&(start as i64).to_le_bytes());
    meta.extend_from_slice(&((dvd_base + data.len() as u64 - start) as i64).to_le_bytes());
    let addr_start = dvd_base + data.len() as u64;
    data.extend_from_slice(&address_buffer);
    meta.extend_from_slice(&(addr_start as i64).to_le_bytes());
    meta.extend_from_slice(&((dvd_base + data.len() as u64 - addr_start) as i64).to_le_bytes());
    Ok(())
}

/// Encode one SORTED field: per-doc single term (or absent). Terms are
/// dictionary-deduplicated and byte-sorted; ords ride `write_values`.
pub fn encode_sorted_field(
    encoder: &dyn NumericEncoder,
    field_number: i32,
    docs: &[u32],
    terms_per_doc: &[&[u8]],
    max_doc: u32,
    dvd_base: u64,
) -> Result<EncodedField> {
    if docs.len() != terms_per_doc.len() {
        return Err(Error::invalid("docs and terms length mismatch"));
    }
    let mut sorted: Vec<&[u8]> = terms_per_doc.to_vec();
    sorted.sort_unstable();
    sorted.dedup();

    let ords: Vec<i64> = terms_per_doc
        .iter()
        .map(|t| sorted.binary_search(t).expect("term present") as i64)
        .collect();

    let mut meta = Vec::new();
    let mut data = Vec::new();
    meta.extend_from_slice(&field_number.to_le_bytes());
    meta.push(crate::consts::TYPE_SORTED);
    write_values(&mut meta, &mut data, &ValuesIn::Flat(&ords), docs, max_doc, dvd_base, encoder)?;
    add_terms_dict(&mut meta, &mut data, &sorted, dvd_base)?;
    Ok(EncodedField { meta, data })
}

/// Encode one BINARY field (per-doc byte payloads).
pub fn encode_binary_field(
    field_number: i32,
    docs: &[u32],
    values: &[&[u8]],
    max_doc: u32,
    dvd_base: u64,
) -> Result<EncodedField> {
    if docs.len() != values.len() {
        return Err(Error::invalid("docs and values length mismatch"));
    }
    let mut meta = Vec::new();
    let mut data = Vec::new();
    meta.extend_from_slice(&field_number.to_le_bytes());
    meta.push(crate::consts::TYPE_BINARY);

    let start = dvd_base + data.len() as u64;
    meta.extend_from_slice(&(start as i64).to_le_bytes());
    let mut min_len = i32::MAX;
    let mut max_len = 0i32;
    for v in values {
        min_len = min_len.min(v.len() as i32);
        max_len = max_len.max(v.len() as i32);
        data.extend_from_slice(v);
    }
    let n = docs.len() as u32;
    if n == 0 {
        min_len = 0;
    }
    meta.extend_from_slice(&((dvd_base + data.len() as u64 - start) as i64).to_le_bytes());

    write_docs_with_field(&mut meta, &mut data, docs, max_doc, dvd_base)?;
    meta.extend_from_slice(&(n as i32).to_le_bytes());
    meta.extend_from_slice(&min_len.to_le_bytes());
    meta.extend_from_slice(&max_len.to_le_bytes());

    if max_len > min_len {
        let mut cumulative = 0i64;
        let mut addr: Vec<i64> = Vec::with_capacity(values.len() + 1);
        for v in values {
            addr.push(cumulative);
            cumulative += v.len() as i64;
        }
        addr.push(cumulative);
        write_addresses(&mut meta, &mut data, &addr, dvd_base)?;
    }
    Ok(EncodedField { meta, data })
}

/// The DISI docs-with-field indicator, shared across field types.
fn write_docs_with_field(
    meta: &mut Vec<u8>,
    data: &mut Vec<u8>,
    docs: &[u32],
    max_doc: u32,
    dvd_base: u64,
) -> Result<()> {
    let n = docs.len() as u32;
    if n == 0 {
        meta.extend_from_slice(&(-2i64).to_le_bytes());
        meta.extend_from_slice(&0i64.to_le_bytes());
        meta.extend_from_slice(&(-1i16).to_le_bytes());
        meta.push(0xFF);
    } else if n == max_doc {
        meta.extend_from_slice(&(-1i64).to_le_bytes());
        meta.extend_from_slice(&0i64.to_le_bytes());
        meta.extend_from_slice(&(-1i16).to_le_bytes());
        meta.push(0xFF);
    } else {
        let offset = dvd_base + data.len() as u64;
        meta.extend_from_slice(&(offset as i64).to_le_bytes());
        let before = data.len();
        let jtec = disi::write_bit_set(docs, data)?;
        meta.extend_from_slice(&((data.len() - before) as i64).to_le_bytes());
        meta.extend_from_slice(&jtec.to_le_bytes());
        meta.push(disi::DEFAULT_DENSE_RANK_POWER);
    }
    Ok(())
}

/// Addresses block: `offset i64, vint blockShift, DM meta, length i64`
/// (meta), packed deltas appended to `.dvd`.
fn write_addresses(
    meta: &mut Vec<u8>,
    data: &mut Vec<u8>,
    addresses: &[i64],
    dvd_base: u64,
) -> Result<()> {
    let start = dvd_base + data.len() as u64;
    meta.extend_from_slice(&(start as i64).to_le_bytes());
    lucene_arrow_core::cursor::write_vint(meta, crate::consts::DIRECT_MONOTONIC_BLOCK_SHIFT);
    let mut buffer = Vec::new();
    crate::monotonic::write(addresses, crate::consts::DIRECT_MONOTONIC_BLOCK_SHIFT, meta, &mut buffer)?;
    data.extend_from_slice(&buffer);
    meta.extend_from_slice(&((dvd_base + data.len() as u64 - start) as i64).to_le_bytes());
    Ok(())
}

/// Encode one SORTED_NUMERIC field (values sorted within each doc, per
/// the Lucene contract; single-valued degrades to the NUMERIC layout).
pub fn encode_sorted_numeric_field(
    encoder: &dyn NumericEncoder,
    field_number: i32,
    docs: &[u32],
    values_per_doc: &[Vec<i64>],
    max_doc: u32,
    dvd_base: u64,
) -> Result<EncodedField> {
    if docs.len() != values_per_doc.len() {
        return Err(Error::invalid("docs and values length mismatch"));
    }
    let mut sorted_per_doc: Vec<Vec<i64>> = values_per_doc.to_vec();
    for v in &mut sorted_per_doc {
        v.sort_unstable();
    }
    let all: Vec<i64> = sorted_per_doc.iter().flatten().copied().collect();
    let n = docs.len();

    let mut meta = Vec::new();
    let mut data = Vec::new();
    meta.extend_from_slice(&field_number.to_le_bytes());
    meta.push(crate::consts::TYPE_SORTED_NUMERIC);
    write_values(&mut meta, &mut data, &ValuesIn::Flat(&all), docs, max_doc, dvd_base, encoder)?;
    meta.extend_from_slice(&(n as i32).to_le_bytes());
    if all.len() > n {
        let mut cumulative = 0i64;
        let mut addr = Vec::with_capacity(n + 1);
        for v in &sorted_per_doc {
            addr.push(cumulative);
            cumulative += v.len() as i64;
        }
        addr.push(cumulative);
        write_addresses(&mut meta, &mut data, &addr, dvd_base)?;
    }
    Ok(EncodedField { meta, data })
}

/// Encode one SORTED_SET field (terms deduplicated + byte-sorted within
/// each doc; single-valued degrades to the SORTED layout under flag 0).
pub fn encode_sorted_set_field(
    encoder: &dyn NumericEncoder,
    field_number: i32,
    docs: &[u32],
    terms_per_doc: &[Vec<Vec<u8>>],
    max_doc: u32,
    dvd_base: u64,
) -> Result<EncodedField> {
    if docs.len() != terms_per_doc.len() {
        return Err(Error::invalid("docs and terms length mismatch"));
    }
    // Global dictionary.
    let mut dict: Vec<&[u8]> = terms_per_doc.iter().flatten().map(|t| t.as_slice()).collect();
    dict.sort_unstable();
    dict.dedup();

    // Per-doc ords, deduped + sorted (== term byte order).
    let per_doc_ords: Vec<Vec<i64>> = terms_per_doc
        .iter()
        .map(|terms| {
            let mut ords: Vec<i64> = terms
                .iter()
                .map(|t| dict.binary_search(&t.as_slice()).expect("term present") as i64)
                .collect();
            ords.sort_unstable();
            ords.dedup();
            ords
        })
        .collect();
    let multi = per_doc_ords.iter().any(|o| o.len() > 1);

    let mut meta = Vec::new();
    let mut data = Vec::new();
    meta.extend_from_slice(&field_number.to_le_bytes());
    meta.push(crate::consts::TYPE_SORTED_SET);
    meta.push(u8::from(multi));

    let all: Vec<i64> = per_doc_ords.iter().flatten().copied().collect();
    write_values(&mut meta, &mut data, &ValuesIn::Flat(&all), docs, max_doc, dvd_base, encoder)?;
    if multi {
        meta.extend_from_slice(&(docs.len() as i32).to_le_bytes());
        let mut cumulative = 0i64;
        let mut addr = Vec::with_capacity(docs.len() + 1);
        for o in &per_doc_ords {
            addr.push(cumulative);
            cumulative += o.len() as i64;
        }
        addr.push(cumulative);
        write_addresses(&mut meta, &mut data, &addr, dvd_base)?;
    }
    add_terms_dict(&mut meta, &mut data, &dict, dvd_base)?;
    Ok(EncodedField { meta, data })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gcd_matches_java_mathutil() {
        assert_eq!(gcd_compute(0, 7), 7);
        assert_eq!(gcd_compute(12, 18), 6);
        assert_eq!(gcd_compute(-25, 100), 25);
        assert_eq!(gcd_compute(1, 999), 1);
    }

    #[test]
    fn rejects_unsorted_docs() {
        assert!(encode_numeric_field(0, &[3, 1], &[10, 20], 10, 0).is_err());
    }
}
