// SPDX-License-Identifier: Apache-2.0

//! Flat vector encoding (SPEC §10.4): packed vectors + `.vemf` entry,
//! mirroring `Lucene99FlatVectorsWriter.writeMeta` +
//! `OrdToDocDISIReaderConfiguration.writeStoredMeta` byte-for-byte.

use crate::{Similarity, VectorEncoding, consts};
use lucene_arrow_core::cursor::{write_vint, write_vlong};
use lucene_arrow_core::{Error, Result};
use lucene_arrow_docvalues::{disi, monotonic};

pub struct EncodedVectorField {
    pub meta: Vec<u8>,
    pub data: Vec<u8>,
}

/// Encode one flat vector field.
///
/// `docs` are the (sorted, unique) docids carrying a vector; `vectors` is
/// the packed payload in the same order (`docs.len() × dim × width`
/// bytes: little-endian f32s or raw i8s). `vec_base` is the `.vec` length
/// before this field's payload lands.
#[allow(clippy::too_many_arguments)]
pub fn encode_flat_field(
    field_number: i32,
    encoding: VectorEncoding,
    similarity: Similarity,
    dim: u32,
    docs: &[u32],
    vectors: &[u8],
    max_doc: u32,
    vec_base: u64,
) -> Result<EncodedVectorField> {
    let count = docs.len();
    if vectors.len() != count * dim as usize * encoding.width() {
        return Err(Error::invalid(format!(
            "vector payload {} bytes != {count} × {dim} × {}",
            vectors.len(),
            encoding.width()
        )));
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

    // Vector payload first: writeMeta records its offset/length.
    let data_offset = vec_base;
    data.extend_from_slice(vectors);
    let data_length = vectors.len() as u64;

    meta.extend_from_slice(&field_number.to_le_bytes());
    meta.extend_from_slice(&encoding.ordinal().to_le_bytes());
    meta.extend_from_slice(&similarity.ordinal().to_le_bytes());
    write_vlong(&mut meta, data_offset);
    write_vlong(&mut meta, data_length);
    write_vint(&mut meta, dim);
    meta.extend_from_slice(&(count as i32).to_le_bytes());

    // OrdToDocDISIReaderConfiguration.writeStoredMeta
    if count == 0 {
        meta.extend_from_slice(&(-2i64).to_le_bytes());
        meta.extend_from_slice(&0i64.to_le_bytes());
        meta.extend_from_slice(&(-1i16).to_le_bytes());
        meta.push(0xFF);
    } else if count as u32 == max_doc {
        meta.extend_from_slice(&(-1i64).to_le_bytes());
        meta.extend_from_slice(&0i64.to_le_bytes());
        meta.extend_from_slice(&(-1i16).to_le_bytes());
        meta.push(0xFF);
    } else {
        let offset = vec_base + data.len() as u64;
        meta.extend_from_slice(&(offset as i64).to_le_bytes());
        let before = data.len();
        let jump_table_entry_count = disi::write_bit_set(docs, &mut data)?;
        meta.extend_from_slice(&((data.len() - before) as i64).to_le_bytes());
        meta.extend_from_slice(&jump_table_entry_count.to_le_bytes());
        meta.push(disi::DEFAULT_DENSE_RANK_POWER);

        // ord → doc map (DirectMonotonic)
        let start = vec_base + data.len() as u64;
        meta.extend_from_slice(&(start as i64).to_le_bytes());
        write_vint(&mut meta, consts::DIRECT_MONOTONIC_BLOCK_SHIFT);
        let ords: Vec<i64> = docs.iter().map(|&d| d as i64).collect();
        monotonic::write(&ords, consts::DIRECT_MONOTONIC_BLOCK_SHIFT, &mut meta, &mut data)?;
        meta.extend_from_slice(&((vec_base + data.len() as u64 - start) as i64).to_le_bytes());
    }

    Ok(EncodedVectorField { meta, data })
}
