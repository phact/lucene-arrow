// SPDX-License-Identifier: Apache-2.0

//! First-party `.fnm` (Lucene94FieldInfos) parser.
//!
//! Bearing's reader parses-and-discards the vector fields (dimension,
//! encoding, similarity) and doesn't expose the doc-values skip-index
//! flag; both matter to us (SPEC §7.2 vector shapes; skip-index changes
//! `.dvm` entry layout). This parser captures everything. Layout per
//! `Lucene94FieldInfosFormat.write()` (Lucene 10.3.2), cross-checked with
//! Bearing's `reference/formats/lucene94-formats.md`.

use std::collections::BTreeMap;

use lucene_arrow_core::cursor::{Cursor, read_index_header, verify_footer};
use lucene_arrow_core::{Error, Result};

pub const CODEC_NAME: &str = "Lucene94FieldInfos";
pub const FORMAT_START: i32 = 0;
pub const FORMAT_DOCVALUE_SKIPPER: i32 = 2;
pub const FORMAT_CURRENT: i32 = 2;

// fieldBits flags.
const STORE_TERMVECTOR: u8 = 0x01;
const OMIT_NORMS: u8 = 0x02;
const STORE_PAYLOADS: u8 = 0x04;
const SOFT_DELETES: u8 = 0x08;
const PARENT_FIELD: u8 = 0x10;
const DOCVALUES_SKIPPER: u8 = 0x20;

/// One `.fnm` field record, fully captured.
#[derive(Debug, Clone)]
pub struct FnmField {
    pub name: String,
    pub number: u32,
    pub store_term_vector: bool,
    pub omit_norms: bool,
    pub store_payloads: bool,
    pub soft_deletes: bool,
    pub parent_field: bool,
    /// Doc-values skip index present (changes `.dvm` entry layout).
    pub doc_values_skip_index: bool,
    /// 0=NONE, 1=DOCS … 4=DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS.
    pub index_options: u8,
    /// 0=NONE, 1=NUMERIC, 2=BINARY, 3=SORTED, 4=SORTED_SET, 5=SORTED_NUMERIC
    /// (the on-disk order — note it differs from Bearing's enum order).
    pub doc_values_type: u8,
    pub doc_values_gen: i64,
    pub attributes: BTreeMap<String, String>,
    pub point_dimension_count: u32,
    pub point_index_dimension_count: u32,
    pub point_num_bytes: u32,
    /// 0 when the field has no vectors.
    pub vector_dimension: u32,
    /// `VectorEncoding` ordinal (0=BYTE, 1=FLOAT32); meaningful only when
    /// `vector_dimension > 0`.
    pub vector_encoding: u8,
    /// `VectorSimilarityFunction` ordinal (0=EUCLIDEAN, 1=DOT_PRODUCT,
    /// 2=COSINE, 3=MAXIMUM_INNER_PRODUCT).
    pub vector_similarity: u8,
}

/// Parse a whole `.fnm` file (header + footer verified).
pub fn parse(bytes: &[u8]) -> Result<Vec<FnmField>> {
    let header = read_index_header(bytes, CODEC_NAME, FORMAT_START, FORMAT_CURRENT)?;
    verify_footer(bytes)?;
    let version = header.version;

    let mut c = Cursor::at(bytes, header.length);
    let count = c.vint()?;
    if count < 0 {
        return Err(Error::corrupt("negative field count"));
    }
    let mut fields = Vec::with_capacity(count as usize);

    for _ in 0..count {
        let name = c.string()?;
        let number = c.vint()?;
        if number < 0 {
            return Err(Error::corrupt(format!("negative field number for {name:?}")));
        }
        let bits = c.u8()?;
        if bits & 0xC0 != 0 {
            return Err(Error::corrupt(format!("reserved field bits set: {bits:#04x}")));
        }
        let index_options = c.u8()?;
        if index_options > 4 {
            return Err(Error::corrupt(format!("invalid index options byte {index_options}")));
        }
        let doc_values_type = c.u8()?;
        if doc_values_type > 5 {
            return Err(Error::corrupt(format!("invalid doc values byte {doc_values_type}")));
        }
        let skip_index_byte = if version >= FORMAT_DOCVALUE_SKIPPER { c.u8()? } else { 0 };
        let doc_values_gen = c.le_i64()?;

        let attr_count = c.vint()?;
        let mut attributes = BTreeMap::new();
        for _ in 0..attr_count {
            let k = c.string()?;
            let v = c.string()?;
            attributes.insert(k, v);
        }

        let point_dimension_count = c.vint()? as u32;
        let (point_index_dimension_count, point_num_bytes) = if point_dimension_count != 0 {
            (c.vint()? as u32, c.vint()? as u32)
        } else {
            (0, 0)
        };
        let vector_dimension = c.vint()? as u32;
        let vector_encoding = c.u8()?;
        let vector_similarity = c.u8()?;

        fields.push(FnmField {
            name,
            number: number as u32,
            store_term_vector: bits & STORE_TERMVECTOR != 0,
            omit_norms: bits & OMIT_NORMS != 0,
            store_payloads: bits & STORE_PAYLOADS != 0,
            soft_deletes: bits & SOFT_DELETES != 0,
            parent_field: bits & PARENT_FIELD != 0,
            doc_values_skip_index: bits & DOCVALUES_SKIPPER != 0 || skip_index_byte != 0,
            index_options,
            doc_values_type,
            doc_values_gen,
            attributes,
            point_dimension_count,
            point_index_dimension_count,
            point_num_bytes,
            vector_dimension,
            vector_encoding,
            vector_similarity,
        });
    }

    Ok(fields)
}
