// SPDX-License-Identifier: Apache-2.0

//! Whole-file assembly for `.dvm`/`.dvd` (headers, sentinel, footers).
//!
//! Container framing belongs to the codec layer in the final architecture
//! (SPEC §10.5); this small builder exists so doc-values round-trip tests
//! and the early write path can produce complete, checksum-valid files
//! without the full segment writer.

use crate::consts;
use crate::write::{NumericEncoder, encode_numeric_field, encode_numeric_field_with};
use lucene_arrow_core::Result;
use lucene_arrow_core::cursor::{SEGMENT_ID_LENGTH, write_footer, write_index_header};

/// Builds one `.dvm` + `.dvd` pair field by field.
pub struct DocValuesFileBuilder {
    dvm: Vec<u8>,
    dvd: Vec<u8>,
}

impl DocValuesFileBuilder {
    /// `suffix` is the per-field format segment suffix Lucene uses for
    /// these files (e.g. `"Lucene90_0"`); empty when unwrapped.
    pub fn new(segment_id: &[u8; SEGMENT_ID_LENGTH], suffix: &str) -> Self {
        let mut dvm = Vec::new();
        let mut dvd = Vec::new();
        write_index_header(&mut dvm, consts::META_CODEC, consts::VERSION, segment_id, suffix);
        write_index_header(&mut dvd, consts::DATA_CODEC, consts::VERSION, segment_id, suffix);
        DocValuesFileBuilder { dvm, dvd }
    }

    /// Append one NUMERIC field. Fields must be added in field-number order
    /// (Lucene writes them ordered; readers tolerate but CheckIndex cares).
    pub fn add_numeric(
        &mut self,
        field_number: i32,
        docs: &[u32],
        values: &[i64],
        max_doc: u32,
    ) -> Result<()> {
        let encoded =
            encode_numeric_field(field_number, docs, values, max_doc, self.dvd.len() as u64)?;
        self.dvm.extend_from_slice(&encoded.meta);
        self.dvd.extend_from_slice(&encoded.data);
        Ok(())
    }

    /// [`add_numeric`](Self::add_numeric) with an explicit stats+pack
    /// executor (CPU reference or the GPU packer).
    pub fn add_numeric_with(
        &mut self,
        encoder: &dyn NumericEncoder,
        field_number: i32,
        docs: &[u32],
        values: &[i64],
        max_doc: u32,
    ) -> Result<()> {
        let encoded = encode_numeric_field_with(
            encoder,
            field_number,
            docs,
            values,
            max_doc,
            self.dvd.len() as u64,
        )?;
        self.dvm.extend_from_slice(&encoded.meta);
        self.dvd.extend_from_slice(&encoded.data);
        Ok(())
    }

    /// Append one **dense** NUMERIC field straight from Arrow batch
    /// slices — the zero-copy ingest lane (no docs array, no host concat).
    pub fn add_numeric_dense_chunks(
        &mut self,
        encoder: &dyn NumericEncoder,
        field_number: i32,
        chunks: &[&[i64]],
        max_doc: u32,
    ) -> Result<()> {
        let encoded = crate::write::encode_numeric_field_dense_chunks(
            encoder,
            field_number,
            chunks,
            max_doc,
            self.dvd.len() as u64,
        )?;
        self.dvm.extend_from_slice(&encoded.meta);
        self.dvd.extend_from_slice(&encoded.data);
        Ok(())
    }

    /// Append one SORTED field (single term per doc-with-value).
    pub fn add_sorted_with(
        &mut self,
        encoder: &dyn NumericEncoder,
        field_number: i32,
        docs: &[u32],
        terms_per_doc: &[&[u8]],
        max_doc: u32,
    ) -> Result<()> {
        let encoded = crate::write::encode_sorted_field(
            encoder,
            field_number,
            docs,
            terms_per_doc,
            max_doc,
            self.dvd.len() as u64,
        )?;
        self.dvm.extend_from_slice(&encoded.meta);
        self.dvd.extend_from_slice(&encoded.data);
        Ok(())
    }

    /// Append one BINARY field.
    pub fn add_binary(
        &mut self,
        field_number: i32,
        docs: &[u32],
        values: &[&[u8]],
        max_doc: u32,
    ) -> Result<()> {
        let e = crate::write::encode_binary_field(field_number, docs, values, max_doc, self.dvd.len() as u64)?;
        self.dvm.extend_from_slice(&e.meta);
        self.dvd.extend_from_slice(&e.data);
        Ok(())
    }

    /// Append one SORTED_NUMERIC field (multi-valued allowed).
    pub fn add_sorted_numeric_with(
        &mut self,
        encoder: &dyn NumericEncoder,
        field_number: i32,
        docs: &[u32],
        values_per_doc: &[Vec<i64>],
        max_doc: u32,
    ) -> Result<()> {
        let e = crate::write::encode_sorted_numeric_field(
            encoder, field_number, docs, values_per_doc, max_doc, self.dvd.len() as u64,
        )?;
        self.dvm.extend_from_slice(&e.meta);
        self.dvd.extend_from_slice(&e.data);
        Ok(())
    }

    /// Append one SORTED_SET field (multi-valued allowed).
    pub fn add_sorted_set_with(
        &mut self,
        encoder: &dyn NumericEncoder,
        field_number: i32,
        docs: &[u32],
        terms_per_doc: &[Vec<Vec<u8>>],
        max_doc: u32,
    ) -> Result<()> {
        let e = crate::write::encode_sorted_set_field(
            encoder, field_number, docs, terms_per_doc, max_doc, self.dvd.len() as u64,
        )?;
        self.dvm.extend_from_slice(&e.meta);
        self.dvd.extend_from_slice(&e.data);
        Ok(())
    }

    /// Close both files: `.dvm` sentinel + footers. Returns `(dvm, dvd)`.
    pub fn finish(mut self) -> (Vec<u8>, Vec<u8>) {
        self.dvm.extend_from_slice(&(-1i32).to_le_bytes());
        write_footer(&mut self.dvm);
        write_footer(&mut self.dvd);
        (self.dvm, self.dvd)
    }
}
