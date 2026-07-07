// SPDX-License-Identifier: Apache-2.0

//! Lucene90 doc values ⇄ Arrow (SPEC crates/docvalues; formats per
//! Bearing `reference/formats/lucene90-formats.md` and Lucene
//! `Lucene90DocValuesConsumer/Producer`).
//!
//! Read side: parse `.dvm` metadata into [`DecodePlan`]s — the CPU
//! metadata-parse stage of the cuIO pattern (SPEC §5). Executors (the `cpu`
//! crate today, `gpu` later) turn plans + `.dvd` bytes into Arrow arrays.
//!
//! Write side: encode values into `.dvd`/`.dvm` bytes, byte-identical to
//! Bearing's writer (which is cross-validated against Java Lucene). v1
//! scope: NUMERIC fields, single-block encoding — the same envelope Bearing
//! emits. Multi-block and table encodings are fully supported on *read*
//! (Java Lucene emits them).
//!
//! P1 limitation, documented: fields carrying a doc-values *skip index*
//! (Lucene 10 `DocValuesSkipIndexType != NONE`; opt-in, off by default in
//! ES/OS) prepend extra metadata this parser only handles when the caller
//! flags the field (`DvField::has_skip_index`).
//!
//! [`DecodePlan`]: lucene_arrow_core::plan::DecodePlan

pub mod direct;
pub mod disi;
pub mod file;
pub mod monotonic;
pub mod ordmap;
pub mod read;
pub mod terms;
pub mod write;

/// Lucene90DocValuesFormat constants.
pub mod consts {
    pub const DATA_CODEC: &str = "Lucene90DocValuesData";
    pub const DATA_EXTENSION: &str = "dvd";
    pub const META_CODEC: &str = "Lucene90DocValuesMetadata";
    pub const META_EXTENSION: &str = "dvm";
    pub const VERSION: i32 = 0;

    pub const TYPE_NUMERIC: u8 = 0;
    pub const TYPE_BINARY: u8 = 1;
    pub const TYPE_SORTED: u8 = 2;
    pub const TYPE_SORTED_SET: u8 = 3;
    pub const TYPE_SORTED_NUMERIC: u8 = 4;

    pub const DIRECT_MONOTONIC_BLOCK_SHIFT: u32 = 16;
    pub const NUMERIC_BLOCK_SHIFT: u32 = 14;
    pub const NUMERIC_BLOCK_SIZE: usize = 1 << NUMERIC_BLOCK_SHIFT;
    /// `.dvm` tableSize marker for multi-block ("blockwise") numeric mode.
    pub const MULTI_BLOCK_TABLE_SIZE: i32 = -2 - NUMERIC_BLOCK_SHIFT as i32;
    /// numBitsPerValue marker for multi-block mode.
    pub const MULTI_BLOCK_BPV: u8 = 0xFF;
}
