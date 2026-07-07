// SPDX-License-Identifier: Apache-2.0

//! Flat vectors ⇄ Arrow (SPEC §7.2, §10.4; format: Lucene99FlatVectorsFormat,
//! files `.vemf` meta + `.vec` data, traced from Lucene 10.3.2 sources).
//!
//! v1 reads/writes **flat** storage only — the HNSW graph (`.vem`/`.vex`)
//! is a serving concern we deliberately ignore (SPEC §7.7); flat vectors
//! give exact answers and DMA straight into cuVS/cuBLAS. `Raw` plan blocks
//! mean no decode kernel at all — the GPU path is a copy (SPEC §11.6).

pub mod file;
pub mod hnsw;
pub mod jvector;
pub mod read;
pub mod write;

/// Lucene99FlatVectorsFormat constants.
pub mod consts {
    pub const META_CODEC: &str = "Lucene99FlatVectorsFormatMeta";
    pub const DATA_CODEC: &str = "Lucene99FlatVectorsFormatData";
    pub const META_EXTENSION: &str = "vemf";
    pub const DATA_EXTENSION: &str = "vec";
    pub const VERSION: i32 = 0;
    pub const DIRECT_MONOTONIC_BLOCK_SHIFT: u32 = 16;
}

/// `org.apache.lucene.index.VectorEncoding` ordinals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorEncoding {
    Byte,
    Float32,
}

impl VectorEncoding {
    pub fn from_ordinal(ord: i32) -> Option<Self> {
        match ord {
            0 => Some(VectorEncoding::Byte),
            1 => Some(VectorEncoding::Float32),
            _ => None,
        }
    }
    pub fn ordinal(self) -> i32 {
        match self {
            VectorEncoding::Byte => 0,
            VectorEncoding::Float32 => 1,
        }
    }
    /// Bytes per dimension.
    pub fn width(self) -> usize {
        match self {
            VectorEncoding::Byte => 1,
            VectorEncoding::Float32 => 4,
        }
    }
}

/// `org.apache.lucene.index.VectorSimilarityFunction` ordinals — carried
/// into Arrow field metadata as `lucene.vector.similarity` (SPEC §7.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Similarity {
    Euclidean,
    DotProduct,
    Cosine,
    MaximumInnerProduct,
}

impl Similarity {
    pub fn from_ordinal(ord: i32) -> Option<Self> {
        match ord {
            0 => Some(Similarity::Euclidean),
            1 => Some(Similarity::DotProduct),
            2 => Some(Similarity::Cosine),
            3 => Some(Similarity::MaximumInnerProduct),
            _ => None,
        }
    }
    pub fn ordinal(self) -> i32 {
        match self {
            Similarity::Euclidean => 0,
            Similarity::DotProduct => 1,
            Similarity::Cosine => 2,
            Similarity::MaximumInnerProduct => 3,
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Similarity::Euclidean => "euclidean",
            Similarity::DotProduct => "dot_product",
            Similarity::Cosine => "cosine",
            Similarity::MaximumInnerProduct => "max_inner_product",
        }
    }
}
