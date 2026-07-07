// SPDX-License-Identifier: Apache-2.0

//! The `lucene.*` metadata namespace — [CONTRACT] keys (SPEC §4, §7.1, §10.2).
//!
//! All lucene-arrow metadata lives under this reserved prefix. Clients must
//! tolerate unknown keys; servers echo effective options (SPEC §8.3) so
//! nothing ever depends on an implicit default. Key *names* here are
//! contract; many *values* are [DEFAULT] and free to change.

/// Wire frame version. Bumped only for breaking changes to the frame
/// contract (segment-scoped batches, system columns, metadata keys).
pub const FRAME_VERSION: u32 = 1;

// --- Batch-level keys (SPEC §7.1) -----------------------------------------
pub const SEGMENT: &str = "lucene.segment";
pub const SEGMENT_ORD: &str = "lucene.segment_ord";
pub const DOC_LO: &str = "lucene.doc_lo";
pub const DOC_HI: &str = "lucene.doc_hi";
pub const DOC_BASE: &str = "lucene.doc_base";
pub const MAX_DOC: &str = "lucene.max_doc";
pub const LIVE_APPLIED: &str = "lucene.live_applied";
pub const CODEC: &str = "lucene.codec";
pub const FRAME_VERSION_KEY: &str = "lucene.frame_version";

// --- System columns (SPEC §7.1) --------------------------------------------
pub const COL_SEG: &str = "_seg";
pub const COL_DOC: &str = "_doc";
pub const COL_GLOBAL_DOC: &str = "_global_doc";
pub const COL_LIVE: &str = "_live";

// --- Field-level keys (SPEC §7.2, §7.3, §10.2) ------------------------------
pub const FIELD_TYPE: &str = "lucene.type";
pub const FIELD_NAME: &str = "lucene.field";
pub const DICT_MODE: &str = "lucene.dict";
pub const DICT_CARDINALITY: &str = "lucene.dict.cardinality";
pub const MULTI: &str = "lucene.multi";
pub const SOURCE_TYPE: &str = "lucene.source_type";
pub const ALLOW_LOSSY: &str = "lucene.allow_lossy";
pub const SCALE_FACTOR: &str = "lucene.scale_factor";
pub const VECTOR_SIMILARITY: &str = "lucene.vector.similarity";
pub const VECTOR_ENCODING: &str = "lucene.vector.encoding";
pub const VECTOR_QUANT: &str = "lucene.vector.quant";
pub const POINTS: &str = "lucene.points";

/// `lucene.type` values (SPEC §10.2).
pub mod field_type {
    pub const NUMERIC: &str = "numeric";
    pub const SORTED: &str = "sorted";
    pub const SORTED_SET: &str = "sorted_set";
    pub const SORTED_NUMERIC: &str = "sorted_numeric";
    pub const BINARY: &str = "binary";
    pub const VECTOR: &str = "vector";
}

/// `lucene.dict` values (SPEC §7.3).
pub mod dict_mode {
    pub const GLOBAL: &str = "global";
    pub const SEGMENT: &str = "segment";
    pub const NONE: &str = "none";
}

/// Row modes — caller must choose; no implicit default (SPEC §7.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowMode {
    /// Deletes dropped server-side.
    Compact,
    /// One row per docid + `_live: Boolean`.
    Positional,
}

impl RowMode {
    pub fn parse(s: &str) -> Option<RowMode> {
        match s {
            "compact" => Some(RowMode::Compact),
            "positional" => Some(RowMode::Positional),
            _ => None,
        }
    }
    pub fn as_str(&self) -> &'static str {
        match self {
            RowMode::Compact => "compact",
            RowMode::Positional => "positional",
        }
    }
}
