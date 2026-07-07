// SPDX-License-Identifier: Apache-2.0

//! Decode/Encode plans (SPEC §6) — the shared core everything hangs off.
//!
//! A plan is a *serializable* description of how byte ranges become a typed
//! Arrow column (decode) or how an Arrow column becomes codec payload blocks
//! (encode). Plans are produced by the metadata-parsing stage (CPU, cheap,
//! sequential — the cuIO pattern) and consumed by executors (CPU reference or
//! GPU), which must produce **bit-identical** output for the same plan.
//!
//! Plans are internal and versioned separately from wire frames (SPEC §4):
//! [`PLAN_VERSION`] may move freely; `lucene.frame_version` may not.

use arrow_schema::DataType;
use serde::{Deserialize, Serialize};

/// Internal plan-format version. Not a wire contract.
pub const PLAN_VERSION: u32 = 1;

/// Identifies one column of one segment: Lucene field number plus name.
///
/// The name is what appears in the Arrow schema; the number is what the
/// Lucene metadata refers to. Both are carried so neither side needs a
/// lookup table.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FieldId {
    pub number: i32,
    pub name: String,
}

impl FieldId {
    pub fn new(number: i32, name: impl Into<String>) -> Self {
        FieldId { number, name: name.into() }
    }
}

/// How to decode one contiguous run of packed values.
///
/// Offsets are absolute byte offsets into the segment's data file (`.dvd`,
/// `.vec`, …) — never into a compound-file slice; the planner resolves `.cfs`
/// slices before emitting blocks, so executors stay slice-unaware.
/// Every packed variant carries `values`: how many logical values the block
/// yields. A `bit_width` of 0 means "no payload bytes; every value equals
/// the epilogue base" (Lucene's constant blocks).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BlockDecode {
    /// PackedInts / DIRECT: `value = unpack(bits)`.
    Direct { offset: u64, len: u64, bit_width: u8, values: u64 },
    /// `value = base + unpack(bits)` (per-block base; blocks self-contained).
    DeltaPacked { offset: u64, len: u64, bit_width: u8, base: i64, values: u64 },
    /// `value = base + gcd * unpack(bits)`.
    GcdPacked { offset: u64, len: u64, bit_width: u8, base: i64, gcd: i64, values: u64 },
    /// `value = table[unpack(bits)]`. The ≤256-entry table lives in `.dvm`
    /// metadata, not the data file, so the plan inlines it (deviation from
    /// the SPEC §6 sketch's `table_off`: inlining keeps executors
    /// single-file).
    Table { offset: u64, len: u64, bit_width: u8, table: Vec<i64>, values: u64 },
    /// DirectMonotonic: `value = base + round(avg * i) + unpack(bits)_i`
    /// (zig-zag deltas). Used for List offsets / addresses blocks.
    Monotonic { offset: u64, len: u64, bit_width: u8, base: i64, avg: f32, values: u64 },
    /// Ordinals for SORTED/SORTED_SET: unpack + (optionally) fused gather.
    Ordinals { offset: u64, len: u64, bit_width: u8, values: u64 },
    /// Verbatim bytes (flat vectors): DMA/memcpy, no kernel.
    Raw { offset: u64, len: u64 },
}

impl BlockDecode {
    pub fn byte_range(&self) -> (u64, u64) {
        match *self {
            BlockDecode::Direct { offset, len, .. }
            | BlockDecode::DeltaPacked { offset, len, .. }
            | BlockDecode::GcdPacked { offset, len, .. }
            | BlockDecode::Table { offset, len, .. }
            | BlockDecode::Monotonic { offset, len, .. }
            | BlockDecode::Ordinals { offset, len, .. }
            | BlockDecode::Raw { offset, len } => (offset, len),
        }
    }

    /// Logical values this block yields (0 for `Raw`, which is byte-oriented).
    pub fn value_count(&self) -> u64 {
        match *self {
            BlockDecode::Direct { values, .. }
            | BlockDecode::DeltaPacked { values, .. }
            | BlockDecode::GcdPacked { values, .. }
            | BlockDecode::Table { values, .. }
            | BlockDecode::Monotonic { values, .. }
            | BlockDecode::Ordinals { values, .. } => values,
            BlockDecode::Raw { .. } => 0,
        }
    }
}

/// IndexedDISI jump-table description for sparse fields (SPEC §7.1: validity
/// means "field absent for this doc", never deletion).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DisiPlan {
    /// Absolute byte offset of the DISI blocks in the data file.
    pub offset: u64,
    pub len: u64,
    /// Number of jump-table entries (−1 ⇒ dense/all as encoded by Lucene).
    pub jump_table_entry_count: i16,
    /// Rank block size for DENSE blocks = 2^dense_rank_power (−1 if absent).
    pub dense_rank_power: u8,
    /// Number of documents that carry a value ("cost" in Lucene terms).
    pub num_values: u64,
}

/// How many documents the plan spans and which carry values.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Coverage {
    /// Every doc in `[0, num_docs)` has a value.
    Dense { num_docs: u32 },
    /// Only `disi.num_values` docs have values; membership via DISI.
    Sparse { num_docs: u32, disi: DisiPlan },
    /// No doc has a value (field absent in this segment → all-null column).
    Empty { num_docs: u32 },
}

/// A serializable description of how to turn byte ranges into one typed
/// Arrow column for one segment (SPEC §6).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecodePlan {
    pub plan_version: u32,
    pub column: FieldId,
    /// Data file the offsets refer to (e.g. `_0.dvd`), relative to the
    /// segment directory.
    pub file: String,
    pub arrow_type: DataType,
    pub blocks: Vec<BlockDecode>,
    pub coverage: Coverage,
    /// Total values across `blocks` (== num_docs when dense; == DISI cost
    /// when sparse).
    pub num_values: u64,
}

impl DecodePlan {
    /// Sum of payload bytes the executor must fetch (excludes DISI).
    pub fn payload_bytes(&self) -> u64 {
        self.blocks.iter().map(|b| b.byte_range().1).sum()
    }
}

/// Per-block encoding choice on the write path (SPEC §6, §11.7). The
/// executor emits packed payload; the codec layer wraps it in Lucene
/// framing and writes the entry metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BlockEncode {
    /// All values equal `base`: zero payload bits.
    Constant { base: i64 },
    /// Pack `(value - base) / gcd` at `bit_width` (gcd == 1, base == 0 for
    /// plain DIRECT; base per-block for delta).
    Pack { bit_width: u8, base: i64, gcd: i64 },
    /// Pack indexes into `table` at `bit_width`.
    Table { bit_width: u8, table: Vec<i64> },
    /// Verbatim bytes (flat vectors, binary payloads).
    Raw,
}

/// A serializable description of how one Arrow column becomes codec payload
/// for one output segment. Mirrors [`DecodePlan`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EncodePlan {
    pub plan_version: u32,
    pub column: FieldId,
    pub arrow_type: DataType,
    /// One entry per value block, in doc order. Value-block size is a codec
    /// constant (Lucene90 doc values: 16384 values) — the planner slices.
    pub blocks: Vec<BlockEncode>,
    /// Docs with a value; None ⇒ dense.
    pub sparse: Option<SparseEncode>,
    pub num_docs: u32,
    pub num_values: u64,
}

/// Sparse membership on the write side: which docids carry values.
/// The codec layer serializes this as IndexedDISI.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SparseEncode {
    /// Sorted docids that have a value. Kept as a plan-level abstraction;
    /// executors produce it from Arrow validity in one popcount/scan pass.
    pub docs_with_value: Vec<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_round_trips_through_serde() {
        let plan = DecodePlan {
            plan_version: PLAN_VERSION,
            column: FieldId::new(3, "price"),
            file: "_0.dvd".into(),
            arrow_type: DataType::Int64,
            blocks: vec![
                BlockDecode::GcdPacked {
                    offset: 64,
                    len: 1024,
                    bit_width: 4,
                    base: 100,
                    gcd: 25,
                    values: 2048,
                },
                BlockDecode::Direct { offset: 1088, len: 2048, bit_width: 12, values: 2048 },
            ],
            coverage: Coverage::Dense { num_docs: 4096 },
            num_values: 4096,
        };
        let json = serde_json::to_string(&plan).unwrap();
        let back: DecodePlan = serde_json::from_str(&json).unwrap();
        assert_eq!(plan, back);
        assert_eq!(plan.payload_bytes(), 3072);
    }
}
