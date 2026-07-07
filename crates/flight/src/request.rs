// SPDX-License-Identifier: Apache-2.0

//! Read/write job vocabulary (SPEC §8.1, §10.1).
//!
//! Field *names* here are [CONTRACT]; values are [DEFAULT] and resolved
//! server-side, then echoed back (SPEC §8.3). The Flight transport that
//! carries these lands in P4; the types exist now so the contract is
//! testable early.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Which segments of a shard a request covers: `"all"` or an explicit list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SegmentSelector {
    All,
    Names(Vec<String>),
}

impl Serialize for SegmentSelector {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            SegmentSelector::All => s.serialize_str("all"),
            SegmentSelector::Names(names) => names.serialize(s),
        }
    }
}

impl<'de> Deserialize<'de> for SegmentSelector {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Repr {
            Keyword(String),
            Names(Vec<String>),
        }
        match Repr::deserialize(d)? {
            Repr::Keyword(k) if k == "all" => Ok(SegmentSelector::All),
            Repr::Keyword(k) => Err(serde::de::Error::custom(format!(
                "invalid segment selector {k:?}: expected \"all\" or a list"
            ))),
            Repr::Names(names) => Ok(SegmentSelector::Names(names)),
        }
    }
}

/// Per-column dictionary mode (SPEC §7.3): `{"col": mode, "*": mode}`;
/// values are `global | segment | none` ([`dict_mode`]).
///
/// [`dict_mode`]: lucene_arrow_core::meta::dict_mode
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct DictModes(pub BTreeMap<String, String>);

/// `GetFlightInfo` descriptor command (SPEC §8.1). `row_mode` is required —
/// no implicit default (SPEC §7.1); parse it with
/// [`RowMode::parse`](lucene_arrow_core::meta::RowMode::parse).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReadRequest {
    pub path: String,
    pub segments: SegmentSelector,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub columns: Option<Vec<String>>,
    pub row_mode: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dict: Option<DictModes>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub batch_rows: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub executor: Option<String>,
    pub frame_version: u32,
    /// Relation to read (SPEC §7.8): `None`/"docvalues" (default) or
    /// "postings".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub relation: Option<String>,
    /// Indexed field for the postings relation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub postings_field: Option<String>,
}

/// `DoAction: "hydrate"` body (SPEC §7.4): stored columns for explicit
/// `(_seg, _doc)` row addresses. Never part of the scan stream.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HydrateRequest {
    pub path: String,
    /// `(_seg ordinal, local docid)` pairs.
    pub pairs: Vec<(u32, u32)>,
    pub columns: Vec<String>,
}

/// One `index_sort` entry (SPEC §10.1): per-segment sort only.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SortSpec {
    pub field: String,
    /// `"asc" | "desc"`.
    pub order: String,
}

/// `DoPut` descriptor command (SPEC §10.1).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WriteJob {
    pub output_dir: String,
    /// Pinned to `"Lucene103"` in v1 (SPEC §3.2).
    pub codec: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub segment_max_docs: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub segment_max_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index_sort: Option<Vec<SortSpec>>,
    /// `"strict" | "auto"` (SPEC §10.2).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub coercion: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compound: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub executor: Option<String>,
    pub frame_version: u32,
}

/// One file of a flushed segment (SPEC §10.1).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileManifest {
    pub name: String,
    pub bytes: u64,
    pub crc32: u32,
}

/// `PutResult` per flushed segment (SPEC §10.1).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SegmentManifest {
    pub name: String,
    pub max_doc: i32,
    pub files: Vec<FileManifest>,
    #[serde(default)]
    pub field_stats: BTreeMap<String, serde_json::Value>,
    pub wall_ms: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_request_round_trips() {
        let json = r#"{"path":"/x","segments":"all","row_mode":"compact","frame_version":1}"#;
        let req: ReadRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.segments, SegmentSelector::All);
        let back = serde_json::to_string(&req).unwrap();
        let again: ReadRequest = serde_json::from_str(&back).unwrap();
        assert_eq!(req, again);

        let named = r#"{"path":"/x","segments":["_0","_1"],"row_mode":"positional","frame_version":1}"#;
        let req: ReadRequest = serde_json::from_str(named).unwrap();
        assert_eq!(
            req.segments,
            SegmentSelector::Names(vec!["_0".into(), "_1".into()])
        );

        let bad = r#"{"path":"/x","segments":"some","row_mode":"compact","frame_version":1}"#;
        assert!(serde_json::from_str::<ReadRequest>(bad).is_err());
    }
}
