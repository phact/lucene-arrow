// SPDX-License-Identifier: Apache-2.0

//! Segment metadata types.
//!
//! [`SegmentInfo`] describes a segment's identity, document count, and
//! diagnostics. [`SegmentCommitInfo`] pairs a `SegmentInfo` with its
//! field metadata and deletion/update generation counters at a particular
//! commit point.

use std::collections::{HashMap, HashSet};

use crate::index::FieldInfos;
use crate::util::string_helper;

/// Information about a segment in the index.
#[derive(Clone, Debug)]
pub struct SegmentInfo {
    pub name: String,
    pub max_doc: i32,
    pub is_compound_file: bool,
    pub id: [u8; string_helper::ID_LENGTH],
    pub diagnostics: HashMap<String, String>,
    pub attributes: HashMap<String, String>,
    pub files: HashSet<String>,
    pub has_blocks: bool,
}

impl SegmentInfo {
    pub fn new(
        name: String,
        max_doc: i32,
        is_compound_file: bool,
        id: [u8; string_helper::ID_LENGTH],
        diagnostics: HashMap<String, String>,
        attributes: HashMap<String, String>,
    ) -> Self {
        Self {
            name,
            max_doc,
            is_compound_file,
            id,
            diagnostics,
            attributes,
            files: HashSet::new(),
            has_blocks: false,
        }
    }
}

/// Information about a segment at a particular commit point.
#[derive(Clone, Debug)]
pub struct SegmentCommitInfo {
    pub info: SegmentInfo,
    pub field_infos: FieldInfos,
    pub del_count: i32,
    pub soft_del_count: i32,
    pub del_gen: i64,
    pub field_infos_gen: i64,
    pub doc_values_gen: i64,
    pub id: Option<[u8; string_helper::ID_LENGTH]>,
}

impl SegmentCommitInfo {
    pub fn new(
        info: SegmentInfo,
        field_infos: FieldInfos,
        id: Option<[u8; string_helper::ID_LENGTH]>,
    ) -> Self {
        Self {
            info,
            field_infos,
            del_count: 0,
            soft_del_count: 0,
            del_gen: -1,
            field_infos_gen: -1,
            doc_values_gen: -1,
            id,
        }
    }

    /// Returns true if this segment has had deletions applied (del_gen != -1).
    pub fn has_deletions(&self) -> bool {
        self.del_gen != -1
    }

    /// Returns true if this segment has had field info updates (field_infos_gen != -1).
    pub fn has_field_updates(&self) -> bool {
        self.field_infos_gen != -1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::FieldInfos;

    #[test]
    fn test_segment_info() {
        let id = [0u8; 16];
        let si = SegmentInfo::new(
            "_0".to_string(),
            3,
            true,
            id,
            HashMap::new(),
            HashMap::new(),
        );
        assert_eq!(si.name, "_0");
        assert_eq!(si.max_doc, 3);
        assert!(si.is_compound_file);
    }

    #[test]
    fn test_segment_commit_info() {
        let id = [0u8; 16];
        let si = SegmentInfo::new(
            "_0".to_string(),
            3,
            true,
            id,
            HashMap::new(),
            HashMap::new(),
        );
        let fis = FieldInfos::new(vec![]);
        let sci = SegmentCommitInfo::new(si, fis, Some([1u8; 16]));
        assert!(!sci.has_deletions());
        assert!(!sci.has_field_updates());
    }
}
