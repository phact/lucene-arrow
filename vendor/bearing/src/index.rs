// SPDX-License-Identifier: Apache-2.0

//! Segment metadata and index reading.
//!
//! Types for segment-level metadata ([`FieldInfo`], [`FieldInfos`],
//! [`SegmentInfo`], [`SegmentCommitInfo`]) and readers for opening and
//! querying indexes ([`directory_reader`], [`segment_reader`]).

pub mod config;
pub mod field;
pub mod field_infos;
pub(crate) mod index_file_names;
pub(crate) mod pipeline;
pub mod segment;
pub mod segment_info;
pub mod writer;

pub use segment_infos::{SegmentEntry, SegmentInfosRead};
pub mod directory_reader;
pub mod doc_values_iterators;
pub mod segment_infos;
pub mod segment_reader;
pub mod terms;

// Re-exports so `use crate::index::{FieldInfo, ...}` continues to work.
pub use field_infos::{FieldInfo, FieldInfos, PointDimensionConfig};
pub use segment_info::{SegmentCommitInfo, SegmentInfo};
