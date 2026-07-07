// SPDX-License-Identifier: Apache-2.0

/// Identity for a segment, assigned by the coordinator.
#[derive(Debug, Clone)]
pub struct SegmentId {
    /// Segment name used as the file name prefix (e.g., "_0", "_1", "_a3").
    /// Derived from a base-36 counter.
    pub name: String,
    /// Random 16-byte identifier written into file headers for integrity.
    pub id: [u8; 16],
}

/// Metadata produced by flushing a segment.
///
/// Returned by `SegmentWorker::flush()` and collected by the coordinator
/// for writing the segments file at commit time.
#[derive(Debug, Clone)]
pub struct FlushedSegment {
    /// Which segment this is.
    pub segment_id: SegmentId,
    /// Number of documents in the segment.
    pub doc_count: i32,
    /// Names of all files written for this segment.
    pub file_names: Vec<String>,
    // TODO: FieldInfos — final field metadata from the registry
}
