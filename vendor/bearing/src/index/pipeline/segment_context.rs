// SPDX-License-Identifier: Apache-2.0

use std::fmt;

use crate::store::SharedDirectory;

/// Segment identity and directory access for codec writers at flush time.
///
/// Passed to [`FieldConsumer::flush`](crate::index::pipeline::consumer::FieldConsumer::flush)
/// so consumers can write correctly named and headered files without
/// storing this context themselves.
pub struct SegmentContext {
    /// Shared directory for creating output files.
    pub directory: SharedDirectory,
    /// Segment name used as file name prefix (e.g., "_0").
    pub segment_name: String,
    /// Random 16-byte identifier written into codec file headers.
    pub segment_id: [u8; 16],
}

impl fmt::Debug for SegmentContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SegmentContext")
            .field("segment_name", &self.segment_name)
            .finish_non_exhaustive()
    }
}
