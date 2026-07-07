// SPDX-License-Identifier: Apache-2.0

//! Extents & coalescing (SPEC §11.2).
//!
//! An **extent** is a contiguous byte range of one file: one column over one
//! segment (or a slice of it), including interleaved block metadata. Extents
//! are the unit of IO (one read/write call each) and of batched kernel
//! launches (a device-side descriptor table batches many extents into one
//! launch). This is what solves many-small-segments *below* the frame layer.

use serde::{Deserialize, Serialize};

use crate::plan::FieldId;

/// Default extent sizing [DEFAULT]: 4–64 MiB, 4 KiB-aligned.
pub const EXTENT_ALIGN: u64 = 4 * 1024;
pub const EXTENT_MIN_BYTES: u64 = 4 * 1024 * 1024;
pub const EXTENT_MAX_BYTES: u64 = 64 * 1024 * 1024;

/// A contiguous byte range of one file, tagged with where it belongs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Extent {
    /// File name relative to the segment directory (e.g. `_0.dvd`).
    pub file: String,
    pub offset: u64,
    pub len: u64,
    /// Ordinal of the segment within the stream (SPEC `_seg`).
    pub seg_ord: u32,
    pub column: FieldId,
}

impl Extent {
    pub fn end(&self) -> u64 {
        self.offset + self.len
    }
}

/// Coalesce sorted, per-block byte ranges of one (file, column, segment)
/// into IO-sized extents.
///
/// Adjacent or near-adjacent ranges (gap ≤ `max_gap`) merge — re-reading a
/// small gap is cheaper than a second IO. Output extents are clamped to
/// `max_bytes` and aligned down/up to [`EXTENT_ALIGN`] where possible
/// without swallowing unrelated data (alignment only widens, never splits).
pub fn coalesce(
    file: &str,
    seg_ord: u32,
    column: &FieldId,
    ranges: &[(u64, u64)],
    max_gap: u64,
    max_bytes: u64,
) -> Vec<Extent> {
    let mut extents: Vec<Extent> = Vec::new();
    let mut cur: Option<(u64, u64)> = None; // (offset, end)

    let push = |off: u64, end: u64, out: &mut Vec<Extent>| {
        // Align outward: widening a read is always safe.
        let a_off = off - (off % EXTENT_ALIGN);
        out.push(Extent {
            file: file.to_string(),
            offset: a_off,
            len: end - a_off,
            seg_ord,
            column: column.clone(),
        });
    };

    for &(off, len) in ranges {
        let end = off + len;
        match cur {
            None => cur = Some((off, end)),
            Some((c_off, c_end)) => {
                let merged_len = end.saturating_sub(c_off);
                if off <= c_end.saturating_add(max_gap) && merged_len <= max_bytes {
                    cur = Some((c_off, c_end.max(end)));
                } else {
                    push(c_off, c_end, &mut extents);
                    cur = Some((off, end));
                }
            }
        }
    }
    if let Some((c_off, c_end)) = cur {
        push(c_off, c_end, &mut extents);
    }
    extents
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coalesces_adjacent_and_respects_cap() {
        let col = FieldId::new(1, "f");
        // Three ranges: first two 1 KiB apart (merge), third far away.
        let ranges = [(8192, 4096), (13312, 4096), (10_000_000, 4096)];
        let ext = coalesce("_0.dvd", 0, &col, &ranges, 64 * 1024, EXTENT_MAX_BYTES);
        assert_eq!(ext.len(), 2);
        assert_eq!(ext[0].offset, 8192); // already aligned
        assert_eq!(ext[0].end(), 17408);
        assert_eq!(ext[1].offset % EXTENT_ALIGN, 0);
        assert!(ext[1].offset <= 10_000_000 && ext[1].end() >= 10_004_096);
    }

    #[test]
    fn splits_when_over_max_bytes() {
        let col = FieldId::new(1, "f");
        let ranges = [(0, 32 * 1024 * 1024), (32 * 1024 * 1024, 48 * 1024 * 1024)];
        let ext = coalesce("_0.dvd", 0, &col, &ranges, 0, EXTENT_MAX_BYTES);
        assert_eq!(ext.len(), 2, "merged extent would exceed 64 MiB cap");
    }
}
