// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::fmt;
use std::mem;

use crate::util::byte_block_pool::ByteBlockPool;

/// Shared state accumulated across all documents in a segment.
///
/// Owned by the [`SegmentWorker`](crate::index::pipeline::segment_worker::SegmentWorker)
/// and passed as `&mut` to each [`FieldConsumer`](crate::index::pipeline::consumer::FieldConsumer)
/// in sequence. Only one consumer borrows at a time — no concurrent
/// access within a worker thread.
///
/// This struct serves two purposes:
///
/// 1. **Shared data pools** — the `term_byte_pool` stores deduplicated term
///    text bytes shared between postings and term vectors consumers, matching
///    Java Lucene's `termBytePool` in `TermsHash`.
///
/// 2. **Cross-consumer metadata** — information that one consumer produces
///    and another consumes. For example, norms computed by
///    [`NormsConsumer`](crate::index::pipeline::norms_consumer::NormsConsumer) are stored
///    here so that the postings consumer can build `NormsLookup` at flush
///    time for competitive impact encoding. The `text_start` hint passes
///    term byte pool offsets from postings to term vectors per token.
pub struct SegmentAccumulator {
    /// Shared term text byte pool. Postings interns term bytes here;
    /// term vectors references them by offset. Matches Java's `termBytePool`.
    term_byte_pool: ByteBlockPool,
    /// Per-token hint from postings to term vectors: the byte pool offset
    /// and term bytes of the most recently interned term.
    text_start_hint: Option<i32>,
    /// field_id → per-field norms data
    norms: HashMap<u32, PerFieldNormsData>,
    /// Total documents processed in this segment.
    doc_count: i32,
}

/// Per-field norms accumulated during document processing.
///
/// Written by `NormsConsumer` in `finish_field`, read by both
/// `NormsConsumer` (at flush for .nvm/.nvd writing) and
/// `PostingsConsumer` (at flush for competitive impact encoding).
#[derive(Debug, mem_dbg::MemSize)]
pub struct PerFieldNormsData {
    /// Field name (for debug logging and codec writing).
    pub field_name: String,
    /// Doc IDs that have a norm value for this field.
    pub docs: Vec<i32>,
    /// Norm values, parallel with `docs`.
    pub values: Vec<i64>,
}

impl SegmentAccumulator {
    /// Creates an empty accumulator with an initialized term byte pool.
    pub fn new() -> Self {
        let term_byte_pool = ByteBlockPool::new(32 * 1024);
        Self {
            term_byte_pool,
            text_start_hint: None,
            norms: HashMap::new(),
            doc_count: 0,
        }
    }

    /// Returns a reference to the shared term byte pool.
    ///
    /// Used at flush time to read term text for sorted output.
    pub fn term_byte_pool(&self) -> &ByteBlockPool {
        &self.term_byte_pool
    }

    /// Returns a mutable reference to the shared term byte pool.
    ///
    /// Used by the postings consumer to intern term bytes during `add_token`.
    pub fn term_byte_pool_mut(&mut self) -> &mut ByteBlockPool {
        &mut self.term_byte_pool
    }

    /// Records the byte pool offset of the most recently interned term.
    ///
    /// Called by the postings consumer after interning a term. The term
    /// vectors consumer reads this via [`take_text_start_hint`](Self::take_text_start_hint).
    ///
    /// Overwrites any previous unconsumed hint. This is expected when
    /// term vectors are not enabled for the current field — postings
    /// sets the hint for every token but TV only consumes it when active.
    pub fn set_text_start_hint(&mut self, text_start: i32) {
        self.text_start_hint = Some(text_start);
    }

    /// Returns and clears the text_start hint.
    ///
    /// Called by the term vectors consumer to get the byte pool offset
    /// for a term already interned by the postings consumer.
    ///
    /// # Panics
    /// Panics if no hint was set.
    pub fn take_text_start_hint(&mut self) -> i32 {
        self.text_start_hint
            .take()
            .expect("no text_start hint set — postings must process token before term vectors")
    }

    /// Clears the text_start hint without panicking.
    ///
    /// Called at `finish_field` to clean up any unconsumed hint from
    /// fields where postings ran but term vectors was not interested.
    pub fn clear_text_start_hint(&mut self) {
        self.text_start_hint = None;
    }

    /// Records a norm value for a field in a document.
    ///
    /// Called by `NormsConsumer::finish_field` after computing the norm
    /// from the token count.
    pub fn record_norm(&mut self, field_id: u32, field_name: &str, doc_id: i32, norm: i64) {
        let entry = self
            .norms
            .entry(field_id)
            .or_insert_with(|| PerFieldNormsData {
                field_name: field_name.to_string(),
                docs: Vec::new(),
                values: Vec::new(),
            });
        entry.docs.push(doc_id);
        entry.values.push(norm);
    }

    /// Returns the accumulated norms data for all fields.
    pub fn norms(&self) -> &HashMap<u32, PerFieldNormsData> {
        &self.norms
    }

    /// Increments the document count by one.
    pub fn increment_doc_count(&mut self) {
        self.doc_count += 1;
    }

    /// Returns the total number of documents processed.
    pub fn doc_count(&self) -> i32 {
        self.doc_count
    }
}

impl Default for SegmentAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for SegmentAccumulator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SegmentAccumulator")
            .field("term_byte_pool_len", &self.term_byte_pool.data.len())
            .field("text_start_hint", &self.text_start_hint)
            .field("norms_fields", &self.norms.len())
            .field("doc_count", &self.doc_count)
            .finish()
    }
}

impl mem_dbg::MemSize for SegmentAccumulator {
    fn mem_size_rec(
        &self,
        flags: mem_dbg::SizeFlags,
        refs: &mut mem_dbg::HashMap<usize, usize>,
    ) -> usize {
        let pool_size = self.term_byte_pool.mem_size_rec(flags, refs);
        let norms_size = self.norms.mem_size_rec(flags, refs);
        mem::size_of::<Self>() + pool_size + norms_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assertables::*;
    use mem_dbg::{MemSize, SizeFlags};

    #[test]
    fn mem_size_empty_is_small() {
        let acc = SegmentAccumulator::new();
        // Empty pool has no used bytes — only struct overhead
        let size = acc.mem_size(SizeFlags::default());
        assert_gt!(size, 0);
        assert_lt!(size, 1_000);
    }

    #[test]
    fn mem_size_grows_with_norms() {
        let mut acc = SegmentAccumulator::new();
        let before = acc.mem_size(SizeFlags::default());
        for doc_id in 0..100 {
            acc.record_norm(0, "body", doc_id, 42);
        }
        assert_gt!(acc.mem_size(SizeFlags::default()), before);
    }

    #[test]
    fn hint_set_and_take() {
        let mut acc = SegmentAccumulator::new();
        acc.set_text_start_hint(42);
        let result = acc.take_text_start_hint();
        assert_eq!(result, 42);
    }

    #[test]
    fn hint_cleared_after_take() {
        let mut acc = SegmentAccumulator::new();
        acc.set_text_start_hint(42);
        acc.take_text_start_hint();
        // Setting again should work (previous was consumed)
        acc.set_text_start_hint(99);
        let result = acc.take_text_start_hint();
        assert_eq!(result, 99);
    }

    #[test]
    fn clear_hint_allows_reset() {
        let mut acc = SegmentAccumulator::new();
        acc.set_text_start_hint(42);
        acc.clear_text_start_hint();
        // Setting again should work after clear
        acc.set_text_start_hint(99);
        let result = acc.take_text_start_hint();
        assert_eq!(result, 99);
    }

    #[test]
    fn hint_overwrites_unconsumed() {
        let mut acc = SegmentAccumulator::new();
        acc.set_text_start_hint(42);
        acc.set_text_start_hint(99); // overwrites, no panic
        let result = acc.take_text_start_hint();
        assert_eq!(result, 99);
    }

    #[test]
    #[should_panic(expected = "no text_start hint set")]
    fn hint_panics_on_missing() {
        let mut acc = SegmentAccumulator::new();
        acc.take_text_start_hint(); // should panic
    }

    #[test]
    fn term_byte_pool_accessible() {
        let acc = SegmentAccumulator::new();
        let pool = acc.term_byte_pool();
        assert_eq!(pool.data.len(), 0);
    }
}
