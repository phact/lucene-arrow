// SPDX-License-Identifier: Apache-2.0

//! Collection pipeline: `Collector`, `LeafCollector`, `CollectorManager`, `ScoreMode`,
//! `DocIdStream`, `SimpleScorable`, and `DocAndFloatFeatureBuffer`.

use std::cell::Cell;
use std::fmt;
use std::io;
use std::rc::Rc;

use super::doc_id_set_iterator::{DocIdSetIterator, NO_MORE_DOCS};
use super::scorable::Scorable;
use crate::index::directory_reader::LeafReaderContext;

// ---------------------------------------------------------------------------
// ScoreMode
// ---------------------------------------------------------------------------

/// Different modes of search.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScoreMode {
    /// Produced scorers will allow visiting all matches and get their score.
    Complete,
    /// Produced scorers will allow visiting all matches but scores won't be available.
    CompleteNoScores,
    /// Produced scorers will optionally allow skipping over non-competitive hits using the
    /// `Scorer::set_min_competitive_score` API.
    TopScores,
    /// ScoreMode for top field collectors that can provide their own iterators, to optionally
    /// allow to skip for non-competitive docs.
    TopDocs,
    /// ScoreMode for top field collectors that can provide their own iterators, to optionally
    /// allow to skip for non-competitive docs. This mode is used when there is a secondary
    /// sort by score.
    TopDocsWithScores,
}

impl ScoreMode {
    /// Whether this `ScoreMode` needs to compute scores.
    pub fn needs_scores(&self) -> bool {
        match self {
            ScoreMode::Complete => true,
            ScoreMode::CompleteNoScores => false,
            ScoreMode::TopScores => true,
            ScoreMode::TopDocs => false,
            ScoreMode::TopDocsWithScores => true,
        }
    }

    /// Returns `true` if for this `ScoreMode` it is necessary to process all documents, or
    /// `false` if it is enough to go through top documents only.
    pub fn is_exhaustive(&self) -> bool {
        match self {
            ScoreMode::Complete => true,
            ScoreMode::CompleteNoScores => true,
            ScoreMode::TopScores => false,
            ScoreMode::TopDocs => false,
            ScoreMode::TopDocsWithScores => false,
        }
    }
}

// ---------------------------------------------------------------------------
// DocIdStream
// ---------------------------------------------------------------------------

/// A stream of doc IDs. Doc IDs may be consumed at most once.
pub trait DocIdStream {
    /// Iterate over doc IDs contained in this stream up to the given `up_to` exclusive,
    /// calling the given consumer on them. It is not possible to iterate these doc IDs again
    /// later on.
    fn for_each_up_to(
        &mut self,
        up_to: i32,
        consumer: &mut dyn FnMut(i32) -> io::Result<()>,
    ) -> io::Result<()>;

    /// Iterate over all remaining doc IDs in this stream, calling the given consumer on them.
    /// This is a terminal operation.
    fn for_each(&mut self, consumer: &mut dyn FnMut(i32) -> io::Result<()>) -> io::Result<()> {
        self.for_each_up_to(NO_MORE_DOCS, consumer)
    }

    /// Count the number of doc IDs in this stream that are below the given `up_to`. These doc
    /// IDs may not be consumed again later.
    fn count_up_to(&mut self, up_to: i32) -> io::Result<i32>;

    /// Count the number of entries in this stream. This is a terminal operation.
    fn count(&mut self) -> io::Result<i32> {
        self.count_up_to(NO_MORE_DOCS)
    }

    /// Return `true` if this stream may have remaining doc IDs. This must eventually return
    /// `false` when the stream is exhausted.
    fn may_have_remaining(&self) -> bool;
}

/// A `DocIdStream` over a contiguous range `[min, max)`.
#[derive(Debug)]
pub struct RangeDocIdStream {
    up_to: i32,
    max: i32,
}

impl RangeDocIdStream {
    /// Creates a new `RangeDocIdStream` over `[min, max)`.
    ///
    /// # Panics
    ///
    /// Panics if `min >= max`.
    pub fn new(min: i32, max: i32) -> Self {
        assert!(min < max, "min = {} >= max = {}", min, max);
        Self { up_to: min, max }
    }
}

impl DocIdStream for RangeDocIdStream {
    fn for_each_up_to(
        &mut self,
        up_to: i32,
        consumer: &mut dyn FnMut(i32) -> io::Result<()>,
    ) -> io::Result<()> {
        if up_to > self.up_to {
            let up_to = up_to.min(self.max);
            for doc in self.up_to..up_to {
                consumer(doc)?;
            }
            self.up_to = up_to;
        }
        Ok(())
    }

    fn count_up_to(&mut self, up_to: i32) -> io::Result<i32> {
        if up_to > self.up_to {
            let up_to = up_to.min(self.max);
            let count = up_to - self.up_to;
            self.up_to = up_to;
            Ok(count)
        } else {
            Ok(0)
        }
    }

    fn may_have_remaining(&self) -> bool {
        self.up_to < self.max
    }
}

// ---------------------------------------------------------------------------
// LeafCollector
// ---------------------------------------------------------------------------

/// Shared scoring context between the BulkScorer and the LeafCollector.
///
/// In Java, `LeafCollector.setScorer(Scorable)` passes a reference that the collector stores
/// and later calls `scorer.score()` on. In Rust, we can't store the `&mut dyn Scorable`
/// reference across `set_scorer`/`collect` calls due to lifetime constraints.
///
/// `ScoreContext` provides safe shared access via `Cell<f32>`: the BulkScorer writes the
/// current score before each `collect()` call, and the collector reads it. The collector
/// writes `min_competitive_score` to signal the scorer to skip non-competitive docs.
#[derive(Debug)]
pub struct ScoreContext {
    /// The current document's score. Written by the BulkScorer before `collect()`.
    pub score: Cell<f32>,
    /// The minimum competitive score. Written by the collector, read by the BulkScorer
    /// to propagate to the scorer via `Scorer::set_min_competitive_score`.
    pub min_competitive_score: Cell<f32>,
}

impl ScoreContext {
    /// Creates a new `ScoreContext` with zeroed values.
    pub fn new() -> Rc<Self> {
        Rc::new(Self {
            score: Cell::new(0.0),
            min_competitive_score: Cell::new(0.0),
        })
    }
}

/// Per-segment collector that receives matching documents and their scores.
///
/// The `set_scorer` method is called before collection begins, passing a shared `ScoreContext`.
/// The BulkScorer writes the current document's score to the context before calling `collect()`.
/// Collectors that need the score read it from the context.
pub trait LeafCollector: fmt::Debug {
    /// Called before successive calls to `collect`. The `score_context` is shared between the
    /// BulkScorer (which writes the score) and this collector (which reads it and may write
    /// `min_competitive_score`).
    fn set_scorer(&mut self, score_context: Rc<ScoreContext>) -> io::Result<()>;

    /// Called once for every document matching a query, with the unbased document number.
    fn collect(&mut self, doc: i32) -> io::Result<()>;

    /// Collect a range of doc IDs, between `min` inclusive and `max` exclusive. `max` is
    /// guaranteed to be greater than `min`.
    ///
    /// The default implementation calls `collect_stream` on a `RangeDocIdStream`.
    fn collect_range(&mut self, min: i32, max: i32) -> io::Result<()> {
        let mut stream = RangeDocIdStream::new(min, max);
        self.collect_stream(&mut stream)
    }

    /// Bulk-collect doc IDs from a `DocIdStream`.
    ///
    /// The default implementation buffers doc IDs from the stream and then collects them.
    fn collect_stream(&mut self, stream: &mut dyn DocIdStream) -> io::Result<()> {
        let mut docs = Vec::new();
        stream.for_each(&mut |doc| {
            docs.push(doc);
            Ok(())
        })?;
        for doc in docs {
            self.collect(doc)?;
        }
        Ok(())
    }

    /// Optionally returns an iterator over competitive documents. Returns `None` by default.
    fn competitive_iterator(&self) -> Option<Box<dyn DocIdSetIterator>> {
        None
    }

    /// Hook that gets called once the leaf that is associated with this collector has finished
    /// collecting successfully. The default implementation does nothing.
    fn finish(&mut self) -> io::Result<()> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Collector
// ---------------------------------------------------------------------------

/// Expert: Collectors are primarily meant to be used to gather raw results from a search, and
/// implement sorting or custom result filtering, collation, etc.
pub trait Collector: fmt::Debug {
    /// The type of `LeafCollector` this collector creates.
    type Leaf: LeafCollector;

    /// Create a new `LeafCollector` to collect the given context.
    fn get_leaf_collector(&mut self, context: &LeafReaderContext) -> io::Result<Self::Leaf>;

    /// Indicates what features are required from the scorer.
    fn score_mode(&self) -> ScoreMode;
}

/// A manager of collectors. This is useful to parallelize execution of search requests.
///
/// - `new_collector()` must return a NEW collector which will be used to collect a certain
///   set of leaves.
/// - `reduce()` will be used to reduce the results of individual collections into a
///   meaningful result. This method is only called after all leaves have been fully collected.
pub trait CollectorManager: fmt::Debug {
    /// The type of `Collector` this manager creates.
    type Coll: Collector;
    /// The result type produced by `reduce`.
    type Result;

    /// Return a new `Collector`. This must return a different instance on each call.
    fn new_collector(&self) -> io::Result<Self::Coll>;

    /// Reduce the results of individual collectors into a meaningful result.
    fn reduce(&self, collectors: Vec<Self::Coll>) -> io::Result<Self::Result>;
}

// ---------------------------------------------------------------------------
// SimpleScorable
// ---------------------------------------------------------------------------

/// Simplest implementation of `Scorable`, implemented via simple getters and setters.
#[derive(Debug)]
pub struct SimpleScorable {
    /// The current score.
    pub score: f32,
    /// The minimum competitive score.
    pub min_competitive_score: f32,
}

impl SimpleScorable {
    /// Sole constructor.
    pub fn new() -> Self {
        Self {
            score: 0.0,
            min_competitive_score: 0.0,
        }
    }

    /// Set the score.
    pub fn set_score(&mut self, score: f32) {
        self.score = score;
    }

    /// Get the min competitive score.
    pub fn min_competitive_score(&self) -> f32 {
        self.min_competitive_score
    }
}

impl Default for SimpleScorable {
    fn default() -> Self {
        Self::new()
    }
}

impl Scorable for SimpleScorable {
    fn score(&mut self) -> io::Result<f32> {
        Ok(self.score)
    }

    fn set_min_competitive_score(&mut self, min_score: f32) -> io::Result<()> {
        self.min_competitive_score = min_score;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// DocAndFloatFeatureBuffer
// ---------------------------------------------------------------------------

/// Wrapper around parallel arrays storing doc IDs and their corresponding features, stored as
/// `f32`. These features may be anything, but are typically a term frequency or a score.
pub struct DocAndFloatFeatureBuffer {
    /// Doc IDs.
    pub docs: Vec<i32>,
    /// Float-valued features.
    pub features: Vec<f32>,
    /// Number of valid entries in the doc ID and float-valued feature arrays.
    pub size: usize,
}

impl fmt::Debug for DocAndFloatFeatureBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DocAndFloatFeatureBuffer")
            .field("size", &self.size)
            .field("capacity", &self.docs.len())
            .finish()
    }
}

impl DocAndFloatFeatureBuffer {
    /// Sole constructor.
    pub fn new() -> Self {
        Self {
            docs: Vec::new(),
            features: Vec::new(),
            size: 0,
        }
    }

    /// Grow both arrays to ensure that they can store at least the given number of entries.
    /// Does not preserve existing contents.
    pub fn grow_no_copy(&mut self, min_size: usize) {
        if self.docs.len() < min_size {
            self.docs.resize(min_size, 0);
            self.features.resize(self.docs.len(), 0.0);
        }
    }
}

impl Default for DocAndFloatFeatureBuffer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// DocAndScoreAccBuffer
// ---------------------------------------------------------------------------

/// Wrapper around parallel arrays storing doc IDs and their corresponding score accumulators,
/// stored as `f64` for precision during multi-clause score accumulation.
pub struct DocAndScoreAccBuffer {
    /// Doc IDs.
    pub docs: Vec<i32>,
    /// Scores (double-precision accumulators).
    pub scores: Vec<f64>,
    /// Number of valid entries in the doc ID and score arrays.
    pub size: usize,
}

impl fmt::Debug for DocAndScoreAccBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DocAndScoreAccBuffer")
            .field("size", &self.size)
            .field("capacity", &self.docs.len())
            .finish()
    }
}

impl DocAndScoreAccBuffer {
    /// Sole constructor.
    pub fn new() -> Self {
        Self {
            docs: Vec::new(),
            scores: Vec::new(),
            size: 0,
        }
    }

    /// Grow both arrays to ensure that they can store at least the given number of entries.
    /// Existing content may be discarded.
    pub fn grow_no_copy(&mut self, min_size: usize) {
        if self.docs.len() < min_size {
            self.docs.resize(min_size, 0);
            self.scores = vec![0.0; self.docs.len()];
        }
    }

    /// Grow both arrays to ensure that they can store at least the given number of entries.
    /// Existing content is preserved.
    pub fn grow(&mut self, min_size: usize) {
        if self.docs.len() < min_size {
            self.docs.resize(min_size, 0);
            self.scores.resize(self.docs.len(), 0.0);
        }
    }

    /// Copy content from the given `DocAndFloatFeatureBuffer`, expanding float scores to
    /// doubles.
    pub fn copy_from(&mut self, buffer: &DocAndFloatFeatureBuffer) {
        self.grow_no_copy(buffer.size);
        self.docs[..buffer.size].copy_from_slice(&buffer.docs[..buffer.size]);
        for i in 0..buffer.size {
            self.scores[i] = buffer.features[i] as f64;
        }
        self.size = buffer.size;
    }
}

impl Default for DocAndScoreAccBuffer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- DocIdStream tests --

    #[test]
    fn test_range_stream_for_each() {
        let mut stream = RangeDocIdStream::new(5, 8);
        let mut docs = Vec::new();
        stream
            .for_each(&mut |doc| {
                docs.push(doc);
                Ok(())
            })
            .unwrap();
        assert_eq!(docs, vec![5, 6, 7]);
        assert!(!stream.may_have_remaining());
    }

    #[test]
    fn test_range_stream_for_each_up_to() {
        let mut stream = RangeDocIdStream::new(0, 10);
        let mut docs = Vec::new();
        stream
            .for_each_up_to(5, &mut |doc| {
                docs.push(doc);
                Ok(())
            })
            .unwrap();
        assert_eq!(docs, vec![0, 1, 2, 3, 4]);
        assert!(stream.may_have_remaining());

        docs.clear();
        stream
            .for_each(&mut |doc| {
                docs.push(doc);
                Ok(())
            })
            .unwrap();
        assert_eq!(docs, vec![5, 6, 7, 8, 9]);
        assert!(!stream.may_have_remaining());
    }

    #[test]
    fn test_range_stream_count() {
        let mut stream = RangeDocIdStream::new(0, 10);
        assert_eq!(stream.count().unwrap(), 10);
        assert!(!stream.may_have_remaining());
    }

    #[test]
    fn test_range_stream_count_up_to() {
        let mut stream = RangeDocIdStream::new(0, 10);
        assert_eq!(stream.count_up_to(5).unwrap(), 5);
        assert!(stream.may_have_remaining());
        assert_eq!(stream.count_up_to(5).unwrap(), 0);
        assert_eq!(stream.count_up_to(20).unwrap(), 5);
        assert!(!stream.may_have_remaining());
    }

    #[test]
    #[should_panic(expected = "min = 5 >= max = 5")]
    fn test_range_stream_invalid() {
        RangeDocIdStream::new(5, 5);
    }

    // -- SimpleScorable tests --

    #[test]
    fn test_simple_scorable_default_score() {
        let mut s = SimpleScorable::new();
        assert_eq!(s.score().unwrap(), 0.0);
    }

    #[test]
    fn test_simple_scorable_set_and_get_score() {
        let mut s = SimpleScorable::new();
        s.set_score(2.5);
        assert_eq!(s.score().unwrap(), 2.5);
    }

    #[test]
    fn test_simple_scorable_min_competitive_score() {
        let mut s = SimpleScorable::new();
        assert_eq!(s.min_competitive_score(), 0.0);
        s.set_min_competitive_score(1.0).unwrap();
        assert_eq!(s.min_competitive_score(), 1.0);
    }

    // -- DocAndFloatFeatureBuffer tests --

    #[test]
    fn test_feature_buffer_new() {
        let buf = DocAndFloatFeatureBuffer::new();
        assert_eq!(buf.size, 0);
        assert_is_empty!(buf.docs);
        assert_is_empty!(buf.features);
    }

    #[test]
    fn test_feature_buffer_grow_no_copy() {
        let mut buf = DocAndFloatFeatureBuffer::new();
        buf.grow_no_copy(128);
        assert_ge!(buf.docs.len(), 128);
        assert_ge!(buf.features.len(), 128);
    }

    #[test]
    fn test_feature_buffer_grow_no_copy_already_large_enough() {
        let mut buf = DocAndFloatFeatureBuffer::new();
        buf.grow_no_copy(128);
        let old_len = buf.docs.len();
        buf.grow_no_copy(64);
        assert_eq!(buf.docs.len(), old_len);
    }

    // -- DocAndScoreAccBuffer tests --

    #[test]
    fn test_score_acc_buffer_new() {
        let buf = DocAndScoreAccBuffer::new();
        assert_eq!(buf.size, 0);
        assert_is_empty!(buf.docs);
        assert_is_empty!(buf.scores);
    }

    #[test]
    fn test_score_acc_buffer_grow_no_copy() {
        let mut buf = DocAndScoreAccBuffer::new();
        buf.grow_no_copy(128);
        assert_ge!(buf.docs.len(), 128);
        assert_ge!(buf.scores.len(), 128);
    }

    #[test]
    fn test_score_acc_buffer_grow_preserves() {
        let mut buf = DocAndScoreAccBuffer::new();
        buf.grow(4);
        buf.docs[0] = 42;
        buf.scores[0] = 1.5;
        buf.size = 1;
        buf.grow(128);
        assert_ge!(buf.docs.len(), 128);
        assert_eq!(buf.docs[0], 42);
        assert_in_delta!(buf.scores[0], 1.5, 1e-10);
    }

    #[test]
    fn test_score_acc_buffer_copy_from() {
        let mut float_buf = DocAndFloatFeatureBuffer::new();
        float_buf.grow_no_copy(3);
        float_buf.docs[0] = 10;
        float_buf.docs[1] = 20;
        float_buf.docs[2] = 30;
        float_buf.features[0] = 1.5;
        float_buf.features[1] = 2.5;
        float_buf.features[2] = 3.5;
        float_buf.size = 3;

        let mut acc_buf = DocAndScoreAccBuffer::new();
        acc_buf.copy_from(&float_buf);

        assert_eq!(acc_buf.size, 3);
        assert_eq!(acc_buf.docs[0], 10);
        assert_eq!(acc_buf.docs[1], 20);
        assert_eq!(acc_buf.docs[2], 30);
        assert_in_delta!(acc_buf.scores[0], 1.5, 1e-10);
        assert_in_delta!(acc_buf.scores[1], 2.5, 1e-10);
        assert_in_delta!(acc_buf.scores[2], 3.5, 1e-10);
    }
}
