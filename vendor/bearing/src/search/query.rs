// SPDX-License-Identifier: Apache-2.0

//! Query execution types: `Query`, `Weight`, `ScorerSupplier`, `BulkScorer`, and
//! `BatchScoreBulkScorer`.

use std::fmt;
use std::io;
use std::rc::Rc;

use super::collector::{DocAndFloatFeatureBuffer, LeafCollector, ScoreContext, ScoreMode};
use super::doc_id_set_iterator::DocIdSetIterator;
use super::index_searcher::IndexSearcher;
use super::scorer::Scorer;
use crate::index::directory_reader::LeafReaderContext;

// ---------------------------------------------------------------------------
// Query
// ---------------------------------------------------------------------------

/// Trait for all query types.
pub trait Query: fmt::Debug {
    /// Expert: Constructs an appropriate `Weight` implementation for this query.
    ///
    /// `score_mode` indicates how the produced scorers will be consumed.
    /// `boost` is the boost propagated by parent queries.
    fn create_weight(
        &self,
        searcher: &IndexSearcher,
        score_mode: ScoreMode,
        boost: f32,
    ) -> io::Result<Box<dyn Weight>>;

    /// Expert: Called to re-write queries into primitive queries. Returns `None` if this
    /// query does not need rewriting.
    ///
    /// // TODO: implement query rewriting
    fn rewrite(&self, _searcher: &IndexSearcher) -> io::Result<Option<Box<dyn Query>>> {
        Ok(None)
    }
}

// ---------------------------------------------------------------------------
// BulkScorer
// ---------------------------------------------------------------------------

/// Scores a range of documents at once, returned by `Weight::bulk_scorer`.
///
/// Only queries that have a more optimized means of scoring across a range of documents
/// need to override this. Otherwise, a default implementation is wrapped around the `Scorer`
/// returned by `Weight::scorer`.
pub trait BulkScorer: fmt::Debug {
    /// Collects matching documents in a range and returns an estimation of the next matching
    /// document which is on or after `max`.
    ///
    /// `min` is the minimum document to be considered for matching. All documents strictly
    /// before this value must be ignored. `max` is the upper bound (exclusive).
    fn score(&mut self, collector: &mut dyn LeafCollector, min: i32, max: i32) -> io::Result<i32>;

    /// Same as `DocIdSetIterator::cost` for bulk scorers.
    fn cost(&self) -> i64;
}

// ---------------------------------------------------------------------------
// ScorerSupplier
// ---------------------------------------------------------------------------

/// A supplier of `Scorer`. This allows getting an estimate of the cost before building the
/// `Scorer`.
///
/// The `'a` lifetime parameter bounds the reader-borrowed state (such as norms iterators)
/// that the produced `Scorer` / `BulkScorer` may hold.
pub trait ScorerSupplier<'a>: fmt::Debug + 'a {
    /// Get the `Scorer`. This may not return `None` and must be called at most once.
    ///
    /// `lead_cost` can be interpreted as an upper bound of the number of times that
    /// `DocIdSetIterator::next_doc`, `DocIdSetIterator::advance` will be called. Under doubt,
    /// pass `i64::MAX`.
    fn get(&mut self, lead_cost: i64) -> io::Result<Box<dyn Scorer + 'a>>;

    /// Optional method: Get a scorer that is optimized for bulk-scoring. The default
    /// implementation iterates matches from the `Scorer`.
    fn bulk_scorer(&mut self) -> io::Result<Box<dyn BulkScorer + 'a>> {
        let scorer = self.get(i64::MAX)?;
        Ok(Box::new(DefaultBulkScorer::new(scorer)))
    }

    /// Get an estimate of the `Scorer` that would be returned by `get`. This may be a costly
    /// operation, so it should only be called if necessary.
    fn cost(&self) -> i64;

    /// Inform this `ScorerSupplier` that its returned scorers produce scores that get passed
    /// to the collector, as opposed to partial scores that then need to get combined.
    fn set_top_level_scoring_clause(&mut self) -> io::Result<()> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Weight
// ---------------------------------------------------------------------------

/// Expert: Calculate query weights and build query scorers.
///
/// The purpose of `Weight` is to ensure searching does not modify a `Query`, so that a
/// `Query` instance can be reused.
///
/// `IndexSearcher` dependent state of the query should reside in the `Weight`.
/// `LeafReader` dependent state should reside in the `Scorer`.
pub trait Weight: fmt::Debug {
    /// Get a `ScorerSupplier`, which allows knowing the cost of the `Scorer` before building
    /// it. Returns `None` if no documents match.
    fn scorer_supplier<'a>(
        &self,
        context: &'a LeafReaderContext,
    ) -> io::Result<Option<Box<dyn ScorerSupplier<'a> + 'a>>>;

    /// Returns a `Scorer` which can iterate in order over all matching documents and assign
    /// them a score. Returns `None` if no documents match.
    ///
    /// The default implementation delegates to `scorer_supplier`.
    fn scorer<'a>(
        &self,
        context: &'a LeafReaderContext,
    ) -> io::Result<Option<Box<dyn Scorer + 'a>>> {
        match self.scorer_supplier(context)? {
            None => Ok(None),
            Some(mut supplier) => Ok(Some(supplier.get(i64::MAX)?)),
        }
    }

    /// Helper method that delegates to `scorer_supplier`. Returns a `BulkScorer` for the
    /// given leaf, or `None` if no documents match.
    ///
    /// The default implementation calls `set_top_level_scoring_clause` then `bulk_scorer`.
    fn bulk_scorer<'a>(
        &self,
        context: &'a LeafReaderContext,
    ) -> io::Result<Option<Box<dyn BulkScorer + 'a>>> {
        match self.scorer_supplier(context)? {
            None => Ok(None),
            Some(mut supplier) => {
                supplier.set_top_level_scoring_clause()?;
                Ok(Some(supplier.bulk_scorer()?))
            }
        }
    }

    /// Counts the number of live documents that match. Returns -1 if the count could not be
    /// computed in sub-linear time.
    fn count(&self, _context: &LeafReaderContext) -> io::Result<i32> {
        Ok(-1)
    }
}

// ---------------------------------------------------------------------------
// DefaultBulkScorer
// ---------------------------------------------------------------------------

/// Just wraps a `Scorer` and performs top scoring using it.
///
/// This is the default `BulkScorer` implementation when a `Weight` doesn't provide a
/// specialized one.
pub struct DefaultBulkScorer<'a> {
    scorer: Box<dyn Scorer + 'a>,
}

impl fmt::Debug for DefaultBulkScorer<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DefaultBulkScorer").finish()
    }
}

impl<'a> DefaultBulkScorer<'a> {
    /// Sole constructor.
    pub fn new(scorer: Box<dyn Scorer + 'a>) -> Self {
        Self { scorer }
    }

    /// Score documents in `[min, max)` using the simple iterator path (no TwoPhaseIterator,
    /// no competitive iterator).
    fn score_iterator(
        collector: &mut dyn LeafCollector,
        iterator: &mut dyn DocIdSetIterator,
        max: i32,
    ) -> io::Result<()> {
        let mut doc = iterator.doc_id();
        while doc < max {
            collector.collect(doc)?;
            doc = iterator.next_doc()?;
        }
        Ok(())
    }

    /// Score documents in `[min, max)` using a competitive iterator that can skip
    /// non-competitive docs.
    fn score_competitive_iterator(
        collector: &mut dyn LeafCollector,
        iterator: &mut dyn DocIdSetIterator,
        competitive_iterator: &mut dyn DocIdSetIterator,
        max: i32,
    ) -> io::Result<()> {
        let mut doc = iterator.doc_id();
        while doc < max {
            debug_assert!(competitive_iterator.doc_id() <= doc);
            if competitive_iterator.doc_id() < doc {
                let competitive_next = competitive_iterator.advance(doc)?;
                if competitive_next != doc {
                    doc = iterator.advance(competitive_next)?;
                    continue;
                }
            }
            collector.collect(doc)?;
            doc = iterator.next_doc()?;
        }
        Ok(())
    }
}

impl BulkScorer for DefaultBulkScorer<'_> {
    fn score(&mut self, collector: &mut dyn LeafCollector, min: i32, max: i32) -> io::Result<i32> {
        let score_context = ScoreContext::new();
        collector.set_scorer(Rc::clone(&score_context))?;
        let competitive_iterator = collector.competitive_iterator();

        // Advance to min. Each borrow of self.scorer.iterator() is temporary so we can
        // also call self.scorer.score() in the loop below.
        if self.scorer.iterator().doc_id() < min {
            if self.scorer.iterator().doc_id() == min - 1 {
                self.scorer.iterator().next_doc()?;
            } else {
                self.scorer.iterator().advance(min)?;
            }
        }

        match competitive_iterator {
            None => {
                // Score documents in [min, max) writing each score into the ScoreContext
                // before calling collect. We can't use a static method here because we need
                // access to both self.scorer.iterator() and self.scorer.score() in the loop,
                // and holding an iterator borrow would prevent calling score().
                while self.scorer.doc_id() < max {
                    score_context.score.set(self.scorer.score()?);
                    collector.collect(self.scorer.doc_id())?;
                    self.scorer.iterator().next_doc()?;
                }
            }
            Some(mut ci) => {
                let ci_doc = ci.doc_id();
                let effective_min = if ci_doc > min { ci_doc.min(max) } else { min };
                if self.scorer.iterator().doc_id() < effective_min {
                    self.scorer.iterator().advance(effective_min)?;
                }
                while self.scorer.doc_id() < max {
                    debug_assert!(ci.doc_id() <= self.scorer.doc_id());
                    if ci.doc_id() < self.scorer.doc_id() {
                        let competitive_next = ci.advance(self.scorer.doc_id())?;
                        if competitive_next != self.scorer.doc_id() {
                            self.scorer.iterator().advance(competitive_next)?;
                            continue;
                        }
                    }
                    score_context.score.set(self.scorer.score()?);
                    collector.collect(self.scorer.doc_id())?;
                    self.scorer.iterator().next_doc()?;
                }
            }
        }

        Ok(self.scorer.doc_id())
    }

    fn cost(&self) -> i64 {
        // TODO: Java returns `iterator.cost()` here, but Rust's `Scorer::iterator()` requires
        // `&mut self` which conflicts with `BulkScorer::cost(&self)`. To fix properly, either
        // cache the cost at construction time, or change the Scorer trait to expose cost
        // without &mut self. Returns 0 as a placeholder — callers currently get cost from
        // ScorerSupplier before constructing the BulkScorer.
        0
    }
}

// ---------------------------------------------------------------------------
// DefaultScorerSupplier
// ---------------------------------------------------------------------------

/// A wrapper for a pre-built `Scorer` that implements `ScorerSupplier`.
pub struct DefaultScorerSupplier<'a> {
    scorer: Option<Box<dyn Scorer + 'a>>,
    cost: i64,
}

impl fmt::Debug for DefaultScorerSupplier<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DefaultScorerSupplier")
            .field("cost", &self.cost)
            .finish()
    }
}

impl<'a> DefaultScorerSupplier<'a> {
    /// Creates a new `DefaultScorerSupplier` wrapping the given scorer.
    pub fn new(scorer: Box<dyn Scorer + 'a>, cost: i64) -> Self {
        Self {
            scorer: Some(scorer),
            cost,
        }
    }
}

impl<'a> ScorerSupplier<'a> for DefaultScorerSupplier<'a> {
    fn get(&mut self, _lead_cost: i64) -> io::Result<Box<dyn Scorer + 'a>> {
        self.scorer
            .take()
            .ok_or_else(|| io::Error::other("ScorerSupplier.get() called more than once"))
    }

    fn cost(&self) -> i64 {
        self.cost
    }
}

// ---------------------------------------------------------------------------
// BatchScoreBulkScorer
// ---------------------------------------------------------------------------

/// A bulk scorer that uses `Scorer::next_docs_and_scores` to score documents in batches.
///
/// Used when `ScoreMode::needs_scores()` is true and `Scorer::next_docs_and_scores` has
/// optimizations to run faster than one-by-one iteration. If the collector has a
/// competitive iterator, falls back to `DefaultBulkScorer`.
pub struct BatchScoreBulkScorer<'a> {
    scorer: Box<dyn Scorer + 'a>,
    buffer: DocAndFloatFeatureBuffer,
}

impl fmt::Debug for BatchScoreBulkScorer<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BatchScoreBulkScorer").finish()
    }
}

impl<'a> BatchScoreBulkScorer<'a> {
    /// Sole constructor.
    pub fn new(scorer: Box<dyn Scorer + 'a>) -> Self {
        Self {
            scorer,
            buffer: DocAndFloatFeatureBuffer::new(),
        }
    }
}

impl BulkScorer for BatchScoreBulkScorer<'_> {
    fn score(&mut self, collector: &mut dyn LeafCollector, min: i32, max: i32) -> io::Result<i32> {
        let score_context = ScoreContext::new();

        // If there is a competitive iterator, fall back to DefaultBulkScorer
        if collector.competitive_iterator().is_some() {
            collector.set_scorer(Rc::clone(&score_context))?;
            let competitive_iterator = collector.competitive_iterator();

            let iterator = self.scorer.iterator();

            if iterator.doc_id() < min {
                if iterator.doc_id() == min - 1 {
                    iterator.next_doc()?;
                } else {
                    iterator.advance(min)?;
                }
            }

            match competitive_iterator {
                None => {
                    DefaultBulkScorer::score_iterator(collector, iterator, max)?;
                }
                Some(mut ci) => {
                    let ci_doc = ci.doc_id();
                    let effective_min = if ci_doc > min { ci_doc.min(max) } else { min };
                    if iterator.doc_id() < effective_min {
                        iterator.advance(effective_min)?;
                    }
                    DefaultBulkScorer::score_competitive_iterator(
                        collector,
                        iterator,
                        ci.as_mut(),
                        max,
                    )?;
                }
            }

            return Ok(self.scorer.iterator().doc_id());
        }

        collector.set_scorer(Rc::clone(&score_context))?;
        self.scorer
            .set_min_competitive_score(score_context.min_competitive_score.get())?;

        if self.scorer.doc_id() < min {
            self.scorer.iterator().advance(min)?;
        }

        loop {
            self.scorer.next_docs_and_scores(max, &mut self.buffer)?;
            if self.buffer.size == 0 {
                break;
            }
            let size = self.buffer.size;
            for i in 0..size {
                let score = self.buffer.features[i];
                score_context.score.set(score);
                if score >= score_context.min_competitive_score.get() {
                    collector.collect(self.buffer.docs[i])?;
                }
            }
            self.scorer
                .set_min_competitive_score(score_context.min_competitive_score.get())?;
        }

        Ok(self.scorer.doc_id())
    }

    fn cost(&self) -> i64 {
        // TODO: Same issue as DefaultBulkScorer — Scorer::iterator() requires &mut self.
        0
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::doc_id_set_iterator::NO_MORE_DOCS;
    use crate::search::scorable::Scorable;
    use assertables::*;

    /// DocIdSetIterator over a fixed list of doc IDs.
    #[derive(Debug)]
    struct MockScorerIterator {
        docs: Vec<i32>,
        index: usize,
    }

    impl DocIdSetIterator for MockScorerIterator {
        fn doc_id(&self) -> i32 {
            if self.index == 0 {
                -1
            } else if self.index > self.docs.len() {
                NO_MORE_DOCS
            } else {
                self.docs[self.index - 1]
            }
        }
        fn next_doc(&mut self) -> io::Result<i32> {
            if self.index >= self.docs.len() {
                self.index = self.docs.len() + 1;
                return Ok(NO_MORE_DOCS);
            }
            self.index += 1;
            Ok(self.docs[self.index - 1])
        }
        fn advance(&mut self, target: i32) -> io::Result<i32> {
            loop {
                let doc = self.next_doc()?;
                if doc >= target {
                    return Ok(doc);
                }
            }
        }
        fn cost(&self) -> i64 {
            self.docs.len() as i64
        }
    }

    /// We need Scorer to delegate to an iterator. Since Scorer::iterator() returns
    /// &mut dyn DocIdSetIterator, MockScorer needs to own an iterator.
    #[derive(Debug)]
    struct FullMockScorer {
        iter: MockScorerIterator,
        scores: Vec<f32>,
    }

    impl FullMockScorer {
        fn new(docs: Vec<i32>, scores: Vec<f32>) -> Self {
            let iter = MockScorerIterator {
                docs: docs.clone(),
                index: 0,
            };
            Self { iter, scores }
        }
    }

    impl Scorable for FullMockScorer {
        fn score(&mut self) -> io::Result<f32> {
            let doc = self.iter.doc_id();
            if doc < 0 || doc == NO_MORE_DOCS {
                return Ok(0.0);
            }
            // Find the index of this doc
            for (i, &d) in self.iter.docs.iter().enumerate() {
                if d == doc {
                    return Ok(self.scores[i]);
                }
            }
            Ok(0.0)
        }
    }

    impl Scorer for FullMockScorer {
        fn doc_id(&self) -> i32 {
            self.iter.doc_id()
        }
        fn iterator(&mut self) -> &mut dyn DocIdSetIterator {
            &mut self.iter
        }
        fn get_max_score(&mut self, _up_to: i32) -> io::Result<f32> {
            Ok(f32::MAX)
        }
    }

    // -- DefaultScorerSupplier tests --

    #[test]
    fn test_default_scorer_supplier_get() {
        let scorer = FullMockScorer::new(vec![0, 1, 2], vec![1.0, 2.0, 3.0]);
        let mut supplier = DefaultScorerSupplier::new(Box::new(scorer), 3);
        assert_eq!(supplier.cost(), 3);
        let s = supplier.get(100);
        assert_ok!(s);
    }

    #[test]
    fn test_default_scorer_supplier_get_twice_fails() {
        let scorer = FullMockScorer::new(vec![0], vec![1.0]);
        let mut supplier = DefaultScorerSupplier::new(Box::new(scorer), 1);
        supplier.get(100).unwrap();
        let result = supplier.get(100);
        assert_err!(result);
    }

    #[test]
    fn test_default_scorer_supplier_bulk_scorer() {
        let scorer = FullMockScorer::new(vec![0, 1, 2], vec![1.0, 2.0, 3.0]);
        let mut supplier = DefaultScorerSupplier::new(Box::new(scorer), 3);
        let bs = supplier.bulk_scorer();
        assert_ok!(bs);
    }

    // -- DefaultBulkScorer tests --

    /// Simple leaf collector that records doc IDs.
    #[derive(Debug)]
    struct DocCollector {
        docs: Vec<i32>,
    }

    impl DocCollector {
        fn new() -> Self {
            Self { docs: Vec::new() }
        }
    }

    impl LeafCollector for DocCollector {
        fn set_scorer(&mut self, _score_context: Rc<ScoreContext>) -> io::Result<()> {
            Ok(())
        }
        fn collect(&mut self, doc: i32) -> io::Result<()> {
            self.docs.push(doc);
            Ok(())
        }
    }

    #[test]
    fn test_default_bulk_scorer_scores_all_docs() {
        let scorer = FullMockScorer::new(vec![0, 5, 10], vec![1.0, 2.0, 3.0]);
        let mut bulk = DefaultBulkScorer::new(Box::new(scorer));
        let mut collector = DocCollector::new();

        bulk.score(&mut collector, 0, NO_MORE_DOCS).unwrap();

        assert_eq!(collector.docs, vec![0, 5, 10]);
    }

    #[test]
    fn test_default_bulk_scorer_respects_range() {
        let scorer = FullMockScorer::new(vec![0, 5, 10, 15], vec![1.0, 2.0, 3.0, 4.0]);
        let mut bulk = DefaultBulkScorer::new(Box::new(scorer));
        let mut collector = DocCollector::new();

        // Score docs in [3, 12) — should get doc 5 and 10
        bulk.score(&mut collector, 3, 12).unwrap();

        assert_eq!(collector.docs, vec![5, 10]);
    }

    #[test]
    fn test_default_bulk_scorer_empty_range() {
        let scorer = FullMockScorer::new(vec![10, 20], vec![1.0, 2.0]);
        let mut bulk = DefaultBulkScorer::new(Box::new(scorer));
        let mut collector = DocCollector::new();

        bulk.score(&mut collector, 0, 5).unwrap();

        assert_is_empty!(collector.docs);
    }

    // -- BatchScoreBulkScorer tests --

    /// Leaf collector that records (doc, score) pairs via ScoreContext.
    #[derive(Debug)]
    struct ScoreCollector {
        docs: Vec<i32>,
        scores: Vec<f32>,
        score_context: Option<Rc<ScoreContext>>,
    }

    impl ScoreCollector {
        fn new() -> Self {
            Self {
                docs: Vec::new(),
                scores: Vec::new(),
                score_context: None,
            }
        }
    }

    impl LeafCollector for ScoreCollector {
        fn set_scorer(&mut self, score_context: Rc<ScoreContext>) -> io::Result<()> {
            self.score_context = Some(score_context);
            Ok(())
        }
        fn collect(&mut self, doc: i32) -> io::Result<()> {
            self.docs.push(doc);
            if let Some(ref ctx) = self.score_context {
                self.scores.push(ctx.score.get());
            }
            Ok(())
        }
    }

    #[test]
    fn test_batch_score_bulk_scorer_collects_all_docs() {
        let scorer = FullMockScorer::new(vec![0, 5, 10], vec![1.0, 2.0, 3.0]);
        let mut bulk = BatchScoreBulkScorer::new(Box::new(scorer));
        let mut collector = ScoreCollector::new();

        bulk.score(&mut collector, 0, NO_MORE_DOCS).unwrap();

        assert_eq!(collector.docs, vec![0, 5, 10]);
        assert_eq!(collector.scores.len(), 3);
        // All scores should be positive (computed by FullMockScorer)
        for &s in &collector.scores {
            assert!(s > 0.0, "expected positive score, got {s}");
        }
    }

    #[test]
    fn test_batch_score_bulk_scorer_respects_range() {
        let scorer = FullMockScorer::new(vec![0, 5, 10, 15], vec![1.0, 2.0, 3.0, 4.0]);
        let mut bulk = BatchScoreBulkScorer::new(Box::new(scorer));
        let mut collector = DocCollector::new();

        // Score docs in [3, 12) — should get doc 5 and 10
        bulk.score(&mut collector, 3, 12).unwrap();

        assert_eq!(collector.docs, vec![5, 10]);
    }

    #[test]
    fn test_batch_score_bulk_scorer_empty_range() {
        let scorer = FullMockScorer::new(vec![10, 20], vec![1.0, 2.0]);
        let mut bulk = BatchScoreBulkScorer::new(Box::new(scorer));
        let mut collector = DocCollector::new();

        bulk.score(&mut collector, 0, 5).unwrap();

        assert_is_empty!(collector.docs);
    }

    #[test]
    fn test_batch_score_bulk_scorer_min_competitive_score() {
        let scorer = FullMockScorer::new(vec![0, 1, 2, 3], vec![0.5, 1.5, 0.3, 2.0]);
        let mut bulk = BatchScoreBulkScorer::new(Box::new(scorer));

        // Collector that sets min_competitive_score to 1.0 after first collect
        #[derive(Debug)]
        struct FilteringCollector {
            docs: Vec<i32>,
            score_context: Option<Rc<ScoreContext>>,
            first_collect: bool,
        }
        impl LeafCollector for FilteringCollector {
            fn set_scorer(&mut self, score_context: Rc<ScoreContext>) -> io::Result<()> {
                self.score_context = Some(score_context);
                Ok(())
            }
            fn collect(&mut self, doc: i32) -> io::Result<()> {
                self.docs.push(doc);
                if !self.first_collect {
                    self.first_collect = true;
                    // After first doc, set min competitive score to 1.0
                    if let Some(ref ctx) = self.score_context {
                        ctx.min_competitive_score.set(1.0);
                    }
                }
                Ok(())
            }
        }

        let mut collector = FilteringCollector {
            docs: Vec::new(),
            score_context: None,
            first_collect: false,
        };

        bulk.score(&mut collector, 0, NO_MORE_DOCS).unwrap();

        // Doc 0 (score 0.5) collected before threshold set
        // Doc 1 (score 1.5) >= 1.0, collected
        // Doc 2 (score 0.3) < 1.0, filtered by BatchScoreBulkScorer
        // Doc 3 (score 2.0) >= 1.0, collected
        assert_eq!(collector.docs, vec![0, 1, 3]);
    }

    // NOTE: Weight::scorer and Weight::bulk_scorer default method tests require a real
    // LeafReaderContext (which needs a SegmentReader with actual index data). These will be
    // tested as integration tests when TermQuery (Tier 5) wires everything together.
    // The default method logic is straightforward delegation and is covered by the
    // DefaultScorerSupplier and DefaultBulkScorer tests above.
}
