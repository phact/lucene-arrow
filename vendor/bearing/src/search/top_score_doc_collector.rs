// SPDX-License-Identifier: Apache-2.0

//! Collects the top-scoring hits, returning them as `TopDocs`. Hits are sorted by score
//! descending and then (when the scores are tied) by doc ID ascending.

use std::cell::RefCell;
use std::fmt;
use std::io;
use std::rc::Rc;
use std::sync::Arc;

use super::collector::{Collector, CollectorManager, LeafCollector, ScoreContext, ScoreMode};
use super::long_heap::LongHeap;
use super::scorer::{DocScoreEncoder, MaxScoreAccumulator};
use super::top_docs::{Relation, ScoreDoc, TopDocs, TotalHits};
use crate::index::directory_reader::LeafReaderContext;
use crate::search::doc_id_set_iterator::NO_MORE_DOCS;

// ---------------------------------------------------------------------------
// CollectorState — shared between parent and leaf via Rc<RefCell<...>>
// ---------------------------------------------------------------------------

/// Mutable state shared between `TopScoreDocCollector` and its leaf collectors.
struct CollectorState {
    heap: LongHeap,
    total_hits: i32,
    total_hits_relation: Relation,
}

// ---------------------------------------------------------------------------
// TopScoreDocCollector
// ---------------------------------------------------------------------------

/// A `Collector` implementation that collects the top-scoring hits, returning them as `TopDocs`.
///
/// Flattens the Java `TopDocsCollector` base class into this struct since Rust doesn't have
/// inheritance.
pub struct TopScoreDocCollector {
    state: Rc<RefCell<CollectorState>>,
    total_hits_threshold: i32,
    min_score_acc: Option<Arc<MaxScoreAccumulator>>,
    after: Option<ScoreDoc>,
}

impl fmt::Debug for TopScoreDocCollector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let state = self.state.borrow();
        f.debug_struct("TopScoreDocCollector")
            .field("total_hits", &state.total_hits)
            .field("total_hits_threshold", &self.total_hits_threshold)
            .finish()
    }
}

impl TopScoreDocCollector {
    /// Creates a new `TopScoreDocCollector`.
    ///
    /// - `num_hits`: the number of top results to collect
    /// - `after`: the previous doc for searchAfter pagination (or `None`)
    /// - `total_hits_threshold`: the number of docs to count accurately
    /// - `min_score_acc`: shared accumulator for propagating minimum competitive score across
    ///   segments
    pub fn new(
        num_hits: i32,
        after: Option<ScoreDoc>,
        total_hits_threshold: i32,
        min_score_acc: Option<Arc<MaxScoreAccumulator>>,
    ) -> Self {
        Self {
            state: Rc::new(RefCell::new(CollectorState {
                heap: LongHeap::new_with_initial_value(
                    num_hits as usize,
                    DocScoreEncoder::LEAST_COMPETITIVE_CODE,
                ),
                total_hits: 0,
                total_hits_relation: Relation::EqualTo,
            })),
            total_hits_threshold,
            min_score_acc,
            after,
        }
    }

    /// Returns the number of valid results in the heap (excluding sentinel values).
    fn top_docs_size(&self) -> usize {
        let state = self.state.borrow();
        let mut cnt = 0;
        for i in 1..=state.heap.size() {
            if state.heap.get(i) != DocScoreEncoder::LEAST_COMPETITIVE_CODE {
                cnt += 1;
            }
        }
        cnt
    }

    /// Fills `results` with the top `how_many` results from the heap, popping in reverse order.
    fn populate_results(&mut self, how_many: usize) -> Vec<ScoreDoc> {
        let mut state = self.state.borrow_mut();
        let mut results = Vec::with_capacity(how_many);
        for _ in 0..how_many {
            results.push(ScoreDoc::new(0, 0.0)); // placeholder
        }
        for i in (0..how_many).rev() {
            let encode = state.heap.pop();
            results[i] = ScoreDoc::new(
                DocScoreEncoder::doc_id(encode),
                DocScoreEncoder::to_score(encode),
            );
        }
        results
    }

    /// Prune the least competitive hits until the number of candidates is less than or equal
    /// to `keep`.
    fn prune_least_competitive_hits_to(&mut self, keep: usize) {
        let mut state = self.state.borrow_mut();
        let size = state.heap.size();
        if size > keep {
            for _ in 0..(size - keep) {
                state.heap.pop();
            }
        }
    }

    /// Returns the top docs that were collected by this collector.
    pub fn top_docs(&mut self) -> TopDocs {
        let size = self.top_docs_size();
        self.top_docs_range(0, size)
    }

    /// Returns the documents in the range `[start .. start + how_many)`.
    pub fn top_docs_range(&mut self, start: usize, how_many: usize) -> TopDocs {
        let size = self.top_docs_size();

        if start >= size || how_many == 0 {
            let state = self.state.borrow();
            return TopDocs::new(
                TotalHits::new(state.total_hits as i64, state.total_hits_relation),
                vec![],
            );
        }

        let how_many = how_many.min(size - start);
        self.prune_least_competitive_hits_to(start + how_many);
        let results = self.populate_results(how_many);

        let state = self.state.borrow();
        TopDocs::new(
            TotalHits::new(state.total_hits as i64, state.total_hits_relation),
            results,
        )
    }
}

impl Collector for TopScoreDocCollector {
    type Leaf = TopScoreDocLeafCollector;

    fn get_leaf_collector(&mut self, context: &LeafReaderContext) -> io::Result<Self::Leaf> {
        let doc_base = context.doc_base;
        let after_score;
        let after_doc;
        if let Some(ref after) = self.after {
            after_score = after.score;
            after_doc = after.doc - context.doc_base;
        } else {
            after_score = f32::INFINITY;
            after_doc = NO_MORE_DOCS;
        }

        let top_code = self.state.borrow().heap.top();
        let top_score = DocScoreEncoder::to_score(top_code);

        Ok(TopScoreDocLeafCollector {
            state: Rc::clone(&self.state),
            score_context: None,
            doc_base,
            after_score,
            after_doc,
            top_code,
            top_score,
            min_competitive_score: 0.0,
            total_hits_threshold: self.total_hits_threshold,
            min_score_acc: self.min_score_acc.clone(),
            has_after: self.after.is_some(),
        })
    }

    fn score_mode(&self) -> ScoreMode {
        if self.total_hits_threshold == i32::MAX {
            ScoreMode::Complete
        } else {
            ScoreMode::TopScores
        }
    }
}

// ---------------------------------------------------------------------------
// TopScoreDocLeafCollector
// ---------------------------------------------------------------------------

/// Per-segment leaf collector for `TopScoreDocCollector`.
pub struct TopScoreDocLeafCollector {
    state: Rc<RefCell<CollectorState>>,
    score_context: Option<Rc<ScoreContext>>,
    doc_base: i32,
    after_score: f32,
    after_doc: i32,
    top_code: i64,
    top_score: f32,
    min_competitive_score: f32,
    total_hits_threshold: i32,
    min_score_acc: Option<Arc<MaxScoreAccumulator>>,
    has_after: bool,
}

impl fmt::Debug for TopScoreDocLeafCollector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TopScoreDocLeafCollector")
            .field("doc_base", &self.doc_base)
            .field("total_hits_threshold", &self.total_hits_threshold)
            .finish()
    }
}

impl TopScoreDocLeafCollector {
    /// Collects a competitive hit by encoding (doc + doc_base, score) into the heap.
    fn collect_competitive_hit(&mut self, doc: i32, score: f32) -> io::Result<()> {
        let code = DocScoreEncoder::encode(doc + self.doc_base, score);
        let mut state = self.state.borrow_mut();
        self.top_code = state.heap.update_top(code);
        self.top_score = DocScoreEncoder::to_score(self.top_code);
        drop(state);
        self.update_min_competitive_score()?;
        Ok(())
    }

    /// Updates the global minimum competitive score from the shared accumulator.
    fn update_global_min_competitive_score(&mut self) -> io::Result<()> {
        let min_score_acc = self
            .min_score_acc
            .as_ref()
            .expect("min_score_acc must be Some");
        let max_min_score = min_score_acc.get_raw();
        if max_min_score != i64::MIN {
            let mut score = DocScoreEncoder::to_score(max_min_score);
            if self.doc_base >= DocScoreEncoder::doc_id(max_min_score) {
                score = next_up(score);
            }
            if score > self.min_competitive_score {
                if let Some(ref ctx) = self.score_context {
                    ctx.min_competitive_score.set(score);
                }
                self.min_competitive_score = score;
                self.state.borrow_mut().total_hits_relation = Relation::GreaterThanOrEqualTo;
            }
        }
        Ok(())
    }

    /// Updates the local minimum competitive score based on the current heap top.
    fn update_min_competitive_score(&mut self) -> io::Result<()> {
        let state = self.state.borrow();
        if state.total_hits > self.total_hits_threshold {
            let local_min_score = next_up(self.top_score);
            if local_min_score > self.min_competitive_score {
                let min_score_acc = self.min_score_acc.clone();
                drop(state);

                if let Some(ref ctx) = self.score_context {
                    ctx.min_competitive_score.set(local_min_score);
                }
                self.min_competitive_score = local_min_score;
                self.state.borrow_mut().total_hits_relation = Relation::GreaterThanOrEqualTo;
                if let Some(ref min_score_acc) = min_score_acc {
                    min_score_acc.accumulate(self.top_code);
                }
            }
        }
        Ok(())
    }
}

impl LeafCollector for TopScoreDocLeafCollector {
    fn set_scorer(&mut self, score_context: Rc<ScoreContext>) -> io::Result<()> {
        self.score_context = Some(score_context);
        if self.min_score_acc.is_none() {
            self.update_min_competitive_score()?;
        } else {
            self.update_global_min_competitive_score()?;
        }
        Ok(())
    }

    fn collect(&mut self, doc: i32) -> io::Result<()> {
        let score = self
            .score_context
            .as_ref()
            .expect("set_scorer must be called before collect")
            .score
            .get();

        {
            let mut state = self.state.borrow_mut();
            state.total_hits += 1;
        }
        let hit_count_so_far = self.state.borrow().total_hits;

        let has_min_score_acc = self.min_score_acc.is_some();
        let mod_check = if has_min_score_acc {
            let interval = self.min_score_acc.as_ref().unwrap().mod_interval;
            (hit_count_so_far as i64 & interval) == 0
        } else {
            false
        };

        if mod_check {
            self.update_global_min_competitive_score()?;
        }

        if self.has_after
            && (score > self.after_score || (score == self.after_score && doc <= self.after_doc))
        {
            // hit was collected on a previous page
            if self.state.borrow().total_hits_relation == Relation::EqualTo {
                self.update_min_competitive_score()?;
            }
            return Ok(());
        }

        if score <= self.top_score {
            if hit_count_so_far == self.total_hits_threshold + 1 {
                self.update_min_competitive_score()?;
            }
        } else {
            self.collect_competitive_hit(doc, score)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// TopScoreDocCollectorManager
// ---------------------------------------------------------------------------

/// Creates `TopScoreDocCollector` instances and reduces their results into a single `TopDocs`.
#[derive(Debug)]
pub struct TopScoreDocCollectorManager {
    num_hits: i32,
    after: Option<ScoreDoc>,
    total_hits_threshold: i32,
    min_score_acc: Option<Arc<MaxScoreAccumulator>>,
}

impl TopScoreDocCollectorManager {
    /// Creates a new `TopScoreDocCollectorManager`.
    ///
    /// # Panics
    ///
    /// Panics if `num_hits <= 0` or `total_hits_threshold < 0`.
    pub fn new(num_hits: i32, after: Option<ScoreDoc>, total_hits_threshold: i32) -> Self {
        assert!(
            total_hits_threshold >= 0,
            "total_hits_threshold must be >= 0, got {}",
            total_hits_threshold
        );
        assert!(num_hits > 0, "num_hits must be > 0; got: {}", num_hits);
        let total_hits_threshold = total_hits_threshold.max(num_hits);
        let min_score_acc = if total_hits_threshold != i32::MAX {
            Some(Arc::new(MaxScoreAccumulator::new()))
        } else {
            None
        };
        Self {
            num_hits,
            after,
            total_hits_threshold,
            min_score_acc,
        }
    }
}

impl CollectorManager for TopScoreDocCollectorManager {
    type Coll = TopScoreDocCollector;
    type Result = TopDocs;

    fn new_collector(&self) -> io::Result<TopScoreDocCollector> {
        let after = self
            .after
            .as_ref()
            .map(|a| ScoreDoc::new_with_shard_index(a.doc, a.score, a.shard_index));
        Ok(TopScoreDocCollector::new(
            self.num_hits,
            after,
            self.total_hits_threshold,
            self.min_score_acc.clone(),
        ))
    }

    fn reduce(&self, mut collectors: Vec<TopScoreDocCollector>) -> io::Result<TopDocs> {
        // TODO: Use TopDocs.merge for multi-segment. For now, take the first collector's results.
        if collectors.is_empty() {
            return Ok(TopDocs::new(TotalHits::new(0, Relation::EqualTo), vec![]));
        }
        if collectors.len() == 1 {
            return Ok(collectors[0].top_docs());
        }
        // Multi-collector: merge results
        let mut all_docs: Vec<ScoreDoc> = Vec::new();
        let mut total_hits: i64 = 0;
        let mut relation = Relation::EqualTo;
        for collector in &mut collectors {
            let top_docs = collector.top_docs();
            total_hits += top_docs.total_hits.value;
            if top_docs.total_hits.relation == Relation::GreaterThanOrEqualTo {
                relation = Relation::GreaterThanOrEqualTo;
            }
            for sd in top_docs.score_docs {
                all_docs.push(sd);
            }
        }
        // Sort by score descending, then doc ascending
        all_docs.sort_by(ScoreDoc::compare);
        all_docs.truncate(self.num_hits as usize);
        Ok(TopDocs::new(TotalHits::new(total_hits, relation), all_docs))
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Returns the next representable `f32` value greater than `value` (equivalent to
/// `Math.nextUp` in Java).
fn next_up(value: f32) -> f32 {
    if value.is_nan() || value == f32::INFINITY {
        return value;
    }
    if value == 0.0 {
        return f32::MIN_POSITIVE;
    }
    let bits = value.to_bits();
    let next_bits = if value > 0.0 { bits + 1 } else { bits - 1 };
    f32::from_bits(next_bits)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use assertables::*;

    // -- next_up tests --

    #[test]
    fn test_next_up_positive() {
        let v = 1.0f32;
        let next = next_up(v);
        assert_gt!(next, v);
    }

    #[test]
    fn test_next_up_zero() {
        assert_eq!(next_up(0.0), f32::MIN_POSITIVE);
    }

    #[test]
    fn test_next_up_negative() {
        let v = -1.0f32;
        let next = next_up(v);
        assert_gt!(next, v);
        assert_lt!(next, 0.0);
    }

    #[test]
    fn test_next_up_infinity() {
        assert_eq!(next_up(f32::INFINITY), f32::INFINITY);
    }

    #[test]
    fn test_next_up_nan() {
        assert!(next_up(f32::NAN).is_nan());
    }

    // -- TopScoreDocCollector tests --

    #[test]
    fn test_collector_construction() {
        let collector = TopScoreDocCollector::new(10, None, i32::MAX, None);
        let state = collector.state.borrow();
        assert_eq!(state.total_hits, 0);
        assert_eq!(state.total_hits_relation, Relation::EqualTo);
        assert_eq!(state.heap.size(), 10);
    }

    #[test]
    fn test_score_mode_complete() {
        let collector = TopScoreDocCollector::new(10, None, i32::MAX, None);
        assert_eq!(collector.score_mode(), ScoreMode::Complete);
    }

    #[test]
    fn test_score_mode_top_scores() {
        let collector = TopScoreDocCollector::new(10, None, 100, None);
        assert_eq!(collector.score_mode(), ScoreMode::TopScores);
    }

    #[test]
    fn test_top_docs_empty() {
        let mut collector = TopScoreDocCollector::new(10, None, i32::MAX, None);
        let top_docs = collector.top_docs();
        assert_eq!(top_docs.total_hits.value, 0);
        assert_eq!(top_docs.total_hits.relation, Relation::EqualTo);
        assert_is_empty!(top_docs.score_docs);
    }

    #[test]
    fn test_top_docs_size_with_sentinels() {
        let collector = TopScoreDocCollector::new(5, None, i32::MAX, None);
        // All sentinel values, so top_docs_size should be 0
        assert_eq!(collector.top_docs_size(), 0);
    }

    #[test]
    fn test_heap_update_and_top_docs() {
        let mut collector = TopScoreDocCollector::new(3, None, i32::MAX, None);
        // Manually insert encoded values into the heap to simulate collection
        let code1 = DocScoreEncoder::encode(0, 3.0);
        let code2 = DocScoreEncoder::encode(1, 1.0);
        let code3 = DocScoreEncoder::encode(2, 2.0);
        {
            let mut state = collector.state.borrow_mut();
            state.heap.update_top(code1);
            state.heap.update_top(code2);
            state.heap.update_top(code3);
            state.total_hits = 3;
        }

        let top_docs = collector.top_docs();
        assert_eq!(top_docs.total_hits.value, 3);
        assert_eq!(top_docs.score_docs.len(), 3);
        // Results should be sorted by score descending
        assert_ge!(top_docs.score_docs[0].score, top_docs.score_docs[1].score);
        assert_ge!(top_docs.score_docs[1].score, top_docs.score_docs[2].score);
    }

    #[test]
    fn test_heap_competitive_insertion() {
        let mut collector = TopScoreDocCollector::new(2, None, i32::MAX, None);
        // Insert 3 values, heap should keep top 2
        let codes = [
            DocScoreEncoder::encode(0, 1.0),
            DocScoreEncoder::encode(1, 3.0),
            DocScoreEncoder::encode(2, 2.0),
        ];
        {
            let mut state = collector.state.borrow_mut();
            for code in codes {
                state.heap.insert_with_overflow(code);
            }
            state.total_hits = 3;
        }

        let top_docs = collector.top_docs();
        assert_eq!(top_docs.score_docs.len(), 2);
        // Should have scores 3.0 and 2.0
        assert_eq!(top_docs.score_docs[0].score, 3.0);
        assert_eq!(top_docs.score_docs[1].score, 2.0);
    }

    #[test]
    fn test_top_docs_range() {
        let mut collector = TopScoreDocCollector::new(5, None, i32::MAX, None);
        // Insert values
        {
            let mut state = collector.state.borrow_mut();
            for i in 0..5 {
                let code = DocScoreEncoder::encode(i, (i + 1) as f32);
                state.heap.update_top(code);
            }
            state.total_hits = 5;
        }

        // Request beyond size
        let top_docs = collector.top_docs_range(10, 5);
        assert_is_empty!(top_docs.score_docs);

        // Request zero
        let top_docs = collector.top_docs_range(0, 0);
        assert_is_empty!(top_docs.score_docs);
    }

    // -- TopScoreDocCollectorManager tests --

    #[test]
    fn test_manager_construction() {
        let manager = TopScoreDocCollectorManager::new(10, None, 100);
        assert_eq!(manager.num_hits, 10);
        assert_some!(manager.min_score_acc);
    }

    #[test]
    fn test_manager_construction_exact_count() {
        let manager = TopScoreDocCollectorManager::new(10, None, i32::MAX);
        assert_none!(manager.min_score_acc);
    }

    #[test]
    #[should_panic(expected = "num_hits must be > 0")]
    fn test_manager_zero_hits_panics() {
        TopScoreDocCollectorManager::new(0, None, 100);
    }

    #[test]
    #[should_panic(expected = "total_hits_threshold must be >= 0")]
    fn test_manager_negative_threshold_panics() {
        TopScoreDocCollectorManager::new(10, None, -1);
    }

    #[test]
    fn test_manager_new_collector() {
        let manager = TopScoreDocCollectorManager::new(5, None, 100);
        let collector = manager.new_collector().unwrap();
        let state = collector.state.borrow();
        assert_eq!(state.heap.size(), 5);
        drop(state);
        assert_eq!(collector.score_mode(), ScoreMode::TopScores);
    }

    #[test]
    fn test_manager_reduce_empty() {
        let manager = TopScoreDocCollectorManager::new(5, None, i32::MAX);
        let result = manager.reduce(vec![]).unwrap();
        assert_eq!(result.total_hits.value, 0);
        assert_is_empty!(result.score_docs);
    }

    #[test]
    fn test_manager_reduce_single() {
        let manager = TopScoreDocCollectorManager::new(3, None, i32::MAX);
        let collector = manager.new_collector().unwrap();
        // Insert some results
        {
            let mut state = collector.state.borrow_mut();
            let code = DocScoreEncoder::encode(0, 5.0);
            state.heap.update_top(code);
            state.total_hits = 1;
        }

        let result = manager.reduce(vec![collector]).unwrap();
        assert_eq!(result.total_hits.value, 1);
        assert_eq!(result.score_docs.len(), 1);
        assert_eq!(result.score_docs[0].score, 5.0);
        assert_eq!(result.score_docs[0].doc, 0);
    }

    #[test]
    fn test_manager_threshold_clamped_to_num_hits() {
        // total_hits_threshold should be at least num_hits
        let manager = TopScoreDocCollectorManager::new(10, None, 5);
        assert_eq!(manager.total_hits_threshold, 10);
    }
}
