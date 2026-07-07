// SPDX-License-Identifier: Apache-2.0

//! BulkScorer for top-level conjunctions that uses block-max impacts for dynamic pruning.

use std::fmt;
use std::io;
use std::rc::Rc;

use super::collector::{
    DocAndFloatFeatureBuffer, DocAndScoreAccBuffer, LeafCollector, ScoreContext,
};
use super::doc_id_set_iterator::NO_MORE_DOCS;
use super::query::BulkScorer;
use super::scorer::Scorer;
use super::scorer_util;
use crate::util::math_util;

const MAX_WINDOW_SIZE: i32 = 65536; // 16bits - 0xFF.

/// BulkScorer implementation that focuses on top-level conjunctions over clauses that do not
/// have two-phase iterators. Computes scores on the fly in order to skip evaluating more
/// clauses if the total score would be under the minimum competitive score anyway.
pub struct BlockMaxConjunctionBulkScorer<'a> {
    scorers: Vec<Box<dyn Scorer + 'a>>,
    sum_of_other_clauses: Vec<f64>,
    max_doc: i32,
    lead_cost: i64,
    doc_and_score_buffer: DocAndFloatFeatureBuffer,
    doc_and_score_acc_buffer: DocAndScoreAccBuffer,
}

impl fmt::Debug for BlockMaxConjunctionBulkScorer<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BlockMaxConjunctionBulkScorer")
            .field("num_scorers", &self.scorers.len())
            .field("max_doc", &self.max_doc)
            .finish()
    }
}

impl<'a> BlockMaxConjunctionBulkScorer<'a> {
    /// Creates a new `BlockMaxConjunctionBulkScorer`.
    ///
    /// # Panics
    ///
    /// Panics if `scorers` has fewer than 2 elements.
    pub fn new(max_doc: i32, mut scorers: Vec<Box<dyn Scorer + 'a>>) -> Self {
        assert!(
            scorers.len() >= 2,
            "Expected 2 or more scorers, got {}",
            scorers.len()
        );
        // Sort by iterator cost ascending (cheapest lead).
        // Java: `Arrays.sort(scorers, Comparator.comparingLong(s -> s.iterator().cost()))`
        // Rust's sort_by_key provides only &T, but iterator().cost() needs &mut self.
        // Precompute costs, then sort by cached cost using sort_by_cached_key pattern.
        let costs: Vec<i64> = scorers.iter_mut().map(|s| s.iterator().cost()).collect();
        // Zip scorers with costs, sort by cost, unzip
        let mut paired: Vec<(Box<dyn Scorer + 'a>, i64)> = scorers.into_iter().zip(costs).collect();
        paired.sort_by_key(|(_, cost)| *cost);
        let lead_cost = paired[0].1;
        let scorers: Vec<Box<dyn Scorer>> = paired.into_iter().map(|(s, _)| s).collect();

        let num_scorers = scorers.len();
        Self {
            scorers,
            sum_of_other_clauses: vec![f64::INFINITY; num_scorers],
            max_doc,
            lead_cost,
            doc_and_score_buffer: DocAndFloatFeatureBuffer::new(),
            doc_and_score_acc_buffer: DocAndScoreAccBuffer::new(),
        }
    }

    /// Compute the maximum possible score for documents in [window_min, window_max].
    /// Also fills `sum_of_other_clauses` with suffix sums so that `sum_of_other_clauses[i]`
    /// is the max score from clauses i..n.
    fn compute_max_score(&mut self, window_min: i32, window_max: i32) -> io::Result<f32> {
        for i in 0..self.scorers.len() {
            self.scorers[i].advance_shallow(window_min)?;
        }

        let mut max_window_score: f64 = 0.0;
        for i in 0..self.scorers.len() {
            let max_clause_score = self.scorers[i].get_max_score(window_max)?;
            self.sum_of_other_clauses[i] = max_clause_score as f64;
            max_window_score += max_clause_score as f64;
        }
        let len = self.sum_of_other_clauses.len();
        for i in (0..len - 1).rev() {
            self.sum_of_other_clauses[i] += self.sum_of_other_clauses[i + 1];
        }
        Ok(max_window_score as f32)
    }

    /// Score a window of doc IDs by first finding agreement between all iterators and only then
    /// compute scores and call the collector until dynamic pruning kicks in.
    fn score_doc_first_until_dynamic_pruning(
        &mut self,
        collector: &mut dyn LeafCollector,
        score_context: &Rc<ScoreContext>,
        min: i32,
        max: i32,
    ) -> io::Result<i32> {
        let mut doc = self.scorers[0].iterator().doc_id();
        if doc < min {
            doc = self.scorers[0].iterator().advance(min)?;
        }

        'outer: while doc < max {
            // acceptDocs == null check omitted (always true in our pipeline)
            for i in 1..self.scorers.len() {
                let mut other_doc = self.scorers[i].iterator().doc_id();
                if other_doc < doc {
                    other_doc = self.scorers[i].iterator().advance(doc)?;
                }
                if doc != other_doc {
                    doc = self.scorers[0].iterator().advance(other_doc)?;
                    continue 'outer;
                }
            }

            let mut score: f64 = 0.0;
            for i in 0..self.scorers.len() {
                score += self.scorers[i].score()? as f64;
            }
            score_context.score.set(score as f32);
            collector.collect(doc)?;
            if score_context.min_competitive_score.get() > 0.0 {
                return self.scorers[0].iterator().next_doc();
            }
            doc = self.scorers[0].iterator().next_doc()?;
        }
        Ok(doc)
    }

    /// Score a window of doc IDs by computing matches and scores on the lead costly clause, then
    /// iterate other clauses one by one to remove documents that do not match and increase the
    /// global score by the score of the current clause.
    fn score_window_score_first(
        &mut self,
        collector: &mut dyn LeafCollector,
        score_context: &Rc<ScoreContext>,
        min: i32,
        max: i32,
        max_window_score: f32,
    ) -> io::Result<()> {
        if max_window_score < score_context.min_competitive_score.get() {
            // no hits are competitive
            return Ok(());
        }

        if self.scorers[0].iterator().doc_id() < min {
            self.scorers[0].iterator().advance(min)?;
        }
        if self.scorers[0].iterator().doc_id() >= max {
            return Ok(());
        }

        // Score the lead clause in batches via next_docs_and_scores
        loop {
            self.scorers[0].next_docs_and_scores(max, &mut self.doc_and_score_buffer)?;
            if self.doc_and_score_buffer.size == 0 {
                break;
            }

            self.doc_and_score_acc_buffer
                .copy_from(&self.doc_and_score_buffer);

            let num_scorers = self.scorers.len() as i32;
            let min_competitive = score_context.min_competitive_score.get();

            for i in 1..self.scorers.len() {
                let sum_of_other_clause = self.sum_of_other_clauses[i];
                if sum_of_other_clause != self.sum_of_other_clauses[i - 1] {
                    // two equal consecutive values mean that the first clause always returns
                    // a score of zero, so we don't need to filter hits by score again.
                    scorer_util::filter_competitive_hits(
                        &mut self.doc_and_score_acc_buffer,
                        sum_of_other_clause,
                        min_competitive,
                        num_scorers,
                    );
                }

                scorer_util::apply_required_clause(
                    &mut self.doc_and_score_acc_buffer,
                    &mut *self.scorers[i],
                )?;
            }

            for i in 0..self.doc_and_score_acc_buffer.size {
                score_context
                    .score
                    .set(self.doc_and_score_acc_buffer.scores[i] as f32);
                collector.collect(self.doc_and_score_acc_buffer.docs[i])?;
            }
        }

        let mut max_other_doc: i32 = -1;
        for i in 1..self.scorers.len() {
            max_other_doc = max_other_doc.max(self.scorers[i].iterator().doc_id());
        }
        if self.scorers[0].iterator().doc_id() < max_other_doc {
            self.scorers[0].iterator().advance(max_other_doc)?;
        }
        Ok(())
    }
}

impl BulkScorer for BlockMaxConjunctionBulkScorer<'_> {
    fn score(&mut self, collector: &mut dyn LeafCollector, min: i32, max: i32) -> io::Result<i32> {
        let score_context = ScoreContext::new();
        collector.set_scorer(Rc::clone(&score_context))?;

        let mut window_min = self.scorers[0].iterator().doc_id().max(min);
        if score_context.min_competitive_score.get() == 0.0 {
            window_min =
                self.score_doc_first_until_dynamic_pruning(collector, &score_context, min, max)?;
        }

        while window_min < max {
            // Use impacts of the least costly scorer to compute windows
            // NOTE: windowMax is inclusive
            let mut window_max = self.scorers[0].advance_shallow(window_min)?.min(max - 1);
            // Ensure the scoring window not too big, this especially works for the default
            // implementation of `Scorer::advance_shallow` which may return NO_MORE_DOCS.
            window_max = math_util::unsigned_min(window_max, window_min + MAX_WINDOW_SIZE);

            let max_window_score = self.compute_max_score(window_min, window_max)?;
            self.score_window_score_first(
                collector,
                &score_context,
                window_min,
                window_max + 1,
                max_window_score,
            )?;
            window_min = self.scorers[0].iterator().doc_id().max(window_max + 1);
        }

        if window_min >= self.max_doc {
            Ok(NO_MORE_DOCS)
        } else {
            Ok(window_min)
        }
    }

    fn cost(&self) -> i64 {
        self.lead_cost
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::collector::DocIdStream;
    use crate::search::doc_id_set_iterator::DocIdSetIterator;
    use crate::search::scorable::Scorable;
    use assertables::*;

    // -----------------------------------------------------------------------
    // Mock infrastructure
    // -----------------------------------------------------------------------

    /// Mock DocIdSetIterator backed by a sorted Vec of doc IDs.
    /// Starts before the first doc (doc_id() == -1).
    #[derive(Debug)]
    struct VecDocIdSetIterator {
        docs: Vec<i32>,
        index: usize,
    }

    impl DocIdSetIterator for VecDocIdSetIterator {
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

    /// Mock Scorer backed by a fixed list of (doc, score) pairs.
    /// Returns a configurable max_score for block-max queries.
    #[derive(Debug)]
    struct MockScorer {
        iter: VecDocIdSetIterator,
        scores: Vec<f32>,
        max_score: f32,
    }

    impl MockScorer {
        fn from_docs_and_scores(
            docs: Vec<i32>,
            scores: Vec<f32>,
            max_score: f32,
        ) -> Box<dyn Scorer> {
            assert_eq!(docs.len(), scores.len());
            let iter = VecDocIdSetIterator { docs, index: 0 };
            Box::new(Self {
                iter,
                scores,
                max_score,
            })
        }

        fn with_uniform_score(docs: Vec<i32>, score: f32) -> Box<dyn Scorer> {
            let len = docs.len();
            Self::from_docs_and_scores(docs, vec![score; len], score)
        }
    }

    impl Scorable for MockScorer {
        fn score(&mut self) -> io::Result<f32> {
            let doc = self.iter.doc_id();
            if doc < 0 || doc == NO_MORE_DOCS {
                return Ok(0.0);
            }
            for (i, &d) in self.iter.docs.iter().enumerate() {
                if d == doc {
                    return Ok(self.scores[i]);
                }
            }
            Ok(0.0)
        }
    }

    impl Scorer for MockScorer {
        fn doc_id(&self) -> i32 {
            self.iter.doc_id()
        }

        fn iterator(&mut self) -> &mut dyn DocIdSetIterator {
            &mut self.iter
        }

        fn get_max_score(&mut self, _up_to: i32) -> io::Result<f32> {
            Ok(self.max_score)
        }
    }

    /// Simple LeafCollector that records (doc, score) pairs.
    #[derive(Debug)]
    struct CollectingLeafCollector {
        docs: Vec<i32>,
        scores: Vec<f32>,
        score_context: Option<Rc<ScoreContext>>,
    }

    impl CollectingLeafCollector {
        fn new() -> Self {
            Self {
                docs: Vec::new(),
                scores: Vec::new(),
                score_context: None,
            }
        }
    }

    impl LeafCollector for CollectingLeafCollector {
        fn set_scorer(&mut self, score_context: Rc<ScoreContext>) -> io::Result<()> {
            self.score_context = Some(score_context);
            Ok(())
        }

        fn collect(&mut self, doc: i32) -> io::Result<()> {
            self.docs.push(doc);
            let score = self.score_context.as_ref().unwrap().score.get();
            self.scores.push(score);
            Ok(())
        }

        fn collect_range(&mut self, min: i32, max: i32) -> io::Result<()> {
            for doc in min..max {
                self.collect(doc)?;
            }
            Ok(())
        }

        fn collect_stream(&mut self, _stream: &mut dyn DocIdStream) -> io::Result<()> {
            todo!()
        }

        fn competitive_iterator(&self) -> Option<Box<dyn DocIdSetIterator>> {
            None
        }

        fn finish(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    // -----------------------------------------------------------------------
    // Tests
    // -----------------------------------------------------------------------

    #[test]
    #[should_panic(expected = "Expected 2 or more scorers")]
    fn test_new_panics_with_one_scorer() {
        let s1 = MockScorer::with_uniform_score(vec![1, 2, 3], 1.0);
        BlockMaxConjunctionBulkScorer::new(100, vec![s1]);
    }

    #[test]
    fn test_new_sorts_by_cost() {
        let s1 = MockScorer::with_uniform_score(vec![1, 2, 3, 4, 5], 1.0); // cost 5
        let s2 = MockScorer::with_uniform_score(vec![1, 3], 2.0); // cost 2
        let scorer = BlockMaxConjunctionBulkScorer::new(100, vec![s1, s2]);
        // Lead (scorers[0]) should be the cheaper one (cost 2)
        assert_eq!(scorer.lead_cost, 2);
    }

    #[test]
    fn test_score_finds_intersection() {
        // docs:  s1=[1,2,3,4,5], s2=[2,4,6]
        // intersection: [2, 4]
        let s1 = MockScorer::with_uniform_score(vec![1, 2, 3, 4, 5], 1.0);
        let s2 = MockScorer::with_uniform_score(vec![2, 4, 6], 2.0);
        let mut bulk = BlockMaxConjunctionBulkScorer::new(100, vec![s1, s2]);
        let mut collector = CollectingLeafCollector::new();

        bulk.score(&mut collector, 0, NO_MORE_DOCS).unwrap();

        assert_eq!(collector.docs, vec![2, 4]);
        // Scores should be sum of both scorers: 1.0 + 2.0 = 3.0
        for &score in &collector.scores {
            assert_in_delta!(score, 3.0, 1e-5);
        }
    }

    #[test]
    fn test_score_no_intersection() {
        let s1 = MockScorer::with_uniform_score(vec![1, 3, 5], 1.0);
        let s2 = MockScorer::with_uniform_score(vec![2, 4, 6], 2.0);
        let mut bulk = BlockMaxConjunctionBulkScorer::new(100, vec![s1, s2]);
        let mut collector = CollectingLeafCollector::new();

        bulk.score(&mut collector, 0, NO_MORE_DOCS).unwrap();

        assert_is_empty!(collector.docs);
    }

    #[test]
    fn test_score_complete_overlap() {
        let s1 = MockScorer::with_uniform_score(vec![1, 2, 3], 0.5);
        let s2 = MockScorer::with_uniform_score(vec![1, 2, 3], 1.5);
        let mut bulk = BlockMaxConjunctionBulkScorer::new(100, vec![s1, s2]);
        let mut collector = CollectingLeafCollector::new();

        bulk.score(&mut collector, 0, NO_MORE_DOCS).unwrap();

        assert_eq!(collector.docs, vec![1, 2, 3]);
        for &score in &collector.scores {
            assert_in_delta!(score, 2.0, 1e-5);
        }
    }

    #[test]
    fn test_score_respects_min_max_range() {
        // intersection is [2, 4], but we restrict to [3, 5)
        let s1 = MockScorer::with_uniform_score(vec![1, 2, 3, 4, 5], 1.0);
        let s2 = MockScorer::with_uniform_score(vec![2, 4, 6], 2.0);
        let mut bulk = BlockMaxConjunctionBulkScorer::new(100, vec![s1, s2]);
        let mut collector = CollectingLeafCollector::new();

        bulk.score(&mut collector, 3, 5).unwrap();

        assert_eq!(collector.docs, vec![4]);
    }

    #[test]
    fn test_score_varying_scores() {
        let s1 = MockScorer::from_docs_and_scores(vec![1, 2, 3], vec![1.0, 2.0, 3.0], 3.0);
        let s2 = MockScorer::from_docs_and_scores(vec![1, 2, 3], vec![0.1, 0.2, 0.3], 0.3);
        let mut bulk = BlockMaxConjunctionBulkScorer::new(100, vec![s1, s2]);
        let mut collector = CollectingLeafCollector::new();

        bulk.score(&mut collector, 0, NO_MORE_DOCS).unwrap();

        assert_eq!(collector.docs, vec![1, 2, 3]);
        assert_in_delta!(collector.scores[0], 1.1, 1e-5);
        assert_in_delta!(collector.scores[1], 2.2, 1e-5);
        assert_in_delta!(collector.scores[2], 3.3, 1e-5);
    }

    #[test]
    fn test_cost_returns_lead_cost() {
        let s1 = MockScorer::with_uniform_score(vec![1, 2, 3, 4, 5], 1.0); // cost 5
        let s2 = MockScorer::with_uniform_score(vec![1, 3], 2.0); // cost 2
        let bulk = BlockMaxConjunctionBulkScorer::new(100, vec![s1, s2]);
        assert_eq!(bulk.cost(), 2);
    }

    #[test]
    fn test_compute_max_score() {
        let s1 = MockScorer::from_docs_and_scores(vec![1, 2, 3], vec![1.0; 3], 5.0);
        let s2 = MockScorer::from_docs_and_scores(vec![1, 2, 3], vec![2.0; 3], 3.0);
        let mut bulk = BlockMaxConjunctionBulkScorer::new(100, vec![s1, s2]);

        let max_score = bulk.compute_max_score(0, 100).unwrap();

        // Max window score = 5.0 + 3.0 = 8.0 (lead is sorted by cost, s1 has cost 3, s2 cost 3)
        assert_in_delta!(max_score, 8.0, 1e-5);
        // sum_of_other_clauses[0] should be 5.0 + 3.0 = 8.0 (suffix sum from 0)
        assert_in_delta!(bulk.sum_of_other_clauses[0], 8.0, 1e-5);
        // sum_of_other_clauses[1] should be 3.0 (just the last clause — but which is which
        // depends on sort order; both have cost 3 so order is stable)
    }

    #[test]
    fn test_dynamic_pruning_transition() {
        // With a collector that sets min_competitive_score after collecting,
        // the scorer should transition from scoreDocFirst to scoreWindowScoreFirst.
        // We verify by checking that all matching docs are still found correctly.
        let docs: Vec<i32> = (0..20).collect();
        let s1 = MockScorer::with_uniform_score(docs.clone(), 1.0);
        let s2 = MockScorer::with_uniform_score(docs, 2.0);
        let mut bulk = BlockMaxConjunctionBulkScorer::new(100, vec![s1, s2]);

        // Collector that sets min_competitive_score after first collect
        #[derive(Debug)]
        struct PruningCollector {
            docs: Vec<i32>,
            score_context: Option<Rc<ScoreContext>>,
            set_threshold: bool,
        }
        impl LeafCollector for PruningCollector {
            fn set_scorer(&mut self, sc: Rc<ScoreContext>) -> io::Result<()> {
                self.score_context = Some(sc);
                Ok(())
            }
            fn collect(&mut self, doc: i32) -> io::Result<()> {
                self.docs.push(doc);
                if !self.set_threshold {
                    // Set a threshold that all docs should pass (score = 3.0 > 0.1)
                    self.score_context
                        .as_ref()
                        .unwrap()
                        .min_competitive_score
                        .set(0.1);
                    self.set_threshold = true;
                }
                Ok(())
            }
            fn collect_range(&mut self, min: i32, max: i32) -> io::Result<()> {
                for doc in min..max {
                    self.collect(doc)?;
                }
                Ok(())
            }
            fn collect_stream(&mut self, _stream: &mut dyn DocIdStream) -> io::Result<()> {
                todo!()
            }
            fn competitive_iterator(&self) -> Option<Box<dyn DocIdSetIterator>> {
                None
            }
            fn finish(&mut self) -> io::Result<()> {
                Ok(())
            }
        }

        let mut collector = PruningCollector {
            docs: Vec::new(),
            score_context: None,
            set_threshold: false,
        };
        bulk.score(&mut collector, 0, NO_MORE_DOCS).unwrap();

        // All 20 docs should be collected (threshold 0.1 is below all scores of 3.0)
        assert_eq!(collector.docs.len(), 20);
    }
}
