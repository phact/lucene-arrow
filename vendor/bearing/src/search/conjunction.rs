// SPDX-License-Identifier: Apache-2.0

//! Conjunction (AND) iteration and scoring.
//!
//! Provides `ConjunctionDISI` for iterating the intersection of multiple `DocIdSetIterator`s,
//! and `intersect_iterators` as the public entry point.

use std::fmt;
use std::io;

use super::doc_id_set_iterator::{DocIdSetIterator, NO_MORE_DOCS};
use super::scorable::Scorable;
use super::scorer::Scorer;

/// Creates a conjunction over the provided `DocIdSetIterator`s.
///
/// Requires at least 2 iterators. The returned iterator yields only document IDs
/// that appear in **all** input iterators.
/// # Simplification
///
/// The reference `intersectIterators` calls `addIterator()` on each input, which flattens
/// nested `ConjunctionDISI` instances and separates `TwoPhaseIterator`s. We pass iterators
/// directly since we have no TwoPhaseIterator support and don't nest conjunctions yet.
pub fn intersect_iterators(
    iterators: Vec<Box<dyn DocIdSetIterator>>,
) -> io::Result<Box<dyn DocIdSetIterator>> {
    if iterators.len() < 2 {
        return Err(io::Error::other(
            "Cannot make a ConjunctionDISI of less than 2 iterators",
        ));
    }
    create_conjunction(iterators)
}

/// Creates a conjunction `DocIdSetIterator` from the given iterators.
///
/// If only one iterator is provided, returns it directly. Otherwise sorts by cost
/// and wraps in a `ConjunctionDISI`.
///
/// # Simplification
///
/// The reference `createConjunction` computes `minCost` to separate `BitSetIterator`s (which
/// are checked via bitwise AND instead of advancing) and wraps with
/// `ConjunctionTwoPhaseIterator` when `TwoPhaseIterator`s are present. We skip both paths
/// since we have no `BitSetIterator` or `TwoPhaseIterator` support.
fn create_conjunction(
    mut iterators: Vec<Box<dyn DocIdSetIterator>>,
) -> io::Result<Box<dyn DocIdSetIterator>> {
    // check that all sub-iterators are on the same doc ID
    let cur_doc = iterators[0].doc_id();
    for iter in &iterators {
        if iter.doc_id() != cur_doc {
            return Err(io::Error::other(
                "Sub-iterators of ConjunctionDISI are not on the same document!",
            ));
        }
    }

    if iterators.len() == 1 {
        return Ok(iterators.remove(0));
    }

    // Sort the list first to allow the sparser iterator to lead the matching.
    iterators.sort_by_key(|it| it.cost());
    Ok(Box::new(ConjunctionDISI::new(iterators)))
}

/// A conjunction of `DocIdSetIterator`s. Requires that all of its sub-iterators must be on
/// the same document all the time. This iterates over the doc IDs that are present in each
/// given `DocIdSetIterator`.
pub(crate) struct ConjunctionDISI {
    lead1: Box<dyn DocIdSetIterator>,
    lead2: Box<dyn DocIdSetIterator>,
    others: Vec<Box<dyn DocIdSetIterator>>,
}

impl fmt::Debug for ConjunctionDISI {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ConjunctionDISI")
            .field("doc_id", &self.lead1.doc_id())
            .field("num_iterators", &(2 + self.others.len()))
            .finish()
    }
}

impl ConjunctionDISI {
    fn new(mut iterators: Vec<Box<dyn DocIdSetIterator>>) -> Self {
        debug_assert!(iterators.len() >= 2);

        // Drain in reverse to avoid shifting: others first, then lead2, then lead1.
        // iterators is sorted by cost ascending, so index 0 = cheapest = lead1.
        let others = iterators.split_off(2);
        let lead2 = iterators.remove(1);
        let lead1 = iterators.remove(0);

        Self {
            lead1,
            lead2,
            others,
        }
    }

    fn assert_iters_on_same_doc(&self) -> bool {
        let cur_doc = self.lead1.doc_id();
        let mut on_same_doc = self.lead2.doc_id() == cur_doc;
        for other in &self.others {
            if !on_same_doc {
                break;
            }
            on_same_doc = on_same_doc && (other.doc_id() == cur_doc);
        }
        on_same_doc
    }

    fn do_next(&mut self, mut doc: i32) -> io::Result<i32> {
        // advanceHead:
        'advance_head: loop {
            debug_assert!(doc == self.lead1.doc_id());

            // find agreement between the two iterators with the lower costs
            // we special case them because they do not need the
            // 'other.docID() < doc' check that the 'others' iterators need
            let next2 = self.lead2.advance(doc)?;
            if next2 != doc {
                doc = self.lead1.advance(next2)?;
                if next2 != doc {
                    continue;
                }
            }

            // then find agreement with other iterators
            for other in &mut self.others {
                // other.doc may already be equal to doc if we "continued advanceHead"
                // on the previous iteration and the advance on the lead scorer exactly matched.
                if other.doc_id() < doc {
                    let next = other.advance(doc)?;

                    if next > doc {
                        // iterator beyond the current doc - advance lead and continue to the
                        // new highest doc.
                        doc = self.lead1.advance(next)?;
                        continue 'advance_head;
                    }
                }
            }

            // success - all iterators are on the same doc
            return Ok(doc);
        }
    }
}

impl DocIdSetIterator for ConjunctionDISI {
    fn doc_id(&self) -> i32 {
        self.lead1.doc_id()
    }

    fn next_doc(&mut self) -> io::Result<i32> {
        debug_assert!(
            self.assert_iters_on_same_doc(),
            "Sub-iterators of ConjunctionDISI are not on the same document!"
        );
        let doc = self.lead1.next_doc()?;
        self.do_next(doc)
    }

    fn advance(&mut self, target: i32) -> io::Result<i32> {
        debug_assert!(
            self.assert_iters_on_same_doc(),
            "Sub-iterators of ConjunctionDISI are not on the same document!"
        );
        let doc = self.lead1.advance(target)?;
        self.do_next(doc)
    }

    // Cost is the cost of the cheapest iterator (lead1, sorted first by cost)
    fn cost(&self) -> i64 {
        self.lead1.cost()
    }
}

/// Scorer for conjunctions, sets of queries, all of which are required.
///
/// # Ownership adaptation
///
/// In the reference implementation, the conjunction iterator (`disi`) extracts and owns
/// iterators from scorers. In Rust, `Scorer::iterator()` returns a borrow, so we cannot
/// extract owned iterators. Instead, `required` is sorted by cost and the conjunction
/// ping-pong algorithm is implemented inline over `required[i].iterator()`. The `scorers`
/// field stores indices into `required` identifying the scoring subset.
pub(crate) struct ConjunctionScorer<'a> {
    /// All required scorers, sorted by cost ascending.
    /// `required[0]` = lead1, `required[1]` = lead2, `required[2..]` = others.
    required: Vec<Box<dyn Scorer + 'a>>,
    /// Indices into `required` identifying the scoring subset.
    scorers: Vec<usize>,
    /// Cached lead cost (from `required[0]`). Needed because `DocIdSetIterator::cost(&self)`
    /// can't call `Scorer::iterator(&mut self)`.
    lead_cost: i64,
}

impl fmt::Debug for ConjunctionScorer<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ConjunctionScorer")
            .field("doc_id", &self.required[0].doc_id())
            .field("num_required", &self.required.len())
            .field("num_scoring", &self.scorers.len())
            .finish()
    }
}

impl<'a> ConjunctionScorer<'a> {
    /// Creates a new `ConjunctionScorer`, note that `scoring` must be a subset of `required`.
    ///
    /// In the reference implementation, both parameters share references to the same scorer
    /// objects. In Rust we must own them, so `required` and `scoring` are disjoint vecs that
    /// get merged internally.
    pub(crate) fn new(
        mut required: Vec<Box<dyn Scorer + 'a>>,
        scoring: Vec<Box<dyn Scorer + 'a>>,
    ) -> Self {
        // Merge scoring into required and record scoring indices before sorting.
        let scoring_start = required.len();
        let scoring_len = scoring.len();
        required.extend(scoring);

        // Collect costs for sorting (iterator() requires &mut, can't do inside sort_by_key).
        let mut costs: Vec<i64> = Vec::with_capacity(required.len());
        for s in required.iter_mut() {
            costs.push(s.iterator().cost());
        }

        // Sort by cost ascending so the cheapest iterator leads the matching.
        let mut indexed_costs: Vec<(usize, i64)> = costs.into_iter().enumerate().collect();
        indexed_costs.sort_by_key(|&(_, cost)| cost);

        let lead_cost = indexed_costs[0].1;

        // Reorder required by cost. Build a mapping from old index to new index.
        let mut old_to_new = vec![0usize; required.len()];
        for (new_idx, &(old_idx, _)) in indexed_costs.iter().enumerate() {
            old_to_new[old_idx] = new_idx;
        }

        let mut sorted: Vec<Option<Box<dyn Scorer + 'a>>> =
            required.into_iter().map(Some).collect();
        let mut required_sorted: Vec<Box<dyn Scorer + 'a>> = Vec::with_capacity(sorted.len());
        for &(old_idx, _) in &indexed_costs {
            required_sorted.push(sorted[old_idx].take().unwrap());
        }

        // Build scoring indices: map original scoring range through old_to_new.
        let scorers: Vec<usize> = (scoring_start..(scoring_start + scoring_len))
            .map(|old_idx| old_to_new[old_idx])
            .collect();

        Self {
            required: required_sorted,
            scorers,
            lead_cost,
        }
    }

    fn do_next(&mut self, mut doc: i32) -> io::Result<i32> {
        'advance_head: loop {
            debug_assert!(doc == self.required[0].doc_id());

            // find agreement between the two iterators with the lower costs
            // we special case them because they do not need the
            // 'other.docID() < doc' check that the 'others' iterators need
            let next2 = self.required[1].iterator().advance(doc)?;
            if next2 != doc {
                doc = self.required[0].iterator().advance(next2)?;
                if next2 != doc {
                    continue;
                }
            }

            // then find agreement with other iterators
            for i in 2..self.required.len() {
                // other.doc may already be equal to doc if we "continued advanceHead"
                // on the previous iteration and the advance on the lead scorer exactly matched.
                if self.required[i].doc_id() < doc {
                    let next = self.required[i].iterator().advance(doc)?;

                    if next > doc {
                        // iterator beyond the current doc - advance lead and continue to the
                        // new highest doc.
                        doc = self.required[0].iterator().advance(next)?;
                        continue 'advance_head;
                    }
                }
            }

            // success - all iterators are on the same doc
            return Ok(doc);
        }
    }
}

impl Scorable for ConjunctionScorer<'_> {
    fn score(&mut self) -> io::Result<f32> {
        let mut sum = 0.0_f64;
        for &idx in &self.scorers {
            sum += self.required[idx].score()? as f64;
        }
        Ok(sum as f32)
    }

    fn set_min_competitive_score(&mut self, min_score: f32) -> io::Result<()> {
        // This scorer is only used for TOP_SCORES when there is a single scoring clause
        if self.scorers.len() == 1 {
            let idx = self.scorers[0];
            self.required[idx].set_min_competitive_score(min_score)?;
        }
        Ok(())
    }
}

impl Scorer for ConjunctionScorer<'_> {
    fn doc_id(&self) -> i32 {
        self.required[0].doc_id()
    }

    fn iterator(&mut self) -> &mut dyn DocIdSetIterator {
        self
    }

    fn advance_shallow(&mut self, target: i32) -> io::Result<i32> {
        if self.scorers.len() == 1 {
            let idx = self.scorers[0];
            return self.required[idx].advance_shallow(target);
        }
        for &idx in &self.scorers {
            self.required[idx].advance_shallow(target)?;
        }
        // super.advanceShallow(target) returns NO_MORE_DOCS by default
        Ok(NO_MORE_DOCS)
    }

    fn get_max_score(&mut self, up_to: i32) -> io::Result<f32> {
        let mut max_score = 0.0_f64;
        for &idx in &self.scorers {
            if self.required[idx].doc_id() <= up_to {
                max_score += self.required[idx].get_max_score(up_to)? as f64;
            }
        }
        Ok(max_score as f32)
    }
}

impl DocIdSetIterator for ConjunctionScorer<'_> {
    fn doc_id(&self) -> i32 {
        self.required[0].doc_id()
    }

    fn next_doc(&mut self) -> io::Result<i32> {
        let doc = self.required[0].iterator().next_doc()?;
        self.do_next(doc)
    }

    fn advance(&mut self, target: i32) -> io::Result<i32> {
        let doc = self.required[0].iterator().advance(target)?;
        self.do_next(doc)
    }

    fn cost(&self) -> i64 {
        self.lead_cost
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::doc_id_set_iterator::NO_MORE_DOCS;

    /// DocIdSetIterator over a fixed sorted list of doc IDs.
    #[derive(Debug)]
    struct VecDocIdSetIterator {
        docs: Vec<i32>,
        index: usize,
    }

    impl VecDocIdSetIterator {
        fn from_docs(docs: Vec<i32>) -> Box<dyn DocIdSetIterator> {
            Box::new(Self { docs, index: 0 })
        }
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

    fn collect_all(iter: &mut dyn DocIdSetIterator) -> Vec<i32> {
        let mut result = Vec::new();
        loop {
            let doc = iter.next_doc().unwrap();
            if doc == NO_MORE_DOCS {
                break;
            }
            result.push(doc);
        }
        result
    }

    #[test]
    fn test_conjunction_two_iterators_basic() {
        let iter1 = VecDocIdSetIterator::from_docs(vec![1, 3, 5, 7, 9]);
        let iter2 = VecDocIdSetIterator::from_docs(vec![2, 3, 6, 7, 10]);
        let mut conj = intersect_iterators(vec![iter1, iter2]).unwrap();

        assert_eq!(collect_all(conj.as_mut()), vec![3, 7]);
    }

    #[test]
    fn test_conjunction_two_iterators_no_overlap() {
        let iter1 = VecDocIdSetIterator::from_docs(vec![1, 3, 5]);
        let iter2 = VecDocIdSetIterator::from_docs(vec![2, 4, 6]);
        let mut conj = intersect_iterators(vec![iter1, iter2]).unwrap();

        assert_eq!(collect_all(conj.as_mut()), Vec::<i32>::new());
    }

    #[test]
    fn test_conjunction_two_iterators_identical() {
        let iter1 = VecDocIdSetIterator::from_docs(vec![1, 2, 3]);
        let iter2 = VecDocIdSetIterator::from_docs(vec![1, 2, 3]);
        let mut conj = intersect_iterators(vec![iter1, iter2]).unwrap();

        assert_eq!(collect_all(conj.as_mut()), vec![1, 2, 3]);
    }

    #[test]
    fn test_conjunction_advance() {
        let iter1 = VecDocIdSetIterator::from_docs(vec![1, 3, 5, 7, 9, 11]);
        let iter2 = VecDocIdSetIterator::from_docs(vec![2, 3, 5, 8, 9, 11]);
        let mut conj = intersect_iterators(vec![iter1, iter2]).unwrap();

        // Advance past 4 — should land on 5
        let doc = conj.advance(4).unwrap();
        assert_eq!(doc, 5);

        // next_doc should get 9
        let doc = conj.next_doc().unwrap();
        assert_eq!(doc, 9);

        // next_doc should get 11
        let doc = conj.next_doc().unwrap();
        assert_eq!(doc, 11);

        // exhausted
        let doc = conj.next_doc().unwrap();
        assert_eq!(doc, NO_MORE_DOCS);
    }

    #[test]
    fn test_conjunction_three_iterators() {
        let iter1 = VecDocIdSetIterator::from_docs(vec![1, 2, 3, 4, 5]);
        let iter2 = VecDocIdSetIterator::from_docs(vec![2, 3, 4, 5, 6]);
        let iter3 = VecDocIdSetIterator::from_docs(vec![3, 4, 5, 6, 7]);
        let mut conj = intersect_iterators(vec![iter1, iter2, iter3]).unwrap();

        assert_eq!(collect_all(conj.as_mut()), vec![3, 4, 5]);
    }

    #[test]
    fn test_conjunction_cost() {
        // iter1 has 3 docs (cost=3), iter2 has 5 docs (cost=5)
        // cost should be the minimum = 3
        let iter1 = VecDocIdSetIterator::from_docs(vec![1, 2, 3]);
        let iter2 = VecDocIdSetIterator::from_docs(vec![1, 2, 3, 4, 5]);
        let conj = intersect_iterators(vec![iter1, iter2]).unwrap();

        assert_eq!(conj.cost(), 3);
    }

    #[test]
    fn test_conjunction_too_few_iterators() {
        let iter1 = VecDocIdSetIterator::from_docs(vec![1, 2, 3]);
        let result = intersect_iterators(vec![iter1]);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // ConjunctionScorer tests
    // -----------------------------------------------------------------------

    use crate::search::scorable::Scorable;

    /// Mock Scorer backed by a fixed list of (doc, score) pairs.
    #[derive(Debug)]
    struct MockScorer {
        iter: VecDocIdSetIterator,
        scores: Vec<f32>,
    }

    impl MockScorer {
        fn from_docs_and_scores(docs: Vec<i32>, scores: Vec<f32>) -> Box<dyn Scorer> {
            assert_eq!(docs.len(), scores.len());
            let iter = VecDocIdSetIterator { docs, index: 0 };
            Box::new(Self { iter, scores })
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
            Ok(f32::MAX)
        }
    }

    #[test]
    fn test_conjunction_scorer_iterates_intersection() {
        let s1 = MockScorer::from_docs_and_scores(vec![1, 3, 5, 7, 9], vec![1.0; 5]);
        let s2 = MockScorer::from_docs_and_scores(vec![2, 3, 6, 7, 10], vec![2.0; 5]);

        // Both are required and scoring.
        let mut scorer = ConjunctionScorer::new(vec![], vec![s1, s2]);

        let mut docs = Vec::new();
        loop {
            let doc = scorer.next_doc().unwrap();
            if doc == NO_MORE_DOCS {
                break;
            }
            docs.push(doc);
        }
        assert_eq!(docs, vec![3, 7]);
    }

    #[test]
    fn test_conjunction_scorer_sums_scores() {
        let s1 = MockScorer::from_docs_and_scores(vec![1, 3, 5], vec![1.0, 2.0, 3.0]);
        let s2 = MockScorer::from_docs_and_scores(vec![1, 3, 5], vec![0.5, 1.5, 2.5]);

        let mut scorer = ConjunctionScorer::new(vec![], vec![s1, s2]);

        // Advance to first doc
        let doc = scorer.next_doc().unwrap();
        assert_eq!(doc, 1);
        let score = scorer.score().unwrap();
        assert!((score - 1.5).abs() < 1e-6, "expected 1.5, got {score}");

        let doc = scorer.next_doc().unwrap();
        assert_eq!(doc, 3);
        let score = scorer.score().unwrap();
        assert!((score - 3.5).abs() < 1e-6, "expected 3.5, got {score}");

        let doc = scorer.next_doc().unwrap();
        assert_eq!(doc, 5);
        let score = scorer.score().unwrap();
        assert!((score - 5.5).abs() < 1e-6, "expected 5.5, got {score}");
    }

    #[test]
    fn test_conjunction_scorer_doc_id() {
        let s1 = MockScorer::from_docs_and_scores(vec![2, 4, 6], vec![1.0; 3]);
        let s2 = MockScorer::from_docs_and_scores(vec![2, 4, 6], vec![1.0; 3]);

        let mut scorer = ConjunctionScorer::new(vec![], vec![s1, s2]);

        // Before first next_doc, doc_id should be -1
        assert_eq!(Scorer::doc_id(&scorer), -1);

        scorer.next_doc().unwrap();
        assert_eq!(Scorer::doc_id(&scorer), 2);

        scorer.next_doc().unwrap();
        assert_eq!(Scorer::doc_id(&scorer), 4);
    }

    #[test]
    fn test_conjunction_scorer_set_min_competitive_score_single_scoring() {
        // Two scorers total, but only one is scoring — set_min_competitive_score should propagate.
        let s1 = MockScorer::from_docs_and_scores(vec![1, 2], vec![1.0, 2.0]);
        let s2_required = MockScorer::from_docs_and_scores(vec![1, 2], vec![0.0, 0.0]);
        let mut scorer = ConjunctionScorer::new(vec![s2_required], vec![s1]);

        // This should not error (single scoring scorer propagation).
        scorer.set_min_competitive_score(0.5).unwrap();
    }
}
