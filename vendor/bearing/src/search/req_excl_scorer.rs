// SPDX-License-Identifier: Apache-2.0

//! A Scorer for queries with a required subscorer and an excluding (prohibited) sub Scorer.

use std::fmt;
use std::io;

use crate::search::doc_id_set_iterator::{DocIdSetIterator, NO_MORE_DOCS};
use crate::search::scorable::Scorable;
use crate::search::scorer::Scorer;

/// A Scorer for queries with a required subscorer and an excluding (prohibited) sub Scorer.
///
/// Iteration yields docs from `req` that are not present in `excl`. Scoring delegates to `req`.
///
/// Deviation from Lucene: `TwoPhaseIterator` is not yet ported, so we treat both sub-scorers
/// as having a null two-phase view. The match check collapses to `excl_doc != doc`.
pub struct ReqExclScorer<'a> {
    req: Box<dyn Scorer + 'a>,
    excl: Box<dyn Scorer + 'a>,
    /// Cached cost from `req.iterator().cost()`. Required because
    /// `DocIdSetIterator::cost(&self)` cannot call `Scorer::iterator(&mut self)`.
    cost: i64,
}

impl fmt::Debug for ReqExclScorer<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ReqExclScorer")
            .field("doc_id", &self.req.doc_id())
            .field("cost", &self.cost)
            .finish()
    }
}

impl<'a> ReqExclScorer<'a> {
    /// Constructs a `ReqExclScorer` wrapping the required scorer and the prohibited (excl) scorer.
    pub fn new(mut req: Box<dyn Scorer + 'a>, excl: Box<dyn Scorer + 'a>) -> Self {
        let cost = req.iterator().cost();
        Self { req, excl, cost }
    }

    /// Walk forward from `doc` until we find a doc not present in `excl`.
    fn skip_excluded(&mut self, mut doc: i32) -> io::Result<i32> {
        loop {
            if doc == NO_MORE_DOCS {
                return Ok(doc);
            }
            let mut excl_doc = self.excl.doc_id();
            if excl_doc < doc {
                excl_doc = self.excl.iterator().advance(doc)?;
            }
            if excl_doc != doc {
                return Ok(doc);
            }
            doc = self.req.iterator().next_doc()?;
        }
    }
}

impl Scorable for ReqExclScorer<'_> {
    fn score(&mut self) -> io::Result<f32> {
        self.req.score()
    }

    fn set_min_competitive_score(&mut self, min_score: f32) -> io::Result<()> {
        self.req.set_min_competitive_score(min_score)
    }
}

impl<'a> Scorer for ReqExclScorer<'a> {
    fn doc_id(&self) -> i32 {
        self.req.doc_id()
    }

    fn iterator(&mut self) -> &mut dyn DocIdSetIterator {
        self
    }

    fn advance_shallow(&mut self, target: i32) -> io::Result<i32> {
        self.req.advance_shallow(target)
    }

    fn get_max_score(&mut self, up_to: i32) -> io::Result<f32> {
        self.req.get_max_score(up_to)
    }
}

impl<'a> DocIdSetIterator for ReqExclScorer<'a> {
    fn doc_id(&self) -> i32 {
        self.req.doc_id()
    }

    fn next_doc(&mut self) -> io::Result<i32> {
        let doc = self.req.iterator().next_doc()?;
        self.skip_excluded(doc)
    }

    fn advance(&mut self, target: i32) -> io::Result<i32> {
        let doc = self.req.iterator().advance(target)?;
        self.skip_excluded(doc)
    }

    fn cost(&self) -> i64 {
        self.cost
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mock scorer with a sorted doc sequence and constant per-doc score.
    struct MockScorer {
        docs: Vec<i32>,
        idx: isize,
        constant_score: f32,
    }

    impl MockScorer {
        fn boxed(docs: Vec<i32>, constant_score: f32) -> Box<dyn Scorer + 'static> {
            Box::new(Self {
                docs,
                idx: -1,
                constant_score,
            })
        }
    }

    impl fmt::Debug for MockScorer {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("MockScorer")
                .field("len", &self.docs.len())
                .finish()
        }
    }

    impl Scorable for MockScorer {
        fn score(&mut self) -> io::Result<f32> {
            Ok(self.constant_score)
        }
    }

    impl Scorer for MockScorer {
        fn doc_id(&self) -> i32 {
            if self.idx < 0 {
                -1
            } else if (self.idx as usize) < self.docs.len() {
                self.docs[self.idx as usize]
            } else {
                NO_MORE_DOCS
            }
        }
        fn iterator(&mut self) -> &mut dyn DocIdSetIterator {
            self
        }
        fn get_max_score(&mut self, _up_to: i32) -> io::Result<f32> {
            Ok(self.constant_score)
        }
    }

    impl DocIdSetIterator for MockScorer {
        fn doc_id(&self) -> i32 {
            Scorer::doc_id(self)
        }
        fn next_doc(&mut self) -> io::Result<i32> {
            self.idx += 1;
            Ok(Scorer::doc_id(self))
        }
        fn advance(&mut self, target: i32) -> io::Result<i32> {
            while Scorer::doc_id(self) < target {
                self.next_doc()?;
            }
            Ok(Scorer::doc_id(self))
        }
        fn cost(&self) -> i64 {
            self.docs.len() as i64
        }
    }

    fn collect(scorer: &mut dyn Scorer) -> Vec<i32> {
        let mut out = Vec::new();
        loop {
            let doc = scorer.iterator().next_doc().unwrap();
            if doc == NO_MORE_DOCS {
                break;
            }
            out.push(doc);
        }
        out
    }

    fn collect_with_scores(scorer: &mut dyn Scorer) -> Vec<(i32, f32)> {
        let mut out = Vec::new();
        loop {
            let doc = scorer.iterator().next_doc().unwrap();
            if doc == NO_MORE_DOCS {
                break;
            }
            let s = scorer.score().unwrap();
            out.push((doc, s));
        }
        out
    }

    // -----------------------------------------------------------------------
    // Iteration behavior
    // -----------------------------------------------------------------------

    #[test]
    fn test_basic_exclusion() {
        // Required: [1, 2, 3, 4, 5]; Excluded: [2, 4] → expect [1, 3, 5]
        let req = MockScorer::boxed(vec![1, 2, 3, 4, 5], 1.0);
        let excl = MockScorer::boxed(vec![2, 4], 0.0);
        let mut s = ReqExclScorer::new(req, excl);
        assert_eq!(collect(&mut s), vec![1, 3, 5]);
    }

    #[test]
    fn test_no_excluded_docs_match() {
        // Required: [1, 2, 3]; Excluded: [10, 20] → all required pass through.
        let req = MockScorer::boxed(vec![1, 2, 3], 1.0);
        let excl = MockScorer::boxed(vec![10, 20], 0.0);
        let mut s = ReqExclScorer::new(req, excl);
        assert_eq!(collect(&mut s), vec![1, 2, 3]);
    }

    #[test]
    fn test_all_required_excluded() {
        // Required: [1, 2, 3]; Excluded: [1, 2, 3] → empty
        let req = MockScorer::boxed(vec![1, 2, 3], 1.0);
        let excl = MockScorer::boxed(vec![1, 2, 3], 0.0);
        let mut s = ReqExclScorer::new(req, excl);
        assert!(collect(&mut s).is_empty());
    }

    #[test]
    fn test_empty_excl_iterator() {
        let req = MockScorer::boxed(vec![1, 2, 3], 1.0);
        let excl = MockScorer::boxed(vec![], 0.0);
        let mut s = ReqExclScorer::new(req, excl);
        assert_eq!(collect(&mut s), vec![1, 2, 3]);
    }

    #[test]
    fn test_empty_req_iterator() {
        let req = MockScorer::boxed(vec![], 1.0);
        let excl = MockScorer::boxed(vec![1, 2], 0.0);
        let mut s = ReqExclScorer::new(req, excl);
        assert!(collect(&mut s).is_empty());
    }

    #[test]
    fn test_excl_extends_past_required() {
        // Required: [1, 2, 3]; Excluded: [2, 5, 100] → exclude only doc 2.
        let req = MockScorer::boxed(vec![1, 2, 3], 1.0);
        let excl = MockScorer::boxed(vec![2, 5, 100], 0.0);
        let mut s = ReqExclScorer::new(req, excl);
        assert_eq!(collect(&mut s), vec![1, 3]);
    }

    #[test]
    fn test_excl_starts_after_required() {
        // Required: [1, 2, 3]; Excluded: [10, 20] → no overlap, all pass.
        let req = MockScorer::boxed(vec![1, 2, 3], 1.0);
        let excl = MockScorer::boxed(vec![10, 20], 0.0);
        let mut s = ReqExclScorer::new(req, excl);
        assert_eq!(collect(&mut s), vec![1, 2, 3]);
    }

    // -----------------------------------------------------------------------
    // advance() behavior
    // -----------------------------------------------------------------------

    #[test]
    fn test_advance_to_non_excluded() {
        let req = MockScorer::boxed(vec![1, 5, 10, 20], 1.0);
        let excl = MockScorer::boxed(vec![10], 0.0);
        let mut s = ReqExclScorer::new(req, excl);
        // advance past 7 → first non-excluded doc >= 7 is 20 (10 is excluded).
        let d = s.iterator().advance(7).unwrap();
        assert_eq!(d, 20);
    }

    #[test]
    fn test_advance_lands_on_excluded_then_skips() {
        let req = MockScorer::boxed(vec![5, 10, 15, 20], 1.0);
        let excl = MockScorer::boxed(vec![10, 15], 0.0);
        let mut s = ReqExclScorer::new(req, excl);
        // advance to 10; both 10 and 15 are excluded → land on 20.
        let d = s.iterator().advance(10).unwrap();
        assert_eq!(d, 20);
    }

    #[test]
    fn test_advance_past_all_returns_no_more_docs() {
        let req = MockScorer::boxed(vec![1, 2, 3], 1.0);
        let excl = MockScorer::boxed(vec![], 0.0);
        let mut s = ReqExclScorer::new(req, excl);
        let d = s.iterator().advance(100).unwrap();
        assert_eq!(d, NO_MORE_DOCS);
    }

    // -----------------------------------------------------------------------
    // Scorer delegation
    // -----------------------------------------------------------------------

    #[test]
    fn test_score_delegates_to_required() {
        let req = MockScorer::boxed(vec![1, 3, 5], 2.5);
        let excl = MockScorer::boxed(vec![3], 0.0);
        let mut s = ReqExclScorer::new(req, excl);
        let pairs = collect_with_scores(&mut s);
        assert_eq!(pairs, vec![(1, 2.5), (5, 2.5)]);
    }

    #[test]
    fn test_doc_id_initial_is_negative_one() {
        let req = MockScorer::boxed(vec![1, 2, 3], 1.0);
        let excl = MockScorer::boxed(vec![2], 0.0);
        let s = ReqExclScorer::new(req, excl);
        assert_eq!(Scorer::doc_id(&s), -1);
    }

    #[test]
    fn test_doc_id_advances_with_iteration() {
        let req = MockScorer::boxed(vec![1, 2, 3], 1.0);
        let excl = MockScorer::boxed(vec![2], 0.0);
        let mut s = ReqExclScorer::new(req, excl);
        s.iterator().next_doc().unwrap();
        assert_eq!(Scorer::doc_id(&s), 1);
        s.iterator().next_doc().unwrap();
        assert_eq!(Scorer::doc_id(&s), 3);
    }

    #[test]
    fn test_cost_matches_required_iterator_cost() {
        let req = MockScorer::boxed(vec![1, 2, 3, 4, 5], 1.0);
        let excl = MockScorer::boxed(vec![2, 4], 0.0);
        let s = ReqExclScorer::new(req, excl);
        assert_eq!(DocIdSetIterator::cost(&s), 5);
    }

    #[test]
    fn test_get_max_score_delegates_to_required() {
        let req = MockScorer::boxed(vec![1, 2, 3], 7.5);
        let excl = MockScorer::boxed(vec![], 0.0);
        let mut s = ReqExclScorer::new(req, excl);
        assert_eq!(s.get_max_score(NO_MORE_DOCS).unwrap(), 7.5);
    }

    #[test]
    fn test_set_min_competitive_score_does_not_panic() {
        let req = MockScorer::boxed(vec![1, 2, 3], 1.0);
        let excl = MockScorer::boxed(vec![], 0.0);
        let mut s = ReqExclScorer::new(req, excl);
        s.set_min_competitive_score(0.5).unwrap();
    }

    #[test]
    fn test_advance_shallow_delegates_to_required() {
        let req = MockScorer::boxed(vec![1, 2, 3], 1.0);
        let excl = MockScorer::boxed(vec![], 0.0);
        let mut s = ReqExclScorer::new(req, excl);
        // MockScorer::advance_shallow uses default (NO_MORE_DOCS)
        assert_eq!(s.advance_shallow(2).unwrap(), NO_MORE_DOCS);
    }
}
