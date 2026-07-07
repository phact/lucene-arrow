// SPDX-License-Identifier: Apache-2.0

//! A `Scorer` for OR-like queries — counterpart of `ConjunctionScorer`.

use std::fmt;
use std::io;

use crate::search::collector::ScoreMode;
use crate::search::disjunction_scorer::DisjunctionScorerBase;
use crate::search::doc_id_set_iterator::{DocIdSetIterator, NO_MORE_DOCS};
use crate::search::scorable::Scorable;
use crate::search::scorer::Scorer;
use crate::util::math_util;

/// A `Scorer` for OR-like queries — counterpart of `ConjunctionScorer`.
pub struct DisjunctionSumScorer<'a> {
    base: DisjunctionScorerBase<'a>,
}

impl fmt::Debug for DisjunctionSumScorer<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DisjunctionSumScorer")
            .field("num_clauses", &self.base.num_clauses())
            .finish()
    }
}

impl<'a> DisjunctionSumScorer<'a> {
    /// Construct from at least two sub-scorers.
    pub fn new(
        sub_scorers: Vec<Box<dyn Scorer + 'a>>,
        score_mode: ScoreMode,
        lead_cost: i64,
    ) -> io::Result<Self> {
        Ok(Self {
            base: DisjunctionScorerBase::new(sub_scorers, score_mode, lead_cost)?,
        })
    }
}

impl<'a> Scorable for DisjunctionSumScorer<'a> {
    fn score(&mut self) -> io::Result<f32> {
        let top_list = self.base.get_sub_matches();
        let wrappers = self.base.wrappers_mut();
        let mut score = 0.0_f64;
        for idx in top_list {
            let w = &mut wrappers[idx as usize];
            score += w.scorer.score()? as f64;
        }
        Ok(score as f32)
    }
}

impl<'a> Scorer for DisjunctionSumScorer<'a> {
    fn doc_id(&self) -> i32 {
        self.base.doc_id()
    }

    fn iterator(&mut self) -> &mut dyn DocIdSetIterator {
        self.base.approximation_mut()
    }

    fn advance_shallow(&mut self, target: i32) -> io::Result<i32> {
        let mut min = NO_MORE_DOCS;
        for w in self.base.wrappers_mut() {
            if w.scorer.doc_id() <= target {
                min = min.min(w.scorer.advance_shallow(target)?);
            }
        }
        Ok(min)
    }

    fn get_max_score(&mut self, up_to: i32) -> io::Result<f32> {
        let num = self.base.num_clauses() as i32;
        let mut max_score = 0.0_f64;
        for w in self.base.wrappers_mut() {
            if w.scorer.doc_id() <= up_to {
                max_score += w.scorer.get_max_score(up_to)? as f64;
            }
        }
        Ok(math_util::sum_upper_bound(max_score, num) as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assertables::*;

    /// Mock scorer with constant per-doc score and a sorted doc sequence.
    struct MockScorer {
        docs: Vec<i32>,
        idx: isize,
        constant_score: f32,
    }

    impl MockScorer {
        fn new(docs: Vec<i32>, constant_score: f32) -> Self {
            Self {
                docs,
                idx: -1,
                constant_score,
            }
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

    fn mock(docs: Vec<i32>, score: f32) -> Box<dyn Scorer + 'static> {
        Box::new(MockScorer::new(docs, score))
    }

    #[test]
    fn test_two_sub_scorers_errors() {
        let subs: Vec<Box<dyn Scorer + 'static>> = vec![mock(vec![1, 2, 3], 1.0)];
        let err = DisjunctionSumScorer::new(subs, ScoreMode::Complete, 10).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::Other);
    }

    #[test]
    fn test_three_subs_union_and_sum_scores() {
        // Scorer A matches [1, 3, 5] @ 1.0
        // Scorer B matches [2, 3, 6] @ 2.0
        // Scorer C matches [3, 5, 7] @ 4.0
        // Union: 1, 2, 3, 5, 6, 7
        // Scores: 1→1, 2→2, 3→7, 5→5, 6→2, 7→4
        let subs: Vec<Box<dyn Scorer + 'static>> = vec![
            mock(vec![1, 3, 5], 1.0),
            mock(vec![2, 3, 6], 2.0),
            mock(vec![3, 5, 7], 4.0),
        ];
        let mut s = DisjunctionSumScorer::new(subs, ScoreMode::Complete, 10).unwrap();

        let mut observed: Vec<(i32, f32)> = Vec::new();
        loop {
            let doc = s.iterator().next_doc().unwrap();
            if doc == NO_MORE_DOCS {
                break;
            }
            observed.push((doc, s.score().unwrap()));
        }

        let expected: Vec<(i32, f32)> =
            vec![(1, 1.0), (2, 2.0), (3, 7.0), (5, 5.0), (6, 2.0), (7, 4.0)];
        assert_eq!(observed.len(), expected.len());
        for ((d, score), (ed, escore)) in observed.iter().zip(expected.iter()) {
            assert_eq!(d, ed);
            assert_in_delta!(*score, *escore, 1e-5);
        }
    }

    #[test]
    fn test_five_subs_union() {
        let subs: Vec<Box<dyn Scorer + 'static>> = vec![
            mock(vec![1, 5], 1.0),
            mock(vec![2, 5], 1.0),
            mock(vec![3, 5], 1.0),
            mock(vec![4, 5], 1.0),
            mock(vec![5], 1.0),
        ];
        let mut s = DisjunctionSumScorer::new(subs, ScoreMode::Complete, 3).unwrap();

        let mut docs: Vec<i32> = Vec::new();
        let mut score_at_5: Option<f32> = None;
        loop {
            let d = s.iterator().next_doc().unwrap();
            if d == NO_MORE_DOCS {
                break;
            }
            if d == 5 {
                score_at_5 = Some(s.score().unwrap());
            }
            docs.push(d);
        }
        assert_eq!(docs, vec![1, 2, 3, 4, 5]);
        // All 5 clauses match doc 5 → score = 5.0
        assert_in_delta!(score_at_5.unwrap(), 5.0, 1e-5);
    }

    #[test]
    fn test_get_max_score_sums_over_active_subs() {
        let subs: Vec<Box<dyn Scorer + 'static>> = vec![
            mock(vec![1, 2, 3], 0.5),
            mock(vec![1, 2, 3], 0.5),
            mock(vec![1, 2, 3], 0.5),
        ];
        let mut s = DisjunctionSumScorer::new(subs, ScoreMode::Complete, 10).unwrap();

        // advance_shallow sets up max_score state
        s.advance_shallow(3).unwrap();
        // With 3 subs each max 0.5, sum_upper_bound(1.5, 3) ≈ 1.5 * (1 + 2b)
        let max = s.get_max_score(3).unwrap();
        assert_gt!(max, 1.5 - 1e-3);
    }

    #[test]
    fn test_no_scores_mode_still_iterates() {
        let subs: Vec<Box<dyn Scorer + 'static>> = vec![
            mock(vec![1, 3], 1.0),
            mock(vec![2, 3], 1.0),
            mock(vec![3, 4], 1.0),
        ];
        let mut s = DisjunctionSumScorer::new(subs, ScoreMode::CompleteNoScores, 10).unwrap();

        let mut docs: Vec<i32> = Vec::new();
        loop {
            let d = s.iterator().next_doc().unwrap();
            if d == NO_MORE_DOCS {
                break;
            }
            docs.push(d);
        }
        assert_eq!(docs, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_zero_subs_errors() {
        let subs: Vec<Box<dyn Scorer + 'static>> = vec![];
        let err = DisjunctionSumScorer::new(subs, ScoreMode::Complete, 10).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::Other);
    }

    #[test]
    fn test_doc_id_initial() {
        let subs: Vec<Box<dyn Scorer + 'static>> =
            vec![mock(vec![1, 2], 1.0), mock(vec![3, 4], 1.0)];
        let s = DisjunctionSumScorer::new(subs, ScoreMode::Complete, 5).unwrap();
        assert_eq!(s.doc_id(), -1);
    }

    #[test]
    fn test_doc_id_advances_with_iterator() {
        let subs: Vec<Box<dyn Scorer + 'static>> =
            vec![mock(vec![10, 20], 1.0), mock(vec![15, 25], 1.0)];
        let mut s = DisjunctionSumScorer::new(subs, ScoreMode::Complete, 5).unwrap();
        s.iterator().next_doc().unwrap();
        assert_eq!(s.doc_id(), 10);
        s.iterator().next_doc().unwrap();
        assert_eq!(s.doc_id(), 15);
    }

    #[test]
    fn test_two_subs_minimum_valid_case() {
        // Smallest valid construction: exactly 2 sub-scorers.
        let subs: Vec<Box<dyn Scorer + 'static>> =
            vec![mock(vec![1, 3], 0.5), mock(vec![2, 3], 1.5)];
        let mut s = DisjunctionSumScorer::new(subs, ScoreMode::Complete, 5).unwrap();

        let mut observed: Vec<(i32, f32)> = Vec::new();
        loop {
            let d = s.iterator().next_doc().unwrap();
            if d == NO_MORE_DOCS {
                break;
            }
            observed.push((d, s.score().unwrap()));
        }
        assert_eq!(observed.len(), 3);
        assert_eq!(observed[0].0, 1);
        assert_in_delta!(observed[0].1, 0.5, 1e-5);
        assert_eq!(observed[1].0, 2);
        assert_in_delta!(observed[1].1, 1.5, 1e-5);
        assert_eq!(observed[2].0, 3);
        assert_in_delta!(observed[2].1, 2.0, 1e-5);
    }

    #[test]
    fn test_advance_to_specific_doc() {
        let subs: Vec<Box<dyn Scorer + 'static>> = vec![
            mock(vec![1, 5, 10, 20], 1.0),
            mock(vec![2, 6, 11, 21], 1.0),
            mock(vec![3, 7, 12, 22], 1.0),
        ];
        let mut s = DisjunctionSumScorer::new(subs, ScoreMode::Complete, 10).unwrap();
        assert_eq!(s.iterator().advance(15).unwrap(), 20);
        assert_eq!(s.doc_id(), 20);
    }

    #[test]
    fn test_advance_past_all_returns_no_more_docs() {
        let subs: Vec<Box<dyn Scorer + 'static>> =
            vec![mock(vec![1, 2], 1.0), mock(vec![3, 4], 1.0)];
        let mut s = DisjunctionSumScorer::new(subs, ScoreMode::Complete, 5).unwrap();
        assert_eq!(s.iterator().advance(100).unwrap(), NO_MORE_DOCS);
    }

    #[test]
    fn test_score_at_doc_with_no_overlap_uses_single_sub() {
        // Doc 1 is only in sub A; doc 2 is only in sub B.
        let subs: Vec<Box<dyn Scorer + 'static>> = vec![mock(vec![1], 3.5), mock(vec![2], 7.25)];
        let mut s = DisjunctionSumScorer::new(subs, ScoreMode::Complete, 5).unwrap();

        assert_eq!(s.iterator().next_doc().unwrap(), 1);
        assert_in_delta!(s.score().unwrap(), 3.5, 1e-5);
        assert_eq!(s.iterator().next_doc().unwrap(), 2);
        assert_in_delta!(s.score().unwrap(), 7.25, 1e-5);
    }

    #[test]
    fn test_score_at_full_overlap_sums_all_subs() {
        // All 4 subs match doc 5 with distinct scores.
        let subs: Vec<Box<dyn Scorer + 'static>> = vec![
            mock(vec![5], 1.0),
            mock(vec![5], 2.0),
            mock(vec![5], 4.0),
            mock(vec![5], 8.0),
        ];
        let mut s = DisjunctionSumScorer::new(subs, ScoreMode::Complete, 1).unwrap();
        assert_eq!(s.iterator().next_doc().unwrap(), 5);
        assert_in_delta!(s.score().unwrap(), 15.0, 1e-5);
    }

    #[test]
    fn test_advance_shallow_returns_no_more_docs_when_all_past_target() {
        // After iterating past doc 100, advance_shallow(50) finds no scorers <= 50.
        let subs: Vec<Box<dyn Scorer + 'static>> =
            vec![mock(vec![100, 200], 1.0), mock(vec![150, 250], 1.0)];
        let mut s = DisjunctionSumScorer::new(subs, ScoreMode::Complete, 5).unwrap();
        // Advance both subs past doc 50.
        s.iterator().advance(200).unwrap();
        let result = s.advance_shallow(50).unwrap();
        assert_eq!(result, NO_MORE_DOCS);
    }

    #[test]
    fn test_get_max_score_returns_zero_when_no_subs_below_target() {
        let subs: Vec<Box<dyn Scorer + 'static>> =
            vec![mock(vec![100, 200], 1.0), mock(vec![150, 250], 1.0)];
        let mut s = DisjunctionSumScorer::new(subs, ScoreMode::Complete, 5).unwrap();
        s.iterator().advance(200).unwrap();
        let max = s.get_max_score(50).unwrap();
        assert_eq!(max, 0.0);
    }

    #[test]
    fn test_score_after_skipping_empty_overlaps() {
        // Iteration should produce correct scores at every emitted doc.
        let subs: Vec<Box<dyn Scorer + 'static>> = vec![
            mock(vec![1, 4, 7], 1.0),
            mock(vec![2, 4, 8], 2.0),
            mock(vec![3, 7, 9], 4.0),
        ];
        let mut s = DisjunctionSumScorer::new(subs, ScoreMode::Complete, 3).unwrap();

        let expected: Vec<(i32, f32)> = vec![
            (1, 1.0),
            (2, 2.0),
            (3, 4.0),
            (4, 3.0), // 1.0 + 2.0
            (7, 5.0), // 1.0 + 4.0
            (8, 2.0),
            (9, 4.0),
        ];
        let mut observed: Vec<(i32, f32)> = Vec::new();
        loop {
            let d = s.iterator().next_doc().unwrap();
            if d == NO_MORE_DOCS {
                break;
            }
            observed.push((d, s.score().unwrap()));
        }
        assert_eq!(observed.len(), expected.len());
        for ((d, score), (ed, escore)) in observed.iter().zip(expected.iter()) {
            assert_eq!(d, ed);
            assert_in_delta!(*score, *escore, 1e-5);
        }
    }
}
