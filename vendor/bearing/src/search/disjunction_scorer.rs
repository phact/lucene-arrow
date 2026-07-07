// SPDX-License-Identifier: Apache-2.0

//! Base composition for scorers that score disjunctions.

use std::fmt;
use std::io;

use crate::search::collector::ScoreMode;
use crate::search::disi_wrapper::DisiWrapper;
use crate::search::disjunction_disi_approximation::DisjunctionDISIApproximation;
use crate::search::doc_id_set_iterator::DocIdSetIterator;
use crate::search::scorer::Scorer;

/// Shared state for disjunction scorers.
///
/// Deviations from Lucene's `DisjunctionScorer`:
/// - Uses composition rather than inheritance. Concrete subclasses (e.g.
///   `DisjunctionSumScorer`) embed a `DisjunctionScorerBase` and implement their
///   own `score()` via the `Scorer` trait.
/// - The `TwoPhase` inner class (lines 79–167 in the Java source) is omitted because
///   the Rust `Scorer` trait does not yet surface two-phase iterators; sub-scorers
///   are therefore always considered implicitly verified.
pub struct DisjunctionScorerBase<'a> {
    approximation: DisjunctionDISIApproximation<'a>,
    num_clauses: usize,
    #[expect(dead_code)]
    needs_scores: bool,
}

impl fmt::Debug for DisjunctionScorerBase<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DisjunctionScorerBase")
            .field("num_clauses", &self.num_clauses)
            .finish()
    }
}

impl<'a> DisjunctionScorerBase<'a> {
    /// Build from at least two sub-scorers. Matches Java's invariant `subScorers.size() > 1`.
    pub fn new(
        sub_scorers: Vec<Box<dyn Scorer + 'a>>,
        score_mode: ScoreMode,
        lead_cost: i64,
    ) -> io::Result<Self> {
        if sub_scorers.len() <= 1 {
            return Err(io::Error::other("There must be at least 2 subScorers"));
        }
        let num_clauses = sub_scorers.len();
        let needs_scores = score_mode != ScoreMode::CompleteNoScores;

        let wrappers: Vec<DisiWrapper<'a>> =
            sub_scorers.into_iter().map(DisiWrapper::new).collect();
        let approximation = DisjunctionDISIApproximation::new(wrappers, lead_cost);

        Ok(Self {
            approximation,
            num_clauses,
            needs_scores,
        })
    }

    pub fn num_clauses(&self) -> usize {
        self.num_clauses
    }

    pub fn doc_id(&self) -> i32 {
        self.approximation.doc_id()
    }

    /// Indices of sub-iterators positioned on the current doc.
    pub fn get_sub_matches(&self) -> Vec<u32> {
        self.approximation.top_list()
    }

    /// Mutable access to the underlying approximation (used as the iterator).
    pub fn approximation_mut(&mut self) -> &mut DisjunctionDISIApproximation<'a> {
        &mut self.approximation
    }

    /// Shared access to the sub-iterator wrappers.
    pub fn wrappers(&self) -> &[DisiWrapper<'a>] {
        self.approximation.wrappers()
    }

    /// Mutable access to the sub-iterator wrappers (for scoring).
    pub fn wrappers_mut(&mut self) -> &mut [DisiWrapper<'a>] {
        self.approximation.wrappers_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::doc_id_set_iterator::NO_MORE_DOCS;
    use crate::search::scorable::Scorable;

    /// Mock scorer with constant per-doc score and a sorted doc sequence.
    struct MockScorer {
        docs: Vec<i32>,
        idx: isize,
    }

    impl MockScorer {
        fn new(docs: Vec<i32>) -> Self {
            Self { docs, idx: -1 }
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
            Ok(1.0)
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
            Ok(1.0)
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

    fn mock(docs: Vec<i32>) -> Box<dyn Scorer + 'static> {
        Box::new(MockScorer::new(docs))
    }

    #[test]
    fn test_new_errors_with_zero_subs() {
        let subs: Vec<Box<dyn Scorer + 'static>> = vec![];
        let err = DisjunctionScorerBase::new(subs, ScoreMode::Complete, 10).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::Other);
        assert!(err.to_string().contains("at least 2 subScorers"));
    }

    #[test]
    fn test_new_errors_with_one_sub() {
        let subs: Vec<Box<dyn Scorer + 'static>> = vec![mock(vec![1, 2])];
        let err = DisjunctionScorerBase::new(subs, ScoreMode::Complete, 10).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::Other);
    }

    #[test]
    fn test_new_succeeds_with_two_subs() {
        let subs: Vec<Box<dyn Scorer + 'static>> = vec![mock(vec![1, 2]), mock(vec![3, 4])];
        let base = DisjunctionScorerBase::new(subs, ScoreMode::Complete, 5).unwrap();
        assert_eq!(base.num_clauses(), 2);
    }

    #[test]
    fn test_num_clauses_returns_count() {
        let subs: Vec<Box<dyn Scorer + 'static>> = vec![
            mock(vec![1]),
            mock(vec![2]),
            mock(vec![3]),
            mock(vec![4]),
            mock(vec![5]),
        ];
        let base = DisjunctionScorerBase::new(subs, ScoreMode::Complete, 5).unwrap();
        assert_eq!(base.num_clauses(), 5);
    }

    #[test]
    fn test_doc_id_initial_value() {
        let subs: Vec<Box<dyn Scorer + 'static>> = vec![mock(vec![10, 20]), mock(vec![15, 25])];
        let base = DisjunctionScorerBase::new(subs, ScoreMode::Complete, 5).unwrap();
        assert_eq!(base.doc_id(), -1);
    }

    #[test]
    fn test_doc_id_after_iteration_matches_approximation() {
        let subs: Vec<Box<dyn Scorer + 'static>> = vec![mock(vec![10, 20]), mock(vec![15, 25])];
        let mut base = DisjunctionScorerBase::new(subs, ScoreMode::Complete, 5).unwrap();
        let next = base.approximation_mut().next_doc().unwrap();
        assert_eq!(next, 10);
        assert_eq!(base.doc_id(), 10);
    }

    #[test]
    fn test_get_sub_matches_returns_indices_at_top_doc() {
        // Two subs match doc 5; one subs matches doc 7.
        let subs: Vec<Box<dyn Scorer + 'static>> =
            vec![mock(vec![5, 10]), mock(vec![5, 12]), mock(vec![7, 9])];
        let mut base = DisjunctionScorerBase::new(subs, ScoreMode::Complete, 3).unwrap();
        assert_eq!(base.approximation_mut().next_doc().unwrap(), 5);
        let matches = base.get_sub_matches();
        assert_eq!(matches.len(), 2);
        for &idx in &matches {
            assert_eq!(base.wrappers()[idx as usize].doc, 5);
        }
    }

    #[test]
    fn test_wrappers_accessor_returns_all_sub_iterators() {
        let subs: Vec<Box<dyn Scorer + 'static>> =
            vec![mock(vec![1]), mock(vec![2]), mock(vec![3])];
        let base = DisjunctionScorerBase::new(subs, ScoreMode::Complete, 5).unwrap();
        assert_eq!(base.wrappers().len(), 3);
    }

    #[test]
    fn test_wrappers_mut_allows_mutation() {
        let subs: Vec<Box<dyn Scorer + 'static>> = vec![mock(vec![1, 2]), mock(vec![3, 4])];
        let mut base = DisjunctionScorerBase::new(subs, ScoreMode::Complete, 5).unwrap();
        // Mutating doc through wrappers_mut should be visible via wrappers()
        base.wrappers_mut()[0].doc = 99;
        assert_eq!(base.wrappers()[0].doc, 99);
    }

    #[test]
    fn test_score_mode_complete_no_scores_does_not_panic() {
        // Construction should succeed for any score mode.
        let subs: Vec<Box<dyn Scorer + 'static>> = vec![mock(vec![1, 2]), mock(vec![3, 4])];
        let base = DisjunctionScorerBase::new(subs, ScoreMode::CompleteNoScores, 5).unwrap();
        assert_eq!(base.num_clauses(), 2);
    }

    #[test]
    fn test_score_mode_top_scores_does_not_panic() {
        let subs: Vec<Box<dyn Scorer + 'static>> = vec![mock(vec![1, 2]), mock(vec![3, 4])];
        let base = DisjunctionScorerBase::new(subs, ScoreMode::TopScores, 5).unwrap();
        assert_eq!(base.num_clauses(), 2);
    }

    #[test]
    fn test_iteration_full_union_via_approximation_mut() {
        let subs: Vec<Box<dyn Scorer + 'static>> = vec![
            mock(vec![1, 5, 10]),
            mock(vec![2, 5, 11]),
            mock(vec![3, 6, 10]),
        ];
        let mut base = DisjunctionScorerBase::new(subs, ScoreMode::Complete, 5).unwrap();
        let mut docs = Vec::new();
        loop {
            let d = base.approximation_mut().next_doc().unwrap();
            if d == NO_MORE_DOCS {
                break;
            }
            docs.push(d);
        }
        assert_eq!(docs, vec![1, 2, 3, 5, 6, 10, 11]);
    }
}
