// SPDX-License-Identifier: Apache-2.0

//! Scorer for queries with a required part and an optional part. The required scorer drives
//! iteration; the optional scorer is advanced lazily to add bonus scores when it matches.

use std::fmt;
use std::io;

use super::collector::ScoreMode;
use super::doc_id_set_iterator::{DocIdSetIterator, NO_MORE_DOCS};
use super::scorable::Scorable;
use super::scorer::Scorer;

/// Combines a required scorer with an optional scorer. The required scorer must match for a
/// document to be collected; the optional scorer's score is added when it also matches.
///
/// In TOP_SCORES mode, uses block-level impact information to skip over non-competitive
/// documents. When the required scorer's max score alone can't beat the minimum competitive
/// score, the optional scorer becomes required (conjunction mode) to find docs where the
/// combined score is competitive.
pub struct ReqOptSumScorer<'a> {
    req_scorer: Box<dyn Scorer + 'a>,
    opt_scorer: Box<dyn Scorer + 'a>,
    req_cost: i64,
    is_top_scores: bool,
    // TOP_SCORES iterator state
    up_to: i32,
    max_score: f32,
    // Shared mutable state accessed by both iterator and scorer logic
    min_score: f32,
    req_max_score: f32,
    opt_is_required: bool,
}

impl fmt::Debug for ReqOptSumScorer<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ReqOptSumScorer")
            .field("min_score", &self.min_score)
            .field("opt_is_required", &self.opt_is_required)
            .field("is_top_scores", &self.is_top_scores)
            .finish()
    }
}

impl<'a> ReqOptSumScorer<'a> {
    /// Creates a new `ReqOptSumScorer`.
    ///
    /// Rust adaptation: no TwoPhaseIterator support. Iterators are accessed via
    /// `scorer.iterator()` rather than cached as fields.
    pub fn new(
        mut req_scorer: Box<dyn Scorer + 'a>,
        mut opt_scorer: Box<dyn Scorer + 'a>,
        score_mode: ScoreMode,
    ) -> io::Result<Self> {
        let req_cost = req_scorer.iterator().cost();

        let (is_top_scores, req_max_score) = if score_mode != ScoreMode::TopScores {
            (false, f32::INFINITY)
        } else {
            req_scorer.advance_shallow(0)?;
            opt_scorer.advance_shallow(0)?;
            let req_max = req_scorer.get_max_score(NO_MORE_DOCS)?;
            (true, req_max)
        };

        Ok(Self {
            req_scorer,
            opt_scorer,
            req_cost,
            is_top_scores,
            up_to: -1,
            max_score: 0.0,
            min_score: 0.0,
            req_max_score,
            opt_is_required: false,
        })
    }

    fn move_to_next_block(&mut self, target: i32) -> io::Result<()> {
        self.up_to = self.advance_shallow_impl(target)?;
        let req_max_score_block = self.req_scorer.get_max_score(self.up_to)?;
        self.max_score = self.get_max_score_impl(self.up_to)?;
        // Potentially move to a conjunction
        self.opt_is_required = req_max_score_block < self.min_score;
        Ok(())
    }

    fn advance_impacts(&mut self, mut target: i32) -> io::Result<i32> {
        if target > self.up_to {
            self.move_to_next_block(target)?;
        }

        loop {
            if self.max_score >= self.min_score {
                return Ok(target);
            }

            if self.up_to == NO_MORE_DOCS {
                return Ok(NO_MORE_DOCS);
            }

            target = self.up_to + 1;

            self.move_to_next_block(target)?;
        }
    }

    /// Core TOP_SCORES iteration logic. Skips non-competitive blocks and, when
    /// `opt_is_required` is true, seeks a conjunction between req and opt within the
    /// current block.
    fn advance_internal(&mut self, target: i32) -> io::Result<i32> {
        if target == NO_MORE_DOCS {
            self.req_scorer.iterator().advance(target)?;
            return Ok(NO_MORE_DOCS);
        }
        let mut req_doc = target;
        'advance_head: loop {
            if self.min_score != 0.0 {
                req_doc = self.advance_impacts(req_doc)?;
            }
            if self.req_scorer.doc_id() < req_doc {
                req_doc = self.req_scorer.iterator().advance(req_doc)?;
            }
            if req_doc == NO_MORE_DOCS || !self.opt_is_required {
                return Ok(req_doc);
            }

            let upper_bound = if self.req_max_score < self.min_score {
                NO_MORE_DOCS
            } else {
                self.up_to
            };
            if req_doc > upper_bound {
                continue;
            }

            // Find the next common doc within the current block
            loop {
                // invariant: req_doc >= opt_doc
                let mut opt_doc = self.opt_scorer.doc_id();
                if opt_doc < req_doc {
                    opt_doc = self.opt_scorer.iterator().advance(req_doc)?;
                }
                if opt_doc > upper_bound {
                    req_doc = upper_bound + 1;
                    continue 'advance_head;
                }

                if opt_doc != req_doc {
                    req_doc = self.req_scorer.iterator().advance(opt_doc)?;
                    if req_doc > upper_bound {
                        continue 'advance_head;
                    }
                }

                if req_doc == NO_MORE_DOCS || opt_doc == req_doc {
                    return Ok(req_doc);
                }
            }
        }
    }

    /// Helper: calls advance_shallow on both scorers and combines results.
    /// Factored out to avoid borrow conflicts with `self` in move_to_next_block.
    fn advance_shallow_impl(&mut self, target: i32) -> io::Result<i32> {
        let mut up_to = self.req_scorer.advance_shallow(target)?;
        if self.opt_scorer.doc_id() <= target {
            up_to = up_to.min(self.opt_scorer.advance_shallow(target)?);
        } else if self.opt_scorer.doc_id() != NO_MORE_DOCS {
            up_to = up_to.min(self.opt_scorer.doc_id() - 1);
        }
        Ok(up_to)
    }

    /// Helper: calls get_max_score on both scorers and combines results.
    fn get_max_score_impl(&mut self, up_to: i32) -> io::Result<f32> {
        let mut max_score = self.req_scorer.get_max_score(up_to)?;
        if self.opt_scorer.doc_id() <= up_to {
            max_score += self.opt_scorer.get_max_score(up_to)?;
        }
        Ok(max_score)
    }
}

impl Scorable for ReqOptSumScorer<'_> {
    fn score(&mut self) -> io::Result<f32> {
        let cur_doc = self.req_scorer.doc_id();
        let mut score = self.req_scorer.score()?;

        let mut opt_scorer_doc = self.opt_scorer.doc_id();
        if opt_scorer_doc < cur_doc {
            opt_scorer_doc = self.opt_scorer.iterator().advance(cur_doc)?;
        }
        if opt_scorer_doc == cur_doc {
            score += self.opt_scorer.score()?;
        }

        Ok(score)
    }

    fn set_min_competitive_score(&mut self, min_score: f32) -> io::Result<()> {
        self.min_score = min_score;
        // Potentially move to a conjunction
        if self.req_max_score < min_score {
            self.opt_is_required = true;
            if self.req_max_score == 0.0 {
                self.opt_scorer.set_min_competitive_score(min_score)?;
            }
        }
        Ok(())
    }
}

impl Scorer for ReqOptSumScorer<'_> {
    fn doc_id(&self) -> i32 {
        self.req_scorer.doc_id()
    }

    fn iterator(&mut self) -> &mut dyn DocIdSetIterator {
        self
    }

    fn advance_shallow(&mut self, target: i32) -> io::Result<i32> {
        self.advance_shallow_impl(target)
    }

    fn get_max_score(&mut self, up_to: i32) -> io::Result<f32> {
        self.get_max_score_impl(up_to)
    }
}

impl DocIdSetIterator for ReqOptSumScorer<'_> {
    fn doc_id(&self) -> i32 {
        Scorer::doc_id(self)
    }

    fn next_doc(&mut self) -> io::Result<i32> {
        if self.is_top_scores {
            let target = self.req_scorer.doc_id() + 1;
            self.advance_internal(target)
        } else {
            self.req_scorer.iterator().next_doc()
        }
    }

    fn advance(&mut self, target: i32) -> io::Result<i32> {
        if self.is_top_scores {
            self.advance_internal(target)
        } else {
            self.req_scorer.iterator().advance(target)
        }
    }

    fn cost(&self) -> i64 {
        self.req_cost
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assertables::*;

    // -----------------------------------------------------------------------
    // Mock scorer
    // -----------------------------------------------------------------------

    struct MockScorer {
        docs: Vec<i32>,
        scores: Vec<f32>,
        idx: usize,
    }

    impl MockScorer {
        fn new(docs: Vec<i32>, score: f32) -> Self {
            let scores = vec![score; docs.len()];
            Self {
                docs,
                scores,
                idx: usize::MAX, // unpositioned: doc_id() returns -1
            }
        }
    }

    impl fmt::Debug for MockScorer {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("MockScorer").finish()
        }
    }

    impl Scorable for MockScorer {
        fn score(&mut self) -> io::Result<f32> {
            if self.idx < self.scores.len() {
                Ok(self.scores[self.idx])
            } else {
                Ok(0.0)
            }
        }
    }

    impl Scorer for MockScorer {
        fn doc_id(&self) -> i32 {
            if self.idx == usize::MAX {
                -1
            } else if self.idx < self.docs.len() {
                self.docs[self.idx]
            } else {
                NO_MORE_DOCS
            }
        }
        fn iterator(&mut self) -> &mut dyn DocIdSetIterator {
            self
        }
        fn get_max_score(&mut self, _up_to: i32) -> io::Result<f32> {
            Ok(f32::MAX)
        }
    }

    impl DocIdSetIterator for MockScorer {
        fn doc_id(&self) -> i32 {
            Scorer::doc_id(self)
        }
        fn next_doc(&mut self) -> io::Result<i32> {
            if self.idx == usize::MAX {
                self.idx = 0;
            } else {
                self.idx += 1;
            }
            Ok(Scorer::doc_id(self))
        }
        fn advance(&mut self, target: i32) -> io::Result<i32> {
            if self.idx == usize::MAX {
                self.idx = 0;
            }
            while Scorer::doc_id(self) < target {
                self.idx += 1;
            }
            Ok(Scorer::doc_id(self))
        }
        fn cost(&self) -> i64 {
            self.docs.len() as i64
        }
    }

    // -----------------------------------------------------------------------
    // Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_basic_req_opt() {
        // Required: docs [1, 3, 5] with score 1.0
        // Optional: docs [2, 3, 4] with score 2.0
        // Expected: iterate [1, 3, 5], scores [1.0, 3.0, 1.0]
        let req = MockScorer::new(vec![1, 3, 5], 1.0);
        let opt = MockScorer::new(vec![2, 3, 4], 2.0);
        let mut scorer =
            ReqOptSumScorer::new(Box::new(req), Box::new(opt), ScoreMode::Complete).unwrap();

        assert_eq!(scorer.next_doc().unwrap(), 1);
        assert_eq!(Scorer::doc_id(&scorer), 1);
        assert_in_delta!(scorer.score().unwrap(), 1.0, 1e-5); // req only

        assert_eq!(scorer.next_doc().unwrap(), 3);
        assert_in_delta!(scorer.score().unwrap(), 3.0, 1e-5); // req + opt

        assert_eq!(scorer.next_doc().unwrap(), 5);
        assert_in_delta!(scorer.score().unwrap(), 1.0, 1e-5); // req only

        assert_eq!(scorer.next_doc().unwrap(), NO_MORE_DOCS);
    }

    #[test]
    fn test_no_optional_matches() {
        let req = MockScorer::new(vec![1, 2, 3], 1.5);
        let opt = MockScorer::new(vec![10, 20], 2.0);
        let mut scorer =
            ReqOptSumScorer::new(Box::new(req), Box::new(opt), ScoreMode::Complete).unwrap();

        assert_eq!(scorer.next_doc().unwrap(), 1);
        assert_in_delta!(scorer.score().unwrap(), 1.5, 1e-5);

        assert_eq!(scorer.next_doc().unwrap(), 2);
        assert_in_delta!(scorer.score().unwrap(), 1.5, 1e-5);

        assert_eq!(scorer.next_doc().unwrap(), 3);
        assert_in_delta!(scorer.score().unwrap(), 1.5, 1e-5);
    }

    #[test]
    fn test_all_optional_matches() {
        let req = MockScorer::new(vec![1, 2, 3], 1.0);
        let opt = MockScorer::new(vec![1, 2, 3], 0.5);
        let mut scorer =
            ReqOptSumScorer::new(Box::new(req), Box::new(opt), ScoreMode::Complete).unwrap();

        assert_eq!(scorer.next_doc().unwrap(), 1);
        assert_in_delta!(scorer.score().unwrap(), 1.5, 1e-5);

        assert_eq!(scorer.next_doc().unwrap(), 2);
        assert_in_delta!(scorer.score().unwrap(), 1.5, 1e-5);

        assert_eq!(scorer.next_doc().unwrap(), 3);
        assert_in_delta!(scorer.score().unwrap(), 1.5, 1e-5);
    }

    #[test]
    fn test_advance() {
        let req = MockScorer::new(vec![1, 5, 10, 15], 1.0);
        let opt = MockScorer::new(vec![5, 15], 2.0);
        let mut scorer =
            ReqOptSumScorer::new(Box::new(req), Box::new(opt), ScoreMode::Complete).unwrap();

        assert_eq!(scorer.advance(5).unwrap(), 5);
        assert_in_delta!(scorer.score().unwrap(), 3.0, 1e-5); // req + opt

        assert_eq!(scorer.advance(12).unwrap(), 15);
        assert_in_delta!(scorer.score().unwrap(), 3.0, 1e-5); // req + opt
    }

    #[test]
    fn test_doc_id_delegates_to_req() {
        let req = MockScorer::new(vec![42], 1.0);
        let opt = MockScorer::new(vec![42], 2.0);
        let mut scorer =
            ReqOptSumScorer::new(Box::new(req), Box::new(opt), ScoreMode::Complete).unwrap();

        assert_eq!(scorer.next_doc().unwrap(), 42);
        assert_eq!(Scorer::doc_id(&scorer), 42);
    }

    #[test]
    fn test_top_scores_basic() {
        // In TOP_SCORES mode with mock scorers that return f32::MAX for max_score,
        // the behavior should be identical to Complete mode since all blocks are competitive.
        let req = MockScorer::new(vec![1, 3, 5], 1.0);
        let opt = MockScorer::new(vec![2, 3, 4], 2.0);
        let mut scorer =
            ReqOptSumScorer::new(Box::new(req), Box::new(opt), ScoreMode::TopScores).unwrap();

        assert_eq!(scorer.next_doc().unwrap(), 1);
        assert_in_delta!(scorer.score().unwrap(), 1.0, 1e-5);

        assert_eq!(scorer.next_doc().unwrap(), 3);
        assert_in_delta!(scorer.score().unwrap(), 3.0, 1e-5);

        assert_eq!(scorer.next_doc().unwrap(), 5);
        assert_in_delta!(scorer.score().unwrap(), 1.0, 1e-5);

        assert_eq!(scorer.next_doc().unwrap(), NO_MORE_DOCS);
    }
}
