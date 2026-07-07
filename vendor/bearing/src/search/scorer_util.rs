// SPDX-License-Identifier: Apache-2.0

//! Utility functions for scorer-related operations.

use std::collections::BinaryHeap;
use std::io;

use super::collector::DocAndScoreAccBuffer;
use super::scorer::Scorer;
use crate::util::math_util;

/// Compute a minimum required score, so that
/// `(float) sum_upper_bound(min_required_score + max_remaining_score, num_scorers) <= min_competitive_score`.
///
/// The computed value may not be the greatest value that meets this condition, which means that
/// we may fail to filter out some docs. However, this doesn't hurt correctness, it just means
/// that these docs will be filtered out later, and the extra work required to compute an optimal
/// value would unlikely result in a speedup.
pub fn min_required_score(
    max_remaining_score: f64,
    min_competitive_score: f32,
    num_scorers: i32,
) -> f64 {
    let mut min_required_score = min_competitive_score as f64 - max_remaining_score;
    // note: we want the float ulp in order to converge faster, not the double ulp
    let subtraction = ulp_f32(min_competitive_score) as f64;
    while min_required_score > 0.0
        && (math_util::sum_upper_bound(min_required_score + max_remaining_score, num_scorers)
            as f32)
            >= min_competitive_score
    {
        min_required_score -= subtraction;
    }
    min_required_score
}

/// Computes the estimated cost of a disjunction with `min_should_match`.
///
/// The cost is the sum of the `(num_scorers - min_should_match + 1)` least costly scorers.
pub fn cost_with_min_should_match(costs: &[i64], num_scorers: usize, min_should_match: i32) -> i64 {
    // Keep the (num_scorers - minShouldMatch + 1) least costly scorers using a max-heap
    // of bounded size. Java uses a PriorityQueue with reversed lessThan (a > b) so the
    // top is the largest, and insertWithOverflow evicts the top (largest) when over capacity.
    // Rust's BinaryHeap is a max-heap natively, so we just pop when over capacity.
    let capacity = num_scorers as i64 - min_should_match as i64 + 1;
    if capacity <= 0 {
        return 0;
    }
    let capacity = capacity as usize;
    let mut pq = BinaryHeap::with_capacity(capacity + 1);
    for &cost in costs {
        pq.push(cost);
        if pq.len() > capacity {
            pq.pop(); // evict the largest
        }
    }
    pq.into_iter().sum()
}

/// Filters competitive hits from the provided `DocAndScoreAccBuffer`.
///
/// This method removes documents from the buffer that cannot possibly have a score competitive
/// enough to exceed the minimum competitive score, given the maximum remaining score and the
/// number of scorers.
pub fn filter_competitive_hits(
    buffer: &mut DocAndScoreAccBuffer,
    max_remaining_score: f64,
    min_competitive_score: f32,
    num_scorers: i32,
) {
    let min_req = min_required_score(max_remaining_score, min_competitive_score, num_scorers);

    if min_req <= 0.0 {
        return;
    }

    buffer.size = filter_by_score(&mut buffer.docs, &mut buffer.scores, min_req, buffer.size);
}

/// Apply the provided `Scorer` as a required clause on the given `DocAndScoreAccBuffer`.
/// This filters out documents from the buffer that do not match, and adds the scores of this
/// `Scorer` to the scores.
///
/// NOTE: The provided buffer must contain doc IDs in sorted order, with no duplicates.
///
/// **Departure from Java:** Java's `ScorerUtil.applyRequiredClause` takes separate
/// `DocIdSetIterator` and `Scorable` parameters. These are always extracted from the same
/// `Scorer` — the separation exists solely for JVM bimorphic dispatch optimization
/// (`likelyImpactsEnum`/`likelyTermScorer` wrappers that help JIT inlining). That
/// optimization doesn't apply to Rust. Taking a single `&mut dyn Scorer` avoids Rust
/// ownership conflicts from splitting one object into two mutable borrows, and uses
/// sequential borrows (NLL) to alternate between `iterator()` and `score()` calls.
pub fn apply_required_clause(
    buffer: &mut DocAndScoreAccBuffer,
    scorer: &mut dyn Scorer,
) -> io::Result<()> {
    let mut intersection_size = 0;
    let mut cur_doc = scorer.doc_id();
    for i in 0..buffer.size {
        let target_doc = buffer.docs[i];
        if cur_doc < target_doc {
            cur_doc = scorer.iterator().advance(target_doc)?;
        }
        if cur_doc == target_doc {
            buffer.docs[intersection_size] = target_doc;
            buffer.scores[intersection_size] = buffer.scores[i] + scorer.score()? as f64;
            intersection_size += 1;
        }
    }
    buffer.size = intersection_size;
    Ok(())
}

/// Filter parallel doc/score arrays in-place, keeping only entries where
/// `score >= min_score_inclusive`. Returns the new size.
///
/// Scalar implementation — filters entries where `score >= min_score_inclusive`.
fn filter_by_score(
    docs: &mut [i32],
    scores: &mut [f64],
    min_score_inclusive: f64,
    up_to: usize,
) -> usize {
    let mut new_size = 0;
    for i in 0..up_to {
        let doc = docs[i];
        let score = scores[i];
        docs[new_size] = doc;
        scores[new_size] = score;
        if score >= min_score_inclusive {
            new_size += 1;
        }
    }
    new_size
}

/// Returns the ULP (unit in the last place) of an `f32` value.
/// Returns the distance between `value` and the next larger `f32`.
fn ulp_f32(value: f32) -> f32 {
    if value.is_nan() {
        return f32::NAN;
    }
    if value.is_infinite() {
        return f32::INFINITY;
    }
    let abs = value.abs();
    let next = f32::from_bits(abs.to_bits() + 1);
    next - abs
}

#[cfg(test)]
mod tests {
    use std::fmt;

    use super::*;
    use crate::search::collector::DocAndFloatFeatureBuffer;
    use crate::search::doc_id_set_iterator::{DocIdSetIterator, NO_MORE_DOCS};
    use crate::search::scorable::Scorable;
    use assertables::*;

    #[test]
    fn test_ulp_f32_one() {
        let u = ulp_f32(1.0);
        assert_gt!(u, 0.0);
        // ULP of 1.0f32 is 2^-23
        assert_in_delta!(u, 1.1920929e-7, 1e-14);
    }

    #[test]
    fn test_ulp_f32_zero() {
        let u = ulp_f32(0.0);
        assert_gt!(u, 0.0);
        // Smallest positive f32 (subnormal)
        assert_eq!(u, f32::from_bits(1));
    }

    #[test]
    fn test_filter_by_score() {
        let mut docs = vec![1, 2, 3, 4, 5];
        let mut scores = vec![0.5, 1.5, 0.3, 2.0, 0.8];
        let new_size = filter_by_score(&mut docs, &mut scores, 1.0, 5);
        assert_eq!(new_size, 2);
        assert_eq!(docs[0], 2);
        assert_eq!(docs[1], 4);
        assert_in_delta!(scores[0], 1.5, 1e-10);
        assert_in_delta!(scores[1], 2.0, 1e-10);
    }

    #[test]
    fn test_filter_by_score_all_pass() {
        let mut docs = vec![1, 2, 3];
        let mut scores = vec![1.0, 2.0, 3.0];
        let new_size = filter_by_score(&mut docs, &mut scores, 0.5, 3);
        assert_eq!(new_size, 3);
    }

    #[test]
    fn test_filter_by_score_none_pass() {
        let mut docs = vec![1, 2, 3];
        let mut scores = vec![0.1, 0.2, 0.3];
        let new_size = filter_by_score(&mut docs, &mut scores, 1.0, 3);
        assert_eq!(new_size, 0);
    }

    #[test]
    fn test_min_required_score_no_filtering() {
        // When max_remaining_score is huge, min_required_score should be <= 0
        let result = min_required_score(100.0, 1.0, 3);
        assert_le!(result, 0.0);
    }

    #[test]
    fn test_min_required_score_tight() {
        // When max_remaining_score is 0, min_required_score should be close to min_competitive
        let result = min_required_score(0.0, 1.0, 3);
        assert_gt!(result, 0.0);
    }

    #[test]
    fn test_cost_with_min_should_match_msm_1() {
        // msm=1, 3 scorers: capacity = 3-1+1 = 3, keeps all -> sum = 60
        let costs = vec![10, 20, 30];
        let result = cost_with_min_should_match(&costs, 3, 1);
        assert_eq!(result, 60);
    }

    #[test]
    fn test_cost_with_min_should_match_msm_2() {
        // msm=2, 3 scorers: capacity = 3-2+1 = 2, keeps 2 least costly [10, 20] -> sum = 30
        let costs = vec![10, 20, 30];
        let result = cost_with_min_should_match(&costs, 3, 2);
        assert_eq!(result, 30);
    }

    #[test]
    fn test_cost_with_min_should_match_msm_equals_num() {
        // msm=3, 3 scorers: capacity = 3-3+1 = 1, keeps the least costly [10]
        let costs = vec![10, 20, 30];
        let result = cost_with_min_should_match(&costs, 3, 3);
        assert_eq!(result, 10);
    }

    #[test]
    fn test_cost_with_min_should_match_single_scorer() {
        let costs = vec![42];
        let result = cost_with_min_should_match(&costs, 1, 1);
        assert_eq!(result, 42);
    }

    #[test]
    fn test_cost_with_min_should_match_zero_capacity() {
        // msm > num_scorers: capacity <= 0 -> returns 0
        let costs = vec![10, 20];
        let result = cost_with_min_should_match(&costs, 2, 3);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_filter_competitive_hits_none_filtered() {
        let mut buffer = DocAndScoreAccBuffer::new();
        buffer.grow_no_copy(3);
        buffer.docs[0] = 1;
        buffer.docs[1] = 2;
        buffer.docs[2] = 3;
        buffer.scores[0] = 5.0;
        buffer.scores[1] = 6.0;
        buffer.scores[2] = 7.0;
        buffer.size = 3;

        // max_remaining_score is huge, so nothing should be filtered
        filter_competitive_hits(&mut buffer, 100.0, 1.0, 3);
        assert_eq!(buffer.size, 3);
    }

    #[test]
    fn test_apply_required_clause() {
        // Mock scorer that matches docs 1, 3, 5 with score 0.5
        struct MockScorer {
            docs: Vec<i32>,
            idx: usize,
        }
        impl fmt::Debug for MockScorer {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_struct("MockScorer").finish()
            }
        }
        impl Scorable for MockScorer {
            fn score(&mut self) -> io::Result<f32> {
                Ok(0.5)
            }
        }
        impl Scorer for MockScorer {
            fn doc_id(&self) -> i32 {
                if self.idx < self.docs.len() {
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
            fn next_docs_and_scores(
                &mut self,
                _up_to: i32,
                _buffer: &mut DocAndFloatFeatureBuffer,
            ) -> io::Result<()> {
                Ok(())
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

        let mut buffer = DocAndScoreAccBuffer::new();
        buffer.grow_no_copy(5);
        buffer.docs[0] = 1;
        buffer.docs[1] = 2;
        buffer.docs[2] = 3;
        buffer.docs[3] = 4;
        buffer.docs[4] = 5;
        buffer.scores[0] = 1.0;
        buffer.scores[1] = 1.0;
        buffer.scores[2] = 1.0;
        buffer.scores[3] = 1.0;
        buffer.scores[4] = 1.0;
        buffer.size = 5;

        let mut scorer = MockScorer {
            docs: vec![1, 3, 5],
            idx: 0,
        };

        apply_required_clause(&mut buffer, &mut scorer).unwrap();

        // Only docs 1, 3, 5 match — scores should be 1.0 + 0.5 = 1.5
        assert_eq!(buffer.size, 3);
        assert_eq!(buffer.docs[0], 1);
        assert_eq!(buffer.docs[1], 3);
        assert_eq!(buffer.docs[2], 5);
        assert_in_delta!(buffer.scores[0], 1.5, 1e-10);
        assert_in_delta!(buffer.scores[1], 1.5, 1e-10);
        assert_in_delta!(buffer.scores[2], 1.5, 1e-10);
    }
}
