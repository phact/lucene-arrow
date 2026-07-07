// SPDX-License-Identifier: Apache-2.0

//! A `DocIdSetIterator` which is a disjunction of the approximations of the provided iterators.

use std::cmp::Reverse;
use std::io;

use crate::search::disi_priority_queue::DisiPriorityQueue;
use crate::search::disi_wrapper::DisiWrapper;
use crate::search::doc_id_set_iterator::DocIdSetIterator;

/// A `DocIdSetIterator` which is a disjunction of the approximations of the provided iterators.
///
/// To avoid `O(N * log(N))` reordering overhead per `advance()` when intersecting with a
/// selective filter, only clauses whose costs fit in `1.5 * lead_cost` go into the priority
/// queue; the rest are scanned linearly on every call.
///
/// Deviations from Lucene:
/// - Uses index-based access instead of `DisiWrapper` references; `top_list` returns
///   `Vec<u32>` of indices rather than a linked list.
/// - Lucene's `approximation` field (set from `TwoPhaseIterator.approximation()` or the
///   iterator itself) is not modeled separately; we use `scorer.iterator()` directly since
///   `TwoPhaseIterator` is not yet ported.
/// - `fill_bit_set` is not overridden (uses trait default).
pub struct DisjunctionDISIApproximation<'a> {
    wrappers: Vec<DisiWrapper<'a>>,
    lead_iterators: DisiPriorityQueue,
    other_indices: Box<[u32]>,
    cost: i64,
    lead_top: u32,
    min_other_doc: i32,
    doc: i32,
}

impl<'a> DisjunctionDISIApproximation<'a> {
    /// Construct from a non-empty collection of sub-iterator wrappers.
    ///
    /// Panics if `sub_iterators` is empty.
    pub fn new(sub_iterators: Vec<DisiWrapper<'a>>, lead_cost: i64) -> Self {
        assert!(
            !sub_iterators.is_empty(),
            "DisjunctionDISIApproximation requires at least one sub-iterator"
        );

        let mut wrappers = sub_iterators;
        wrappers.sort_by_key(|w| Reverse(w.cost));

        let mut reorder_threshold = lead_cost.wrapping_add(lead_cost >> 1);
        if reorder_threshold < 0 {
            reorder_threshold = i64::MAX;
        }

        let mut cost: i64 = 0;
        let mut reorder_cost: i64 = 0;
        let mut last_idx: i32 = wrappers.len() as i32 - 1;
        while last_idx >= 0 {
            let last_cost = wrappers[last_idx as usize].cost;
            let inc = last_cost.min(lead_cost);
            let sum = reorder_cost.wrapping_add(inc);
            if sum < 0 || sum > reorder_threshold {
                break;
            }
            reorder_cost = sum;
            cost += last_cost;
            last_idx -= 1;
        }

        if last_idx == wrappers.len() as i32 - 1 {
            cost += wrappers[last_idx as usize].cost;
            last_idx -= 1;
        }

        debug_assert!(last_idx >= -1 && last_idx < wrappers.len() as i32 - 1);
        let pq_len = (wrappers.len() as i32 - last_idx - 1) as usize;
        let mut lead_iterators = DisiPriorityQueue::of_max_size(pq_len);
        for (i, w) in wrappers.iter().enumerate().skip((last_idx + 1) as usize) {
            lead_iterators.add(i as u32, w.doc);
        }

        let other_indices: Box<[u32]> = (0..=last_idx)
            .map(|i| i as u32)
            .collect::<Vec<_>>()
            .into_boxed_slice();

        let mut min_other_doc = i32::MAX;
        for &i in other_indices.iter() {
            cost += wrappers[i as usize].cost;
            min_other_doc = min_other_doc.min(wrappers[i as usize].doc);
        }

        let lead_top = lead_iterators
            .top()
            .expect("leadIterators must be non-empty");

        Self {
            wrappers,
            lead_iterators,
            other_indices,
            cost,
            lead_top,
            min_other_doc,
            doc: -1,
        }
    }

    fn lead_top_doc(&self) -> i32 {
        self.wrappers[self.lead_top as usize].doc
    }

    /// Return the indices of sub-iterators positioned on the current doc.
    pub fn top_list(&self) -> Vec<u32> {
        if self.lead_top_doc() < self.min_other_doc {
            self.lead_iterators.top_list()
        } else {
            self.compute_top_list()
        }
    }

    fn compute_top_list(&self) -> Vec<u32> {
        debug_assert!(self.lead_top_doc() >= self.min_other_doc);
        let mut list = Vec::new();
        if self.lead_top_doc() == self.min_other_doc {
            list.extend(self.lead_iterators.top_list());
        }
        for &idx in self.other_indices.iter() {
            if self.wrappers[idx as usize].doc == self.min_other_doc {
                list.push(idx);
            }
        }
        list
    }

    /// Shared access to sub-iterator wrappers.
    pub fn wrappers(&self) -> &[DisiWrapper<'a>] {
        &self.wrappers
    }

    /// Mutable access to sub-iterator wrappers (for scoring).
    pub fn wrappers_mut(&mut self) -> &mut [DisiWrapper<'a>] {
        &mut self.wrappers
    }
}

impl<'a> DocIdSetIterator for DisjunctionDISIApproximation<'a> {
    fn doc_id(&self) -> i32 {
        self.doc
    }

    fn next_doc(&mut self) -> io::Result<i32> {
        if self.lead_top_doc() < self.min_other_doc {
            let cur_doc = self.lead_top_doc();
            loop {
                let idx = self.lead_top as usize;
                let new_doc = self.wrappers[idx].scorer.iterator().next_doc()?;
                self.wrappers[idx].doc = new_doc;
                self.lead_top = self
                    .lead_iterators
                    .update_top(new_doc)
                    .expect("leadIterators non-empty");
                if self.lead_top_doc() != cur_doc {
                    break;
                }
            }
            self.doc = self.lead_top_doc().min(self.min_other_doc);
            Ok(self.doc)
        } else {
            self.advance(self.min_other_doc + 1)
        }
    }

    fn advance(&mut self, target: i32) -> io::Result<i32> {
        while self.lead_top_doc() < target {
            let idx = self.lead_top as usize;
            let new_doc = self.wrappers[idx].scorer.iterator().advance(target)?;
            self.wrappers[idx].doc = new_doc;
            self.lead_top = self
                .lead_iterators
                .update_top(new_doc)
                .expect("leadIterators non-empty");
        }

        self.min_other_doc = i32::MAX;
        for i in 0..self.other_indices.len() {
            let idx = self.other_indices[i] as usize;
            if self.wrappers[idx].doc < target {
                let new_doc = self.wrappers[idx].scorer.iterator().advance(target)?;
                self.wrappers[idx].doc = new_doc;
            }
            self.min_other_doc = self.min_other_doc.min(self.wrappers[idx].doc);
        }

        self.doc = self.lead_top_doc().min(self.min_other_doc);
        Ok(self.doc)
    }

    fn cost(&self) -> i64 {
        self.cost
    }

    fn doc_id_run_end(&mut self) -> io::Result<i32> {
        let mut max_run_end = self.doc + 1;
        for idx in self.top_list() {
            let run_end = self.wrappers[idx as usize]
                .scorer
                .iterator()
                .doc_id_run_end()?;
            max_run_end = max_run_end.max(run_end);
        }
        Ok(max_run_end)
    }
}

#[cfg(test)]
mod tests {
    use std::fmt;

    use super::*;
    use crate::search::doc_id_set_iterator::{self, NO_MORE_DOCS, RangeDocIdSetIterator};
    use crate::search::scorable::Scorable;
    use crate::search::scorer::Scorer;

    /// Simple mock scorer backed by a sorted Vec<i32> of doc IDs.
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

    fn wrappers_for(seqs: Vec<Vec<i32>>) -> Vec<DisiWrapper<'static>> {
        seqs.into_iter()
            .map(|docs| DisiWrapper::new(Box::new(MockScorer::new(docs))))
            .collect()
    }

    /// Mock scorer that wraps a `RangeDocIdSetIterator` and inherits its optimized
    /// `doc_id_run_end` (returns the range max, not just `doc + 1`).
    struct RangeScorer {
        range: RangeDocIdSetIterator,
    }

    impl RangeScorer {
        fn new(min: i32, max: i32) -> Self {
            Self {
                range: doc_id_set_iterator::range(min, max),
            }
        }
    }

    impl fmt::Debug for RangeScorer {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("RangeScorer").finish()
        }
    }

    impl Scorable for RangeScorer {
        fn score(&mut self) -> io::Result<f32> {
            Ok(1.0)
        }
    }

    impl Scorer for RangeScorer {
        fn doc_id(&self) -> i32 {
            self.range.doc_id()
        }
        fn iterator(&mut self) -> &mut dyn DocIdSetIterator {
            &mut self.range
        }
        fn get_max_score(&mut self, _up_to: i32) -> io::Result<f32> {
            Ok(1.0)
        }
    }

    fn union_sorted(seqs: &[Vec<i32>]) -> Vec<i32> {
        let mut all: Vec<i32> = seqs.iter().flatten().copied().collect();
        all.sort();
        all.dedup();
        all
    }

    #[test]
    fn test_two_subs() {
        let seqs = vec![vec![1, 3, 5], vec![2, 3, 4]];
        let wrappers = wrappers_for(seqs.clone());
        let mut disi = DisjunctionDISIApproximation::new(wrappers, 10);

        let expected = union_sorted(&seqs);
        let mut observed = Vec::new();
        loop {
            let doc = disi.next_doc().unwrap();
            if doc == NO_MORE_DOCS {
                break;
            }
            observed.push(doc);
        }
        assert_eq!(observed, expected);
    }

    #[test]
    fn test_five_subs_next_doc_yields_union() {
        let seqs = vec![
            vec![1, 10, 20],
            vec![2, 10, 21],
            vec![3, 11, 20],
            vec![4, 12, 22],
            vec![5, 13, 20],
        ];
        let wrappers = wrappers_for(seqs.clone());
        let mut disi = DisjunctionDISIApproximation::new(wrappers, 3);

        let expected = union_sorted(&seqs);
        let mut observed = Vec::new();
        loop {
            let doc = disi.next_doc().unwrap();
            if doc == NO_MORE_DOCS {
                break;
            }
            observed.push(doc);
        }
        assert_eq!(observed, expected);
    }

    #[test]
    fn test_ten_subs_next_doc_yields_union() {
        let mut seqs: Vec<Vec<i32>> = Vec::new();
        for i in 0..10u32 {
            let mut s: Vec<i32> = (0..8).map(|k| (k * 7 + i as i32 * 3) % 60).collect();
            s.sort();
            s.dedup();
            seqs.push(s);
        }
        let wrappers = wrappers_for(seqs.clone());
        let mut disi = DisjunctionDISIApproximation::new(wrappers, 5);

        let expected = union_sorted(&seqs);
        let mut observed = Vec::new();
        loop {
            let doc = disi.next_doc().unwrap();
            if doc == NO_MORE_DOCS {
                break;
            }
            observed.push(doc);
        }
        assert_eq!(observed, expected);
    }

    #[test]
    fn test_advance_skips_forward() {
        let seqs = vec![vec![1, 5, 10, 20], vec![2, 6, 11, 21], vec![3, 7, 12, 22]];
        let wrappers = wrappers_for(seqs);
        let mut disi = DisjunctionDISIApproximation::new(wrappers, 10);

        assert_eq!(disi.advance(15).unwrap(), 20);
        assert_eq!(disi.next_doc().unwrap(), 21);
        assert_eq!(disi.next_doc().unwrap(), 22);
        assert_eq!(disi.next_doc().unwrap(), NO_MORE_DOCS);
    }

    #[test]
    fn test_top_list_returns_matching_indices() {
        // Two subs at doc 5, one sub at doc 10: top_list at doc 5 should be the two matching.
        let seqs = vec![vec![5, 10], vec![5, 7], vec![6, 10]];
        let wrappers = wrappers_for(seqs);
        let mut disi = DisjunctionDISIApproximation::new(wrappers, 5);

        let doc = disi.next_doc().unwrap();
        assert_eq!(doc, 5);
        let tl = disi.top_list();
        assert_eq!(tl.len(), 2);

        // Each wrapper at current doc should be at doc 5.
        for &idx in &tl {
            assert_eq!(disi.wrappers()[idx as usize].doc, 5);
        }
    }

    #[test]
    fn test_cost_is_sum_of_sub_costs() {
        // Three iterators with costs 3, 3, 3 → total should be 9.
        let seqs = vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]];
        let wrappers = wrappers_for(seqs);
        let disi = DisjunctionDISIApproximation::new(wrappers, 3);
        assert_eq!(disi.cost(), 9);
    }

    #[test]
    #[should_panic(expected = "at least one sub-iterator")]
    fn test_empty_panics() {
        let wrappers: Vec<DisiWrapper<'static>> = Vec::new();
        let _ = DisjunctionDISIApproximation::new(wrappers, 10);
    }

    /// Ported from `TestDisjunctionDISIApproximation.testDocIDRunEnd`.
    /// Verifies that `doc_id_run_end` reports the longest run across sub-iterators
    /// at the current doc.
    #[test]
    fn test_doc_id_run_end() {
        let wrappers: Vec<DisiWrapper<'static>> = vec![
            DisiWrapper::new(Box::new(RangeScorer::new(10_000, 30_000))),
            DisiWrapper::new(Box::new(RangeScorer::new(20_000, 50_000))),
            DisiWrapper::new(Box::new(RangeScorer::new(60_000, 60_001))),
        ];
        let lead_cost = 100_000;
        let mut iter = DisjunctionDISIApproximation::new(wrappers, lead_cost);

        assert_eq!(iter.next_doc().unwrap(), 10_000);
        assert_eq!(iter.doc_id_run_end().unwrap(), 30_000);

        assert_eq!(iter.advance(25_000).unwrap(), 25_000);
        assert_eq!(iter.doc_id_run_end().unwrap(), 50_000);

        assert_eq!(iter.advance(50_000).unwrap(), 60_000);
        assert_eq!(iter.doc_id_run_end().unwrap(), 60_001);
    }

    #[test]
    fn test_doc_id_run_end_default_is_doc_plus_one() {
        // Sub-iterators backed by MockScorer don't override doc_id_run_end → default doc+1.
        // top_list at doc 5 has 2 sub-iterators, both return 6. max = max(doc+1=6, 6, 6) = 6.
        let seqs = vec![vec![5, 10], vec![5, 7]];
        let wrappers = wrappers_for(seqs);
        let mut disi = DisjunctionDISIApproximation::new(wrappers, 5);
        assert_eq!(disi.next_doc().unwrap(), 5);
        assert_eq!(disi.doc_id_run_end().unwrap(), 6);
    }

    #[test]
    fn test_doc_id_initial_value() {
        let wrappers = wrappers_for(vec![vec![1], vec![2]]);
        let disi = DisjunctionDISIApproximation::new(wrappers, 1);
        assert_eq!(disi.doc_id(), -1);
    }

    #[test]
    fn test_advance_to_first_doc() {
        let seqs = vec![vec![10, 20], vec![15, 25]];
        let wrappers = wrappers_for(seqs);
        let mut disi = DisjunctionDISIApproximation::new(wrappers, 5);
        assert_eq!(disi.advance(0).unwrap(), 10);
        assert_eq!(disi.doc_id(), 10);
    }

    #[test]
    fn test_advance_past_all_returns_no_more_docs() {
        let seqs = vec![vec![1, 2, 3], vec![4, 5]];
        let wrappers = wrappers_for(seqs);
        let mut disi = DisjunctionDISIApproximation::new(wrappers, 5);
        assert_eq!(disi.advance(100).unwrap(), NO_MORE_DOCS);
        assert_eq!(disi.doc_id(), NO_MORE_DOCS);
    }

    #[test]
    fn test_top_list_after_advance() {
        // 3 wrappers — only those at the current doc should be in top_list.
        let seqs = vec![vec![5, 10], vec![5, 12], vec![6, 10]];
        let wrappers = wrappers_for(seqs);
        let mut disi = DisjunctionDISIApproximation::new(wrappers, 3);
        assert_eq!(disi.next_doc().unwrap(), 5);
        let tl = disi.top_list();
        assert_eq!(tl.len(), 2);
        assert!(tl.iter().all(|&i| disi.wrappers()[i as usize].doc == 5));
    }

    #[test]
    fn test_wrappers_accessor_count() {
        let seqs = vec![vec![1], vec![2], vec![3], vec![4]];
        let wrappers = wrappers_for(seqs);
        let disi = DisjunctionDISIApproximation::new(wrappers, 5);
        assert_eq!(disi.wrappers().len(), 4);
    }
}
