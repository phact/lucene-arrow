// SPDX-License-Identifier: Apache-2.0

//! Min-heap of `DisiWrapper` indices ordered by current doc ID.
//!
//! Deviations from Lucene's `DisiPriorityQueue`:
//! - Stores `(idx, doc)` pairs rather than `DisiWrapper` references. Callers own
//!   wrappers in a parallel `Vec<DisiWrapper>`; the queue holds u32 indices.
//!   This avoids self-referential borrows around `Box<dyn Scorer>` without `unsafe`.
//! - `top_list()` returns `Vec<u32>` indices instead of a linked list via
//!   `DisiWrapper.next`. Rust's no-unsafe rule prevents a safe mutable linked
//!   list; the slice form preserves behavior (matching subs at the top doc).
//! - Backed by `std::collections::BinaryHeap`. `top2()` and `top_list()` scan
//!   all entries (`O(n)`) because stdlib does not expose the heap's internal
//!   tree layout. N is typically small (handful of clauses per boolean query).

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fmt;

/// Heap entry pairing a `DisiWrapper` index with its cached doc.
///
/// Ordered descending so that `BinaryHeap` (a max-heap) yields the smallest doc
/// at the top — a min-heap by doc.
#[derive(Copy, Clone, Debug)]
pub struct Entry {
    pub idx: u32,
    pub doc: i32,
}

impl PartialEq for Entry {
    fn eq(&self, other: &Self) -> bool {
        self.doc == other.doc
    }
}

impl Eq for Entry {}

impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Entry {
    fn cmp(&self, other: &Self) -> Ordering {
        other.doc.cmp(&self.doc)
    }
}

/// A priority queue of `DisiWrapper` indices, ordered by current doc ID (min-heap).
pub struct DisiPriorityQueue {
    heap: BinaryHeap<Entry>,
    max_size: usize,
}

impl fmt::Debug for DisiPriorityQueue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DisiPriorityQueue")
            .field("size", &self.heap.len())
            .field("max_size", &self.max_size)
            .finish()
    }
}

impl DisiPriorityQueue {
    /// Create a queue with the given maximum capacity.
    pub fn of_max_size(max_size: usize) -> Self {
        Self {
            heap: BinaryHeap::with_capacity(max_size),
            max_size,
        }
    }

    /// Number of entries in the queue.
    pub fn size(&self) -> usize {
        self.heap.len()
    }

    /// Returns true if the queue has no entries.
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Index of the entry with the smallest doc, or `None` if empty.
    pub fn top(&self) -> Option<u32> {
        self.heap.peek().map(|e| e.idx)
    }

    /// Doc of the top entry, or `None` if empty.
    pub fn top_doc(&self) -> Option<i32> {
        self.heap.peek().map(|e| e.doc)
    }

    /// Index of the 2nd smallest entry, or `None` if fewer than two entries.
    pub fn top2(&self) -> Option<u32> {
        let top_doc = self.top_doc()?;
        let mut skipped_top = false;
        let mut best: Option<Entry> = None;
        for &e in self.heap.iter() {
            if !skipped_top && e.doc == top_doc {
                skipped_top = true;
                continue;
            }
            best = match best {
                None => Some(e),
                Some(b) if e.doc < b.doc => Some(e),
                other => other,
            };
        }
        best.map(|e| e.idx)
    }

    /// Indices of all entries whose doc equals the top's doc.
    pub fn top_list(&self) -> Vec<u32> {
        let Some(top_doc) = self.top_doc() else {
            return Vec::new();
        };
        self.heap
            .iter()
            .filter(|e| e.doc == top_doc)
            .map(|e| e.idx)
            .collect()
    }

    /// Add an entry and return the new top's index.
    ///
    /// Panics if adding would exceed `max_size`.
    pub fn add(&mut self, idx: u32, doc: i32) -> u32 {
        assert!(
            self.heap.len() < self.max_size,
            "DisiPriorityQueue::add: would exceed max_size={}",
            self.max_size
        );
        self.heap.push(Entry { idx, doc });
        self.heap.peek().unwrap().idx
    }

    /// Bulk-add entries. Panics if the total would exceed `max_size`.
    pub fn add_all(&mut self, entries: impl IntoIterator<Item = (u32, i32)>) {
        for (idx, doc) in entries {
            self.add(idx, doc);
        }
    }

    /// Remove and return the top entry's index, or `None` if empty.
    pub fn pop(&mut self) -> Option<u32> {
        self.heap.pop().map(|e| e.idx)
    }

    /// Update the top entry's cached doc to `new_doc`, rebalance, and return
    /// the new top's index (may differ if the old top was demoted).
    pub fn update_top(&mut self, new_doc: i32) -> Option<u32> {
        if let Some(mut peek) = self.heap.peek_mut() {
            peek.doc = new_doc;
        }
        self.heap.peek().map(|e| e.idx)
    }

    /// Replace the top entry with `(new_idx, new_doc)`, rebalance, and return
    /// the new top's index.
    ///
    /// Panics if the queue is empty.
    pub fn update_top_with(&mut self, new_idx: u32, new_doc: i32) -> u32 {
        {
            let mut peek = self.heap.peek_mut().expect("queue is empty");
            peek.idx = new_idx;
            peek.doc = new_doc;
        }
        self.heap.peek().unwrap().idx
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.heap.clear();
    }

    /// Iterator over entries in arbitrary order.
    pub fn iter(&self) -> impl Iterator<Item = &Entry> + '_ {
        self.heap.iter()
    }
}

#[cfg(test)]
mod tests {
    use std::panic;

    use assertables::*;

    use super::*;

    /// Ported from `TestDisiPriorityQueue.testDisiPriorityQueue2`.
    /// Uses `max_size=3` since our queue has no 2-element specialization;
    /// the 3rd entry exercises the overflow guard.
    #[test]
    fn test_basic_and_update_top() {
        let mut pq = DisiPriorityQueue::of_max_size(3);
        assert_none!(pq.top());
        assert_eq!(pq.size(), 0);

        // idx 0 at doc 1
        assert_eq!(pq.add(0, 1), 0);
        assert_eq!(pq.top(), Some(0));
        assert_eq!(pq.size(), 1);

        // idx 1 at doc 0 becomes new top (smaller doc)
        assert_eq!(pq.add(1, 0), 1);
        assert_eq!(pq.top(), Some(1));
        assert_eq!(pq.size(), 2);

        // Update top (idx 1) from doc 0 to doc 1.
        // Both entries now at doc=1; top_list should contain both.
        pq.update_top(1);
        let mut top_list = pq.top_list();
        top_list.sort();
        assert_eq!(top_list, vec![0, 1]);

        // Update the current top to doc 2; the other (doc=1) becomes top.
        pq.update_top(2);
        assert_eq!(pq.top_doc(), Some(1));
        assert_eq!(pq.top_list().len(), 1);

        // Pop the top (doc 1); remaining entry (doc 2) becomes top.
        let popped = pq.pop().unwrap();
        assert_eq!(pq.size(), 1);
        assert_eq!(pq.top_doc(), Some(2));
        // popped was the idx with doc=1, and the remaining entry has doc=2
        assert_ne!(pq.top(), Some(popped));
    }

    #[test]
    fn test_overflow_panics() {
        let mut pq = DisiPriorityQueue::of_max_size(2);
        pq.add(0, 1);
        pq.add(1, 0);
        let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            pq.add(2, 2);
        }));
        assert_err!(result);
    }

    #[test]
    fn test_clear_and_empty() {
        let mut pq = DisiPriorityQueue::of_max_size(5);
        pq.add(0, 10);
        pq.add(1, 5);
        pq.add(2, 20);
        assert_eq!(pq.size(), 3);
        pq.clear();
        assert_eq!(pq.size(), 0);
        assert!(pq.is_empty());
        assert_none!(pq.top());
    }

    #[test]
    fn test_top2() {
        let mut pq = DisiPriorityQueue::of_max_size(5);
        assert_none!(pq.top2());
        pq.add(0, 10);
        assert_none!(pq.top2());
        pq.add(1, 5);
        assert_eq!(pq.top2(), Some(0));
        pq.add(2, 20);
        assert_eq!(pq.top2(), Some(0));
        pq.add(3, 7);
        // Top is idx=1 (doc=5); 2nd smallest is idx=3 (doc=7)
        assert_eq!(pq.top2(), Some(3));
    }

    #[test]
    fn test_top_list_single_at_top() {
        let mut pq = DisiPriorityQueue::of_max_size(3);
        pq.add(0, 5);
        pq.add(1, 10);
        pq.add(2, 7);
        assert_eq!(pq.top_list(), vec![0]);
    }

    #[test]
    fn test_top_list_multiple_at_top() {
        let mut pq = DisiPriorityQueue::of_max_size(5);
        pq.add(0, 5);
        pq.add(1, 5);
        pq.add(2, 5);
        pq.add(3, 10);
        let mut tl = pq.top_list();
        tl.sort();
        assert_eq!(tl, vec![0, 1, 2]);
    }

    #[test]
    fn test_update_top_with_replaces_top() {
        let mut pq = DisiPriorityQueue::of_max_size(3);
        pq.add(0, 5);
        pq.add(1, 10);
        // Replace top (idx=0, doc=5) with (idx=99, doc=15).
        // Now idx=1 (doc=10) should be top.
        pq.update_top_with(99, 15);
        assert_eq!(pq.top(), Some(1));
        assert_eq!(pq.top_doc(), Some(10));
        // Both entries still present.
        let all: Vec<u32> = pq.iter().map(|e| e.idx).collect();
        assert_eq!(all.len(), 2);
        assert!(all.contains(&1));
        assert!(all.contains(&99));
    }

    /// Ported from `TestDisiPriorityQueue.testRandom`.
    /// Simulates iterating multiple sub-iterators in doc order by advancing
    /// the heap's top.
    #[test]
    fn test_random_merges_sorted_docs() {
        // Build 10 wrappers each with sorted-unique doc sequences.
        let size = 10u32;
        let mut docs_per_wrapper: Vec<Vec<i32>> = Vec::new();
        for i in 0..size {
            let mut seq: Vec<i32> = (0..8).map(|k| (k * 7 + i as i32 * 3) % 60).collect();
            seq.sort();
            seq.dedup();
            docs_per_wrapper.push(seq);
        }

        // Positions into each wrapper's doc sequence.
        let mut positions = vec![0usize; size as usize];

        let mut pq = DisiPriorityQueue::of_max_size(size as usize);
        for i in 0..size {
            let seq = &docs_per_wrapper[i as usize];
            if !seq.is_empty() {
                pq.add(i, seq[0]);
            }
        }

        let mut expected: Vec<i32> = docs_per_wrapper.iter().flatten().copied().collect();
        expected.sort();

        let mut observed: Vec<i32> = Vec::new();
        while let Some(top_idx) = pq.top() {
            let top_doc = pq.top_doc().unwrap();
            observed.push(top_doc);

            let i = top_idx as usize;
            positions[i] += 1;
            let seq = &docs_per_wrapper[i];
            if positions[i] < seq.len() {
                pq.update_top(seq[positions[i]]);
            } else {
                pq.pop();
            }
        }

        assert_eq!(observed, expected);
    }
}
