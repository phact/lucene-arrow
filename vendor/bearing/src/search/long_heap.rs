// SPDX-License-Identifier: Apache-2.0

//! A min-heap of `i64` values. Wraps Rust's `BinaryHeap<Reverse<i64>>`

use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// A min heap that stores `i64` values; a primitive priority queue that maintains a partial
/// ordering of its elements such that the least element can always be found in constant time.
/// Push and pop require O(log n). This heap provides unbounded growth via [`push`](Self::push),
/// and bounded-size insertion based on its nominal `max_size` via
/// [`insert_with_overflow`](Self::insert_with_overflow).
#[derive(Debug)]
pub struct LongHeap {
    max_size: usize,
    heap: BinaryHeap<Reverse<i64>>,
}

impl LongHeap {
    /// Create an empty priority queue of the configured initial size.
    ///
    /// # Panics
    ///
    /// Panics if `max_size` is zero.
    pub fn new(max_size: usize) -> Self {
        assert!(max_size >= 1, "max_size must be > 0; got: {}", max_size);
        Self {
            max_size,
            heap: BinaryHeap::with_capacity(max_size),
        }
    }

    /// Constructs a heap with specified size and initializes all elements with the given value.
    ///
    /// # Panics
    ///
    /// Panics if `size` is zero.
    pub fn new_with_initial_value(size: usize, initial_value: i64) -> Self {
        let mut h = Self::new(size);
        for _ in 0..size {
            h.heap.push(Reverse(initial_value));
        }
        h
    }

    /// Adds a value in log(size) time. Grows unbounded as needed to accommodate new values.
    ///
    /// Returns the new 'top' element in the queue.
    pub fn push(&mut self, element: i64) -> i64 {
        self.heap.push(Reverse(element));
        self.top()
    }

    /// Adds a value in log(size) time. If the number of values would exceed the heap's
    /// `max_size`, the least value is discarded.
    ///
    /// Returns whether the value was added (unless the heap is full and the new value is less
    /// than or equal to the top value).
    pub fn insert_with_overflow(&mut self, value: i64) -> bool {
        if self.heap.len() >= self.max_size {
            if value < self.top() {
                return false;
            }
            self.update_top(value);
            return true;
        }
        self.push(value);
        true
    }

    /// Returns the least element of the heap in constant time.
    ///
    /// If no elements have been added, returns 0.
    pub fn top(&self) -> i64 {
        self.heap.peek().map_or(0, |r| r.0)
    }

    /// Removes and returns the least element of the heap in log(size) time.
    ///
    /// # Panics
    ///
    /// Panics if the heap is empty.
    pub fn pop(&mut self) -> i64 {
        self.heap.pop().expect("The heap is empty").0
    }

    /// Replace the top of the heap with `value`. Still log(n) worst case, but it's at least
    /// twice as fast as pop + push.
    ///
    /// Returns the new 'top' element after shuffling the heap.
    pub fn update_top(&mut self, value: i64) -> i64 {
        self.heap.pop();
        self.heap.push(Reverse(value));
        self.top()
    }

    /// Returns the number of elements currently stored in the heap.
    pub fn size(&self) -> usize {
        self.heap.len()
    }

    /// Removes all entries from the heap.
    pub fn clear(&mut self) {
        self.heap.clear();
    }

    /// Return the element at the `i`th location in the heap array. Used for iterating over
    /// elements when the order doesn't matter. Valid arguments range from `[1, size]`.
    ///
    /// Note: This is O(n) because BinaryHeap doesn't support random access. The underlying
    /// slice is accessed directly via `into_vec` would destroy the heap. Instead we use
    /// the iterator. This method is only used by `top_docs_size` which iterates all elements.
    ///
    /// # Panics
    ///
    /// Panics if `i` is out of range.
    pub fn get(&self, i: usize) -> i64 {
        assert!(
            i >= 1 && i <= self.heap.len(),
            "index {} out of range [1, {}]",
            i,
            self.heap.len()
        );
        // BinaryHeap stores as a Vec internally; we can iterate it
        // The i-th element (1-based) maps to the (i-1)th element in the internal slice
        let slice = self.heap.iter().collect::<Vec<_>>();
        slice[i - 1].0
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use assertables::*;

    #[test]
    fn test_new_empty() {
        let heap = LongHeap::new(10);
        assert_eq!(heap.size(), 0);
    }

    #[test]
    #[should_panic(expected = "max_size must be > 0")]
    fn test_new_zero_panics() {
        LongHeap::new(0);
    }

    #[test]
    fn test_new_with_initial_value() {
        let heap = LongHeap::new_with_initial_value(5, 42);
        assert_eq!(heap.size(), 5);
        assert_eq!(heap.top(), 42);
    }

    #[test]
    fn test_push_and_top() {
        let mut heap = LongHeap::new(10);
        heap.push(5);
        assert_eq!(heap.top(), 5);
        heap.push(3);
        assert_eq!(heap.top(), 3);
        heap.push(7);
        assert_eq!(heap.top(), 3);
    }

    #[test]
    fn test_push_returns_new_top() {
        let mut heap = LongHeap::new(10);
        assert_eq!(heap.push(5), 5);
        assert_eq!(heap.push(3), 3);
        assert_eq!(heap.push(7), 3);
        assert_eq!(heap.push(1), 1);
    }

    #[test]
    fn test_pop() {
        let mut heap = LongHeap::new(10);
        heap.push(5);
        heap.push(3);
        heap.push(7);
        assert_eq!(heap.pop(), 3);
        assert_eq!(heap.pop(), 5);
        assert_eq!(heap.pop(), 7);
        assert_eq!(heap.size(), 0);
    }

    #[test]
    #[should_panic(expected = "The heap is empty")]
    fn test_pop_empty_panics() {
        let mut heap = LongHeap::new(10);
        heap.pop();
    }

    #[test]
    fn test_update_top() {
        let mut heap = LongHeap::new(10);
        heap.push(3);
        heap.push(5);
        heap.push(7);
        // Replace top (3) with 6
        let new_top = heap.update_top(6);
        assert_eq!(new_top, 5);
        assert_eq!(heap.top(), 5);
    }

    #[test]
    fn test_update_top_with_smaller() {
        let mut heap = LongHeap::new(10);
        heap.push(3);
        heap.push(5);
        heap.push(7);
        // Replace top (3) with 1
        let new_top = heap.update_top(1);
        assert_eq!(new_top, 1);
    }

    #[test]
    fn test_insert_with_overflow_not_full() {
        let mut heap = LongHeap::new(3);
        assert!(heap.insert_with_overflow(5));
        assert!(heap.insert_with_overflow(3));
        assert!(heap.insert_with_overflow(7));
        assert_eq!(heap.size(), 3);
        assert_eq!(heap.top(), 3);
    }

    #[test]
    fn test_insert_with_overflow_full_rejected() {
        let mut heap = LongHeap::new(3);
        heap.push(5);
        heap.push(3);
        heap.push(7);
        // Value 2 is less than top (3), rejected
        assert!(!heap.insert_with_overflow(2));
        assert_eq!(heap.size(), 3);
        assert_eq!(heap.top(), 3);
    }

    #[test]
    fn test_insert_with_overflow_full_accepted() {
        let mut heap = LongHeap::new(3);
        heap.push(5);
        heap.push(3);
        heap.push(7);
        // Value 6 replaces top (3)
        assert!(heap.insert_with_overflow(6));
        assert_eq!(heap.size(), 3);
        assert_eq!(heap.top(), 5);
    }

    #[test]
    fn test_size() {
        let mut heap = LongHeap::new(10);
        assert_eq!(heap.size(), 0);
        heap.push(1);
        assert_eq!(heap.size(), 1);
        heap.push(2);
        assert_eq!(heap.size(), 2);
        heap.pop();
        assert_eq!(heap.size(), 1);
    }

    #[test]
    fn test_clear() {
        let mut heap = LongHeap::new(10);
        heap.push(1);
        heap.push(2);
        heap.push(3);
        heap.clear();
        assert_eq!(heap.size(), 0);
    }

    #[test]
    fn test_get() {
        let mut heap = LongHeap::new(10);
        heap.push(5);
        heap.push(3);
        heap.push(7);
        // Collect all elements via get
        let mut elements: Vec<i64> = (1..=heap.size()).map(|i| heap.get(i)).collect();
        elements.sort();
        assert_eq!(elements, vec![3, 5, 7]);
    }

    #[test]
    #[should_panic(expected = "index 0 out of range")]
    fn test_get_zero_panics() {
        let heap = LongHeap::new_with_initial_value(3, 1);
        heap.get(0);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn test_get_beyond_size_panics() {
        let heap = LongHeap::new_with_initial_value(3, 1);
        heap.get(4);
    }

    #[test]
    fn test_min_heap_property() {
        let mut heap = LongHeap::new(10);
        let values = [10, 4, 15, 1, 8, 3, 12, 6];
        for v in values {
            heap.push(v);
        }
        // Pop all and verify sorted order
        let mut sorted = Vec::new();
        while heap.size() > 0 {
            sorted.push(heap.pop());
        }
        assert_eq!(sorted, vec![1, 3, 4, 6, 8, 10, 12, 15]);
    }

    #[test]
    fn test_duplicate_values() {
        let mut heap = LongHeap::new(10);
        heap.push(5);
        heap.push(5);
        heap.push(5);
        assert_eq!(heap.size(), 3);
        assert_eq!(heap.pop(), 5);
        assert_eq!(heap.pop(), 5);
        assert_eq!(heap.pop(), 5);
    }

    #[test]
    fn test_negative_values() {
        let mut heap = LongHeap::new(10);
        heap.push(-10);
        heap.push(-5);
        heap.push(-20);
        assert_eq!(heap.top(), -20);
        assert_eq!(heap.pop(), -20);
        assert_eq!(heap.pop(), -10);
        assert_eq!(heap.pop(), -5);
    }

    #[test]
    fn test_initial_value_heap_sorted_output() {
        let heap = LongHeap::new_with_initial_value(3, i64::MIN);
        // All elements are the same sentinel value
        for i in 1..=heap.size() {
            assert_eq!(heap.get(i), i64::MIN);
        }
    }

    #[test]
    fn test_grow_beyond_max_size() {
        let mut heap = LongHeap::new(2);
        // push grows unbounded
        heap.push(1);
        heap.push(2);
        heap.push(3);
        assert_eq!(heap.size(), 3);
        assert_eq!(heap.top(), 1);
    }

    #[test]
    fn test_insert_with_overflow_boundary() {
        let mut heap = LongHeap::new(3);
        heap.push(3);
        heap.push(5);
        heap.push(7);
        // Value equal to top: Java uses `value < heap[1]` so equal is NOT rejected
        assert_gt!(3, heap.top() - 1);
        assert!(heap.insert_with_overflow(3));
    }
}
