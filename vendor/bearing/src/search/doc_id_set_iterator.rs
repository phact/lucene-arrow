// SPDX-License-Identifier: Apache-2.0

//! Defines methods to iterate over a set of non-decreasing doc ids.

use fixedbitset::FixedBitSet;
use std::io;

/// Sentinel value: when returned by `next_doc()`, `advance()`, or `doc_id()` it means there
/// are no more docs in the iterator.
pub const NO_MORE_DOCS: i32 = i32::MAX;

/// Defines methods to iterate over a set of non-decreasing doc ids.
///
/// Note that this trait assumes it iterates on doc ids, and therefore `NO_MORE_DOCS` is set
/// to `i32::MAX` in order to be used as a sentinel object. Implementations are expected to
/// consider `i32::MAX` as an invalid value.
pub trait DocIdSetIterator {
    /// Returns the following:
    /// - `-1` if `next_doc()` or `advance()` were not called yet.
    /// - `NO_MORE_DOCS` if the iterator has exhausted.
    /// - Otherwise it should return the doc ID it is currently on.
    fn doc_id(&self) -> i32;

    /// Advances to the next document in the set and returns the doc it is currently on, or
    /// `NO_MORE_DOCS` if there are no more docs in the set.
    ///
    /// **NOTE:** after the iterator has exhausted you should not call this method, as it may
    /// result in unpredicted behavior.
    fn next_doc(&mut self) -> io::Result<i32>;

    /// Advances to the first beyond the current whose document number is greater than or equal
    /// to `target`, and returns the document number itself. Exhausts the iterator and returns
    /// `NO_MORE_DOCS` if `target` is greater than the highest document number in the set.
    ///
    /// The behavior of this method is **undefined** when called with `target <= current`, or
    /// after the iterator has exhausted. Both cases may result in unpredicted behavior.
    ///
    /// **NOTE:** this method may be called with `NO_MORE_DOCS` for efficiency by some Scorers.
    /// If your implementation cannot efficiently determine that it should exhaust, it is
    /// recommended that you check for that value in each call to this method.
    fn advance(&mut self, target: i32) -> io::Result<i32>;

    /// Returns the estimated cost of this `DocIdSetIterator`.
    ///
    /// This is generally an upper bound of the number of documents this iterator might match,
    /// but may be a rough heuristic, hardcoded value, or otherwise completely inaccurate.
    fn cost(&self) -> i64;

    /// Returns the end of the run of consecutive doc IDs that match this iterator and that
    /// contains the current `doc_id()`, that is: one plus the last doc ID of the run.
    ///
    /// The default implementation assumes runs of a single doc ID and returns `doc_id() + 1`.
    ///
    /// **Note:** It is illegal to call this method when the iterator is exhausted or not
    /// positioned.
    ///
    /// Takes `&mut self` (deviation from Java's implicitly read-only signature) so that
    /// composite iterators can delegate to sub-iterators without requiring a separate
    /// shared accessor on the `Scorer` trait.
    fn doc_id_run_end(&mut self) -> io::Result<i32> {
        Ok(self.doc_id() + 1)
    }

    /// Load doc IDs into a bitset. All doc IDs below `up_to` (exclusive) are set in the given
    /// bit set at position `doc_id - offset`.
    ///
    /// The default implementation iterates doc IDs one at a time via `next_doc()`.
    /// Implementations backed by encoded blocks (e.g. postings) can override for bulk loading.
    fn fill_bit_set(
        &mut self,
        up_to: i32,
        bit_set: &mut FixedBitSet,
        offset: i32,
    ) -> io::Result<()> {
        debug_assert!(offset <= self.doc_id());
        let mut doc = self.doc_id();
        while doc < up_to {
            bit_set.insert((doc - offset) as usize);
            doc = self.next_doc()?;
        }
        Ok(())
    }
}

/// Slow (linear) implementation of `advance` relying on `next_doc()` to advance beyond the
/// target position. Provided as a free function since Rust traits can't have `final` methods
/// that call other trait methods with `&mut self`.
pub fn slow_advance(iter: &mut dyn DocIdSetIterator, target: i32) -> io::Result<i32> {
    debug_assert!(iter.doc_id() < target);
    let mut doc;
    loop {
        doc = iter.next_doc()?;
        if doc >= target {
            break;
        }
    }
    Ok(doc)
}

/// An empty `DocIdSetIterator` instance.
pub fn empty() -> RangeDocIdSetIterator {
    RangeDocIdSetIterator::new(0, 0)
}

/// A `DocIdSetIterator` that matches all documents up to `max_doc - 1`.
///
/// # Panics
///
/// Panics if `max_doc` is negative.
pub fn all(max_doc: i32) -> RangeDocIdSetIterator {
    assert!(
        max_doc >= 0,
        "maxDoc must be >= 0, but got maxDoc={}",
        max_doc
    );
    RangeDocIdSetIterator::new(0, max_doc)
}

/// A `DocIdSetIterator` that matches a range of documents from `min_doc` (inclusive) to
/// `max_doc` (exclusive).
///
/// # Panics
///
/// Panics if `min_doc >= max_doc` or `min_doc < 0`.
pub fn range(min_doc: i32, max_doc: i32) -> RangeDocIdSetIterator {
    assert!(
        min_doc < max_doc,
        "minDoc must be < maxDoc but got minDoc={} maxDoc={}",
        min_doc,
        max_doc
    );
    assert!(
        min_doc >= 0,
        "minDoc must be >= 0 but got minDoc={}",
        min_doc
    );
    RangeDocIdSetIterator::new(min_doc, max_doc)
}

/// A `DocIdSetIterator` over a contiguous range of doc IDs.
#[derive(Debug)]
pub struct RangeDocIdSetIterator {
    min_doc: i32,
    max_doc: i32,
    doc: i32,
}

impl RangeDocIdSetIterator {
    fn new(min_doc: i32, max_doc: i32) -> Self {
        debug_assert!(min_doc <= max_doc);
        Self {
            min_doc,
            max_doc,
            doc: -1,
        }
    }
}

impl DocIdSetIterator for RangeDocIdSetIterator {
    fn doc_id(&self) -> i32 {
        self.doc
    }

    fn next_doc(&mut self) -> io::Result<i32> {
        self.advance(self.doc + 1)
    }

    fn advance(&mut self, target: i32) -> io::Result<i32> {
        if target >= self.max_doc {
            self.doc = NO_MORE_DOCS;
        } else if target < self.min_doc {
            self.doc = self.min_doc;
        } else {
            self.doc = target;
        }
        Ok(self.doc)
    }

    fn cost(&self) -> i64 {
        (self.max_doc - self.min_doc) as i64
    }

    fn doc_id_run_end(&mut self) -> io::Result<i32> {
        Ok(self.max_doc)
    }

    /// Bulk set for contiguous range: sets bits `[doc - offset, min(up_to, max_doc) - offset)`.
    fn fill_bit_set(
        &mut self,
        up_to: i32,
        bit_set: &mut FixedBitSet,
        offset: i32,
    ) -> io::Result<()> {
        debug_assert!(offset <= self.doc);
        let up_to = up_to.min(self.max_doc);
        if up_to > self.doc {
            let from = (self.doc - offset) as usize;
            let to = (up_to - offset) as usize;
            bit_set.insert_range(from..to);
            self.advance(up_to)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty() {
        let mut disi = empty();
        assert_eq!(disi.doc_id(), -1);
        assert_eq!(disi.next_doc().unwrap(), NO_MORE_DOCS);
        assert_eq!(disi.doc_id(), NO_MORE_DOCS);

        let mut disi = empty();
        assert_eq!(disi.doc_id(), -1);
        assert_eq!(disi.advance(42).unwrap(), NO_MORE_DOCS);
        assert_eq!(disi.doc_id(), NO_MORE_DOCS);
    }

    #[test]
    fn test_range_basic() {
        let mut disi = range(5, 8);
        assert_eq!(disi.doc_id(), -1);
        assert_eq!(disi.next_doc().unwrap(), 5);
        assert_eq!(disi.next_doc().unwrap(), 6);
        assert_eq!(disi.next_doc().unwrap(), 7);
        assert_eq!(disi.next_doc().unwrap(), NO_MORE_DOCS);
    }

    #[test]
    #[should_panic(expected = "minDoc must be < maxDoc")]
    fn test_invalid_range() {
        range(5, 4);
    }

    #[test]
    #[should_panic(expected = "minDoc must be >= 0")]
    fn test_invalid_range_min() {
        range(-1, 4);
    }

    #[test]
    #[should_panic(expected = "minDoc must be < maxDoc")]
    fn test_empty_range() {
        range(7, 7);
    }

    #[test]
    fn test_range_advance() {
        let mut disi = range(5, 20);
        assert_eq!(disi.doc_id(), -1);
        assert_eq!(disi.next_doc().unwrap(), 5);
        assert_eq!(disi.advance(17).unwrap(), 17);
        assert_eq!(disi.next_doc().unwrap(), 18);
        assert_eq!(disi.next_doc().unwrap(), 19);
        assert_eq!(disi.next_doc().unwrap(), NO_MORE_DOCS);
    }

    #[test]
    fn test_all_basic() {
        let mut disi = all(3);
        assert_eq!(disi.doc_id(), -1);
        assert_eq!(disi.next_doc().unwrap(), 0);
        assert_eq!(disi.next_doc().unwrap(), 1);
        assert_eq!(disi.next_doc().unwrap(), 2);
        assert_eq!(disi.next_doc().unwrap(), NO_MORE_DOCS);
    }

    #[test]
    fn test_all_cost() {
        let disi = all(100);
        assert_eq!(disi.cost(), 100);
    }

    #[test]
    fn test_range_cost() {
        let disi = range(5, 20);
        assert_eq!(disi.cost(), 15);
    }

    #[test]
    fn test_doc_id_run_end_all() {
        let mut it = all(13);
        assert_eq!(it.next_doc().unwrap(), 0);
        assert_eq!(it.doc_id_run_end().unwrap(), 13);
        assert_eq!(it.advance(10).unwrap(), 10);
        assert_eq!(it.doc_id_run_end().unwrap(), 13);
        assert_eq!(it.advance(13).unwrap(), NO_MORE_DOCS);
    }

    #[test]
    fn test_doc_id_run_end_range() {
        let mut it = range(4, 13);
        assert_eq!(it.next_doc().unwrap(), 4);
        assert_eq!(it.doc_id_run_end().unwrap(), 13);
        assert_eq!(it.advance(10).unwrap(), 10);
        assert_eq!(it.doc_id_run_end().unwrap(), 13);
        assert_eq!(it.advance(13).unwrap(), NO_MORE_DOCS);
    }

    #[test]
    fn test_slow_advance() {
        let mut disi = range(0, 100);
        disi.next_doc().unwrap(); // position at 0
        let result = slow_advance(&mut disi, 50).unwrap();
        assert_eq!(result, 50);
        assert_eq!(disi.doc_id(), 50);
    }

    #[test]
    fn test_advance_before_range() {
        let mut disi = range(10, 20);
        assert_eq!(disi.advance(5).unwrap(), 10);
        assert_eq!(disi.doc_id(), 10);
    }

    #[test]
    fn test_advance_past_range() {
        let mut disi = range(10, 20);
        assert_eq!(disi.advance(25).unwrap(), NO_MORE_DOCS);
        assert_eq!(disi.doc_id(), NO_MORE_DOCS);
    }

    #[test]
    #[should_panic(expected = "maxDoc must be >= 0")]
    fn test_all_negative_max_doc() {
        all(-1);
    }

    #[test]
    fn test_empty_cost() {
        let disi = empty();
        assert_eq!(disi.cost(), 0);
    }
}
