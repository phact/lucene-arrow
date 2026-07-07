// SPDX-License-Identifier: Apache-2.0

//! Per-document doc values iterator traits.
//!
//! [`DocValuesIterator`] extends [`DocIdSetIterator`] with point-lookup
//! semantics via [`advance_exact`](DocValuesIterator::advance_exact).
//!
//! Five value-type traits build on it:
//! - [`NumericDocValues`]: single `i64` per document (also used for norms)
//! - [`BinaryDocValues`]: single byte sequence per document
//! - [`SortedDocValues`]: single ordinal per document with term dictionary lookup
//! - [`SortedNumericDocValues`]: multiple sorted `i64` values per document
//! - [`SortedSetDocValues`]: multiple sorted ordinals per document with term dictionary lookup

use std::fmt;
use std::io;

use crate::search::DocIdSetIterator;

/// Base trait for doc values iterators, adding point-lookup to [`DocIdSetIterator`].
///
/// Callers use [`advance_exact`](Self::advance_exact) to position the iterator
/// at a specific document and test whether that document has a value.
pub trait DocValuesIterator: DocIdSetIterator + fmt::Debug {
    /// Advances to exactly `target` and returns whether `target` has a value.
    ///
    /// `target` must be >= the current [`doc_id`](DocIdSetIterator::doc_id) and
    /// a valid doc ID (>= 0 and < `max_doc`). After this returns, the iterator
    /// is positioned at `target`.
    fn advance_exact(&mut self, target: i32) -> io::Result<bool>;
}

/// A per-document numeric value.
///
/// Provides lazy access to norm (or doc value) data. Callers position the
/// iterator with [`advance_exact`](DocValuesIterator::advance_exact), then read
/// the value with [`long_value`](Self::long_value).
pub trait NumericDocValues: DocValuesIterator {
    /// Returns the numeric value for the current document.
    ///
    /// Must only be called after [`advance_exact`](DocValuesIterator::advance_exact)
    /// returned `true`.
    fn long_value(&self) -> io::Result<i64>;
}

/// A per-document binary value.
///
/// Callers position the iterator with [`advance_exact`](DocValuesIterator::advance_exact),
/// then read the value with [`binary_value`](Self::binary_value).
pub trait BinaryDocValues: DocValuesIterator {
    /// Returns the binary value for the current document.
    ///
    /// Must only be called after [`advance_exact`](DocValuesIterator::advance_exact)
    /// returned `true`.
    fn binary_value(&self) -> io::Result<&[u8]>;
}

/// A per-document sorted byte value with ordinal-based access.
///
/// Values are deduplicated, dereferenced, and sorted into a dictionary of
/// unique values. Each document maps to a single ordinal. Ordinals are dense,
/// starting at 0, incrementing by 1 in sorted order.
pub trait SortedDocValues: DocValuesIterator {
    /// Returns the ordinal for the current document.
    ///
    /// Must only be called after [`advance_exact`](DocValuesIterator::advance_exact)
    /// returned `true` or after [`next_doc`](DocIdSetIterator::next_doc) returned a valid doc.
    fn ord_value(&self) -> io::Result<i32>;

    /// Retrieves the value for the specified ordinal.
    ///
    /// `ord` must be >= 0 and < [`value_count`](Self::value_count).
    fn lookup_ord(&self, ord: i32) -> io::Result<&[u8]>;

    /// Returns the number of unique values (one plus the maximum ordinal).
    fn value_count(&self) -> i32;
}

/// A per-document list of sorted numeric values.
///
/// Each document may have multiple values, sorted in ascending order.
pub trait SortedNumericDocValues: DocValuesIterator {
    /// Returns the number of values for the current document.
    ///
    /// Always greater than zero. Must only be called after
    /// [`advance_exact`](DocValuesIterator::advance_exact) returned `true`.
    fn doc_value_count(&self) -> i32;

    /// Returns the next value for the current document.
    ///
    /// Must not be called more than [`doc_value_count`](Self::doc_value_count) times
    /// per document.
    fn next_value(&mut self) -> io::Result<i64>;
}

/// A per-document set of sorted byte values with ordinal-based access.
///
/// Values are deduplicated, dereferenced, and sorted into a dictionary of
/// unique values. Each document maps to one or more ordinals. Ordinals are
/// dense, starting at 0, incrementing by 1 in sorted order.
pub trait SortedSetDocValues: DocValuesIterator {
    /// Returns the number of unique ordinals for the current document.
    ///
    /// Always greater than zero. Must only be called after
    /// [`advance_exact`](DocValuesIterator::advance_exact) returned `true`.
    fn doc_value_count(&self) -> i32;

    /// Returns the next ordinal for the current document.
    ///
    /// Must not be called more than [`doc_value_count`](Self::doc_value_count) times
    /// per document.
    fn next_ord(&mut self) -> io::Result<i64>;

    /// Retrieves the value for the specified ordinal.
    ///
    /// `ord` must be >= 0 and < [`value_count`](Self::value_count).
    fn lookup_ord(&self, ord: i64) -> io::Result<&[u8]>;

    /// Returns the number of unique values (one plus the maximum ordinal).
    fn value_count(&self) -> i64;
}
