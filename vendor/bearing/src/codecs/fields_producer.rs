// SPDX-License-Identifier: Apache-2.0

//! Codec producer traits for the flush and merge write paths.
//!
//! These traits define the interface between the indexing chain and the codec
//! writers. At flush time, `FreqProxFields` implements these traits to read
//! from in-memory byte pool streams. At merge time, different implementations
//! will wrap multiple segment readers.

use std::io;

use crate::document::{IndexOptions, TermOffset};

/// Sentinel value returned by [`PostingsEnumProducer::next_doc`] when iteration
/// is exhausted.
pub const NO_MORE_DOCS: i32 = i32::MAX;

/// Access to all fields in a segment for codec writing.
///
/// Consumed by `BlockTreeTermsWriter::write()` during flush and merge.
pub trait FieldsProducer {
    /// Returns field names in the order they should be written.
    fn field_names(&self) -> Vec<String>;

    /// Returns the terms for the given field, or `None` if the field has no terms.
    fn terms(&self, field: &str) -> Option<Box<dyn TermsProducer + '_>>;
}

/// Access to the terms of a single field for codec writing.
///
/// Consumed by `BlockTreeTermsWriter` to get a term iterator and field metadata.
pub trait TermsProducer {
    /// Returns an iterator over all terms in sorted order.
    fn iterator(&self) -> io::Result<Box<dyn TermsEnumProducer + '_>>;

    /// Whether this field indexes term frequencies.
    fn has_freqs(&self) -> bool;

    /// Whether this field indexes positions.
    fn has_positions(&self) -> bool;

    /// Whether this field indexes offsets.
    fn has_offsets(&self) -> bool;

    /// Whether any token in this field had a payload.
    fn has_payloads(&self) -> bool;
}

/// Iterator over terms in sorted order for codec writing.
///
/// The codec calls `next()` to advance to each term, then `postings()` to get
/// the posting data for that term.
pub trait TermsEnumProducer {
    /// Advances to the next term. Returns the term bytes, or `None` at end.
    fn next(&mut self) -> io::Result<Option<&[u8]>>;

    /// Returns a postings iterator for the current term.
    ///
    /// Must be called after a successful `next()`.
    fn postings(&mut self) -> io::Result<Box<dyn PostingsEnumProducer + '_>>;

    // TODO(lucene-alignment): seekCeil() — not called by codec during flush.
    // Needed for merge path. Add when segment merging is implemented.
}

/// Random-access terms for a single field.
///
/// Provides indexed access to term bytes and streaming postings. The key design
/// property: `term_bytes(&self, i)` returns shared borrows from the backing
/// store, so multiple calls produce simultaneously valid `&[u8]` slices. This
/// enables `PendingTerm<'a>` to accumulate borrowed term bytes across all terms
/// in a field.
///
/// `postings(&self, i)` returns an owned `Box` that borrows from `self` and is
/// consumed within one loop iteration, so it never conflicts with accumulated
/// term byte borrows.
pub trait FieldTerms {
    /// Number of terms in this field.
    fn term_count(&self) -> usize;

    /// Returns the term bytes at the given sorted index.
    fn term_bytes(&self, index: usize) -> &[u8];

    /// Returns a postings producer for the term at the given sorted index.
    fn postings(&self, index: usize) -> io::Result<Box<dyn PostingsEnumProducer + '_>>;

    /// Returns what information is indexed for this field.
    fn index_options(&self) -> IndexOptions;

    /// Whether any token in this field had a payload.
    fn has_payloads(&self) -> bool;

    /// The field number in the segment's `FieldInfos`.
    fn field_number(&self) -> u32;

    /// The field name.
    fn field_name(&self) -> &str;
}

/// Iterator over postings (doc/freq/position/offset/payload) for a single term.
///
/// Consumed by `PostingsWriter::write_term()` during flush and merge.
///
/// Call pattern: `next_doc()` in a loop, for each doc optionally call `freq()`,
/// then `next_position()` x freq times, each optionally followed by
/// `offset()`, `payload()`.
pub trait PostingsEnumProducer {
    /// Number of documents containing this term.
    ///
    /// Available before iteration begins. Used by the postings writer to choose
    /// encoding paths (singleton vs vint vs block).
    fn doc_freq(&self) -> i32;

    /// Total number of term occurrences across all documents, or -1 if
    /// frequencies are not indexed.
    fn total_term_freq(&self) -> i64;

    /// Advances to the next document. Returns the doc ID, or
    /// [`NO_MORE_DOCS`] when exhausted.
    fn next_doc(&mut self) -> io::Result<i32>;

    /// Returns the frequency of the current term in the current document.
    fn freq(&self) -> i32;

    /// Advances to the next position within the current document.
    /// Returns the position.
    fn next_position(&mut self) -> io::Result<i32>;

    /// Returns the character offset of the current position, or `None`
    /// if offsets were not indexed.
    fn offset(&self) -> Option<TermOffset>;

    /// Returns the payload at the current position, if any.
    fn payload(&self) -> Option<&[u8]>;
}
