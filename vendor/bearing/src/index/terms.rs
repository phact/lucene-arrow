// SPDX-License-Identifier: Apache-2.0

//! Term access abstractions for reading an index.
//!
//! [`Terms`] provides access to the terms in a specific field, including
//! statistics and an [`iterator()`](Terms::iterator) to step through terms.
//!
//! [`TermsEnum`] is the iterator returned by [`Terms::iterator`]. It supports
//! seeking to specific terms, reading term statistics, and obtaining the
//! internal [`IntBlockTermState`] for postings access.

use std::io;

use crate::codecs::lucene103::postings_format::IntBlockTermState;

/// Access to the terms in a specific field.
///
/// Provides aggregate statistics and an [`iterator()`](Self::iterator) to
/// enumerate terms. Implementations are per-field: each indexed field has its
/// own `Terms` instance.
pub trait Terms {
    /// Returns an iterator that will step through all terms.
    fn iterator(&self) -> io::Result<Box<dyn TermsEnum + '_>>;

    /// Returns the number of terms for this field, or -1 if not available.
    fn size(&self) -> i64;

    /// Returns the sum of [`TermsEnum::total_term_freq`] for all terms in this field.
    fn get_sum_total_term_freq(&self) -> i64;

    /// Returns the sum of [`TermsEnum::doc_freq`] for all terms in this field.
    fn get_sum_doc_freq(&self) -> i64;

    /// Returns the number of documents that have at least one term for this field.
    fn get_doc_count(&self) -> i32;

    /// Returns true if documents in this field store per-document term frequency.
    fn has_freqs(&self) -> bool;

    /// Returns true if documents in this field store offsets.
    fn has_offsets(&self) -> bool;

    /// Returns true if documents in this field store positions.
    fn has_positions(&self) -> bool;

    /// Returns true if documents in this field store payloads.
    fn has_payloads(&self) -> bool;

    /// Returns the smallest term (in lexicographic order) in the field,
    /// or `None` when there are no terms.
    fn get_min(&self) -> Option<&[u8]>;

    /// Returns the largest term (in lexicographic order) in the field,
    /// or `None` when there are no terms.
    fn get_max(&self) -> Option<&[u8]>;
}

/// Returned result from [`TermsEnum::seek_ceil`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeekStatus {
    /// The term was not found, and the end of iteration was hit.
    End,
    /// The precise term was found.
    Found,
    /// A different term was found after the requested term.
    NotFound,
}

/// Iterator to seek or step through terms, obtaining frequency information
/// and postings for the current term.
///
/// Term enumerations are always ordered by byte comparison (Unicode sort order
/// for UTF-8). Each term in the enumeration is greater than the one before it.
///
/// The enum is unpositioned when first obtained — you must call
/// [`seek_exact`](Self::seek_exact) or [`next`](Self::next) first.
pub trait TermsEnum {
    // Java's TermsEnum also defines:
    //   postings(PostingsEnum reuse, int flags) -> PostingsEnum
    //   impacts(int flags) -> ImpactsEnum
    // These belong on this trait but are not yet implemented. The Rust
    // signature needs a PostingsReader reference that Java obtains via
    // a parent back-reference chain, which creates ownership issues.

    /// Attempts to seek to the exact term, returning `true` if found.
    ///
    /// If this returns `false`, the enum is unpositioned.
    fn seek_exact(&mut self, target: &[u8]) -> io::Result<bool>;

    /// Seeks a specific position by [`IntBlockTermState`] previously obtained
    /// from [`term_state()`](Self::term_state).
    ///
    /// Low-level implementations may position the enum without re-seeking the
    /// term dictionary.
    fn seek_exact_with_state(&mut self, term: &[u8], state: IntBlockTermState);

    /// Returns the current term. Do not call this when the enum is unpositioned.
    fn term(&self) -> &[u8];

    /// Returns the number of documents containing the current term.
    ///
    /// May trigger lazy metadata decoding.
    fn doc_freq(&mut self) -> io::Result<i32>;

    /// Returns the total number of occurrences of this term across all documents.
    ///
    /// May trigger lazy metadata decoding.
    fn total_term_freq(&mut self) -> io::Result<i64>;

    /// Returns the internal state to position the enum without re-seeking.
    ///
    /// May trigger lazy metadata decoding.
    fn term_state(&mut self) -> io::Result<IntBlockTermState>;

    /// Seeks to the specified term, if it exists, or to the next (ceiling) term.
    fn seek_ceil(&mut self, _target: &[u8]) -> io::Result<SeekStatus> {
        todo!("seek_ceil not yet implemented")
    }

    /// Advances to the next term in the enumeration. Returns `None` at end.
    fn next(&mut self) -> io::Result<Option<&[u8]>> {
        todo!("next not yet implemented")
    }
}
