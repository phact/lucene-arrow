// SPDX-License-Identifier: Apache-2.0

use std::io;

use mem_dbg::MemSize;

use crate::analysis::Token;
use crate::index::field::Field;
use crate::index::pipeline::segment_accumulator::SegmentAccumulator;
use crate::index::pipeline::segment_context::SegmentContext;

/// Indicates whether a consumer wants to receive tokens for a field.
///
/// Returned by [`FieldConsumer::start_field`] so the worker knows
/// which consumers to include in the token loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenInterest {
    /// This consumer wants to receive `add_token` calls for this field.
    WantsTokens,
    /// This consumer does not process tokens for this field.
    NoTokens,
}

/// Lifecycle trait for processing document data during indexing.
///
/// Each consumer handles one aspect of field data (postings, stored
/// fields, norms, term vectors, doc values, points). The SegmentWorker
/// calls these methods in order for every document — the consumer
/// decides internally whether to act on a given field or ignore it.
///
/// This trait is the core of the indexing pipeline. All data flows
/// through these lifecycle methods.
///
/// # Segment accumulator
///
/// Methods that accumulate data receive `&mut SegmentAccumulator` —
/// the shared state owned by the worker. This includes memory pools
/// for data accumulation and cross-consumer metadata (e.g., field
/// properties discovered during processing). Only one consumer
/// borrows the accumulator at a time; the worker passes it
/// sequentially.
///
/// # Call sequence per document
///
/// ```text
/// start_document(doc_id)
///   for each field:
///     interest = start_field(field_id, field, &mut accumulator)
///     if tokenized and interest == WantsTokens:
///       for each token from analyzer:
///         add_token(field_id, field, token, &mut accumulator)
///     finish_field(field_id, field, &mut accumulator)
/// finish_document(doc_id, &mut accumulator, &context)
/// ```
///
/// # Flush
///
/// After one or more documents have been processed, `flush` is called
/// to write accumulated data to segment files. Consumers are flushed
/// in the order they appear in the worker's consumer list. This order
/// matters — some consumers may read files written by earlier consumers
/// during their own flush. The consumer is then dropped along with
/// the worker.
pub trait FieldConsumer: MemSize {
    /// A new document is beginning.
    fn start_document(&mut self, doc_id: i32) -> io::Result<()>;

    /// A field is beginning within the current document.
    ///
    /// Called once per field, before any tokens. The consumer should
    /// initialize or locate its per-field state and prepare to receive
    /// data. For example, a stored fields consumer would buffer the
    /// field's value here. A postings consumer would look up (or create)
    /// its per-field term hash and set it as the active target for
    /// incoming tokens.
    ///
    /// Returns [`TokenInterest`] to indicate whether this consumer
    /// wants to receive `add_token` calls for this field.
    fn start_field(
        &mut self,
        field_id: u32,
        field: &Field,
        accumulator: &mut SegmentAccumulator,
    ) -> io::Result<TokenInterest>;

    /// A single token from a tokenized field. Only called on consumers
    /// that returned [`TokenInterest::WantsTokens`] from `start_field`.
    ///
    /// The field reference is provided so the consumer has full context
    /// without needing to track "current field" state.
    fn add_token(
        &mut self,
        field_id: u32,
        field: &Field,
        token: &Token<'_>,
        accumulator: &mut SegmentAccumulator,
    ) -> io::Result<()>;

    /// A field is complete within the current document.
    ///
    /// Called once per field, after all tokens have been delivered.
    /// The consumer should finalize any per-field per-document state
    /// that depends on the complete field content. For example, a
    /// postings consumer would record the final term frequency for
    /// the last term seen. A norms consumer would compute the field's
    /// norm value from accumulated statistics (token count, unique
    /// terms, field length, etc.).
    ///
    /// Consumers that discover field properties during processing
    /// (e.g., detecting payloads in a token stream) should record
    /// those properties in the accumulator here so they are available
    /// to other consumers at flush time.
    fn finish_field(
        &mut self,
        field_id: u32,
        field: &Field,
        accumulator: &mut SegmentAccumulator,
    ) -> io::Result<()>;

    /// The current document is complete. FieldConsumers may finalize
    /// per-document state (e.g., flush term vectors, store norms).
    ///
    /// `context` provides directory access for consumers that stream
    /// data to codec writers incrementally (e.g., stored fields, term
    /// vectors).
    fn finish_document(
        &mut self,
        doc_id: i32,
        accumulator: &mut SegmentAccumulator,
        context: &SegmentContext,
    ) -> io::Result<()>;

    /// Write all accumulated data to segment files.
    ///
    /// `context` provides the directory, segment name, and segment ID
    /// needed to create correctly named and headered codec files.
    /// The accumulator is borrowed immutably — consumers read
    /// accumulated data but do not modify it.
    /// Returns the names of the files written.
    fn flush(
        &mut self,
        context: &SegmentContext,
        accumulator: &SegmentAccumulator,
    ) -> io::Result<Vec<String>>;
}
