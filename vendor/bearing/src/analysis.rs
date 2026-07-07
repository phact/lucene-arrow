// SPDX-License-Identifier: Apache-2.0

//! Text analysis pipeline: tokenizers and analyzers.
//!
//! The [`Analyzer`] trait is the main entry point — a stateful, pull-based
//! token iterator. [`StandardAnalyzer`] provides Unicode-aware tokenization
//! with lowercase normalization, matching Lucene's `StandardAnalyzer`.

use std::fmt::Debug;
use std::io::{self, Read};

use crate::document::TermOffset;

pub(crate) mod chunk_reader;
pub mod standard;
pub mod unicode;

pub use standard::{StandardAnalyzer, StandardAnalyzerFactory};
pub use unicode::{UnicodeAnalyzer, UnicodeAnalyzerFactory};

/// A single token produced by the analyzer during tokenization.
#[derive(Debug)]
pub struct Token<'a> {
    /// The token text, borrowed from the analyzer's buffer.
    pub text: &'a str,
    /// Character offset of this token in the original input.
    pub offset: TermOffset,
    /// Position increment (distance from previous token). Usually 1.
    pub position_increment: i32,
}

/// Breaks text into a stream of tokens for indexing.
///
/// The analyzer owns its input reader. Call [`set_reader`](Analyzer::set_reader)
/// to provide input for a new field, then call [`next_token`](Analyzer::next_token)
/// repeatedly until it returns `None`. Each `set_reader` call replaces the
/// previous reader and resets internal state.
///
/// The returned [`Token`] borrows from the analyzer's internal buffer.
/// The caller must let the token drop before calling `next_token` again
/// (which the natural loop does).
pub trait Analyzer: Send {
    /// Sets the input reader for a new field.
    ///
    /// Replaces any previous reader and resets internal state (buffered
    /// tokens, offsets, etc.). The old reader is dropped.
    fn set_reader(&mut self, reader: Box<dyn Read + Send>);

    /// Returns the next token, or `None` when input is exhausted.
    ///
    /// The returned `Token` borrows its text from the analyzer's internal
    /// buffer. The caller must drop the token before calling `next_token`
    /// again.
    fn next_token(&mut self) -> io::Result<Option<Token<'_>>>;
}

/// Creates [`Analyzer`] instances for indexing workers.
///
/// Each worker thread receives its own `Analyzer` via [`create`](AnalyzerFactory::create).
/// The factory is shared across threads via `Arc` in [`IndexWriterConfig`](crate::index::config::IndexWriterConfig).
///
/// # Example
///
/// ```
/// use bearing::analysis::{Analyzer, AnalyzerFactory, StandardAnalyzerFactory};
///
/// let factory = StandardAnalyzerFactory;
/// let mut analyzer = factory.create();
/// ```
pub trait AnalyzerFactory: Send + Sync + Debug {
    /// Creates a new analyzer instance.
    fn create(&self) -> Box<dyn Analyzer>;
}
