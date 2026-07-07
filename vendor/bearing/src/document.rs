// SPDX-License-Identifier: Apache-2.0

//! Core document types: index options, doc values types, stored values,
//! and the [`Document`] / [`DocumentBuilder`] API for constructing documents.

use std::fmt;

use mem_dbg::MemSize;

use crate::index::field::Field;

/// Specifies what information is stored in the index for a field.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, MemSize)]
#[mem_size(flat)]
pub enum IndexOptions {
    /// Not indexed.
    None = 0,
    /// Only documents are indexed: term frequencies and positions are omitted.
    Docs = 1,
    /// Documents and term frequencies are indexed. Positions are omitted.
    DocsAndFreqs = 2,
    /// Documents, frequencies, and positions are indexed.
    DocsAndFreqsAndPositions = 3,
    /// Documents, frequencies, positions, and offsets are indexed.
    DocsAndFreqsAndPositionsAndOffsets = 4,
}

impl IndexOptions {
    /// Returns true if this option includes term frequencies.
    pub fn has_freqs(self) -> bool {
        self >= IndexOptions::DocsAndFreqs
    }

    /// Returns true if this option includes positions.
    pub fn has_positions(self) -> bool {
        self >= IndexOptions::DocsAndFreqsAndPositions
    }

    /// Returns true if this option includes offsets.
    pub fn has_offsets(self) -> bool {
        self >= IndexOptions::DocsAndFreqsAndPositionsAndOffsets
    }
}

/// Character-level offset of a term occurrence within a document.
///
/// Represents where a token appears in the original text as a start
/// character position and a length in characters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TermOffset {
    /// Start character position in the document (absolute, zero-based).
    pub start: u32,
    /// Length of the token in characters.
    pub length: u16,
}

/// Specifies the type of doc values stored for a field.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, MemSize)]
#[mem_size(flat)]
pub enum DocValuesType {
    /// No doc values.
    None = 0,
    /// A per-document numeric value.
    Numeric = 1,
    /// A per-document byte[].
    Binary = 2,
    /// A pre-sorted byte[]. Only a single value per document is allowed.
    Sorted = 3,
    /// A pre-sorted numeric value. Multiple values per document are allowed.
    SortedNumeric = 4,
    /// A pre-sorted Set<byte[]>. Multiple values per document are allowed.
    SortedSet = 5,
}

/// A stored field value, used for both writing and reading stored fields.
///
/// Each variant carries a single typed value. The type is fixed at field
/// creation time and determines how the value is serialized.
#[derive(Clone, PartialEq, MemSize)]
pub enum StoredValue {
    /// UTF-8 string value.
    String(String),
    /// Raw byte array.
    Bytes(Vec<u8>),
    /// 32-bit signed integer.
    Int(i32),
    /// 64-bit signed integer.
    Long(i64),
    /// 32-bit floating point.
    Float(f32),
    /// 64-bit floating point.
    Double(f64),
}

impl fmt::Debug for StoredValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StoredValue::String(s) => f.debug_tuple("String").field(s).finish(),
            StoredValue::Bytes(b) => f.debug_tuple("Bytes.len()").field(&b.len()).finish(),
            StoredValue::Int(v) => f.debug_tuple("Int").field(v).finish(),
            StoredValue::Long(v) => f.debug_tuple("Long").field(v).finish(),
            StoredValue::Float(v) => f.debug_tuple("Float").field(v).finish(),
            StoredValue::Double(v) => f.debug_tuple("Double").field(v).finish(),
        }
    }
}

/// An immutable collection of fields to be indexed.
///
/// Created via [`DocumentBuilder`]. Once built, the document is read-only
/// and consumed by the indexing pipeline.
#[derive(Debug)]
pub struct Document {
    fields: Vec<Field>,
}

impl Document {
    /// Returns the fields in this document.
    ///
    /// Mutable access is needed to consume reader-backed field values
    /// during tokenization.
    pub fn fields(&mut self) -> &mut [Field] {
        &mut self.fields
    }
}

/// Builds a [`Document`] by accumulating fields.
///
/// ```
/// use bearing::prelude::{DocumentBuilder, keyword, numeric_dv, stored, text};
///
/// let doc = DocumentBuilder::new()
///     .add_field(text("body").value("the quick brown fox"))
///     .add_field(keyword("category").value("animals"))
///     .add_field(numeric_dv("timestamp").value(1_700_000_000i64))
///     .add_field(stored("raw").string("payload data"))
///     .build();
/// ```
#[derive(Debug, Default)]
pub struct DocumentBuilder {
    fields: Vec<Field>,
}

impl DocumentBuilder {
    /// Creates an empty builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a field to the document.
    pub fn add_field(mut self, field: Field) -> Self {
        self.fields.push(field);
        self
    }

    /// Consumes the builder and produces an immutable [`Document`].
    pub fn build(self) -> Document {
        Document {
            fields: self.fields,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_options_ordering() {
        use assertables::*;
        assert_lt!(IndexOptions::None, IndexOptions::Docs);
        assert_lt!(IndexOptions::Docs, IndexOptions::DocsAndFreqs);
        assert_lt!(
            IndexOptions::DocsAndFreqs,
            IndexOptions::DocsAndFreqsAndPositions
        );
        assert_lt!(
            IndexOptions::DocsAndFreqsAndPositions,
            IndexOptions::DocsAndFreqsAndPositionsAndOffsets
        );
    }

    #[test]
    fn test_index_options_has_freqs() {
        assert!(!IndexOptions::None.has_freqs());
        assert!(!IndexOptions::Docs.has_freqs());
        assert!(IndexOptions::DocsAndFreqs.has_freqs());
        assert!(IndexOptions::DocsAndFreqsAndPositions.has_freqs());
        assert!(IndexOptions::DocsAndFreqsAndPositionsAndOffsets.has_freqs());
    }

    #[test]
    fn test_index_options_has_positions() {
        assert!(!IndexOptions::None.has_positions());
        assert!(!IndexOptions::Docs.has_positions());
        assert!(!IndexOptions::DocsAndFreqs.has_positions());
        assert!(IndexOptions::DocsAndFreqsAndPositions.has_positions());
        assert!(IndexOptions::DocsAndFreqsAndPositionsAndOffsets.has_positions());
    }
}
