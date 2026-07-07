// SPDX-License-Identifier: Apache-2.0

//! Convenience re-exports for common types.
//!
//! # Write path
//!
//! - **Analysis**: [`AnalyzerFactory`], [`StandardAnalyzerFactory`],
//!   [`UnicodeAnalyzerFactory`]
//! - **Documents**: [`Document`], [`DocumentBuilder`]
//! - **Index writer**: [`IndexWriter`], [`IndexWriterConfig`]
//! - **Field builders**: [`text`], [`keyword`], [`string`], [`stored`],
//!   [`int_field`], [`long_field`], [`float_field`], [`double_field`],
//!   [`lat_lon`], [`feature`], [`int_range`], [`long_range`], [`float_range`],
//!   [`double_range`], [`numeric_dv`], [`binary_dv`], [`sorted_dv`],
//!   [`sorted_set_dv`], [`sorted_numeric_dv`], [`TermVectorOptions`]
//! - **Storage**: [`Directory`], [`FSDirectory`], [`MemoryDirectory`],
//!   [`SharedDirectory`]
//!
//! # Read path
//!
//! - **Index reader**: [`DirectoryReader`], [`LeafReaderContext`]
//! - **Terms**: [`Terms`], [`TermsEnum`], [`SeekStatus`]
//! - **Searcher**: [`IndexSearcher`]
//! - **Queries**: [`Query`], [`TermQuery`], [`BooleanQuery`], [`BooleanClause`],
//!   [`Occur`]
//! - **Results**: [`TopDocs`], [`ScoreDoc`], [`TotalHits`]

pub use crate::analysis::{AnalyzerFactory, StandardAnalyzerFactory, UnicodeAnalyzerFactory};
pub use crate::document::{Document, DocumentBuilder};
pub use crate::index::config::IndexWriterConfig;
pub use crate::index::directory_reader::{DirectoryReader, LeafReaderContext};
pub use crate::index::field::{
    TermVectorOptions, TokenizedIndexOptions, binary_dv, double_field, double_range, feature,
    float_field, float_range, int_field, int_range, keyword, lat_lon, long_field, long_range,
    numeric_dv, sorted_dv, sorted_numeric_dv, sorted_set_dv, stored, string, text,
};
pub use crate::index::terms::{SeekStatus, Terms, TermsEnum};
pub use crate::index::writer::IndexWriter;
pub use crate::search::{
    BooleanClause, BooleanQuery, IndexSearcher, Occur, Query, ScoreDoc, TermQuery, TopDocs,
    TotalHits,
};
pub use crate::store::{Directory, FSDirectory, MemoryDirectory, SharedDirectory};
