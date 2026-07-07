// SPDX-License-Identifier: Apache-2.0

//! A Rust port of Apache Lucene.
//!
//! Bearing implements both the indexing (write) and search (read) paths of
//! Apache Lucene 10.3.2 under the Lucene103 codec. Indexes are byte-compatible
//! with Java Lucene — readable by either side.
//!
//! # Indexing
//!
//! ```no_run
//! use bearing::prelude::{
//!     DocumentBuilder, FSDirectory, IndexWriter, IndexWriterConfig,
//!     keyword, text,
//! };
//!
//! let directory = FSDirectory::open(std::path::Path::new("/tmp/my-index")).unwrap();
//! let writer = IndexWriter::new(IndexWriterConfig::default(), directory);
//!
//! let doc = DocumentBuilder::new()
//!     .add_field(text("body").value("the quick brown fox"))
//!     .add_field(keyword("category").value("animals"))
//!     .build();
//! writer.add_document(doc).unwrap();
//!
//! writer.commit().unwrap();
//! ```
//!
//! # Searching
//!
//! ```no_run
//! use bearing::prelude::{DirectoryReader, FSDirectory, IndexSearcher, TermQuery};
//!
//! let directory = FSDirectory::open(std::path::Path::new("/tmp/my-index")).unwrap();
//! let reader = DirectoryReader::open(&*directory).unwrap();
//! let searcher = IndexSearcher::new(&reader);
//!
//! let query = TermQuery::new("body", b"fox");
//! let top_docs = searcher.search(&query, 10).unwrap();
//!
//! for hit in &top_docs.score_docs {
//!     println!("doc={} score={}", hit.doc, hit.score);
//! }
//! ```
//!
//! # Modules
//!
//! - [`prelude`] — Convenience re-exports for common types.
//! - [`analysis`] — Text analysis pipeline (tokenizers, filters, analyzers).
//! - [`codecs`] — Codec implementations (Lucene103 and supporting formats).
//! - [`document`] — Document model and field factory functions.
//! - [`encoding`] — Data encoding/decoding algorithms (varint, zigzag, packed ints, compression).
//! - [`index`] — Index writer, reader, configuration, and segment metadata.
//! - [`search`] — Queries, scorers, collectors, and the index searcher.
//! - [`store`] — Storage abstraction (directories, file backing, input/output).
//! - [`util`] — Utility types (byte block pool, bytes ref hash).

#[cfg(test)]
#[macro_use]
extern crate assertables;

pub mod analysis;
pub mod codecs;
pub mod document;
pub mod encoding;
pub use index::field;
pub mod index;
pub mod prelude;
pub mod search;
pub mod store;
#[cfg(test)]
pub mod test_util;
pub mod util;
