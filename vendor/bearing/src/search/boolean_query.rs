// SPDX-License-Identifier: Apache-2.0

//! Boolean query types: `Occur`, `BooleanClause`, `BooleanQuery`, and `BooleanQuery::Builder`.

use std::fmt;
use std::io;

use super::boolean_weight::BooleanWeight;
use super::collector::ScoreMode;
use super::index_searcher::IndexSearcher;
use super::query::{Query, Weight};

/// Specifies how clauses are to occur in matching documents.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Occur {
    /// Use this operator for clauses that *must* appear in the matching documents.
    Must,

    /// Like `Must` except that these clauses do not participate in scoring.
    Filter,

    /// Use this operator for clauses that *should* appear in the matching documents. For a
    /// `BooleanQuery` with no `Must` clauses one or more `Should` clauses must match a document
    /// for the `BooleanQuery` to match.
    Should,

    /// Use this operator for clauses that *must not* appear in the matching documents. Note
    /// that it is not possible to search for queries that only consist of a `MustNot` clause.
    /// These clauses do not contribute to the score of documents.
    MustNot,
}

impl fmt::Display for Occur {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Occur::Must => write!(f, "+"),
            Occur::Filter => write!(f, "#"),
            Occur::Should => Ok(()),
            Occur::MustNot => write!(f, "-"),
        }
    }
}

/// A clause in a `BooleanQuery`.
pub struct BooleanClause {
    query: Box<dyn Query>,
    occur: Occur,
}

impl BooleanClause {
    /// Constructs a `BooleanClause`.
    pub fn new(query: Box<dyn Query>, occur: Occur) -> Self {
        Self { query, occur }
    }

    /// Returns the sub-query of this clause.
    pub fn query(&self) -> &dyn Query {
        &*self.query
    }

    /// Returns how this clause occurs in matching documents.
    pub fn occur(&self) -> Occur {
        self.occur
    }

    /// Returns `true` if this clause is prohibited (`MustNot`).
    pub fn is_prohibited(&self) -> bool {
        self.occur == Occur::MustNot
    }

    /// Returns `true` if this clause is required (`Must` or `Filter`).
    pub fn is_required(&self) -> bool {
        self.occur == Occur::Must || self.occur == Occur::Filter
    }

    /// Returns `true` if this clause participates in scoring (`Must` or `Should`).
    pub fn is_scoring(&self) -> bool {
        self.occur == Occur::Must || self.occur == Occur::Should
    }
}

impl fmt::Debug for BooleanClause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BooleanClause")
            .field("occur", &self.occur)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// BooleanQuery
// ---------------------------------------------------------------------------

/// A `Query` that matches documents matching boolean combinations of other queries, e.g.
/// `TermQuery`s or other `BooleanQuery`s.
pub struct BooleanQuery {
    minimum_number_should_match: i32,
    clauses: Vec<BooleanClause>,
}

impl BooleanQuery {
    /// Creates a new `Builder` for constructing a `BooleanQuery`.
    pub fn builder() -> Builder {
        Builder::new()
    }

    /// Gets the minimum number of the optional `BooleanClause`s which must be satisfied.
    pub fn get_minimum_number_should_match(&self) -> i32 {
        self.minimum_number_should_match
    }

    /// Returns the clauses of this `BooleanQuery`.
    pub fn clauses(&self) -> &[BooleanClause] {
        &self.clauses
    }
}

impl fmt::Debug for BooleanQuery {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BooleanQuery")
            .field("min_should_match", &self.minimum_number_should_match)
            .field("num_clauses", &self.clauses.len())
            .finish()
    }
}

impl Query for BooleanQuery {
    fn create_weight(
        &self,
        searcher: &IndexSearcher,
        score_mode: ScoreMode,
        boost: f32,
    ) -> io::Result<Box<dyn Weight>> {
        Ok(Box::new(BooleanWeight::new(
            &self.clauses,
            searcher,
            score_mode,
            self.minimum_number_should_match,
            boost,
        )?))
    }
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// A builder for `BooleanQuery`.
pub struct Builder {
    minimum_number_should_match: i32,
    clauses: Vec<BooleanClause>,
}

impl fmt::Debug for Builder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BooleanQuery::Builder")
            .field("min_should_match", &self.minimum_number_should_match)
            .field("num_clauses", &self.clauses.len())
            .finish()
    }
}

impl Builder {
    /// Sole constructor.
    pub fn new() -> Self {
        Self {
            minimum_number_should_match: 0,
            clauses: Vec::new(),
        }
    }

    /// Specifies a minimum number of the optional `BooleanClause`s which must be satisfied.
    pub fn set_minimum_number_should_match(&mut self, min: i32) -> &mut Self {
        self.minimum_number_should_match = min;
        self
    }

    /// Add a new clause to this `Builder`.
    pub fn add(&mut self, clause: BooleanClause) -> &mut Self {
        self.clauses.push(clause);
        self
    }

    /// Add a new clause to this `Builder` from a query and occur.
    pub fn add_query(&mut self, query: Box<dyn Query>, occur: Occur) -> &mut Self {
        self.add(BooleanClause::new(query, occur))
    }

    /// Create a new `BooleanQuery` based on the parameters that have been set on this builder.
    pub fn build(self) -> BooleanQuery {
        BooleanQuery {
            minimum_number_should_match: self.minimum_number_should_match,
            clauses: self.clauses,
        }
    }
}

impl Default for Builder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assertables::*;

    #[test]
    fn test_occur_display() {
        assert_eq!(format!("{}", Occur::Must), "+");
        assert_eq!(format!("{}", Occur::Filter), "#");
        assert_eq!(format!("{}", Occur::Should), "");
        assert_eq!(format!("{}", Occur::MustNot), "-");
    }

    #[test]
    fn test_boolean_clause_is_prohibited() {
        assert!(!make_clause(Occur::Must).is_prohibited());
        assert!(!make_clause(Occur::Filter).is_prohibited());
        assert!(!make_clause(Occur::Should).is_prohibited());
        assert!(make_clause(Occur::MustNot).is_prohibited());
    }

    #[test]
    fn test_boolean_clause_is_required() {
        assert!(make_clause(Occur::Must).is_required());
        assert!(make_clause(Occur::Filter).is_required());
        assert!(!make_clause(Occur::Should).is_required());
        assert!(!make_clause(Occur::MustNot).is_required());
    }

    #[test]
    fn test_boolean_clause_is_scoring() {
        assert!(make_clause(Occur::Must).is_scoring());
        assert!(!make_clause(Occur::Filter).is_scoring());
        assert!(make_clause(Occur::Should).is_scoring());
        assert!(!make_clause(Occur::MustNot).is_scoring());
    }

    /// Dummy query for testing BooleanClause.
    #[derive(Debug)]
    struct DummyQuery;
    impl Query for DummyQuery {
        fn create_weight(
            &self,
            _searcher: &IndexSearcher,
            _score_mode: ScoreMode,
            _boost: f32,
        ) -> io::Result<Box<dyn Weight>> {
            unimplemented!()
        }
    }

    fn make_clause(occur: Occur) -> BooleanClause {
        BooleanClause::new(Box::new(DummyQuery), occur)
    }

    // -----------------------------------------------------------------------
    // BooleanQuery integration tests
    // -----------------------------------------------------------------------

    use std::sync::Arc;

    use crate::document::DocumentBuilder;
    use crate::index::config::IndexWriterConfig;
    use crate::index::directory_reader::DirectoryReader;
    use crate::index::field::text;
    use crate::index::writer::IndexWriter;
    use crate::search::index_searcher::IndexSearcher;
    use crate::search::term_query::TermQuery;
    use crate::store::{MemoryDirectory, SharedDirectory};

    fn build_test_index() -> (SharedDirectory, DirectoryReader) {
        let config = IndexWriterConfig::default().num_threads(1);
        let directory: SharedDirectory = MemoryDirectory::create();
        let writer = IndexWriter::new(config, Arc::clone(&directory));

        writer
            .add_document(
                DocumentBuilder::new()
                    .add_field(text("content").value("hello world"))
                    .build(),
            )
            .unwrap();

        writer
            .add_document(
                DocumentBuilder::new()
                    .add_field(text("content").value("hello there"))
                    .build(),
            )
            .unwrap();

        writer
            .add_document(
                DocumentBuilder::new()
                    .add_field(text("content").value("world peace"))
                    .build(),
            )
            .unwrap();

        writer.commit().unwrap();
        let reader = DirectoryReader::open(&*directory).unwrap();
        (directory, reader)
    }

    #[test]
    fn test_boolean_query_two_must_clauses() {
        let (_dir, reader) = build_test_index();
        let searcher = IndexSearcher::new(&reader);

        // MUST("hello") AND MUST("world") — only doc 0 has both terms
        let mut builder = BooleanQuery::builder();
        builder.add_query(Box::new(TermQuery::new("content", b"hello")), Occur::Must);
        builder.add_query(Box::new(TermQuery::new("content", b"world")), Occur::Must);
        let query = builder.build();

        let top_docs = searcher.search(&query, 10).unwrap();
        assert_eq!(top_docs.total_hits.value, 1);
        assert_eq!(top_docs.score_docs.len(), 1);
    }

    #[test]
    fn test_boolean_query_must_with_nonexistent() {
        let (_dir, reader) = build_test_index();
        let searcher = IndexSearcher::new(&reader);

        // MUST("hello") AND MUST("nonexistent") — no matches
        let mut builder = BooleanQuery::builder();
        builder.add_query(Box::new(TermQuery::new("content", b"hello")), Occur::Must);
        builder.add_query(
            Box::new(TermQuery::new("content", b"nonexistent")),
            Occur::Must,
        );
        let query = builder.build();

        let top_docs = searcher.search(&query, 10).unwrap();
        assert_eq!(top_docs.total_hits.value, 0);
        assert_is_empty!(top_docs.score_docs);
    }

    #[test]
    fn test_boolean_query_score_is_sum_of_term_scores() {
        let (_dir, reader) = build_test_index();
        let searcher = IndexSearcher::new(&reader);

        // Get individual term scores
        let hello_docs = searcher
            .search(&TermQuery::new("content", b"hello"), 10)
            .unwrap();
        let world_docs = searcher
            .search(&TermQuery::new("content", b"world"), 10)
            .unwrap();

        // Find the score for doc 0 from each term query
        let hello_score_doc0 = hello_docs
            .score_docs
            .iter()
            .find(|sd| sd.doc == 0)
            .unwrap()
            .score;
        let world_score_doc0 = world_docs
            .score_docs
            .iter()
            .find(|sd| sd.doc == 0)
            .unwrap()
            .score;
        let expected_sum = hello_score_doc0 + world_score_doc0;

        // Boolean AND query score should equal the sum
        let mut builder = BooleanQuery::builder();
        builder.add_query(Box::new(TermQuery::new("content", b"hello")), Occur::Must);
        builder.add_query(Box::new(TermQuery::new("content", b"world")), Occur::Must);
        let query = builder.build();

        let bool_docs = searcher.search(&query, 10).unwrap();
        assert_eq!(bool_docs.score_docs.len(), 1);
        let bool_score = bool_docs.score_docs[0].score;
        assert!(
            (bool_score - expected_sum).abs() < 1e-5,
            "expected {expected_sum}, got {bool_score}"
        );
    }

    #[test]
    fn test_boolean_query_single_must_clause() {
        let (_dir, reader) = build_test_index();
        let searcher = IndexSearcher::new(&reader);

        // Single MUST clause should work (delegates through single-clause path)
        let mut builder = BooleanQuery::builder();
        builder.add_query(Box::new(TermQuery::new("content", b"hello")), Occur::Must);
        let query = builder.build();

        let top_docs = searcher.search(&query, 10).unwrap();
        assert_eq!(top_docs.total_hits.value, 2);
    }
}
