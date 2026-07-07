// SPDX-License-Identifier: Apache-2.0

//! Search result types: `ScoreDoc`, `TotalHits`, and `TopDocs`.

use std::cmp::Ordering;
use std::fmt;

// ---------------------------------------------------------------------------
// ScoreDoc
// ---------------------------------------------------------------------------

/// Holds one hit in `TopDocs`.
#[derive(Debug)]
pub struct ScoreDoc {
    /// A hit document's number.
    pub doc: i32,
    /// The score of this document for the query.
    pub score: f32,
    /// Only set by `TopDocs::merge`.
    pub shard_index: i32,
}

impl ScoreDoc {
    /// Constructs a `ScoreDoc`.
    pub fn new(doc: i32, score: f32) -> Self {
        Self::new_with_shard_index(doc, score, -1)
    }

    /// Constructs a `ScoreDoc` with a shard index.
    pub fn new_with_shard_index(doc: i32, score: f32, shard_index: i32) -> Self {
        Self {
            doc,
            score,
            shard_index,
        }
    }

    /// Comparator that sorts by score descending, then by doc id ascending.
    pub fn compare(a: &ScoreDoc, b: &ScoreDoc) -> Ordering {
        if a.score > b.score {
            Ordering::Less
        } else if a.score < b.score {
            Ordering::Greater
        } else {
            a.doc.cmp(&b.doc)
        }
    }
}

impl fmt::Display for ScoreDoc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "doc={} score={} shardIndex={}",
            self.doc, self.score, self.shard_index
        )
    }
}

// ---------------------------------------------------------------------------
// TotalHits
// ---------------------------------------------------------------------------

/// How the `TotalHits::value` should be interpreted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Relation {
    /// The total hit count is equal to `TotalHits::value`.
    EqualTo,
    /// The total hit count is greater than or equal to `TotalHits::value`.
    GreaterThanOrEqualTo,
}

/// Description of the total number of hits of a query. The total hit count can't generally be
/// computed accurately without visiting all matches, which is costly for queries that match lots
/// of documents. Given that it is often enough to have a lower bounds of the number of hits,
/// such as "there are more than 1000 hits", Lucene has options to stop counting as soon as a
/// threshold has been reached in order to improve query times.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TotalHits {
    /// The value of the total hit count. Must be interpreted in the context of `relation`.
    pub value: i64,
    /// Whether `value` is the exact hit count or a lower bound.
    pub relation: Relation,
}

impl TotalHits {
    /// Creates a new `TotalHits`.
    ///
    /// # Panics
    ///
    /// Panics if `value` is negative.
    pub fn new(value: i64, relation: Relation) -> Self {
        assert!(value >= 0, "value must be >= 0, got {}", value);
        Self { value, relation }
    }
}

impl fmt::Display for TotalHits {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}{} hits",
            self.value,
            if self.relation == Relation::EqualTo {
                ""
            } else {
                "+"
            }
        )
    }
}

// ---------------------------------------------------------------------------
// TopDocs
// ---------------------------------------------------------------------------

/// Represents hits returned by `IndexSearcher::search`.
#[derive(Debug)]
pub struct TopDocs {
    /// The total number of hits for the query.
    pub total_hits: TotalHits,
    /// The top hits for the query.
    pub score_docs: Vec<ScoreDoc>,
}

impl TopDocs {
    /// Constructs a `TopDocs`.
    pub fn new(total_hits: TotalHits, score_docs: Vec<ScoreDoc>) -> Self {
        Self {
            total_hits,
            score_docs,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use assertables::*;

    // -- TotalHits tests --

    #[test]
    fn test_total_hits_new_valid() {
        let hits = TotalHits::new(42, Relation::EqualTo);
        assert_eq!(hits.value, 42);
        assert_eq!(hits.relation, Relation::EqualTo);
    }

    #[test]
    fn test_total_hits_new_zero() {
        let hits = TotalHits::new(0, Relation::GreaterThanOrEqualTo);
        assert_eq!(hits.value, 0);
        assert_eq!(hits.relation, Relation::GreaterThanOrEqualTo);
    }

    #[test]
    #[should_panic(expected = "value must be >= 0")]
    fn test_total_hits_new_negative_panics() {
        TotalHits::new(-1, Relation::EqualTo);
    }

    #[test]
    fn test_total_hits_display_equal_to() {
        let hits = TotalHits::new(1000, Relation::EqualTo);
        assert_eq!(hits.to_string(), "1000 hits");
    }

    #[test]
    fn test_total_hits_display_greater_than_or_equal_to() {
        let hits = TotalHits::new(1000, Relation::GreaterThanOrEqualTo);
        assert_eq!(hits.to_string(), "1000+ hits");
    }

    #[test]
    fn test_total_hits_equality() {
        let hits1 = TotalHits::new(42, Relation::EqualTo);
        let hits2 = TotalHits::new(42, Relation::EqualTo);
        assert_eq!(hits1, hits2);
    }

    #[test]
    fn test_total_hits_inequality_value() {
        let hits1 = TotalHits::new(42, Relation::EqualTo);
        let hits2 = TotalHits::new(43, Relation::EqualTo);
        assert_ne!(hits1, hits2);
    }

    #[test]
    fn test_total_hits_inequality_relation() {
        let hits1 = TotalHits::new(42, Relation::EqualTo);
        let hits2 = TotalHits::new(42, Relation::GreaterThanOrEqualTo);
        assert_ne!(hits1, hits2);
    }

    // -- TopDocs tests --

    #[test]
    fn test_top_docs_construction() {
        let total_hits = TotalHits::new(2, Relation::EqualTo);
        let score_docs = vec![ScoreDoc::new(0, 1.5), ScoreDoc::new(1, 0.8)];
        let top_docs = TopDocs::new(total_hits, score_docs);
        assert_eq!(top_docs.total_hits.value, 2);
        assert_eq!(top_docs.total_hits.relation, Relation::EqualTo);
        assert_eq!(top_docs.score_docs.len(), 2);
        assert_eq!(top_docs.score_docs[0].doc, 0);
        assert_eq!(top_docs.score_docs[1].doc, 1);
    }

    #[test]
    fn test_top_docs_empty() {
        let total_hits = TotalHits::new(0, Relation::EqualTo);
        let top_docs = TopDocs::new(total_hits, vec![]);
        assert_eq!(top_docs.total_hits.value, 0);
        assert_is_empty!(top_docs.score_docs);
    }
}
