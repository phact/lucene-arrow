// SPDX-License-Identifier: Apache-2.0

//! `IndexSearcher` implements search over a `DirectoryReader`.

use std::fmt;
use std::io;

use crate::index::directory_reader::{DirectoryReader, LeafReaderContext};
use crate::index::terms::Terms;
use crate::search::collector::{Collector, CollectorManager, LeafCollector, ScoreMode};
use crate::search::doc_id_set_iterator::NO_MORE_DOCS;
use crate::search::query::{Query, Weight};
use crate::search::similarity::{BM25Similarity, CollectionStatistics, Similarity, TermStatistics};
use crate::search::top_docs::{ScoreDoc, TopDocs};
use crate::search::top_score_doc_collector::TopScoreDocCollectorManager;

/// By default, we count hits accurately up to 1,000.
const TOTAL_HITS_THRESHOLD: i32 = 1000;

/// Implements search over a single `DirectoryReader`.
///
/// Applications usually need only call `search`. For performance reasons, if your index is
/// unchanging, you should share a single `IndexSearcher` instance across multiple searches
/// instead of creating a new one per search.
///
/// The `search` and `search_after` methods are configured to only count top hits accurately
/// up to 1,000 and may return a lower bound of the hit count if the hit count is greater than
/// or equal to 1,000.
pub struct IndexSearcher<'a> {
    reader: &'a DirectoryReader,
    similarity: Box<dyn Similarity>,
}

impl fmt::Debug for IndexSearcher<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IndexSearcher")
            .field("max_doc", &self.reader.max_doc())
            .field("num_segments", &self.reader.leaves().len())
            .finish()
    }
}

impl<'a> IndexSearcher<'a> {
    /// Creates a searcher searching the provided reader, using `BM25Similarity` as the default.
    pub fn new(reader: &'a DirectoryReader) -> Self {
        Self {
            reader,
            similarity: Box::new(BM25Similarity::default()),
        }
    }

    /// Creates a searcher searching the provided reader with a custom similarity.
    pub fn new_with_similarity(
        reader: &'a DirectoryReader,
        similarity: Box<dyn Similarity>,
    ) -> Self {
        Self { reader, similarity }
    }

    /// Returns the similarity used for scoring.
    pub fn get_similarity(&self) -> &dyn Similarity {
        &*self.similarity
    }

    /// Returns the underlying `DirectoryReader`.
    pub fn get_reader(&self) -> &DirectoryReader {
        self.reader
    }

    /// Finds the top `n` hits for `query`.
    pub fn search(&self, query: &dyn Query, n: i32) -> io::Result<TopDocs> {
        self.search_after(None, query, n)
    }

    /// Finds the top `n` hits for `query` where all results are after `after`.
    ///
    /// By passing the bottom result from a previous page as `after`, this method can be used
    /// for efficient deep-paging across potentially large result sets.
    pub fn search_after(
        &self,
        after: Option<ScoreDoc>,
        query: &dyn Query,
        num_hits: i32,
    ) -> io::Result<TopDocs> {
        let limit = 1.max(self.reader.max_doc());
        if let Some(ref after_doc) = after
            && after_doc.doc >= limit
        {
            return Err(io::Error::other(format!(
                "after.doc exceeds the number of documents in the reader: after.doc={} limit={}",
                after_doc.doc, limit
            )));
        }

        let capped_num_hits = num_hits.min(limit);
        let manager =
            TopScoreDocCollectorManager::new(capped_num_hits, after, TOTAL_HITS_THRESHOLD);

        self.search_with_collector_manager(query, &manager)
    }

    /// Lower-level search API using a `CollectorManager`.
    pub fn search_with_collector_manager<M: CollectorManager>(
        &self,
        query: &dyn Query,
        manager: &M,
    ) -> io::Result<M::Result> {
        let mut collector = manager.new_collector()?;
        let score_mode = collector.score_mode();
        let weight = self.create_weight(query, score_mode, 1.0)?;

        for leaf in self.reader.leaves() {
            self.search_leaf(leaf, weight.as_ref(), &mut collector)?;
        }

        manager.reduce(vec![collector])
    }

    /// Rewrites the given query, returning the rewritten form. Returns the original query
    /// if no rewriting is needed.
    ///
    /// // TODO: implement the iterative rewriting loop
    pub fn rewrite(&self, query: Box<dyn Query>) -> io::Result<Box<dyn Query>> {
        Ok(query)
    }

    /// Creates a `Weight` for the given query.
    pub fn create_weight(
        &self,
        query: &dyn Query,
        score_mode: ScoreMode,
        boost: f32,
    ) -> io::Result<Box<dyn Weight>> {
        // No query cache — delegates directly to the query.
        query.create_weight(self, score_mode, boost)
    }

    /// Searches a single leaf segment.
    fn search_leaf<C: Collector>(
        &self,
        ctx: &LeafReaderContext,
        weight: &dyn Weight,
        collector: &mut C,
    ) -> io::Result<()> {
        let mut leaf_collector = collector.get_leaf_collector(ctx)?;

        let scorer_supplier = weight.scorer_supplier(ctx)?;
        if let Some(mut supplier) = scorer_supplier {
            supplier.set_top_level_scoring_clause()?;
            let mut bulk_scorer = supplier.bulk_scorer()?;
            bulk_scorer.score(&mut leaf_collector, 0, NO_MORE_DOCS)?;
        }

        leaf_collector.finish()?;
        Ok(())
    }

    /// Returns `CollectionStatistics` for a field, or `None` if the field does not exist
    /// (has no indexed terms).
    ///
    /// Aggregates `doc_count`, `sum_total_term_freq`, and `sum_doc_freq` across all segments.
    pub fn collection_statistics(&self, field: &str) -> io::Result<Option<CollectionStatistics>> {
        let mut doc_count: i64 = 0;
        let mut sum_total_term_freq: i64 = 0;
        let mut sum_doc_freq: i64 = 0;

        for leaf in self.reader.leaves() {
            let terms = match leaf.reader.terms(field) {
                Some(t) => t,
                None => continue,
            };

            doc_count += terms.get_doc_count() as i64;
            sum_total_term_freq += terms.get_sum_total_term_freq();
            sum_doc_freq += terms.get_sum_doc_freq();
        }

        if doc_count == 0 {
            return Ok(None);
        }

        Ok(Some(CollectionStatistics::new(
            field.to_string(),
            self.reader.max_doc() as i64,
            doc_count,
            sum_total_term_freq,
            sum_doc_freq,
        )))
    }

    /// Returns `TermStatistics` for a term.
    pub fn term_statistics(
        &self,
        term: &[u8],
        doc_freq: i64,
        total_term_freq: i64,
    ) -> io::Result<TermStatistics> {
        Ok(TermStatistics::new(
            term.to_vec(),
            doc_freq,
            total_term_freq,
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::document::DocumentBuilder;
    use crate::index::config::IndexWriterConfig;
    use crate::index::directory_reader::DirectoryReader;
    use crate::index::field::text;
    use crate::index::writer::IndexWriter;
    use crate::search::term_query::TermQuery;
    use crate::store::{MemoryDirectory, SharedDirectory};
    use assertables::*;

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
    fn test_search_basic() {
        let (_dir, reader) = build_test_index();
        let searcher = IndexSearcher::new(&reader);
        let query = TermQuery::new("content", b"hello");
        let top_docs = searcher.search(&query, 10).unwrap();

        // "hello" appears in docs 0 and 1
        assert_eq!(top_docs.total_hits.value, 2);
        assert_eq!(top_docs.score_docs.len(), 2);
    }

    #[test]
    fn test_search_nonexistent_term() {
        let (_dir, reader) = build_test_index();
        let searcher = IndexSearcher::new(&reader);
        let query = TermQuery::new("content", b"nonexistent");
        let top_docs = searcher.search(&query, 10).unwrap();

        assert_eq!(top_docs.total_hits.value, 0);
        assert_is_empty!(top_docs.score_docs);
    }

    #[test]
    fn test_search_top_n_limits_results() {
        let (_dir, reader) = build_test_index();
        let searcher = IndexSearcher::new(&reader);
        // "hello" matches 2 docs, but request only 1
        let query = TermQuery::new("content", b"hello");
        let top_docs = searcher.search(&query, 1).unwrap();

        assert_eq!(top_docs.score_docs.len(), 1);
        assert_ge!(top_docs.total_hits.value, 1);
    }

    #[test]
    fn test_search_scores_sorted_descending() {
        let (_dir, reader) = build_test_index();
        let searcher = IndexSearcher::new(&reader);
        let query = TermQuery::new("content", b"hello");
        let top_docs = searcher.search(&query, 10).unwrap();

        for i in 1..top_docs.score_docs.len() {
            assert_ge!(
                top_docs.score_docs[i - 1].score,
                top_docs.score_docs[i].score,
                "scores should be sorted descending"
            );
        }
    }

    #[test]
    fn test_collection_statistics() {
        let (_dir, reader) = build_test_index();
        let searcher = IndexSearcher::new(&reader);
        let stats = searcher.collection_statistics("content").unwrap();

        let stats = assert_some!(stats);
        assert_eq!(stats.max_doc(), 3);
        assert_gt!(stats.doc_count(), 0);
        assert_gt!(stats.sum_doc_freq(), 0);
        assert_gt!(stats.sum_total_term_freq(), 0);
    }

    #[test]
    fn test_collection_statistics_nonexistent_field() {
        let (_dir, reader) = build_test_index();
        let searcher = IndexSearcher::new(&reader);
        let stats = searcher.collection_statistics("no_such_field").unwrap();
        assert_none!(stats);
    }

    #[test]
    fn test_search_rare_term_higher_score() {
        let (_dir, reader) = build_test_index();
        let searcher = IndexSearcher::new(&reader);

        // "world" is in 2 of 3 docs
        let query_common = TermQuery::new("content", b"world");
        let top_common = searcher.search(&query_common, 10).unwrap();

        // "peace" is in 1 of 3 docs — rarer term should have higher IDF
        let query_rare = TermQuery::new("content", b"peace");
        let top_rare = searcher.search(&query_rare, 10).unwrap();

        assert_not_empty!(top_common.score_docs);
        assert_not_empty!(top_rare.score_docs);
        assert_gt!(
            top_rare.score_docs[0].score,
            top_common.score_docs[0].score,
            "rare term should score higher"
        );
    }
}
