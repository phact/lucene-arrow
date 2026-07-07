// SPDX-License-Identifier: Apache-2.0

//! `TermQuery` matches documents containing a specific term, producing BM25 scores.

use std::fmt;
use std::io;
use std::slice;

use crate::codecs::lucene103::postings_reader::{BlockPostingsEnum, IndexFeatures};
use crate::document::IndexOptions;
use crate::index::directory_reader::LeafReaderContext;
use crate::index::doc_values_iterators::NumericDocValues;
use crate::search::collector::{DocAndFloatFeatureBuffer, ScoreMode};
use crate::search::doc_id_set_iterator::{DocIdSetIterator, NO_MORE_DOCS};
use crate::search::index_searcher::IndexSearcher;
use crate::search::query::{
    BatchScoreBulkScorer, BulkScorer, DefaultBulkScorer, Query, ScorerSupplier, Weight,
};
use crate::search::scorable::Scorable;
use crate::search::scorer::{MaxScoreCache, Scorer};
use crate::search::similarity::{CollectionStatistics, SimScorer, TermStatistics};
use crate::search::term_states::TermStates;

// ---------------------------------------------------------------------------
// TermQuery
// ---------------------------------------------------------------------------

/// A query that matches documents containing a term.
///
/// This may be combined with other queries via a `BooleanQuery`.
pub struct TermQuery {
    field: String,
    term: Vec<u8>,
}

impl fmt::Debug for TermQuery {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TermQuery")
            .field("field", &self.field)
            .field("term", &String::from_utf8_lossy(&self.term))
            .finish()
    }
}

impl TermQuery {
    /// Constructs a query for the given field and term bytes.
    pub fn new(field: &str, term: &[u8]) -> Self {
        Self {
            field: field.to_string(),
            term: term.to_vec(),
        }
    }

    /// Returns the field name.
    pub fn field(&self) -> &str {
        &self.field
    }

    /// Returns the term bytes.
    pub fn term(&self) -> &[u8] {
        &self.term
    }
}

impl Query for TermQuery {
    fn create_weight(
        &self,
        searcher: &IndexSearcher,
        score_mode: ScoreMode,
        boost: f32,
    ) -> io::Result<Box<dyn Weight>> {
        // Java: TermStates.build(searcher, term, scoreMode.needsScores())
        let term_states = TermStates::build(searcher, &self.field, &self.term)?;

        // Java: TermWeight constructor (L57-94)
        let similarity = searcher.get_similarity();

        let collection_stats;
        let term_stats;
        if score_mode.needs_scores() {
            collection_stats = searcher.collection_statistics(&self.field)?;
            term_stats = if term_states.doc_freq() > 0 {
                Some(searcher.term_statistics(
                    &self.term,
                    term_states.doc_freq() as i64,
                    term_states.total_term_freq(),
                )?)
            } else {
                None
            };
        } else {
            // We do not need actual stats, use fake stats with docFreq=maxDoc=ttf=1
            collection_stats = Some(CollectionStatistics::new(self.field.clone(), 1, 1, 1, 1));
            term_stats = Some(TermStatistics::new(self.term.clone(), 1, 1));
        }

        // Java: L75-94 — create simScorer once, or null if term doesn't exist
        let sim_scorer = match (&collection_stats, &term_stats) {
            (Some(cs), Some(ts)) if score_mode.needs_scores() => {
                Some(similarity.scorer(boost, cs, slice::from_ref(ts)))
            }
            (Some(cs), Some(ts)) => {
                // Not scoring — assign a dummy scorer to avoid unnecessary allocations
                Some(similarity.scorer(boost, cs, slice::from_ref(ts)))
            }
            _ => None,
        };

        Ok(Box::new(TermWeight {
            field: self.field.clone(),
            term: self.term.clone(),
            sim_scorer,
            term_states,
            score_mode,
        }))
    }
}

// ---------------------------------------------------------------------------
// TermWeight
// ---------------------------------------------------------------------------

/// Expert: Calculate query weights and build scorers for a `TermQuery`.
///
/// Holds the field, term, a single pre-computed `SimScorer`, cached per-leaf term states,
/// and score mode. The `SimScorer` is created once in `create_weight` from aggregated
/// stats and cloned per-leaf.
pub struct TermWeight {
    field: String,
    term: Vec<u8>,
    sim_scorer: Option<Box<dyn SimScorer>>,
    term_states: TermStates,
    score_mode: ScoreMode,
}

impl fmt::Debug for TermWeight {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TermWeight")
            .field("field", &self.field)
            .field("term", &String::from_utf8_lossy(&self.term))
            .field("score_mode", &self.score_mode)
            .finish()
    }
}

impl Weight for TermWeight {
    fn scorer_supplier<'a>(
        &self,
        context: &'a LeafReaderContext,
    ) -> io::Result<Option<Box<dyn ScorerSupplier<'a> + 'a>>> {
        // Java: termStates.get(context) — use cached state, no trie navigation
        let state = match self.term_states.get(context.ord) {
            Some(s) => s,
            None => return Ok(None),
        };

        let reader = &context.reader;

        // Look up the field info (needed for index_options and norms check)
        let field_info = match reader.field_infos().field_info_by_name(&self.field) {
            Some(fi) => fi,
            None => return Ok(None),
        };

        // Clone the SimScorer created in create_weight
        let sim_scorer = self.sim_scorer.as_ref().map(|s| s.box_clone());

        // Build postings enum — use impacts mode for TopScores
        let postings_reader = match reader.postings_reader() {
            Some(pr) => pr,
            None => return Ok(None),
        };

        let index_has_offsets =
            field_info.index_options() >= IndexOptions::DocsAndFreqsAndPositionsAndOffsets;
        let index_features = IndexFeatures {
            has_freq: field_info.index_options().has_freqs(),
            has_pos: field_info.index_options().has_positions(),
            has_offsets_or_payloads: index_has_offsets || field_info.has_payloads(),
        };
        let needs_freq = self.score_mode.needs_scores();
        let postings_enum = if self.score_mode == ScoreMode::TopScores {
            postings_reader.impacts(&state, index_features, needs_freq)?
        } else {
            postings_reader.postings(&state, index_features, needs_freq)?
        };

        // Java: norms = context.reader().getNormValues(term.field())
        let norms = if self.score_mode.needs_scores() {
            reader.get_norm_values(&self.field)?
        } else {
            None
        };

        Ok(Some(Box::new(TermScorerSupplier {
            postings_enum: Some(postings_enum),
            sim_scorer,
            norms,
            doc_freq: state.doc_freq,
            score_mode: self.score_mode,
            top_level_scoring_clause: false,
        })))
    }
}

// ---------------------------------------------------------------------------
// TermScorerSupplier
// ---------------------------------------------------------------------------

/// Supplies a `TermScorer` for a single leaf, holding pre-built components.
struct TermScorerSupplier<'a> {
    postings_enum: Option<BlockPostingsEnum<'a>>,
    sim_scorer: Option<Box<dyn SimScorer>>,
    norms: Option<Box<dyn NumericDocValues + 'a>>,
    doc_freq: i32,
    score_mode: ScoreMode,
    top_level_scoring_clause: bool,
}

impl fmt::Debug for TermScorerSupplier<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TermScorerSupplier")
            .field("doc_freq", &self.doc_freq)
            .field("score_mode", &self.score_mode)
            .finish()
    }
}

impl<'a> ScorerSupplier<'a> for TermScorerSupplier<'a> {
    fn get(&mut self, _lead_cost: i64) -> io::Result<Box<dyn Scorer + 'a>> {
        let postings_enum = self
            .postings_enum
            .take()
            .ok_or_else(|| io::Error::other("ScorerSupplier.get() called more than once"))?;
        let sim_scorer = self
            .sim_scorer
            .take()
            .ok_or_else(|| io::Error::other("ScorerSupplier.get() called more than once"))?;
        let norms = self.norms.take();

        let scorer = TermScorer::new(
            postings_enum,
            sim_scorer,
            norms,
            self.score_mode,
            self.top_level_scoring_clause,
        );
        Ok(Box::new(scorer))
    }

    fn bulk_scorer(&mut self) -> io::Result<Box<dyn BulkScorer + 'a>> {
        if !self.score_mode.needs_scores() {
            let scorer = self.get(i64::MAX)?;
            return Ok(Box::new(DefaultBulkScorer::new(scorer)));
        }
        let scorer = self.get(i64::MAX)?;
        Ok(Box::new(BatchScoreBulkScorer::new(scorer)))
    }

    fn cost(&self) -> i64 {
        self.doc_freq as i64
    }

    fn set_top_level_scoring_clause(&mut self) -> io::Result<()> {
        self.top_level_scoring_clause = true;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// TermScorer
// ---------------------------------------------------------------------------

/// Expert: A `Scorer` for documents matching a `Term`.
///
/// Iterates over matching documents and computes BM25 scores. When
/// `top_level_scoring_clause` is true, includes ImpactsDISI logic inline to skip
/// non-competitive blocks using `MaxScoreCache`.
pub struct TermScorer<'a> {
    postings_enum: BlockPostingsEnum<'a>,
    sim_scorer: Box<dyn SimScorer>,
    max_score_cache: MaxScoreCache,
    norms: Option<Box<dyn NumericDocValues + 'a>>,
    norm_values: Vec<i64>,
    // ImpactsDISI state (inlined — see Java ImpactsDISI)
    min_competitive_score: f32,
    up_to: i32,
    max_score: f32,
    top_level_scoring_clause: bool,
}

impl fmt::Debug for TermScorer<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TermScorer")
            .field("doc_id", &self.postings_enum.doc_id())
            .finish()
    }
}

impl<'a> TermScorer<'a> {
    /// Constructs a new `TermScorer`.
    fn new(
        postings_enum: BlockPostingsEnum<'a>,
        sim_scorer: Box<dyn SimScorer>,
        norms: Option<Box<dyn NumericDocValues + 'a>>,
        _score_mode: ScoreMode,
        top_level_scoring_clause: bool,
    ) -> Self {
        let max_score_cache = MaxScoreCache::new(sim_scorer.as_ref());
        Self {
            postings_enum,
            sim_scorer,
            max_score_cache,
            norms,
            norm_values: Vec::new(),
            min_competitive_score: 0.0,
            up_to: NO_MORE_DOCS,
            max_score: f32::MAX,
            top_level_scoring_clause,
        }
    }

    /// Returns term frequency in the current document.
    pub fn freq(&mut self) -> io::Result<i32> {
        self.postings_enum.freq()
    }

    // -- ImpactsDISI logic inlined (from Java ImpactsDISI) --

    /// Compute the target to advance to, skipping non-competitive blocks.
    /// Matches Java's `ImpactsDISI.advanceTarget(int)`.
    fn advance_target(&mut self, target: i32) -> io::Result<i32> {
        if target <= self.up_to {
            // Still in the current block, considered competitive
            return Ok(target);
        }

        self.up_to = self
            .max_score_cache
            .advance_shallow(&mut self.postings_enum, target)?;
        self.max_score = self
            .max_score_cache
            .get_max_score_for_level_zero(&mut self.postings_enum, self.sim_scorer.as_ref())?;

        let mut target = target;
        loop {
            debug_assert!(self.up_to >= target);

            if self.max_score >= self.min_competitive_score {
                return Ok(target);
            }

            if self.up_to == NO_MORE_DOCS {
                return Ok(NO_MORE_DOCS);
            }

            let skip_up_to = self.max_score_cache.get_skip_up_to(
                &mut self.postings_enum,
                self.sim_scorer.as_ref(),
                self.min_competitive_score,
            )?;
            if skip_up_to == -1 {
                // no further skipping
                target = self.up_to + 1;
            } else if skip_up_to == NO_MORE_DOCS {
                return Ok(NO_MORE_DOCS);
            } else {
                target = skip_up_to + 1;
            }
            self.up_to = self
                .max_score_cache
                .advance_shallow(&mut self.postings_enum, target)?;
            self.max_score = self
                .max_score_cache
                .get_max_score_for_level_zero(&mut self.postings_enum, self.sim_scorer.as_ref())?;
        }
    }

    /// If the current doc is not competitive, advance to a competitive one.
    /// Matches Java's `ImpactsDISI.ensureCompetitive()`.
    fn ensure_competitive(&mut self) -> io::Result<()> {
        if !self.top_level_scoring_clause {
            return Ok(());
        }
        let doc = self.postings_enum.doc_id();
        let advance_target = self.advance_target(doc)?;
        if advance_target != doc {
            self.postings_enum.advance(advance_target)?;
        }
        Ok(())
    }
}

impl Scorable for TermScorer<'_> {
    fn score(&mut self) -> io::Result<f32> {
        // Java TermScorer.score() L96-105
        let freq = self.postings_enum.freq()? as f32;
        let doc_id = self.postings_enum.doc_id();
        let mut norm = 1i64;
        if let Some(ref mut norms) = self.norms
            && norms.advance_exact(doc_id)?
        {
            norm = norms.long_value()?;
        }
        Ok(self.sim_scorer.score(freq, norm))
    }

    fn smoothing_score(&mut self, doc_id: i32) -> io::Result<f32> {
        let mut norm = 1i64;
        if let Some(ref mut norms) = self.norms
            && norms.advance_exact(doc_id)?
        {
            norm = norms.long_value()?;
        }
        Ok(self.sim_scorer.score(0.0, norm))
    }

    fn set_min_competitive_score(&mut self, min_score: f32) -> io::Result<()> {
        if self.top_level_scoring_clause {
            // Matches Java ImpactsDISI.setMinCompetitiveScore
            debug_assert!(min_score >= self.min_competitive_score);
            if min_score > self.min_competitive_score {
                self.min_competitive_score = min_score;
                // Force up_to and max_score to be recomputed
                self.up_to = -1;
            }
        }
        Ok(())
    }
}

impl Scorer for TermScorer<'_> {
    fn doc_id(&self) -> i32 {
        self.postings_enum.doc_id()
    }

    fn iterator(&mut self) -> &mut dyn DocIdSetIterator {
        // NOTE: When top_level_scoring_clause is true, Java returns ImpactsDISI here.
        // We inline the ImpactsDISI logic in next_docs_and_scores/ensureCompetitive
        // instead, so we always return the raw postings_enum. The competitive skipping
        // happens in ensureCompetitive() which is called from next_docs_and_scores().
        &mut self.postings_enum
    }

    fn advance_shallow(&mut self, target: i32) -> io::Result<i32> {
        self.max_score_cache
            .advance_shallow(&mut self.postings_enum, target)
    }

    fn get_max_score(&mut self, up_to: i32) -> io::Result<f32> {
        self.max_score_cache
            .get_max_score(&mut self.postings_enum, self.sim_scorer.as_ref(), up_to)
    }

    fn next_docs_and_scores(
        &mut self,
        up_to: i32,
        buffer: &mut DocAndFloatFeatureBuffer,
    ) -> io::Result<()> {
        // Matches Java TermScorer.nextDocsAndScores:
        // 1. ensureCompetitive (ImpactsDISI)
        // 2. fill docs + freqs
        // 3. fill norms
        // 4. bulk score

        // Note: Java loops here to retry if liveDocs filtering empties the buffer.
        // We skip liveDocs (Bits not ported), so a single pass suffices.
        self.ensure_competitive()?;

        // Fill buffer with docs and freqs
        let batch_size = 64;
        buffer.grow_no_copy(batch_size);
        let mut size = 0;
        let mut doc = self.postings_enum.doc_id();
        while doc < up_to && size < batch_size {
            buffer.docs[size] = doc;
            buffer.features[size] = self.postings_enum.freq()? as f32;
            size += 1;
            doc = self.postings_enum.next_doc()?;
        }
        buffer.size = size;

        // Fill norm values for the batch — Java TermScorer.nextDocsAndScores L134-173
        if self.norm_values.len() < size {
            self.norm_values.resize(size, 1);
            if self.norms.is_none() {
                self.norm_values.fill(1);
            }
        }
        if let Some(ref mut norms) = self.norms {
            for i in 0..size {
                if norms.advance_exact(buffer.docs[i])? {
                    self.norm_values[i] = norms.long_value()?;
                } else {
                    self.norm_values[i] = 1;
                }
            }
        }

        // Bulk score: compute scores from freqs (in features) + norms.
        // Rust can't alias features as both input and output, so score in-place per element.
        // Java uses BulkSimScorer which allows the same array; Rust's borrowing rules prevent it.
        for i in 0..size {
            buffer.features[i] = self
                .sim_scorer
                .score(buffer.features[i], self.norm_values[i]);
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::rc::Rc;
    use std::sync::Arc;

    use super::*;
    use crate::document::DocumentBuilder;
    use crate::index::config::IndexWriterConfig;
    use crate::index::directory_reader::DirectoryReader;
    use crate::index::field::text;
    use crate::index::writer::IndexWriter;
    use crate::search::collector::{LeafCollector, ScoreContext};
    use crate::search::doc_id_set_iterator::NO_MORE_DOCS;
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

    // -- TermQuery construction tests --

    #[test]
    fn test_term_query_new() {
        let q = TermQuery::new("content", b"hello");
        assert_eq!(q.field(), "content");
        assert_eq!(q.term(), b"hello");
    }

    #[test]
    fn test_term_query_create_weight() {
        let (_dir, reader) = build_test_index();
        let searcher = IndexSearcher::new(&reader);
        let q = TermQuery::new("content", b"hello");
        let weight = q.create_weight(&searcher, ScoreMode::Complete, 1.0);
        assert_ok!(weight);
    }

    // -- TermScorer basic scoring tests --

    #[test]
    fn test_term_scorer_iterates_matching_docs() {
        let (_dir, reader) = build_test_index();
        let searcher = IndexSearcher::new(&reader);
        let q = TermQuery::new("content", b"hello");
        let weight = q
            .create_weight(&searcher, ScoreMode::Complete, 1.0)
            .unwrap();

        let leaf = &reader.leaves()[0];
        let supplier = weight.scorer_supplier(leaf).unwrap();
        let mut supplier = assert_some!(supplier);

        let mut scorer = supplier.get(i64::MAX).unwrap();
        let iter = scorer.iterator();

        // "hello" is in docs 0 and 1
        let doc0 = iter.next_doc().unwrap();
        assert_eq!(doc0, 0);
        let doc1 = iter.next_doc().unwrap();
        assert_eq!(doc1, 1);
        let end = iter.next_doc().unwrap();
        assert_eq!(end, NO_MORE_DOCS);
    }

    #[test]
    fn test_term_scorer_scores_are_positive() {
        let (_dir, reader) = build_test_index();
        let searcher = IndexSearcher::new(&reader);
        let q = TermQuery::new("content", b"hello");
        let weight = q
            .create_weight(&searcher, ScoreMode::Complete, 1.0)
            .unwrap();

        let leaf = &reader.leaves()[0];
        let mut scorer = weight
            .scorer_supplier(leaf)
            .unwrap()
            .unwrap()
            .get(i64::MAX)
            .unwrap();

        scorer.iterator().next_doc().unwrap();
        let score0 = scorer.score().unwrap();
        assert!(score0 > 0.0, "score should be positive, got {score0}");

        scorer.iterator().next_doc().unwrap();
        let score1 = scorer.score().unwrap();
        assert!(score1 > 0.0, "score should be positive, got {score1}");
    }

    #[test]
    fn test_term_scorer_rare_term_scores_higher() {
        let (_dir, reader) = build_test_index();
        let searcher = IndexSearcher::new(&reader);

        // "world" is in 2 of 3 docs
        let q_common = TermQuery::new("content", b"world");
        let weight_common = q_common
            .create_weight(&searcher, ScoreMode::Complete, 1.0)
            .unwrap();
        let leaf = &reader.leaves()[0];
        let mut scorer_common = weight_common
            .scorer_supplier(leaf)
            .unwrap()
            .unwrap()
            .get(i64::MAX)
            .unwrap();
        scorer_common.iterator().next_doc().unwrap();
        let score_common = scorer_common.score().unwrap();

        // "peace" is in 1 of 3 docs — rarer term should have higher IDF
        let q_rare = TermQuery::new("content", b"peace");
        let weight_rare = q_rare
            .create_weight(&searcher, ScoreMode::Complete, 1.0)
            .unwrap();
        let mut scorer_rare = weight_rare
            .scorer_supplier(leaf)
            .unwrap()
            .unwrap()
            .get(i64::MAX)
            .unwrap();
        scorer_rare.iterator().next_doc().unwrap();
        let score_rare = scorer_rare.score().unwrap();

        assert!(
            score_rare > score_common,
            "rare term should score higher: peace={score_rare} vs world={score_common}"
        );
    }

    #[test]
    fn test_term_scorer_nonexistent_term() {
        let (_dir, reader) = build_test_index();
        let searcher = IndexSearcher::new(&reader);
        let q = TermQuery::new("content", b"nonexistent");
        let weight = q
            .create_weight(&searcher, ScoreMode::Complete, 1.0)
            .unwrap();

        let leaf = &reader.leaves()[0];
        let supplier = weight.scorer_supplier(leaf).unwrap();
        assert_none!(supplier);
    }

    #[test]
    fn test_term_scorer_nonexistent_field() {
        let (_dir, reader) = build_test_index();
        let searcher = IndexSearcher::new(&reader);
        let q = TermQuery::new("no_such_field", b"hello");
        let weight = q
            .create_weight(&searcher, ScoreMode::Complete, 1.0)
            .unwrap();

        let leaf = &reader.leaves()[0];
        let supplier = weight.scorer_supplier(leaf).unwrap();
        assert_none!(supplier);
    }

    #[test]
    fn test_term_scorer_boost_scales_score() {
        let (_dir, reader) = build_test_index();
        let searcher = IndexSearcher::new(&reader);
        let leaf = &reader.leaves()[0];

        let q1 = TermQuery::new("content", b"hello");
        let weight1 = q1
            .create_weight(&searcher, ScoreMode::Complete, 1.0)
            .unwrap();
        let mut scorer1 = weight1
            .scorer_supplier(leaf)
            .unwrap()
            .unwrap()
            .get(i64::MAX)
            .unwrap();
        scorer1.iterator().next_doc().unwrap();
        let score1 = scorer1.score().unwrap();

        let q2 = TermQuery::new("content", b"hello");
        let weight2 = q2
            .create_weight(&searcher, ScoreMode::Complete, 2.0)
            .unwrap();
        let mut scorer2 = weight2
            .scorer_supplier(leaf)
            .unwrap()
            .unwrap()
            .get(i64::MAX)
            .unwrap();
        scorer2.iterator().next_doc().unwrap();
        let score2 = scorer2.score().unwrap();

        assert!(
            (score2 - score1 * 2.0).abs() < 0.01,
            "2x boost should ~double the score: {score1} vs {score2}"
        );
    }

    // -- BatchScoreBulkScorer tests --

    #[test]
    fn test_batch_score_bulk_scorer_collects_docs() {
        let (_dir, reader) = build_test_index();
        let searcher = IndexSearcher::new(&reader);
        let q = TermQuery::new("content", b"hello");
        let weight = q
            .create_weight(&searcher, ScoreMode::Complete, 1.0)
            .unwrap();

        let leaf = &reader.leaves()[0];
        let scorer = weight
            .scorer_supplier(leaf)
            .unwrap()
            .unwrap()
            .get(i64::MAX)
            .unwrap();

        let mut bulk = BatchScoreBulkScorer::new(scorer);

        #[derive(Debug)]
        struct CollectedDoc {
            docs: Vec<i32>,
        }
        impl LeafCollector for CollectedDoc {
            fn set_scorer(&mut self, _score_context: Rc<ScoreContext>) -> io::Result<()> {
                Ok(())
            }
            fn collect(&mut self, doc: i32) -> io::Result<()> {
                self.docs.push(doc);
                Ok(())
            }
        }

        let mut collector = CollectedDoc { docs: Vec::new() };
        bulk.score(&mut collector, 0, NO_MORE_DOCS).unwrap();

        // "hello" is in docs 0 and 1
        assert_eq!(collector.docs, vec![0, 1]);
    }

    #[test]
    fn test_batch_score_bulk_scorer_respects_range() {
        let (_dir, reader) = build_test_index();
        let searcher = IndexSearcher::new(&reader);
        let q = TermQuery::new("content", b"world");
        let weight = q
            .create_weight(&searcher, ScoreMode::Complete, 1.0)
            .unwrap();

        let leaf = &reader.leaves()[0];
        let scorer = weight
            .scorer_supplier(leaf)
            .unwrap()
            .unwrap()
            .get(i64::MAX)
            .unwrap();

        let mut bulk = BatchScoreBulkScorer::new(scorer);

        #[derive(Debug)]
        struct CollectedDoc {
            docs: Vec<i32>,
        }
        impl LeafCollector for CollectedDoc {
            fn set_scorer(&mut self, _score_context: Rc<ScoreContext>) -> io::Result<()> {
                Ok(())
            }
            fn collect(&mut self, doc: i32) -> io::Result<()> {
                self.docs.push(doc);
                Ok(())
            }
        }

        let mut collector = CollectedDoc { docs: Vec::new() };
        // "world" is in docs 0 and 2. Score only [1, 3) should get doc 2.
        bulk.score(&mut collector, 1, 3).unwrap();

        assert_eq!(collector.docs, vec![2]);
    }
}
