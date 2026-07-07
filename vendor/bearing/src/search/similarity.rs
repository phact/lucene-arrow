// SPDX-License-Identifier: Apache-2.0

//! Scoring similarity models, collection/term statistics, and BM25.

use crate::util::small_float;

// ---------------------------------------------------------------------------
// CollectionStatistics
// ---------------------------------------------------------------------------

/// Contains statistics for a collection (field).
///
/// This struct holds statistics across all documents for scoring purposes:
/// - `max_doc`: number of documents.
/// - `doc_count`: number of documents that contain this field.
/// - `sum_doc_freq`: number of postings-list entries.
/// - `sum_total_term_freq`: number of tokens.
///
/// The following conditions are always true:
/// - All statistics are positive integers: never zero or negative.
/// - `doc_count` <= `max_doc`
/// - `doc_count` <= `sum_doc_freq` <= `sum_total_term_freq`
#[derive(Debug, Clone)]
pub struct CollectionStatistics {
    field: String,
    max_doc: i64,
    doc_count: i64,
    sum_total_term_freq: i64,
    sum_doc_freq: i64,
}

impl CollectionStatistics {
    /// Creates statistics instance for a collection (field).
    ///
    /// # Panics
    ///
    /// Panics if any of the invariants are violated:
    /// - `max_doc` must be positive.
    /// - `doc_count` must be positive and not exceed `max_doc`.
    /// - `sum_doc_freq` must be positive and at least `doc_count`.
    /// - `sum_total_term_freq` must be positive and at least `sum_doc_freq`.
    pub fn new(
        field: String,
        max_doc: i64,
        doc_count: i64,
        sum_total_term_freq: i64,
        sum_doc_freq: i64,
    ) -> Self {
        assert!(max_doc > 0, "maxDoc must be positive, maxDoc: {}", max_doc);
        assert!(
            doc_count > 0,
            "docCount must be positive, docCount: {}",
            doc_count
        );
        assert!(
            doc_count <= max_doc,
            "docCount must not exceed maxDoc, docCount: {}, maxDoc: {}",
            doc_count,
            max_doc
        );
        assert!(
            sum_doc_freq > 0,
            "sumDocFreq must be positive, sumDocFreq: {}",
            sum_doc_freq
        );
        assert!(
            sum_doc_freq >= doc_count,
            "sumDocFreq must be at least docCount, sumDocFreq: {}, docCount: {}",
            sum_doc_freq,
            doc_count
        );
        assert!(
            sum_total_term_freq > 0,
            "sumTotalTermFreq must be positive, sumTotalTermFreq: {}",
            sum_total_term_freq
        );
        assert!(
            sum_total_term_freq >= sum_doc_freq,
            "sumTotalTermFreq must be at least sumDocFreq, sumTotalTermFreq: {}, sumDocFreq: {}",
            sum_total_term_freq,
            sum_doc_freq
        );
        Self {
            field,
            max_doc,
            doc_count,
            sum_total_term_freq,
            sum_doc_freq,
        }
    }

    /// Field's name.
    pub fn field(&self) -> &str {
        &self.field
    }

    /// The total number of documents, regardless of whether they all contain values for this
    /// field.
    pub fn max_doc(&self) -> i64 {
        self.max_doc
    }

    /// The total number of documents that have at least one term for this field.
    pub fn doc_count(&self) -> i64 {
        self.doc_count
    }

    /// The total number of tokens for this field.
    pub fn sum_total_term_freq(&self) -> i64 {
        self.sum_total_term_freq
    }

    /// The total number of posting list entries for this field.
    pub fn sum_doc_freq(&self) -> i64 {
        self.sum_doc_freq
    }
}

// ---------------------------------------------------------------------------
// TermStatistics
// ---------------------------------------------------------------------------

/// Contains statistics for a specific term.
///
/// This struct holds statistics for this term across all documents for scoring purposes:
/// - `doc_freq`: number of documents this term occurs in.
/// - `total_term_freq`: number of tokens for this term.
///
/// The following conditions are always true:
/// - All statistics are positive integers: never zero or negative.
/// - `doc_freq` <= `total_term_freq`
#[derive(Debug, Clone)]
pub struct TermStatistics {
    term: Vec<u8>,
    doc_freq: i64,
    total_term_freq: i64,
}

impl TermStatistics {
    /// Creates statistics instance for a term.
    ///
    /// # Panics
    ///
    /// Panics if `doc_freq` is not positive, `total_term_freq` is not positive,
    /// or `total_term_freq` is less than `doc_freq`.
    pub fn new(term: Vec<u8>, doc_freq: i64, total_term_freq: i64) -> Self {
        assert!(
            doc_freq > 0,
            "docFreq must be positive, docFreq: {}",
            doc_freq
        );
        assert!(
            total_term_freq > 0,
            "totalTermFreq must be positive, totalTermFreq: {}",
            total_term_freq
        );
        assert!(
            total_term_freq >= doc_freq,
            "totalTermFreq must be at least docFreq, totalTermFreq: {}, docFreq: {}",
            total_term_freq,
            doc_freq
        );
        Self {
            term,
            doc_freq,
            total_term_freq,
        }
    }

    /// Term bytes.
    pub fn term(&self) -> &[u8] {
        &self.term
    }

    /// Number of documents containing the term in the collection.
    pub fn doc_freq(&self) -> i64 {
        self.doc_freq
    }

    /// Number of occurrences of the term in the collection.
    pub fn total_term_freq(&self) -> i64 {
        self.total_term_freq
    }
}

// ---------------------------------------------------------------------------
// Similarity / SimScorer / BulkSimScorer traits
// ---------------------------------------------------------------------------

/// Stores the weight for a query across the indexed collection.
///
/// Subclasses define the statistics they require (e.g. IDF, average field length).
/// The `score` method is called for every matching document to compute its score.
pub trait SimScorer {
    /// Score a single document. `freq` is the document-term sloppy frequency and must be
    /// finite and positive. `norm` is the encoded normalization factor as computed by
    /// `Similarity::compute_norm` at index time, or `1` if norms are disabled.
    /// `norm` is never `0`.
    ///
    /// Score must not decrease when `freq` increases. Score must not increase when the
    /// unsigned `norm` increases.
    fn score(&self, freq: f32, norm: i64) -> f32;

    /// Returns a boxed clone of this scorer.
    fn box_clone(&self) -> Box<dyn SimScorer>;
}

/// Specialization of `SimScorer` for bulk-computation of scores.
pub trait BulkSimScorer {
    /// Bulk computation of scores. For each index `i` in `[0, size)`, `scores[i]` is computed
    /// as `score(freqs[i], norms[i])`.
    ///
    /// **NOTE:** It is legal to pass the same `freqs` and `scores` slices.
    fn score(&self, freqs: &[f32], norms: &[i64], scores: &mut [f32]);
}

/// Default `BulkSimScorer` implementation that delegates to a `SimScorer`.
pub struct DefaultBulkSimScorer<'a> {
    scorer: &'a dyn SimScorer,
}

impl<'a> DefaultBulkSimScorer<'a> {
    /// Creates a new `DefaultBulkSimScorer` wrapping the given `SimScorer`.
    pub fn new(scorer: &'a dyn SimScorer) -> Self {
        Self { scorer }
    }
}

impl BulkSimScorer for DefaultBulkSimScorer<'_> {
    fn score(&self, freqs: &[f32], norms: &[i64], scores: &mut [f32]) {
        let size = freqs.len().min(norms.len()).min(scores.len());
        for i in 0..size {
            scores[i] = self.scorer.score(freqs[i], norms[i]);
        }
    }
}

/// Similarity defines the components of Lucene scoring.
///
/// Similarity determines how Lucene weights terms. At index time, the indexer calls
/// `compute_norm`, allowing the Similarity implementation to set a per-document value for
/// the field. At query time, `scorer` is called to compute collection-level weight (e.g.
/// IDF), and then `SimScorer::score` is called for every matching document.
pub trait Similarity {
    /// Whether overlap tokens (tokens with a position increment of zero) are discounted from
    /// the document's length.
    fn get_discount_overlaps(&self) -> bool;

    /// Computes the normalization value for a field at index-time.
    ///
    /// The default implementation is provided by concrete implementations (e.g. BM25Similarity).
    fn compute_norm(&self, num_terms: i32) -> i64;

    /// Compute any collection-level weight (e.g. IDF, average document length, etc) needed for
    /// scoring a query.
    ///
    /// `boost` is a multiplicative factor to apply to the produced scores.
    /// `collection_stats` contains collection-level statistics.
    /// `term_stats` contains term-level statistics for each term.
    fn scorer(
        &self,
        boost: f32,
        collection_stats: &CollectionStatistics,
        term_stats: &[TermStatistics],
    ) -> Box<dyn SimScorer>;

    /// Clone this `Similarity` into a `Box`. Required for storing trait objects.
    fn box_clone(&self) -> Box<dyn Similarity>;
}

// ---------------------------------------------------------------------------
// BM25Similarity
// ---------------------------------------------------------------------------

/// Cache of decoded bytes. Maps norm byte → document length.
static LENGTH_TABLE: [f32; 256] = {
    let mut table = [0.0f32; 256];
    let mut i = 0u32;
    while i < 256 {
        table[i as usize] = small_float::byte4_to_int(i as u8) as f32;
        i += 1;
    }
    table
};

/// BM25 Similarity. Introduced in Stephen E. Robertson, Steve Walker, Susan Jones,
/// Micheline Hancock-Beaulieu, and Mike Gatford. Okapi at TREC-3.
#[derive(Debug)]
pub struct BM25Similarity {
    k1: f32,
    b: f32,
    discount_overlaps: bool,
}

impl BM25Similarity {
    /// BM25 with the supplied parameter values.
    ///
    /// # Panics
    ///
    /// Panics if `k1` is infinite or negative, or if `b` is not within the range `[0..1]`.
    pub fn new(k1: f32, b: f32, discount_overlaps: bool) -> Self {
        assert!(
            k1.is_finite() && k1 >= 0.0,
            "illegal k1 value: {}, must be a non-negative finite value",
            k1
        );
        assert!(
            !b.is_nan() && (0.0..=1.0).contains(&b),
            "illegal b value: {}, must be between 0 and 1",
            b
        );
        Self {
            k1,
            b,
            discount_overlaps,
        }
    }

    /// BM25 with the supplied parameter values and `discount_overlaps = true`.
    pub fn new_with_defaults(k1: f32, b: f32) -> Self {
        Self::new(k1, b, true)
    }

    /// Implemented as `log(1 + (docCount - docFreq + 0.5) / (docFreq + 0.5))`.
    fn idf(doc_freq: i64, doc_count: i64) -> f32 {
        ((1.0_f64 + (doc_count as f64 - doc_freq as f64 + 0.5) / (doc_freq as f64 + 0.5)).ln())
            as f32
    }

    /// The default implementation computes the average as `sumTotalTermFreq / docCount`.
    fn avg_field_length(collection_stats: &CollectionStatistics) -> f32 {
        (collection_stats.sum_total_term_freq() as f64 / collection_stats.doc_count() as f64) as f32
    }

    /// Returns the `k1` parameter.
    pub fn get_k1(&self) -> f32 {
        self.k1
    }

    /// Returns the `b` parameter.
    pub fn get_b(&self) -> f32 {
        self.b
    }
}

impl Default for BM25Similarity {
    /// BM25 with default values: k1 = 1.2, b = 0.75, discountOverlaps = true.
    fn default() -> Self {
        Self::new(1.2, 0.75, true)
    }
}

impl Similarity for BM25Similarity {
    fn get_discount_overlaps(&self) -> bool {
        self.discount_overlaps
    }

    fn compute_norm(&self, num_terms: i32) -> i64 {
        small_float::int_to_byte4(num_terms) as i64
    }

    fn scorer(
        &self,
        boost: f32,
        collection_stats: &CollectionStatistics,
        term_stats: &[TermStatistics],
    ) -> Box<dyn SimScorer> {
        let idf = if term_stats.len() == 1 {
            Self::idf(term_stats[0].doc_freq(), collection_stats.doc_count())
        } else {
            let mut idf_sum = 0.0_f64;
            for ts in term_stats {
                idf_sum += Self::idf(ts.doc_freq(), collection_stats.doc_count()) as f64;
            }
            idf_sum as f32
        };

        let avgdl = Self::avg_field_length(collection_stats);

        let mut cache = [0.0f32; 256];
        for (i, entry) in cache.iter_mut().enumerate() {
            *entry = 1.0 / (self.k1 * ((1.0 - self.b) + self.b * LENGTH_TABLE[i] / avgdl));
        }

        let weight = boost * idf;
        Box::new(BM25Scorer { weight, cache })
    }

    fn box_clone(&self) -> Box<dyn Similarity> {
        Box::new(BM25Similarity::new(self.k1, self.b, self.discount_overlaps))
    }
}

/// Collection statistics for the BM25 model.
struct BM25Scorer {
    /// weight (idf * boost)
    weight: f32,
    /// precomputed `norm[256]` with `k1 * ((1 - b) + b * dl / avgdl)`
    cache: [f32; 256],
}

impl BM25Scorer {
    fn do_score(&self, freq: f32, norm_inverse: f32) -> f32 {
        // In order to guarantee monotonicity with both freq and norm without
        // promoting to doubles, we rewrite freq / (freq + norm) to
        // 1 - 1 / (1 + freq * 1/norm).
        // Finally we expand weight * (1 - 1 / (1 + freq * 1/norm)) to
        // weight - weight / (1 + freq * 1/norm), which runs slightly faster.
        self.weight - self.weight / (1.0 + freq * norm_inverse)
    }
}

impl SimScorer for BM25Scorer {
    fn score(&self, freq: f32, encoded_norm: i64) -> f32 {
        let norm_inverse = self.cache[(encoded_norm as u8) as usize];
        self.do_score(freq, norm_inverse)
    }

    fn box_clone(&self) -> Box<dyn SimScorer> {
        Box::new(BM25Scorer {
            weight: self.weight,
            cache: self.cache,
        })
    }
}

/// BM25 bulk scorer that pre-decodes norms for vectorization.
#[expect(dead_code)]
pub(crate) struct BM25BulkSimScorer {
    weight: f32,
    cache: [f32; 256],
    norm_inverses: Vec<f32>,
}

impl BM25BulkSimScorer {
    #[expect(dead_code)]
    pub(crate) fn new(weight: f32, cache: [f32; 256]) -> Self {
        Self {
            weight,
            cache,
            norm_inverses: Vec::new(),
        }
    }
}

impl BulkSimScorer for BM25BulkSimScorer {
    fn score(&self, freqs: &[f32], norms: &[i64], scores: &mut [f32]) {
        let size = freqs.len().min(norms.len()).min(scores.len());

        // This two-pass approach matches Java's BM25Scorer.asBulkSimScorer():
        // first decode all norm inverses, then compute all scores.
        // The second loop auto-vectorizes.
        let mut norm_inverses = vec![0.0f32; size];
        for i in 0..size {
            norm_inverses[i] = self.cache[(norms[i] as u8) as usize];
        }

        let weight = self.weight;
        for i in 0..size {
            scores[i] = weight - weight / (1.0 + freqs[i] * norm_inverses[i]);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::slice;

    use super::*;
    use assertables::*;

    // -- CollectionStatistics tests --

    #[test]
    fn test_collection_stats_valid() {
        let stats = CollectionStatistics::new("body".to_string(), 100, 50, 500, 200);
        assert_eq!(stats.field(), "body");
        assert_eq!(stats.max_doc(), 100);
        assert_eq!(stats.doc_count(), 50);
        assert_eq!(stats.sum_total_term_freq(), 500);
        assert_eq!(stats.sum_doc_freq(), 200);
    }

    #[test]
    #[should_panic(expected = "maxDoc must be positive")]
    fn test_collection_stats_max_doc_zero() {
        CollectionStatistics::new("f".to_string(), 0, 1, 1, 1);
    }

    #[test]
    #[should_panic(expected = "maxDoc must be positive")]
    fn test_collection_stats_max_doc_negative() {
        CollectionStatistics::new("f".to_string(), -1, 1, 1, 1);
    }

    #[test]
    #[should_panic(expected = "docCount must be positive")]
    fn test_collection_stats_doc_count_zero() {
        CollectionStatistics::new("f".to_string(), 10, 0, 1, 1);
    }

    #[test]
    #[should_panic(expected = "docCount must not exceed maxDoc")]
    fn test_collection_stats_doc_count_exceeds_max_doc() {
        CollectionStatistics::new("f".to_string(), 10, 11, 20, 15);
    }

    #[test]
    #[should_panic(expected = "sumDocFreq must be positive")]
    fn test_collection_stats_sum_doc_freq_zero() {
        CollectionStatistics::new("f".to_string(), 10, 5, 10, 0);
    }

    #[test]
    #[should_panic(expected = "sumDocFreq must be at least docCount")]
    fn test_collection_stats_sum_doc_freq_less_than_doc_count() {
        CollectionStatistics::new("f".to_string(), 10, 5, 10, 3);
    }

    #[test]
    #[should_panic(expected = "sumTotalTermFreq must be positive")]
    fn test_collection_stats_sum_total_term_freq_zero() {
        CollectionStatistics::new("f".to_string(), 10, 5, 0, 5);
    }

    #[test]
    #[should_panic(expected = "sumTotalTermFreq must be at least sumDocFreq")]
    fn test_collection_stats_sum_total_term_freq_less_than_sum_doc_freq() {
        CollectionStatistics::new("f".to_string(), 10, 5, 4, 5);
    }

    #[test]
    fn test_collection_stats_minimum_valid() {
        let stats = CollectionStatistics::new("f".to_string(), 1, 1, 1, 1);
        assert_eq!(stats.max_doc(), 1);
        assert_eq!(stats.doc_count(), 1);
        assert_eq!(stats.sum_total_term_freq(), 1);
        assert_eq!(stats.sum_doc_freq(), 1);
    }

    // -- TermStatistics tests --

    fn term(s: &str) -> Vec<u8> {
        s.as_bytes().to_vec()
    }

    #[test]
    fn test_term_stats_valid() {
        let stats = TermStatistics::new(term("hello"), 10, 50);
        assert_eq!(stats.term(), &term("hello"));
        assert_eq!(stats.doc_freq(), 10);
        assert_eq!(stats.total_term_freq(), 50);
    }

    #[test]
    #[should_panic(expected = "docFreq must be positive")]
    fn test_term_stats_doc_freq_zero() {
        TermStatistics::new(term("t"), 0, 1);
    }

    #[test]
    #[should_panic(expected = "docFreq must be positive")]
    fn test_term_stats_doc_freq_negative() {
        TermStatistics::new(term("t"), -1, 1);
    }

    #[test]
    #[should_panic(expected = "totalTermFreq must be positive")]
    fn test_term_stats_total_term_freq_zero() {
        TermStatistics::new(term("t"), 1, 0);
    }

    #[test]
    #[should_panic(expected = "totalTermFreq must be positive")]
    fn test_term_stats_total_term_freq_negative() {
        TermStatistics::new(term("t"), 1, -1);
    }

    #[test]
    #[should_panic(expected = "totalTermFreq must be at least docFreq")]
    fn test_term_stats_total_term_freq_less_than_doc_freq() {
        TermStatistics::new(term("t"), 10, 5);
    }

    #[test]
    fn test_term_stats_minimum_valid() {
        let stats = TermStatistics::new(term("t"), 1, 1);
        assert_eq!(stats.doc_freq(), 1);
        assert_eq!(stats.total_term_freq(), 1);
    }

    // -- SimScorer / DefaultBulkSimScorer tests --

    struct ConstantSimScorer {
        value: f32,
    }

    impl SimScorer for ConstantSimScorer {
        fn score(&self, _freq: f32, _norm: i64) -> f32 {
            self.value
        }

        fn box_clone(&self) -> Box<dyn SimScorer> {
            Box::new(ConstantSimScorer { value: self.value })
        }
    }

    #[test]
    fn test_sim_scorer() {
        let scorer = ConstantSimScorer { value: 2.5 };
        assert_eq!(scorer.score(1.0, 1), 2.5);
        assert_eq!(scorer.score(5.0, 100), 2.5);
    }

    #[test]
    fn test_default_bulk_sim_scorer() {
        let scorer = ConstantSimScorer { value: 3.0 };
        let bulk = DefaultBulkSimScorer::new(&scorer);
        let freqs = [1.0, 2.0, 3.0];
        let norms = [1, 2, 3];
        let mut scores = [0.0f32; 3];
        bulk.score(&freqs, &norms, &mut scores);
        assert_eq!(scores, [3.0, 3.0, 3.0]);
    }

    #[test]
    fn test_default_bulk_sim_scorer_varying_scores() {
        struct LinearSimScorer;
        impl SimScorer for LinearSimScorer {
            fn score(&self, freq: f32, _norm: i64) -> f32 {
                freq * 2.0
            }

            fn box_clone(&self) -> Box<dyn SimScorer> {
                Box::new(LinearSimScorer)
            }
        }

        let scorer = LinearSimScorer;
        let bulk = DefaultBulkSimScorer::new(&scorer);
        let freqs = [1.0, 2.0, 4.0];
        let norms = [1, 1, 1];
        let mut scores = [0.0f32; 3];
        bulk.score(&freqs, &norms, &mut scores);
        assert_eq!(scores, [2.0, 4.0, 8.0]);
    }

    // -- BM25Similarity tests --

    fn test_collection_stats() -> CollectionStatistics {
        CollectionStatistics::new("body".to_string(), 100, 100, 1000, 500)
    }

    fn test_term_stats() -> TermStatistics {
        TermStatistics::new(term("test"), 10, 50)
    }

    #[test]
    fn test_bm25_default_parameters() {
        let sim = BM25Similarity::default();
        assert_in_delta!(sim.get_k1(), 1.2, 0.001);
        assert_in_delta!(sim.get_b(), 0.75, 0.001);
        assert!(sim.get_discount_overlaps());
    }

    #[test]
    fn test_bm25_idf_single_doc() {
        let result = BM25Similarity::idf(1, 1);
        assert_in_delta!(result, (1.0_f64 + 0.5 / 1.5).ln() as f32, 0.0001);
    }

    #[test]
    fn test_bm25_idf_rare_term() {
        let result = BM25Similarity::idf(1, 10000);
        let expected = (1.0_f64 + (10000.0 - 1.0 + 0.5) / (1.0 + 0.5)).ln() as f32;
        assert_in_delta!(result, expected, 0.0001);
        assert_gt!(result, 0.0);
    }

    #[test]
    fn test_bm25_idf_common_term() {
        let result = BM25Similarity::idf(9999, 10000);
        assert_gt!(result, 0.0);
        assert_lt!(result, BM25Similarity::idf(1, 10000));
    }

    #[test]
    fn test_bm25_score_increases_with_freq() {
        let sim = BM25Similarity::default();
        let coll = test_collection_stats();
        let ts = test_term_stats();
        let scorer = sim.scorer(1.0, &coll, &[ts]);

        let norm: i64 = 10;
        let score1 = scorer.score(1.0, norm);
        let score2 = scorer.score(5.0, norm);
        let score3 = scorer.score(20.0, norm);

        assert_gt!(score1, 0.0);
        assert_gt!(score2, score1);
        assert_gt!(score3, score2);
    }

    #[test]
    fn test_bm25_score_with_boost() {
        let sim = BM25Similarity::default();
        let coll = test_collection_stats();
        let ts = test_term_stats();

        let scorer1 = sim.scorer(1.0, &coll, slice::from_ref(&ts));
        let scorer2 = sim.scorer(2.0, &coll, &[ts]);

        let norm: i64 = 10;
        let score1 = scorer1.score(5.0, norm);
        let score2 = scorer2.score(5.0, norm);

        assert_in_delta!(score2, score1 * 2.0, 0.001);
    }

    #[test]
    fn test_bm25_length_table_small_values() {
        assert_in_delta!(LENGTH_TABLE[0], 0.0, 0.001);
        assert_in_delta!(LENGTH_TABLE[1], 1.0, 0.001);
        assert_in_delta!(LENGTH_TABLE[10], 10.0, 0.001);
    }

    #[test]
    fn test_bm25_length_table_monotonic() {
        for i in 1..256 {
            assert_ge!(
                LENGTH_TABLE[i],
                LENGTH_TABLE[i - 1],
                "LENGTH_TABLE not monotonic at index {i}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "illegal k1 value")]
    fn test_bm25_negative_k1() {
        BM25Similarity::new(-1.0, 0.75, true);
    }

    #[test]
    #[should_panic(expected = "illegal k1 value")]
    fn test_bm25_infinite_k1() {
        BM25Similarity::new(f32::INFINITY, 0.75, true);
    }

    #[test]
    #[should_panic(expected = "illegal b value")]
    fn test_bm25_b_out_of_range() {
        BM25Similarity::new(1.2, 1.5, true);
    }

    #[test]
    #[should_panic(expected = "illegal b value")]
    fn test_bm25_negative_b() {
        BM25Similarity::new(1.2, -0.1, true);
    }

    #[test]
    fn test_bm25_bulk_scorer_matches_individual() {
        let sim = BM25Similarity::default();
        let coll = test_collection_stats();
        let ts = test_term_stats();
        let scorer = sim.scorer(1.0, &coll, &[ts]);

        let bulk = DefaultBulkSimScorer::new(scorer.as_ref());

        let freqs = [1.0, 3.0, 5.0, 10.0];
        let norms = [10i64, 20, 30, 40];
        let mut bulk_scores = [0.0f32; 4];
        bulk.score(&freqs, &norms, &mut bulk_scores);

        for i in 0..4 {
            let individual = scorer.score(freqs[i], norms[i]);
            assert_in_delta!(bulk_scores[i], individual, 0.0001);
        }
    }
}
