// SPDX-License-Identifier: Apache-2.0

//! Scorer hierarchy: `Scorer` trait, `MaxScoreCache`, `ImpactsDISI`, `MaxScoreAccumulator`,
//! and `DocScoreEncoder`.

use std::fmt;
use std::io;
use std::sync::atomic;

use crate::codecs::competitive_impact::Impact;
use crate::search::collector::DocAndFloatFeatureBuffer;
use crate::search::doc_id_set_iterator::{DocIdSetIterator, NO_MORE_DOCS};
use crate::search::scorable::Scorable;
use crate::search::similarity::SimScorer;

// ---------------------------------------------------------------------------
// Impacts / ImpactsSource
// ---------------------------------------------------------------------------

/// Information about upcoming impacts, ie. (freq, norm) pairs.
pub trait Impacts {
    /// Return the number of levels on which we have impacts. The returned value is always
    /// greater than 0 and may not always be the same, even on a single postings list.
    fn num_levels(&self) -> usize;

    /// Return the maximum inclusive doc ID until which the list of impacts returned by
    /// `get_impacts` is valid. This is a non-decreasing function of `level`.
    fn get_doc_id_up_to(&self, level: usize) -> i32;

    /// Return impacts on the given level. Sorted by increasing frequency and increasing
    /// unsigned norm.
    fn get_impacts(&self, level: usize) -> &[Impact];
}

/// Source of `Impacts`.
pub trait ImpactsSource {
    /// Shallow-advance to `target`. This is cheaper than calling `DocIdSetIterator::advance`
    /// and allows further calls to `get_impacts` to ignore doc IDs that are less than `target`.
    fn advance_shallow(&mut self, target: i32) -> io::Result<()>;

    /// Get information about upcoming impacts for doc IDs >= the maximum of the current doc ID
    /// and the last target passed to `advance_shallow`.
    fn get_impacts(&mut self) -> io::Result<&dyn Impacts>;
}

// ---------------------------------------------------------------------------
// Scorer
// ---------------------------------------------------------------------------

/// Expert: Common scoring functionality for different types of queries.
///
/// A `Scorer` exposes an iterator over documents matching a query in increasing order of
/// doc id.
pub trait Scorer: Scorable + fmt::Debug {
    /// Returns the doc ID that is currently being scored.
    fn doc_id(&self) -> i32;

    /// Return a reference to the `DocIdSetIterator` over matching documents.
    fn iterator(&mut self) -> &mut dyn DocIdSetIterator;

    /// Advance to the block of documents that contains `target` in order to get scoring
    /// information about this block. Returns a number >= all documents in the current block,
    /// but < any doc IDs of the next block.
    ///
    /// The default implementation returns `NO_MORE_DOCS`.
    fn advance_shallow(&mut self, _target: i32) -> io::Result<i32> {
        Ok(NO_MORE_DOCS)
    }

    /// Return the maximum score that documents between the last `target` that this iterator
    /// was shallow-advanced to included and `up_to` included.
    fn get_max_score(&mut self, up_to: i32) -> io::Result<f32>;

    /// Return a new batch of doc IDs and scores, starting at the current doc ID, and ending
    /// before `up_to`.
    ///
    /// The default implementation fills a batch of up to 64 entries.
    fn next_docs_and_scores(
        &mut self,
        up_to: i32,
        buffer: &mut DocAndFloatFeatureBuffer,
    ) -> io::Result<()> {
        let batch_size = 64;
        buffer.grow_no_copy(batch_size);
        let mut size = 0;
        let doc_id = self.doc_id();
        let mut doc = doc_id;
        while doc < up_to && size < batch_size {
            buffer.docs[size] = doc;
            buffer.features[size] = self.score()?;
            size += 1;
            doc = self.iterator().next_doc()?;
        }
        buffer.size = size;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// MaxScoreCache
// ---------------------------------------------------------------------------

/// Compute maximum scores based on `Impacts` and keep them in a cache in order not to run
/// expensive similarity score computations multiple times on the same data.
///
/// Unlike Java's `MaxScoreCache` which holds a reference to the `ImpactsSource`, this Rust
/// version takes the `ImpactsSource` as a parameter on each method call. This avoids
/// borrow checker conflicts when `TermScorer` needs to own both the iterator (which is the
/// `ImpactsSource`) and the cache simultaneously.
/// Compute maximum scores based on `Impacts` and keep them in a cache in order not to run
/// expensive similarity score computations multiple times on the same data.
///
/// Unlike Java's `MaxScoreCache`, this Rust version takes both the `ImpactsSource` and
/// `SimScorer` as parameters on each method call. This avoids borrow checker conflicts
/// when `TermScorer` needs to own the iterator, scorer, and cache simultaneously.
pub struct MaxScoreCache {
    global_max_score: f32,
    max_score_cache: Vec<f32>,
    max_score_cache_up_to: Vec<i32>,
}

impl fmt::Debug for MaxScoreCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MaxScoreCache")
            .field("global_max_score", &self.global_max_score)
            .field("cache_size", &self.max_score_cache.len())
            .finish()
    }
}

impl MaxScoreCache {
    /// Sole constructor. Computes the global max score from the given scorer.
    pub fn new(scorer: &dyn SimScorer) -> Self {
        let global_max_score = scorer.score(f32::MAX, 1);
        Self {
            global_max_score,
            max_score_cache: Vec::new(),
            max_score_cache_up_to: Vec::new(),
        }
    }

    /// Implement the contract of `Scorer::advance_shallow` based on the given `ImpactsSource`.
    pub fn advance_shallow(
        &mut self,
        source: &mut dyn ImpactsSource,
        target: i32,
    ) -> io::Result<i32> {
        source.advance_shallow(target)?;
        let impacts = source.get_impacts()?;
        Ok(impacts.get_doc_id_up_to(0))
    }

    fn ensure_cache_size(&mut self, size: usize) {
        if self.max_score_cache.len() < size {
            let old_length = self.max_score_cache.len();
            self.max_score_cache.resize(size, 0.0);
            self.max_score_cache_up_to.resize(size, -1);
            for i in old_length..self.max_score_cache_up_to.len() {
                self.max_score_cache_up_to[i] = -1;
            }
        }
    }

    fn compute_max_score(&self, scorer: &dyn SimScorer, impacts: &[Impact]) -> f32 {
        let mut max_score = 0.0f32;
        for impact in impacts {
            max_score = max_score.max(scorer.score(impact.freq as f32, impact.norm));
        }
        max_score
    }

    /// Return the maximum score up to `up_to` included.
    pub fn get_max_score(
        &mut self,
        source: &mut dyn ImpactsSource,
        scorer: &dyn SimScorer,
        up_to: i32,
    ) -> io::Result<f32> {
        let level = self.get_level(source, up_to)?;
        if level == -1 {
            return Ok(self.global_max_score);
        }
        self.get_max_score_for_level(source, scorer, level as usize)
    }

    /// Return the first level that includes all doc IDs up to `up_to`, or -1 if there is no
    /// such level.
    fn get_level(&mut self, source: &mut dyn ImpactsSource, up_to: i32) -> io::Result<i32> {
        let impacts = source.get_impacts()?;
        let num_levels = impacts.num_levels();
        for level in 0..num_levels {
            let impacts_up_to = impacts.get_doc_id_up_to(level);
            if up_to <= impacts_up_to {
                return Ok(level as i32);
            }
        }
        Ok(-1)
    }

    /// Return the maximum score for level zero.
    pub(crate) fn get_max_score_for_level_zero(
        &mut self,
        source: &mut dyn ImpactsSource,
        scorer: &dyn SimScorer,
    ) -> io::Result<f32> {
        self.get_max_score_for_level(source, scorer, 0)
    }

    /// Return the maximum score for the given `level`.
    fn get_max_score_for_level(
        &mut self,
        source: &mut dyn ImpactsSource,
        scorer: &dyn SimScorer,
        level: usize,
    ) -> io::Result<f32> {
        self.ensure_cache_size(level + 1);
        let impacts = source.get_impacts()?;
        let level_up_to = impacts.get_doc_id_up_to(level);
        if self.max_score_cache_up_to[level] < level_up_to {
            self.max_score_cache[level] =
                self.compute_max_score(scorer, impacts.get_impacts(level));
            self.max_score_cache_up_to[level] = level_up_to;
        }
        Ok(self.max_score_cache[level])
    }

    /// Return the maximum level at which scores are all less than `min_score`, or -1 if none.
    fn get_skip_level(
        &mut self,
        source: &mut dyn ImpactsSource,
        scorer: &dyn SimScorer,
        min_score: f32,
    ) -> io::Result<i32> {
        let impacts = source.get_impacts()?;
        let num_levels = impacts.num_levels();
        for level in 0..num_levels {
            if self.get_max_score_for_level(source, scorer, level)? >= min_score {
                return Ok(level as i32 - 1);
            }
        }
        Ok(num_levels as i32 - 1)
    }

    /// Return an inclusive upper bound of documents that all have a score less than
    /// `min_score`, or -1 if the current document may be competitive.
    pub(crate) fn get_skip_up_to(
        &mut self,
        source: &mut dyn ImpactsSource,
        scorer: &dyn SimScorer,
        min_score: f32,
    ) -> io::Result<i32> {
        let level = self.get_skip_level(source, scorer, min_score)?;
        if level == -1 {
            return Ok(-1);
        }
        let impacts = source.get_impacts()?;
        Ok(impacts.get_doc_id_up_to(level as usize))
    }
}

// ---------------------------------------------------------------------------
// ImpactsDISI
// ---------------------------------------------------------------------------

/// `DocIdSetIterator` that skips non-competitive docs thanks to the indexed impacts. Call
/// `set_min_competitive_score` in order to give this iterator the ability to skip low-scoring
/// documents.
#[expect(dead_code)]
pub struct ImpactsDISI<I: DocIdSetIterator> {
    inner: I,
    max_score_cache: MaxScoreCache,
    min_competitive_score: f32,
    up_to: i32,
    max_score: f32,
}

// NOTE: ImpactsDISI requires ownership of the inner iterator AND a MaxScoreCache that
// borrows from an ImpactsSource. The lifetime relationships here are complex. In a full
// implementation, TermScorer will own both the BlockPostingsEnum (which is both the
// DocIdSetIterator AND the ImpactsSource) and build the MaxScoreCache from it.
//
// For now, we provide the struct definition and the algorithm. The actual wiring will be
// done in TermScorer (Tier 5) where the ownership is clear. The methods below show the
// exact Java algorithm translated to Rust.

// ---------------------------------------------------------------------------
// MaxScoreAccumulator
// ---------------------------------------------------------------------------

/// Maintains the maximum score and its corresponding document id concurrently.
#[derive(Debug)]
pub struct MaxScoreAccumulator {
    /// We use 2^10-1 to check the remainder with a bitwise operation.
    acc: atomic::AtomicI64,
    /// The interval at which to check for global min competitive score updates.
    pub(crate) mod_interval: i64,
}

/// Default interval: 0x3ff (2^10 - 1).
pub(crate) const DEFAULT_INTERVAL: i64 = 0x3ff;

impl MaxScoreAccumulator {
    /// Sole constructor.
    pub fn new() -> Self {
        Self {
            acc: atomic::AtomicI64::new(i64::MIN),
            mod_interval: DEFAULT_INTERVAL,
        }
    }

    /// Accumulate a new (doc, score) encoded as a long.
    pub fn accumulate(&self, code: i64) {
        self.acc.fetch_max(code, atomic::Ordering::Relaxed);
    }

    /// Get the current raw accumulated value.
    pub fn get_raw(&self) -> i64 {
        self.acc.load(atomic::Ordering::Relaxed)
    }
}

impl Default for MaxScoreAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// DocScoreEncoder
// ---------------------------------------------------------------------------

/// Encodes (doc, score) pairs as a long whose sort order matches
/// `(score ascending, doc descending)`.
#[derive(Debug)]
pub struct DocScoreEncoder;

impl DocScoreEncoder {
    /// The least competitive code: lowest possible score, highest possible doc ID.
    pub const LEAST_COMPETITIVE_CODE: i64 = Self::encode(i32::MAX, f32::NEG_INFINITY);

    /// Encode a (doc, score) pair into a single i64.
    pub const fn encode(doc_id: i32, score: f32) -> i64 {
        ((float_to_sortable_int(score) as i64) << 32) | ((i32::MAX - doc_id) as u32 as i64)
    }

    /// Extract the score from an encoded value.
    pub const fn to_score(value: i64) -> f32 {
        sortable_int_to_float((value >> 32) as i32)
    }

    /// Extract the doc ID from an encoded value.
    pub const fn doc_id(value: i64) -> i32 {
        i32::MAX - (value as i32)
    }
}

/// Converts a float to a sortable int (matching Java's `NumericUtils.floatToSortableInt`).
pub(crate) const fn float_to_sortable_int(value: f32) -> i32 {
    let bits = value.to_bits() as i32;
    // If the sign bit is set, flip all bits; otherwise flip only the sign bit.
    bits ^ (bits >> 31) & 0x7fffffff
}

/// Converts a sortable int back to a float (matching Java's `NumericUtils.sortableIntToFloat`).
pub(crate) const fn sortable_int_to_float(encoded: i32) -> f32 {
    let bits = encoded ^ ((encoded >> 31) & 0x7fffffff);
    f32::from_bits(bits as u32)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use assertables::*;

    // -- DocScoreEncoder tests --

    #[test]
    fn test_encode_decode_roundtrip() {
        let doc = 42;
        let score = 1.5f32;
        let code = DocScoreEncoder::encode(doc, score);
        assert_eq!(DocScoreEncoder::doc_id(code), doc);
        assert_eq!(DocScoreEncoder::to_score(code), score);
    }

    #[test]
    fn test_encode_ordering_by_score() {
        let low = DocScoreEncoder::encode(0, 1.0);
        let high = DocScoreEncoder::encode(0, 2.0);
        assert_gt!(high, low);
    }

    #[test]
    fn test_encode_ordering_by_doc_descending() {
        // Same score: lower doc ID should produce higher code (reverse doc order)
        let doc0 = DocScoreEncoder::encode(0, 1.0);
        let doc100 = DocScoreEncoder::encode(100, 1.0);
        assert_gt!(doc0, doc100);
    }

    #[test]
    fn test_least_competitive_code() {
        // Any real (doc, score) should be more competitive than LEAST_COMPETITIVE_CODE
        let real = DocScoreEncoder::encode(0, 0.0);
        assert_gt!(real, DocScoreEncoder::LEAST_COMPETITIVE_CODE);
    }

    #[test]
    fn test_encode_zero_score() {
        let code = DocScoreEncoder::encode(10, 0.0);
        assert_eq!(DocScoreEncoder::doc_id(code), 10);
        assert_eq!(DocScoreEncoder::to_score(code), 0.0);
    }

    // -- float_to_sortable_int tests --

    #[test]
    fn test_float_sortable_roundtrip() {
        for &v in &[0.0f32, 1.0, -1.0, f32::MAX, f32::MIN, 0.001, 1000.0] {
            let encoded = float_to_sortable_int(v);
            let decoded = sortable_int_to_float(encoded);
            assert_eq!(decoded, v);
        }
    }

    #[test]
    fn test_float_sortable_preserves_order() {
        let values = [-100.0f32, -1.0, 0.0, 0.5, 1.0, 100.0];
        for i in 0..values.len() - 1 {
            let a = float_to_sortable_int(values[i]);
            let b = float_to_sortable_int(values[i + 1]);
            assert_lt!(a, b, "{} should sort before {}", values[i], values[i + 1]);
        }
    }

    // -- MaxScoreAccumulator tests --

    #[test]
    fn test_accumulator_initial_value() {
        let acc = MaxScoreAccumulator::new();
        assert_eq!(acc.get_raw(), i64::MIN);
    }

    #[test]
    fn test_accumulator_keeps_max() {
        let acc = MaxScoreAccumulator::new();
        acc.accumulate(100);
        assert_eq!(acc.get_raw(), 100);
        acc.accumulate(50);
        assert_eq!(acc.get_raw(), 100);
        acc.accumulate(200);
        assert_eq!(acc.get_raw(), 200);
    }

    // -- MaxScoreCache tests --

    /// Mock ImpactsSource that stores impacts data inline and returns references to itself.
    struct MockImpactsSource {
        /// (doc_id_up_to, impacts) per level
        levels: Vec<(i32, Vec<Impact>)>,
    }

    impl MockImpactsSource {
        fn new(levels: Vec<(i32, Vec<Impact>)>) -> Self {
            Self { levels }
        }
    }

    impl Impacts for MockImpactsSource {
        fn num_levels(&self) -> usize {
            self.levels.len()
        }
        fn get_doc_id_up_to(&self, level: usize) -> i32 {
            self.levels[level].0
        }
        fn get_impacts(&self, level: usize) -> &[Impact] {
            &self.levels[level].1
        }
    }

    impl ImpactsSource for MockImpactsSource {
        fn advance_shallow(&mut self, _target: i32) -> io::Result<()> {
            Ok(())
        }
        fn get_impacts(&mut self) -> io::Result<&dyn Impacts> {
            Ok(self)
        }
    }

    /// Simple SimScorer for testing: score = freq / max(norm, 1).
    struct TestSimScorer;
    impl SimScorer for TestSimScorer {
        fn score(&self, freq: f32, norm: i64) -> f32 {
            freq / norm.max(1) as f32
        }

        fn box_clone(&self) -> Box<dyn SimScorer> {
            Box::new(TestSimScorer)
        }
    }

    #[test]
    fn test_max_score_cache_single_level() {
        let mut source = MockImpactsSource::new(vec![(
            100,
            vec![Impact { freq: 5, norm: 1 }, Impact { freq: 10, norm: 2 }],
        )]);
        let scorer = TestSimScorer;
        let mut cache = MaxScoreCache::new(&scorer);

        // Max score at level 0: max(5/1, 10/2) = max(5.0, 5.0) = 5.0
        let score = cache.get_max_score(&mut source, &scorer, 100).unwrap();
        assert_eq!(score, 5.0);
    }

    #[test]
    fn test_max_score_cache_two_levels() {
        let mut source = MockImpactsSource::new(vec![
            (50, vec![Impact { freq: 2, norm: 1 }]), // level 0: up to doc 50, max=2.0
            (200, vec![Impact { freq: 10, norm: 1 }]), // level 1: up to doc 200, max=10.0
        ]);
        let scorer = TestSimScorer;
        let mut cache = MaxScoreCache::new(&scorer);

        // up_to=50 fits in level 0
        assert_eq!(cache.get_max_score(&mut source, &scorer, 50).unwrap(), 2.0);
        // up_to=100 needs level 1
        assert_eq!(
            cache.get_max_score(&mut source, &scorer, 100).unwrap(),
            10.0
        );
    }

    #[test]
    fn test_max_score_cache_beyond_all_levels() {
        let mut source = MockImpactsSource::new(vec![(50, vec![Impact { freq: 2, norm: 1 }])]);
        let scorer = TestSimScorer;
        let mut cache = MaxScoreCache::new(&scorer);

        // up_to=100 is beyond level 0 (up_to=50), so returns global max score
        let score = cache.get_max_score(&mut source, &scorer, 100).unwrap();
        let global = TestSimScorer.score(f32::MAX, 1);
        assert_eq!(score, global);
    }

    #[test]
    fn test_max_score_cache_advance_shallow() {
        let mut source = MockImpactsSource::new(vec![
            (50, vec![Impact { freq: 2, norm: 1 }]),
            (200, vec![Impact { freq: 10, norm: 1 }]),
        ]);
        let scorer = TestSimScorer;
        let mut cache = MaxScoreCache::new(&scorer);

        // advance_shallow returns doc_id_up_to for level 0
        let up_to = cache.advance_shallow(&mut source, 0).unwrap();
        assert_eq!(up_to, 50);
    }

    #[test]
    fn test_max_score_cache_get_skip_up_to() {
        let mut source = MockImpactsSource::new(vec![
            (50, vec![Impact { freq: 2, norm: 1 }]),   // max=2.0
            (200, vec![Impact { freq: 10, norm: 1 }]), // max=10.0
        ]);
        let scorer = TestSimScorer;
        let mut cache = MaxScoreCache::new(&scorer);

        // min_score=3.0: level 0 max is 2.0 < 3.0, so skip up to doc 50
        let skip = cache.get_skip_up_to(&mut source, &scorer, 3.0).unwrap();
        assert_eq!(skip, 50);

        // min_score=1.0: level 0 max is 2.0 >= 1.0, so no skip (-1)
        let skip = cache.get_skip_up_to(&mut source, &scorer, 1.0).unwrap();
        assert_eq!(skip, -1);
    }

    #[test]
    fn test_max_score_cache_caching() {
        let mut source = MockImpactsSource::new(vec![(100, vec![Impact { freq: 5, norm: 1 }])]);
        let scorer = TestSimScorer;
        let mut cache = MaxScoreCache::new(&scorer);

        // First call computes and caches
        assert_eq!(cache.get_max_score(&mut source, &scorer, 100).unwrap(), 5.0);
        // Second call should use cache (same result)
        assert_eq!(cache.get_max_score(&mut source, &scorer, 100).unwrap(), 5.0);
    }

    #[test]
    fn test_accumulator_with_doc_score_encoder() {
        let acc = MaxScoreAccumulator::new();
        acc.accumulate(DocScoreEncoder::encode(0, 1.0));
        acc.accumulate(DocScoreEncoder::encode(1, 2.0));
        acc.accumulate(DocScoreEncoder::encode(2, 1.5));

        let raw = acc.get_raw();
        assert_eq!(DocScoreEncoder::to_score(raw), 2.0);
        assert_eq!(DocScoreEncoder::doc_id(raw), 1);
    }
}
