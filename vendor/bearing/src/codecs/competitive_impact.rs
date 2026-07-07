// SPDX-License-Identifier: Apache-2.0
//! Competitive impact accumulation for block skip data.
//!
//! Collects per-document (freq, norm) pairs and reduces them to a Pareto-optimal
//! set for encoding in `.doc` skip blocks. This allows readers to skip blocks that
//! cannot contribute to top-K results during scoring.

use std::cmp::Ordering;
use std::collections::BTreeSet;
use std::fmt;

/// Per-document scoring factors: term frequency and norm factor.
///
/// Ordered by ascending frequency, then descending unsigned norm (matching
/// the Java `CompetitiveImpactAccumulator`'s TreeSet comparator).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Impact {
    /// Term frequency of the term in the document.
    pub freq: i32,
    /// Norm factor of the document.
    pub norm: i64,
}

impl Ord for Impact {
    fn cmp(&self, other: &Self) -> Ordering {
        self.freq
            .cmp(&other.freq)
            .then_with(|| (other.norm as u64).cmp(&(self.norm as u64)))
    }
}

impl PartialOrd for Impact {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Accumulates (freq, norm) pairs and reduces them to a Pareto-optimal set.
///
/// Uses a fast path for norm values in the byte range (-128..=127), which covers
/// all norms produced by BM25 (the default similarity). Norms outside this range
/// fall back to a `BTreeSet`.
pub struct CompetitiveImpactAccumulator {
    /// Maps norm values in -128..127 (as unsigned byte index 0..255) to the
    /// maximum frequency observed for that norm.
    max_freqs: [i32; 256],
    /// Competitive (freq, norm) pairs for norm values outside the byte range.
    /// Always empty with the default BM25 similarity.
    other_freq_norm_pairs: BTreeSet<Impact>,
}

impl Default for CompetitiveImpactAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl CompetitiveImpactAccumulator {
    /// Creates a new empty accumulator.
    pub fn new() -> Self {
        Self {
            max_freqs: [0; 256],
            other_freq_norm_pairs: BTreeSet::new(),
        }
    }

    /// Resets to the initial empty state.
    pub fn clear(&mut self) {
        self.max_freqs.fill(0);
        self.other_freq_norm_pairs.clear();
    }

    /// Accumulates a (freq, norm) pair.
    pub fn add(&mut self, freq: i32, norm: i64) {
        if norm >= i8::MIN as i64 && norm <= i8::MAX as i64 {
            let index = norm as i8 as u8 as usize;
            self.max_freqs[index] = self.max_freqs[index].max(freq);
        } else {
            let entry = Impact { freq, norm };
            add_to_set(entry, &mut self.other_freq_norm_pairs);
        }
    }

    /// Merges all entries from `other` into this accumulator.
    pub fn add_all(&mut self, other: &Self) {
        for i in 0..self.max_freqs.len() {
            self.max_freqs[i] = self.max_freqs[i].max(other.max_freqs[i]);
        }
        for &entry in &other.other_freq_norm_pairs {
            add_to_set(entry, &mut self.other_freq_norm_pairs);
        }
    }

    /// Returns the Pareto-optimal set of competitive (freq, norm) pairs,
    /// ordered by ascending frequency and ascending unsigned norm.
    pub fn get_competitive_freq_norm_pairs(&self) -> Vec<Impact> {
        let mut impacts = Vec::new();
        let mut max_freq_for_lower_norms = 0;

        // Iterate byte-range norms in unsigned order (0..255).
        // Index i maps to norm value: i as u8 as i8 as i64.
        for i in 0..self.max_freqs.len() {
            let max_freq = self.max_freqs[i];
            if max_freq > max_freq_for_lower_norms {
                let norm = i as u8 as i8 as i64;
                impacts.push(Impact {
                    freq: max_freq,
                    norm,
                });
                max_freq_for_lower_norms = max_freq;
            }
        }

        if self.other_freq_norm_pairs.is_empty() {
            // Common case: all norms are bytes
            return impacts;
        }

        // Merge byte-range impacts into a copy of the out-of-range set
        let mut freq_norm_pairs = self.other_freq_norm_pairs.clone();
        for impact in impacts {
            add_to_set(impact, &mut freq_norm_pairs);
        }
        freq_norm_pairs.into_iter().collect()
    }
}

/// Default norm value for fields without norms (omit_norms).
///
/// Norms are a multiplicative scoring factor, so 1 means no boost.
pub const NO_BOOST: i64 = 1;

/// Provides norm lookups by doc ID for a single field.
///
/// Implementations resolve doc IDs to norm values for competitive impact
/// computation. Returns [`NO_BOOST`] for documents without norms.
pub trait NormsLookup: fmt::Debug {
    /// Returns the norm for the given doc ID, or [`NO_BOOST`] if not found.
    fn get(&self, doc_id: i32) -> i64;
}

/// In-memory norms lookup backed by sorted doc ID and value slices.
///
/// Binary-searches the sorted `norms_docs` array to find the norm value.
/// For fields without norms (omit_norms), construct with [`no_norms`](Self::no_norms)
/// which uses empty slices and always returns 1.
#[derive(Debug)]
pub struct BufferedNormsLookup<'a> {
    norms: &'a [i64],
    norms_docs: &'a [i32],
}

impl<'a> BufferedNormsLookup<'a> {
    /// Creates a lookup for a field that has norms.
    pub fn new(norms: &'a [i64], norms_docs: &'a [i32]) -> Self {
        Self { norms, norms_docs }
    }

    /// Creates a lookup that always returns [`NO_BOOST`] (for fields with omit_norms).
    pub fn no_norms() -> Self {
        Self {
            norms: &[],
            norms_docs: &[],
        }
    }
}

impl NormsLookup for BufferedNormsLookup<'_> {
    fn get(&self, doc_id: i32) -> i64 {
        match self.norms_docs.binary_search(&doc_id) {
            Ok(idx) => self.norms[idx],
            Err(_) => NO_BOOST,
        }
    }
}

/// Inserts `new_entry` into the Pareto-optimal set, pruning dominated entries.
fn add_to_set(new_entry: Impact, set: &mut BTreeSet<Impact>) {
    // ceiling: smallest element >= new_entry in the ordering
    let ceiling = set.range(new_entry..).next().copied();

    if let Some(next) = ceiling
        && (next.norm as u64) <= (new_entry.norm as u64)
    {
        // Already have this entry or a more competitive one
        return;
    }

    set.insert(new_entry);

    // Prune entries with lower freq but equal or worse (higher unsigned) norm
    let to_remove: Vec<Impact> = set
        .range(..new_entry)
        .rev()
        .copied()
        .take_while(|e| (e.norm as u64) >= (new_entry.norm as u64))
        .collect();
    for e in to_remove {
        set.remove(&e);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- Impact ordering tests --

    #[test]
    fn test_impact_ord_ascending_freq() {
        let a = Impact { freq: 3, norm: 5 };
        let b = Impact { freq: 7, norm: 5 };
        assert_lt!(a, b);
    }

    #[test]
    fn test_impact_ord_descending_unsigned_norm_for_equal_freq() {
        // Higher unsigned norm compares lower (descending unsigned norm)
        let a = Impact { freq: 5, norm: 10 };
        let b = Impact { freq: 5, norm: 5 };
        assert_lt!(a, b); // norm 10 (unsigned larger) comes first
    }

    // -- CompetitiveImpactAccumulator tests --
    // Ported from TestCompetitiveFreqNormAccumulator

    fn impacts_to_pairs(impacts: &[Impact]) -> Vec<(i32, i64)> {
        impacts.iter().map(|i| (i.freq, i.norm)).collect()
    }

    #[test]
    fn test_basics() {
        let mut acc = CompetitiveImpactAccumulator::new();

        acc.add(3, 5);
        assert_eq!(
            impacts_to_pairs(&acc.get_competitive_freq_norm_pairs()),
            [(3, 5)]
        );

        acc.add(6, 11);
        assert_eq!(
            impacts_to_pairs(&acc.get_competitive_freq_norm_pairs()),
            [(3, 5), (6, 11)]
        );

        acc.add(10, 13);
        assert_eq!(
            impacts_to_pairs(&acc.get_competitive_freq_norm_pairs()),
            [(3, 5), (6, 11), (10, 13)]
        );

        acc.add(1, 2);
        assert_eq!(
            impacts_to_pairs(&acc.get_competitive_freq_norm_pairs()),
            [(1, 2), (3, 5), (6, 11), (10, 13)]
        );

        // (7, 9) dominates (6, 11): freq 7 > 6, norm 9 < 11 (lower unsigned = more competitive)
        acc.add(7, 9);
        assert_eq!(
            impacts_to_pairs(&acc.get_competitive_freq_norm_pairs()),
            [(1, 2), (3, 5), (7, 9), (10, 13)]
        );

        // (8, 2) dominates (1, 2), (3, 5), (7, 9): all have norm >= 2
        acc.add(8, 2);
        assert_eq!(
            impacts_to_pairs(&acc.get_competitive_freq_norm_pairs()),
            [(8, 2), (10, 13)]
        );
    }

    #[test]
    fn test_extreme_norms() {
        let mut acc = CompetitiveImpactAccumulator::new();

        acc.add(3, 5);
        assert_eq!(
            impacts_to_pairs(&acc.get_competitive_freq_norm_pairs()),
            [(3, 5)]
        );

        // norm 10000 is outside byte range, goes to BTreeSet
        acc.add(10, 10000);
        assert_eq!(
            impacts_to_pairs(&acc.get_competitive_freq_norm_pairs()),
            [(3, 5), (10, 10000)]
        );

        // norm 200 is outside byte range (> 127)
        acc.add(5, 200);
        assert_eq!(
            impacts_to_pairs(&acc.get_competitive_freq_norm_pairs()),
            [(3, 5), (5, 200), (10, 10000)]
        );

        // norm -100 is in byte range, unsigned value is very large (0xFFFFFFFFFFFFFF9C)
        acc.add(20, -100);
        assert_eq!(
            impacts_to_pairs(&acc.get_competitive_freq_norm_pairs()),
            [(3, 5), (5, 200), (10, 10000), (20, -100)]
        );

        // norm -3 is in byte range, unsigned value even larger
        acc.add(30, -3);
        assert_eq!(
            impacts_to_pairs(&acc.get_competitive_freq_norm_pairs()),
            [(3, 5), (5, 200), (10, 10000), (20, -100), (30, -3)]
        );
    }

    #[test]
    fn test_add_all() {
        let mut acc = CompetitiveImpactAccumulator::new();
        acc.add(3, 5);

        let mut merged = CompetitiveImpactAccumulator::new();
        merged.add_all(&acc);
        assert_eq!(
            acc.get_competitive_freq_norm_pairs(),
            merged.get_competitive_freq_norm_pairs()
        );

        acc.add(10, 10000);
        merged.clear();
        merged.add_all(&acc);
        assert_eq!(
            acc.get_competitive_freq_norm_pairs(),
            merged.get_competitive_freq_norm_pairs()
        );

        acc.add(5, 200);
        merged.clear();
        merged.add_all(&acc);
        assert_eq!(
            acc.get_competitive_freq_norm_pairs(),
            merged.get_competitive_freq_norm_pairs()
        );

        acc.add(20, -100);
        merged.clear();
        merged.add_all(&acc);
        assert_eq!(
            acc.get_competitive_freq_norm_pairs(),
            merged.get_competitive_freq_norm_pairs()
        );

        acc.add(30, -3);
        merged.clear();
        merged.add_all(&acc);
        assert_eq!(
            acc.get_competitive_freq_norm_pairs(),
            merged.get_competitive_freq_norm_pairs()
        );
    }

    #[test]
    fn test_omit_freqs() {
        // All same freq: only the lowest unsigned norm survives
        let mut acc = CompetitiveImpactAccumulator::new();
        acc.add(1, 5);
        acc.add(1, 7);
        acc.add(1, 4);
        assert_eq!(
            impacts_to_pairs(&acc.get_competitive_freq_norm_pairs()),
            [(1, 4)]
        );
    }

    #[test]
    fn test_omit_norms() {
        // All same norm: only the highest freq survives
        let mut acc = CompetitiveImpactAccumulator::new();
        acc.add(5, 1);
        acc.add(7, 1);
        acc.add(4, 1);
        assert_eq!(
            impacts_to_pairs(&acc.get_competitive_freq_norm_pairs()),
            [(7, 1)]
        );
    }

    #[test]
    fn test_single_impact() {
        let mut acc = CompetitiveImpactAccumulator::new();
        acc.add(42, 10);
        assert_eq!(
            impacts_to_pairs(&acc.get_competitive_freq_norm_pairs()),
            [(42, 10)]
        );
    }

    #[test]
    fn test_clear() {
        let mut acc = CompetitiveImpactAccumulator::new();
        acc.add(5, 10);
        acc.add(10, 20);
        acc.clear();
        assert_is_empty!(acc.get_competitive_freq_norm_pairs());
    }

    #[test]
    fn test_byte_range_boundaries() {
        // Verify norms at byte range boundaries use the correct path
        let mut acc = CompetitiveImpactAccumulator::new();
        acc.add(1, -128); // i8::MIN — byte path, index 128
        acc.add(2, 127); // i8::MAX — byte path, index 127
        acc.add(3, 128); // outside byte range — BTreeSet

        let pairs = impacts_to_pairs(&acc.get_competitive_freq_norm_pairs());
        // (1, -128) is dominated by (2, 127): freq 2 > 1, unsigned(127) < unsigned(-128)
        // Unsigned order: 127 < 128, so both (2, 127) and (3, 128) survive
        assert_eq!(pairs, [(2, 127), (3, 128)]);
    }

    #[test]
    fn test_same_norm_max_freq_wins() {
        let mut acc = CompetitiveImpactAccumulator::new();
        acc.add(3, 10);
        acc.add(7, 10);
        acc.add(5, 10);
        assert_eq!(
            impacts_to_pairs(&acc.get_competitive_freq_norm_pairs()),
            [(7, 10)]
        );
    }
}
