// SPDX-License-Identifier: Apache-2.0

//! WAND (Weak AND) scorer with dynamic pruning.
//!
//! Implements the WAND algorithm described in "Efficient Query Evaluation using a Two-Level
//! Retrieval Process" by Broder et al., enhanced with block-max techniques from
//! "Faster Top-k Document Retrieval Using Block-Max Indexes" by Ding and Suel.
//!
//! Used for disjunctions when:
//! - `min_should_match > 1`, regardless of score mode, OR
//! - `score_mode == TopScores` and the disjunction is the top-level scoring clause.
//!
//! Sub-scorers live in three places:
//! - **lead**: scorers positioned on the current candidate doc.
//! - **head**: scorers beyond the current candidate, ordered by doc ID (advance quickly).
//! - **tail**: scorers behind the current candidate, ordered by max score (least costly first).
//!
//! Deviations from Lucene:
//! - No `TwoPhaseIterator`. The approximation/match split is inlined into `next_doc`/`advance`.
//! - DisiWrapper has no `next` field; the lead linked list is represented by a `Vec<u32>` of
//!   wrapper indices.
//! - The tail max-heap is a private array-based heap of indices, since stdlib `BinaryHeap`
//!   doesn't support mutation of stored entries' ordering keys.

use std::fmt;
use std::io;
use std::mem;

use crate::search::collector::ScoreMode;
use crate::search::disi_priority_queue::DisiPriorityQueue;
use crate::search::disi_wrapper::DisiWrapper;
use crate::search::doc_id_set_iterator::{DocIdSetIterator, NO_MORE_DOCS};
use crate::search::scorable::Scorable;
use crate::search::scorer::Scorer;
use crate::search::scorer_util;
use crate::util::math_util;

/// Number of mantissa bits in a single-precision float; used for scaling factor selection.
pub const FLOAT_MANTISSA_BITS: i32 = 24;
const MAX_SCALED_SCORE: i64 = (1_i64 << 24) - 1;

/// Returns the scaling factor for `f` such that `f * 2^scaling_factor` is in `[2^23, 2^24)`.
///
/// Special cases:
/// - `scaling_factor(0) = scaling_factor(MIN_VALUE) + 1`
/// - `scaling_factor(+Infinity) = scaling_factor(MAX_VALUE) - 1`
pub fn scaling_factor(f: f32) -> i32 {
    if f < 0.0 {
        panic!("Scores must be positive or null, got {f}");
    } else if f == 0.0 {
        scaling_factor(f32::MIN_POSITIVE) + 1
    } else if f.is_infinite() {
        scaling_factor(f32::MAX) - 1
    } else {
        // Extract the unbiased IEEE 754 exponent (stable Rust has no frexp).
        let bits = (f as f64).to_bits();
        let raw_exp = ((bits >> 52) & 0x7ff) as i32;
        let exponent = raw_exp - 1023;
        FLOAT_MANTISSA_BITS - 1 - exponent
    }
}

/// Scale `max_score` into an unsigned integer, rounding up so we never miss a match.
pub fn scale_max_score(max_score: f32, scaling_factor: i32) -> i64 {
    debug_assert!(!max_score.is_nan());
    debug_assert!(max_score >= 0.0);
    let scaled = scalb(max_score as f64, scaling_factor);
    if scaled > MAX_SCALED_SCORE as f64 {
        return MAX_SCALED_SCORE;
    }
    scaled.ceil() as i64
}

/// Scale `min_score` into an unsigned integer, rounding down so we never miss a match.
fn scale_min_score(min_score: f32, scaling_factor: i32) -> i64 {
    debug_assert!(min_score.is_finite());
    debug_assert!(min_score >= 0.0);
    let scaled = scalb(min_score as f64, scaling_factor);
    scaled.floor() as i64
}

/// `Math.scalb(d, n) = d * 2^n` with proper handling of subnormals/overflow.
/// Rust's `f64::powi` is sufficient here because the scaling factor never causes the
/// intermediate value to exceed f64's range in our usage.
fn scalb(d: f64, n: i32) -> f64 {
    d * 2.0f64.powi(n)
}

// ---------------------------------------------------------------------------
// Tail max-heap (private)
// ---------------------------------------------------------------------------

/// Max-heap of `DisiWrapper` indices, ordered by `scaled_max_score` (descending),
/// tiebreaker by `cost` (ascending — least costly first, so they advance further).
struct TailHeap {
    heap: Box<[u32]>,
    size: usize,
}

impl TailHeap {
    fn new(capacity: usize) -> Self {
        Self {
            heap: vec![0u32; capacity].into_boxed_slice(),
            size: 0,
        }
    }

    fn size(&self) -> usize {
        self.size
    }

    fn top(&self) -> u32 {
        debug_assert!(self.size > 0);
        self.heap[0]
    }

    /// Returns the entry's index in the underlying slice for direct access.
    fn at(&self, i: usize) -> u32 {
        debug_assert!(i < self.size);
        self.heap[i]
    }

    /// Add an entry. Caller is responsible for capacity (panics if full).
    fn add(&mut self, idx: u32, wrappers: &[DisiWrapper<'_>]) {
        assert!(self.size < self.heap.len(), "TailHeap over capacity");
        self.heap[self.size] = idx;
        self.up_heap(self.size, wrappers);
        self.size += 1;
    }

    /// Remove and return the top.
    fn pop(&mut self, wrappers: &[DisiWrapper<'_>]) -> u32 {
        debug_assert!(self.size > 0);
        let result = self.heap[0];
        self.size -= 1;
        if self.size > 0 {
            self.heap[0] = self.heap[self.size];
            self.down_heap(wrappers);
        }
        result
    }

    /// Re-sift the entry at index `i` upward after its `scaled_max_score` may have increased.
    fn up_heap(&mut self, mut i: usize, wrappers: &[DisiWrapper<'_>]) {
        let node = self.heap[i];
        let node_w = &wrappers[node as usize];
        while i > 0 {
            let parent = (i - 1) / 2;
            let parent_w = &wrappers[self.heap[parent] as usize];
            if greater_max_score(node_w, parent_w) {
                self.heap[i] = self.heap[parent];
                i = parent;
            } else {
                break;
            }
        }
        self.heap[i] = node;
    }

    /// Re-sift index 0 downward.
    fn down_heap(&mut self, wrappers: &[DisiWrapper<'_>]) {
        let mut i = 0;
        let node = self.heap[0];
        let node_w = &wrappers[node as usize];
        loop {
            let left = 2 * i + 1;
            if left >= self.size {
                break;
            }
            let right = left + 1;
            let mut child = left;
            if right < self.size {
                let lw = &wrappers[self.heap[left] as usize];
                let rw = &wrappers[self.heap[right] as usize];
                if greater_max_score(rw, lw) {
                    child = right;
                }
            }
            let child_w = &wrappers[self.heap[child] as usize];
            if greater_max_score(child_w, node_w) {
                self.heap[i] = self.heap[child];
                i = child;
            } else {
                break;
            }
        }
        self.heap[i] = node;
    }
}

/// Tail ordering: greater `scaled_max_score` wins; tie broken by lower `cost`.
fn greater_max_score(w1: &DisiWrapper<'_>, w2: &DisiWrapper<'_>) -> bool {
    if w1.scaled_max_score > w2.scaled_max_score {
        true
    } else if w1.scaled_max_score < w2.scaled_max_score {
        false
    } else {
        w1.cost < w2.cost
    }
}

// ---------------------------------------------------------------------------
// WANDScorer
// ---------------------------------------------------------------------------

/// WAND (Weak AND) scorer for disjunctions with `min_should_match > 1` or top-level TOP_SCORES.
pub struct WANDScorer<'a> {
    /// All sub-scorer wrappers, owned. Indices into this slice are used by `head`, `tail`, `lead`.
    wrappers: Vec<DisiWrapper<'a>>,
    /// Indices of scorers positioned on `doc` (the lead set).
    lead: Vec<u32>,
    /// Indices of scorers beyond the current `doc`, ordered by doc ID.
    head: DisiPriorityQueue,
    /// Indices of scorers behind the current `doc`, ordered by `scaled_max_score`.
    tail: TailHeap,
    /// Sum of `scaled_max_score` for entries currently in `tail`.
    tail_max_score: i64,
    /// Current candidate doc.
    doc: i32,
    /// Sum of lead scorers' scores at `doc` (only maintained for TOP_SCORES).
    lead_score: f64,
    /// Number of scorers currently in `lead`.
    freq: i32,
    /// Doc upper bound for which max scores are valid.
    up_to: i32,
    /// Scaled minimum competitive score (TOP_SCORES feedback).
    min_competitive_score: i64,
    /// Scaling factor for max scores (only meaningful for TOP_SCORES).
    scaling_factor: i32,
    min_should_match: i32,
    score_mode: ScoreMode,
    lead_cost: i64,
    cost: i64,
}

impl fmt::Debug for WANDScorer<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WANDScorer")
            .field("doc", &self.doc)
            .field("min_should_match", &self.min_should_match)
            .field("freq", &self.freq)
            .field("num_clauses", &self.wrappers.len())
            .finish()
    }
}

impl<'a> WANDScorer<'a> {
    /// Construct a `WANDScorer`. `min_should_match` must be < `scorers.len()`.
    pub fn new(
        scorers: Vec<Box<dyn Scorer + 'a>>,
        min_should_match: i32,
        score_mode: ScoreMode,
        lead_cost: i64,
    ) -> io::Result<Self> {
        if min_should_match >= scorers.len() as i32 {
            return Err(io::Error::other(
                "minShouldMatch should be < the number of scorers",
            ));
        }
        if min_should_match < 0 {
            return Err(io::Error::other("minShouldMatch should not be negative"));
        }

        let num = scorers.len();

        // Build wrappers (with cost cached); compute scaling factor for TOP_SCORES.
        let mut wrappers: Vec<DisiWrapper<'a>> =
            scorers.into_iter().map(DisiWrapper::new).collect();

        let scaling = if score_mode == ScoreMode::TopScores {
            let mut max_score_sum = 0.0_f64;
            for w in wrappers.iter_mut() {
                w.scorer.advance_shallow(0)?;
                max_score_sum += w.scorer.get_max_score(NO_MORE_DOCS)? as f64;
            }
            let max_score_sum_f = math_util::sum_upper_bound(max_score_sum, num as i32) as f32;
            scaling_factor(max_score_sum_f)
        } else {
            0
        };

        let costs: Vec<i64> = wrappers.iter().map(|w| w.cost).collect();
        let cost = scorer_util::cost_with_min_should_match(&costs, num, min_should_match);

        // All wrappers start in `lead` (unpositioned, doc = -1). They'll be pushed back on first
        // advance.
        let lead: Vec<u32> = (0..num as u32).collect();
        let freq = num as i32;

        Ok(Self {
            wrappers,
            lead,
            head: DisiPriorityQueue::of_max_size(num),
            tail: TailHeap::new(num),
            tail_max_score: 0,
            doc: -1,
            lead_score: 0.0,
            freq,
            up_to: -1,
            min_competitive_score: 0,
            scaling_factor: scaling,
            min_should_match,
            score_mode,
            lead_cost,
            cost,
        })
    }

    // -----------------------------------------------------------------------
    // Internal moves
    // -----------------------------------------------------------------------

    /// Insert into `tail`; if capacity is reached, evict the least-contributing entry per
    /// `greater_max_score`. Returns `Some(idx)` of the evicted entry, or `None` if the new
    /// entry was simply added.
    fn insert_tail_with_overflow(&mut self, s: u32) -> Option<u32> {
        let s_w = &self.wrappers[s as usize];
        let s_scaled = s_w.scaled_max_score;
        // Free room: keep the new entry whenever the running tail max-score sum still cannot
        // make a competitive hit, or when we still need more entries to satisfy minShouldMatch.
        if self.tail_max_score + s_scaled < self.min_competitive_score
            || (self.tail.size() as i32) + 1 < self.min_should_match
        {
            self.tail.add(s, &self.wrappers);
            self.tail_max_score += s_scaled;
            None
        } else if self.tail.size() == 0 {
            Some(s)
        } else {
            let top_idx = self.tail.top();
            let top_w = &self.wrappers[top_idx as usize];
            if !greater_max_score(top_w, s_w) {
                Some(s)
            } else {
                // Swap top and s.
                let evicted = top_idx;
                let evicted_scaled = top_w.scaled_max_score;
                self.tail.heap[0] = s;
                self.tail.down_heap(&self.wrappers);
                self.tail_max_score = self.tail_max_score - evicted_scaled + s_scaled;
                Some(evicted)
            }
        }
    }

    /// Move all entries currently in `lead` back into `tail` (or `head` if evicted from tail),
    /// targeting `target`.
    fn push_back_leads(&mut self, target: i32) -> io::Result<()> {
        let lead = mem::take(&mut self.lead);
        for idx in lead {
            if let Some(evicted) = self.insert_tail_with_overflow(idx) {
                let new_doc = self.wrappers[evicted as usize]
                    .scorer
                    .iterator()
                    .advance(target)?;
                self.wrappers[evicted as usize].doc = new_doc;
                self.head.add(evicted, new_doc);
            }
        }
        self.freq = 0;
        self.lead_score = 0.0;
        Ok(())
    }

    /// Make sure all entries in `head` are on or after `target`. Returns the new top idx (or None).
    fn advance_head(&mut self, target: i32) -> io::Result<Option<u32>> {
        loop {
            let top_idx = match self.head.top() {
                Some(t) => t,
                None => return Ok(None),
            };
            if self.wrappers[top_idx as usize].doc >= target {
                return Ok(Some(top_idx));
            }
            // top is behind target; move it through tail/head.
            if let Some(evicted) = self.insert_tail_with_overflow(top_idx) {
                let new_doc = self.wrappers[evicted as usize]
                    .scorer
                    .iterator()
                    .advance(target)?;
                self.wrappers[evicted as usize].doc = new_doc;
                self.head.update_top_with(evicted, new_doc);
            } else {
                self.head.pop();
            }
        }
    }

    /// Pop the highest-scoring tail entry, advance it to `doc`; if it lands on `doc`, add to lead;
    /// otherwise push to head.
    fn advance_tail(&mut self) -> io::Result<()> {
        let top = self.tail.pop(&self.wrappers);
        let top_scaled = self.wrappers[top as usize].scaled_max_score;
        self.tail_max_score -= top_scaled;
        self.advance_one(top)
    }

    fn advance_one(&mut self, idx: u32) -> io::Result<()> {
        let new_doc = self.wrappers[idx as usize]
            .scorer
            .iterator()
            .advance(self.doc)?;
        self.wrappers[idx as usize].doc = new_doc;
        if new_doc == self.doc {
            self.add_lead(idx)?;
        } else {
            self.head.add(idx, new_doc);
        }
        Ok(())
    }

    /// Append `idx` to the lead set, updating `freq` and (for TOP_SCORES) `lead_score`.
    fn add_lead(&mut self, idx: u32) -> io::Result<()> {
        self.lead.push(idx);
        self.freq += 1;
        if self.score_mode == ScoreMode::TopScores {
            self.lead_score += self.wrappers[idx as usize].scorer.score()? as f64;
        }
        Ok(())
    }

    /// Pop all `head` entries equal to `doc` into `lead`. Sets initial `freq` and `lead_score`.
    fn move_to_next_candidate(&mut self) -> io::Result<()> {
        debug_assert!(self.lead.is_empty());
        let first_idx = self
            .head
            .pop()
            .expect("move_to_next_candidate called with empty head");
        debug_assert_eq!(self.doc, self.wrappers[first_idx as usize].doc);
        self.lead.push(first_idx);
        self.freq = 1;
        self.lead_score = if self.score_mode == ScoreMode::TopScores {
            self.wrappers[first_idx as usize].scorer.score()? as f64
        } else {
            0.0
        };
        while let Some(top_doc) = self.head.top_doc() {
            if top_doc != self.doc {
                break;
            }
            let next = self.head.pop().unwrap();
            self.add_lead(next)?;
        }
        Ok(())
    }

    /// Advance every tail entry to `doc`, redistributing them to lead/head.
    fn advance_all_tail(&mut self) -> io::Result<()> {
        while self.tail.size() > 0 {
            let last = self.tail.size() - 1;
            let idx = self.tail.at(last);
            self.tail.size -= 1;
            self.advance_one(idx)?;
        }
        self.tail_max_score = 0;
        Ok(())
    }

    /// Refresh `up_to` and per-clause `scaled_max_score` so that `up_to >= target` and the
    /// next candidate (top of head) is included in the current block.
    fn move_to_next_block(&mut self, mut target: i32) -> io::Result<()> {
        debug_assert!(self.lead.is_empty());
        while self.up_to < NO_MORE_DOCS {
            if self.head.is_empty() {
                target = target.max(self.up_to + 1);
                self.update_max_scores(target)?;
            } else {
                let top_doc = self.head.top_doc().unwrap();
                if top_doc > self.up_to {
                    self.update_max_scores(target)?;
                    break;
                } else {
                    break;
                }
            }
        }
        Ok(())
    }

    fn update_max_scores(&mut self, target: i32) -> io::Result<()> {
        let mut new_up_to = NO_MORE_DOCS;

        // Look at head entries that are within the next block boundary and have cost <= lead_cost.
        // Collect indices first to avoid borrow conflicts.
        let head_entries: Vec<u32> = self.head.iter().map(|e| e.idx).collect();
        for idx in &head_entries {
            let w = &self.wrappers[*idx as usize];
            if w.doc <= new_up_to && w.cost <= self.lead_cost {
                let advance_target = w.doc;
                let s = self.wrappers[*idx as usize]
                    .scorer
                    .advance_shallow(advance_target)?;
                new_up_to = new_up_to.min(s);
            }
        }

        // If nothing in head qualified, fall back to tail[0] if cheap enough.
        if new_up_to == NO_MORE_DOCS && self.tail.size() > 0 {
            let tail_top = self.tail.at(0);
            if self.wrappers[tail_top as usize].cost <= self.lead_cost {
                new_up_to = self.wrappers[tail_top as usize]
                    .scorer
                    .advance_shallow(target)?;
                if let Some(head_top_doc) = self.head.top_doc() {
                    new_up_to = new_up_to.max(head_top_doc);
                }
            }
        }
        self.up_to = new_up_to;

        // Update scaled_max_score for head entries within up_to.
        let scaling = self.scaling_factor;
        for &idx in &head_entries {
            let w = &mut self.wrappers[idx as usize];
            if w.doc <= self.up_to {
                let m = w.scorer.get_max_score(new_up_to)?;
                w.scaled_max_score = scale_max_score(m, scaling);
            }
        }

        // Update tail entries' max scores and re-heapify, recompute tail_max_score.
        self.tail_max_score = 0;
        for i in 0..self.tail.size() {
            let idx = self.tail.heap[i];
            self.wrappers[idx as usize].scorer.advance_shallow(target)?;
            let m = self.wrappers[idx as usize]
                .scorer
                .get_max_score(self.up_to)?;
            self.wrappers[idx as usize].scaled_max_score = scale_max_score(m, scaling);
            self.tail.up_heap(i, &self.wrappers);
            self.tail_max_score += self.wrappers[idx as usize].scaled_max_score;
        }

        // If tail alone could be competitive, evict its top into head until that's no longer true.
        while self.tail.size() > 0 && self.tail_max_score >= self.min_competitive_score {
            let idx = self.tail.pop(&self.wrappers);
            let evicted_scaled = self.wrappers[idx as usize].scaled_max_score;
            self.tail_max_score -= evicted_scaled;
            let new_doc = self.wrappers[idx as usize]
                .scorer
                .iterator()
                .advance(target)?;
            self.wrappers[idx as usize].doc = new_doc;
            self.head.add(idx, new_doc);
        }

        Ok(())
    }

    /// Approximation: drive `doc` to the next candidate at or after `target`.
    fn advance_approx(&mut self, target: i32) -> io::Result<i32> {
        self.push_back_leads(target)?;
        let mut head_top = self.advance_head(target)?;
        if self.score_mode == ScoreMode::TopScores
            && (head_top.is_none() || self.wrappers[head_top.unwrap() as usize].doc > self.up_to)
        {
            self.move_to_next_block(target)?;
            head_top = self.head.top();
        }
        self.doc = match head_top {
            None => NO_MORE_DOCS,
            Some(idx) => self.wrappers[idx as usize].doc,
        };
        Ok(self.doc)
    }

    /// Match check: confirm the current `doc` satisfies minShouldMatch / minCompetitiveScore,
    /// promoting tail entries until it does or it's clear the doc cannot match.
    fn matches_current(&mut self) -> io::Result<bool> {
        debug_assert!(self.lead.is_empty());
        self.move_to_next_candidate()?;

        let mut scaled_lead_score = if self.score_mode == ScoreMode::TopScores {
            scale_max_score(
                math_util::sum_upper_bound(self.lead_score, FLOAT_MANTISSA_BITS) as f32,
                self.scaling_factor,
            )
        } else {
            0
        };

        while scaled_lead_score < self.min_competitive_score || self.freq < self.min_should_match {
            if scaled_lead_score + self.tail_max_score < self.min_competitive_score
                || self.freq + (self.tail.size() as i32) < self.min_should_match
            {
                return Ok(false);
            }
            let prev_lead_len = self.lead.len();
            self.advance_tail()?;
            if self.score_mode == ScoreMode::TopScores && self.lead.len() != prev_lead_len {
                scaled_lead_score = scale_max_score(
                    math_util::sum_upper_bound(self.lead_score, FLOAT_MANTISSA_BITS) as f32,
                    self.scaling_factor,
                );
            }
        }
        Ok(true)
    }
}

impl Scorable for WANDScorer<'_> {
    fn score(&mut self) -> io::Result<f32> {
        // We need to know about all matches at the current doc.
        self.advance_all_tail()?;
        let mut total = self.lead_score;
        if self.score_mode != ScoreMode::TopScores {
            // For TOP_SCORES we tracked lead_score on the fly; otherwise sum now.
            for &idx in &self.lead {
                total += self.wrappers[idx as usize].scorer.score()? as f64;
            }
        }
        Ok(total as f32)
    }

    fn set_min_competitive_score(&mut self, min_score: f32) -> io::Result<()> {
        debug_assert_eq!(self.score_mode, ScoreMode::TopScores);
        debug_assert!(min_score >= 0.0);
        let scaled = scale_min_score(min_score, self.scaling_factor);
        debug_assert!(scaled >= self.min_competitive_score);
        self.min_competitive_score = scaled;
        Ok(())
    }
}

impl<'a> Scorer for WANDScorer<'a> {
    fn doc_id(&self) -> i32 {
        self.doc
    }

    fn iterator(&mut self) -> &mut dyn DocIdSetIterator {
        self
    }

    fn advance_shallow(&mut self, target: i32) -> io::Result<i32> {
        // Propagate to improve score bounds on every clause.
        for w in self.wrappers.iter_mut() {
            if w.scorer.doc_id() < target {
                w.scorer.advance_shallow(target)?;
            }
        }
        if target <= self.up_to {
            return Ok(self.up_to);
        }
        Ok(NO_MORE_DOCS)
    }

    fn get_max_score(&mut self, up_to: i32) -> io::Result<f32> {
        let mut sum = 0.0_f64;
        for w in self.wrappers.iter_mut() {
            if w.scorer.doc_id() <= up_to {
                sum += w.scorer.get_max_score(up_to)? as f64;
            }
        }
        let n = self.wrappers.len() as i32;
        Ok(math_util::sum_upper_bound(sum, n) as f32)
    }
}

impl<'a> DocIdSetIterator for WANDScorer<'a> {
    fn doc_id(&self) -> i32 {
        self.doc
    }

    fn next_doc(&mut self) -> io::Result<i32> {
        self.advance(self.doc + 1)
    }

    fn advance(&mut self, target: i32) -> io::Result<i32> {
        let mut current_target = target;
        loop {
            let doc = self.advance_approx(current_target)?;
            if doc == NO_MORE_DOCS {
                return Ok(NO_MORE_DOCS);
            }
            if self.matches_current()? {
                return Ok(doc);
            }
            current_target = self.doc + 1;
        }
    }

    fn cost(&self) -> i64 {
        self.cost
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mock scorer with a sorted doc sequence and a fixed max score per doc.
    struct MockScorer {
        docs: Vec<i32>,
        idx: isize,
        per_doc_score: f32,
        max_score: f32,
    }

    impl MockScorer {
        fn boxed(docs: Vec<i32>, per_doc_score: f32) -> Box<dyn Scorer + 'static> {
            Box::new(Self {
                docs,
                idx: -1,
                per_doc_score,
                max_score: per_doc_score,
            })
        }

        fn boxed_with_max(
            docs: Vec<i32>,
            per_doc_score: f32,
            max_score: f32,
        ) -> Box<dyn Scorer + 'static> {
            Box::new(Self {
                docs,
                idx: -1,
                per_doc_score,
                max_score,
            })
        }
    }

    impl fmt::Debug for MockScorer {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("MockScorer")
                .field("len", &self.docs.len())
                .finish()
        }
    }

    impl Scorable for MockScorer {
        fn score(&mut self) -> io::Result<f32> {
            Ok(self.per_doc_score)
        }
    }

    impl Scorer for MockScorer {
        fn doc_id(&self) -> i32 {
            if self.idx < 0 {
                -1
            } else if (self.idx as usize) < self.docs.len() {
                self.docs[self.idx as usize]
            } else {
                NO_MORE_DOCS
            }
        }
        fn iterator(&mut self) -> &mut dyn DocIdSetIterator {
            self
        }
        fn get_max_score(&mut self, _up_to: i32) -> io::Result<f32> {
            Ok(self.max_score)
        }
    }

    impl DocIdSetIterator for MockScorer {
        fn doc_id(&self) -> i32 {
            Scorer::doc_id(self)
        }
        fn next_doc(&mut self) -> io::Result<i32> {
            self.idx += 1;
            Ok(Scorer::doc_id(self))
        }
        fn advance(&mut self, target: i32) -> io::Result<i32> {
            while Scorer::doc_id(self) < target {
                self.next_doc()?;
            }
            Ok(Scorer::doc_id(self))
        }
        fn cost(&self) -> i64 {
            self.docs.len() as i64
        }
    }

    fn collect(scorer: &mut WANDScorer<'_>) -> Vec<i32> {
        let mut out = Vec::new();
        loop {
            let d = scorer.next_doc().unwrap();
            if d == NO_MORE_DOCS {
                break;
            }
            out.push(d);
        }
        out
    }

    fn collect_with_scores(scorer: &mut WANDScorer<'_>) -> Vec<(i32, f32)> {
        let mut out = Vec::new();
        loop {
            let d = scorer.next_doc().unwrap();
            if d == NO_MORE_DOCS {
                break;
            }
            let s = scorer.score().unwrap();
            out.push((d, s));
        }
        out
    }

    // -----------------------------------------------------------------------
    // Construction invariants
    // -----------------------------------------------------------------------

    #[test]
    fn test_msm_equal_to_num_scorers_errors() {
        let scorers: Vec<Box<dyn Scorer + 'static>> = vec![
            MockScorer::boxed(vec![1], 1.0),
            MockScorer::boxed(vec![2], 1.0),
        ];
        let err = WANDScorer::new(scorers, 2, ScoreMode::Complete, 5).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::Other);
    }

    #[test]
    fn test_negative_msm_errors() {
        let scorers: Vec<Box<dyn Scorer + 'static>> = vec![
            MockScorer::boxed(vec![1], 1.0),
            MockScorer::boxed(vec![2], 1.0),
        ];
        let err = WANDScorer::new(scorers, -1, ScoreMode::Complete, 5).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::Other);
    }

    // -----------------------------------------------------------------------
    // minShouldMatch enforcement (msm > 1)
    // -----------------------------------------------------------------------

    #[test]
    fn test_three_subs_msm_2() {
        // A: [1, 2, 3]; B: [2, 3, 4]; C: [3, 4, 5]
        // msm=2 → docs matching ≥ 2 clauses: {2 (A,B), 3 (A,B,C), 4 (B,C)}
        let scorers: Vec<Box<dyn Scorer + 'static>> = vec![
            MockScorer::boxed(vec![1, 2, 3], 1.0),
            MockScorer::boxed(vec![2, 3, 4], 1.0),
            MockScorer::boxed(vec![3, 4, 5], 1.0),
        ];
        let mut s = WANDScorer::new(scorers, 2, ScoreMode::Complete, 5).unwrap();
        assert_eq!(collect(&mut s), vec![2, 3, 4]);
    }

    #[test]
    fn test_four_subs_msm_3_intersection_of_top_3() {
        // 4 subs with msm=3. Doc must match ≥ 3 clauses.
        // A:[1,3,5]; B:[3,5,7]; C:[3,5,9]; D:[3,5,100]
        // doc 3 → A,B,C,D (4) ✓
        // doc 5 → A,B,C,D (4) ✓
        // others < 3 matches
        let scorers: Vec<Box<dyn Scorer + 'static>> = vec![
            MockScorer::boxed(vec![1, 3, 5], 1.0),
            MockScorer::boxed(vec![3, 5, 7], 1.0),
            MockScorer::boxed(vec![3, 5, 9], 1.0),
            MockScorer::boxed(vec![3, 5, 100], 1.0),
        ];
        let mut s = WANDScorer::new(scorers, 3, ScoreMode::Complete, 5).unwrap();
        assert_eq!(collect(&mut s), vec![3, 5]);
    }

    #[test]
    fn test_five_subs_msm_3() {
        // 5 subs, msm=3 → only docs matching ≥ 3 clauses.
        // A:[1,5,10], B:[2,5,10], C:[3,5,11], D:[4,6,10], E:[5,7,12]
        // doc 5: A,B,C,E → 4 matches ✓
        // doc 10: A,B,D → 3 matches ✓
        // doc 11: only C → 1, no
        let scorers: Vec<Box<dyn Scorer + 'static>> = vec![
            MockScorer::boxed(vec![1, 5, 10], 1.0),
            MockScorer::boxed(vec![2, 5, 10], 1.0),
            MockScorer::boxed(vec![3, 5, 11], 1.0),
            MockScorer::boxed(vec![4, 6, 10], 1.0),
            MockScorer::boxed(vec![5, 7, 12], 1.0),
        ];
        let mut s = WANDScorer::new(scorers, 3, ScoreMode::Complete, 5).unwrap();
        assert_eq!(collect(&mut s), vec![5, 10]);
    }

    #[test]
    fn test_msm_2_scores_sum_only_matching_clauses() {
        // A:[1,2] @1.0; B:[2,3] @2.0; C:[2,3] @4.0
        // doc 2 matches A,B,C → score = 1+2+4 = 7.0
        // doc 3 matches B,C → score = 2+4 = 6.0
        let scorers: Vec<Box<dyn Scorer + 'static>> = vec![
            MockScorer::boxed(vec![1, 2], 1.0),
            MockScorer::boxed(vec![2, 3], 2.0),
            MockScorer::boxed(vec![2, 3], 4.0),
        ];
        let mut s = WANDScorer::new(scorers, 2, ScoreMode::Complete, 5).unwrap();
        let pairs = collect_with_scores(&mut s);
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[0].0, 2);
        assert!((pairs[0].1 - 7.0).abs() < 1e-5);
        assert_eq!(pairs[1].0, 3);
        assert!((pairs[1].1 - 6.0).abs() < 1e-5);
    }

    // -----------------------------------------------------------------------
    // Iteration mechanics
    // -----------------------------------------------------------------------

    #[test]
    fn test_doc_id_initial_is_negative_one() {
        let scorers: Vec<Box<dyn Scorer + 'static>> = vec![
            MockScorer::boxed(vec![1], 1.0),
            MockScorer::boxed(vec![2], 1.0),
        ];
        let s = WANDScorer::new(scorers, 1, ScoreMode::Complete, 5).unwrap();
        assert_eq!(Scorer::doc_id(&s), -1);
    }

    #[test]
    fn test_advance_to_specific_doc() {
        let scorers: Vec<Box<dyn Scorer + 'static>> = vec![
            MockScorer::boxed(vec![1, 5, 10], 1.0),
            MockScorer::boxed(vec![2, 5, 10], 1.0),
            MockScorer::boxed(vec![3, 5, 11], 1.0),
        ];
        let mut s = WANDScorer::new(scorers, 2, ScoreMode::Complete, 5).unwrap();
        let d = s.advance(7).unwrap();
        assert_eq!(d, 10);
    }

    #[test]
    fn test_advance_past_all_returns_no_more_docs() {
        let scorers: Vec<Box<dyn Scorer + 'static>> = vec![
            MockScorer::boxed(vec![1, 2], 1.0),
            MockScorer::boxed(vec![3, 4], 1.0),
        ];
        let mut s = WANDScorer::new(scorers, 1, ScoreMode::Complete, 5).unwrap();
        let d = s.advance(100).unwrap();
        assert_eq!(d, NO_MORE_DOCS);
    }

    // -----------------------------------------------------------------------
    // TOP_SCORES min-competitive feedback
    // -----------------------------------------------------------------------

    #[test]
    fn test_top_scores_set_min_competitive_score_does_not_panic() {
        let scorers: Vec<Box<dyn Scorer + 'static>> = vec![
            MockScorer::boxed_with_max(vec![1, 2, 3], 1.0, 2.0),
            MockScorer::boxed_with_max(vec![1, 2, 3], 1.0, 2.0),
            MockScorer::boxed_with_max(vec![1, 2, 3], 1.0, 2.0),
        ];
        let mut s = WANDScorer::new(scorers, 2, ScoreMode::TopScores, 5).unwrap();
        s.set_min_competitive_score(0.1).unwrap();
    }

    #[test]
    fn test_top_scores_iterates_correctly_with_min_competitive() {
        // 4 subs, msm=2, TopScores. Set a moderate min_competitive after iterating a bit.
        let scorers: Vec<Box<dyn Scorer + 'static>> = vec![
            MockScorer::boxed_with_max(vec![1, 5, 10], 1.0, 1.0),
            MockScorer::boxed_with_max(vec![2, 5, 10], 1.0, 1.0),
            MockScorer::boxed_with_max(vec![3, 5, 11], 1.0, 1.0),
            MockScorer::boxed_with_max(vec![4, 6, 10], 1.0, 1.0),
        ];
        let mut s = WANDScorer::new(scorers, 2, ScoreMode::TopScores, 5).unwrap();
        // doc 5 matches 3 subs (A,B,C) → score = 3.0
        // doc 6 matches B,D? No, B has [2,5,10]; D has [4,6,10]. Only D matches doc 6 → no, msm=2.
        // doc 10 matches A,B,D → score = 3.0
        let mut docs_seen = Vec::new();
        loop {
            let d = s.next_doc().unwrap();
            if d == NO_MORE_DOCS {
                break;
            }
            docs_seen.push(d);
        }
        assert!(docs_seen.contains(&5));
        assert!(docs_seen.contains(&10));
    }

    // -----------------------------------------------------------------------
    // Cost and getMaxScore
    // -----------------------------------------------------------------------

    #[test]
    fn test_cost_uses_min_should_match() {
        // 3 scorers with costs [3, 3, 3], msm=2 → keep (3-2+1)=2 least costly = 3+3 = 6
        let scorers: Vec<Box<dyn Scorer + 'static>> = vec![
            MockScorer::boxed(vec![1, 2, 3], 1.0),
            MockScorer::boxed(vec![1, 2, 3], 1.0),
            MockScorer::boxed(vec![1, 2, 3], 1.0),
        ];
        let s = WANDScorer::new(scorers, 2, ScoreMode::Complete, 5).unwrap();
        assert_eq!(s.cost(), 6);
    }

    #[test]
    fn test_get_max_score_sums_active_subs() {
        let scorers: Vec<Box<dyn Scorer + 'static>> = vec![
            MockScorer::boxed_with_max(vec![1, 2, 3], 0.5, 0.5),
            MockScorer::boxed_with_max(vec![1, 2, 3], 0.5, 0.5),
            MockScorer::boxed_with_max(vec![1, 2, 3], 0.5, 0.5),
        ];
        let mut s = WANDScorer::new(scorers, 2, ScoreMode::Complete, 5).unwrap();
        // No iteration yet → all sub-scorer doc_ids are -1, which is <= NO_MORE_DOCS.
        let m = s.get_max_score(NO_MORE_DOCS).unwrap();
        assert!(m >= 1.5);
    }

    // -----------------------------------------------------------------------
    // Scaling helpers
    // -----------------------------------------------------------------------

    #[test]
    fn test_scaling_factor_one() {
        // For 1.0, the unbiased exponent is 0; scaling = 24-1-0 = 23.
        assert_eq!(scaling_factor(1.0), 23);
    }

    #[test]
    fn test_scaling_factor_zero() {
        // scaling_factor(0) = scaling_factor(MIN_VALUE) + 1
        let a = scaling_factor(0.0);
        let b = scaling_factor(f32::MIN_POSITIVE);
        assert_eq!(a, b + 1);
    }

    #[test]
    fn test_scale_max_score_round_trip_close() {
        let f = 1.5_f32;
        let sf = scaling_factor(f);
        let scaled = scale_max_score(f, sf);
        // Should be in [2^23, 2^24)
        assert!(scaled >= 1 << 23);
        assert!(scaled < 1 << 24);
    }
}
