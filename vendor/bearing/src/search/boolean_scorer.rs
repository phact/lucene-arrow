// SPDX-License-Identifier: Apache-2.0

//! BulkScorer for pure disjunctions that scores documents in batches of 4,096 docs.

use std::fmt;
use std::io;
use std::rc::Rc;

use fixedbitset::FixedBitSet;

use super::collector::{DocAndFloatFeatureBuffer, DocIdStream, LeafCollector, ScoreContext};
use super::disi_wrapper::DisiWrapper;
use super::query::BulkScorer;
use super::scorer::Scorer;
use super::scorer_util;

const SHIFT: i32 = 12;
const SIZE: i32 = 1 << SHIFT;
const MASK: i32 = SIZE - 1;

// ---------------------------------------------------------------------------
// Bucket
// ---------------------------------------------------------------------------

/// Accumulates score and frequency for a single doc within the current window.
struct Bucket {
    score: f64,
    freq: i32,
}

// ---------------------------------------------------------------------------
// HeadPQ — min-heap of wrapper indices, ordered by wrappers[idx].doc
// ---------------------------------------------------------------------------

/// Min-heap of `usize` indices into a `Vec<DisiWrapper>`, ordered by `wrapper.doc`.
struct HeadPQ {
    heap: Vec<usize>, // 1-based; heap[0] is unused sentinel
    size: usize,
    max_size: usize,
}

impl HeadPQ {
    fn new(max_size: usize) -> Self {
        let mut heap = Vec::with_capacity(max_size + 1);
        heap.push(0); // unused sentinel at index 0
        Self {
            heap,
            size: 0,
            max_size,
        }
    }

    #[inline]
    fn less_than(a: usize, b: usize, wrappers: &[DisiWrapper<'_>]) -> bool {
        wrappers[a].doc < wrappers[b].doc
    }

    fn add(&mut self, idx: usize, wrappers: &[DisiWrapper<'_>]) -> usize {
        self.size += 1;
        if self.size < self.heap.len() {
            self.heap[self.size] = idx;
        } else {
            self.heap.push(idx);
        }
        self.up_heap(self.size, wrappers);
        self.heap[1]
    }

    fn pop(&mut self, wrappers: &[DisiWrapper<'_>]) -> Option<usize> {
        if self.size > 0 {
            let result = self.heap[1];
            self.heap[1] = self.heap[self.size];
            self.size -= 1;
            self.down_heap(1, wrappers);
            Some(result)
        } else {
            None
        }
    }

    fn top(&self) -> Option<usize> {
        if self.size > 0 {
            Some(self.heap[1])
        } else {
            None
        }
    }

    /// Re-sift after top's key changed.
    fn update_top(&mut self, wrappers: &[DisiWrapper<'_>]) -> usize {
        self.down_heap(1, wrappers);
        self.heap[1]
    }

    /// Replace top with new element and re-sift.
    fn update_top_with(&mut self, new_idx: usize, wrappers: &[DisiWrapper<'_>]) -> usize {
        self.heap[1] = new_idx;
        self.down_heap(1, wrappers);
        self.heap[1]
    }

    /// Bounded insert. Returns the evicted index, or None if nothing was evicted.
    fn insert_with_overflow(&mut self, idx: usize, wrappers: &[DisiWrapper<'_>]) -> Option<usize> {
        if self.size < self.max_size {
            self.add(idx, wrappers);
            None
        } else if self.size > 0 && Self::less_than(self.heap[1], idx, wrappers) {
            let ret = self.heap[1];
            self.heap[1] = idx;
            self.update_top(wrappers);
            Some(ret)
        } else {
            Some(idx)
        }
    }

    fn size(&self) -> usize {
        self.size
    }

    fn up_heap(&mut self, orig_pos: usize, wrappers: &[DisiWrapper<'_>]) {
        let mut i = orig_pos;
        let node = self.heap[i];
        let mut j = i >> 1;
        while j > 0 && Self::less_than(node, self.heap[j], wrappers) {
            self.heap[i] = self.heap[j];
            i = j;
            j >>= 1;
        }
        self.heap[i] = node;
    }

    fn down_heap(&mut self, start: usize, wrappers: &[DisiWrapper<'_>]) {
        let mut i = start;
        let node = self.heap[i];
        let mut j = i << 1;
        let mut k = j + 1;
        if k <= self.size && Self::less_than(self.heap[k], self.heap[j], wrappers) {
            j = k;
        }
        while j <= self.size && Self::less_than(self.heap[j], node, wrappers) {
            self.heap[i] = self.heap[j];
            i = j;
            j = i << 1;
            k = j + 1;
            if k <= self.size && Self::less_than(self.heap[k], self.heap[j], wrappers) {
                j = k;
            }
        }
        self.heap[i] = node;
    }
}

// ---------------------------------------------------------------------------
// TailPQ — min-heap of wrapper indices, ordered by wrappers[idx].cost
// ---------------------------------------------------------------------------

/// Min-heap of `usize` indices into a `Vec<DisiWrapper>`, ordered by `wrapper.cost`.
struct TailPQ {
    heap: Vec<usize>, // 1-based; heap[0] is unused sentinel
    size: usize,
    max_size: usize,
}

impl TailPQ {
    fn new(max_size: usize) -> Self {
        let mut heap = Vec::with_capacity(max_size + 1);
        heap.push(0); // unused sentinel at index 0
        Self {
            heap,
            size: 0,
            max_size,
        }
    }

    #[inline]
    fn less_than(a: usize, b: usize, wrappers: &[DisiWrapper<'_>]) -> bool {
        wrappers[a].cost < wrappers[b].cost
    }

    fn add(&mut self, idx: usize, wrappers: &[DisiWrapper<'_>]) -> usize {
        self.size += 1;
        if self.size < self.heap.len() {
            self.heap[self.size] = idx;
        } else {
            self.heap.push(idx);
        }
        self.up_heap(self.size, wrappers);
        self.heap[1]
    }

    fn pop(&mut self, wrappers: &[DisiWrapper<'_>]) -> Option<usize> {
        if self.size > 0 {
            let result = self.heap[1];
            self.heap[1] = self.heap[self.size];
            self.size -= 1;
            self.down_heap(1, wrappers);
            Some(result)
        } else {
            None
        }
    }

    fn top(&self) -> Option<usize> {
        if self.size > 0 {
            Some(self.heap[1])
        } else {
            None
        }
    }

    /// Replace top with new element and re-sift.
    fn update_top_with(&mut self, new_idx: usize, wrappers: &[DisiWrapper<'_>]) -> usize {
        self.heap[1] = new_idx;
        self.down_heap(1, wrappers);
        self.heap[1]
    }

    /// Bounded insert. Returns the evicted index, or None if nothing was evicted.
    fn insert_with_overflow(&mut self, idx: usize, wrappers: &[DisiWrapper<'_>]) -> Option<usize> {
        if self.size < self.max_size {
            self.add(idx, wrappers);
            None
        } else if self.size > 0 && Self::less_than(self.heap[1], idx, wrappers) {
            let ret = self.heap[1];
            self.heap[1] = idx;
            self.down_heap(1, wrappers);
            Some(ret)
        } else {
            Some(idx)
        }
    }

    fn size(&self) -> usize {
        self.size
    }

    fn clear(&mut self) {
        self.size = 0;
    }

    /// Direct access to the i-th element (0-based).
    fn get(&self, i: usize) -> usize {
        assert!(i < self.size, "index {} out of range [0, {})", i, self.size);
        self.heap[1 + i]
    }

    fn up_heap(&mut self, orig_pos: usize, wrappers: &[DisiWrapper<'_>]) {
        let mut i = orig_pos;
        let node = self.heap[i];
        let mut j = i >> 1;
        while j > 0 && Self::less_than(node, self.heap[j], wrappers) {
            self.heap[i] = self.heap[j];
            i = j;
            j >>= 1;
        }
        self.heap[i] = node;
    }

    fn down_heap(&mut self, start: usize, wrappers: &[DisiWrapper<'_>]) {
        let mut i = start;
        let node = self.heap[i];
        let mut j = i << 1;
        let mut k = j + 1;
        if k <= self.size && Self::less_than(self.heap[k], self.heap[j], wrappers) {
            j = k;
        }
        while j <= self.size && Self::less_than(self.heap[j], node, wrappers) {
            self.heap[i] = self.heap[j];
            i = j;
            j = i << 1;
            k = j + 1;
            if k <= self.size && Self::less_than(self.heap[k], self.heap[j], wrappers) {
                j = k;
            }
        }
        self.heap[i] = node;
    }
}

// ---------------------------------------------------------------------------
// BitSetDocIdStream — DocIdStream over a FixedBitSet
// ---------------------------------------------------------------------------

/// Streams doc IDs from set bits in a `FixedBitSet`, offset by a base doc ID.
struct BitSetDocIdStream<'a> {
    bit_set: &'a FixedBitSet,
    base: i32,
    word_idx: usize,
}

impl<'a> BitSetDocIdStream<'a> {
    fn new(bit_set: &'a FixedBitSet, base: i32) -> Self {
        Self {
            bit_set,
            base,
            word_idx: 0,
        }
    }
}

impl DocIdStream for BitSetDocIdStream<'_> {
    fn for_each_up_to(
        &mut self,
        up_to: i32,
        consumer: &mut dyn FnMut(i32) -> io::Result<()>,
    ) -> io::Result<()> {
        let words = self.bit_set.as_slice();
        while self.word_idx < words.len() {
            let mut bits = words[self.word_idx] as u64;
            while bits != 0 {
                let ntz = bits.trailing_zeros() as usize;
                let doc = self.base + (self.word_idx * 64 + ntz) as i32;
                if doc >= up_to {
                    return Ok(());
                }
                consumer(doc)?;
                bits ^= 1u64 << ntz;
            }
            self.word_idx += 1;
        }
        Ok(())
    }

    fn count_up_to(&mut self, up_to: i32) -> io::Result<i32> {
        let mut count = 0i32;
        let words = self.bit_set.as_slice();
        while self.word_idx < words.len() {
            let mut bits = words[self.word_idx] as u64;
            while bits != 0 {
                let ntz = bits.trailing_zeros() as usize;
                let doc = self.base + (self.word_idx * 64 + ntz) as i32;
                if doc >= up_to {
                    return Ok(count);
                }
                count += 1;
                bits ^= 1u64 << ntz;
            }
            self.word_idx += 1;
        }
        Ok(count)
    }

    fn may_have_remaining(&self) -> bool {
        self.word_idx < self.bit_set.as_slice().len()
    }
}

// ---------------------------------------------------------------------------
// BooleanScorer
// ---------------------------------------------------------------------------

/// BulkScorer for pure disjunctions that scores documents in batches of 4,096 docs.
pub struct BooleanScorer<'a> {
    wrappers: Vec<DisiWrapper<'a>>,
    buckets: Option<Vec<Bucket>>,
    matching: FixedBitSet,
    leads: Vec<usize>,
    head: HeadPQ,
    tail: TailPQ,
    score_context: Rc<ScoreContext>,
    min_should_match: i32,
    cost: i64,
    needs_scores: bool,
    doc_and_score_buffer: DocAndFloatFeatureBuffer,
}

impl fmt::Debug for BooleanScorer<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BooleanScorer")
            .field("wrappers_count", &self.wrappers.len())
            .field("min_should_match", &self.min_should_match)
            .field("cost", &self.cost)
            .field("needs_scores", &self.needs_scores)
            .finish()
    }
}

impl<'a> BooleanScorer<'a> {
    /// Creates a new BooleanScorer from a collection of scorers.
    pub fn new(
        scorers: Vec<Box<dyn Scorer + 'a>>,
        min_should_match: i32,
        needs_scores: bool,
    ) -> io::Result<Self> {
        if min_should_match < 1 || min_should_match > scorers.len() as i32 {
            return Err(io::Error::other(format!(
                "minShouldMatch should be within 1..num_scorers. Got {}",
                min_should_match
            )));
        }
        if scorers.len() <= 1 {
            return Err(io::Error::other(format!(
                "This scorer can only be used with two scorers or more, got {}",
                scorers.len()
            )));
        }

        let buckets = if needs_scores || min_should_match > 1 {
            let mut b = Vec::with_capacity(SIZE as usize);
            for _ in 0..SIZE {
                b.push(Bucket {
                    score: 0.0,
                    freq: 0,
                });
            }
            Some(b)
        } else {
            None
        };

        let num_scorers = scorers.len();
        let leads = vec![0usize; num_scorers];
        let head_max_size = num_scorers - min_should_match as usize + 1;
        let tail_max_size = min_should_match as usize - 1;
        let mut head = HeadPQ::new(head_max_size);
        let mut tail = TailPQ::new(tail_max_size);

        let mut wrappers = Vec::with_capacity(num_scorers);
        let mut costs = Vec::with_capacity(num_scorers);
        for scorer in scorers {
            let w = DisiWrapper::new(scorer);
            costs.push(w.cost);
            wrappers.push(w);
            let idx = wrappers.len() - 1;
            let evicted = tail.insert_with_overflow(idx, &wrappers);
            if let Some(evicted_idx) = evicted {
                head.add(evicted_idx, &wrappers);
            }
        }

        let cost = scorer_util::cost_with_min_should_match(&costs, num_scorers, min_should_match);

        Ok(Self {
            wrappers,
            buckets,
            matching: FixedBitSet::with_capacity(SIZE as usize),
            leads,
            head,
            tail,
            score_context: ScoreContext::new(),
            min_should_match,
            cost,
            needs_scores,
            doc_and_score_buffer: DocAndFloatFeatureBuffer::new(),
        })
    }

    /// Advance all scorers to position >= min.
    fn advance(&mut self, min: i32) -> io::Result<usize> {
        debug_assert!(self.tail.size() == self.min_should_match as usize - 1);
        let mut head_top_idx = self.head.top().unwrap();
        let mut tail_top_idx = self.tail.top();

        while self.wrappers[head_top_idx].doc < min {
            if tail_top_idx.is_none()
                || self.wrappers[head_top_idx].cost <= self.wrappers[tail_top_idx.unwrap()].cost
            {
                let doc = self.wrappers[head_top_idx].scorer.iterator().advance(min)?;
                self.wrappers[head_top_idx].doc = doc;
                head_top_idx = self.head.update_top(&self.wrappers);
            } else {
                // swap top of head and tail
                let previous_head_top = head_top_idx;
                let tail_top = tail_top_idx.unwrap();
                let doc = self.wrappers[tail_top].scorer.iterator().advance(min)?;
                self.wrappers[tail_top].doc = doc;
                head_top_idx = self.head.update_top_with(tail_top, &self.wrappers);
                tail_top_idx = Some(self.tail.update_top_with(previous_head_top, &self.wrappers));
            }
        }
        Ok(head_top_idx)
    }

    /// Process one 4096-doc window.
    fn score_window(
        &mut self,
        top_idx: usize,
        collector: &mut dyn LeafCollector,
        min: i32,
        max: i32,
    ) -> io::Result<usize> {
        let window_base = self.wrappers[top_idx].doc & !MASK;
        let window_min = min.max(window_base);
        let window_max = max.min(window_base + SIZE);

        // Fill leads with all scorers from head that are in this window
        self.leads[0] = self.head.pop(&self.wrappers).unwrap();
        let mut max_freq = 1usize;
        while self.head.size() > 0 && self.wrappers[self.head.top().unwrap()].doc < window_max {
            self.leads[max_freq] = self.head.pop(&self.wrappers).unwrap();
            max_freq += 1;
        }

        if self.min_should_match == 1 && max_freq == 1 {
            // special case: only one scorer can match in the current window
            let bulk_scorer_idx = self.leads[0];
            self.score_window_single_scorer(
                bulk_scorer_idx,
                collector,
                window_min,
                window_max,
                max,
            )?;
            let new_top = self.head.add(bulk_scorer_idx, &self.wrappers);
            Ok(new_top)
        } else {
            // general case, collect through a bit set first and then replay
            self.score_window_multiple_scorers(
                collector,
                window_base,
                window_min,
                window_max,
                max_freq,
            )?;
            Ok(self.head.top().unwrap())
        }
    }

    /// Fast path for a single scorer in the current window.
    fn score_window_single_scorer(
        &mut self,
        w_idx: usize,
        collector: &mut dyn LeafCollector,
        window_min: i32,
        window_max: i32,
        max: i32,
    ) -> io::Result<()> {
        debug_assert!(self.tail.size() == 0);
        let next_window_base = self.wrappers[self.head.top().unwrap()].doc & !MASK;
        let end = window_max.max(max.min(next_window_base));

        let mut doc = self.wrappers[w_idx].doc;
        if doc < window_min {
            doc = self.wrappers[w_idx].scorer.iterator().advance(window_min)?;
        }

        // Rust adaptation: eagerly compute score into ScoreContext before each collect(),
        // since we can't pass a mutable scorer reference through setScorer like Java does.
        while doc < end {
            if self.needs_scores {
                self.score_context
                    .score
                    .set(self.wrappers[w_idx].scorer.score()?);
            }
            collector.collect(doc)?;
            doc = self.wrappers[w_idx].scorer.iterator().next_doc()?;
        }
        self.wrappers[w_idx].doc = doc;

        Ok(())
    }

    /// General path for multiple matching scorers in the current window.
    fn score_window_multiple_scorers(
        &mut self,
        collector: &mut dyn LeafCollector,
        window_base: i32,
        window_min: i32,
        window_max: i32,
        mut max_freq: usize,
    ) -> io::Result<()> {
        while (max_freq as i32) < self.min_should_match
            && max_freq + self.tail.size() >= self.min_should_match as usize
        {
            let candidate_idx = self.tail.pop(&self.wrappers).unwrap();
            if self.wrappers[candidate_idx].doc < window_min {
                let doc = self.wrappers[candidate_idx]
                    .scorer
                    .iterator()
                    .advance(window_min)?;
                self.wrappers[candidate_idx].doc = doc;
            }
            if self.wrappers[candidate_idx].doc < window_max {
                self.leads[max_freq] = candidate_idx;
                max_freq += 1;
            } else {
                self.head.add(candidate_idx, &self.wrappers);
            }
        }

        if max_freq as i32 >= self.min_should_match {
            // There might be matches in other scorers from the tail too
            for i in 0..self.tail.size() {
                self.leads[max_freq] = self.tail.get(i);
                max_freq += 1;
            }
            self.tail.clear();

            self.score_window_into_bit_set_and_replay(
                collector,
                window_base,
                window_min,
                window_max,
                max_freq,
            )?;
        }

        // Push back scorers into head and tail
        for i in 0..max_freq {
            let lead_idx = self.leads[i];
            let evicted = self.head.insert_with_overflow(lead_idx, &self.wrappers);
            if let Some(evicted_idx) = evicted {
                self.tail.add(evicted_idx, &self.wrappers);
            }
        }

        Ok(())
    }

    /// Collect docs into a bitset and replay to the collector.
    fn score_window_into_bit_set_and_replay(
        &mut self,
        collector: &mut dyn LeafCollector,
        base: i32,
        min: i32,
        max: i32,
        num_scorers: usize,
    ) -> io::Result<()> {
        for i in 0..num_scorers {
            let w_idx = self.leads[i];
            debug_assert!(self.wrappers[w_idx].doc < max);

            if self.wrappers[w_idx].doc < min {
                let doc = self.wrappers[w_idx].scorer.iterator().advance(min)?;
                self.wrappers[w_idx].doc = doc;
            }

            match &mut self.buckets {
                None => {
                    let w = &mut self.wrappers[w_idx];
                    w.scorer
                        .iterator()
                        .fill_bit_set(max, &mut self.matching, base)?;
                    w.doc = w.scorer.iterator().doc_id();
                }
                Some(buckets) if self.needs_scores => {
                    loop {
                        self.wrappers[w_idx]
                            .scorer
                            .next_docs_and_scores(max, &mut self.doc_and_score_buffer)?;
                        if self.doc_and_score_buffer.size == 0 {
                            break;
                        }
                        for index in 0..self.doc_and_score_buffer.size {
                            let doc = self.doc_and_score_buffer.docs[index];
                            let score = self.doc_and_score_buffer.features[index];
                            let d = (doc & MASK) as usize;
                            self.matching.insert(d);
                            buckets[d].freq += 1;
                            buckets[d].score += score as f64;
                        }
                    }
                    self.wrappers[w_idx].doc = self.wrappers[w_idx].scorer.doc_id();
                }
                Some(buckets) => {
                    debug_assert!(self.min_should_match > 1);
                    let w = &mut self.wrappers[w_idx];
                    let mut doc = w.scorer.iterator().doc_id();
                    while doc < max {
                        let d = (doc & MASK) as usize;
                        self.matching.insert(d);
                        buckets[d].freq += 1;
                        doc = w.scorer.iterator().next_doc()?;
                    }
                    w.doc = w.scorer.iterator().doc_id();
                }
            }
        }

        // Replay
        match &mut self.buckets {
            None => {
                let mut stream = BitSetDocIdStream::new(&self.matching, base);
                collector.collect_stream(&mut stream)?;
            }
            Some(buckets) => {
                for (idx, &word) in self.matching.as_slice().iter().enumerate() {
                    let mut bits = word as u64;
                    while bits != 0 {
                        let ntz = bits.trailing_zeros() as usize;
                        let index_in_window = (idx << 6) | ntz;
                        let bucket = &mut buckets[index_in_window];
                        if bucket.freq >= self.min_should_match {
                            self.score_context.score.set(bucket.score as f32);
                            collector.collect(base | index_in_window as i32)?;
                        }
                        bucket.freq = 0;
                        bucket.score = 0.0;
                        bits ^= 1u64 << ntz;
                    }
                }
            }
        }

        self.matching.clear();
        Ok(())
    }
}

impl BulkScorer for BooleanScorer<'_> {
    fn score(&mut self, collector: &mut dyn LeafCollector, min: i32, max: i32) -> io::Result<i32> {
        collector.set_scorer(Rc::clone(&self.score_context))?;

        let mut top_idx = self.advance(min)?;
        while self.wrappers[top_idx].doc < max {
            top_idx = self.score_window(top_idx, collector, min, max)?;
        }

        Ok(self.wrappers[top_idx].doc)
    }

    fn cost(&self) -> i64 {
        self.cost
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::doc_id_set_iterator::{DocIdSetIterator, NO_MORE_DOCS};
    use crate::search::scorable::Scorable;
    use assertables::*;

    // -----------------------------------------------------------------------
    // Mock scorer for testing
    // -----------------------------------------------------------------------

    struct MockScorer {
        docs: Vec<i32>,
        scores: Vec<f32>,
        idx: usize,
    }

    impl MockScorer {
        fn new(docs: Vec<i32>, score: f32) -> Self {
            let scores = vec![score; docs.len()];
            Self {
                docs,
                scores,
                idx: 0,
            }
        }
    }

    impl fmt::Debug for MockScorer {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("MockScorer")
                .field("docs_len", &self.docs.len())
                .finish()
        }
    }

    impl Scorable for MockScorer {
        fn score(&mut self) -> io::Result<f32> {
            if self.idx < self.scores.len() {
                Ok(self.scores[self.idx])
            } else {
                Ok(0.0)
            }
        }
    }

    impl Scorer for MockScorer {
        fn doc_id(&self) -> i32 {
            if self.idx < self.docs.len() {
                self.docs[self.idx]
            } else {
                NO_MORE_DOCS
            }
        }
        fn iterator(&mut self) -> &mut dyn DocIdSetIterator {
            self
        }
        fn get_max_score(&mut self, _up_to: i32) -> io::Result<f32> {
            Ok(f32::MAX)
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

    // -----------------------------------------------------------------------
    // Simple collecting LeafCollector for tests
    // -----------------------------------------------------------------------

    #[derive(Debug)]
    struct CollectingLeafCollector {
        score_context: Option<Rc<ScoreContext>>,
        collected_docs: Vec<i32>,
        collected_scores: Vec<f32>,
    }

    impl CollectingLeafCollector {
        fn new() -> Self {
            Self {
                score_context: None,
                collected_docs: Vec::new(),
                collected_scores: Vec::new(),
            }
        }
    }

    impl LeafCollector for CollectingLeafCollector {
        fn set_scorer(&mut self, score_context: Rc<ScoreContext>) -> io::Result<()> {
            self.score_context = Some(score_context);
            Ok(())
        }

        fn collect(&mut self, doc: i32) -> io::Result<()> {
            self.collected_docs.push(doc);
            if let Some(ref ctx) = self.score_context {
                self.collected_scores.push(ctx.score.get());
            }
            Ok(())
        }
    }

    // -----------------------------------------------------------------------
    // FixedBitSet tests (matching window usage)
    // -----------------------------------------------------------------------

    #[test]
    fn test_fixed_bit_set_insert_and_read() {
        let mut bs = FixedBitSet::with_capacity(SIZE as usize);
        bs.insert(0);
        bs.insert(63);
        bs.insert(64);
        bs.insert(4095);
        assert!(bs.contains(0));
        assert!(bs.contains(63));
        assert!(bs.contains(64));
        assert!(bs.contains(4095));
        assert!(!bs.contains(1));
    }

    #[test]
    fn test_fixed_bit_set_clear() {
        let mut bs = FixedBitSet::with_capacity(SIZE as usize);
        bs.insert(100);
        bs.insert(200);
        bs.clear();
        assert!(!bs.contains(100));
        assert!(!bs.contains(200));
    }

    // -----------------------------------------------------------------------
    // HeadPQ tests
    // -----------------------------------------------------------------------

    fn test_wrapper(cost: i64, doc: i32) -> DisiWrapper<'static> {
        DisiWrapper {
            scorer: Box::new(MockScorer::new(vec![], 0.0)),
            cost,
            match_cost: 0.0,
            doc,
            scaled_max_score: 0,
        }
    }

    #[test]
    fn test_head_pq_basic() {
        let wrappers = vec![
            test_wrapper(10, 5),
            test_wrapper(20, 3),
            test_wrapper(30, 8),
        ];

        let mut pq = HeadPQ::new(3);
        pq.add(0, &wrappers);
        pq.add(1, &wrappers);
        pq.add(2, &wrappers);

        // Min by doc: wrapper[1].doc = 3
        assert_eq!(pq.top(), Some(1));
        assert_eq!(pq.pop(&wrappers), Some(1));
        assert_eq!(pq.top(), Some(0)); // doc=5
        assert_eq!(pq.pop(&wrappers), Some(0));
        assert_eq!(pq.top(), Some(2)); // doc=8
        assert_eq!(pq.pop(&wrappers), Some(2));
        assert_eq!(pq.top(), None);
    }

    #[test]
    fn test_head_pq_update_top() {
        let mut wrappers = vec![
            test_wrapper(10, 3),
            test_wrapper(20, 5),
            test_wrapper(30, 8),
        ];

        let mut pq = HeadPQ::new(3);
        pq.add(0, &wrappers);
        pq.add(1, &wrappers);
        pq.add(2, &wrappers);

        // Update wrapper[0].doc from 3 to 10, then re-heap
        wrappers[0].doc = 10;
        let new_top = pq.update_top(&wrappers);
        assert_eq!(new_top, 1); // wrapper[1].doc = 5 is now smallest
    }

    // -----------------------------------------------------------------------
    // TailPQ tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_tail_pq_basic() {
        let wrappers = vec![
            test_wrapper(30, 0),
            test_wrapper(10, 0),
            test_wrapper(20, 0),
        ];

        let mut pq = TailPQ::new(3);
        pq.add(0, &wrappers);
        pq.add(1, &wrappers);
        pq.add(2, &wrappers);

        // Min by cost: wrapper[1].cost = 10
        assert_eq!(pq.top(), Some(1));
    }

    #[test]
    fn test_tail_pq_get() {
        let wrappers = vec![
            test_wrapper(30, 0),
            test_wrapper(10, 0),
            test_wrapper(20, 0),
        ];

        let mut pq = TailPQ::new(3);
        pq.add(0, &wrappers);
        pq.add(1, &wrappers);
        pq.add(2, &wrappers);

        // get(i) accesses heap[1+i] — collect all and verify all indices present
        let mut indices: Vec<usize> = (0..pq.size()).map(|i| pq.get(i)).collect();
        indices.sort();
        assert_eq!(indices, vec![0, 1, 2]);
    }

    // -----------------------------------------------------------------------
    // BooleanScorer tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_two_scorers_needs_scores() {
        // Scorer A: docs [1, 3, 5] with score 1.0
        // Scorer B: docs [2, 3, 6] with score 2.0
        // Expected union: [1, 2, 3, 5, 6]
        // Doc 3 overlaps: score = 1.0 + 2.0 = 3.0
        let scorer_a = MockScorer::new(vec![1, 3, 5], 1.0);
        let scorer_b = MockScorer::new(vec![2, 3, 6], 2.0);

        let mut bs =
            BooleanScorer::new(vec![Box::new(scorer_a), Box::new(scorer_b)], 1, true).unwrap();

        let mut collector = CollectingLeafCollector::new();
        bs.score(&mut collector, 0, NO_MORE_DOCS).unwrap();

        assert_eq!(collector.collected_docs, vec![1, 2, 3, 5, 6]);
        assert_in_delta!(collector.collected_scores[0], 1.0, 1e-5); // doc 1: only A
        assert_in_delta!(collector.collected_scores[1], 2.0, 1e-5); // doc 2: only B
        assert_in_delta!(collector.collected_scores[2], 3.0, 1e-5); // doc 3: A+B
        assert_in_delta!(collector.collected_scores[3], 1.0, 1e-5); // doc 5: only A
        assert_in_delta!(collector.collected_scores[4], 2.0, 1e-5); // doc 6: only B
    }

    #[test]
    fn test_two_scorers_no_scores() {
        // Same as above but needsScores=false, msm=1 -> takes buckets==None path
        let scorer_a = MockScorer::new(vec![1, 3, 5], 1.0);
        let scorer_b = MockScorer::new(vec![2, 3, 6], 2.0);

        let mut bs =
            BooleanScorer::new(vec![Box::new(scorer_a), Box::new(scorer_b)], 1, false).unwrap();

        let mut collector = CollectingLeafCollector::new();
        bs.score(&mut collector, 0, NO_MORE_DOCS).unwrap();

        assert_eq!(collector.collected_docs, vec![1, 2, 3, 5, 6]);
    }

    #[test]
    fn test_three_scorers_msm_2() {
        // Scorer A: docs [1, 2, 3] with score 1.0
        // Scorer B: docs [2, 3, 4] with score 2.0
        // Scorer C: docs [3, 4, 5] with score 3.0
        // msm=2: need at least 2 matches
        // Doc 2: A+B (freq=2) -> score = 3.0
        // Doc 3: A+B+C (freq=3) -> score = 6.0
        // Doc 4: B+C (freq=2) -> score = 5.0
        let scorer_a = MockScorer::new(vec![1, 2, 3], 1.0);
        let scorer_b = MockScorer::new(vec![2, 3, 4], 2.0);
        let scorer_c = MockScorer::new(vec![3, 4, 5], 3.0);

        let mut bs = BooleanScorer::new(
            vec![Box::new(scorer_a), Box::new(scorer_b), Box::new(scorer_c)],
            2,
            true,
        )
        .unwrap();

        let mut collector = CollectingLeafCollector::new();
        bs.score(&mut collector, 0, NO_MORE_DOCS).unwrap();

        assert_eq!(collector.collected_docs, vec![2, 3, 4]);
        assert_in_delta!(collector.collected_scores[0], 3.0, 1e-5); // doc 2: A+B
        assert_in_delta!(collector.collected_scores[1], 6.0, 1e-5); // doc 3: A+B+C
        assert_in_delta!(collector.collected_scores[2], 5.0, 1e-5); // doc 4: B+C
    }

    #[test]
    fn test_window_boundary() {
        // Docs spanning two 4096-doc windows: one doc in window 0, one in window 1
        let scorer_a = MockScorer::new(vec![100, 4200], 1.0);
        let scorer_b = MockScorer::new(vec![100, 4200], 2.0);

        let mut bs =
            BooleanScorer::new(vec![Box::new(scorer_a), Box::new(scorer_b)], 1, true).unwrap();

        let mut collector = CollectingLeafCollector::new();
        bs.score(&mut collector, 0, NO_MORE_DOCS).unwrap();

        assert_eq!(collector.collected_docs, vec![100, 4200]);
        assert_in_delta!(collector.collected_scores[0], 3.0, 1e-5);
        assert_in_delta!(collector.collected_scores[1], 3.0, 1e-5);
    }

    #[test]
    fn test_single_scorer_fast_path() {
        // Two scorers but only one has docs in a given window -> single scorer path
        let scorer_a = MockScorer::new(vec![1, 2, 3], 1.5);
        let scorer_b = MockScorer::new(vec![5000, 5001], 2.5);

        let mut bs =
            BooleanScorer::new(vec![Box::new(scorer_a), Box::new(scorer_b)], 1, true).unwrap();

        let mut collector = CollectingLeafCollector::new();
        bs.score(&mut collector, 0, NO_MORE_DOCS).unwrap();

        assert_eq!(collector.collected_docs, vec![1, 2, 3, 5000, 5001]);
        assert_in_delta!(collector.collected_scores[0], 1.5, 1e-5);
        assert_in_delta!(collector.collected_scores[3], 2.5, 1e-5);
    }

    #[test]
    fn test_validation_min_should_match() {
        let scorer_a = MockScorer::new(vec![1], 1.0);
        let scorer_b = MockScorer::new(vec![2], 1.0);

        // msm=0 should fail
        let result = BooleanScorer::new(vec![Box::new(scorer_a), Box::new(scorer_b)], 0, true);
        assert!(result.is_err());
    }

    #[test]
    fn test_validation_single_scorer() {
        let scorer_a = MockScorer::new(vec![1], 1.0);

        // Single scorer should fail
        let result = BooleanScorer::new(vec![Box::new(scorer_a)], 1, true);
        assert!(result.is_err());
    }

    #[test]
    fn test_cost() {
        let scorer_a = MockScorer::new(vec![1, 2, 3], 1.0);
        let scorer_b = MockScorer::new(vec![4, 5], 1.0);

        let bs = BooleanScorer::new(vec![Box::new(scorer_a), Box::new(scorer_b)], 1, true).unwrap();
        // msm=1, 2 scorers: cost = sum of all costs = 3 + 2 = 5
        assert_eq!(bs.cost(), 5);
    }
}
