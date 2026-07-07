// SPDX-License-Identifier: Apache-2.0

//! Wraps a [`Scorer`] with cached document ID and cost for use in priority queues.

use std::fmt;

use super::scorer::Scorer;

/// Wraps a `Scorer` with cached `doc` and `cost` fields for efficient priority queue operations.
pub struct DisiWrapper<'a> {
    /// The wrapped scorer, providing both scoring and iteration.
    pub scorer: Box<dyn Scorer + 'a>,
    /// Cost of the underlying iterator, cached at construction.
    pub cost: i64,
    /// The match cost for two-phase iterators, 0 otherwise.
    pub match_cost: f32,
    /// Current document ID. Updated during iteration.
    pub doc: i32,
    /// Scaled maximum score used by WANDScorer.
    pub scaled_max_score: i64,
}

impl<'a> DisiWrapper<'a> {
    /// Creates a new `DisiWrapper` from a scorer.
    pub fn new(mut scorer: Box<dyn Scorer + 'a>) -> Self {
        let cost = scorer.iterator().cost();
        Self {
            scorer,
            cost,
            match_cost: 0.0,
            doc: -1,
            scaled_max_score: 0,
        }
    }
}

impl fmt::Debug for DisiWrapper<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DisiWrapper")
            .field("cost", &self.cost)
            .field("match_cost", &self.match_cost)
            .field("doc", &self.doc)
            .field("scaled_max_score", &self.scaled_max_score)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::doc_id_set_iterator::{DocIdSetIterator, NO_MORE_DOCS};
    use crate::search::scorable::Scorable;
    use std::io;

    struct MockScorer {
        docs: Vec<i32>,
        idx: usize,
    }

    impl fmt::Debug for MockScorer {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("MockScorer").finish()
        }
    }

    impl Scorable for MockScorer {
        fn score(&mut self) -> io::Result<f32> {
            Ok(1.0)
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

    #[test]
    fn test_new_wrapper() {
        let scorer = MockScorer {
            docs: vec![1, 5, 10],
            idx: 0,
        };
        let wrapper = DisiWrapper::new(Box::new(scorer));
        assert_eq!(wrapper.doc, -1);
        assert_eq!(wrapper.cost, 3);
    }

    #[test]
    fn test_advance_through_wrapper() {
        let scorer = MockScorer {
            docs: vec![1, 5, 10],
            idx: 0,
        };
        let mut wrapper = DisiWrapper::new(Box::new(scorer));

        // Advance to first doc
        wrapper.doc = wrapper.scorer.iterator().advance(1).unwrap();
        assert_eq!(wrapper.doc, 1);

        // Advance to 5
        wrapper.doc = wrapper.scorer.iterator().advance(5).unwrap();
        assert_eq!(wrapper.doc, 5);

        // Next doc
        wrapper.doc = wrapper.scorer.iterator().next_doc().unwrap();
        assert_eq!(wrapper.doc, 10);

        // Exhausted
        wrapper.doc = wrapper.scorer.iterator().next_doc().unwrap();
        assert_eq!(wrapper.doc, NO_MORE_DOCS);
    }
}
