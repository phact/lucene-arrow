// SPDX-License-Identifier: Apache-2.0

//! BulkScorer that filters a required scorer by excluding documents matching a prohibited scorer.

use std::fmt;
use std::io;

use super::query::BulkScorer;
use super::scorer::Scorer;

/// Wraps a positive `BulkScorer` and filters out documents that match a prohibited `Scorer`.
///
/// Scores documents in ranges between excluded doc IDs, using `doc_id_run_end()` to skip
/// over contiguous runs of excluded documents efficiently.
pub struct ReqExclBulkScorer<'a> {
    req: Box<dyn BulkScorer + 'a>,
    excl: Box<dyn Scorer + 'a>,
}

impl fmt::Debug for ReqExclBulkScorer<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ReqExclBulkScorer")
            .field("req", &self.req)
            .finish()
    }
}

impl<'a> ReqExclBulkScorer<'a> {
    /// Creates a new `ReqExclBulkScorer`.
    ///
    /// Rust adaptation: no TwoPhaseIterator support; the excluded scorer's iterator is
    /// accessed via `excl.iterator()`
    pub fn new(req: Box<dyn BulkScorer + 'a>, excl: Box<dyn Scorer + 'a>) -> Self {
        Self { req, excl }
    }
}

impl BulkScorer for ReqExclBulkScorer<'_> {
    fn score(
        &mut self,
        collector: &mut dyn super::collector::LeafCollector,
        min: i32,
        max: i32,
    ) -> io::Result<i32> {
        let mut up_to = min;
        let excl_iter = self.excl.iterator();
        let mut excl_doc = excl_iter.doc_id();

        while up_to < max {
            if excl_doc < up_to {
                excl_doc = excl_iter.advance(up_to)?;
            }
            if excl_doc == up_to {
                // Skip over the excluded run using doc_id_run_end()
                up_to = excl_iter.doc_id_run_end()?.min(max);
                excl_doc = excl_iter.next_doc()?;
            } else {
                up_to = self.req.score(collector, up_to, excl_doc.min(max))?;
            }
        }

        if up_to == max {
            up_to = self.req.score(collector, up_to, up_to)?;
        }

        Ok(up_to)
    }

    fn cost(&self) -> i64 {
        self.req.cost()
    }
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use super::*;
    use crate::search::collector::{LeafCollector, ScoreContext};
    use crate::search::doc_id_set_iterator::{DocIdSetIterator, NO_MORE_DOCS};
    use crate::search::scorable::Scorable;
    use assertables::*;

    // -----------------------------------------------------------------------
    // Mock BulkScorer: collects all docs in [min, max) into a shared vec
    // -----------------------------------------------------------------------

    #[derive(Debug)]
    struct MockBulkScorer {
        docs: Vec<i32>,
    }

    impl BulkScorer for MockBulkScorer {
        fn score(
            &mut self,
            collector: &mut dyn LeafCollector,
            min: i32,
            max: i32,
        ) -> io::Result<i32> {
            for &doc in &self.docs {
                if doc >= min && doc < max {
                    collector.collect(doc)?;
                }
            }
            // Return the next doc >= max, or NO_MORE_DOCS
            for &doc in &self.docs {
                if doc >= max {
                    return Ok(doc);
                }
            }
            Ok(NO_MORE_DOCS)
        }

        fn cost(&self) -> i64 {
            self.docs.len() as i64
        }
    }

    // -----------------------------------------------------------------------
    // Mock exclusion DISI
    // -----------------------------------------------------------------------

    struct MockDISI {
        docs: Vec<i32>,
        idx: usize,
    }

    impl MockDISI {
        fn new(docs: Vec<i32>) -> Self {
            Self { docs, idx: 0 }
        }

        fn current_doc(&self) -> i32 {
            if self.idx < self.docs.len() {
                self.docs[self.idx]
            } else {
                NO_MORE_DOCS
            }
        }
    }

    impl DocIdSetIterator for MockDISI {
        fn doc_id(&self) -> i32 {
            self.current_doc()
        }

        fn next_doc(&mut self) -> io::Result<i32> {
            self.idx += 1;
            Ok(self.current_doc())
        }

        fn advance(&mut self, target: i32) -> io::Result<i32> {
            while self.current_doc() < target {
                self.idx += 1;
            }
            Ok(self.current_doc())
        }

        fn cost(&self) -> i64 {
            self.docs.len() as i64
        }
    }

    // -----------------------------------------------------------------------
    // Mock Scorer wrapping a DISI
    // -----------------------------------------------------------------------

    struct MockExclScorer {
        disi: MockDISI,
    }

    impl fmt::Debug for MockExclScorer {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("MockExclScorer").finish()
        }
    }

    impl Scorable for MockExclScorer {
        fn score(&mut self) -> io::Result<f32> {
            Ok(0.0)
        }
    }

    impl Scorer for MockExclScorer {
        fn doc_id(&self) -> i32 {
            self.disi.doc_id()
        }
        fn iterator(&mut self) -> &mut dyn DocIdSetIterator {
            &mut self.disi
        }
        fn get_max_score(&mut self, _up_to: i32) -> io::Result<f32> {
            Ok(0.0)
        }
    }

    // -----------------------------------------------------------------------
    // Simple collecting LeafCollector
    // -----------------------------------------------------------------------

    #[derive(Debug)]
    struct CollectingLeafCollector {
        collected_docs: Vec<i32>,
    }

    impl CollectingLeafCollector {
        fn new() -> Self {
            Self {
                collected_docs: Vec::new(),
            }
        }
    }

    impl LeafCollector for CollectingLeafCollector {
        fn set_scorer(&mut self, _score_context: Rc<ScoreContext>) -> io::Result<()> {
            Ok(())
        }
        fn collect(&mut self, doc: i32) -> io::Result<()> {
            self.collected_docs.push(doc);
            Ok(())
        }
    }

    // -----------------------------------------------------------------------
    // Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_basic_exclusion() {
        // Required: docs [1, 2, 3, 4, 5]
        // Excluded: docs [2, 4]
        // Expected: [1, 3, 5]
        let req = MockBulkScorer {
            docs: vec![1, 2, 3, 4, 5],
        };
        let excl = MockExclScorer {
            disi: MockDISI::new(vec![2, 4]),
        };
        let mut scorer = ReqExclBulkScorer::new(Box::new(req), Box::new(excl));
        let mut collector = CollectingLeafCollector::new();
        scorer.score(&mut collector, 0, NO_MORE_DOCS).unwrap();
        assert_eq!(collector.collected_docs, vec![1, 3, 5]);
    }

    #[test]
    fn test_no_exclusions_match() {
        // Required: docs [1, 2, 3]
        // Excluded: docs [10, 20]
        // Expected: all required docs pass through
        let req = MockBulkScorer {
            docs: vec![1, 2, 3],
        };
        let excl = MockExclScorer {
            disi: MockDISI::new(vec![10, 20]),
        };
        let mut scorer = ReqExclBulkScorer::new(Box::new(req), Box::new(excl));
        let mut collector = CollectingLeafCollector::new();
        scorer.score(&mut collector, 0, NO_MORE_DOCS).unwrap();
        assert_eq!(collector.collected_docs, vec![1, 2, 3]);
    }

    #[test]
    fn test_all_excluded() {
        // Required: docs [1, 2, 3]
        // Excluded: docs [1, 2, 3]
        // Expected: empty
        let req = MockBulkScorer {
            docs: vec![1, 2, 3],
        };
        let excl = MockExclScorer {
            disi: MockDISI::new(vec![1, 2, 3]),
        };
        let mut scorer = ReqExclBulkScorer::new(Box::new(req), Box::new(excl));
        let mut collector = CollectingLeafCollector::new();
        scorer.score(&mut collector, 0, NO_MORE_DOCS).unwrap();
        assert_is_empty!(collector.collected_docs);
    }

    #[test]
    fn test_empty_exclusion() {
        // Required: docs [1, 2, 3]
        // Excluded: empty
        // Expected: all docs
        let req = MockBulkScorer {
            docs: vec![1, 2, 3],
        };
        let excl = MockExclScorer {
            disi: MockDISI::new(vec![]),
        };
        let mut scorer = ReqExclBulkScorer::new(Box::new(req), Box::new(excl));
        let mut collector = CollectingLeafCollector::new();
        scorer.score(&mut collector, 0, NO_MORE_DOCS).unwrap();
        assert_eq!(collector.collected_docs, vec![1, 2, 3]);
    }

    #[test]
    fn test_cost_delegates_to_req() {
        let req = MockBulkScorer {
            docs: vec![1, 2, 3, 4, 5],
        };
        let excl = MockExclScorer {
            disi: MockDISI::new(vec![2]),
        };
        let scorer = ReqExclBulkScorer::new(Box::new(req), Box::new(excl));
        assert_eq!(scorer.cost(), 5);
    }
}
