// SPDX-License-Identifier: Apache-2.0

//! Allows access to the score of a Query.

use std::fmt;
use std::io;

/// A child Scorer and its relationship to its parent. The meaning of the relationship
/// depends upon the parent query.
pub struct ChildScorable<'a> {
    /// Child Scorer. (note this is typically a direct child, and may itself also have children).
    pub child: &'a dyn Scorable,
    /// An arbitrary string relating this scorer to the parent.
    pub relationship: String,
}

impl fmt::Debug for ChildScorable<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ChildScorable")
            .field("relationship", &self.relationship)
            .finish()
    }
}

/// Allows access to the score of a Query.
pub trait Scorable {
    /// Returns the score of the current document matching the query.
    fn score(&mut self) -> io::Result<f32>;

    /// Returns the smoothing score of the current document matching the query. This score is
    /// used when the query/term does not appear in the document, and behaves like an idf. The
    /// smoothing score is particularly important when the Scorer returns a product of
    /// probabilities so that the document score does not go to zero when one probability is
    /// zero. This can return 0 or a smoothing score.
    fn smoothing_score(&mut self, _doc_id: i32) -> io::Result<f32> {
        Ok(0.0)
    }

    /// Optional method: Tell the scorer that its iterator may safely ignore all documents whose
    /// score is less than the given `min_score`. This is a no-op by default.
    ///
    /// This method may only be called from collectors that use `ScoreMode::TopScores`, and
    /// successive calls may only set increasing values of `min_score`.
    fn set_min_competitive_score(&mut self, _min_score: f32) -> io::Result<()> {
        Ok(())
    }

    /// Returns child sub-scorers positioned on the current document.
    fn get_children(&self) -> io::Result<Vec<ChildScorable<'_>>> {
        Ok(vec![])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assertables::*;

    struct TestScorable {
        current_score: f32,
    }

    impl Scorable for TestScorable {
        fn score(&mut self) -> io::Result<f32> {
            Ok(self.current_score)
        }
    }

    #[test]
    fn test_score() {
        let mut s = TestScorable { current_score: 1.5 };
        assert_eq!(s.score().unwrap(), 1.5);
    }

    #[test]
    fn test_default_smoothing_score() {
        let mut s = TestScorable { current_score: 1.0 };
        assert_eq!(s.smoothing_score(0).unwrap(), 0.0);
    }

    #[test]
    fn test_default_set_min_competitive_score() {
        let mut s = TestScorable { current_score: 1.0 };
        s.set_min_competitive_score(0.5).unwrap();
    }

    #[test]
    fn test_default_get_children() {
        let s = TestScorable { current_score: 1.0 };
        let children = s.get_children().unwrap();
        assert_is_empty!(children);
    }
}
