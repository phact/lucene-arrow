// SPDX-License-Identifier: Apache-2.0

//! `BooleanWeight` and `BooleanScorerSupplier` — weight and scorer creation for `BooleanQuery`.

use std::cell::Cell;
use std::collections::HashMap;
use std::fmt;
use std::io;

use super::block_max_conjunction::BlockMaxConjunctionBulkScorer;
use super::boolean_query::{BooleanClause, Occur};
use super::boolean_scorer::BooleanScorer;
use super::collector::ScoreMode;
use super::conjunction::ConjunctionScorer;
use super::disjunction_sum_scorer::DisjunctionSumScorer;
use super::index_searcher::IndexSearcher;
use super::query::{BulkScorer, DefaultBulkScorer, ScorerSupplier, Weight};
use super::req_excl_bulk_scorer::ReqExclBulkScorer;
use super::req_excl_scorer::ReqExclScorer;
use super::req_opt_sum_scorer::ReqOptSumScorer;
use super::scorer::Scorer;
use super::scorer_util;
use super::wand_scorer::WANDScorer;
use crate::index::directory_reader::LeafReaderContext;

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

/// A `BooleanClause` paired with its `Weight`.
///
/// The reference implementation stores a `BooleanClause` reference here. In Rust, the clause
/// is borrowed during construction and only the `Occur` is retained — the `Query` inside the
/// clause has already been consumed by `create_weight`.
struct WeightedBooleanClause {
    occur: Occur,
    weight: Box<dyn Weight>,
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

/// Expert: the Weight for BooleanQuery, used to score and explain boolean queries.
pub(crate) struct BooleanWeight {
    weighted_clauses: Vec<WeightedBooleanClause>,
    score_mode: ScoreMode,
    min_should_match: i32,
}

impl fmt::Debug for BooleanWeight {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BooleanWeight")
            .field("num_clauses", &self.weighted_clauses.len())
            .field("score_mode", &self.score_mode)
            .field("min_should_match", &self.min_should_match)
            .finish()
    }
}

impl BooleanWeight {
    /// Creates a new `BooleanWeight` for the given clauses.
    pub(crate) fn new(
        clauses: &[BooleanClause],
        searcher: &IndexSearcher,
        score_mode: ScoreMode,
        min_should_match: i32,
        boost: f32,
    ) -> io::Result<Self> {
        let mut weighted_clauses = Vec::with_capacity(clauses.len());
        for c in clauses {
            let w = searcher.create_weight(
                c.query(),
                if c.occur() == Occur::Must || c.occur() == Occur::Should {
                    score_mode
                } else {
                    ScoreMode::CompleteNoScores
                },
                boost,
            )?;
            weighted_clauses.push(WeightedBooleanClause {
                occur: c.occur(),
                weight: w,
            });
        }
        Ok(Self {
            weighted_clauses,
            score_mode,
            min_should_match,
        })
    }
}

impl Weight for BooleanWeight {
    fn scorer_supplier<'a>(
        &self,
        context: &'a LeafReaderContext,
    ) -> io::Result<Option<Box<dyn ScorerSupplier<'a> + 'a>>> {
        let mut min_should_match = self.min_should_match;

        let mut scorers: HashMap<Occur, Vec<Box<dyn ScorerSupplier<'a> + 'a>>> = HashMap::new();
        scorers.insert(Occur::Must, Vec::new());
        scorers.insert(Occur::Filter, Vec::new());
        scorers.insert(Occur::Should, Vec::new());
        scorers.insert(Occur::MustNot, Vec::new());

        for wc in &self.weighted_clauses {
            let sub_scorer = wc.weight.scorer_supplier(context)?;
            match sub_scorer {
                None => {
                    if wc.occur == Occur::Must || wc.occur == Occur::Filter {
                        return Ok(None);
                    }
                }
                Some(supplier) => {
                    scorers.get_mut(&wc.occur).unwrap().push(supplier);
                }
            }
        }

        // scorer simplifications:

        if scorers[&Occur::Should].len() == min_should_match as usize {
            // any optional clauses are in fact required
            let should = scorers.remove(&Occur::Should).unwrap();
            scorers.get_mut(&Occur::Must).unwrap().extend(should);
            scorers.insert(Occur::Should, Vec::new());
            min_should_match = 0;
        }

        if scorers[&Occur::Filter].is_empty()
            && scorers[&Occur::Must].is_empty()
            && scorers[&Occur::Should].is_empty()
        {
            // no required and optional clauses.
            return Ok(None);
        } else if (scorers[&Occur::Should].len() as i32) < min_should_match {
            // either >1 req scorer, or there are 0 req scorers and at least 1
            // optional scorer. Therefore if there are not enough optional scorers
            // no documents will be matched by the query
            return Ok(None);
        }

        if !self.score_mode.needs_scores()
            && min_should_match == 0
            && scorers[&Occur::Must].len() + scorers[&Occur::Filter].len() > 0
        {
            // Purely optional clauses are useless without scoring.
            scorers.get_mut(&Occur::Should).unwrap().clear();
        }

        Ok(Some(Box::new(BooleanScorerSupplier::new(
            scorers,
            self.score_mode,
            min_should_match,
            context.reader.max_doc(),
        )?)))
    }
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

/// Supplier of scorers for `BooleanQuery`.
struct BooleanScorerSupplier<'a> {
    subs: HashMap<Occur, Vec<Box<dyn ScorerSupplier<'a> + 'a>>>,
    score_mode: ScoreMode,
    min_should_match: i32,
    max_doc: i32,
    cost: Cell<i64>,
    top_level_scoring_clause: bool,
}

impl fmt::Debug for BooleanScorerSupplier<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BooleanScorerSupplier")
            .field("score_mode", &self.score_mode)
            .field("min_should_match", &self.min_should_match)
            .finish()
    }
}

impl<'a> BooleanScorerSupplier<'a> {
    fn new(
        subs: HashMap<Occur, Vec<Box<dyn ScorerSupplier<'a> + 'a>>>,
        score_mode: ScoreMode,
        min_should_match: i32,
        max_doc: i32,
    ) -> io::Result<Self> {
        if min_should_match < 0 {
            return Err(io::Error::other(format!(
                "minShouldMatch must be positive, but got: {min_should_match}"
            )));
        }
        if min_should_match != 0 && min_should_match as usize >= subs[&Occur::Should].len() {
            return Err(io::Error::other(
                "minShouldMatch must be strictly less than the number of SHOULD clauses",
            ));
        }
        if !score_mode.needs_scores()
            && min_should_match == 0
            && !subs[&Occur::Should].is_empty()
            && subs[&Occur::Must].len() + subs[&Occur::Filter].len() > 0
        {
            return Err(io::Error::other(
                "Cannot pass purely optional clauses if scores are not needed",
            ));
        }
        if subs[&Occur::Should].len() + subs[&Occur::Must].len() + subs[&Occur::Filter].len() == 0 {
            return Err(io::Error::other(
                "There should be at least one positive clause",
            ));
        }

        Ok(Self {
            subs,
            score_mode,
            min_should_match,
            max_doc,
            cost: Cell::new(-1),
            top_level_scoring_clause: false,
        })
    }

    fn compute_cost(&self) -> i64 {
        let min_required_cost = self.subs[&Occur::Must]
            .iter()
            .chain(self.subs[&Occur::Filter].iter())
            .map(|s| s.cost())
            .min();

        if let Some(min_cost) = min_required_cost
            && self.min_should_match == 0
        {
            return min_cost;
        }

        // No required clauses or minShouldMatch > 0: need should cost.
        let should_cost = self.compute_should_cost();
        min_required_cost.unwrap_or(i64::MAX).min(should_cost)
    }

    fn compute_should_cost(&self) -> i64 {
        let optional_scorers = &self.subs[&Occur::Should];
        let costs: Vec<i64> = optional_scorers.iter().map(|s| s.cost()).collect();
        scorer_util::cost_with_min_should_match(
            &costs,
            optional_scorers.len(),
            self.min_should_match,
        )
    }

    fn get_internal(&mut self, lead_cost: i64) -> io::Result<Box<dyn Scorer + 'a>> {
        // three cases: conjunction, disjunction, or mix
        let lead_cost = lead_cost.min(self.cost());
        let top_level_scoring_clause = self.top_level_scoring_clause;

        // pure conjunction
        if self.subs[&Occur::Should].is_empty() {
            let filter_suppliers = self.subs.remove(&Occur::Filter).unwrap_or_default();
            let must_suppliers = self.subs.remove(&Occur::Must).unwrap_or_default();
            let must_not_suppliers = self.subs.remove(&Occur::MustNot).unwrap_or_default();
            let req_scorer = Self::req(
                filter_suppliers,
                must_suppliers,
                lead_cost,
                top_level_scoring_clause,
                self.score_mode,
            )?;
            return Self::excl(req_scorer, must_not_suppliers, lead_cost);
        }

        // pure disjunction
        if self.subs[&Occur::Filter].is_empty() && self.subs[&Occur::Must].is_empty() {
            let should_suppliers = self.subs.remove(&Occur::Should).unwrap_or_default();
            let must_not_suppliers = self.subs.remove(&Occur::MustNot).unwrap_or_default();
            let opt_scorer = Self::opt(
                should_suppliers,
                self.min_should_match,
                self.score_mode,
                lead_cost,
                top_level_scoring_clause,
            )?;
            return Self::excl(opt_scorer, must_not_suppliers, lead_cost);
        }

        // conjunction-disjunction mix
        let filter_suppliers = self.subs.remove(&Occur::Filter).unwrap_or_default();
        let must_suppliers = self.subs.remove(&Occur::Must).unwrap_or_default();
        let must_not_suppliers = self.subs.remove(&Occur::MustNot).unwrap_or_default();
        let should_suppliers = self.subs.remove(&Occur::Should).unwrap_or_default();

        if self.min_should_match > 0 {
            let req_scorer = Self::excl(
                Self::req(
                    filter_suppliers,
                    must_suppliers,
                    lead_cost,
                    false,
                    self.score_mode,
                )?,
                must_not_suppliers,
                lead_cost,
            )?;
            let opt_scorer = Self::opt(
                should_suppliers,
                self.min_should_match,
                self.score_mode,
                lead_cost,
                false,
            )?;
            return Ok(Box::new(ConjunctionScorer::new(
                Vec::new(),
                vec![req_scorer, opt_scorer],
            )));
        }

        let req_scorer = Self::excl(
            Self::req(
                filter_suppliers,
                must_suppliers,
                lead_cost,
                false,
                self.score_mode,
            )?,
            must_not_suppliers,
            lead_cost,
        )?;
        let opt_scorer = Self::opt(
            should_suppliers,
            self.min_should_match,
            self.score_mode,
            lead_cost,
            false,
        )?;
        Ok(Box::new(ReqOptSumScorer::new(
            req_scorer,
            opt_scorer,
            self.score_mode,
        )?))
    }

    fn boolean_scorer(&mut self) -> io::Result<Option<Box<dyn BulkScorer + 'a>>> {
        let num_optional_clauses = self.subs[&Occur::Should].len();
        let num_must_clauses = self.subs[&Occur::Must].len();
        let num_required_clauses = num_must_clauses + self.subs[&Occur::Filter].len();

        let positive_scorer;
        if num_required_clauses == 0 {
            let cost_threshold: i64 = if self.min_should_match <= 1 {
                // when all clauses are optional, use BooleanScorer aggressively
                -1
            } else {
                // when a minimum number of clauses should match, BooleanScorer is
                // going to score all windows that have at least minNrShouldMatch
                // matches in the window
                (self.max_doc / 3) as i64
            };

            if self.cost() < cost_threshold {
                return Ok(None);
            }

            match self.optional_bulk_scorer()? {
                Some(s) => positive_scorer = s,
                None => return Ok(None),
            }
        } else if num_must_clauses == 0 && num_optional_clauses > 1 && self.min_should_match >= 1 {
            // filteredOptionalBulkScorer: not yet implemented.
            return Ok(None);
        } else if num_required_clauses > 0
            && num_optional_clauses == 0
            && self.min_should_match == 0
        {
            match self.required_bulk_scorer()? {
                Some(s) => positive_scorer = s,
                None => return Ok(None),
            }
        } else {
            return Ok(None);
        }

        let positive_scorer_cost = positive_scorer.cost();

        let mut prohibited: Vec<Box<dyn Scorer + 'a>> = Vec::new();
        for ss in self.subs.get_mut(&Occur::MustNot).unwrap() {
            prohibited.push(ss.get(positive_scorer_cost)?);
        }

        if prohibited.is_empty() {
            Ok(Some(positive_scorer))
        } else if prohibited.len() == 1 {
            let prohibited_scorer = prohibited.remove(0);
            Ok(Some(Box::new(ReqExclBulkScorer::new(
                positive_scorer,
                prohibited_scorer,
            ))))
        } else {
            let prohibited_scorer: Box<dyn Scorer + 'a> = Box::new(DisjunctionSumScorer::new(
                prohibited,
                ScoreMode::CompleteNoScores,
                positive_scorer_cost,
            )?);
            Ok(Some(Box::new(ReqExclBulkScorer::new(
                positive_scorer,
                prohibited_scorer,
            ))))
        }
    }

    /// Returns a `BulkScorer` for the optional (SHOULD) clauses only, or `None` if not
    /// applicable.
    fn optional_bulk_scorer(&mut self) -> io::Result<Option<Box<dyn BulkScorer + 'a>>> {
        let should = self.subs.get_mut(&Occur::Should).unwrap();
        if should.is_empty() {
            return Ok(None);
        }
        // Single SHOULD clause with msm <= 1: delegate directly
        if should.len() == 1 && self.min_should_match <= 1 {
            return Ok(Some(should[0].bulk_scorer()?));
        }

        // TODO: TOP_SCORES + msm <= 1 should use MaxScoreBulkScorer.
        // Fall through to BooleanScorer for now.

        let should_cost = self.compute_should_cost();
        let should = self.subs.get_mut(&Occur::Should).unwrap();
        let mut optional: Vec<Box<dyn Scorer + 'a>> = Vec::new();
        for ss in should {
            optional.push(ss.get(should_cost)?);
        }

        Ok(Some(Box::new(BooleanScorer::new(
            optional,
            self.min_should_match.max(1),
            self.score_mode.needs_scores(),
        )?)))
    }

    fn required_bulk_scorer(&mut self) -> io::Result<Option<Box<dyn BulkScorer + 'a>>> {
        if self.subs[&Occur::Must].len() + self.subs[&Occur::Filter].len() == 0 {
            return Ok(None);
        }
        if self.subs[&Occur::Must].len() + self.subs[&Occur::Filter].len() == 1 {
            if !self.subs[&Occur::Must].is_empty() {
                return Ok(Some(
                    self.subs.get_mut(&Occur::Must).unwrap()[0].bulk_scorer()?,
                ));
            } else {
                let scorer = self.subs.get_mut(&Occur::Filter).unwrap()[0].bulk_scorer()?;
                // TODO: disableScoring if scoreMode.needsScores()
                return Ok(Some(scorer));
            }
        }

        let must_lead_cost = self.subs[&Occur::Must]
            .iter()
            .map(|s| s.cost())
            .min()
            .unwrap_or(i64::MAX);
        let filter_lead_cost = self.subs[&Occur::Filter]
            .iter()
            .map(|s| s.cost())
            .min()
            .unwrap_or(i64::MAX);
        let lead_cost = must_lead_cost.min(filter_lead_cost);

        let mut required_no_scoring: Vec<Box<dyn Scorer + 'a>> = Vec::new();
        for ss in self.subs.get_mut(&Occur::Filter).unwrap() {
            required_no_scoring.push(ss.get(lead_cost)?);
        }
        let mut required_scoring: Vec<Box<dyn Scorer + 'a>> = Vec::new();
        let required_scoring_supplier_size = self.subs[&Occur::Must].len();
        for ss in self.subs.get_mut(&Occur::Must).unwrap() {
            if required_scoring_supplier_size == 1 {
                ss.set_top_level_scoring_clause()?;
            }
            required_scoring.push(ss.get(lead_cost)?);
        }

        // Java lines 387-397: TOP_SCORES with multiple scoring scorers and no two-phase
        // iterators → use BlockMaxConjunctionBulkScorer for dynamic pruning.
        // Note: We don't have twoPhaseIterator in Rust yet, so the two-phase check is
        // trivially true (our TermQuery scorers never have two-phase iterators).
        if self.score_mode == ScoreMode::TopScores && required_scoring.len() > 1 {
            // Java wraps filter scorers as ConstantScoreScorer(0f) here.
            // We don't have ConstantScoreScorer yet — assert no filters for now.
            if !required_no_scoring.is_empty() {
                todo!(
                    "ConstantScoreScorer wrapping for FILTER clauses in BlockMaxConjunctionBulkScorer"
                );
            }
            return Ok(Some(Box::new(BlockMaxConjunctionBulkScorer::new(
                self.max_doc,
                required_scoring,
            ))));
        }

        // Java lines 399-411: non-TOP_SCORES paths (ConjunctionBulkScorer,
        // DenseConjunctionBulkScorer) — not yet implemented, fall through to
        // DefaultBulkScorer wrapping a ConjunctionScorer.

        let conjunction_scorer: Box<dyn Scorer + 'a>;
        if required_no_scoring.len() + required_scoring.len() == 1 {
            if required_scoring.len() == 1 {
                conjunction_scorer = required_scoring.remove(0);
            } else {
                conjunction_scorer = required_no_scoring.remove(0);
                // TODO: if scoreMode.needsScores(), wrap in FilterScorer returning 0
            }
        } else {
            // Rust: ConjunctionScorer::new takes (non-scoring required, scoring) as disjoint vecs
            conjunction_scorer = Box::new(ConjunctionScorer::new(
                required_no_scoring,
                required_scoring,
            ));
            // TOP_SCORES with no scoring — not yet implemented.
        }
        Ok(Some(Box::new(DefaultBulkScorer::new(conjunction_scorer))))
    }

    fn req(
        mut required_no_scoring: Vec<Box<dyn ScorerSupplier<'a> + 'a>>,
        mut required_scoring: Vec<Box<dyn ScorerSupplier<'a> + 'a>>,
        lead_cost: i64,
        top_level_scoring_clause: bool,
        score_mode: ScoreMode,
    ) -> io::Result<Box<dyn Scorer + 'a>> {
        if required_no_scoring.len() + required_scoring.len() == 1 {
            let req = if required_no_scoring.is_empty() {
                required_scoring[0].get(lead_cost)?
            } else {
                required_no_scoring[0].get(lead_cost)?
            };

            if !score_mode.needs_scores() {
                return Ok(req);
            }

            if required_scoring.is_empty() {
                // Scores are needed but we only have a filter clause.
                // TODO: wrap in FilterScorer returning score=0
                return Ok(req);
            }

            return Ok(req);
        }

        let mut required_scorers: Vec<Box<dyn Scorer + 'a>> = Vec::new();
        let mut scoring_scorers: Vec<Box<dyn Scorer + 'a>> = Vec::new();
        for s in &mut required_no_scoring {
            required_scorers.push(s.get(lead_cost)?);
        }
        for s in &mut required_scoring {
            scoring_scorers.push(s.get(lead_cost)?);
        }

        // — not yet implemented, fall through.
        let _ = top_level_scoring_clause;

        // Rust: ConjunctionScorer::new takes (non-scoring required, scoring) as disjoint vecs
        Ok(Box::new(ConjunctionScorer::new(
            required_scorers,
            scoring_scorers,
        )))
    }

    fn excl(
        main: Box<dyn Scorer + 'a>,
        prohibited: Vec<Box<dyn ScorerSupplier<'a> + 'a>>,
        lead_cost: i64,
    ) -> io::Result<Box<dyn Scorer + 'a>> {
        if prohibited.is_empty() {
            Ok(main)
        } else {
            let inner = Self::opt(prohibited, 1, ScoreMode::CompleteNoScores, lead_cost, false)?;
            Ok(Box::new(ReqExclScorer::new(main, inner)))
        }
    }

    /// Creates a Scorer for the optional (SHOULD) clauses.
    fn opt(
        mut optional: Vec<Box<dyn ScorerSupplier<'a> + 'a>>,
        min_should_match: i32,
        score_mode: ScoreMode,
        lead_cost: i64,
        top_level_scoring_clause: bool,
    ) -> io::Result<Box<dyn Scorer + 'a>> {
        if optional.len() == 1 {
            return optional.remove(0).get(lead_cost);
        }

        let mut optional_scorers: Vec<Box<dyn Scorer + 'a>> = Vec::with_capacity(optional.len());
        for mut sup in optional {
            optional_scorers.push(sup.get(lead_cost)?);
        }

        if (score_mode == ScoreMode::TopScores && top_level_scoring_clause) || min_should_match > 1
        {
            return Ok(Box::new(WANDScorer::new(
                optional_scorers,
                min_should_match,
                score_mode,
                lead_cost,
            )?));
        }
        Ok(Box::new(DisjunctionSumScorer::new(
            optional_scorers,
            score_mode,
            lead_cost,
        )?))
    }
}

impl<'a> ScorerSupplier<'a> for BooleanScorerSupplier<'a> {
    fn get(&mut self, lead_cost: i64) -> io::Result<Box<dyn Scorer + 'a>> {
        let scorer = self.get_internal(lead_cost)?;
        // with no scoring clauses — not yet implemented.
        Ok(scorer)
    }

    fn bulk_scorer(&mut self) -> io::Result<Box<dyn BulkScorer + 'a>> {
        let bulk_scorer = self.boolean_scorer()?;
        if let Some(bs) = bulk_scorer {
            // bulk scoring is applicable, use it
            Ok(bs)
        } else {
            // use a Scorer-based impl (BS2)
            let scorer = self.get(i64::MAX)?;
            Ok(Box::new(DefaultBulkScorer::new(scorer)))
        }
    }

    fn cost(&self) -> i64 {
        if self.cost.get() == -1 {
            self.cost.set(self.compute_cost());
        }
        self.cost.get()
    }

    fn set_top_level_scoring_clause(&mut self) -> io::Result<()> {
        self.top_level_scoring_clause = true;
        if self.subs[&Occur::Should].len() + self.subs[&Occur::Must].len() == 1 {
            // If there is a single scoring clause, propagate the call.
            for ss in self.subs.get_mut(&Occur::Should).unwrap() {
                ss.set_top_level_scoring_clause()?;
            }
            for ss in self.subs.get_mut(&Occur::Must).unwrap() {
                ss.set_top_level_scoring_clause()?;
            }
        }
        Ok(())
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
    use crate::search::BooleanQuery;
    use crate::search::doc_id_set_iterator::NO_MORE_DOCS;
    use crate::search::query::Query;
    use crate::search::term_query::TermQuery;
    use crate::store::{MemoryDirectory, SharedDirectory};

    fn build_test_index() -> (SharedDirectory, DirectoryReader) {
        index_docs(&["hello world", "hello there", "world peace"])
    }

    /// Indexes the given strings as docs into a fresh in-memory index under field "content".
    fn index_docs(docs: &[&str]) -> (SharedDirectory, DirectoryReader) {
        let config = IndexWriterConfig::default().num_threads(1);
        let directory: SharedDirectory = MemoryDirectory::create();
        let writer = IndexWriter::new(config, Arc::clone(&directory));
        for content in docs {
            writer
                .add_document(
                    DocumentBuilder::new()
                        .add_field(text("content").value(*content))
                        .build(),
                )
                .unwrap();
        }
        writer.commit().unwrap();
        let reader = DirectoryReader::open(&*directory).unwrap();
        (directory, reader)
    }

    fn tq(term: &'static [u8]) -> Box<dyn Query> {
        Box::new(TermQuery::new("content", term))
    }

    /// Returns the scorer supplier for the (single) leaf, or None.
    fn supplier_for<'a>(
        weight: &'a BooleanWeight,
        reader: &'a DirectoryReader,
    ) -> Option<Box<dyn ScorerSupplier<'a> + 'a>> {
        let leaf = &reader.leaves()[0];
        weight.scorer_supplier(leaf).unwrap()
    }

    /// Drains a Scorer's iterator into a sorted Vec<(doc, score)>.
    fn collect_scorer(scorer: &mut dyn Scorer) -> Vec<(i32, f32)> {
        let mut out = Vec::new();
        loop {
            let doc = scorer.iterator().next_doc().unwrap();
            if doc == NO_MORE_DOCS {
                break;
            }
            let s = scorer.score().unwrap();
            out.push((doc, s));
        }
        out
    }

    // ----------------------------------------------------------------
    // Group A: Weight::scorer_supplier — clause partition + simplifications
    // ----------------------------------------------------------------

    #[test]
    fn test_scorer_supplier_only_must_not_returns_none() {
        let (_dir, reader) = build_test_index();
        let searcher = IndexSearcher::new(&reader);
        let clauses = vec![BooleanClause::new(tq(b"hello"), Occur::MustNot)];
        let weight = BooleanWeight::new(&clauses, &searcher, ScoreMode::Complete, 0, 1.0).unwrap();
        assert!(supplier_for(&weight, &reader).is_none());
    }

    #[test]
    fn test_scorer_supplier_filter_no_postings_returns_none() {
        let (_dir, reader) = build_test_index();
        let searcher = IndexSearcher::new(&reader);
        let clauses = vec![BooleanClause::new(tq(b"missing"), Occur::Filter)];
        let weight = BooleanWeight::new(&clauses, &searcher, ScoreMode::Complete, 0, 1.0).unwrap();
        assert!(supplier_for(&weight, &reader).is_none());
    }

    #[test]
    fn test_scorer_supplier_should_no_postings_does_not_block() {
        let (_dir, reader) = build_test_index();
        let searcher = IndexSearcher::new(&reader);
        // 1 SHOULD with no postings + 1 SHOULD with postings → still produces a supplier.
        let clauses = vec![
            BooleanClause::new(tq(b"missing"), Occur::Should),
            BooleanClause::new(tq(b"hello"), Occur::Should),
        ];
        let weight = BooleanWeight::new(&clauses, &searcher, ScoreMode::Complete, 0, 1.0).unwrap();
        assert!(supplier_for(&weight, &reader).is_some());
    }

    #[test]
    fn test_scorer_supplier_msm_exceeds_should_count_returns_none() {
        let (_dir, reader) = build_test_index();
        let searcher = IndexSearcher::new(&reader);
        // 2 SHOULDs with msm=3 → impossible to match → None.
        let clauses = vec![
            BooleanClause::new(tq(b"hello"), Occur::Should),
            BooleanClause::new(tq(b"world"), Occur::Should),
        ];
        let weight = BooleanWeight::new(&clauses, &searcher, ScoreMode::Complete, 3, 1.0).unwrap();
        assert!(supplier_for(&weight, &reader).is_none());
    }

    #[test]
    fn test_scorer_supplier_shoulds_equal_msm_promotes_to_must() {
        let (_dir, reader) = build_test_index();
        let searcher = IndexSearcher::new(&reader);
        // 2 SHOULDs with msm=2 → simplification promotes them to MUSTs.
        // hello+world both match only doc 0 ("hello world").
        let clauses = vec![
            BooleanClause::new(tq(b"hello"), Occur::Should),
            BooleanClause::new(tq(b"world"), Occur::Should),
        ];
        let weight = BooleanWeight::new(&clauses, &searcher, ScoreMode::Complete, 2, 1.0).unwrap();
        let mut supplier = supplier_for(&weight, &reader).unwrap();
        let mut scorer = supplier.get(i64::MAX).unwrap();
        let docs = collect_scorer(&mut *scorer);
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].0, 0);
    }

    #[test]
    fn test_scorer_supplier_no_scores_with_mixed_clears_optional() {
        let (_dir, reader) = build_test_index();
        let searcher = IndexSearcher::new(&reader);
        // CompleteNoScores + msm=0 + (must + should) → simplification clears shoulds,
        // and the remaining MUST drives matching alone.
        let clauses = vec![
            BooleanClause::new(tq(b"hello"), Occur::Must),
            BooleanClause::new(tq(b"peace"), Occur::Should),
        ];
        let weight =
            BooleanWeight::new(&clauses, &searcher, ScoreMode::CompleteNoScores, 0, 1.0).unwrap();
        let mut supplier = supplier_for(&weight, &reader).unwrap();
        let mut scorer = supplier.get(i64::MAX).unwrap();
        let docs = collect_scorer(&mut *scorer);
        // hello matches docs 0 and 1; peace would have added doc 2, but it's cleared.
        let just_docs: Vec<i32> = docs.iter().map(|(d, _)| *d).collect();
        assert_eq!(just_docs, vec![0, 1]);
    }

    // ----------------------------------------------------------------
    // Group B: Construction invariants
    // ----------------------------------------------------------------

    #[test]
    fn test_negative_msm_returns_error() {
        let (_dir, reader) = build_test_index();
        let searcher = IndexSearcher::new(&reader);
        let clauses = vec![
            BooleanClause::new(tq(b"hello"), Occur::Should),
            BooleanClause::new(tq(b"world"), Occur::Should),
        ];
        let weight = BooleanWeight::new(&clauses, &searcher, ScoreMode::Complete, -1, 1.0).unwrap();
        // The error surfaces during scorer_supplier (BooleanScorerSupplier::new).
        let leaf = &reader.leaves()[0];
        let err = weight.scorer_supplier(leaf).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::Other);
        assert!(err.to_string().contains("minShouldMatch"));
    }

    // ----------------------------------------------------------------
    // Group C: Cost computation
    // ----------------------------------------------------------------

    #[test]
    fn test_cost_pure_conjunction_uses_min_required() {
        // hello matches 2 docs; world matches 2 docs.
        // Pure conjunction with msm=0 → min(2, 2) = 2.
        let (_dir, reader) = build_test_index();
        let searcher = IndexSearcher::new(&reader);
        let clauses = vec![
            BooleanClause::new(tq(b"hello"), Occur::Must),
            BooleanClause::new(tq(b"world"), Occur::Must),
        ];
        let weight = BooleanWeight::new(&clauses, &searcher, ScoreMode::Complete, 0, 1.0).unwrap();
        let supplier = supplier_for(&weight, &reader).unwrap();
        assert_eq!(supplier.cost(), 2);
    }

    #[test]
    fn test_cost_pure_disjunction_uses_should_cost() {
        // hello (cost 2) + peace (cost 1). msm=0 → should cost = sum of (n-msm+1) least-costly = (1+2) = 3.
        let (_dir, reader) = build_test_index();
        let searcher = IndexSearcher::new(&reader);
        let clauses = vec![
            BooleanClause::new(tq(b"hello"), Occur::Should),
            BooleanClause::new(tq(b"peace"), Occur::Should),
        ];
        let weight = BooleanWeight::new(&clauses, &searcher, ScoreMode::Complete, 0, 1.0).unwrap();
        let supplier = supplier_for(&weight, &reader).unwrap();
        // Cost includes both (msm=0 → all clauses contribute).
        assert_eq!(supplier.cost(), 3);
    }

    #[test]
    fn test_cost_mixed_with_msm_zero_uses_required_min() {
        // MUST hello (cost 2) + SHOULD peace (cost 1), msm=0 → required min = 2.
        let (_dir, reader) = build_test_index();
        let searcher = IndexSearcher::new(&reader);
        let clauses = vec![
            BooleanClause::new(tq(b"hello"), Occur::Must),
            BooleanClause::new(tq(b"peace"), Occur::Should),
        ];
        let weight = BooleanWeight::new(&clauses, &searcher, ScoreMode::Complete, 0, 1.0).unwrap();
        let supplier = supplier_for(&weight, &reader).unwrap();
        assert_eq!(supplier.cost(), 2);
    }

    // ----------------------------------------------------------------
    // Group D: get_internal Scorer-level dispatch
    // ----------------------------------------------------------------

    #[test]
    fn test_get_pure_conjunction_returns_intersection() {
        let (_dir, reader) = build_test_index();
        let searcher = IndexSearcher::new(&reader);
        let clauses = vec![
            BooleanClause::new(tq(b"hello"), Occur::Must),
            BooleanClause::new(tq(b"world"), Occur::Must),
        ];
        let weight = BooleanWeight::new(&clauses, &searcher, ScoreMode::Complete, 0, 1.0).unwrap();
        let mut supplier = supplier_for(&weight, &reader).unwrap();
        let mut scorer = supplier.get(i64::MAX).unwrap();
        let docs: Vec<i32> = collect_scorer(&mut *scorer)
            .into_iter()
            .map(|(d, _)| d)
            .collect();
        assert_eq!(docs, vec![0]);
    }

    #[test]
    fn test_get_pure_disjunction_three_should_returns_union() {
        // 3 SHOULDs: hello (0,1) + peace (2) + there (1) → union {0,1,2}
        let (_dir, reader) = build_test_index();
        let searcher = IndexSearcher::new(&reader);
        let clauses = vec![
            BooleanClause::new(tq(b"hello"), Occur::Should),
            BooleanClause::new(tq(b"peace"), Occur::Should),
            BooleanClause::new(tq(b"there"), Occur::Should),
        ];
        let weight = BooleanWeight::new(&clauses, &searcher, ScoreMode::Complete, 0, 1.0).unwrap();
        let mut supplier = supplier_for(&weight, &reader).unwrap();
        let mut scorer = supplier.get(i64::MAX).unwrap();
        let docs: Vec<i32> = collect_scorer(&mut *scorer)
            .into_iter()
            .map(|(d, _)| d)
            .collect();
        assert_eq!(docs, vec![0, 1, 2]);
    }

    #[test]
    fn test_get_mixed_must_should_msm_zero_intersects_with_optional_score_bonus() {
        // MUST hello (0, 1) + SHOULD world (0, 2). Intersection-required = {0,1}.
        // doc 0 should score higher than doc 1 (world matches doc 0, not doc 1).
        let (_dir, reader) = build_test_index();
        let searcher = IndexSearcher::new(&reader);
        let clauses = vec![
            BooleanClause::new(tq(b"hello"), Occur::Must),
            BooleanClause::new(tq(b"world"), Occur::Should),
        ];
        let weight = BooleanWeight::new(&clauses, &searcher, ScoreMode::Complete, 0, 1.0).unwrap();
        let mut supplier = supplier_for(&weight, &reader).unwrap();
        let mut scorer = supplier.get(i64::MAX).unwrap();
        let pairs = collect_scorer(&mut *scorer);
        let docs: Vec<i32> = pairs.iter().map(|(d, _)| *d).collect();
        assert_eq!(docs, vec![0, 1]);
        // Doc 0 has world matching → higher score than doc 1.
        assert!(pairs[0].1 > pairs[1].1);
    }

    #[test]
    fn test_get_pure_disjunction_msm_one_explicit() {
        // 3 SHOULDs with msm=1 → at least one clause must match. With our corpus,
        // every doc matches at least one of {hello, world, peace}, so all 3 docs.
        let (_dir, reader) = build_test_index();
        let searcher = IndexSearcher::new(&reader);
        let clauses = vec![
            BooleanClause::new(tq(b"hello"), Occur::Should),
            BooleanClause::new(tq(b"world"), Occur::Should),
            BooleanClause::new(tq(b"peace"), Occur::Should),
        ];
        let weight = BooleanWeight::new(&clauses, &searcher, ScoreMode::Complete, 1, 1.0).unwrap();
        let mut supplier = supplier_for(&weight, &reader).unwrap();
        let mut scorer = supplier.get(i64::MAX).unwrap();
        let docs: Vec<i32> = collect_scorer(&mut *scorer)
            .into_iter()
            .map(|(d, _)| d)
            .collect();
        assert_eq!(docs, vec![0, 1, 2]);
    }

    #[test]
    fn test_get_pure_disjunction_msm_two_uses_wand() {
        // Phase 7: msm > 1 routes to WANDScorer.
        // hello (0,1), world (0,2), peace (2). msm=2 → docs matching ≥ 2 clauses:
        // doc 0 → hello+world (2) ✓; doc 2 → world+peace (2) ✓.
        let (_dir, reader) = build_test_index();
        let searcher = IndexSearcher::new(&reader);
        let clauses = vec![
            BooleanClause::new(tq(b"hello"), Occur::Should),
            BooleanClause::new(tq(b"world"), Occur::Should),
            BooleanClause::new(tq(b"peace"), Occur::Should),
        ];
        let weight = BooleanWeight::new(&clauses, &searcher, ScoreMode::Complete, 2, 1.0).unwrap();
        let mut supplier = supplier_for(&weight, &reader).unwrap();
        let mut scorer = supplier.get(i64::MAX).unwrap();
        let docs: Vec<i32> = collect_scorer(&mut *scorer)
            .into_iter()
            .map(|(d, _)| d)
            .collect();
        assert_eq!(docs, vec![0, 2]);
    }

    #[test]
    fn test_get_mixed_must_should_msm_one_uses_disjunction_in_opt() {
        // Mixed MUST + 2 SHOULDs with msm=1: opt() returns DisjunctionSumScorer (msm <=1),
        // and get_internal wraps as ConjunctionScorer([req, opt]).
        // hello (0,1) MUST + (world(0,2), peace(2)) msm=1.
        // Required: hello → {0, 1}.
        // Optional must contribute ≥ 1 → docs in (world ∪ peace) = {0, 2}.
        // Intersection: {0}.
        let (_dir, reader) = build_test_index();
        let searcher = IndexSearcher::new(&reader);
        let clauses = vec![
            BooleanClause::new(tq(b"hello"), Occur::Must),
            BooleanClause::new(tq(b"world"), Occur::Should),
            BooleanClause::new(tq(b"peace"), Occur::Should),
        ];
        let weight = BooleanWeight::new(&clauses, &searcher, ScoreMode::Complete, 1, 1.0).unwrap();
        let mut supplier = supplier_for(&weight, &reader).unwrap();
        let mut scorer = supplier.get(i64::MAX).unwrap();
        let docs: Vec<i32> = collect_scorer(&mut *scorer)
            .into_iter()
            .map(|(d, _)| d)
            .collect();
        assert_eq!(docs, vec![0]);
    }

    #[test]
    fn test_get_mixed_must_should_msm_two_uses_wand_in_opt() {
        // Mixed MUST + 3 SHOULDs with msm=2: opt() routes to WANDScorer.
        // MUST hello (0, 1).
        // SHOULDs: world (0, 2), peace (2), there (1). msm=2 → ≥2 of these.
        // doc 0: world only → 1. doc 1: there only → 1. doc 2: world+peace → 2 ✓.
        // Required is hello (0,1). Required ∩ msm>=2 of optional: {0,1} ∩ {2} = {}
        let (_dir, reader) = build_test_index();
        let searcher = IndexSearcher::new(&reader);
        let clauses = vec![
            BooleanClause::new(tq(b"hello"), Occur::Must),
            BooleanClause::new(tq(b"world"), Occur::Should),
            BooleanClause::new(tq(b"peace"), Occur::Should),
            BooleanClause::new(tq(b"there"), Occur::Should),
        ];
        let weight = BooleanWeight::new(&clauses, &searcher, ScoreMode::Complete, 2, 1.0).unwrap();
        let mut supplier = supplier_for(&weight, &reader).unwrap();
        let mut scorer = supplier.get(i64::MAX).unwrap();
        let docs: Vec<i32> = collect_scorer(&mut *scorer)
            .into_iter()
            .map(|(d, _)| d)
            .collect();
        assert!(docs.is_empty());
    }

    // ----------------------------------------------------------------
    // Group E: bulk_scorer dispatch (BulkScorer level)
    // ----------------------------------------------------------------

    #[test]
    fn test_bulk_scorer_pure_must_returns_some() {
        let (_dir, reader) = build_test_index();
        let searcher = IndexSearcher::new(&reader);
        let clauses = vec![
            BooleanClause::new(tq(b"hello"), Occur::Must),
            BooleanClause::new(tq(b"world"), Occur::Must),
        ];
        let weight = BooleanWeight::new(&clauses, &searcher, ScoreMode::Complete, 0, 1.0).unwrap();
        let mut supplier = supplier_for(&weight, &reader).unwrap();
        let _ = supplier.bulk_scorer().unwrap();
    }

    #[test]
    fn test_bulk_scorer_pure_should_msm_one() {
        let (_dir, reader) = build_test_index();
        let searcher = IndexSearcher::new(&reader);
        let clauses = vec![
            BooleanClause::new(tq(b"hello"), Occur::Should),
            BooleanClause::new(tq(b"world"), Occur::Should),
            BooleanClause::new(tq(b"peace"), Occur::Should),
        ];
        let weight = BooleanWeight::new(&clauses, &searcher, ScoreMode::Complete, 1, 1.0).unwrap();
        let mut supplier = supplier_for(&weight, &reader).unwrap();
        let _ = supplier.bulk_scorer().unwrap();
    }

    #[test]
    fn test_get_must_with_single_must_not_via_scorer_path() {
        // Phase 5: excl() is now wired. Test via Scorer-level get(), which
        // exercises the excl() → ReqExclScorer path directly.
        // hello matches (0, 1); world matches (0, 2). hello & !world → doc 1.
        let (_dir, reader) = build_test_index();
        let searcher = IndexSearcher::new(&reader);
        let clauses = vec![
            BooleanClause::new(tq(b"hello"), Occur::Must),
            BooleanClause::new(tq(b"world"), Occur::MustNot),
        ];
        let weight = BooleanWeight::new(&clauses, &searcher, ScoreMode::Complete, 0, 1.0).unwrap();
        let mut supplier = supplier_for(&weight, &reader).unwrap();
        let mut scorer = supplier.get(i64::MAX).unwrap();
        let docs: Vec<i32> = collect_scorer(&mut *scorer)
            .into_iter()
            .map(|(d, _)| d)
            .collect();
        assert_eq!(docs, vec![1]);
    }

    #[test]
    fn test_get_must_with_multi_must_not_via_scorer_path() {
        // Phase 5: excl() now handles >1 prohibited via DisjunctionSumScorer + ReqExclScorer.
        // Corpus: doc 0 "hello world", doc 1 "hello there", doc 2 "world peace".
        // MUST hello (0, 1); MUST_NOT world (0, 2); MUST_NOT there (1).
        // Result: hello \ {world ∪ there} = {0, 1} \ {0, 2, 1} = {}.
        let (_dir, reader) = build_test_index();
        let searcher = IndexSearcher::new(&reader);
        let clauses = vec![
            BooleanClause::new(tq(b"hello"), Occur::Must),
            BooleanClause::new(tq(b"world"), Occur::MustNot),
            BooleanClause::new(tq(b"there"), Occur::MustNot),
        ];
        let weight = BooleanWeight::new(&clauses, &searcher, ScoreMode::Complete, 0, 1.0).unwrap();
        let mut supplier = supplier_for(&weight, &reader).unwrap();
        let mut scorer = supplier.get(i64::MAX).unwrap();
        let docs: Vec<i32> = collect_scorer(&mut *scorer)
            .into_iter()
            .map(|(d, _)| d)
            .collect();
        assert!(docs.is_empty());
    }

    #[test]
    fn test_bulk_scorer_must_with_multi_must_not() {
        // Phase 6: bulk path now wraps prohibited.len() > 1 in DisjunctionSumScorer
        // and uses ReqExclBulkScorer. Verify via searcher.search end-to-end.
        // hello matches (0, 1); world matches (0, 2); peace matches (2).
        // hello & !world & !peace → {0, 1} \ {0, 2} = {1}.
        let (_dir, reader) = build_test_index();
        let searcher = IndexSearcher::new(&reader);
        let mut b = BooleanQuery::builder();
        b.add_query(tq(b"hello"), Occur::Must);
        b.add_query(tq(b"world"), Occur::MustNot);
        b.add_query(tq(b"peace"), Occur::MustNot);
        let query = b.build();

        let top = searcher.search(&query, 10).unwrap();
        let docs: Vec<i32> = top.score_docs.iter().map(|sd| sd.doc).collect();
        assert_eq!(docs, vec![1]);
    }

    #[test]
    fn test_get_must_with_multi_must_not_partial_exclusion() {
        // MUST hello (0, 1); MUST_NOT world (0, 2); MUST_NOT peace (2).
        // hello \ {world ∪ peace} = {0, 1} \ {0, 2} = {1}.
        let (_dir, reader) = build_test_index();
        let searcher = IndexSearcher::new(&reader);
        let clauses = vec![
            BooleanClause::new(tq(b"hello"), Occur::Must),
            BooleanClause::new(tq(b"world"), Occur::MustNot),
            BooleanClause::new(tq(b"peace"), Occur::MustNot),
        ];
        let weight = BooleanWeight::new(&clauses, &searcher, ScoreMode::Complete, 0, 1.0).unwrap();
        let mut supplier = supplier_for(&weight, &reader).unwrap();
        let mut scorer = supplier.get(i64::MAX).unwrap();
        let docs: Vec<i32> = collect_scorer(&mut *scorer)
            .into_iter()
            .map(|(d, _)| d)
            .collect();
        assert_eq!(docs, vec![1]);
    }

    #[test]
    fn test_bulk_scorer_must_with_single_must_not() {
        let (_dir, reader) = build_test_index();
        let searcher = IndexSearcher::new(&reader);
        // MUST hello + MUST_NOT world → docs containing hello but not world = doc 1
        let clauses = vec![
            BooleanClause::new(tq(b"hello"), Occur::Must),
            BooleanClause::new(tq(b"world"), Occur::MustNot),
        ];
        let query = {
            let mut b = BooleanQuery::builder();
            b.add_query(tq(b"hello"), Occur::Must);
            b.add_query(tq(b"world"), Occur::MustNot);
            b.build()
        };
        let weight = BooleanWeight::new(&clauses, &searcher, ScoreMode::Complete, 0, 1.0).unwrap();
        let _ = weight; // verify construct
        let top = searcher.search(&query, 10).unwrap();
        let docs: Vec<i32> = top.score_docs.iter().map(|sd| sd.doc).collect();
        assert_eq!(docs, vec![1]);
    }

    // ----------------------------------------------------------------
    // Group F: required_bulk_scorer
    // ----------------------------------------------------------------

    #[test]
    fn test_required_bulk_scorer_top_scores_multi_must_works() {
        let (_dir, reader) = build_test_index();
        let searcher = IndexSearcher::new(&reader);
        let clauses = vec![
            BooleanClause::new(tq(b"hello"), Occur::Must),
            BooleanClause::new(tq(b"world"), Occur::Must),
        ];
        let weight = BooleanWeight::new(&clauses, &searcher, ScoreMode::TopScores, 0, 1.0).unwrap();
        let mut supplier = supplier_for(&weight, &reader).unwrap();
        let _ = supplier.bulk_scorer().unwrap();
    }

    #[test]
    fn test_required_bulk_scorer_single_must_delegates() {
        let (_dir, reader) = build_test_index();
        let searcher = IndexSearcher::new(&reader);
        let clauses = vec![BooleanClause::new(tq(b"hello"), Occur::Must)];
        let weight = BooleanWeight::new(&clauses, &searcher, ScoreMode::Complete, 0, 1.0).unwrap();
        let mut supplier = supplier_for(&weight, &reader).unwrap();
        let _ = supplier.bulk_scorer().unwrap();
    }

    // ----------------------------------------------------------------
    // Group G: optional_bulk_scorer
    // ----------------------------------------------------------------

    #[test]
    fn test_optional_bulk_scorer_single_should_delegates() {
        let (_dir, reader) = build_test_index();
        let searcher = IndexSearcher::new(&reader);
        let clauses = vec![BooleanClause::new(tq(b"hello"), Occur::Should)];
        let weight = BooleanWeight::new(&clauses, &searcher, ScoreMode::Complete, 0, 1.0).unwrap();
        let mut supplier = supplier_for(&weight, &reader).unwrap();
        let _ = supplier.bulk_scorer().unwrap();
    }

    #[test]
    fn test_optional_bulk_scorer_multi_should_uses_boolean_scorer() {
        let (_dir, reader) = build_test_index();
        let searcher = IndexSearcher::new(&reader);
        let clauses = vec![
            BooleanClause::new(tq(b"hello"), Occur::Should),
            BooleanClause::new(tq(b"world"), Occur::Should),
        ];
        let weight = BooleanWeight::new(&clauses, &searcher, ScoreMode::Complete, 0, 1.0).unwrap();
        let mut supplier = supplier_for(&weight, &reader).unwrap();
        let _ = supplier.bulk_scorer().unwrap();
    }

    // ----------------------------------------------------------------
    // Group H: set_top_level_scoring_clause
    // ----------------------------------------------------------------

    #[test]
    fn test_set_top_level_scoring_clause_does_not_panic() {
        let (_dir, reader) = build_test_index();
        let searcher = IndexSearcher::new(&reader);
        let clauses = vec![BooleanClause::new(tq(b"hello"), Occur::Should)];
        let weight = BooleanWeight::new(&clauses, &searcher, ScoreMode::TopScores, 0, 1.0).unwrap();
        let mut supplier = supplier_for(&weight, &reader).unwrap();
        supplier.set_top_level_scoring_clause().unwrap();
    }

    #[test]
    fn test_set_top_level_scoring_clause_results_unchanged_with_multi() {
        let (_dir, reader) = build_test_index();
        let searcher = IndexSearcher::new(&reader);
        let clauses = vec![
            BooleanClause::new(tq(b"hello"), Occur::Must),
            BooleanClause::new(tq(b"world"), Occur::Must),
        ];
        let weight = BooleanWeight::new(&clauses, &searcher, ScoreMode::TopScores, 0, 1.0).unwrap();
        let mut supplier = supplier_for(&weight, &reader).unwrap();
        supplier.set_top_level_scoring_clause().unwrap();
        let mut scorer = supplier.get(i64::MAX).unwrap();
        let docs: Vec<i32> = collect_scorer(&mut *scorer)
            .into_iter()
            .map(|(d, _)| d)
            .collect();
        assert_eq!(docs, vec![0]);
    }

    // ----------------------------------------------------------------
    // Existing tests (kept for regression coverage)
    // ----------------------------------------------------------------

    #[test]
    fn test_boolean_weight_required_clause_no_match() {
        let (_dir, reader) = build_test_index();
        let searcher = IndexSearcher::new(&reader);

        let clauses = vec![
            BooleanClause::new(Box::new(TermQuery::new("content", b"hello")), Occur::Must),
            BooleanClause::new(
                Box::new(TermQuery::new("content", b"nonexistent")),
                Occur::Must,
            ),
        ];
        let weight = BooleanWeight::new(&clauses, &searcher, ScoreMode::Complete, 0, 1.0).unwrap();

        // "nonexistent" has no postings, so scorer_supplier should return None
        for leaf in reader.leaves() {
            let supplier = weight.scorer_supplier(leaf).unwrap();
            assert!(
                supplier.is_none(),
                "expected None when a MUST clause has no matches"
            );
        }
    }

    #[test]
    fn test_boolean_weight_two_must_clauses() {
        let (_dir, reader) = build_test_index();
        let searcher = IndexSearcher::new(&reader);

        let clauses = vec![
            BooleanClause::new(Box::new(TermQuery::new("content", b"hello")), Occur::Must),
            BooleanClause::new(Box::new(TermQuery::new("content", b"world")), Occur::Must),
        ];
        let weight = BooleanWeight::new(&clauses, &searcher, ScoreMode::Complete, 0, 1.0).unwrap();

        for leaf in reader.leaves() {
            let supplier = weight.scorer_supplier(leaf).unwrap();
            assert!(
                supplier.is_some(),
                "expected Some when both MUST clauses have matches"
            );
        }
    }
}
