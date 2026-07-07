// SPDX-License-Identifier: Apache-2.0

//! Integration tests for BooleanQuery, cross-validated against Java Lucene 10.3.2.
//!
//! Uses the golden-docs corpus (15 documents from `testdata/golden-docs/`), indexed with
//! field name "contents" using `text_field` (DOCS_AND_FREQS_AND_POSITIONS, tokenized,
//! no term vectors). Files are indexed in sorted filename order for deterministic doc IDs.
//!
//! Expected results were obtained by indexing the same documents in the same order with
//! Java Lucene 10.3.2 (StandardAnalyzer, BM25Similarity default k1=1.2 b=0.75) and
//! running the same BooleanQuery with two MUST TermQuery clauses.

use std::fs;
use std::path::Path;
use std::sync::Arc;

use assertables::*;
use bearing::document::DocumentBuilder;
use bearing::document::StoredValue;
use bearing::index::config::IndexWriterConfig;
use bearing::index::directory_reader::DirectoryReader;
use bearing::index::field::{keyword, text};
use bearing::index::writer::IndexWriter;
use bearing::search::*;
use bearing::store::{MemoryDirectory, SharedDirectory};

/// Indexes golden-docs into an in-memory directory, returning the directory and reader.
/// Files are indexed in sorted filename order to match the Java indexing order.
fn build_golden_docs_index() -> (SharedDirectory, DirectoryReader) {
    let config = IndexWriterConfig::default().num_threads(1);
    let directory: SharedDirectory = MemoryDirectory::create();
    let writer = IndexWriter::new(config, Arc::clone(&directory));

    let docs_dir = Path::new("testdata/golden-docs");
    let mut paths: Vec<_> = fs::read_dir(docs_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "txt"))
        .map(|e| e.path())
        .collect();
    paths.sort();

    for path in &paths {
        let contents = fs::read_to_string(path).unwrap();
        let stem = path.file_stem().unwrap().to_string_lossy().into_owned();
        let doc = DocumentBuilder::new()
            .add_field(text("contents").value(contents.as_str()))
            .add_field(keyword("filename").stored().value(stem))
            .build();
        writer.add_document(doc).unwrap();
    }

    writer.commit().unwrap();

    let reader = DirectoryReader::open(&*directory).unwrap();
    (directory, reader)
}

/// Loads the stored "filename" field for every doc into a Vec indexed by doc_id.
/// Used by tests that derived expected results from Java QueryIndex output:
/// Java assigns doc IDs in `Files.walkFileTree` (filesystem) order while the
/// Rust test indexes in sorted order, so doc IDs differ. Filenames are stable.
fn load_filenames_by_doc(directory: &SharedDirectory) -> Vec<String> {
    use bearing::index::segment_infos;
    use bearing::index::segment_reader::SegmentReader;

    let files = directory.list_all().unwrap();
    let segments_file = segment_infos::get_last_commit_segments_file_name(&files).unwrap();
    let infos = segment_infos::read(&**directory, &segments_file).unwrap();
    assert_eq!(
        infos.segments.len(),
        1,
        "load_filenames_by_doc assumes a single segment"
    );
    let seg = &infos.segments[0];
    let mut sr = SegmentReader::open(&**directory, &seg.name, &seg.id).unwrap();
    let max_doc = sr.max_doc() as u32;
    let field_no = sr
        .field_infos()
        .field_info_by_name("filename")
        .expect("filename field missing")
        .number();
    let sfr = sr
        .get_fields_reader()
        .expect("stored fields reader missing");

    let mut out = Vec::with_capacity(max_doc as usize);
    for doc in 0..max_doc {
        let fields = sfr.document(doc).unwrap();
        let value = fields
            .iter()
            .find(|f| f.field_number == field_no)
            .map(|f| match &f.value {
                StoredValue::String(s) => s.clone(),
                other => panic!("filename has non-string value: {other:?}"),
            })
            .expect("filename not stored on doc");
        out.push(value);
    }
    out
}

/// Helper to build a two-MUST boolean query.
fn must_must_query(field: &str, term1: &[u8], term2: &[u8]) -> BooleanQuery {
    let mut builder = BooleanQuery::builder();
    builder.add_query(Box::new(TermQuery::new(field, term1)), Occur::Must);
    builder.add_query(Box::new(TermQuery::new(field, term2)), Occur::Must);
    builder.build()
}

// -------------------------------------------------------------------------
// Query 1: +algorithms +data → 4 hits
// Java results:
//   doc=11 score=0.9744643
//   doc=0  score=0.8020670
//   doc=14 score=0.8000477
//   doc=3  score=0.6710463
// -------------------------------------------------------------------------

#[test]
fn test_boolean_must_algorithms_data() {
    let (_dir, reader) = build_golden_docs_index();
    let searcher = IndexSearcher::new(&reader);
    let query = must_must_query("contents", b"algorithms", b"data");

    let top_docs = searcher.search(&query, 10).unwrap();

    assert_eq!(top_docs.total_hits.value, 4);
    assert_eq!(top_docs.score_docs.len(), 4);

    assert_eq!(top_docs.score_docs[0].doc, 11);
    assert_in_delta!(top_docs.score_docs[0].score, 0.9744643_f32, 1e-5);

    assert_eq!(top_docs.score_docs[1].doc, 0);
    assert_in_delta!(top_docs.score_docs[1].score, 0.802067_f32, 1e-5);

    assert_eq!(top_docs.score_docs[2].doc, 14);
    assert_in_delta!(top_docs.score_docs[2].score, 0.8000477_f32, 1e-5);

    assert_eq!(top_docs.score_docs[3].doc, 3);
    assert_in_delta!(top_docs.score_docs[3].score, 0.6710463_f32, 1e-5);
}

// -------------------------------------------------------------------------
// Query 2: +distributed +systems → 6 hits
// Java results:
//   doc=13 score=0.6466637
//   doc=2  score=0.5788049
//   doc=7  score=0.5369343
//   doc=8  score=0.5369343
//   doc=6  score=0.5074845
//   doc=0  score=0.4957356
// -------------------------------------------------------------------------

#[test]
fn test_boolean_must_distributed_systems() {
    let (_dir, reader) = build_golden_docs_index();
    let searcher = IndexSearcher::new(&reader);
    let query = must_must_query("contents", b"distributed", b"systems");

    let top_docs = searcher.search(&query, 10).unwrap();

    assert_eq!(top_docs.total_hits.value, 6);
    assert_eq!(top_docs.score_docs.len(), 6);

    assert_eq!(top_docs.score_docs[0].doc, 13);
    assert_in_delta!(top_docs.score_docs[0].score, 0.6466637_f32, 1e-5);

    assert_eq!(top_docs.score_docs[1].doc, 2);
    assert_in_delta!(top_docs.score_docs[1].score, 0.5788049_f32, 1e-5);

    assert_eq!(top_docs.score_docs[2].doc, 7);
    assert_in_delta!(top_docs.score_docs[2].score, 0.5369343_f32, 1e-5);

    assert_eq!(top_docs.score_docs[3].doc, 8);
    assert_in_delta!(top_docs.score_docs[3].score, 0.5369343_f32, 1e-5);

    assert_eq!(top_docs.score_docs[4].doc, 6);
    assert_in_delta!(top_docs.score_docs[4].score, 0.5074845_f32, 1e-5);

    assert_eq!(top_docs.score_docs[5].doc, 0);
    assert_in_delta!(top_docs.score_docs[5].score, 0.4957356_f32, 1e-5);
}

// -------------------------------------------------------------------------
// Query 3: +memory +performance → 1 hit
// Java results:
//   doc=0 score=1.1177642
// -------------------------------------------------------------------------

#[test]
fn test_boolean_must_memory_performance() {
    let (_dir, reader) = build_golden_docs_index();
    let searcher = IndexSearcher::new(&reader);
    let query = must_must_query("contents", b"memory", b"performance");

    let top_docs = searcher.search(&query, 10).unwrap();

    assert_eq!(top_docs.total_hits.value, 1);
    assert_eq!(top_docs.score_docs.len(), 1);

    assert_eq!(top_docs.score_docs[0].doc, 0);
    assert_in_delta!(top_docs.score_docs[0].score, 1.1177642_f32, 1e-5);
}

// -------------------------------------------------------------------------
// Query 4: +quantum +computing → 1 hit
// Java results:
//   doc=9 score=1.7729266
// -------------------------------------------------------------------------

#[test]
fn test_boolean_must_quantum_computing() {
    let (_dir, reader) = build_golden_docs_index();
    let searcher = IndexSearcher::new(&reader);
    let query = must_must_query("contents", b"quantum", b"computing");

    let top_docs = searcher.search(&query, 10).unwrap();

    assert_eq!(top_docs.total_hits.value, 1);
    assert_eq!(top_docs.score_docs.len(), 1);

    assert_eq!(top_docs.score_docs[0].doc, 9);
    assert_in_delta!(top_docs.score_docs[0].score, 1.7729266_f32, 1e-5);
}

// -------------------------------------------------------------------------
// Query 5: +nonexistent1 +nonexistent2 → 0 hits
// -------------------------------------------------------------------------

#[test]
fn test_boolean_must_nonexistent_terms() {
    let (_dir, reader) = build_golden_docs_index();
    let searcher = IndexSearcher::new(&reader);
    let query = must_must_query("contents", b"nonexistent1", b"nonexistent2");

    let top_docs = searcher.search(&query, 10).unwrap();

    assert_eq!(top_docs.total_hits.value, 0);
    assert_is_empty!(top_docs.score_docs);
}

// -------------------------------------------------------------------------
// SHOULD queries — pure disjunction
// Cross-validated against Java Lucene 10.3.2. Java index built with sorted
// filenames, StandardAnalyzer(EMPTY_SET), text_field("contents").
// Java queries via QueryParser default operator (OR = SHOULD).
// -------------------------------------------------------------------------

/// Helper to build a two-SHOULD boolean query.
fn should_should_query(field: &str, term1: &[u8], term2: &[u8]) -> BooleanQuery {
    let mut builder = BooleanQuery::builder();
    builder.add_query(Box::new(TermQuery::new(field, term1)), Occur::Should);
    builder.add_query(Box::new(TermQuery::new(field, term2)), Occur::Should);
    builder.build()
}

// -------------------------------------------------------------------------
// SHOULD Query 1: algorithms data → 10 hits
// Java results:
//   doc=11  score=0.9744643
//   doc=0   score=0.8020670
//   doc=14  score=0.8000477
//   doc=3   score=0.6710463
//   doc=5   score=0.2621497
//   doc=1   score=0.2506270
//   doc=10  score=0.2121821
//   doc=12  score=0.1924969
//   doc=6   score=0.1707188
//   doc=9   score=0.1577401
// -------------------------------------------------------------------------

#[test]
fn test_boolean_should_algorithms_data() {
    let (_dir, reader) = build_golden_docs_index();
    let searcher = IndexSearcher::new(&reader);
    let query = should_should_query("contents", b"algorithms", b"data");

    let top_docs = searcher.search(&query, 15).unwrap();

    assert_eq!(top_docs.total_hits.value, 10);
    assert_eq!(top_docs.score_docs.len(), 10);

    let expected = [
        (11, 0.9744643_f32),
        (0, 0.802067),
        (14, 0.8000477),
        (3, 0.6710463),
        (5, 0.2621497),
        (1, 0.250627),
        (10, 0.2121821),
        (12, 0.1924969),
        (6, 0.1707188),
        (9, 0.1577401),
    ];
    for (i, &(doc, score)) in expected.iter().enumerate() {
        assert_eq!(top_docs.score_docs[i].doc, doc);
        assert_in_delta!(top_docs.score_docs[i].score, score, 1e-5);
    }
}

// -------------------------------------------------------------------------
// SHOULD Query 2: distributed systems → 12 hits
// Java results:
//   doc=13  score=0.6466637
//   doc=2   score=0.5788049
//   doc=7   score=0.5369343
//   doc=8   score=0.5369343
//   doc=6   score=0.5074845
//   doc=0   score=0.4957356
//   doc=12  score=0.1548606
//   doc=3   score=0.1403393
//   doc=4   score=0.1385187
//   doc=9   score=0.1345176
//   doc=10  score=0.1243533
//   doc=11  score=0.0942376
// -------------------------------------------------------------------------

#[test]
fn test_boolean_should_distributed_systems() {
    let (_dir, reader) = build_golden_docs_index();
    let searcher = IndexSearcher::new(&reader);
    let query = should_should_query("contents", b"distributed", b"systems");

    let top_docs = searcher.search(&query, 15).unwrap();

    assert_eq!(top_docs.total_hits.value, 12);
    assert_eq!(top_docs.score_docs.len(), 12);

    let expected = [
        (13, 0.6466637_f32),
        (2, 0.5788049),
        (7, 0.5369343),
        (8, 0.5369343),
        (6, 0.5074845),
        (0, 0.4957356),
        (12, 0.1548606),
        (3, 0.1403393),
        (4, 0.1385187),
        (9, 0.1345176),
        (10, 0.1243533),
        (11, 0.0942376),
    ];
    for (i, &(doc, score)) in expected.iter().enumerate() {
        assert_eq!(top_docs.score_docs[i].doc, doc);
        assert_in_delta!(top_docs.score_docs[i].score, score, 1e-5);
    }
}

// -------------------------------------------------------------------------
// SHOULD Query 3: memory performance → 7 hits
// Java results:
//   doc=0   score=1.1177642
//   doc=8   score=0.7110608
//   doc=1   score=0.6353776
//   doc=9   score=0.5691590
//   doc=14  score=0.5055991
//   doc=2   score=0.4995965
//   doc=12  score=0.4880089
// -------------------------------------------------------------------------

#[test]
fn test_boolean_should_memory_performance() {
    let (_dir, reader) = build_golden_docs_index();
    let searcher = IndexSearcher::new(&reader);
    let query = should_should_query("contents", b"memory", b"performance");

    let top_docs = searcher.search(&query, 15).unwrap();

    assert_eq!(top_docs.total_hits.value, 7);
    assert_eq!(top_docs.score_docs.len(), 7);

    let expected = [
        (0, 1.1177642_f32),
        (8, 0.7110608),
        (1, 0.6353776),
        (9, 0.569159),
        (14, 0.5055991),
        (2, 0.4995965),
        (12, 0.4880089),
    ];
    for (i, &(doc, score)) in expected.iter().enumerate() {
        assert_eq!(top_docs.score_docs[i].doc, doc);
        assert_in_delta!(top_docs.score_docs[i].score, score, 1e-5);
    }
}

// -------------------------------------------------------------------------
// MUST_NOT queries — exclusion
// Cross-validated against Java Lucene 10.3.2. Same index setup as SHOULD tests.
// Java queries via QueryParser: `+word1 -word2` and `word1 word2 -word3`.
// -------------------------------------------------------------------------

/// Helper to build a MUST + single MUST_NOT boolean query.
fn must_must_not_query(field: &str, must_term: &[u8], not_term: &[u8]) -> BooleanQuery {
    let mut builder = BooleanQuery::builder();
    builder.add_query(Box::new(TermQuery::new(field, must_term)), Occur::Must);
    builder.add_query(Box::new(TermQuery::new(field, not_term)), Occur::MustNot);
    builder.build()
}

// -------------------------------------------------------------------------
// MUST_NOT Query 1: +distributed -security → 6 hits
// Java results:
//   doc=13  score=0.4596379
//   doc=2   score=0.4214391
//   doc=7   score=0.4214391
//   doc=8   score=0.4214391
//   doc=0   score=0.3891023
//   doc=6   score=0.3650910
// -------------------------------------------------------------------------

#[test]
fn test_boolean_must_not_distributed_security() {
    let (_dir, reader) = build_golden_docs_index();
    let searcher = IndexSearcher::new(&reader);
    let query = must_must_not_query("contents", b"distributed", b"security");

    let top_docs = searcher.search(&query, 15).unwrap();

    assert_eq!(top_docs.total_hits.value, 6);
    assert_eq!(top_docs.score_docs.len(), 6);

    let expected = [
        (13, 0.4596379_f32),
        (2, 0.4214391),
        (7, 0.4214391),
        (8, 0.4214391),
        (0, 0.3891023),
        (6, 0.365091),
    ];
    for (i, &(doc, score)) in expected.iter().enumerate() {
        assert_eq!(top_docs.score_docs[i].doc, doc);
        assert_in_delta!(top_docs.score_docs[i].score, score, 1e-5);
    }
}

// -------------------------------------------------------------------------
// MUST_NOT Query 2: +memory -quantum → 2 hits
// Java results:
//   doc=8   score=0.7110608
//   doc=0   score=0.6565015
// -------------------------------------------------------------------------

#[test]
fn test_boolean_must_not_memory_quantum() {
    let (_dir, reader) = build_golden_docs_index();
    let searcher = IndexSearcher::new(&reader);
    let query = must_must_not_query("contents", b"memory", b"quantum");

    let top_docs = searcher.search(&query, 15).unwrap();

    assert_eq!(top_docs.total_hits.value, 2);
    assert_eq!(top_docs.score_docs.len(), 2);

    let expected = [(8, 0.7110608_f32), (0, 0.6565015)];
    for (i, &(doc, score)) in expected.iter().enumerate() {
        assert_eq!(top_docs.score_docs[i].doc, doc);
        assert_in_delta!(top_docs.score_docs[i].score, score, 1e-5);
    }
}

// -------------------------------------------------------------------------
// MUST_NOT Query 3: distributed -security → 6 hits (SHOULD + single MUST_NOT)
// Java results:
//   doc=13  score=0.4596379
//   doc=2   score=0.4214391
//   doc=7   score=0.4214391
//   doc=8   score=0.4214391
//   doc=0   score=0.3891023
//   doc=6   score=0.3650910
// -------------------------------------------------------------------------

#[test]
fn test_boolean_should_must_not_distributed_security() {
    let (_dir, reader) = build_golden_docs_index();
    let searcher = IndexSearcher::new(&reader);

    let mut builder = BooleanQuery::builder();
    builder.add_query(
        Box::new(TermQuery::new("contents", b"distributed")),
        Occur::Should,
    );
    builder.add_query(
        Box::new(TermQuery::new("contents", b"security")),
        Occur::MustNot,
    );
    let query = builder.build();

    let top_docs = searcher.search(&query, 15).unwrap();

    assert_eq!(top_docs.total_hits.value, 6);
    assert_eq!(top_docs.score_docs.len(), 6);

    let expected = [
        (13, 0.4596379_f32),
        (2, 0.4214391),
        (7, 0.4214391),
        (8, 0.4214391),
        (0, 0.3891023),
        (6, 0.365091),
    ];
    for (i, &(doc, score)) in expected.iter().enumerate() {
        assert_eq!(top_docs.score_docs[i].doc, doc);
        assert_in_delta!(top_docs.score_docs[i].score, score, 1e-5);
    }
}

// -------------------------------------------------------------------------
// MUST_NOT Query 4: memory -quantum → 2 hits (SHOULD + single MUST_NOT)
// Java results:
//   doc=8   score=0.7110608
//   doc=0   score=0.6565015
// -------------------------------------------------------------------------

#[test]
fn test_boolean_should_must_not_memory_quantum() {
    let (_dir, reader) = build_golden_docs_index();
    let searcher = IndexSearcher::new(&reader);

    let mut builder = BooleanQuery::builder();
    builder.add_query(
        Box::new(TermQuery::new("contents", b"memory")),
        Occur::Should,
    );
    builder.add_query(
        Box::new(TermQuery::new("contents", b"quantum")),
        Occur::MustNot,
    );
    let query = builder.build();

    let top_docs = searcher.search(&query, 15).unwrap();

    assert_eq!(top_docs.total_hits.value, 2);
    assert_eq!(top_docs.score_docs.len(), 2);

    let expected = [(8, 0.7110608_f32), (0, 0.6565015)];
    for (i, &(doc, score)) in expected.iter().enumerate() {
        assert_eq!(top_docs.score_docs[i].doc, doc);
        assert_in_delta!(top_docs.score_docs[i].score, score, 1e-5);
    }
}

// -------------------------------------------------------------------------
// Mixed MUST + SHOULD queries
// Cross-validated against Java Lucene 10.3.2. Same index setup.
// Java queries via QueryParser: `+word1 word2` (MUST + SHOULD).
// -------------------------------------------------------------------------

/// Helper to build a MUST + SHOULD boolean query.
fn must_should_query(field: &str, must_term: &[u8], should_term: &[u8]) -> BooleanQuery {
    let mut builder = BooleanQuery::builder();
    builder.add_query(Box::new(TermQuery::new(field, must_term)), Occur::Must);
    builder.add_query(Box::new(TermQuery::new(field, should_term)), Occur::Should);
    builder.build()
}

// -------------------------------------------------------------------------
// Mixed Query 1: +algorithms data → 4 hits (only docs with "algorithms")
// Scores: docs that also contain "data" get boosted
// Java results:
//   doc=11  score=0.9744643
//   doc=0   score=0.8020670
//   doc=14  score=0.8000477
//   doc=3   score=0.6710463
// -------------------------------------------------------------------------

#[test]
fn test_boolean_mixed_algorithms_data() {
    let (_dir, reader) = build_golden_docs_index();
    let searcher = IndexSearcher::new(&reader);
    let query = must_should_query("contents", b"algorithms", b"data");

    let top_docs = searcher.search(&query, 15).unwrap();

    assert_eq!(top_docs.total_hits.value, 4);
    assert_eq!(top_docs.score_docs.len(), 4);

    let expected = [
        (11, 0.9744643_f32),
        (0, 0.802067),
        (14, 0.8000477),
        (3, 0.6710463),
    ];
    for (i, &(doc, score)) in expected.iter().enumerate() {
        assert_eq!(top_docs.score_docs[i].doc, doc);
        assert_in_delta!(top_docs.score_docs[i].score, score, 1e-5);
    }
}

// -------------------------------------------------------------------------
// Mixed Query 2: +distributed systems → 6 hits
// Java results:
//   doc=13  score=0.6466637
//   doc=2   score=0.5788049
//   doc=7   score=0.5369343
//   doc=8   score=0.5369343
//   doc=6   score=0.5074845
//   doc=0   score=0.4957356
// -------------------------------------------------------------------------

#[test]
fn test_boolean_mixed_distributed_systems() {
    let (_dir, reader) = build_golden_docs_index();
    let searcher = IndexSearcher::new(&reader);
    let query = must_should_query("contents", b"distributed", b"systems");

    let top_docs = searcher.search(&query, 15).unwrap();

    assert_eq!(top_docs.total_hits.value, 6);
    assert_eq!(top_docs.score_docs.len(), 6);

    let expected = [
        (13, 0.6466637_f32),
        (2, 0.5788049),
        (7, 0.5369343),
        (8, 0.5369343),
        (6, 0.5074845),
        (0, 0.4957356),
    ];
    for (i, &(doc, score)) in expected.iter().enumerate() {
        assert_eq!(top_docs.score_docs[i].doc, doc);
        assert_in_delta!(top_docs.score_docs[i].score, score, 1e-5);
    }
}

// -------------------------------------------------------------------------
// Mixed Query 3: +memory performance → 3 hits
// Java results:
//   doc=0   score=1.1177642
//   doc=8   score=0.7110608
//   doc=9   score=0.5691590
// -------------------------------------------------------------------------

#[test]
fn test_boolean_mixed_memory_performance() {
    let (_dir, reader) = build_golden_docs_index();
    let searcher = IndexSearcher::new(&reader);
    let query = must_should_query("contents", b"memory", b"performance");

    let top_docs = searcher.search(&query, 15).unwrap();

    assert_eq!(top_docs.total_hits.value, 3);
    assert_eq!(top_docs.score_docs.len(), 3);

    let expected = [(0, 1.1177642_f32), (8, 0.7110608), (9, 0.569159)];
    for (i, &(doc, score)) in expected.iter().enumerate() {
        assert_eq!(top_docs.score_docs[i].doc, doc);
        assert_in_delta!(top_docs.score_docs[i].score, score, 1e-5);
    }
}

// -------------------------------------------------------------------------
// Conjunction scoring pruning: verify that BooleanQuery conjunctions interact
// correctly with the totalHits threshold (TOTAL_HITS_THRESHOLD = 1000).
//
// Java's BooleanScorerSupplier uses BlockMaxConjunctionBulkScorer for
// TOP_SCORES mode conjunctions with 2+ scoring clauses. This scorer prunes
// non-competitive docs via setMinCompetitiveScore, which means fewer docs
// reach the collector and totalHits is a lower bound (< actual match count).
//
// For TermQuery, BatchScoreBulkScorer already does this pruning correctly.
// This test verifies that BooleanQuery conjunctions prune the same way.
// -------------------------------------------------------------------------

/// Builds an in-memory index with `doc_count` documents, each containing
/// the given terms in a "content" text field.
fn build_large_index(doc_count: usize, terms: &[&str]) -> (SharedDirectory, DirectoryReader) {
    let config = IndexWriterConfig::default().num_threads(1);
    let directory: SharedDirectory = MemoryDirectory::create();
    let writer = IndexWriter::new(config, Arc::clone(&directory));

    for i in 0..doc_count {
        // Each doc gets the common terms plus filler words to vary document length,
        // which produces varying BM25 scores across docs.
        let filler_count = (i % 20) + 1;
        let filler: String = (0..filler_count)
            .map(|j| format!("word{}", j))
            .collect::<Vec<_>>()
            .join(" ");
        let content = format!("{} {}", terms.join(" "), filler);
        let doc = DocumentBuilder::new()
            .add_field(text("content").value(content.as_str()))
            .build();
        writer.add_document(doc).unwrap();
    }

    writer.commit().unwrap();

    let reader = DirectoryReader::open(&*directory).unwrap();
    (directory, reader)
}

#[test]
fn test_boolean_conjunction_pruning_matches_term_query() {
    let doc_count = 1500;
    let (_dir, reader) = build_large_index(doc_count, &["alpha", "beta"]);
    let searcher = IndexSearcher::new(&reader);

    // TermQuery with default threshold (1000): should prune via BatchScoreBulkScorer,
    // giving totalHits < doc_count.
    let term_query = TermQuery::new("content", b"alpha");
    let term_result = searcher.search(&term_query, 10).unwrap();
    assert_lt!(
        term_result.total_hits.value,
        doc_count as i64,
        "TermQuery should prune non-competitive docs when totalHits > threshold"
    );

    // BooleanQuery +alpha +beta with default threshold: should also prune via
    // BlockMaxConjunctionBulkScorer, giving totalHits < doc_count.
    // Currently FAILS because Rust uses DefaultBulkScorer which counts every doc.
    let bool_query = must_must_query("content", b"alpha", b"beta");
    let bool_result = searcher.search(&bool_query, 10).unwrap();
    assert_lt!(
        bool_result.total_hits.value,
        doc_count as i64,
        "BooleanQuery conjunction should prune non-competitive docs like TermQuery does"
    );
}

// -------------------------------------------------------------------------
// 3+ SHOULD term queries — pure disjunction
//
// Cross-validated against Java Lucene 10.3.2: indexed golden-docs with Java
// IndexAllFields (which uses Files.walkFileTree, NOT sorted), then ran the
// queries via QueryIndex with QueryParser default operator (OR).
//
// Java assigns doc IDs in walk order while the Rust integration test indexes
// in sorted order — so doc IDs differ between the two systems. Filenames
// are stable, so we assert against the stored "filename" field instead.
// -------------------------------------------------------------------------

/// Build a pure-SHOULD BooleanQuery from N terms.
fn should_n_query(field: &str, terms: &[&[u8]]) -> BooleanQuery {
    let mut builder = BooleanQuery::builder();
    for &term in terms {
        builder.add_query(Box::new(TermQuery::new(field, term)), Occur::Should);
    }
    builder.build()
}

/// Map top_docs results to (filename, score) pairs.
fn results_by_filename(top_docs: &TopDocs, filenames: &[String]) -> Vec<(String, f32)> {
    top_docs
        .score_docs
        .iter()
        .map(|sd| (filenames[sd.doc as usize].clone(), sd.score))
        .collect()
}

// -------------------------------------------------------------------------
// SHOULD Query: algorithms data systems → 15 hits
// -------------------------------------------------------------------------

#[test]
fn test_boolean_should_3_terms_algorithms_data_systems() {
    let (dir, reader) = build_golden_docs_index();
    let filenames = load_filenames_by_doc(&dir);
    let searcher = IndexSearcher::new(&reader);
    let query = should_n_query("contents", &[b"algorithms", b"data", b"systems"]);

    let top_docs = searcher.search(&query, 15).unwrap();
    assert_eq!(top_docs.total_hits.value, 15);
    assert_eq!(top_docs.score_docs.len(), 15);

    let actual = results_by_filename(&top_docs, &filenames);

    let expected: [(&str, f32); 10] = [
        ("security_004", 1.068_702),
        ("algorithms_001", 0.9087003),
        ("climate_015", 0.8113856),
        ("testing_008", 0.8000477),
        ("storage_009", 0.3473575),
        ("robotics_011", 0.3365355),
        ("graphics_006", 0.3131124),
        ("quantum_013", 0.2922577),
        ("databases_002", 0.2621497),
        ("analysis_010", 0.250_627),
    ];
    for (i, (name, score)) in expected.iter().enumerate() {
        assert_eq!(actual[i].0, *name, "rank {i} filename mismatch");
        assert_in_delta!(actual[i].1, *score, 1e-5);
    }
}

// -------------------------------------------------------------------------
// SHOULD Query: distributed systems networks → 12 hits
// -------------------------------------------------------------------------

#[test]
fn test_boolean_should_3_terms_distributed_systems_networks() {
    let (dir, reader) = build_golden_docs_index();
    let filenames = load_filenames_by_doc(&dir);
    let searcher = IndexSearcher::new(&reader);
    let query = should_n_query("contents", &[b"distributed", b"systems", b"networks"]);

    let top_docs = searcher.search(&query, 15).unwrap();
    assert_eq!(top_docs.total_hits.value, 12);
    assert_eq!(top_docs.score_docs.len(), 12);

    let actual = results_by_filename(&top_docs, &filenames);

    let expected: [(&str, f32); 10] = [
        ("networks_003", 1.6444092),
        ("systems_007", 0.6466637),
        ("biology_014", 0.5788049),
        ("language_012", 0.5369343),
        ("graphics_006", 0.5074845),
        ("algorithms_001", 0.4957356),
        ("storage_009", 0.1548606),
        ("climate_015", 0.1403393),
        ("compilers_005", 0.1385187),
        ("quantum_013", 0.1345176),
    ];
    for (i, (name, score)) in expected.iter().enumerate() {
        assert_eq!(actual[i].0, *name, "rank {i} filename mismatch");
        assert_in_delta!(actual[i].1, *score, 1e-5);
    }
}

// -------------------------------------------------------------------------
// SHOULD Query: quantum biology security → 2 hits
// -------------------------------------------------------------------------

#[test]
fn test_boolean_should_3_terms_quantum_biology_security() {
    let (dir, reader) = build_golden_docs_index();
    let filenames = load_filenames_by_doc(&dir);
    let searcher = IndexSearcher::new(&reader);
    let query = should_n_query("contents", &[b"quantum", b"biology", b"security"]);

    let top_docs = searcher.search(&query, 15).unwrap();
    assert_eq!(top_docs.total_hits.value, 2);
    assert_eq!(top_docs.score_docs.len(), 2);

    let actual = results_by_filename(&top_docs, &filenames);

    let expected = [
        ("biology_014", 1.1074749_f32),
        ("quantum_013", 0.8864633_f32),
    ];
    for (i, (name, score)) in expected.iter().enumerate() {
        assert_eq!(actual[i].0, *name, "rank {i} filename mismatch");
        assert_in_delta!(actual[i].1, *score, 1e-5);
    }
}

// -------------------------------------------------------------------------
// SHOULD Query: algorithms data systems networks compilers (5 terms) → 15 hits
// -------------------------------------------------------------------------

#[test]
fn test_boolean_should_5_terms() {
    let (dir, reader) = build_golden_docs_index();
    let filenames = load_filenames_by_doc(&dir);
    let searcher = IndexSearcher::new(&reader);
    let query = should_n_query(
        "contents",
        &[
            b"algorithms",
            b"data",
            b"systems",
            b"networks",
            b"compilers",
        ],
    );

    let top_docs = searcher.search(&query, 15).unwrap();
    assert_eq!(top_docs.total_hits.value, 15);
    assert_eq!(top_docs.score_docs.len(), 15);

    let actual = results_by_filename(&top_docs, &filenames);

    let expected: [(&str, f32); 10] = [
        ("compilers_005", 1.4667643),
        ("networks_003", 1.2229701),
        ("security_004", 1.068_702),
        ("algorithms_001", 0.9087003),
        ("climate_015", 0.8113856),
        ("testing_008", 0.8000477),
        ("storage_009", 0.3473575),
        ("robotics_011", 0.3365355),
        ("graphics_006", 0.3131124),
        ("quantum_013", 0.2922577),
    ];
    for (i, (name, score)) in expected.iter().enumerate() {
        assert_eq!(actual[i].0, *name, "rank {i} filename mismatch");
        assert_in_delta!(actual[i].1, *score, 1e-5);
    }
}

// -------------------------------------------------------------------------
// Phase 6: multi-MUST_NOT bulk path + mixed msm=0 multi-SHOULD via search()
//
// Cross-validated against Java Lucene 10.3.2 via QueryIndex on /tmp/golden-3term
// (Java IndexAllFields uses unsorted Files.walkFileTree; assertions resolve by
// stored "filename" field).
// -------------------------------------------------------------------------

#[test]
fn test_boolean_must_with_two_must_not() {
    // +algorithms -systems -networks → 1 hit: testing_008 score=0.6006123
    let (dir, reader) = build_golden_docs_index();
    let filenames = load_filenames_by_doc(&dir);
    let searcher = IndexSearcher::new(&reader);
    let mut b = BooleanQuery::builder();
    b.add_query(
        Box::new(TermQuery::new("contents", b"algorithms")),
        Occur::Must,
    );
    b.add_query(
        Box::new(TermQuery::new("contents", b"systems")),
        Occur::MustNot,
    );
    b.add_query(
        Box::new(TermQuery::new("contents", b"networks")),
        Occur::MustNot,
    );
    let query = b.build();

    let top = searcher.search(&query, 10).unwrap();
    assert_eq!(top.total_hits.value, 1);

    let actual = results_by_filename(&top, &filenames);
    assert_eq!(actual[0].0, "testing_008");
    assert_in_delta!(actual[0].1, 0.6006123_f32, 1e-5);
}

#[test]
fn test_boolean_two_must_with_two_must_not() {
    // +algorithms +data -systems -networks → 1 hit: testing_008 score=0.8000477
    let (dir, reader) = build_golden_docs_index();
    let filenames = load_filenames_by_doc(&dir);
    let searcher = IndexSearcher::new(&reader);
    let mut b = BooleanQuery::builder();
    b.add_query(
        Box::new(TermQuery::new("contents", b"algorithms")),
        Occur::Must,
    );
    b.add_query(Box::new(TermQuery::new("contents", b"data")), Occur::Must);
    b.add_query(
        Box::new(TermQuery::new("contents", b"systems")),
        Occur::MustNot,
    );
    b.add_query(
        Box::new(TermQuery::new("contents", b"networks")),
        Occur::MustNot,
    );
    let query = b.build();

    let top = searcher.search(&query, 10).unwrap();
    assert_eq!(top.total_hits.value, 1);

    let actual = results_by_filename(&top, &filenames);
    assert_eq!(actual[0].0, "testing_008");
    assert_in_delta!(actual[0].1, 0.8000477_f32, 1e-5);
}

#[test]
fn test_boolean_mixed_must_with_multi_should_msm_zero() {
    // +algorithms data systems (1 MUST, 2 SHOULD, msm=0) → 4 hits.
    // Same scores as pure 3-SHOULD on the docs that have "algorithms".
    let (dir, reader) = build_golden_docs_index();
    let filenames = load_filenames_by_doc(&dir);
    let searcher = IndexSearcher::new(&reader);
    let mut b = BooleanQuery::builder();
    b.add_query(
        Box::new(TermQuery::new("contents", b"algorithms")),
        Occur::Must,
    );
    b.add_query(Box::new(TermQuery::new("contents", b"data")), Occur::Should);
    b.add_query(
        Box::new(TermQuery::new("contents", b"systems")),
        Occur::Should,
    );
    let query = b.build();

    let top = searcher.search(&query, 10).unwrap();
    assert_eq!(top.total_hits.value, 4);

    let actual = results_by_filename(&top, &filenames);
    let expected = [
        ("security_004", 1.068_702_f32),
        ("algorithms_001", 0.9087003_f32),
        ("climate_015", 0.8113856_f32),
        ("testing_008", 0.8000477_f32),
    ];
    for (i, (name, score)) in expected.iter().enumerate() {
        assert_eq!(actual[i].0, *name, "rank {i} filename mismatch");
        assert_in_delta!(actual[i].1, *score, 1e-5);
    }
}
