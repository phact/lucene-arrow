// Query a Bearing-readable index with a list of query strings and report per-query timing.
//
// Each line of the queries file is a single JSON object:
//
//   {"q": "<query string in Lucene syntax>", "msm": <int, optional, default 0>}
//
// Supported query strings:
//   - bare word:           algorithms         (TermQuery)
//   - boolean MUST:        +algorithms +data  (BooleanQuery with MUST clauses)
//   - boolean SHOULD:      algorithms data    (BooleanQuery with SHOULD clauses)
//   - boolean MUST_NOT:    +algorithms -data  (BooleanQuery with MUST and MUST_NOT clauses)
//
// `msm` (if > 0 and the parsed query has at least msm SHOULD clauses) sets
// `set_minimum_number_should_match(msm)` on the BooleanQuery builder.
//
// Usage:
//   cargo run --release --bin queryindex -- -index <DIR> -queries <FILE.jsonl> [-output <FILE>]

use std::env;
use std::fmt::Write as _;
use std::fs;
use std::io::Write;
use std::panic;
use std::path::Path;
use std::process;
use std::time::Instant;

use bearing::index::directory_reader::DirectoryReader;
use bearing::search::index_searcher::IndexSearcher;
use bearing::search::query::Query;
use bearing::search::term_query::TermQuery;
use bearing::search::top_score_doc_collector::TopScoreDocCollectorManager;
use bearing::search::{BooleanQuery, Occur};
use bearing::store::FSDirectory;
use serde::Deserialize;

#[derive(Deserialize)]
struct QueryEntry {
    q: String,
    #[serde(default)]
    msm: i32,
}

/// Parses a query string in Lucene standard syntax into a Query.
fn parse_query(query_str: &str, field: &str, msm: i32) -> Box<dyn Query> {
    let tokens: Vec<&str> = query_str.split_whitespace().collect();

    let has_plus = tokens.iter().any(|t| t.starts_with('+'));
    let has_minus = tokens.iter().any(|t| t.starts_with('-'));

    if has_plus || has_minus {
        let mut builder = BooleanQuery::builder();
        let mut should_count = 0;
        for token in &tokens {
            if let Some(term) = token.strip_prefix('+') {
                builder.add_query(
                    Box::new(TermQuery::new(field, term.as_bytes())),
                    Occur::Must,
                );
            } else if let Some(term) = token.strip_prefix('-') {
                builder.add_query(
                    Box::new(TermQuery::new(field, term.as_bytes())),
                    Occur::MustNot,
                );
            } else {
                builder.add_query(
                    Box::new(TermQuery::new(field, token.as_bytes())),
                    Occur::Should,
                );
                should_count += 1;
            }
        }
        if msm > 0 && should_count >= msm {
            builder.set_minimum_number_should_match(msm);
        }
        Box::new(builder.build())
    } else if tokens.len() > 1 {
        let mut builder = BooleanQuery::builder();
        for token in &tokens {
            builder.add_query(
                Box::new(TermQuery::new(field, token.as_bytes())),
                Occur::Should,
            );
        }
        if msm > 0 && tokens.len() as i32 >= msm {
            builder.set_minimum_number_should_match(msm);
        }
        Box::new(builder.build())
    } else {
        Box::new(TermQuery::new(field, tokens[0].as_bytes()))
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();

    let mut index_path = String::new();
    let mut queries_path = String::new();
    let mut output_path = String::new();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-index" => {
                i += 1;
                index_path = args[i].clone();
            }
            "-queries" => {
                i += 1;
                queries_path = args[i].clone();
            }
            "-output" => {
                i += 1;
                output_path = args[i].clone();
            }
            other => {
                eprintln!("Unknown argument: {other}");
                process::exit(1);
            }
        }
        i += 1;
    }

    if index_path.is_empty() || queries_path.is_empty() {
        eprintln!(
            "Usage: queryindex -index <INDEX_DIR> -queries <QUERIES_FILE.jsonl> [-output <RESULTS_FILE>]"
        );
        process::exit(1);
    }

    let queries_content = fs::read_to_string(&queries_path).expect("Failed to read queries file");
    let entries: Vec<(String, QueryEntry)> = queries_content
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(|line| {
            let entry: QueryEntry = serde_json::from_str(line)
                .unwrap_or_else(|e| panic!("Invalid JSON line {line:?}: {e}"));
            (line.to_string(), entry)
        })
        .collect();

    let dir = FSDirectory::open(Path::new(&index_path)).expect("Failed to open index directory");
    let reader = DirectoryReader::open(&dir).expect("Failed to open DirectoryReader");
    let searcher = IndexSearcher::new(&reader);

    let mut results: Vec<String> = Vec::with_capacity(entries.len());
    let mut errors = 0;

    let start = Instant::now();

    for (raw, entry) in &entries {
        let query = parse_query(&entry.q, "contents", entry.msm);
        let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            let manager = TopScoreDocCollectorManager::new(10, None, i32::MAX);
            searcher.search_with_collector_manager(query.as_ref(), &manager)
        }));
        match result {
            Ok(Ok(top_docs)) => {
                let mut line = String::new();
                write!(line, "{:<40} hits={:<6}", raw, top_docs.total_hits.value).unwrap();
                for sd in &top_docs.score_docs {
                    write!(line, "  doc={:<5} score={:.5}", sd.doc, sd.score).unwrap();
                }
                results.push(line);
            }
            Ok(Err(e)) => {
                results.push(format!("{raw:<40} ERROR: {e}"));
                errors += 1;
            }
            Err(_) => {
                results.push(format!("{raw:<40} PANIC"));
                errors += 1;
            }
        }
    }

    let elapsed = start.elapsed();

    if !output_path.is_empty() {
        let mut f = fs::File::create(&output_path).expect("Failed to create output file");
        for line in &results {
            writeln!(f, "{line}").unwrap();
        }
    } else {
        for line in &results {
            println!("{line}");
        }
    }

    println!(
        "Queried {} queries in {elapsed:.2?} ({errors} errors)",
        entries.len()
    );
    println!(
        "Average: {:.2} µs/query",
        elapsed.as_micros() as f64 / entries.len() as f64
    );
}
