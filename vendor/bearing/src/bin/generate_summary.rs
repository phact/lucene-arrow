// SPDX-License-Identifier: Apache-2.0

//! Generates a JSON summary of an index's structure and statistics.
//!
//! Produces the same format as the Java `GenerateIndexSummary` tool for
//! golden-file comparison.
//!
//! Usage: `generate_summary -index <path>`

use std::env;
use std::path::Path;
use std::process;

use serde::Serialize;

use bearing::document::{DocValuesType, IndexOptions};
use bearing::index::FieldInfo;
use bearing::index::directory_reader::DirectoryReader;
use bearing::index::terms::Terms;
use bearing::store::FSDirectory;

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Summary {
    total_docs: i32,
    max_doc: i32,
    segments: Vec<SegmentSummary>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SegmentSummary {
    index: usize,
    max_doc: i32,
    num_docs: i32,
    tv_chunks: i64,
    fields: Vec<FieldSummary>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct FieldSummary {
    name: String,
    number: u32,
    index_options: String,
    has_norms: bool,
    store_term_vector: bool,
    has_payloads: bool,
    doc_values_type: String,
    point_dimension_count: u32,
    point_index_dimension_count: u32,
    point_num_bytes: u32,
    term_count: i64,
    term_count_verified: i64,
    sum_total_term_freq: i64,
    sum_doc_freq: i64,
    terms_doc_count: i64,
    dv_doc_count: i64,
    norms_doc_count: i64,
    point_doc_count: i64,
    point_count: i64,
}

fn main() {
    let index_path = parse_args();

    let dir = FSDirectory::open(Path::new(&index_path)).unwrap_or_else(|e| {
        eprintln!("Failed to open index directory '{index_path}': {e}");
        process::exit(1);
    });

    let reader = DirectoryReader::open(&dir).unwrap_or_else(|e| {
        eprintln!("Failed to open index: {e}");
        process::exit(1);
    });

    let mut summary = Summary {
        total_docs: reader.max_doc(),
        max_doc: reader.max_doc(),
        segments: Vec::new(),
    };

    for leaf in reader.leaves() {
        let seg = &leaf.reader;

        let mut fields: Vec<&FieldInfo> = seg.field_infos().iter().collect();
        fields.sort_by_key(|fi| fi.number());

        let field_summaries = fields
            .iter()
            .map(|fi| {
                let terms = seg.terms(fi.name());

                let term_count = terms.as_ref().map_or(0, |t| t.size());
                let term_count_verified = count_terms(terms.as_ref().map(|t| t as &dyn Terms));
                let sum_total_term_freq = terms.as_ref().map_or(0, |t| t.get_sum_total_term_freq());
                let sum_doc_freq = terms.as_ref().map_or(0, |t| t.get_sum_doc_freq());
                let terms_doc_count = terms.as_ref().map_or(0, |t| t.get_doc_count() as i64);

                let norms_doc_count = seg
                    .norms_reader()
                    .and_then(|r| r.num_docs_with_field(fi.number()))
                    .unwrap_or(0) as i64;

                let dv_doc_count = seg
                    .doc_values_reader()
                    .and_then(|r| r.num_docs_with_field(fi.number()))
                    .unwrap_or(0) as i64;

                let point_doc_count = seg
                    .points_reader()
                    .and_then(|r| r.doc_count(fi.number()))
                    .unwrap_or(0) as i64;

                let point_count = seg
                    .points_reader()
                    .and_then(|r| r.point_count(fi.number()))
                    .unwrap_or(0);

                FieldSummary {
                    name: fi.name().to_string(),
                    number: fi.number(),
                    index_options: index_options_str(fi.index_options()).to_string(),
                    has_norms: fi.has_norms(),
                    store_term_vector: fi.store_term_vector(),
                    has_payloads: fi.has_payloads(),
                    doc_values_type: doc_values_type_str(fi.doc_values_type()).to_string(),
                    point_dimension_count: fi.point_config().dimension_count,
                    point_index_dimension_count: fi.point_config().index_dimension_count,
                    point_num_bytes: fi.point_config().num_bytes,
                    term_count,
                    term_count_verified,
                    sum_total_term_freq,
                    sum_doc_freq,
                    terms_doc_count,
                    dv_doc_count,
                    norms_doc_count,
                    point_doc_count,
                    point_count,
                }
            })
            .collect();

        let tv_chunks = seg.term_vectors_reader().map_or(0, |r| r.num_chunks());

        summary.segments.push(SegmentSummary {
            index: leaf.ord,
            max_doc: seg.max_doc(),
            num_docs: seg.max_doc(),
            tv_chunks,
            fields: field_summaries,
        });
    }

    let json = serde_json::to_string_pretty(&summary).unwrap_or_else(|e| {
        eprintln!("Failed to serialize JSON: {e}");
        process::exit(1);
    });
    println!("{json}");
    println!();
}

fn count_terms(terms: Option<&dyn Terms>) -> i64 {
    let Some(t) = terms else { return 0 };
    let mut te = match t.iterator() {
        Ok(te) => te,
        Err(_) => return 0,
    };
    let mut count = 0i64;
    while let Ok(Some(_)) = te.next() {
        count += 1;
    }
    count
}

fn index_options_str(opt: IndexOptions) -> &'static str {
    match opt {
        IndexOptions::None => "NONE",
        IndexOptions::Docs => "DOCS",
        IndexOptions::DocsAndFreqs => "DOCS_AND_FREQS",
        IndexOptions::DocsAndFreqsAndPositions => "DOCS_AND_FREQS_AND_POSITIONS",
        IndexOptions::DocsAndFreqsAndPositionsAndOffsets => {
            "DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS"
        }
    }
}

fn doc_values_type_str(dvt: DocValuesType) -> &'static str {
    match dvt {
        DocValuesType::None => "NONE",
        DocValuesType::Numeric => "NUMERIC",
        DocValuesType::Binary => "BINARY",
        DocValuesType::Sorted => "SORTED",
        DocValuesType::SortedNumeric => "SORTED_NUMERIC",
        DocValuesType::SortedSet => "SORTED_SET",
    }
}

fn parse_args() -> String {
    let args: Vec<String> = env::args().collect();
    let mut index_path = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-index" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Missing value for -index");
                    process::exit(1);
                }
                index_path = Some(args[i].clone());
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                print_usage();
                process::exit(1);
            }
        }
        i += 1;
    }

    match index_path {
        Some(p) => p,
        None => {
            print_usage();
            process::exit(1);
        }
    }
}

fn print_usage() {
    eprintln!("Usage: generate_summary -index <index_path>");
    eprintln!("\nGenerates a JSON summary of a Lucene index's structure.");
}
