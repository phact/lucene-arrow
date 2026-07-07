// SPDX-License-Identifier: Apache-2.0

//! P9b: markdown corpus → BM25 segment, ours vs `BenchMdIngest` (JVM).
//! Sort-based aggregation — the exact plan the GPU path mirrors:
//! tokenize+intern → sort (term_id<<32|doc) → RLE → term-order gather.
//!
//! Run: cargo bench -p lucene-arrow-postings --bench bm25_ingest -- <corpus.txt>

use std::time::Instant;

use lucene_arrow_codec::norms::NormsFilesBuilder;
use lucene_arrow_codec::writer::{
    WriteField, commit_segments, random_segment_id, write_segment_files_full,
};
use lucene_arrow_postings::segment::write_postings_files;

fn main() {
    let corpus = std::env::args().skip(1).find(|a| !a.starts_with('-')).expect("corpus path");
    let text = std::fs::read_to_string(&corpus).unwrap();
    let t_total = Instant::now();

    // Phases 1-3 fused: parallel tokenize+intern, sort+RLE, CSR
    // (build_parallel — byte-identical to the serial reference).
    let t = Instant::now();
    let lines: Vec<&str> = text.lines().collect();
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);
    let inv = lucene_arrow_postings::build::build_parallel(&lines, threads);
    let num_docs = inv.norms.len() as u32;
    let t_build = t.elapsed();
    let vocab_len = inv.num_terms();

    // Phase 4: write the segment.
    let t = Instant::now();
    let tmp = tempfile::tempdir().unwrap();
    let id = random_segment_id();
    let postings_files = write_postings_files(tmp.path(), "_0", &id, "body", 0, &inv).unwrap();
    let mut nb = NormsFilesBuilder::new(&id, "");
    nb.add_dense_field(0, &inv.norms).unwrap();
    let (nvm, nvd) = nb.finish();
    let dv = lucene_arrow_docvalues::file::DocValuesFileBuilder::new(&id, "Lucene90_0");
    let (dvm, dvd) = dv.finish();
    let field = WriteField {
        name: "body".into(),
        number: 0,
        doc_values_type: 0,
        vector_dim: 0,
        vector_encoding: 0,
        vector_similarity: 0,
        index_options: 2,
    };
    let extra = [("_0.nvm".to_string(), nvm.as_slice()), ("_0.nvd".to_string(), nvd.as_slice())];
    let seg = write_segment_files_full(
        tmp.path(), "_0", id, &[field], num_docs, &dvm, &dvd, &extra, &postings_files,
    )
    .unwrap();
    commit_segments(tmp.path(), &[seg]).unwrap();
    let t_write = t.elapsed();
    let total = t_total.elapsed();

    println!("ours: {num_docs} docs, {vocab_len} terms, {} postings", inv.docs.len());
    println!("  build (parallel tokenize + sort/RLE): {:>7.0} ms", t_build.as_secs_f64() * 1e3);
    println!("  segment write   : {:>7.0} ms", t_write.as_secs_f64() * 1e3);
    println!(
        "  TOTAL           : {:>7.0} ms = {:.0} kdocs/s",
        total.as_secs_f64() * 1e3,
        num_docs as f64 / total.as_secs_f64() / 1e3
    );
    println!("  segment kept at {:?}", tmp.keep());
}
