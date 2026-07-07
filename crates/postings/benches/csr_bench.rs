// SPDX-License-Identifier: Apache-2.0

//! P7 throughput vs the JVM TermsEnum/PostingsEnum sweep (harness
//! `BenchText scan`). Two modes: pure scan (iterate + checksum, matching
//! the Java loop) and full CSR materialization.
//!
//! Run: cargo bench -p lucene-arrow-postings --bench csr_bench

use std::time::Instant;

use lucene_arrow_postings::coo::read_csr;
use lucene_arrow_postings::doc::scan_postings;
use lucene_arrow_postings::walk::{FieldTraits, walk_terms};
use lucene_arrow_postings::{parse_tmd, root_block};

fn main() {
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../harness/bench-text");
    if !dir.exists() {
        return eprintln!("bench index absent: java BenchText ingest harness/bench-text 4000000");
    }
    let tmd = std::fs::read(dir.join("_0_Lucene103_0.tmd")).unwrap();
    let tim = std::fs::read(dir.join("_0_Lucene103_0.tim")).unwrap();
    let tip = std::fs::read(dir.join("_0_Lucene103_0.tip")).unwrap();
    let doc = std::fs::read(dir.join("_0_Lucene103_0.doc")).unwrap();

    let traits = FieldTraits { has_freqs: true, has_positions: true, has_offsets: false };
    let m = &parse_tmd(&tmd, |_| true).unwrap()[0];
    let tip_slice = &tip[m.index_start_fp as usize..m.index_end_fp as usize];
    let root = root_block(tip_slice, m.trie_root_fp).unwrap();

    for round in 0..3 {
        let t = Instant::now();
        let mut postings = 0u64;
        let mut terms = 0u64;
        let mut sum = 0u64;
        walk_terms(&tim, root.fp, traits, |_term, df, ttf, tm| {
            terms += 1;
            scan_postings(&doc, df, ttf, tm, traits, |d, f| {
                sum += (d + f) as u64;
                postings += 1;
                Ok(())
            })
        })
        .unwrap();
        let secs = t.elapsed().as_secs_f64();
        println!(
            "scan round {round}: {terms} terms, {postings} postings in {secs:.2} s = {:.1} Mpostings/s (sum {sum})",
            postings as f64 / secs / 1e6
        );
    }

    let t = Instant::now();
    let csr = read_csr(&tim, &doc, root.fp, traits).unwrap();
    let secs = t.elapsed().as_secs_f64();
    println!(
        "read_csr: {} terms, {} rows in {secs:.2} s = {:.1} Mpostings/s (materialized)",
        csr.num_terms(),
        csr.num_rows(),
        csr.num_rows() as f64 / secs / 1e6
    );
}
