// SPDX-License-Identifier: Apache-2.0

//! P9c: GPU BM25 disjunctive scoring throughput over the 300k-doc corpus
//! CSR vs JVM BooleanQuery (harness BenchBM25Query, same queries file).

use std::time::Instant;

use lucene_arrow_gpu::GpuDecoder;
use lucene_arrow_gpu::bm25::{Bm25Scorer, QueryTerm};
use lucene_arrow_postings::build::build_parallel;

fn main() {
    let corpus = std::env::args().skip(1).find(|a| !a.starts_with('-')).expect("corpus");
    let Ok(gpu) = GpuDecoder::new() else { return eprintln!("no CUDA") };
    let text = std::fs::read_to_string(&corpus).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    let inv = build_parallel(&lines, 16);
    let num_docs = inv.norms.len() as u32;
    let avgdl = inv.sum_total_term_freq as f32 / num_docs as f32;
    eprintln!("csr: {} docs, {} terms, {} rows", num_docs, inv.num_terms(), inv.docs.len());

    // Two query sets: "selective" (random vocab — mostly rare terms, the
    // shape Lucene's skipping loves) and "heavy" (top-df head terms —
    // the analytics shape where exhaustive scoring pays off).
    let nq = 256usize;
    let nt = inv.num_terms();
    let mut by_df: Vec<usize> = (0..nt).collect();
    by_df.sort_by_key(|&t| std::cmp::Reverse(inv.row_offsets[t + 1] - inv.row_offsets[t]));
    let selective: Vec<[usize; 3]> = (0..nq)
        .map(|i| [(i * 131) % nt, (i * 7919 + 13) % nt, (i * 104729 + 7) % nt])
        .collect();
    let heavy: Vec<[usize; 3]> = (0..nq)
        .map(|i| [by_df[i % 500], by_df[(i * 7 + 3) % 500], by_df[(i * 13 + 11) % 500]])
        .collect();
    let queries = selective;
    for (name, set) in [("bm25_queries.txt", &queries), ("bm25_queries_heavy.txt", &heavy)] {
        let mut qtext = String::new();
        for q in set {
            for (j, &t) in q.iter().enumerate() {
                if j > 0 { qtext.push(' '); }
                qtext.push_str(std::str::from_utf8(inv.term(t)).unwrap());
            }
            qtext.push('\n');
        }
        std::fs::write(std::env::temp_dir().join(name), qtext).unwrap();
    }

    let scorer = Bm25Scorer::new(&gpu).unwrap();
    let norm_bytes: Vec<u8> = inv.norms.iter().map(|&n| n as u8).collect();
    let (d, f, n) = scorer.upload(&gpu, &inv.docs, &inv.freqs, &norm_bytes).unwrap();

    let make_terms = |q: &[usize; 3]| -> Vec<QueryTerm> {
        q.iter()
            .map(|&t| {
                let df = (inv.row_offsets[t + 1] - inv.row_offsets[t]) as f32;
                QueryTerm {
                    row_start: inv.row_offsets[t],
                    row_end: inv.row_offsets[t + 1],
                    idf: (1.0 + (num_docs as f32 - df + 0.5) / (df + 0.5)).ln(),
                    _pad: 0.0,
                }
            })
            .collect()
    };

    for (setname, queries) in [("selective", &queries), ("heavy", &heavy)] {
    for round in 0..3 {
        let t0 = Instant::now();
        let mut rows = 0u64;
        let mut top1 = 0u32;
        for q in queries.iter() {
            let terms = make_terms(q);
            rows += terms.iter().map(|t| t.row_end - t.row_start).sum::<u64>();
            let scores = scorer.score(&gpu, &d, &f, &n, &terms, num_docs, avgdl).unwrap();
            // top-1 on host (dense array), keeps the result honest.
            let mut best = 0usize;
            for i in 1..scores.len() {
                if scores[i] > scores[best] {
                    best = i;
                }
            }
            top1 ^= best as u32;
        }
        let secs = t0.elapsed().as_secs_f64();
        println!(
            "{setname} round {round}: {nq} queries in {:.3} s = {:.0} qps, {:.1} Mrows/s scored (x{top1})",
            secs,
            nq as f64 / secs,
            rows as f64 / secs / 1e6
        );
    }
    // Batched: all queries in ONE launch + device top-10 (the shape that
    // actually fills the GPU — per-query launches pay a fixed floor).
    let batch: Vec<Vec<QueryTerm>> = queries.iter().map(make_terms).collect();
    let rows: u64 = batch.iter().flatten().map(|t| t.row_end - t.row_start).sum();
    for round in 0..3 {
        let t0 = Instant::now();
        let top =
            scorer.score_batch(&gpu, &d, &f, &n, &batch, num_docs, avgdl, 10).unwrap();
        let secs = t0.elapsed().as_secs_f64();
        let top1: u32 = top.iter().map(|t| t[0].0).fold(0, |a, d| a ^ d);
        println!(
            "{setname} BATCHED round {round}: {nq} queries in {:.3} s = {:.0} qps, {:.1} Mrows/s scored (x{top1})",
            secs,
            nq as f64 / secs,
            rows as f64 / secs / 1e6
        );
    }
    }
}
