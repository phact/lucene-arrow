// SPDX-License-Identifier: Apache-2.0

//! P10 differential gate: GPU text ingest must produce a byte-identical
//! `InvertedField` to the CPU `build_parallel` reference — including
//! Unicode content (exercises the dirty-span hybrid path).

#![cfg(feature = "gpu")]

use lucene_arrow_gpu::GpuDecoder;
use lucene_arrow_gpu::text_ingest::GpuTextIngest;
use lucene_arrow_postings::build::build_parallel;

fn assert_identical(
    a: &lucene_arrow_postings::build::InvertedField,
    b: &lucene_arrow_postings::build::InvertedField,
    label: &str,
) {
    assert_eq!(a.term_bytes, b.term_bytes, "{label}: term_bytes");
    assert_eq!(a.term_offsets, b.term_offsets, "{label}: term_offsets");
    assert_eq!(a.row_offsets, b.row_offsets, "{label}: row_offsets");
    assert_eq!(a.docs, b.docs, "{label}: docs");
    assert_eq!(a.freqs, b.freqs, "{label}: freqs");
    assert_eq!(a.norms, b.norms, "{label}: norms");
    assert_eq!(a.sum_total_term_freq, b.sum_total_term_freq, "{label}: ttf");
}

#[test]
fn gpu_ingest_matches_cpu_reference() {
    let Ok(gpu) = GpuDecoder::new() else { return };
    let ingest = GpuTextIngest::new(&gpu).unwrap();

    // Mixed corpus: markdown syntax, unicode words, math symbols, long
    // tokens, repeated terms, empty-ish docs.
    let mut lines: Vec<String> = (0..2000)
        .map(|i| {
            let mut s = format!("# Doc{i} the QUICK brown w{} fox_{}", i % 97, i % 7);
            if i % 3 == 0 {
                s.push_str(" naïve café Σίγμα ün%20d — ß");
            }
            if i % 5 == 0 {
                s.push_str(" α=β+γ x≤y");
            }
            if i % 11 == 0 {
                s.push_str(&format!(" {}", "z".repeat(300))); // dropped by both
            }
            s
        })
        .collect();
    lines.push("...".into()); // doc with zero tokens
    lines.push("The the THE ThE".into()); // freq aggregation
    let refs: Vec<&str> = lines.iter().map(|s| s.as_str()).collect();
    let corpus = lines.join("\n");

    let cpu = build_parallel(&refs, 8);
    let (gpu_inv, stats) = ingest.build(&gpu, &corpus, 8).unwrap();
    eprintln!(
        "gpu: {} clean pairs, {} dirty spans, kernel {:.2} ms",
        stats.clean_pairs, stats.dirty_spans, stats.kernel_ms
    );
    assert!(stats.dirty_spans > 0, "corpus must exercise the hybrid path");
    assert_identical(&gpu_inv, &cpu, "mixed corpus");

    // Real corpus when present (arXiv markdown conversions).
    let arxiv = std::path::PathBuf::from(
        "/tmp/claude-1000/-home-tato-Desktop-arrow-lucene/34b4da2f-450e-4716-9bfc-25dc7d79ebb4/scratchpad/corpus-arxiv.txt",
    );
    if let Ok(text) = std::fs::read_to_string(&arxiv) {
        let refs: Vec<&str> = text.lines().collect();
        let cpu = build_parallel(&refs, 16);
        let (gpu_inv, stats) = ingest.build(&gpu, &text, 16).unwrap();
        eprintln!(
            "arxiv: {} clean pairs, {} dirty spans, kernel {:.2} ms",
            stats.clean_pairs, stats.dirty_spans, stats.kernel_ms
        );
        assert_identical(&gpu_inv, &cpu, "arxiv");
    }
}
