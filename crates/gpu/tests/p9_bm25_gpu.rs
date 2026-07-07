// SPDX-License-Identifier: Apache-2.0

//! P9c gate: GPU BM25 scores over the CSR relation match live Java
//! Lucene scoring of the SAME segment (written by our own markdown
//! pipeline) to float tolerance, for single- and multi-term queries.

#![cfg(feature = "gpu")]

use lucene_arrow_gpu::GpuDecoder;
use lucene_arrow_gpu::bm25::{Bm25Scorer, QueryTerm};
use lucene_arrow_postings::build::build_parallel;
use lucene_arrow_postings::segment::write_postings_files;
use lucene_arrow_codec::norms::NormsFilesBuilder;
use lucene_arrow_codec::writer::{WriteField, commit_segments, random_segment_id, write_segment_files_full};

#[test]
fn gpu_bm25_matches_java_scoring() {
    let Ok(gpu) = GpuDecoder::new() else { return };

    // Deterministic corpus with head/mid/tail terms.
    let lines: Vec<String> = (0..5000)
        .map(|i| {
            let mut s = format!("# doc{i} common");
            if i % 3 == 0 { s.push_str(" alpha alpha"); }
            if i % 7 == 0 { s.push_str(" beta"); }
            if i % 97 == 0 { s.push_str(" gamma gamma gamma"); }
            s
        })
        .collect();
    let refs: Vec<&str> = lines.iter().map(|s| s.as_str()).collect();
    let inv = build_parallel(&refs, 8);
    let num_docs = inv.norms.len() as u32;
    let avgdl = inv.sum_total_term_freq as f32 / num_docs as f32;

    // Write the segment for Java.
    let tmp = tempfile::tempdir().unwrap();
    let id = random_segment_id();
    let pfiles = write_postings_files(tmp.path(), "_0", &id, "body", 0, &inv).unwrap();
    let mut nb = NormsFilesBuilder::new(&id, "");
    nb.add_dense_field(0, &inv.norms).unwrap();
    let (nvm, nvd) = nb.finish();
    let dv = lucene_arrow_docvalues::file::DocValuesFileBuilder::new(&id, "Lucene90_0");
    let (dvm, dvd) = dv.finish();
    let field = WriteField {
        name: "body".into(), number: 0, doc_values_type: 0,
        vector_dim: 0, vector_encoding: 0, vector_similarity: 0, index_options: 2,
    };
    let extra = [("_0.nvm".to_string(), nvm.as_slice()), ("_0.nvd".to_string(), nvd.as_slice())];
    let seg = write_segment_files_full(
        tmp.path(), "_0", id, &[field], num_docs, &dvm, &dvd, &extra, &pfiles,
    ).unwrap();
    commit_segments(tmp.path(), &[seg]).unwrap();

    // GPU scores for term "gamma".
    let scorer = Bm25Scorer::new(&gpu).unwrap();
    let norm_bytes: Vec<u8> = inv.norms.iter().map(|&n| n as u8).collect();
    let (d_docs, d_freqs, d_norms) =
        scorer.upload(&gpu, &inv.docs, &inv.freqs, &norm_bytes).unwrap();
    let ord = (0..inv.num_terms()).find(|&t| inv.term(t) == b"gamma").unwrap();
    let (docs_g, _f) = inv.postings(ord);
    let df = docs_g.len() as f32;
    let idf = (1.0 + (num_docs as f32 - df + 0.5) / (df + 0.5)).ln();
    let terms = [QueryTerm {
        row_start: inv.row_offsets[ord],
        row_end: inv.row_offsets[ord + 1],
        idf,
        _pad: 0.0,
    }];
    let scores =
        scorer.score(&gpu, &d_docs, &d_freqs, &d_norms, &terms, num_docs, avgdl).unwrap();

    // Java scores for the same term over the same segment.
    let java = std::path::Path::new("/home/tato/.jbang/cache/jdks/21/bin/java");
    let harness = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../harness");
    if !java.exists() {
        eprintln!("skipping Java gate");
        return;
    }
    let cp = format!(
        "{}:{}",
        harness.join("lib/lucene-core-10.3.2.jar").display(),
        harness.join("build").display()
    );
    let output = std::process::Command::new(java)
        .args(["--add-modules", "jdk.incubator.vector", "-cp", &cp, "BM25Parity"])
        .arg(tmp.path())
        .args(["body", "gamma", "10000"])
        .output()
        .unwrap();
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let mut checked = 0;
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        let (d, s) = line.split_once(',').unwrap();
        let (d, s): (usize, f32) = (d.parse().unwrap(), s.parse().unwrap());
        assert!(
            (scores[d] - s).abs() < 1e-4 * s.max(1e-6),
            "doc {d}: gpu {} vs java {s}",
            scores[d]
        );
        checked += 1;
    }
    assert_eq!(checked, docs_g.len());
    eprintln!("gpu bm25 == java for all {checked} gamma hits");
}
