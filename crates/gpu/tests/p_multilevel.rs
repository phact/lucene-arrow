// SPDX-License-Identifier: Apache-2.0

//! Multi-level HNSW write gate: cuVS CAGRA -> HNSW hierarchy (hierarchy=CPU,
//! standard hnswlib) -> parse -> our multi-level `.vem`/`.vex` -> CheckIndex
//! clean AND Java `KnnFloatVectorQuery` finds the true neighbors.

#![cfg(feature = "cuvs")]

use lucene_arrow_codec::writer::{
    WriteField, commit_segments, random_segment_id, write_segment_files_ext,
};
use lucene_arrow_gpu::cuvs_knn::CuvsContext;
use lucene_arrow_vectors::file::VectorsFileBuilder;
use lucene_arrow_vectors::hnsw::{HnswFilesBuilder, parse_hnswlib};
use lucene_arrow_vectors::{Similarity, VectorEncoding};

const N: usize = 5000;
const DIM: usize = 64;
const M: usize = 16;

fn vecf(seed: usize) -> Vec<f32> {
    let cluster = seed % 100;
    (0..DIM)
        .map(|k| {
            let hc = ((cluster as u64) ^ ((k as u64) << 32)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let hj = ((seed as u64) ^ ((k as u64) << 32) ^ 0xABCD)
                .wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
            (hc >> 11) as f32 / (1u64 << 53) as f32 * 2.0 - 1.0
                + ((hj >> 11) as f32 / (1u64 << 53) as f32 * 2.0 - 1.0) * 0.05
        })
        .collect()
}

#[test]
fn multilevel_hnsw_checkindex_and_knn() {
    let Ok(ctx) = CuvsContext::new() else { return };
    let vectors: Vec<f32> = (0..N).flat_map(vecf).collect();

    // CAGRA -> HNSW hierarchy -> standard hnswlib file -> parse.
    let tmp = tempfile::tempdir().unwrap();
    let hnsw_file = tmp.path().join("h.hnsw");
    ctx.cagra_to_hnswlib(&vectors, DIM, M, 100, hnsw_file.to_str().unwrap()).unwrap();
    let parsed = parse_hnswlib(&std::fs::read(&hnsw_file).unwrap()).unwrap();
    eprintln!("parsed: {} levels, level sizes {:?}", parsed.levels.len(),
        parsed.levels.iter().map(|l| l.len()).collect::<Vec<_>>());
    assert!(parsed.levels.len() >= 2, "expected a multi-level graph");
    assert_eq!(parsed.levels[0].len(), N, "level 0 must hold all nodes");

    // Write the Lucene segment (flat + multi-level graph).
    let seg_id = random_segment_id();
    let suffix = "Lucene99HnswVectorsFormat_0";
    let payload: Vec<u8> = vectors.iter().flat_map(|v| v.to_le_bytes()).collect();
    let docs: Vec<u32> = (0..N as u32).collect();
    let mut flat = VectorsFileBuilder::new(&seg_id, suffix);
    flat.add_field(0, VectorEncoding::Float32, Similarity::Euclidean, DIM as u32, &docs, &payload, N as u32)
        .unwrap();
    let (vemf, vecb) = flat.finish();
    let mut hb = HnswFilesBuilder::new(&seg_id, suffix);
    hb.add_field_multi(0, VectorEncoding::Float32, Similarity::Euclidean, DIM as u32, N, parsed.m, &parsed.levels)
        .unwrap();
    let (vem, vex) = hb.finish();

    let dv = lucene_arrow_docvalues::file::DocValuesFileBuilder::new(&seg_id, "Lucene90_0");
    let (dvm, dvd) = dv.finish();
    let field = WriteField {
        name: "emb".into(), number: 0, doc_values_type: 0,
        vector_dim: DIM as u32, vector_encoding: 1, vector_similarity: 0, index_options: 0,
    };
    let extra = [
        (format!("_0_{suffix}.vemf"), vemf.as_slice()),
        (format!("_0_{suffix}.vec"), vecb.as_slice()),
        (format!("_0_{suffix}.vem"), vem.as_slice()),
        (format!("_0_{suffix}.vex"), vex.as_slice()),
    ];
    let seg = write_segment_files_ext(tmp.path(), "_0", seg_id, &[field], N as u32, &dvm, &dvd, &extra).unwrap();
    commit_segments(tmp.path(), &[seg]).unwrap();

    // Java gates.
    let java = std::path::Path::new("/home/tato/.jbang/cache/jdks/21/bin/java");
    let harness = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../harness");
    let jar = harness.join("lib/lucene-core-10.3.2.jar");
    if !java.exists() || !jar.exists() {
        eprintln!("skipping Java gates");
        return;
    }
    let ci = std::process::Command::new(java)
        .args(["--add-modules", "jdk.incubator.vector", "-cp"]).arg(&jar)
        .arg("org.apache.lucene.index.CheckIndex").arg(tmp.path()).args(["-level", "2"])
        .output().unwrap();
    let out = String::from_utf8_lossy(&ci.stdout);
    assert!(ci.status.success() && out.contains("No problems were detected"), "CheckIndex failed:\n{out}");
    assert!(out.contains("test: vectors"), "vectors not validated");

    // KNN: query = an exact dataset vector; its own id must come back top-1.
    let build = harness.join("build");
    std::process::Command::new(java.parent().unwrap().join("javac"))
        .args(["-cp"]).arg(&jar).args(["-d"]).arg(&build).arg(harness.join("src/VerifyKnn.java"))
        .status().unwrap();
    let mut checked = 0;
    for &qdoc in &[7usize, 1234, 4999] {
        let o = std::process::Command::new(java)
            .args(["--add-modules", "jdk.incubator.vector", "-cp"])
            .arg(format!("{}:{}", jar.display(), build.display()))
            .arg("VerifyKnn").arg(tmp.path()).args(["emb", &qdoc.to_string(), "10"])
            .output().unwrap();
        let hits: Vec<u32> = String::from_utf8_lossy(&o.stdout).lines()
            .filter_map(|l| l.split(',').next()?.parse().ok()).collect();
        assert_eq!(hits.first(), Some(&(qdoc as u32)), "doc {qdoc}: own vector not top-1, got {hits:?}");
        checked += 1;
    }
    eprintln!("multi-level HNSW: CheckIndex clean + KNN top-1 exact for {checked} queries");
}
