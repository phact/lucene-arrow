// SPDX-License-Identifier: Apache-2.0

//! GPU rebuild-merge: read N jVector `OnDiskGraphIndex` files, extract
//! their vectors, build ONE CAGRA → HNSW hierarchy on the GPU, and emit
//! ONE merged multi-layer jVector file. Gate: the real jVector library
//! opens the merged file and searches it correctly.

#![cfg(feature = "cuvs")]

use std::time::Instant;

use lucene_arrow_gpu::cuvs_knn::CuvsContext;
use lucene_arrow_vectors::hnsw::parse_hnswlib;
use lucene_arrow_vectors::jvector::{read_vectors_file, write_index, write_index_multi};

const NUM_SRC: usize = 5;
const CHUNK: usize = 4000;
const DIM: usize = 64;

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
fn gpu_rebuild_merge_of_jvector_files() {
    let Ok(ctx) = CuvsContext::new() else { return };
    let tmp = tempfile::tempdir().unwrap();

    // Write NUM_SRC source jVector files (trivial ring graph — only the
    // vectors matter for a rebuild-merge).
    let mut src_files = Vec::new();
    let mut expected: Vec<f32> = Vec::new();
    for s in 0..NUM_SRC {
        let vecs: Vec<f32> = (0..CHUNK).flat_map(|i| vecf(s * CHUNK + i)).collect();
        expected.extend_from_slice(&vecs);
        let ring: Vec<Vec<u32>> = (0..CHUNK)
            .map(|i| vec![((i + 1) % CHUNK) as u32, ((i + CHUNK - 1) % CHUNK) as u32])
            .collect();
        let p = tmp.path().join(format!("src{s}.jvector"));
        std::fs::write(&p, write_index(&vecs, DIM, &ring, 0).unwrap()).unwrap();
        src_files.push(p);
    }

    // --- read each source's vectors back (reader round-trip) + concat ---
    let t_read = Instant::now();
    let mut merged_vecs: Vec<f32> = Vec::new();
    for p in &src_files {
        let (v, d) = read_vectors_file(p).unwrap();
        assert_eq!(d, DIM);
        merged_vecs.extend_from_slice(&v);
    }
    assert_eq!(merged_vecs, expected, "extracted vectors must match sources exactly");
    let read_ms = t_read.elapsed().as_secs_f64() * 1e3;
    let n = merged_vecs.len() / DIM;

    // --- GPU rebuild: one CAGRA → HNSW hierarchy over the combined set ---
    let t_build = Instant::now();
    let hf = tmp.path().join("m.hnsw");
    ctx.cagra_to_hnswlib(&merged_vecs, DIM, 16, 100, hf.to_str().unwrap()).unwrap();
    let parsed = parse_hnswlib(&std::fs::read(&hf).unwrap()).unwrap();
    let merged = write_index_multi(&merged_vecs, DIM, &parsed).unwrap();
    let build_ms = t_build.elapsed().as_secs_f64() * 1e3;
    let mp = tmp.path().join("merged.jvector");
    std::fs::write(&mp, &merged).unwrap();
    eprintln!(
        "merged {NUM_SRC} files → 1 ({n} vectors, {} levels): read {read_ms:.0} ms, GPU rebuild+write {build_ms:.0} ms",
        parsed.levels.len()
    );

    // --- verify: the real jVector library opens + searches the merged file ---
    let java = std::path::Path::new("/home/tato/.jbang/cache/jdks/21/bin/java");
    let harness = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../harness");
    if !java.exists() {
        eprintln!("skipping jVector-library verify");
        return;
    }
    let cp = format!(
        "{0}/lib/jvector-4.0.0-beta.6.jar:{0}/lib/commons-math3-3.6.1.jar:{0}/lib/agrona-1.20.0.jar:{0}/lib/slf4j-api-2.0.13.jar:{0}/build",
        harness.display()
    );
    std::process::Command::new(java.parent().unwrap().join("javac"))
        .args(["-cp", &cp, "-d"]).arg(harness.join("build"))
        .arg(harness.join("src/VerifyJVector.java")).status().unwrap();
    let mut checked = 0;
    for &qdoc in &[7usize, 9999, n - 1] {
        let o = std::process::Command::new(java)
            .args(["--add-modules", "jdk.incubator.vector", "-cp", &cp, "VerifyJVector"])
            .arg(&mp).args(["x", &qdoc.to_string(), "10"]).output().unwrap();
        assert!(o.status.success(), "VerifyJVector: {}", String::from_utf8_lossy(&o.stderr));
        let hits: Vec<u32> = String::from_utf8_lossy(&o.stdout).lines()
            .filter_map(|l| l.split(',').next()?.parse().ok()).collect();
        assert_eq!(hits.first(), Some(&(qdoc as u32)), "merged doc {qdoc}: own vector not top-1, got {hits:?}");
        checked += 1;
    }
    eprintln!("jVector library opened + searched the merged file: top-1 exact for {checked} queries");
}
