// SPDX-License-Identifier: Apache-2.0

//! Fused GPU extract correctness: the `gather_be_f32` kernel must produce
//! exactly the vectors the CPU reader does, and CAGRA built from that
//! on-device buffer (`cagra_to_hnswlib_device`, no host round-trip) must
//! yield a graph the real jVector library searches correctly.

#![cfg(feature = "cuvs")]

use lucene_arrow_gpu::GpuDecoder;
use lucene_arrow_gpu::cuvs_knn::CuvsContext;
use lucene_arrow_vectors::hnsw::parse_hnswlib;
use lucene_arrow_vectors::jvector::{l0_layout, read_vectors_file, write_index, write_index_multi};

const N: usize = 6000;
const DIM: usize = 128;

fn vecf(seed: usize) -> Vec<f32> {
    (0..DIM)
        .map(|k| {
            let h = ((seed % 120) as u64 ^ ((k as u64) << 32)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let j = ((seed as u64) ^ ((k as u64) << 32) ^ 0xABCD).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
            (h >> 11) as f32 / (1u64 << 53) as f32 * 2.0 - 1.0
                + ((j >> 11) as f32 / (1u64 << 53) as f32 * 2.0 - 1.0) * 0.05
        })
        .collect()
}

#[test]
fn fused_gather_matches_cpu_and_feeds_cagra() {
    let (Ok(gpu), Ok(ctx)) = (GpuDecoder::new(), CuvsContext::new()) else { return };
    let tmp = tempfile::tempdir().unwrap();

    let vecs: Vec<f32> = (0..N).flat_map(vecf).collect();
    let ring: Vec<Vec<u32>> = (0..N).map(|i| vec![((i + 1) % N) as u32]).collect();
    let src = tmp.path().join("src.jvector");
    std::fs::write(&src, write_index(&vecs, DIM, &ring, 0).unwrap()).unwrap();

    // CPU reference extraction.
    let (host, dim) = read_vectors_file(&src).unwrap();
    assert_eq!(dim, DIM);

    // GPU gather → download → must be byte-for-byte identical.
    let raw = std::fs::read(&src).unwrap();
    let (header, record, gdim, gn) = l0_layout(&raw).unwrap();
    assert_eq!((gdim, gn), (DIM, N));
    let dev = gpu.gather_be_f32(&raw, header, record, DIM, N).unwrap();
    gpu.sync().unwrap();
    let gathered = gpu.download_f32(&dev).unwrap();
    assert_eq!(gathered, host, "GPU gather must match CPU read exactly");

    // Device-fed CAGRA (no host round-trip) → hierarchy → jVector file.
    let hf = tmp.path().join("h.hnsw");
    let ptr = gpu.device_ptr_f32(&dev);
    // Safety: dev outlives the call; same primary CUDA context; kernel synced.
    unsafe { ctx.cagra_to_hnswlib_device(ptr, N, DIM, 16, 100, hf.to_str().unwrap()).unwrap() };
    drop(dev);
    let parsed = parse_hnswlib(&std::fs::read(&hf).unwrap()).unwrap();
    assert!(parsed.levels.len() >= 2, "expected a multi-level graph");
    let merged = tmp.path().join("merged.jvector");
    std::fs::write(&merged, write_index_multi(&vecs, DIM, &parsed).unwrap()).unwrap();

    // The real jVector library must open + search the fused-built file.
    let java = std::path::Path::new("/home/tato/.jbang/cache/jdks/21/bin/java");
    let harness = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../harness");
    if !java.exists() {
        eprintln!("fused gather matched CPU; skipping jVector-library search");
        return;
    }
    let cp = format!(
        "{0}/lib/jvector-4.0.0-beta.6.jar:{0}/lib/commons-math3-3.6.1.jar:{0}/lib/agrona-1.20.0.jar:{0}/lib/slf4j-api-2.0.13.jar:{0}/build",
        harness.display()
    );
    std::process::Command::new(java.parent().unwrap().join("javac"))
        .args(["-cp", &cp, "-d"]).arg(harness.join("build"))
        .arg(harness.join("src/VerifyJVector.java")).status().unwrap();
    for &q in &[7usize, 3000, N - 1] {
        let o = std::process::Command::new(java)
            .args(["--add-modules", "jdk.incubator.vector", "-cp", &cp, "VerifyJVector"])
            .arg(&merged).args(["x", &q.to_string(), "10"]).output().unwrap();
        assert!(o.status.success(), "VerifyJVector: {}", String::from_utf8_lossy(&o.stderr));
        let hits: Vec<u32> = String::from_utf8_lossy(&o.stdout).lines()
            .filter_map(|l| l.split(',').next()?.parse().ok()).collect();
        assert_eq!(hits.first(), Some(&(q as u32)), "fused doc {q}: own vector not top-1, got {hits:?}");
    }
    eprintln!("fused extract: GPU gather == CPU read; device-fed CAGRA searched top-1 exact");
}
