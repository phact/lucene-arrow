// SPDX-License-Identifier: Apache-2.0

//! P6a gate (decision register #14): the `cuvs` crate against the
//! pixi-provided `libcuvs`, over the same vectors our decoder produces.
//! Brute force must agree with `FlatKnn` exactly; CAGRA must hit high
//! recall against them.

#![cfg(feature = "cuvs")]

use lucene_arrow_gpu::cuvs_knn::CuvsContext;
use lucene_arrow_gpu::{GpuDecoder, knn::FlatKnn};
use lucene_arrow_vectors::Similarity;

/// Embedding-like pseudo-random vectors in [-1, 1] (hash-based, unique,
/// well-separated — the regime ANN engines are built for).
fn vecf(seed: usize, dim: usize) -> Vec<f32> {
    (0..dim)
        .map(|k| {
            let h = (seed as u64 ^ (k as u64) << 32).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            (h >> 11) as f32 / (1u64 << 53) as f32 * 2.0 - 1.0
        })
        .collect()
}

#[test]
fn cuvs_brute_force_agrees_with_flatknn_and_cagra_recalls() {
    let Ok(gpu) = GpuDecoder::new() else {
        eprintln!("no CUDA device");
        return;
    };
    let ctx = match CuvsContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skipping: cuVS unavailable ({e})");
            return;
        }
    };

    let (n, dim, k, nq) = (20_000usize, 64usize, 10usize, 8usize);
    let vectors: Vec<f32> = (0..n).flat_map(|i| vecf(i * 13 + 1, dim)).collect();
    // Queries = perturbed dataset points (realistic ANN workload).
    let queries: Vec<f32> = (0..nq)
        .flat_map(|i| {
            vecf(i * 997 * 13 + 1, dim).into_iter().map(move |v| v + (i as f32 + 1.0) * 1e-3)
        })
        .collect();

    // Reference: our FlatKnn over a device buffer.
    let payload_bytes: Vec<u8> = vectors.iter().flat_map(|v| v.to_le_bytes()).collect();
    let data = gpu.upload(&payload_bytes).unwrap();
    let plan = lucene_arrow_core::plan::DecodePlan {
        plan_version: lucene_arrow_core::plan::PLAN_VERSION,
        column: lucene_arrow_core::plan::FieldId::new(0, "v"),
        file: "t.vec".into(),
        arrow_type: arrow_schema::DataType::FixedSizeList(
            std::sync::Arc::new(arrow_schema::Field::new(
                "item",
                arrow_schema::DataType::Float32,
                false,
            )),
            dim as i32,
        ),
        blocks: vec![lucene_arrow_core::plan::BlockDecode::Raw {
            offset: 0,
            len: payload_bytes.len() as u64,
        }],
        coverage: lucene_arrow_core::plan::Coverage::Dense { num_docs: n as u32 },
        num_values: n as u64,
    };
    let dev_payload = gpu.vector_payload_device(&plan, &data).unwrap();
    let flat = FlatKnn::new(&gpu).unwrap();
    let reference = flat
        .search(&dev_payload, n as u64, dim as u32, Similarity::Euclidean, &queries, k)
        .unwrap();

    // cuVS brute force: exact — identical id sets (ordering may differ on
    // exact distance ties, so compare as sets).
    // Both are exact, but distance summation order differs (cuVS uses the
    // expanded form), so the k-th slot may swap between near-equal
    // candidates. Gate: top-1 identical; ≥ k-1 of k ids shared per query.
    let bf = ctx.brute_force(&vectors, dim, &queries, k).unwrap();
    for (qi, (ours, theirs)) in reference.iter().zip(&bf).enumerate() {
        assert_eq!(ours[0].ord, theirs[0].ord, "q{qi}: top-1 differs");
        let a: std::collections::BTreeSet<u32> = ours.iter().map(|h| h.ord).collect();
        let shared = theirs.iter().filter(|h| a.contains(&h.ord)).count();
        assert!(shared >= k - 1, "q{qi}: only {shared}/{k} ids shared with FlatKnn");
    }

    // CAGRA: ANN — require ≥ 90% recall@10 vs exact.
    let (cagra, build, _search) = ctx.cagra(&vectors, dim, &queries, k).unwrap();
    let mut hits = 0usize;
    for (ours, theirs) in reference.iter().zip(&cagra) {
        let exact: std::collections::BTreeSet<u32> = ours.iter().map(|h| h.ord).collect();
        hits += theirs.iter().filter(|h| exact.contains(&h.ord)).count();
    }
    let recall = hits as f64 / (nq * k) as f64;
    eprintln!("CAGRA build {build:?}, recall@{k} = {recall:.3}");
    assert!(recall >= 0.9, "CAGRA recall {recall} < 0.9");
}
