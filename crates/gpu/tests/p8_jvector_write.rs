// SPDX-License-Identifier: Apache-2.0

//! P8 acceptance: GPU-built graph (NN-Descent) → our jVector
//! `OnDiskGraphIndex` v5 serializer → the REAL jVector 4.0.0-beta library
//! opens it and its graph search returns the true nearest neighbors our
//! exact GPU search reports.

#![cfg(feature = "cuvs")]

use lucene_arrow_gpu::cuvs_knn::CuvsContext;
use lucene_arrow_gpu::{GpuDecoder, knn::FlatKnn};
use lucene_arrow_vectors::Similarity;
use lucene_arrow_vectors::hnsw::navigable_from_knn;
use lucene_arrow_vectors::jvector::write_index;

const N: usize = 4000;
const DIM: usize = 64;
const DEGREE: usize = 32;

fn vecf(seed: usize) -> Vec<f32> {
    let cluster = seed % 100;
    (0..DIM)
        .map(|k| {
            let hc =
                ((cluster as u64) ^ ((k as u64) << 32)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let hj = ((seed as u64) ^ ((k as u64) << 32) ^ 0xABCD)
                .wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
            let center = (hc >> 11) as f32 / (1u64 << 53) as f32 * 2.0 - 1.0;
            let jitter = ((hj >> 11) as f32 / (1u64 << 53) as f32 * 2.0 - 1.0) * 0.05;
            center + jitter
        })
        .collect()
}

#[test]
fn jvector_reads_and_searches_our_file() {
    let Ok(gpu) = GpuDecoder::new() else { return };
    let Ok(ctx) = CuvsContext::new() else {
        eprintln!("cuVS unavailable");
        return;
    };

    let vectors: Vec<f32> = (0..N).flat_map(vecf).collect();
    let (graph, _t) = ctx.knn_graph(&vectors, DIM, DEGREE).unwrap();
    let neighbors = navigable_from_knn(&graph, N, DEGREE, DEGREE);

    let file = write_index(&vectors, DIM, &neighbors, 0).unwrap();
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("graph.jvector");
    std::fs::write(&path, &file).unwrap();

    // Exact reference: FlatKnn on device.
    let k = 10usize;
    let query_doc = 7usize;
    let query = vecf(query_doc);
    let payload: Vec<u8> = vectors.iter().flat_map(|v| v.to_le_bytes()).collect();
    let data = gpu.upload(&payload).unwrap();
    let plan = lucene_arrow_core::plan::DecodePlan {
        plan_version: lucene_arrow_core::plan::PLAN_VERSION,
        column: lucene_arrow_core::plan::FieldId::new(0, "v"),
        file: "x.vec".into(),
        arrow_type: arrow_schema::DataType::FixedSizeList(
            std::sync::Arc::new(arrow_schema::Field::new(
                "item",
                arrow_schema::DataType::Float32,
                false,
            )),
            DIM as i32,
        ),
        blocks: vec![lucene_arrow_core::plan::BlockDecode::Raw {
            offset: 0,
            len: payload.len() as u64,
        }],
        coverage: lucene_arrow_core::plan::Coverage::Dense { num_docs: N as u32 },
        num_values: N as u64,
    };
    let dev = gpu.vector_payload_device(&plan, &data).unwrap();
    let flat = FlatKnn::new(&gpu).unwrap();
    let exact =
        &flat.search(&dev, N as u64, DIM as u32, Similarity::Euclidean, &query, k).unwrap()[0];
    assert_eq!(exact[0].ord as usize, query_doc);

    // jVector opens + searches our bytes.
    let java = std::path::Path::new("/home/tato/.jbang/cache/jdks/21/bin/java");
    let harness = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../harness");
    if !java.exists() {
        eprintln!("skipping Java gate: JDK missing");
        return;
    }
    let cp = format!(
        "{0}/lib/jvector-4.0.0-beta.6.jar:{0}/lib/commons-math3-3.6.1.jar:{0}/lib/agrona-1.20.0.jar:{0}/lib/slf4j-api-2.0.13.jar:{0}/build",
        harness.display()
    );
    let output = std::process::Command::new(java)
        .args(["--add-modules", "jdk.incubator.vector", "--enable-preview", "-cp"])
        .arg(&cp)
        .arg("VerifyJVector")
        .arg(&path)
        .args([&DIM.to_string(), &query_doc.to_string(), "100"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "VerifyJVector failed: {}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let jv_hits: Vec<u32> = String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|l| l.split(',').next()?.parse().ok())
        .take(k)
        .collect();
    eprintln!("jvector: {jv_hits:?}");
    eprintln!("exact  : {:?}", exact.iter().map(|h| h.ord).collect::<Vec<_>>());
    assert!(jv_hits.contains(&(query_doc as u32)), "jVector missed the exact top-1");
    let exact_set: std::collections::BTreeSet<u32> = exact.iter().map(|h| h.ord).collect();
    let overlap = jv_hits.iter().filter(|h| exact_set.contains(h)).count();
    assert!(overlap * 2 >= k, "jVector top-{k} overlaps exact only {overlap}/{k}");
    eprintln!("jVector-over-our-file recall@{k}: {overlap}/{k}");
}
