// SPDX-License-Identifier: Apache-2.0

//! Vector-search throughput + recall on ONE dataset, five ways: our exact
//! GPU FlatKnn (Rust); jVector and Lucene HNSW reading the graph *we*
//! wrote; and jVector and Lucene searching graphs built by their *own*
//! native builders. The native rows isolate graph quality — is our graph
//! bad in general, or bad for a specific engine? Both our files now carry
//! a real multi-layer hierarchy (CAGRA -> cuvsHnswFromCagra -> parse ->
//! multi-level Lucene .vem/.vex and jVector OnDiskGraphIndex) and land at
//! or near their native builders.)
//! Ground-truth top-k is the FlatKnn exact result; node/doc ids share the
//! ordinal space, so recall is directly comparable.
//!
//! Run: CONDA_PREFIX=... PATH=...pixi.../bin:$PATH LD_LIBRARY_PATH=...pixi.../lib \
//!      cargo bench -p lucene-arrow-gpu --features cuvs --bench vector_search

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use lucene_arrow_codec::writer::{
    WriteField, commit_segments, random_segment_id, write_segment_files_ext,
};
use lucene_arrow_gpu::GpuDecoder;
use lucene_arrow_gpu::cuvs_knn::CuvsContext;
use lucene_arrow_gpu::knn::FlatKnn;
use lucene_arrow_vectors::file::VectorsFileBuilder;
use lucene_arrow_vectors::hnsw::{HnswFilesBuilder, small_world_from_cagra};
use lucene_arrow_vectors::jvector::write_index_multi;
use lucene_arrow_vectors::{Similarity, VectorEncoding};

const N: usize = 100_000;
const DIM: usize = 128;
const DEGREE: usize = 32;
const Q: usize = 1000;
const K: usize = 10;
const EF: usize = 100; // graph-search beam width (jVector / Lucene); FlatKnn is exact

fn vecf(seed: usize) -> Vec<f32> {
    let cluster = seed % 200;
    (0..DIM)
        .map(|k| {
            let hc = ((cluster as u64) ^ ((k as u64) << 32)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let hj =
                ((seed as u64) ^ ((k as u64) << 32) ^ 0xABCD).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
            let center = (hc >> 11) as f32 / (1u64 << 53) as f32 * 2.0 - 1.0;
            let jitter = ((hj >> 11) as f32 / (1u64 << 53) as f32 * 2.0 - 1.0) * 0.1;
            center + jitter
        })
        .collect()
}

fn main() {
    let Ok(gpu) = GpuDecoder::new() else { return eprintln!("no CUDA device") };
    let Ok(ctx) = CuvsContext::new() else { return eprintln!("cuVS unavailable") };

    eprintln!("building {N}×{DIM} dataset + graph...");
    let vectors: Vec<f32> = (0..N).flat_map(vecf).collect();
    // Queries = perturbed dataset points, so a true near neighbor exists.
    let queries: Vec<f32> = (0..Q)
        .flat_map(|qi| {
            let d = (qi * 131) % N;
            let base = &vectors[d * DIM..(d + 1) * DIM];
            base.iter().map(|&v| v + (qi as f32 % 7.0 - 3.0) * 1e-3).collect::<Vec<f32>>()
        })
        .collect();

    let (graph, gdeg) = ctx.cagra_graph(&vectors, DIM, DEGREE).unwrap();
    let neighbors = small_world_from_cagra(&graph, N, gdeg, DEGREE);

    let tmp = tempfile::tempdir().unwrap();
    let seg_id = random_segment_id();
    let suffix = "Lucene99HnswVectorsFormat_0";
    let payload: Vec<u8> = vectors.iter().flat_map(|v| v.to_le_bytes()).collect();

    // Multi-level HNSW hierarchy for the Lucene segment: CAGRA → cuVS HNSW
    // hierarchy (hierarchy=CPU, standard hnswlib) → parse → our multi-level
    // .vem/.vex writer.
    let hfile = tmp.path().join("h.hnsw");
    let efc: i32 = std::env::var("EFC").ok().and_then(|v| v.parse().ok()).unwrap_or(100);
    ctx.cagra_to_hnswlib(&vectors, DIM, DEGREE / 2, efc, hfile.to_str().unwrap()).unwrap();
    let parsed =
        lucene_arrow_vectors::hnsw::parse_hnswlib(&std::fs::read(&hfile).unwrap()).unwrap();

    // jVector file — multi-layer, from the same HNSW hierarchy.
    let jv_path = tmp.path().join("graph.jvector");
    std::fs::write(&jv_path, write_index_multi(&vectors, DIM, &parsed).unwrap()).unwrap();
    let _ = &neighbors; // (single-level small-world still available for A/B)

    // Lucene HNSW segment (flat + graph).
    let docs: Vec<u32> = (0..N as u32).collect();
    let mut flatb = VectorsFileBuilder::new(&seg_id, suffix);
    flatb
        .add_field(0, VectorEncoding::Float32, Similarity::Euclidean, DIM as u32, &docs, &payload, N as u32)
        .unwrap();
    let (vemf, vecbytes) = flatb.finish();
    let mut hnsw = HnswFilesBuilder::new(&seg_id, suffix);
    hnsw.add_field_multi(0, VectorEncoding::Float32, Similarity::Euclidean, DIM as u32, N, parsed.m, &parsed.levels)
        .unwrap();
    let (vem, vex) = hnsw.finish();
    let lucdir = tmp.path().join("lucene");
    std::fs::create_dir_all(&lucdir).unwrap();
    let dv = lucene_arrow_docvalues::file::DocValuesFileBuilder::new(&seg_id, "Lucene90_0");
    let (dvm, dvd) = dv.finish();
    let field = WriteField {
        name: "emb".into(),
        number: 0,
        doc_values_type: 0,
        vector_dim: DIM as u32,
        vector_encoding: 1,
        vector_similarity: 0,
        index_options: 0,
    };
    let extra = [
        (format!("_0_{suffix}.vemf"), vemf.as_slice()),
        (format!("_0_{suffix}.vec"), vecbytes.as_slice()),
        (format!("_0_{suffix}.vem"), vem.as_slice()),
        (format!("_0_{suffix}.vex"), vex.as_slice()),
    ];
    let seg = write_segment_files_ext(&lucdir, "_0", seg_id, &[field], N as u32, &dvm, &dvd, &extra)
        .unwrap();
    commit_segments(&lucdir, &[seg]).unwrap();

    // FlatKnn: exact top-k (ground truth) + QPS.
    let data = gpu.upload(&payload).unwrap();
    let plan = lucene_arrow_core::plan::DecodePlan {
        plan_version: lucene_arrow_core::plan::PLAN_VERSION,
        column: lucene_arrow_core::plan::FieldId::new(0, "emb"),
        file: "x.vec".into(),
        arrow_type: arrow_schema::DataType::FixedSizeList(
            Arc::new(arrow_schema::Field::new("item", arrow_schema::DataType::Float32, false)),
            DIM as i32,
        ),
        blocks: vec![lucene_arrow_core::plan::BlockDecode::Raw { offset: 0, len: payload.len() as u64 }],
        coverage: lucene_arrow_core::plan::Coverage::Dense { num_docs: N as u32 },
        num_values: N as u64,
    };
    let dev = gpu.vector_payload_device(&plan, &data).unwrap();
    let flat = FlatKnn::new(&gpu).unwrap();
    let exact = flat
        .search(&dev, N as u64, DIM as u32, Similarity::Euclidean, &queries, K)
        .unwrap();
    let mut best = f64::MAX;
    for _ in 0..3 {
        let t = Instant::now();
        let _ = flat.search(&dev, N as u64, DIM as u32, Similarity::Euclidean, &queries, K).unwrap();
        best = best.min(t.elapsed().as_secs_f64());
    }
    let flat_qps = Q as f64 / best;

    // Ground-truth ids + query vectors → files for the Java harness.
    let gt: Vec<i32> = exact.iter().flat_map(|hs| hs.iter().map(|h| h.ord as i32)).collect();
    let qbin = tmp.path().join("queries.bin");
    std::fs::write(&qbin, queries.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())
        .unwrap();
    let gtbin = tmp.path().join("gt.bin");
    std::fs::write(&gtbin, gt.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
    // Raw dataset too, so Java can build each engine's native graph.
    let vbin = tmp.path().join("vectors.bin");
    std::fs::write(&vbin, &payload).unwrap();

    // Java: jVector + Lucene HNSW search over the files we wrote.
    let java = std::path::Path::new("/home/tato/.jbang/cache/jdks/21/bin/java");
    let harness = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../harness");
    if !java.exists() {
        eprintln!("no JDK 21 — printing FlatKnn only");
        print_table(flat_qps, None, None, None, None);
        return;
    }
    let cp = [
        "jvector-4.0.0-beta.6.jar",
        "commons-math3-3.6.1.jar",
        "agrona-1.20.0.jar",
        "slf4j-api-2.0.13.jar",
        "lucene-core-10.3.2.jar",
    ]
    .iter()
    .map(|j| harness.join("lib").join(j).display().to_string())
    .collect::<Vec<_>>()
    .join(":");
    let build = tmp.path().join("classes");
    std::fs::create_dir_all(&build).unwrap();
    let javac = java.parent().unwrap().join("javac");
    let ok = std::process::Command::new(&javac)
        .args(["-cp", &cp, "-d"])
        .arg(&build)
        .arg(harness.join("src/BenchVectorSearch.java"))
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if !ok {
        eprintln!("javac failed — printing FlatKnn only");
        print_table(flat_qps, None, None, None, None);
        return;
    }
    let out = std::process::Command::new(java)
        .args(["--add-modules", "jdk.incubator.vector", "-cp"])
        .arg(format!("{cp}:{}", build.display()))
        .arg("BenchVectorSearch")
        .arg(&jv_path)
        .arg(&lucdir)
        .arg(&qbin)
        .arg(&gtbin)
        .args([&DIM.to_string(), &K.to_string(), &EF.to_string()])
        .arg(&vbin)
        .output()
        .unwrap();
    if !out.status.success() {
        eprintln!("BenchVectorSearch failed:\n{}", String::from_utf8_lossy(&out.stderr));
        print_table(flat_qps, None, None, None, None);
        return;
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    let row = |name: &str| -> Option<(f64, f64)> {
        stdout.lines().find(|l| l.split(',').next() == Some(name)).and_then(|l| {
            let mut p = l.split(',').skip(1);
            Some((p.next()?.parse().ok()?, p.next()?.parse().ok()?))
        })
    };
    print_table(
        flat_qps,
        row("jvector"),
        row("lucene_hnsw"),
        row("jvector_native"),
        row("lucene_native"),
    );
}

fn print_table(
    flat_qps: f64,
    jvector: Option<(f64, f64)>,
    lucene: Option<(f64, f64)>,
    jvector_native: Option<(f64, f64)>,
    lucene_native: Option<(f64, f64)>,
) {
    let line = |name: &str, v: Option<(f64, f64)>| match v {
        Some((qps, r)) => println!("  {name:<30} | {qps:>6.0} | {r:>8.3}"),
        None => println!("  {name:<30} |      - |        -"),
    };
    println!();
    println!("{N} × {DIM} f32, {Q} queries, k={K}, ef={EF}, single-thread search (RTX 5090)");
    println!("  engine                         |    QPS | recall@{K}");
    println!("  -------------------------------+--------+----------");
    println!("  {:<30} | {flat_qps:>6.0} |    1.000 (ref)", "FlatKnn (ours, exact GPU)");
    line("jVector — OUR graph", jvector);
    line("jVector — its NATIVE graph", jvector_native);
    line("Lucene HNSW — OUR graph", lucene);
    line("Lucene HNSW — its NATIVE graph", lucene_native);
}
