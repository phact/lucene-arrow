// SPDX-License-Identifier: Apache-2.0

//! P6c acceptance: GPU-built graph (NN-Descent, the CAGRA substrate) →
//! our `.vem`/`.vex` + flat files → complete segment commit. Gates:
//! CheckIndex passes (it validates graph structure), and a **Java Lucene
//! KNN query** over the segment finds the true nearest neighbors our
//! exact GPU search reports.

#![cfg(feature = "cuvs")]

use lucene_arrow_codec::writer::{WriteField, commit_segments, random_segment_id, write_segment_files_ext};
use lucene_arrow_gpu::cuvs_knn::CuvsContext;
use lucene_arrow_gpu::{GpuDecoder, knn::FlatKnn};
use lucene_arrow_vectors::file::VectorsFileBuilder;
use lucene_arrow_vectors::hnsw::HnswFilesBuilder;
use lucene_arrow_vectors::{Similarity, VectorEncoding};

const N: usize = 4000;
const DIM: usize = 64;
const DEGREE: usize = 32;

/// Clustered data (100 centers + small jitter): the regime graph search
/// is built for — uniform random high-dim is adversarial for greedy
/// routing and says nothing about the file format.
fn vecf(seed: usize) -> Vec<f32> {
    let cluster = seed % 100;
    (0..DIM)
        .map(|k| {
            let hc = ((cluster as u64) ^ ((k as u64) << 32))
                .wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let hj = ((seed as u64) ^ ((k as u64) << 32) ^ 0xABCD)
                .wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
            let center = (hc >> 11) as f32 / (1u64 << 53) as f32 * 2.0 - 1.0;
            let jitter = ((hj >> 11) as f32 / (1u64 << 53) as f32 * 2.0 - 1.0) * 0.05;
            center + jitter
        })
        .collect()
}

#[test]
fn gpu_graph_segment_passes_checkindex_and_java_knn() {
    let Ok(gpu) = GpuDecoder::new() else { return };
    let Ok(ctx) = CuvsContext::new() else {
        eprintln!("cuVS unavailable");
        return;
    };

    let vectors: Vec<f32> = (0..N).flat_map(vecf).collect();
    let (graph, build) = ctx.knn_graph(&vectors, DIM, DEGREE).unwrap();
    eprintln!("NN-Descent graph {N}x{DEGREE} built in {build:?}");
    let neighbors =
        lucene_arrow_vectors::hnsw::navigable_from_knn(&graph, N, DEGREE, DEGREE);

    // Files: flat pair + graph pair, all under the HNSW per-field suffix.
    let segment_id = random_segment_id();
    let suffix = "Lucene99HnswVectorsFormat_0";
    let payload: Vec<u8> = vectors.iter().flat_map(|v| v.to_le_bytes()).collect();
    let docs: Vec<u32> = (0..N as u32).collect();
    let mut flat = VectorsFileBuilder::new(&segment_id, suffix);
    flat.add_field(0, VectorEncoding::Float32, Similarity::Euclidean, DIM as u32, &docs, &payload, N as u32)
        .unwrap();
    let (vemf, vec) = flat.finish();
    let mut hnsw = HnswFilesBuilder::new(&segment_id, suffix);
    hnsw.add_field(0, VectorEncoding::Float32, Similarity::Euclidean, DIM as u32, &neighbors, (DEGREE / 2) as u32)
        .unwrap();
    let (vem, vex) = hnsw.finish();

    // Segment: one vector field, no doc values.
    let tmp = tempfile::tempdir().unwrap();
    let dv = lucene_arrow_docvalues::file::DocValuesFileBuilder::new(&segment_id, "Lucene90_0");
    let (dvm, dvd) = dv.finish();
    let field = WriteField {
        name: "emb".into(),
        number: 0,
        doc_values_type: 0,
        vector_dim: DIM as u32,
        vector_encoding: 1, // FLOAT32
        vector_similarity: 0, // EUCLIDEAN
        index_options: 0,
    };
    let extra = [
        (format!("_0_{suffix}.vemf"), vemf.as_slice()),
        (format!("_0_{suffix}.vec"), vec.as_slice()),
        (format!("_0_{suffix}.vem"), vem.as_slice()),
        (format!("_0_{suffix}.vex"), vex.as_slice()),
    ];
    let seg = write_segment_files_ext(
        tmp.path(), "_0", segment_id, &[field], N as u32, &dvm, &dvd, &extra,
    )
    .unwrap();
    commit_segments(tmp.path(), &[seg]).unwrap();

    // Self-parse: replicate Java's reader over our .vem/.vex and require
    // the decoded adjacency to equal what we fed in (writer correctness,
    // independent of search behavior).
    {
        use lucene_arrow_core::cursor::Cursor;
        let header = lucene_arrow_core::cursor::read_index_header(
            &vem, "Lucene99HnswVectorsFormatMeta", 1, 1).unwrap();
        let mut c = Cursor::at(&vem, header.length);
        assert_eq!(c.le_i32().unwrap(), 0); // field number
        c.le_i32().unwrap(); // encoding
        c.le_i32().unwrap(); // similarity
        let vex_off = c.vlong().unwrap() as usize;
        let _vex_len = c.vlong().unwrap();
        assert_eq!(c.vint().unwrap() as usize, DIM);
        assert_eq!(c.le_i32().unwrap() as usize, N);
        let _m = c.vint().unwrap();
        assert_eq!(c.vint().unwrap(), 1); // numLevels
        let offsets_start = c.le_i64().unwrap() as usize;
        let shift = c.vint().unwrap() as u32;
        let meta_blocks =
            lucene_arrow_docvalues::monotonic::read_meta(&mut c, N as u64, shift).unwrap();
        let _offsets_len = c.le_i64().unwrap();
        let starts = lucene_arrow_docvalues::monotonic::decode(
            &meta_blocks, &vex[offsets_start..], N as u64, shift).unwrap();
        for (node, want) in neighbors.iter().enumerate() {
            let mut want: Vec<u32> = want.clone();
            want.sort_unstable();
            want.dedup();
            let mut r = Cursor::at(&vex, vex_off + starts[node] as usize);
            let cnt = r.vint().unwrap() as usize;
            let mut vals = vec![0i32; cnt];
            let mut cur = std::io::Cursor::new(&vex[r.pos()..]);
            bearing::encoding::group_vint::read_group_vints(&mut cur, &mut vals, cnt).unwrap();
            for i in 1..cnt {
                vals[i] += vals[i - 1];
            }
            let got: Vec<u32> = vals.iter().map(|&x| x as u32).collect();
            assert_eq!(got, want, "node {node} adjacency mismatch");
        }
        eprintln!("self-parse: adjacency round-trips for all {N} nodes");
    }

    // Reopen with our reader: vectors decode back.
    let dir = lucene_arrow_codec::SegmentDirectory::open(tmp.path()).unwrap();
    let smeta = &dir.segments()[0];
    assert!(smeta.field("emb").unwrap().has_vectors);
    assert_eq!(smeta.field("emb").unwrap().vector_dimension, DIM as u32);

    // Exact reference for query doc 7: our FlatKnn.
    let k = 10usize;
    let query_doc = 7usize;
    let query: Vec<f32> = vecf(query_doc);
    let data = gpu.upload(&payload).unwrap();
    let plan = lucene_arrow_core::plan::DecodePlan {
        plan_version: lucene_arrow_core::plan::PLAN_VERSION,
        column: lucene_arrow_core::plan::FieldId::new(0, "emb"),
        file: "x.vec".into(),
        arrow_type: arrow_schema::DataType::FixedSizeList(
            std::sync::Arc::new(arrow_schema::Field::new("item", arrow_schema::DataType::Float32, false)),
            DIM as i32,
        ),
        blocks: vec![lucene_arrow_core::plan::BlockDecode::Raw { offset: 0, len: payload.len() as u64 }],
        coverage: lucene_arrow_core::plan::Coverage::Dense { num_docs: N as u32 },
        num_values: N as u64,
    };
    let dev = gpu.vector_payload_device(&plan, &data).unwrap();
    let flatknn = FlatKnn::new(&gpu).unwrap();
    let exact =
        &flatknn.search(&dev, N as u64, DIM as u32, Similarity::Euclidean, &query, k).unwrap()[0];
    assert_eq!(exact[0].ord as usize, query_doc, "query doc is its own NN");

    // CheckIndex (validates the graph structurally).
    let java = std::path::Path::new("/home/tato/.jbang/cache/jdks/21/bin/java");
    let jar = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../harness/lib/lucene-core-10.3.2.jar");
    if !java.exists() || !jar.exists() {
        eprintln!("skipping Java gates: JDK/jar missing");
        return;
    }
    let output = std::process::Command::new(java)
        .args(["--add-modules", "jdk.incubator.vector", "-cp"])
        .arg(&jar)
        .arg("org.apache.lucene.index.CheckIndex")
        .arg(tmp.path())
        .args(["-level", "2"])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success() && stdout.contains("No problems were detected"),
        "CheckIndex failed:\n{stdout}\n{}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Java KNN query over OUR graph: its hits must include the exact
    // top-1 (the query doc itself) and overlap the exact top-k.
    let build_dir = tempfile::tempdir().unwrap();
    let harness = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../harness");
    assert!(
        std::process::Command::new(java.parent().unwrap().join("javac"))
            .args(["-cp"])
            .arg(&jar)
            .args(["-d"])
            .arg(build_dir.path())
            .arg(harness.join("src/VerifyKnn.java"))
            .status()
            .unwrap()
            .success()
    );
    let output = std::process::Command::new(java)
        .args(["--add-modules", "jdk.incubator.vector", "-cp"])
        .arg(format!("{}:{}", jar.display(), build_dir.path().display()))
        .arg("VerifyKnn")
        .arg(tmp.path())
        .args(["emb", &query_doc.to_string(), "100"]) // wide beam; we grade its top-k
        .output()
        .unwrap();
    assert!(output.status.success(), "VerifyKnn failed: {}", String::from_utf8_lossy(&output.stderr));
    let java_hits: Vec<u32> = String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|l| l.split(',').next()?.parse().ok())
        .take(k)
        .collect();
    eprintln!("java KNN: {java_hits:?}");
    eprintln!("exact   : {:?}", exact.iter().map(|h| h.ord).collect::<Vec<_>>());
    assert!(java_hits.contains(&(query_doc as u32)), "Java missed the exact top-1");
    let exact_set: std::collections::BTreeSet<u32> = exact.iter().map(|h| h.ord).collect();
    let overlap = java_hits.iter().filter(|h| exact_set.contains(h)).count();
    assert!(overlap * 2 >= k, "Java top-{k} overlaps exact only {overlap}/{k}");
    eprintln!("Java-over-our-graph recall@{k}: {}/{k}", overlap);
}
