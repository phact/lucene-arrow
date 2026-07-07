// SPDX-License-Identifier: Apache-2.0

//! P6d: GPU vector indexing (NN-Descent graph + our Lucene file writers +
//! commit) vs the JVM `BenchKnnIngest` baseline on identical data.
//!
//! Run: CONDA_PREFIX=... PATH=... LD_LIBRARY_PATH=... \
//!      cargo bench -p lucene-arrow-gpu --features cuvs --bench hnsw_build

use std::time::Instant;

use lucene_arrow_codec::writer::{WriteField, commit_segments, random_segment_id, write_segment_files_ext};
use lucene_arrow_gpu::GpuDecoder;
use lucene_arrow_gpu::cuvs_knn::CuvsContext;
use lucene_arrow_vectors::file::VectorsFileBuilder;
use lucene_arrow_vectors::hnsw::{HnswFilesBuilder, navigable_from_knn};
use lucene_arrow_vectors::{Similarity, VectorEncoding};

const N: usize = 200_000;
const DIM: usize = 128;
const DEGREE: usize = 32;

fn main() {
    if GpuDecoder::new().is_err() {
        return eprintln!("no CUDA device");
    }
    let Ok(ctx) = CuvsContext::new() else {
        return eprintln!("cuVS unavailable");
    };

    eprintln!("generating {N}x{DIM} clustered vectors...");
    let vectors: Vec<f32> = (0..N)
        .flat_map(|d| {
            (0..DIM).map(move |k| {
                let cluster = (d % 100) as u64;
                let hc = (cluster ^ ((k as u64) << 32)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                let hj = ((d as u64) ^ ((k as u64) << 32) ^ 0xABCD)
                    .wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
                let center = (hc >> 11) as f32 / (1u64 << 53) as f32 * 2.0 - 1.0;
                let jitter = ((hj >> 11) as f32 / (1u64 << 53) as f32 * 2.0 - 1.0) * 0.05;
                center + jitter
            })
        })
        .collect();

    let t_total = Instant::now();
    let (graph, t_graph) = ctx.knn_graph(&vectors, DIM, DEGREE).unwrap();

    let t = Instant::now();
    let neighbors = navigable_from_knn(&graph, N, DEGREE, DEGREE);
    let t_convert = t.elapsed();

    let t = Instant::now();
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

    let tmp = tempfile::tempdir().unwrap();
    let dv = lucene_arrow_docvalues::file::DocValuesFileBuilder::new(&segment_id, "Lucene90_0");
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
        (format!("_0_{suffix}.vec"), vec.as_slice()),
        (format!("_0_{suffix}.vem"), vem.as_slice()),
        (format!("_0_{suffix}.vex"), vex.as_slice()),
    ];
    let seg =
        write_segment_files_ext(tmp.path(), "_0", segment_id, &[field], N as u32, &dvm, &dvd, &extra)
            .unwrap();
    commit_segments(tmp.path(), &[seg]).unwrap();
    let t_write = t.elapsed();
    let total = t_total.elapsed();

    println!();
    println!("GPU vector indexing, {N} x {DIM} f32, degree {DEGREE}:");
    println!("  graph (NN-Descent, 5090): {:>8.0} ms", t_graph.as_secs_f64() * 1e3);
    println!("  convert (navigable)     : {:>8.0} ms", t_convert.as_secs_f64() * 1e3);
    println!("  files + commit          : {:>8.0} ms", t_write.as_secs_f64() * 1e3);
    println!("  TOTAL                   : {:>8.0} ms = {:.0} kdocs/s",
        total.as_secs_f64() * 1e3, N as f64 / total.as_secs_f64() / 1e3);
    println!("  segment at {:?} (CheckIndex it via harness/run.sh check)", tmp.keep());
    println!("JVM baseline on identical data: harness BenchKnnIngest.");
}
