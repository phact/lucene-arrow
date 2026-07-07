// SPDX-License-Identifier: Apache-2.0

//! §11.6 / P6b: FlatKnn (ours, exact) vs cuVS brute force (exact) vs
//! CAGRA (graph ANN) on identical data — recall, QPS, build cost.
//!
//! Run: CONDA_PREFIX=$PWD/.pixi/envs/default \
//!      PATH=$PWD/.pixi/envs/default/bin:$PATH \
//!      LD_LIBRARY_PATH=$PWD/.pixi/envs/default/lib \
//!      cargo bench -p lucene-arrow-gpu --features cuvs --bench knn_threeway

use std::time::Instant;

use lucene_arrow_gpu::cuvs_knn::CuvsContext;
use lucene_arrow_gpu::{GpuDecoder, knn::FlatKnn};
use lucene_arrow_vectors::Similarity;

const N: usize = 1 << 20;
const DIM: usize = 128;
const K: usize = 10;
const NQ: usize = 64;

fn vecf(seed: usize) -> impl Iterator<Item = f32> {
    (0..DIM).map(move |k| {
        let h = (seed as u64 ^ (k as u64) << 32).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        (h >> 11) as f32 / (1u64 << 53) as f32 * 2.0 - 1.0
    })
}

fn main() {
    let gpu = match GpuDecoder::new() {
        Ok(g) => g,
        Err(e) => return eprintln!("no CUDA device: {e}"),
    };
    let ctx = match CuvsContext::new() {
        Ok(c) => c,
        Err(e) => return eprintln!("cuVS unavailable: {e}"),
    };

    eprintln!("generating {}x{DIM} vectors...", N);
    let vectors: Vec<f32> = (0..N).flat_map(vecf).collect();
    let queries: Vec<f32> =
        (0..NQ).flat_map(|i| vecf(i * 8191 + 3).map(move |v| v + (i as f32) * 1e-3)).collect();

    // --- FlatKnn over device-resident payload ---
    let payload: Vec<u8> = vectors.iter().flat_map(|v| v.to_le_bytes()).collect();
    let ring = gpu.new_pinned_ring(32 << 20, 4).unwrap();
    let data = gpu.upload_pipelined(&payload, &ring).unwrap();
    let plan = lucene_arrow_core::plan::DecodePlan {
        plan_version: lucene_arrow_core::plan::PLAN_VERSION,
        column: lucene_arrow_core::plan::FieldId::new(0, "v"),
        file: "b.vec".into(),
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
    let dev_payload = gpu.vector_payload_device(&plan, &data).unwrap();
    let flat = FlatKnn::new(&gpu).unwrap();

    let time = |f: &mut dyn FnMut()| -> f64 {
        let mut best = f64::MAX;
        for _ in 0..3 {
            let t = Instant::now();
            f();
            best = best.min(t.elapsed().as_secs_f64());
        }
        best
    };

    let mut flat_hits = Vec::new();
    let t_flat = time(&mut || {
        flat_hits = flat
            .search(&dev_payload, N as u64, DIM as u32, Similarity::Euclidean, &queries, K)
            .unwrap();
    });

    // --- cuVS brute force (exact; includes its own H2D of dataset once) ---
    let t_bf_total = Instant::now();
    let bf_hits = ctx.brute_force(&vectors, DIM, &queries, K).unwrap();
    let t_bf = t_bf_total.elapsed().as_secs_f64();

    // --- CAGRA build + search (times measured inside) ---
    let (cagra_hits, build, search) = ctx.cagra(&vectors, DIM, &queries, K).unwrap();
    let t_cagra_search = search.as_secs_f64();

    let recall = |hits: &Vec<Vec<lucene_arrow_gpu::knn::Hit>>| -> f64 {
        let mut shared = 0usize;
        for (a, b) in flat_hits.iter().zip(hits) {
            let set: std::collections::BTreeSet<u32> = a.iter().map(|h| h.ord).collect();
            shared += b.iter().filter(|h| set.contains(&h.ord)).count();
        }
        shared as f64 / (NQ * K) as f64
    };

    println!();
    println!("{N} x {DIM} f32, {NQ} queries, k={K} (RTX 5090)");
    println!("  engine            |  build ms | search ms |    QPS | recall vs FlatKnn");
    println!("  ------------------+-----------+-----------+--------+------------------");
    println!(
        "  FlatKnn (ours)    | {:>9} | {:>9.1} | {:>6.0} | {:>17}",
        "-", t_flat * 1e3, NQ as f64 / t_flat, "1.000 (reference)"
    );
    println!(
        "  cuVS brute force  | {:>9} | {:>9.1} | {:>6.0} | {:>17.3}",
        "-", t_bf * 1e3, NQ as f64 / t_bf, recall(&bf_hits)
    );
    println!(
        "  cuVS CAGRA        | {:>9.0} | {:>9.1} | {:>6.0} | {:>17.3}",
        build.as_secs_f64() * 1e3,
        t_cagra_search * 1e3,
        NQ as f64 / t_cagra_search.max(1e-9),
        recall(&cagra_hits)
    );
}
