// SPDX-License-Identifier: Apache-2.0

//! Exact-KNN scale bench: 1M × 128-d f32, device-resident (SPEC §11.6 —
//! scoring is memory-bound; this measures how fast flat search really is
//! before anyone reaches for a graph).
//!
//! Run: `cargo bench -p lucene-arrow-gpu --features gpu --bench knn_scale`

use std::time::Instant;

use lucene_arrow_gpu::{GpuDecoder, knn::FlatKnn};
use lucene_arrow_vectors::Similarity;

const N: u64 = 1 << 20;
const DIM: u32 = 128;
const K: usize = 10;
const ITERS: usize = 10;

fn main() {
    let gpu = match GpuDecoder::new() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("no CUDA device: {e}");
            return;
        }
    };
    let knn = FlatKnn::new(&gpu).unwrap();

    eprintln!("generating {} × {DIM}d vectors...", N);
    let payload: Vec<u8> = (0..N as usize * DIM as usize)
        .map(|i| ((i as u64).wrapping_mul(0x9E37_79B9) % 2003) as f32 * 0.03125 - 31.0)
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let gb = payload.len() as f64 / 1e9;

    let ring = gpu.new_pinned_ring(32 << 20, 4).unwrap();
    let t = Instant::now();
    let data = gpu.upload_pipelined(&payload, &ring).unwrap();
    let up = t.elapsed().as_secs_f64();
    eprintln!("uploaded {gb:.2} GB in {:.1} ms ({:.1} GB/s)", up * 1e3, gb / up);

    // The payload IS the device data here (one Raw block over everything).
    let payload_dev = {
        use lucene_arrow_core::plan::{BlockDecode, Coverage, DecodePlan, FieldId, PLAN_VERSION};
        let plan = DecodePlan {
            plan_version: PLAN_VERSION,
            column: FieldId::new(0, "bench"),
            file: "bench.vec".into(),
            arrow_type: arrow_schema::DataType::FixedSizeList(
                std::sync::Arc::new(arrow_schema::Field::new(
                    "item",
                    arrow_schema::DataType::Float32,
                    false,
                )),
                DIM as i32,
            ),
            blocks: vec![BlockDecode::Raw { offset: 0, len: payload.len() as u64 }],
            coverage: Coverage::Dense { num_docs: N as u32 },
            num_values: N,
        };
        gpu.vector_payload_device(&plan, &data).unwrap()
    };

    for (name, sim, nq) in [
        ("euclidean, 1 query", Similarity::Euclidean, 1usize),
        ("euclidean, 16 queries", Similarity::Euclidean, 16),
        ("dot_product, 16 queries", Similarity::DotProduct, 16),
    ] {
        let queries: Vec<f32> = (0..nq * DIM as usize)
            .map(|i| (i % 97) as f32 * 0.5 - 24.0)
            .collect();
        let mut secs = Vec::new();
        for i in 0..3 + ITERS {
            let t = Instant::now();
            let hits =
                knn.search(&payload_dev, N, DIM, sim, &queries, K).unwrap();
            let dt = t.elapsed().as_secs_f64();
            std::hint::black_box(&hits);
            if i >= 3 {
                secs.push(dt);
            }
        }
        secs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let t_med = secs[secs.len() / 2];
        let pairs = N as f64 * nq as f64;
        println!(
            "  {:<24} {:>8.2} ms | {:>7.2} Gpair/s | {:>8.1} GB/s scanned | {:>6.1} GFLOP/s",
            name,
            t_med * 1e3,
            pairs / t_med / 1e9,
            gb * nq as f64 / t_med,
            pairs * (DIM as f64 * 3.0) / t_med / 1e9,
        );
    }
}
