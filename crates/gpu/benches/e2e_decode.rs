// SPDX-License-Identifier: Apache-2.0

//! End-to-end decode of a real-format `.dvd`/`.dvm` pair (SPEC §15 metric
//! 3): plan → transfer → kernel → columns, versus the CPU reference.
//!
//! Run: `cargo bench -p lucene-arrow-gpu --features gpu --bench e2e_decode`
//!
//! The file lives in host RAM (warm page-cache regime, §11.0 row 3).
//! Three configs: CPU fused single-thread; GPU with a pageable one-shot
//! upload; GPU through a pinned write-combined staging buffer. GPU output
//! stays device-resident (the analytics hand-off shape); the dtoh cost is
//! reported separately.

use std::time::Instant;

use lucene_arrow_docvalues::file::DocValuesFileBuilder;
use lucene_arrow_docvalues::read::{DvField, DvKind, plan_doc_values};
use lucene_arrow_gpu::GpuDecoder;

const NUM_DOCS: u32 = 32 << 20; // 32 Mi docs
const ITERS: usize = 5;

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

fn main() {
    let gpu = match GpuDecoder::new() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("no CUDA device: {e}");
            return;
        }
    };

    eprintln!("building {} Mi-doc segment files in memory...", NUM_DOCS >> 20);
    let all_docs: Vec<u32> = (0..NUM_DOCS).collect();
    let sparse_docs: Vec<u32> = (0..NUM_DOCS).filter(|d| d % 4 != 3).collect();

    let mut builder = DocValuesFileBuilder::new(b"e2e-bench-seg-id", "");
    // f0: gcd-packed (prices), f1: direct 20-bit, f2: full 64-bit,
    // f3: sparse 75% coverage (DENSE DISI blocks) 16-bit.
    builder
        .add_numeric(
            0,
            &all_docs,
            &all_docs.iter().map(|&d| 1_000_000 + (d as i64 % 4096) * 25).collect::<Vec<_>>(),
            NUM_DOCS,
        )
        .unwrap();
    builder
        .add_numeric(
            1,
            &all_docs,
            &all_docs.iter().map(|&d| (d as i64).wrapping_mul(0x9E37) & 0xF_FFFF).collect::<Vec<_>>(),
            NUM_DOCS,
        )
        .unwrap();
    builder
        .add_numeric(
            2,
            &all_docs,
            &all_docs.iter().map(|&d| (d as i64).wrapping_mul(0x9E37_79B9_7F4A_7C15u64 as i64)).collect::<Vec<_>>(),
            NUM_DOCS,
        )
        .unwrap();
    builder
        .add_numeric(
            3,
            &sparse_docs,
            &sparse_docs.iter().map(|&d| (d as i64) & 0xFFFF).collect::<Vec<_>>(),
            NUM_DOCS,
        )
        .unwrap();
    let (dvm, dvd) = builder.finish();
    let payload_gb = dvd.len() as f64 / 1e9;
    let total_rows = NUM_DOCS as u64 * 4;
    eprintln!(".dvd = {:.2} GB, 4 columns × {} Mi docs", payload_gb, NUM_DOCS >> 20);

    let fields: Vec<DvField> = (0..4)
        .map(|n| DvField {
            number: n,
            name: format!("f{n}"),
            kind: DvKind::Numeric,
            has_skip_index: false,
        })
        .collect();
    let plans = plan_doc_values(&dvm, &dvd, &fields, NUM_DOCS, "_0.dvd").unwrap();
    assert_eq!(plans.plans.len(), 4);

    // --- CPU reference (fused, single thread) ---
    let mut cpu_secs = Vec::new();
    for _ in 0..3 {
        let t = Instant::now();
        for plan in &plans.plans {
            std::hint::black_box(lucene_arrow_cpu::decode_numeric(plan, &dvd).unwrap());
        }
        cpu_secs.push(t.elapsed().as_secs_f64());
    }
    let cpu_t = median(cpu_secs);

    // --- GPU: pageable one-shot upload, decode all columns, stay resident ---
    let mut pageable_secs = Vec::new();
    for _ in 0..ITERS {
        let t = Instant::now();
        let data = gpu.upload(&dvd).unwrap();
        let outs: Vec<_> =
            plans.plans.iter().map(|p| gpu.decode_values_device(p, &data).unwrap()).collect();
        gpu.synchronize().unwrap();
        pageable_secs.push(t.elapsed().as_secs_f64());
        drop(outs);
    }
    let pageable_t = median(pageable_secs);

    // --- GPU: pinned ring, chunked copy/DMA overlap (SPEC §11.2) ---
    let ring = gpu.new_pinned_ring(32 << 20, 4).unwrap();
    let mut pinned_secs = Vec::new();
    let mut upload_secs = Vec::new();
    let mut kernel_secs = Vec::new();
    for _ in 0..ITERS {
        let t = Instant::now();
        let data = gpu.upload_pipelined(&dvd, &ring).unwrap();
        let t_up = t.elapsed().as_secs_f64();
        let tk = Instant::now();
        let outs: Vec<_> =
            plans.plans.iter().map(|p| gpu.decode_values_device(p, &data).unwrap()).collect();
        gpu.synchronize().unwrap();
        kernel_secs.push(tk.elapsed().as_secs_f64());
        pinned_secs.push(t.elapsed().as_secs_f64());
        upload_secs.push(t_up);
        drop(outs);
    }
    let pinned_t = median(pinned_secs);

    println!();
    println!(
        "end-to-end: 4 columns × {} Mi docs, {:.2} GB packed (warm host RAM), median",
        NUM_DOCS >> 20,
        payload_gb
    );
    println!();
    println!("  config                    |  wall ms | payload GB/s | Grows/s");
    println!("  --------------------------+----------+--------------+--------");
    for (name, t) in [
        ("CPU fused, 1 thread", cpu_t),
        ("GPU, pageable upload", pageable_t),
        ("GPU, pinned ring (32MB x4)", pinned_t),
    ] {
        println!(
            "  {:<25} | {:>8.1} | {:>12.2} | {:>6.2}",
            name,
            t * 1e3,
            payload_gb / t,
            total_rows as f64 / t / 1e9,
        );
    }
    println!();
    println!(
        "  pinned breakdown: upload {:.1} ms ({:.1} GB/s H2D), kernels+sync {:.1} ms",
        median(upload_secs.clone()) * 1e3,
        payload_gb / median(upload_secs),
        median(kernel_secs) * 1e3,
    );
    // Headline in the same unit as the JVM BenchScan baseline, for the
    // bench-all report's speedup (best config = pinned ring).
    println!("  best (pinned ring): {:.1} Mvals/s", total_rows as f64 / pinned_t / 1e6);
}
