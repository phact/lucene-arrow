// SPDX-License-Identifier: Apache-2.0

//! §15.3/§15.4 head-to-head: full numeric-column scan of a real segment
//! directory (default: the 16M-doc index BenchIngest writes) — our CPU and
//! GPU executors vs the recorded JVM `BenchScan` baseline.
//!
//! Run: cargo bench -p lucene-arrow-gpu --features gpu --bench scan_dir \
//!      [-- <index-dir>]

use std::time::Instant;

use lucene_arrow_codec::{DocValuesKind, SegmentDirectory};
use lucene_arrow_docvalues::read::{DocValuesPlans, DvField, DvKind, plan_doc_values};
use lucene_arrow_gpu::GpuDecoder;

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

fn main() {
    let dir_path = std::env::args()
        .skip(1)
        .find(|a| !a.starts_with('-'))
        .unwrap_or_else(|| concat!(env!("CARGO_MANIFEST_DIR"), "/../../harness/bench-index").into());
    let dir = match SegmentDirectory::open(&dir_path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("no bench index at {dir_path}: {e} (run harness BenchIngest first)");
            return;
        }
    };

    // Plan every numeric column of every segment; keep bytes in RAM
    // (warm page-cache regime).
    let mut per_seg: Vec<(DocValuesPlans, Vec<u8>)> = Vec::new();
    let mut total_values = 0u64;
    let mut payload_bytes = 0u64;
    for seg in dir.segments() {
        let fields: Vec<DvField> = seg
            .fields
            .iter()
            .filter(|f| f.doc_values == DocValuesKind::Numeric)
            .map(|f| DvField {
                number: f.number as i32,
                name: f.name.clone(),
                kind: DvKind::Numeric,
                has_skip_index: f.has_skip_index,
            })
            .collect();
        let dvm_name = seg.files.iter().find(|f| f.ends_with(".dvm")).unwrap();
        let dvd_name = seg.files.iter().find(|f| f.ends_with(".dvd")).unwrap();
        let dvm_r = dir.open_input(&seg.name, dvm_name).unwrap();
        let dvd_r = dir.open_input(&seg.name, dvd_name).unwrap();
        let dvm = dvm_r.slice(0, dvm_r.len()).unwrap().to_vec();
        let dvd = dvd_r.slice(0, dvd_r.len()).unwrap().to_vec();
        let plans = plan_doc_values(&dvm, &dvd, &fields, seg.max_doc as u32, dvd_name).unwrap();
        for p in &plans.plans {
            total_values += p.num_values;
            payload_bytes += p.payload_bytes();
        }
        per_seg.push((plans, dvd));
    }
    println!(
        "{} segment(s), {:.1} M values across numeric columns, {:.1} MiB packed payload",
        per_seg.len(),
        total_values as f64 / 1e6,
        payload_bytes as f64 / (1 << 20) as f64
    );

    // CPU: fused single-thread decode of every column.
    let mut cpu_secs = Vec::new();
    for _ in 0..5 {
        let t = Instant::now();
        for (plans, dvd) in &per_seg {
            for plan in &plans.plans {
                std::hint::black_box(lucene_arrow_cpu::decode_numeric(plan, dvd).unwrap());
            }
        }
        cpu_secs.push(t.elapsed().as_secs_f64());
    }
    let cpu = median(cpu_secs);
    println!("  CPU fused 1-thread : {:>7.3} s = {:>8.1} Mvals/s", cpu, total_values as f64 / cpu / 1e6);

    // GPU: pinned-ring upload + kernels, device-resident output
    // (e2e includes H2D; kernel-only shown separately).
    let Ok(gpu) = GpuDecoder::new() else {
        eprintln!("no CUDA device — GPU pass skipped");
        return;
    };
    let ring = gpu.new_pinned_ring(32 << 20, 4).unwrap();
    let mut e2e_secs = Vec::new();
    let mut kernel_secs = Vec::new();
    for _ in 0..5 {
        let t = Instant::now();
        let mut kernels = 0.0f64;
        for (plans, dvd) in &per_seg {
            let data = gpu.upload_pipelined(dvd, &ring).unwrap();
            let tk = Instant::now();
            for plan in &plans.plans {
                std::hint::black_box(gpu.decode_values_device(plan, &data).unwrap());
            }
            gpu.synchronize().unwrap();
            kernels += tk.elapsed().as_secs_f64();
        }
        e2e_secs.push(t.elapsed().as_secs_f64());
        kernel_secs.push(kernels);
    }
    let e2e = median(e2e_secs);
    let kern = median(kernel_secs);
    println!("  GPU e2e (H2D+kern) : {:>7.3} s = {:>8.1} Mvals/s", e2e, total_values as f64 / e2e / 1e6);
    println!("  GPU kernels only   : {:>7.3} s = {:>8.1} Mvals/s", kern, total_values as f64 / kern / 1e6);
    println!();
    println!("baseline for the same index (record separately): JVM BenchScan.");
}
