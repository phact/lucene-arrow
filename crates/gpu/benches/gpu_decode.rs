// SPDX-License-Identifier: Apache-2.0

//! Raw-kernel decode throughput on device-resident input (SPEC §11.0, §15
//! metric 1), with the single-thread CPU reference for scale.
//!
//! Run: `cargo bench -p lucene-arrow-gpu --features gpu`
//!
//! Reports, per bit width: packed input GB/s (payload bytes / kernel wall)
//! and output Gvals/s. Input is uploaded once; timing covers launch +
//! stream sync only — storage-utilization numbers (§15 metric 2) come
//! later with the extent pipeline.

use std::time::Instant;

use lucene_arrow_core::plan::{BlockDecode, Coverage, DecodePlan, FieldId, PLAN_VERSION};
use lucene_arrow_docvalues::direct;
use lucene_arrow_gpu::GpuDecoder;

const NUM_VALUES: usize = 64 << 20; // 64 Mi values
const WARMUP: usize = 3;
const ITERS: usize = 10;

fn plan_for(bit_width: u8, payload_len: u64, num_values: u64) -> DecodePlan {
    DecodePlan {
        plan_version: PLAN_VERSION,
        column: FieldId::new(0, "bench"),
        file: "bench.dvd".into(),
        arrow_type: arrow_schema::DataType::Int64,
        blocks: vec![BlockDecode::GcdPacked {
            offset: 0,
            len: payload_len,
            bit_width,
            base: 12_345,
            gcd: 25,
            values: num_values,
        }],
        coverage: Coverage::Dense { num_docs: num_values as u32 },
        num_values,
    }
}

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

    println!(
        "decode kernel, {} Mi values, median of {ITERS} (device-resident input)",
        NUM_VALUES >> 20
    );
    println!();
    println!("  bpv | packed MiB |  GPU GB/s in | GPU Gval/s |  CPU GB/s in | CPU Gval/s | speedup");
    println!("  ----+------------+--------------+------------+--------------+------------+--------");

    for &bpv in &direct::SUPPORTED_BITS_PER_VALUE {
        let mask = if bpv == 64 { u64::MAX } else { (1u64 << bpv) - 1 };
        let values: Vec<i64> = (0..NUM_VALUES)
            .map(|i| ((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) & mask) as i64)
            .collect();
        let mut packed = Vec::new();
        direct::pack(&values, bpv, &mut packed);
        let payload = packed.len() as u64;
        let plan = plan_for(bpv, payload, NUM_VALUES as u64);

        // GPU: upload once, time kernel + sync.
        let dev_data = gpu.upload(&packed).unwrap();
        let mut gpu_secs = Vec::new();
        for i in 0..WARMUP + ITERS {
            gpu.synchronize().unwrap();
            let t = Instant::now();
            let out = gpu.decode_values_device(&plan, &dev_data).unwrap();
            gpu.synchronize().unwrap();
            let dt = t.elapsed().as_secs_f64();
            drop(out);
            if i >= WARMUP {
                gpu_secs.push(dt);
            }
        }
        let gpu_t = median(gpu_secs);

        // CPU reference: fused single-pass unpack + epilogue, one thread.
        let mut cpu_secs = Vec::new();
        let mut decoded: Vec<i64> = Vec::with_capacity(NUM_VALUES);
        for i in 0..2 + 3 {
            decoded.clear();
            let t = Instant::now();
            direct::for_each_unpacked(&packed, bpv, NUM_VALUES, |x| {
                decoded.push(12_345i64.wrapping_add(25i64.wrapping_mul(x as i64)))
            })
            .unwrap();
            let dt = t.elapsed().as_secs_f64();
            std::hint::black_box(&decoded);
            if i >= 2 {
                cpu_secs.push(dt);
            }
        }
        let cpu_t = median(cpu_secs);

        let gib = payload as f64 / 1e9;
        println!(
            "  {:>3} | {:>10.1} | {:>12.1} | {:>10.2} | {:>12.2} | {:>10.3} | {:>6.1}x",
            bpv,
            payload as f64 / (1 << 20) as f64,
            gib / gpu_t,
            NUM_VALUES as f64 / gpu_t / 1e9,
            gib / cpu_t,
            NUM_VALUES as f64 / cpu_t / 1e9,
            cpu_t / gpu_t,
        );
    }
}
