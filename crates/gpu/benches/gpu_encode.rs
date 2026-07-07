// SPDX-License-Identifier: Apache-2.0

//! Encode-kernel throughput (SPEC §11.7, §15 write-path metric): GPU
//! bit-pack of device-resident values vs single-thread CPU `direct::pack`.
//!
//! Run: cargo bench -p lucene-arrow-gpu --features gpu --bench gpu_encode

use std::time::Instant;

use lucene_arrow_docvalues::direct;
use lucene_arrow_gpu::{GpuDecoder, encode::GpuPacker};

const N: usize = 64 << 20;
const ITERS: usize = 10;

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

fn main() {
    let Ok(gpu) = GpuDecoder::new() else {
        eprintln!("no CUDA device");
        return;
    };
    let packer = GpuPacker::new(&gpu).unwrap();

    println!("pack kernel, {} Mi values, median of {ITERS} (device-resident in/out)", N >> 20);
    println!();
    println!("  bpv | packed MiB | GPU GB/s out | GPU Gval/s | CPU GB/s out | speedup");
    println!("  ----+------------+--------------+------------+--------------+--------");
    for &bpv in &direct::SUPPORTED_BITS_PER_VALUE {
        let mask = if bpv == 64 { u64::MAX } else { (1u64 << bpv) - 1 };
        let encoded: Vec<i64> = (0..N)
            .map(|i| ((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) & mask & (i64::MAX as u64)) as i64)
            .collect();
        let payload = direct::packed_len(N, bpv) as f64;

        let dev = packer.upload_values(&encoded).unwrap();
        let mut gpu_secs = Vec::new();
        for i in 0..3 + ITERS {
            packer.synchronize().unwrap();
            let t = Instant::now();
            let out = packer.pack_device(&dev, N as u64, bpv, 0, 1).unwrap();
            packer.synchronize().unwrap();
            let dt = t.elapsed().as_secs_f64();
            drop(out);
            if i >= 3 {
                gpu_secs.push(dt);
            }
        }
        let g = median(gpu_secs);

        let mut cpu_secs = Vec::new();
        let mut out = Vec::with_capacity(payload as usize);
        for i in 0..2 + 3 {
            out.clear();
            let t = Instant::now();
            direct::pack(&encoded, bpv, &mut out);
            let dt = t.elapsed().as_secs_f64();
            std::hint::black_box(&out);
            if i >= 2 {
                cpu_secs.push(dt);
            }
        }
        let c = median(cpu_secs);

        println!(
            "  {:>3} | {:>10.1} | {:>12.1} | {:>10.2} | {:>12.2} | {:>6.1}x",
            bpv,
            payload / (1 << 20) as f64,
            payload / 1e9 / g,
            N as f64 / g / 1e9,
            payload / 1e9 / c,
            c / g,
        );
    }
}
