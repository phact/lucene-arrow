// SPDX-License-Identifier: Apache-2.0

//! P10 e2e: GPU ingest vs CPU build_parallel vs full segment write, on
//! the arXiv corpus (and any corpus passed as arg).
//! Run: cargo bench -p lucene-arrow-gpu --features gpu --bench gpu_ingest -- <corpus>

use std::time::Instant;

use lucene_arrow_gpu::GpuDecoder;
use lucene_arrow_gpu::text_ingest::GpuTextIngest;
use lucene_arrow_postings::build::build_parallel;

fn main() {
    let corpus = std::env::args().skip(1).find(|a| !a.starts_with('-')).expect("corpus");
    let Ok(gpu) = GpuDecoder::new() else { return eprintln!("no CUDA") };
    let text = std::fs::read_to_string(&corpus).unwrap();
    let mb = text.len() as f64 / 1e6;
    let lines: Vec<&str> = text.lines().collect();
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);

    // CPU reference timing.
    let t = Instant::now();
    let cpu = build_parallel(&lines, threads);
    let t_cpu = t.elapsed();

    let ingest = GpuTextIngest::new(&gpu).unwrap();
    // Warm-up + correctness spot check.
    let (g, _stats) = ingest.build(&gpu, &text, threads).unwrap();
    assert_eq!(g.docs, cpu.docs);
    assert_eq!(g.sum_total_term_freq, cpu.sum_total_term_freq);

    for round in 0..3 {
        let t = Instant::now();
        let (_g, st) = ingest.build(&gpu, &text, threads).unwrap();
        let secs = t.elapsed().as_secs_f64();
        println!(
            "gpu round {round}: {:.0} ms e2e = {:.0} MB/s (kernel {:.1} dl {:.1} vocab {:.1} dirty {:.1} remap {:.1} csr {:.1}) | cpu {:.0} ms",
            secs * 1e3,
            mb / secs,
            st.kernel_ms,
            st.download_ms,
            st.vocab_ms,
            st.dirty_ms,
            st.remap_ms,
            st.csr_ms,
            t_cpu.as_secs_f64() * 1e3,
        );
    }
}
