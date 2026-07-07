// SPDX-License-Identifier: Apache-2.0

//! GPU postings doc-block decode vs CPU (and the JVM 360 Mpostings/s
//! baseline). Differential: every GPU-decoded doc id must equal the CPU
//! scan. Run with --features gpu against harness/bench-text.

use std::time::Instant;

use lucene_arrow_gpu::GpuDecoder;
use lucene_arrow_gpu::postings_gpu::{GpuPostings, plan_doc_blocks};
use lucene_arrow_postings::doc::scan_postings;
use lucene_arrow_postings::walk::{FieldTraits, walk_terms};
use lucene_arrow_postings::{parse_tmd, root_block};

fn main() {
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../harness/bench-text");
    if !dir.exists() {
        return eprintln!("bench index absent: java BenchText ingest harness/bench-text 4000000");
    }
    let Ok(gpu) = GpuDecoder::new() else { return eprintln!("no CUDA device") };
    let tmd = std::fs::read(dir.join("_0_Lucene103_0.tmd")).unwrap();
    let tim = std::fs::read(dir.join("_0_Lucene103_0.tim")).unwrap();
    let tip = std::fs::read(dir.join("_0_Lucene103_0.tip")).unwrap();
    let doc = std::fs::read(dir.join("_0_Lucene103_0.doc")).unwrap();

    let traits = FieldTraits { has_freqs: true, has_positions: true, has_offsets: false };
    let m = &parse_tmd(&tmd, |_| true).unwrap()[0];
    let tip_slice = &tip[m.index_start_fp as usize..m.index_end_fp as usize];
    let root = root_block(tip_slice, m.trie_root_fp).unwrap();

    let t = Instant::now();
    let descs = plan_doc_blocks(&tim, &doc, root.fp, traits).unwrap();
    let t_plan = t.elapsed();
    let packed_docs = descs.len() * 128;
    eprintln!("planned {} blocks ({packed_docs} docs) in {t_plan:?}", descs.len());

    let engine = GpuPostings::new(&gpu).unwrap();
    let doc_dev = gpu.upload(&doc).unwrap();
    // Warm-up + differential
    let out = engine.decode_blocks(&gpu, &doc_dev, &descs).unwrap();
    let gpu_docs: Vec<u32> = engine.download(&gpu, &out).unwrap();

    // CPU reference for the packed spans.
    let mut cpu_docs = Vec::with_capacity(packed_docs);
    walk_terms(&tim, root.fp, traits, |_t, df, ttf, tm| {
        if df < 128 {
            return Ok(());
        }
        let full = (df as usize / 128) * 128;
        let mut i = 0usize;
        scan_postings(&doc, df, ttf, tm, traits, |d, _f| {
            if i < full {
                cpu_docs.push(d);
            }
            i += 1;
            Ok(())
        })
    })
    .unwrap();
    assert_eq!(gpu_docs.len(), cpu_docs.len());
    assert_eq!(gpu_docs, cpu_docs, "GPU/CPU doc mismatch");
    eprintln!("differential: {} packed docs bit-identical", cpu_docs.len());

    // Kernel-only timing (data resident).
    for _ in 0..3 {
        let t = Instant::now();
        let _ = engine.decode_blocks(&gpu, &doc_dev, &descs).unwrap();
        let secs = t.elapsed().as_secs_f64();
        println!(
            "gpu decode: {} blocks / {packed_docs} docs in {:.2} ms = {:.1} Gdocs/s",
            descs.len(),
            secs * 1e3,
            packed_docs as f64 / secs / 1e9
        );
    }
}
