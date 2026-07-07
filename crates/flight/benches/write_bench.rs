// SPDX-License-Identifier: Apache-2.0

//! Kill-criterion (c) head-to-head, SPEC §15.4: our write job vs JVM
//! `BenchIngest` on identical data (4 numeric fields, one sparse,
//! 16M docs — same value formulas).
//!
//! Run: cargo bench -p lucene-arrow-flight --bench write_bench

use std::sync::Arc;
use std::time::Instant;

use arrow_array::{Int64Array, RecordBatch};
use lucene_arrow_flight::write::WriteSession;

const NUM_DOCS: usize = 16_000_000;
const BATCH: usize = 128 * 1024;

fn main() {
    let schema = Arc::new(arrow_schema::Schema::new(vec![
        arrow_schema::Field::new("f0", arrow_schema::DataType::Int64, false),
        arrow_schema::Field::new("f1", arrow_schema::DataType::Int64, false),
        arrow_schema::Field::new("f2", arrow_schema::DataType::Int64, false),
        arrow_schema::Field::new("f3", arrow_schema::DataType::Int64, true),
    ]));

    eprintln!("materializing {} Mi docs of Arrow input...", NUM_DOCS >> 20);
    let batches: Vec<RecordBatch> = (0..NUM_DOCS.div_ceil(BATCH))
        .map(|b| {
            let base = b * BATCH;
            let n = BATCH.min(NUM_DOCS - base);
            RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int64Array::from_iter_values(
                        (0..n).map(|i| 1_000_000 + ((base + i) as i64 % 4096) * 25),
                    )),
                    Arc::new(Int64Array::from_iter_values(
                        (0..n).map(|i| ((base + i) as i64).wrapping_mul(0x9E37) & 0xF_FFFF),
                    )),
                    Arc::new(Int64Array::from_iter_values(
                        (0..n).map(|i| ((base + i) as i64).wrapping_mul(0x9E37_79B9_7F4A_7C15u64 as i64)),
                    )),
                    Arc::new(Int64Array::from_iter(
                        (0..n).map(|i| ((base + i) % 4 != 3).then_some((base + i) as i64 & 0xFFFF)),
                    )),
                ],
            )
            .unwrap()
        })
        .collect();

    let tmp = tempfile::tempdir().unwrap();
    let job = lucene_arrow_flight::WriteJob {
        output_dir: tmp.path().join("out").to_string_lossy().into_owned(),
        codec: "Lucene103".into(),
        segment_max_docs: Some(NUM_DOCS as u64), // one segment, like the JVM run
        segment_max_bytes: None,
        index_sort: None,
        coercion: Some("auto".into()),
        compound: Some(false),
        executor: None,
        frame_version: 1,
    };

    let t = Instant::now();
    let mut session = WriteSession::new(&job, &schema).unwrap();
    for batch in &batches {
        session.push(batch).unwrap();
    }
    let manifests = session.finish().unwrap();
    let secs = t.elapsed().as_secs_f64();

    let total_bytes: u64 = manifests
        .iter()
        .flat_map(|m| m.files.iter())
        .map(|f| f.bytes)
        .sum();
    println!(
        "write job: {} docs × 4 fields in {:.2} s = {:.2} Mdocs/s ({:.2} Mvals/s), {} segment(s), {:.0} MiB",
        NUM_DOCS,
        secs,
        NUM_DOCS as f64 / secs / 1e6,
        NUM_DOCS as f64 * 4.0 / secs / 1e6,
        manifests.len(),
        total_bytes as f64 / (1 << 20) as f64,
    );
    println!("JVM baseline on identical data: BenchIngest (record separately).");
}
