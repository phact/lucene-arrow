// SPDX-License-Identifier: Apache-2.0

//! §10.4 e2e: FixedSizeList<Float32> columns through DoPut become
//! Java-servable vector segments (flat + GPU-built graph), readable back
//! through DoGet and CheckIndex-clean.

#![cfg(feature = "cuvs")]

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{Int64Array, RecordBatch};
use arrow_flight::client::FlightClient;
use arrow_flight::{FlightDescriptor, Ticket};
use futures::TryStreamExt;
use tonic::transport::Channel;

use lucene_arrow_flight::{LuceneFlightService, ReadRequest, SegmentSelector, WriteJob};

const N: usize = 6000; // > 4096 → exercises the GPU path
const DIM: usize = 32;

fn vecf(seed: usize) -> Vec<f32> {
    let cluster = seed % 50;
    (0..DIM)
        .map(|k| {
            let hc =
                ((cluster as u64) ^ ((k as u64) << 32)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let hj = ((seed as u64) ^ ((k as u64) << 32) ^ 0xABCD)
                .wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
            (hc >> 11) as f32 / (1u64 << 53) as f32 * 2.0 - 1.0
                + ((hj >> 11) as f32 / (1u64 << 53) as f32 * 2.0 - 1.0) * 0.05
        })
        .collect()
}

async fn start_server() -> FlightClient {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(
        tonic::transport::Server::builder()
            .add_service(LuceneFlightService::default().into_server())
            .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener)),
    );
    let channel = Channel::from_shared(format!("http://{addr}")).unwrap().connect().await.unwrap();
    FlightClient::new(channel)
}

#[tokio::test(flavor = "multi_thread")]
async fn vectors_through_doput_roundtrip_and_checkindex() {
    if lucene_arrow_gpu::GpuDecoder::new().is_err() {
        eprintln!("no CUDA device");
        return;
    }
    let mut client = start_server().await;

    let schema = Arc::new(arrow_schema::Schema::new(vec![
        arrow_schema::Field::new(
            "emb",
            arrow_schema::DataType::FixedSizeList(
                Arc::new(arrow_schema::Field::new("item", arrow_schema::DataType::Float32, false)),
                DIM as i32,
            ),
            false,
        ),
        arrow_schema::Field::new("price", arrow_schema::DataType::Int64, false),
    ]));
    let flat: Vec<f32> = (0..N).flat_map(vecf).collect();
    let emb = arrow_array::FixedSizeListArray::new(
        Arc::new(arrow_schema::Field::new("item", arrow_schema::DataType::Float32, false)),
        DIM as i32,
        Arc::new(arrow_array::Float32Array::from(flat.clone())),
        None,
    );
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(emb), Arc::new(Int64Array::from_iter_values((0..N as i64).map(|i| i * 3)))],
    )
    .unwrap();

    let tmp = tempfile::tempdir().unwrap();
    let out_dir = tmp.path().join("out");
    let job = WriteJob {
        output_dir: out_dir.to_string_lossy().into_owned(),
        codec: "Lucene103".into(),
        segment_max_docs: None,
        segment_max_bytes: None,
        index_sort: None,
        coercion: Some("strict".into()),
        compound: Some(false),
        executor: Some("cpu".into()),
        frame_version: 1,
    };
    let mut fds = arrow_flight::utils::batches_to_flight_data(&schema, vec![batch]).unwrap();
    fds[0].flight_descriptor =
        Some(FlightDescriptor::new_cmd(serde_json::to_vec(&job).unwrap()));
    let results: Vec<arrow_flight::PutResult> = client
        .do_put(futures::stream::iter(fds.into_iter().map(Ok)))
        .await
        .unwrap()
        .try_collect()
        .await
        .unwrap();
    assert_eq!(results.len(), 1);

    // Read back: vectors decode to the exact input floats.
    let req = ReadRequest {
        path: out_dir.to_string_lossy().into_owned(),
        segments: SegmentSelector::All,
        columns: Some(vec!["emb".into()]),
        row_mode: "positional".into(),
        dict: None,
        batch_rows: None,
        executor: Some("cpu".into()),
        frame_version: 1,
        relation: None,
        postings_field: None,
    };
    let got: Vec<RecordBatch> = client
        .do_get(Ticket::new(serde_json::to_vec(&req).unwrap()))
        .await
        .unwrap()
        .try_collect()
        .await
        .unwrap();
    let mut g = 0usize;
    for b in &got {
        let col = b.column_by_name("emb").unwrap();
        let l = col.as_fixed_size_list();
        let vals = l.values().as_primitive::<arrow_array::types::Float32Type>();
        for r in 0..b.num_rows() {
            for k in 0..DIM {
                assert_eq!(vals.value(r * DIM + k), flat[g * DIM + k], "doc {g} dim {k}");
            }
            g += 1;
        }
    }
    assert_eq!(g, N);

    // CheckIndex validates the graph we built on the GPU.
    let java = std::path::Path::new("/home/tato/.jbang/cache/jdks/21/bin/java");
    let jar = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../harness/lib/lucene-core-10.3.2.jar");
    if !java.exists() || !jar.exists() {
        eprintln!("skipping CheckIndex gate");
        return;
    }
    let output = std::process::Command::new(java)
        .args(["--add-modules", "jdk.incubator.vector", "-cp"])
        .arg(&jar)
        .arg("org.apache.lucene.index.CheckIndex")
        .arg(&out_dir)
        .args(["-level", "2"])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success() && stdout.contains("No problems were detected"),
        "CheckIndex failed:\n{stdout}"
    );
    assert!(stdout.contains("test: vectors"), "vectors not validated:\n{stdout}");
}
