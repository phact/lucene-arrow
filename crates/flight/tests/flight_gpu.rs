// SPDX-License-Identifier: Apache-2.0

//! GPU-behind-the-engine gate: `executor: "gpu"` DoGet must return byte-
//! identical batches to `"cpu"` (SPEC §12.3 applies at the wire level
//! too), and the echoed config must say what actually ran (§8.3).

#![cfg(feature = "gpu")]

use std::path::PathBuf;

use arrow_array::RecordBatch;
use arrow_flight::client::FlightClient;
use arrow_flight::{FlightDescriptor, Ticket};
use futures::TryStreamExt;
use tonic::transport::Channel;

use lucene_arrow_flight::{LuceneFlightService, ReadRequest, SegmentSelector};

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

#[tokio::test]
async fn gpu_do_get_matches_cpu() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../harness/golden/numerics");
    if !root.is_dir() {
        eprintln!("skipping: goldens not generated");
        return;
    }
    let mut client = start_server().await;

    let mut collect = async |executor: &str| -> (serde_json::Value, Vec<RecordBatch>) {
        let req = ReadRequest {
            path: root.to_string_lossy().into_owned(),
            segments: SegmentSelector::All,
            columns: None,
            row_mode: "positional".to_string(),
            dict: None,
            batch_rows: Some(2048),
            executor: Some(executor.to_string()),
            frame_version: 1,
            relation: None,
            postings_field: None,
        };
        let cmd = serde_json::to_vec(&req).unwrap();
        let info =
            client.get_flight_info(FlightDescriptor::new_cmd(cmd.clone())).await.unwrap();
        let config = serde_json::from_slice(&info.app_metadata).unwrap();
        let batches =
            client.do_get(Ticket::new(cmd)).await.unwrap().try_collect().await.unwrap();
        (config, batches)
    };

    let (cpu_config, cpu_batches) = collect("cpu").await;
    assert_eq!(cpu_config["executor"], "cpu");

    let (gpu_config, gpu_batches) = collect("gpu").await;
    assert_eq!(gpu_config["executor"], "gpu", "echo must reflect what ran");
    assert_eq!(cpu_batches.len(), gpu_batches.len());
    for (c, g) in cpu_batches.iter().zip(&gpu_batches) {
        assert_eq!(c, g, "GPU DoGet batch differs from CPU");
    }
}
