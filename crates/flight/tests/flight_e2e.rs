// SPDX-License-Identifier: Apache-2.0

//! P4 end-to-end: a real Flight server + client over TCP, streaming the
//! Java-written golden indexes as segment-scoped RecordBatches (SPEC §7.1,
//! §8). Config echo (§8.3) and loud failures (§8.2, row-mode contract)
//! are asserted too.

use std::path::PathBuf;

use arrow_array::cast::AsArray;
use arrow_array::types::{Int32Type, Int64Type};
use arrow_array::{Array, RecordBatch, StringArray};
use arrow_flight::client::FlightClient;
use arrow_flight::{FlightDescriptor, Ticket};
use futures::TryStreamExt;
use tonic::transport::Channel;

use lucene_arrow_flight::{LuceneFlightService, ReadRequest, SegmentSelector};

fn golden(name: &str) -> Option<String> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../harness/golden").join(name);
    if p.is_dir() {
        Some(p.to_string_lossy().into_owned())
    } else {
        eprintln!("skipping: harness/golden/{name} not generated (needs JDK 21)");
        None
    }
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

fn request(path: &str) -> ReadRequest {
    ReadRequest {
        path: path.to_string(),
        segments: SegmentSelector::All,
        columns: None,
        row_mode: "positional".to_string(),
        dict: None,
        batch_rows: Some(1024), // force multiple batches per segment
        executor: Some("cpu".to_string()), // deterministic across gpu-feature builds
        frame_version: 1,
        relation: None,
        postings_field: None,
    }
}

#[tokio::test]
async fn do_get_streams_keywords_shard() {
    let Some(path) = golden("keywords") else { return };
    let mut client = start_server().await;

    let req = request(&path);
    let cmd = serde_json::to_vec(&req).unwrap();
    let info = client
        .get_flight_info(FlightDescriptor::new_cmd(cmd.clone()))
        .await
        .unwrap();

    // §8.3: effective config echoed; nothing implicit.
    let config: serde_json::Value = serde_json::from_slice(&info.app_metadata).unwrap();
    assert_eq!(config["row_mode"], "positional");
    assert_eq!(config["batch_rows"], 1024);
    assert_eq!(config["executor"], "cpu");
    assert_eq!(config["frame_version"], 1);
    assert_eq!(config["segments"].as_array().unwrap().len(), 3);
    assert_eq!(config["dict"]["*"], "segment");
    assert_eq!(info.total_records, 3000);

    let batches: Vec<RecordBatch> = client
        .do_get(Ticket::new(cmd))
        .await
        .unwrap()
        .try_collect()
        .await
        .unwrap();

    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 3000);
    // 3 segments × 1000 docs at batch_rows=1024 → one full slice per
    // segment boundary (batches never span segments, SPEC §7.1).
    assert_eq!(batches.len(), 3);

    let mut global = 0i64;
    for (seg_ord, batch) in batches.iter().enumerate() {
        let seg = batch.column_by_name("_seg").unwrap().as_primitive::<Int32Type>();
        let doc = batch.column_by_name("_doc").unwrap().as_primitive::<Int32Type>();
        let gdoc = batch.column_by_name("_global_doc").unwrap().as_primitive::<Int64Type>();
        let cat = batch.column_by_name("cat").unwrap().as_dictionary::<Int32Type>();
        let cat_terms: &StringArray = cat.values().as_string();
        let nums = batch.column_by_name("nums").unwrap().as_list::<i32>();
        let nums_child = nums.values().as_primitive::<Int64Type>();

        for d in 0..batch.num_rows() {
            let g = global as usize;
            assert_eq!(seg.value(d), seg_ord as i32);
            assert_eq!(doc.value(d), d as i32);
            assert_eq!(gdoc.value(d), global);
            if g % 5 != 2 {
                assert_eq!(
                    cat_terms.value(cat.keys().value(d) as usize),
                    format!("cat-{:04}", g * 7 % 501),
                    "cat g{g}"
                );
            } else {
                assert!(cat.is_null(d), "cat g{g}");
            }
            let (s, e) = (nums.value_offsets()[d] as usize, nums.value_offsets()[d + 1] as usize);
            let mut expected: Vec<i64> = (0..1 + g % 3)
                .map(|j| match j {
                    0 => g as i64 * 5,
                    1 => g as i64 * 5 - 100,
                    _ => 7,
                })
                .collect();
            expected.sort();
            let got: Vec<i64> = (s..e).map(|i| nums_child.value(i)).collect();
            assert_eq!(got, expected, "nums g{g}");
            global += 1;
        }
    }
}

#[tokio::test]
async fn do_get_streams_vectors_and_numerics() {
    let Some(path) = golden("numerics") else { return };
    let mut client = start_server().await;

    let mut req = request(&path);
    req.columns = Some(vec!["_doc".into(), "price".into(), "rare".into()]);
    let cmd = serde_json::to_vec(&req).unwrap();

    let batches: Vec<RecordBatch> =
        client.do_get(Ticket::new(cmd)).await.unwrap().try_collect().await.unwrap();
    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total, 5000);
    assert_eq!(batches[0].num_columns(), 3, "column projection respected");

    let mut g = 0usize;
    for batch in &batches {
        let price = batch.column_by_name("price").unwrap().as_primitive::<Int64Type>();
        let rare = batch.column_by_name("rare").unwrap().as_primitive::<Int64Type>();
        for d in 0..batch.num_rows() {
            assert_eq!(price.value(d), 1000 + g as i64 * 25);
            if g % 6 == 1 {
                assert_eq!(rare.value(d), g as i64 * -13);
            } else {
                assert!(rare.is_null(d));
            }
            g += 1;
        }
    }
}

#[tokio::test]
async fn loud_failures() {
    let mut client = start_server().await;

    // row_mode is caller-chosen, no default (SPEC §7.1).
    if let Some(path) = golden("numerics") {
        let mut req = request(&path);
        req.row_mode = "".into();
        let cmd = serde_json::to_vec(&req).unwrap();
        let err = client.get_flight_info(FlightDescriptor::new_cmd(cmd)).await.unwrap_err();
        let arrow_flight::error::FlightError::Tonic(status) = err else {
            panic!("expected tonic status, got {err:?}")
        };
        assert_eq!(status.code(), tonic::Code::InvalidArgument, "{status}");
    }

}

/// Deletes (SPEC §7.1 row modes): positional keeps tombstones with
/// `_live = false`; compact drops them server-side. Docs i % 9 == 4 are
/// deleted in the golden index.
#[tokio::test]
async fn row_modes_respect_tombstones() {
    let Some(path) = golden("deletes") else { return };
    let mut client = start_server().await;
    let num_docs = 2000usize;
    let deleted = |d: usize| d % 9 == 4;
    let live_count = (0..num_docs).filter(|&d| !deleted(d)).count();

    // Positional: every docid present, _live flags tombstones, doc values
    // still readable on deleted rows.
    let mut req = request(&path);
    req.columns = Some(vec!["_doc".into(), "_live".into(), "val".into()]);
    let cmd = serde_json::to_vec(&req).unwrap();
    let batches: Vec<RecordBatch> =
        client.do_get(Ticket::new(cmd)).await.unwrap().try_collect().await.unwrap();
    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total, num_docs);
    let mut d = 0usize;
    for batch in &batches {
        let live = batch.column_by_name("_live").unwrap().as_boolean();
        let val = batch.column_by_name("val").unwrap().as_primitive::<Int64Type>();
        for r in 0..batch.num_rows() {
            assert_eq!(live.value(r), !deleted(d), "_live doc {d}");
            assert_eq!(val.value(r), d as i64 * 3, "val doc {d} (tombstones keep values)");
            d += 1;
        }
    }

    // Compact: tombstones dropped; _doc still names the original docid.
    let mut req = request(&path);
    req.row_mode = "compact".into();
    req.columns = Some(vec!["_doc".into(), "val".into()]);
    let cmd = serde_json::to_vec(&req).unwrap();
    let batches: Vec<RecordBatch> =
        client.do_get(Ticket::new(cmd)).await.unwrap().try_collect().await.unwrap();
    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total, live_count);
    let mut expected_docs = (0..num_docs).filter(|&d| !deleted(d));
    for batch in &batches {
        let doc = batch.column_by_name("_doc").unwrap().as_primitive::<Int32Type>();
        let val = batch.column_by_name("val").unwrap().as_primitive::<Int64Type>();
        for r in 0..batch.num_rows() {
            let d = expected_docs.next().unwrap();
            assert_eq!(doc.value(r), d as i32);
            assert_eq!(val.value(r), d as i64 * 3);
        }
    }
}

/// Hydration (SPEC §7.4): stored columns for explicit (_seg,_doc) pairs
/// via DoAction, IPC-decoded, pair order preserved.
#[tokio::test]
async fn hydrate_action_returns_stored_columns() {
    use bearing::prelude::{
        DocumentBuilder, FSDirectory, IndexWriter, IndexWriterConfig, numeric_dv, stored,
    };

    let tmp = tempfile::tempdir().unwrap();
    {
        let directory = FSDirectory::open(tmp.path()).unwrap();
        let config = IndexWriterConfig::default().num_threads(1).use_compound_file(false);
        let writer = IndexWriter::new(config, directory);
        for i in 0..500i64 {
            let doc = DocumentBuilder::new()
                .add_field(numeric_dv("id").value(i))
                .add_field(stored("title").string(format!("doc number {i}")))
                .add_field(stored("weight").long(i * 7))
                .build();
            writer.add_document(doc).unwrap();
        }
        writer.commit().unwrap();
    }

    let mut client = start_server().await;
    let body = serde_json::to_vec(&lucene_arrow_flight::HydrateRequest {
        path: tmp.path().to_string_lossy().into_owned(),
        pairs: vec![(0, 411), (0, 3), (0, 77)],
        columns: vec!["title".into(), "weight".into()],
    })
    .unwrap();
    let results = client
        .do_action(arrow_flight::Action { r#type: "hydrate".into(), body: body.into() })
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    assert_eq!(results.len(), 1);

    let mut reader =
        arrow_ipc::reader::StreamReader::try_new(std::io::Cursor::new(results[0].to_vec()), None)
            .unwrap();
    let batch = reader.next().unwrap().unwrap();
    assert_eq!(batch.num_rows(), 3);
    let title = batch.column_by_name("title").unwrap().as_string::<i32>();
    let weight = batch.column_by_name("weight").unwrap().as_primitive::<Int64Type>();
    let doc = batch.column_by_name("_doc").unwrap().as_primitive::<Int32Type>();
    for (r, expected_doc) in [411i64, 3, 77].iter().enumerate() {
        assert_eq!(doc.value(r) as i64, *expected_doc);
        assert_eq!(title.value(r), format!("doc number {expected_doc}"));
        assert_eq!(weight.value(r), expected_doc * 7);
    }
}

/// The write job end-to-end (SPEC §10): DoPut a stream of Arrow batches
/// (canonical + lossless-coerced types), get SegmentManifests back, then
/// prove the output: segments cut on thresholds, values readable back via
/// DoGet, and Java CheckIndex passes.
#[tokio::test(flavor = "multi_thread")]
async fn do_put_writes_checkindex_clean_segments() {
    use arrow_array::{Float64Array, Int32Array as I32, Int64Array, TimestampMillisecondArray};
    use std::sync::Arc;

    let out = tempfile::tempdir().unwrap();
    let out_path = out.path().to_string_lossy().into_owned();
    let mut client = start_server().await;

    let schema = Arc::new(arrow_schema::Schema::new(vec![
        arrow_schema::Field::new("id", arrow_schema::DataType::Int64, false),
        arrow_schema::Field::new("small", arrow_schema::DataType::Int32, true),
        arrow_schema::Field::new("price", arrow_schema::DataType::Float64, false),
        arrow_schema::Field::new(
            "ts",
            arrow_schema::DataType::Timestamp(arrow_schema::TimeUnit::Millisecond, None),
            false,
        ),
        arrow_schema::Field::new("tag", arrow_schema::DataType::Utf8, true),
    ]));
    let rows_per = 1000usize;
    let batches: Vec<RecordBatch> = (0..6usize)
        .map(|b| {
            let base = (b * rows_per) as i64;
            RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int64Array::from_iter_values((0..rows_per as i64).map(|i| base + i))),
                    Arc::new(I32::from_iter((0..rows_per as i64).map(|i| {
                        ((base + i) % 3 != 1).then_some((base + i) as i32 * 2)
                    }))),
                    Arc::new(Float64Array::from_iter_values(
                        (0..rows_per as i64).map(|i| (base + i) as f64 * 0.5),
                    )),
                    Arc::new(TimestampMillisecondArray::from_iter_values(
                        (0..rows_per as i64).map(|i| 1_700_000_000_000 + (base + i) * 1000),
                    )),
                    Arc::new(arrow_array::StringArray::from_iter((0..rows_per as i64).map(
                        |i| ((base + i) % 4 != 2).then(|| format!("tag-{:03}", (base + i) % 97)),
                    ))),
                ],
            )
            .unwrap()
        })
        .collect();

    let job = lucene_arrow_flight::WriteJob {
        output_dir: out_path.clone(),
        codec: "Lucene103".into(),
        segment_max_docs: Some(2500), // cut between batches: 3000 + 3000
        segment_max_bytes: None,
        index_sort: None,
        coercion: Some("auto".into()),
        compound: Some(false),
        executor: None,
        frame_version: 1,
    };
    let descriptor = FlightDescriptor::new_cmd(serde_json::to_vec(&job).unwrap());
    let mut fds = arrow_flight::utils::batches_to_flight_data(&schema, batches).unwrap();
    fds[0].flight_descriptor = Some(descriptor);

    let results: Vec<arrow_flight::PutResult> = client
        .do_put(futures::stream::iter(fds.into_iter().map(Ok)))
        .await
        .unwrap()
        .try_collect()
        .await
        .unwrap();
    let manifests: Vec<serde_json::Value> = results
        .iter()
        .map(|r| serde_json::from_slice(&r.app_metadata).unwrap())
        .collect();
    assert_eq!(manifests.len(), 2, "6k rows at max 2500/segment, batch-aligned → 2 segments");
    assert_eq!(manifests[0]["name"], "_0");
    assert_eq!(manifests[1]["name"], "_1");
    assert_eq!(manifests[0]["max_doc"], 3000);
    assert_eq!(manifests[1]["max_doc"], 3000);
    assert!(!manifests[0]["files"].as_array().unwrap().is_empty());

    // Read the job's output back through DoGet (values are the canonical
    // numeric payloads: temporals/floats ride as their i64 payload until
    // lucene.source_type round-trip typing lands).
    let mut req = request(&out_path);
    req.columns =
        Some(vec!["_global_doc".into(), "id".into(), "small".into(), "price".into(), "tag".into()]);
    let cmd = serde_json::to_vec(&req).unwrap();
    let got: Vec<RecordBatch> =
        client.do_get(Ticket::new(cmd)).await.unwrap().try_collect().await.unwrap();
    let total: usize = got.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total, 6000);
    let mut g = 0i64;
    for batch in &got {
        let id = batch.column_by_name("id").unwrap().as_primitive::<Int64Type>();
        let small = batch.column_by_name("small").unwrap().as_primitive::<Int64Type>();
        let price = batch.column_by_name("price").unwrap().as_primitive::<Int64Type>();
        let tag = batch.column_by_name("tag").unwrap().as_dictionary::<Int32Type>();
        let tag_terms: &arrow_array::StringArray = tag.values().as_string();
        for r in 0..batch.num_rows() {
            assert_eq!(id.value(r), g, "id");
            if g % 3 != 1 {
                assert_eq!(small.value(r), g * 2, "small g{g}");
            } else {
                assert!(small.is_null(r), "small g{g}");
            }
            assert_eq!(price.value(r) as u64, (g as f64 * 0.5).to_bits(), "price g{g}");
            if g % 4 != 2 {
                assert!(tag.is_valid(r), "tag g{g}");
                assert_eq!(
                    tag_terms.value(tag.keys().value(r) as usize),
                    format!("tag-{:03}", g % 97),
                    "tag g{g}"
                );
            } else {
                assert!(tag.is_null(r), "tag g{g}");
            }
            g += 1;
        }
    }

    // CheckIndex acceptance gate (SPEC §10.5).
    let java = std::path::Path::new("/home/tato/.jbang/cache/jdks/21/bin/java");
    let jar = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../harness/lib/lucene-core-10.3.2.jar");
    if java.exists() && jar.exists() {
        let output = std::process::Command::new(java)
            .args(["--add-modules", "jdk.incubator.vector", "-cp"])
            .arg(&jar)
            .arg("org.apache.lucene.index.CheckIndex")
            .arg(out.path())
            .args(["-level", "2"])
            .output()
            .expect("run CheckIndex");
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            output.status.success() && stdout.contains("No problems were detected"),
            "CheckIndex failed:\n{stdout}"
        );
    } else {
        eprintln!("skipping CheckIndex gate: JDK/jar missing");
    }
}

/// §10.2 matrix extension: List<Int64>, List<Utf8>, Binary through DoPut,
/// plus strict-mode rejection and the stats action.
#[tokio::test(flavor = "multi_thread")]
async fn lists_binary_strict_and_stats() {
    use arrow_array::builder::{BinaryBuilder, ListBuilder, StringBuilder, Int64Builder};
    use std::sync::Arc;

    let out = tempfile::tempdir().unwrap();
    let out_path = out.path().to_string_lossy().into_owned();
    let mut client = start_server().await;

    let schema = Arc::new(arrow_schema::Schema::new(vec![
        arrow_schema::Field::new("id", arrow_schema::DataType::Int64, false),
        arrow_schema::Field::new(
            "scores",
            arrow_schema::DataType::List(Arc::new(arrow_schema::Field::new(
                "item",
                arrow_schema::DataType::Int64,
                true,
            ))),
            true,
        ),
        arrow_schema::Field::new(
            "tags",
            arrow_schema::DataType::List(Arc::new(arrow_schema::Field::new(
                "item",
                arrow_schema::DataType::Utf8,
                true,
            ))),
            true,
        ),
        arrow_schema::Field::new("blob", arrow_schema::DataType::Binary, true),
    ]));

    let num_docs = 2000usize;
    let mut ids = Int64Builder::new();
    let mut scores = ListBuilder::new(Int64Builder::new());
    let mut tags = ListBuilder::new(StringBuilder::new());
    let mut blob = BinaryBuilder::new();
    for d in 0..num_docs as i64 {
        ids.append_value(d);
        for j in 0..1 + d % 3 {
            scores.values().append_value(d * 5 - j * 100);
        }
        scores.append(true);
        if d % 4 == 0 {
            tags.append_null();
        } else {
            for j in 0..d % 3 {
                tags.values().append_value(format!("tag-{:03}", (d + j * 37) % 113));
            }
            tags.values().append_value(format!("base-{:02}", d % 31));
            tags.append(true);
        }
        if d % 3 != 1 {
            blob.append_value(vec![d as u8; (d % 5) as usize + 1]);
        } else {
            blob.append_null();
        }
    }
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(ids.finish()),
            Arc::new(scores.finish()),
            Arc::new(tags.finish()),
            Arc::new(blob.finish()),
        ],
    )
    .unwrap();

    let job = lucene_arrow_flight::WriteJob {
        output_dir: out_path.clone(),
        codec: "Lucene103".into(),
        segment_max_docs: None,
        segment_max_bytes: None,
        index_sort: None,
        coercion: Some("auto".into()),
        compound: Some(false),
        executor: Some("cpu".into()),
        frame_version: 1,
    };
    let mut fds =
        arrow_flight::utils::batches_to_flight_data(&schema, vec![batch.clone()]).unwrap();
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

    // Read back and spot-check every shape.
    let mut req = request(&out_path);
    req.columns =
        Some(vec!["id".into(), "scores".into(), "tags".into(), "blob".into()]);
    let cmd = serde_json::to_vec(&req).unwrap();
    let got: Vec<RecordBatch> =
        client.do_get(Ticket::new(cmd)).await.unwrap().try_collect().await.unwrap();
    let total: usize = got.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total, num_docs);
    let mut g = 0i64;
    for b in &got {
        let scores = b.column_by_name("scores").unwrap().as_list::<i32>();
        let sc = scores.values().as_primitive::<Int64Type>();
        let tags = b.column_by_name("tags").unwrap().as_list::<i32>();
        let tag_dict = tags.values().as_dictionary::<Int32Type>();
        let tag_terms: &arrow_array::StringArray = tag_dict.values().as_string();
        let blob = b.column_by_name("blob").unwrap().as_binary::<i32>();
        for r in 0..b.num_rows() {
            // scores: per-doc sorted ascending
            let (s, e) =
                (scores.value_offsets()[r] as usize, scores.value_offsets()[r + 1] as usize);
            let mut expected: Vec<i64> = (0..1 + g % 3).map(|j| g * 5 - j * 100).collect();
            expected.sort();
            assert_eq!((s..e).map(|i| sc.value(i)).collect::<Vec<_>>(), expected, "scores g{g}");
            // tags: absent every 4th doc; sorted unique terms otherwise
            if g % 4 == 0 {
                assert!(tags.is_null(r), "tags g{g}");
            } else {
                let mut exp: Vec<String> =
                    (0..g % 3).map(|j| format!("tag-{:03}", (g + j * 37) % 113)).collect();
                exp.push(format!("base-{:02}", g % 31));
                exp.sort();
                exp.dedup();
                let (s, e) =
                    (tags.value_offsets()[r] as usize, tags.value_offsets()[r + 1] as usize);
                let got_terms: Vec<String> = (s..e)
                    .map(|i| tag_terms.value(tag_dict.keys().value(i) as usize).to_string())
                    .collect();
                assert_eq!(got_terms, exp, "tags g{g}");
            }
            // blob
            if g % 3 != 1 {
                assert_eq!(blob.value(r), vec![g as u8; (g % 5) as usize + 1], "blob g{g}");
            } else {
                assert!(blob.is_null(r), "blob g{g}");
            }
            g += 1;
        }
    }

    // CheckIndex gate.
    let java = std::path::Path::new("/home/tato/.jbang/cache/jdks/21/bin/java");
    let jar = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../harness/lib/lucene-core-10.3.2.jar");
    if java.exists() && jar.exists() {
        let output = std::process::Command::new(java)
            .args(["--add-modules", "jdk.incubator.vector", "-cp"])
            .arg(&jar)
            .arg("org.apache.lucene.index.CheckIndex")
            .arg(out.path())
            .args(["-level", "2"])
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            output.status.success() && stdout.contains("No problems were detected"),
            "CheckIndex failed:\n{stdout}"
        );
    }

    // strict mode: Utf8 list is a coercion → rejected at schema time.
    let strict_dir = tempfile::tempdir().unwrap();
    let job = lucene_arrow_flight::WriteJob {
        output_dir: strict_dir.path().join("x").to_string_lossy().into_owned(),
        coercion: Some("strict".into()),
        ..job
    };
    let mut fds = arrow_flight::utils::batches_to_flight_data(&schema, vec![batch]).unwrap();
    fds[0].flight_descriptor =
        Some(FlightDescriptor::new_cmd(serde_json::to_vec(&job).unwrap()));
    let strict_err = match client.do_put(futures::stream::iter(fds.into_iter().map(Ok))).await {
        Err(e) => format!("{e:?}"),
        Ok(stream) => format!("{:?}", stream.try_collect::<Vec<_>>().await.unwrap_err()),
    };
    assert!(strict_err.contains("strict"), "{strict_err}");

    // stats action over the written shard.
    let body = serde_json::json!({ "path": out_path }).to_string();
    let results = client
        .do_action(arrow_flight::Action { r#type: "stats".into(), body: body.into() })
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    let stats: serde_json::Value = serde_json::from_slice(&results[0]).unwrap();
    let fields = stats["segments"][0]["fields"].as_array().unwrap();
    assert_eq!(fields.len(), 4);
    assert!(fields.iter().any(|f| f["shape"] == "sorted_set_multi"));
    assert!(fields.iter().any(|f| f["shape"] == "sorted_numeric_multi"));
    assert!(fields.iter().any(|f| f["shape"] == "binary"));
}

/// Explicit-lossy coercions (§10.2): UInt64 and Decimal128 are rejected
/// without their opt-in metadata, accepted (and correct) with it.
#[tokio::test(flavor = "multi_thread")]
async fn lossy_coercions_are_explicit_only() {
    use std::collections::HashMap;
    use std::sync::Arc;

    let mut client = start_server().await;
    let job_for = |dir: &std::path::Path| lucene_arrow_flight::WriteJob {
        output_dir: dir.join("out").to_string_lossy().into_owned(),
        codec: "Lucene103".into(),
        segment_max_docs: None,
        segment_max_bytes: None,
        index_sort: None,
        coercion: Some("auto".into()),
        compound: Some(false),
        executor: Some("cpu".into()),
        frame_version: 1,
    };

    // Without metadata → schema-time rejection.
    let bare = Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new(
        "big",
        arrow_schema::DataType::UInt64,
        false,
    )]));
    let batch = RecordBatch::try_new(
        bare.clone(),
        vec![Arc::new(arrow_array::UInt64Array::from_iter_values(0..100))],
    )
    .unwrap();
    let tmp = tempfile::tempdir().unwrap();
    let mut fds = arrow_flight::utils::batches_to_flight_data(&bare, vec![batch]).unwrap();
    fds[0].flight_descriptor = Some(FlightDescriptor::new_cmd(
        serde_json::to_vec(&job_for(tmp.path())).unwrap(),
    ));
    let err = match client.do_put(futures::stream::iter(fds.into_iter().map(Ok))).await {
        Err(e) => format!("{e:?}"),
        Ok(s) => format!("{:?}", s.try_collect::<Vec<_>>().await.unwrap_err()),
    };
    assert!(err.contains("allow_lossy"), "{err}");

    // With metadata → accepted; Decimal128 scales per ES convention.
    let schema = Arc::new(arrow_schema::Schema::new(vec![
        arrow_schema::Field::new("big", arrow_schema::DataType::UInt64, false).with_metadata(
            HashMap::from([("lucene.allow_lossy".to_string(), "true".to_string())]),
        ),
        arrow_schema::Field::new("price", arrow_schema::DataType::Decimal128(10, 2), false)
            .with_metadata(HashMap::from([(
                "lucene.scale_factor".to_string(),
                "100".to_string(),
            )])),
    ]));
    let num = 500usize;
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(arrow_array::UInt64Array::from_iter_values(
                (0..num as u64).map(|i| i.wrapping_mul(u64::MAX / 97)),
            )),
            Arc::new(
                arrow_array::Decimal128Array::from_iter_values(
                    // raw cents (scale 2): value = i + 0.25 → raw = 100i + 25
                    (0..num as i128).map(|i| i * 100 + 25),
                )
                .with_precision_and_scale(10, 2)
                .unwrap(),
            ),
        ],
    )
    .unwrap();
    let tmp = tempfile::tempdir().unwrap();
    let out_path = tmp.path().join("out").to_string_lossy().into_owned();
    let mut fds = arrow_flight::utils::batches_to_flight_data(&schema, vec![batch]).unwrap();
    fds[0].flight_descriptor = Some(FlightDescriptor::new_cmd(
        serde_json::to_vec(&job_for(tmp.path())).unwrap(),
    ));
    let results: Vec<arrow_flight::PutResult> = client
        .do_put(futures::stream::iter(fds.into_iter().map(Ok)))
        .await
        .unwrap()
        .try_collect()
        .await
        .unwrap();
    assert_eq!(results.len(), 1);

    let mut req = request(&out_path);
    req.columns = Some(vec!["big".into(), "price".into()]);
    let cmd = serde_json::to_vec(&req).unwrap();
    let got: Vec<RecordBatch> =
        client.do_get(Ticket::new(cmd)).await.unwrap().try_collect().await.unwrap();
    let mut g = 0u64;
    for b in &got {
        let big = b.column_by_name("big").unwrap().as_primitive::<Int64Type>();
        let price = b.column_by_name("price").unwrap().as_primitive::<Int64Type>();
        for r in 0..b.num_rows() {
            assert_eq!(big.value(r), g.wrapping_mul(u64::MAX / 97) as i64, "big g{g}");
            // (i + 0.25) · 100 = 100i + 25, already integral cents.
            assert_eq!(price.value(r), g as i64 * 100 + 25, "price g{g}");
            g += 1;
        }
    }
    assert_eq!(g as usize, num);
}

/// SPEC §7.8: postings relation over Flight — term|doc|freq COO batches
/// from the Java-written text golden.
#[tokio::test(flavor = "multi_thread")]
async fn postings_relation_over_flight() {
    use arrow_array::cast::AsArray;
    use arrow_array::types::{Int32Type, UInt32Type};

    let golden = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../harness/golden/text");
    if !golden.exists() {
        eprintln!("text golden absent");
        return;
    }
    let mut client = start_server().await;
    let mut req = request(&golden.to_string_lossy());
    req.relation = Some("postings".into());
    req.postings_field = Some("body".into());
    req.batch_rows = Some(4096);
    let cmd = serde_json::to_vec(&req).unwrap();
    let batches: Vec<RecordBatch> =
        client.do_get(Ticket::new(cmd)).await.unwrap().try_collect().await.unwrap();

    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total, 9000); // 3000 (common) + 3000 (modK) + 3000 (uN)

    // First rows are term ord 0 = "common": doc i, freq i%3+1.
    let b = &batches[0];
    let term = b.column_by_name("term").unwrap().as_dictionary::<Int32Type>();
    let terms = term.values().as_binary::<i32>();
    assert_eq!(terms.value(0), b"common");
    let docs = b.column_by_name("doc").unwrap().as_primitive::<UInt32Type>();
    let freqs = b.column_by_name("freq").unwrap().as_primitive::<UInt32Type>();
    for i in 0..3000.min(b.num_rows()) {
        assert_eq!(term.key(i).unwrap(), 0);
        assert_eq!(docs.value(i), i as u32);
        assert_eq!(freqs.value(i), i as u32 % 3 + 1);
    }
}
