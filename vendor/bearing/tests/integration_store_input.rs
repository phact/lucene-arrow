// SPDX-License-Identifier: Apache-2.0

//! Integration tests for store-level read functionality through the public API.
//!
//! These tests exercise `Directory` (read_file, file_length, list_all),
//! `CompoundDirectory`, `segment_infos::read`, and `StoredFieldsReader` —
//! the publicly-visible read surface. They do not reach into the crate-private
//! `store2` byte-cursor types; primitive read coverage lives in the
//! per-encoding and per-module unit tests inside `src/`.

#[macro_use]
extern crate assertables;

use std::sync::Arc;

use bearing::store::{CompoundDirectory, Directory, MemoryDirectory, SharedDirectory};

/// First 4 bytes of every codec file are the BE-encoded codec magic.
const CODEC_MAGIC_BYTES: [u8; 4] = [0x3F, 0xD7, 0x6C, 0x17];

#[test]
fn test_index_writer_files_readable() {
    use bearing::document::DocumentBuilder;
    use bearing::index::config::IndexWriterConfig;
    use bearing::index::field::{keyword, text};
    use bearing::index::writer::IndexWriter;

    let config = IndexWriterConfig::default().num_threads(1);
    let directory: SharedDirectory = MemoryDirectory::create();
    let writer = IndexWriter::new(config, Arc::clone(&directory));

    let doc = DocumentBuilder::new()
        .add_field(text("body").value("the quick brown fox"))
        .add_field(keyword("category").value("animals"))
        .build();
    writer.add_document(doc).unwrap();
    writer.commit().unwrap();

    let dir = &*directory;
    let files = dir.list_all().unwrap();
    assert_not_empty!(files);

    for file in &files {
        assert_gt!(
            dir.file_length(file).unwrap(),
            0,
            "file {file} has zero length"
        );
    }
}

#[test]
fn test_index_writer_codec_files_have_valid_headers() {
    use bearing::document::DocumentBuilder;
    use bearing::index::config::IndexWriterConfig;
    use bearing::index::field::{keyword, text};
    use bearing::index::writer::IndexWriter;

    let config = IndexWriterConfig::default().num_threads(1);
    let directory: SharedDirectory = MemoryDirectory::create();
    let writer = IndexWriter::new(config, Arc::clone(&directory));

    let doc = DocumentBuilder::new()
        .add_field(text("body").value("the quick brown fox"))
        .add_field(keyword("category").value("animals"))
        .build();
    writer.add_document(doc).unwrap();
    writer.commit().unwrap();

    let dir = &*directory;
    let files = dir.list_all().unwrap();

    for file in &files {
        if file.starts_with("segments") {
            continue; // segments_N has its own format
        }

        let bytes = dir.read_file(file).unwrap();
        assert_ge!(bytes.len(), 4, "file {file} too short for codec magic");
        assert_eq!(
            &bytes[..4],
            &CODEC_MAGIC_BYTES,
            "file {file} does not start with CODEC_MAGIC"
        );
    }
}

#[test]
fn test_read_segments_from_index_writer() {
    use bearing::document::DocumentBuilder;
    use bearing::index::config::IndexWriterConfig;
    use bearing::index::field::{keyword, text};
    use bearing::index::segment_infos;
    use bearing::index::writer::IndexWriter;

    let config = IndexWriterConfig::default().num_threads(1);
    let directory: SharedDirectory = MemoryDirectory::create();
    let writer = IndexWriter::new(config, Arc::clone(&directory));

    let doc = DocumentBuilder::new()
        .add_field(text("body").value("the quick brown fox"))
        .add_field(keyword("category").value("animals"))
        .build();
    writer.add_document(doc).unwrap();
    writer.commit().unwrap();

    let dir = &*directory;
    let files = dir.list_all().unwrap();
    let segments_file = files.iter().find(|f| f.starts_with("segments_")).unwrap();

    let infos = segment_infos::read(dir, segments_file).unwrap();

    assert_eq!(infos.segments.len(), 1);
    assert_eq!(infos.segments[0].name, "_0");
    assert_eq!(infos.segments[0].codec, "Lucene103");
    assert_eq!(infos.counter, 1);
}

#[test]
fn test_read_segments_multiple_docs() {
    use bearing::document::DocumentBuilder;
    use bearing::index::config::IndexWriterConfig;
    use bearing::index::field::{keyword, long_field, text};
    use bearing::index::segment_infos;
    use bearing::index::writer::IndexWriter;

    let config = IndexWriterConfig::default().num_threads(1);
    let directory: SharedDirectory = MemoryDirectory::create();
    let writer = IndexWriter::new(config, Arc::clone(&directory));

    for i in 0..5 {
        let doc = DocumentBuilder::new()
            .add_field(text("body").value(format!("document number {i}")))
            .add_field(keyword("id").value(format!("doc_{i}")))
            .add_field(long_field("modified").value(1000 + i as i64))
            .build();
        writer.add_document(doc).unwrap();
    }

    writer.commit().unwrap();

    let dir = &*directory;
    let files = dir.list_all().unwrap();
    let segments_file = files.iter().find(|f| f.starts_with("segments_")).unwrap();
    let infos = segment_infos::read(dir, segments_file).unwrap();

    assert_eq!(infos.segments.len(), 1);
    assert_eq!(infos.segments[0].name, "_0");
    assert_eq!(infos.segments[0].codec, "Lucene103");
}

#[test]
fn test_read_segments_memory_directory() {
    use bearing::document::DocumentBuilder;
    use bearing::index::config::IndexWriterConfig;
    use bearing::index::field::{keyword, text};
    use bearing::index::segment_infos;
    use bearing::index::writer::IndexWriter;

    let config = IndexWriterConfig::default().num_threads(1);
    let directory: SharedDirectory = MemoryDirectory::create();
    let writer = IndexWriter::new(config, Arc::clone(&directory));

    let doc = DocumentBuilder::new()
        .add_field(text("body").value("hello world"))
        .add_field(keyword("tag").value("test"))
        .build();
    writer.add_document(doc).unwrap();
    writer.commit().unwrap();

    let dir = &*directory;
    let files = dir.list_all().unwrap();
    let segments_file = files.iter().find(|f| f.starts_with("segments_")).unwrap();
    let infos = segment_infos::read(dir, segments_file).unwrap();

    assert_eq!(infos.segments.len(), 1);
    assert_eq!(infos.segments[0].name, "_0");
}

// --- Compound file tests ---

#[test]
fn test_read_segments_compound_mode() {
    use bearing::document::DocumentBuilder;
    use bearing::index::config::IndexWriterConfig;
    use bearing::index::field::{keyword, text};
    use bearing::index::segment_infos;
    use bearing::index::writer::IndexWriter;

    let config = IndexWriterConfig::default()
        .num_threads(1)
        .use_compound_file(true);
    let directory: SharedDirectory = MemoryDirectory::create();
    let writer = IndexWriter::new(config, Arc::clone(&directory));

    let doc = DocumentBuilder::new()
        .add_field(text("body").value("compound file test"))
        .add_field(keyword("tag").value("test"))
        .build();
    writer.add_document(doc).unwrap();
    writer.commit().unwrap();

    let dir = &*directory;
    let files = dir.list_all().unwrap();
    let segments_file = files.iter().find(|f| f.starts_with("segments_")).unwrap();
    let infos = segment_infos::read(dir, segments_file).unwrap();

    assert_eq!(infos.segments.len(), 1);
    assert_eq!(infos.segments[0].name, "_0");

    assert_any!(files.iter(), |f: &String| f.ends_with(".cfs"));
    assert_any!(files.iter(), |f: &String| f.ends_with(".cfe"));
}

#[test]
fn test_compound_directory_list_files() {
    use bearing::document::DocumentBuilder;
    use bearing::index::config::IndexWriterConfig;
    use bearing::index::field::{keyword, text};
    use bearing::index::segment_infos;
    use bearing::index::writer::IndexWriter;

    let config = IndexWriterConfig::default()
        .num_threads(1)
        .use_compound_file(true);
    let directory: SharedDirectory = MemoryDirectory::create();
    let writer = IndexWriter::new(config, Arc::clone(&directory));

    let doc = DocumentBuilder::new()
        .add_field(text("body").value("compound listing test"))
        .add_field(keyword("tag").value("test"))
        .build();
    writer.add_document(doc).unwrap();
    writer.commit().unwrap();

    let dir = &*directory;
    let files = dir.list_all().unwrap();
    let segments_file = files.iter().find(|f| f.starts_with("segments_")).unwrap();
    let infos = segment_infos::read(dir, segments_file).unwrap();
    let seg = &infos.segments[0];

    let compound_dir = CompoundDirectory::open(dir, &seg.name, &seg.id).unwrap();
    let compound_files = compound_dir.list_all().unwrap();

    assert_not_empty!(compound_files);
    assert_any!(compound_files.iter(), |f: &String| f.ends_with(".fnm"));
}

#[test]
fn test_compound_directory_read_embedded_file() {
    use bearing::document::DocumentBuilder;
    use bearing::index::config::IndexWriterConfig;
    use bearing::index::field::{keyword, text};
    use bearing::index::segment_infos;
    use bearing::index::writer::IndexWriter;

    let config = IndexWriterConfig::default()
        .num_threads(1)
        .use_compound_file(true);
    let directory: SharedDirectory = MemoryDirectory::create();
    let writer = IndexWriter::new(config, Arc::clone(&directory));

    let doc = DocumentBuilder::new()
        .add_field(text("body").value("embedded file test"))
        .add_field(keyword("tag").value("test"))
        .build();
    writer.add_document(doc).unwrap();
    writer.commit().unwrap();

    let dir = &*directory;
    let files = dir.list_all().unwrap();
    let segments_file = files.iter().find(|f| f.starts_with("segments_")).unwrap();
    let infos = segment_infos::read(dir, segments_file).unwrap();
    let seg = &infos.segments[0];

    let compound_dir = CompoundDirectory::open(dir, &seg.name, &seg.id).unwrap();

    let bytes = compound_dir.read_file(".fnm").unwrap();
    assert_ge!(bytes.len(), 4, ".fnm too short for codec magic");
    assert_eq!(
        &bytes[..4],
        &CODEC_MAGIC_BYTES,
        ".fnm in compound should start with CODEC_MAGIC"
    );
}

#[test]
fn test_compound_directory_memory() {
    use bearing::document::DocumentBuilder;
    use bearing::index::config::IndexWriterConfig;
    use bearing::index::field::{keyword, text};
    use bearing::index::segment_infos;
    use bearing::index::writer::IndexWriter;

    let config = IndexWriterConfig::default()
        .num_threads(1)
        .use_compound_file(true);
    let directory: SharedDirectory = MemoryDirectory::create();
    let writer = IndexWriter::new(config, Arc::clone(&directory));

    let doc = DocumentBuilder::new()
        .add_field(text("body").value("memory compound test"))
        .add_field(keyword("tag").value("test"))
        .build();
    writer.add_document(doc).unwrap();
    writer.commit().unwrap();

    let dir = &*directory;
    let files = dir.list_all().unwrap();
    let segments_file = files.iter().find(|f| f.starts_with("segments_")).unwrap();
    let infos = segment_infos::read(dir, segments_file).unwrap();
    let seg = &infos.segments[0];

    let compound_dir = CompoundDirectory::open(dir, &seg.name, &seg.id).unwrap();
    let compound_files = compound_dir.list_all().unwrap();
    assert_not_empty!(compound_files);

    let bytes = compound_dir.read_file(".fnm").unwrap();
    assert_ge!(bytes.len(), 4);
    assert_eq!(&bytes[..4], &CODEC_MAGIC_BYTES);
}

// ============================================================
// Stored fields reader integration tests
// ============================================================

#[test]
fn test_stored_fields_reader_round_trip() {
    use bearing::codecs::lucene90::stored_fields_reader::StoredFieldsReader;
    use bearing::document::DocumentBuilder;
    use bearing::document::StoredValue;
    use bearing::index::config::IndexWriterConfig;
    use bearing::index::field::stored;
    use bearing::index::segment_infos;
    use bearing::index::writer::IndexWriter;

    let config = IndexWriterConfig::default().num_threads(1);
    let directory: SharedDirectory = MemoryDirectory::create();
    let writer = IndexWriter::new(config, Arc::clone(&directory));

    let doc = DocumentBuilder::new()
        .add_field(stored("title").string("Hello World"))
        .add_field(stored("count").int(42))
        .build();
    writer.add_document(doc).unwrap();

    let doc2 = DocumentBuilder::new()
        .add_field(stored("title").string("Second Doc"))
        .add_field(stored("count").int(99))
        .build();
    writer.add_document(doc2).unwrap();

    writer.commit().unwrap();

    let dir = &*directory;
    let files = dir.list_all().unwrap();
    let segments_file = files
        .iter()
        .find(|f| f.starts_with("segments_"))
        .expect("no segments file");
    let infos = segment_infos::read(dir, segments_file).unwrap();
    let seg = &infos.segments[0];

    let mut reader = StoredFieldsReader::open(dir, &seg.name, "", &seg.id).unwrap();

    let fields = reader.document(0).unwrap();
    assert!(
        fields
            .iter()
            .any(|f| matches!(&f.value, StoredValue::String(s) if s == "Hello World")),
        "missing 'Hello World'"
    );
    assert!(
        fields
            .iter()
            .any(|f| matches!(&f.value, StoredValue::Int(42))),
        "missing Int(42)"
    );

    let fields1 = reader.document(1).unwrap();
    assert!(
        fields1
            .iter()
            .any(|f| matches!(&f.value, StoredValue::String(s) if s == "Second Doc")),
        "missing 'Second Doc'"
    );
    assert!(
        fields1
            .iter()
            .any(|f| matches!(&f.value, StoredValue::Int(99))),
        "missing Int(99)"
    );
}

#[test]
fn test_stored_fields_reader_all_types() {
    use bearing::codecs::lucene90::stored_fields_reader::StoredFieldsReader;
    use bearing::document::DocumentBuilder;
    use bearing::document::StoredValue;
    use bearing::index::config::IndexWriterConfig;
    use bearing::index::field::stored;
    use bearing::index::segment_infos;
    use bearing::index::writer::IndexWriter;

    let config = IndexWriterConfig::default().num_threads(1);
    let directory: SharedDirectory = MemoryDirectory::create();
    let writer = IndexWriter::new(config, Arc::clone(&directory));

    let doc = DocumentBuilder::new()
        .add_field(stored("s").string("text value"))
        .add_field(stored("i").int(12345))
        .add_field(stored("l").long(9876543210))
        .add_field(stored("f").float(3.125))
        .add_field(stored("d").double(2.7))
        .add_field(stored("b").bytes(vec![0xDE, 0xAD, 0xBE, 0xEF]))
        .build();
    writer.add_document(doc).unwrap();

    writer.commit().unwrap();

    let dir = &*directory;
    let files = dir.list_all().unwrap();
    let segments_file = files
        .iter()
        .find(|f| f.starts_with("segments_"))
        .expect("no segments file");
    let infos = segment_infos::read(dir, segments_file).unwrap();
    let seg = &infos.segments[0];

    let mut reader = StoredFieldsReader::open(dir, &seg.name, "", &seg.id).unwrap();
    let fields = reader.document(0).unwrap();

    assert!(
        fields
            .iter()
            .any(|f| matches!(&f.value, StoredValue::String(s) if s == "text value")),
        "missing string"
    );
    assert!(
        fields
            .iter()
            .any(|f| matches!(&f.value, StoredValue::Int(12345))),
        "missing int"
    );
    assert!(
        fields
            .iter()
            .any(|f| matches!(&f.value, StoredValue::Long(9876543210))),
        "missing long"
    );
    assert!(
        fields
            .iter()
            .any(|f| matches!(&f.value, StoredValue::Float(v) if (*v - 3.125).abs() < 0.001)),
        "missing float"
    );
    assert!(
        fields
            .iter()
            .any(|f| matches!(&f.value, StoredValue::Double(v) if (*v - 2.7).abs() < 0.001)),
        "missing double"
    );
    assert!(
        fields
            .iter()
            .any(|f| matches!(&f.value, StoredValue::Bytes(b) if b == &[0xDE, 0xAD, 0xBE, 0xEF])),
        "missing bytes"
    );
}

#[test]
fn test_stored_fields_reader_many_docs() {
    use bearing::codecs::lucene90::stored_fields_reader::StoredFieldsReader;
    use bearing::document::DocumentBuilder;
    use bearing::document::StoredValue;
    use bearing::index::config::IndexWriterConfig;
    use bearing::index::field::stored;
    use bearing::index::segment_infos;
    use bearing::index::writer::IndexWriter;

    let config = IndexWriterConfig::default().num_threads(1);
    let directory: SharedDirectory = MemoryDirectory::create();
    let writer = IndexWriter::new(config, Arc::clone(&directory));

    for i in 0..50 {
        let doc = DocumentBuilder::new()
            .add_field(stored("title").string(format!("Document {i}")))
            .add_field(stored("id").int(i))
            .build();
        writer.add_document(doc).unwrap();
    }

    writer.commit().unwrap();

    let dir = &*directory;
    let files = dir.list_all().unwrap();
    let segments_file = files
        .iter()
        .find(|f| f.starts_with("segments_"))
        .expect("no segments file");
    let infos = segment_infos::read(dir, segments_file).unwrap();
    let seg = &infos.segments[0];

    let mut reader = StoredFieldsReader::open(dir, &seg.name, "", &seg.id).unwrap();

    for i in 0..50 {
        let fields = reader.document(i).unwrap();
        let title = format!("Document {i}");
        assert!(
            fields
                .iter()
                .any(|f| matches!(&f.value, StoredValue::String(s) if s == &title)),
            "missing title for doc {i}"
        );
        assert!(
            fields
                .iter()
                .any(|f| matches!(&f.value, StoredValue::Int(v) if *v == i as i32)),
            "missing id for doc {i}"
        );
    }
}
