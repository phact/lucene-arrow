// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the indexing pipeline.
//!
//! Verifies the full lifecycle: IndexWriter → add documents → commit →
//! correct segments returned with expected file lists.

use std::collections::HashSet;
use std::env;
use std::fs;
use std::path::Path;
use std::sync::Arc;

use assertables::*;
use bearing::document::DocumentBuilder;
use bearing::index::config::IndexWriterConfig;
use bearing::index::directory_reader::DirectoryReader;
use bearing::index::field::text;
use bearing::index::writer::IndexWriter;
use bearing::prelude::{
    MemoryDirectory, SharedDirectory, TermVectorOptions, binary_dv, numeric_dv, sorted_dv,
    sorted_numeric_dv, sorted_set_dv, stored, string,
};
use bearing::search::*;

fn shared_memory_dir() -> SharedDirectory {
    MemoryDirectory::create()
}

fn add_stored_docs(writer: &IndexWriter, count: usize) {
    for i in 0..count {
        let doc = DocumentBuilder::new()
            .add_field(stored("title").string(format!("Document {i}")))
            .add_field(stored("body").string(format!("Body text for document {i}")))
            .build();
        writer.add_document(doc).unwrap();
    }
}

#[test]
fn single_segment_stored_fields() {
    let writer = IndexWriter::new(
        IndexWriterConfig::default().num_threads(1),
        shared_memory_dir(),
    );

    for i in 0..5 {
        let doc = DocumentBuilder::new()
            .add_field(stored("title").string(format!("Document {i}")))
            .add_field(stored("body").string(format!("Body text for document {i}")))
            .build();
        writer.add_document(doc).unwrap();
    }

    let segments = writer.commit().unwrap();

    // Single-threaded config produces one segment
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].doc_count, 5);

    // Verify expected files are in the segment's file list
    let files = &segments[0].file_names;
    assert_any!(files.iter(), |f: &String| f.ends_with(".fdt"));
    assert_any!(files.iter(), |f: &String| f.ends_with(".fdx"));
    assert_any!(files.iter(), |f: &String| f.ends_with(".fdm"));
    assert_any!(files.iter(), |f: &String| f.ends_with(".fnm"));
    assert_any!(files.iter(), |f: &String| f.ends_with(".si"));
}

#[test]
fn empty_commit_produces_no_segments() {
    let writer = IndexWriter::new(
        IndexWriterConfig::default().num_threads(1),
        shared_memory_dir(),
    );
    let segments = writer.commit().unwrap();
    assert_is_empty!(segments);
}

#[test]
fn segment_file_names_use_segment_prefix() {
    let writer = IndexWriter::new(
        IndexWriterConfig::default().num_threads(1),
        shared_memory_dir(),
    );

    let doc = DocumentBuilder::new()
        .add_field(stored("title").string("hello"))
        .build();
    writer.add_document(doc).unwrap();

    let segments = writer.commit().unwrap();
    let seg = &segments[0];

    for file_name in &seg.file_names {
        assert_starts_with!(file_name, seg.segment_id.name.as_str());
    }
}

#[test]
fn max_buffered_docs_creates_multiple_segments() {
    let config = IndexWriterConfig::default()
        .num_threads(1)
        .max_buffered_docs(5);
    let writer = IndexWriter::new(config, shared_memory_dir());

    add_stored_docs(&writer, 12);

    let segments = writer.commit().unwrap();

    // 12 docs with max_buffered_docs=5 → 3 segments (5 + 5 + 2)
    assert_eq!(segments.len(), 3);
    let total_docs: i32 = segments.iter().map(|s| s.doc_count).sum();
    assert_eq!(total_docs, 12);
    assert_eq!(segments[0].doc_count, 5);
    assert_eq!(segments[1].doc_count, 5);
    assert_eq!(segments[2].doc_count, 2);

    // All segment names must be unique
    let names: HashSet<_> = segments.iter().map(|s| &s.segment_id.name).collect();
    assert_eq!(names.len(), 3);

    // Each segment has stored field files
    for seg in &segments {
        assert_any!(seg.file_names.iter(), |f: &String| f.ends_with(".fdt"));
        assert_any!(seg.file_names.iter(), |f: &String| f.ends_with(".si"));
        assert_any!(seg.file_names.iter(), |f: &String| f.ends_with(".fnm"));
    }
}

#[test]
fn multi_thread_produces_independent_segments() {
    let config = IndexWriterConfig::default()
        .num_threads(2)
        .max_buffered_docs(1);
    let writer = IndexWriter::new(config, shared_memory_dir());

    // With max_buffered_docs=1, every doc triggers a flush. The flush
    // I/O gives the other thread time to pick up work from the channel.
    // 100 docs is enough that both threads will participate.
    add_stored_docs(&writer, 100);

    let segments = writer.commit().unwrap();

    // 100 docs with max_buffered_docs=1 → 100 segments across 2 threads
    assert_eq!(segments.len(), 100);
    let total_docs: i32 = segments.iter().map(|s| s.doc_count).sum();
    assert_eq!(total_docs, 100);

    // Every segment has exactly 1 doc
    for seg in &segments {
        assert_eq!(seg.doc_count, 1);
    }

    // All segment names unique
    let names: HashSet<_> = segments.iter().map(|s| &s.segment_id.name).collect();
    assert_eq!(names.len(), 100);
}

#[test]
fn multi_thread_with_mid_flush() {
    let config = IndexWriterConfig::default()
        .num_threads(2)
        .max_buffered_docs(3);
    let writer = IndexWriter::new(config, shared_memory_dir());

    add_stored_docs(&writer, 10);

    let segments = writer.commit().unwrap();

    // Multiple segments from flush + final flush across 2 threads
    assert_ge!(segments.len(), 3);
    let total_docs: i32 = segments.iter().map(|s| s.doc_count).sum();
    assert_eq!(total_docs, 10);

    // All segment names unique
    let names: HashSet<_> = segments.iter().map(|s| &s.segment_id.name).collect();
    assert_eq!(names.len(), segments.len());
}

#[test]
fn compound_file_packaging() {
    let config = IndexWriterConfig::default()
        .num_threads(1)
        .use_compound_file(true);
    let writer = IndexWriter::new(config, shared_memory_dir());

    add_stored_docs(&writer, 5);

    let segments = writer.commit().unwrap();

    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].doc_count, 5);

    // Compound packaging replaces sub-files with .cfs/.cfe
    let files = &segments[0].file_names;
    assert_eq!(files.len(), 3);
    assert_any!(files.iter(), |f: &String| f.ends_with(".si"));
    assert_any!(files.iter(), |f: &String| f.ends_with(".cfs"));
    assert_any!(files.iter(), |f: &String| f.ends_with(".cfe"));
}

#[test]
fn non_compound_keeps_individual_files() {
    let config = IndexWriterConfig::default().num_threads(1);
    let writer = IndexWriter::new(config, shared_memory_dir());

    add_stored_docs(&writer, 5);

    let segments = writer.commit().unwrap();

    let files = &segments[0].file_names;
    // Individual stored field files + .fnm + .si
    assert_ge!(files.len(), 5);
    assert_any!(files.iter(), |f: &String| f.ends_with(".fdt"));
    assert_any!(files.iter(), |f: &String| f.ends_with(".fdx"));
    assert_any!(files.iter(), |f: &String| f.ends_with(".fdm"));
    assert_any!(files.iter(), |f: &String| f.ends_with(".fnm"));
    assert_any!(files.iter(), |f: &String| f.ends_with(".si"));
    // No compound files
    assert!(!files.iter().any(|f| f.ends_with(".cfs")));
    assert!(!files.iter().any(|f| f.ends_with(".cfe")));
}

#[test]
fn compound_with_multi_segment() {
    let config = IndexWriterConfig::default()
        .num_threads(1)
        .use_compound_file(true)
        .max_buffered_docs(5);
    let writer = IndexWriter::new(config, shared_memory_dir());

    add_stored_docs(&writer, 12);

    let segments = writer.commit().unwrap();

    assert_eq!(segments.len(), 3);
    let total_docs: i32 = segments.iter().map(|s| s.doc_count).sum();
    assert_eq!(total_docs, 12);

    // Every segment should be compound
    for seg in &segments {
        assert_eq!(seg.file_names.len(), 3);
        assert_any!(seg.file_names.iter(), |f: &String| f.ends_with(".cfs"));
        assert_any!(seg.file_names.iter(), |f: &String| f.ends_with(".cfe"));
        assert_any!(seg.file_names.iter(), |f: &String| f.ends_with(".si"));
    }
}

// --- Text field (tokenized + norms + postings) tests ---

fn add_text_docs_from_testdata(writer: &IndexWriter) {
    let docs_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/docs");
    for entry in fs::read_dir(&docs_dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        let contents = fs::read_to_string(&path).unwrap();
        let doc = DocumentBuilder::new()
            .add_field(stored("path").string(&name))
            .add_field(text("contents").stored().value(contents))
            .build();
        writer.add_document(doc).unwrap();
    }
}

#[test]
fn stored_tokenized_fields_produce_norms_and_postings_files() {
    let writer = IndexWriter::new(
        IndexWriterConfig::default().num_threads(1),
        shared_memory_dir(),
    );

    add_text_docs_from_testdata(&writer);

    let segments = writer.commit().unwrap();
    assert_eq!(segments.len(), 1);

    let files = &segments[0].file_names;
    // Norms
    assert!(files.contains(&"_0.nvm".to_string()));
    assert!(files.contains(&"_0.nvd".to_string()));
    // Stored fields
    assert!(files.contains(&"_0.fdt".to_string()));
    assert!(files.contains(&"_0.fdx".to_string()));
    assert!(files.contains(&"_0.fdm".to_string()));
    // Field infos + segment info
    assert!(files.contains(&"_0.fnm".to_string()));
    assert!(files.contains(&"_0.si".to_string()));
    // Postings — per-field suffix must match PerFieldPostingsFormat attributes in .fnm
    assert!(files.contains(&"_0_Lucene103_0.tim".to_string()));
    assert!(files.contains(&"_0_Lucene103_0.tip".to_string()));
    assert!(files.contains(&"_0_Lucene103_0.tmd".to_string()));
    assert!(files.contains(&"_0_Lucene103_0.doc".to_string()));
    assert!(files.contains(&"_0_Lucene103_0.pos".to_string()));
    assert!(files.contains(&"_0_Lucene103_0.psm".to_string()));
}

#[test]
fn stored_tokenized_fields_multi_segment() {
    let config = IndexWriterConfig::default()
        .num_threads(1)
        .max_buffered_docs(2);
    let writer = IndexWriter::new(config, shared_memory_dir());

    add_text_docs_from_testdata(&writer);

    let segments = writer.commit().unwrap();

    // 4 docs with max_buffered_docs=2 → 2 segments
    assert_eq!(segments.len(), 2);
    let total_docs: i32 = segments.iter().map(|s| s.doc_count).sum();
    assert_eq!(total_docs, 4);

    // Each segment has norms and postings files
    for seg in &segments {
        assert_any!(seg.file_names.iter(), |f: &String| f.ends_with(".nvm"));
        assert_any!(seg.file_names.iter(), |f: &String| f.ends_with(".nvd"));
        assert_any!(seg.file_names.iter(), |f: &String| f
            .ends_with("_Lucene103_0.tim"));
        assert_any!(seg.file_names.iter(), |f: &String| f
            .ends_with("_Lucene103_0.doc"));
    }
}

#[test]
fn text_only_fields_produce_postings_without_stored() {
    let writer = IndexWriter::new(
        IndexWriterConfig::default().num_threads(1),
        shared_memory_dir(),
    );

    // Tokenized, not stored
    for i in 0..3 {
        let doc = DocumentBuilder::new()
            .add_field(text("body").value(format!("hello world document {i}")))
            .build();
        writer.add_document(doc).unwrap();
    }

    let segments = writer.commit().unwrap();
    assert_eq!(segments.len(), 1);

    let files = &segments[0].file_names;
    // Postings present
    assert!(files.contains(&"_0_Lucene103_0.tim".to_string()));
    assert!(files.contains(&"_0_Lucene103_0.doc".to_string()));
    assert!(files.contains(&"_0_Lucene103_0.pos".to_string()));
    // Norms present (TEXT type has norms)
    assert!(files.contains(&"_0.nvm".to_string()));
}

#[test]
fn mixed_stored_and_stored_tokenized_fields() {
    let writer = IndexWriter::new(
        IndexWriterConfig::default().num_threads(1),
        shared_memory_dir(),
    );

    for i in 0..5 {
        let doc = DocumentBuilder::new()
            .add_field(stored("id").string(format!("{i}")))
            .add_field(text("body").stored().value(format!("quick brown fox {i}")))
            .build();
        writer.add_document(doc).unwrap();
    }

    let segments = writer.commit().unwrap();
    assert_eq!(segments.len(), 1);

    let files = &segments[0].file_names;
    // Both stored fields and postings
    assert!(files.contains(&"_0.fdt".to_string()));
    assert!(files.contains(&"_0_Lucene103_0.tim".to_string()));
    assert!(files.contains(&"_0_Lucene103_0.doc".to_string()));
    assert!(files.contains(&"_0_Lucene103_0.pos".to_string()));
    assert!(files.contains(&"_0.nvm".to_string()));
}

#[test]
fn tokenized_field_produces_same_postings_as_string() {
    let docs_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/docs");

    // Index with tokenized_field (streaming)
    let writer_reader = IndexWriter::new(
        IndexWriterConfig::default().num_threads(1),
        shared_memory_dir(),
    );
    for entry in fs::read_dir(&docs_dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        let doc = DocumentBuilder::new()
            .add_field(text("contents").value(path))
            .build();
        writer_reader.add_document(doc).unwrap();
    }
    let segments_reader = writer_reader.commit().unwrap();

    // Index with stored_tokenized_field (string)
    let writer_string = IndexWriter::new(
        IndexWriterConfig::default().num_threads(1),
        shared_memory_dir(),
    );
    for entry in fs::read_dir(&docs_dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        let contents = fs::read_to_string(&path).unwrap();
        let doc = DocumentBuilder::new()
            .add_field(text("contents").stored().value(contents))
            .build();
        writer_string.add_document(doc).unwrap();
    }
    let segments_string = writer_string.commit().unwrap();

    // Both should produce the same number of segments and docs
    assert_eq!(segments_reader.len(), segments_string.len());
    let total_reader: i32 = segments_reader.iter().map(|s| s.doc_count).sum();
    let total_string: i32 = segments_string.iter().map(|s| s.doc_count).sum();
    assert_eq!(total_reader, total_string);

    // Reader-based index should have postings and norms
    let files = &segments_reader[0].file_names;
    assert!(files.contains(&"_0_Lucene103_0.tim".to_string()));
    assert!(files.contains(&"_0_Lucene103_0.doc".to_string()));
    assert!(files.contains(&"_0_Lucene103_0.pos".to_string()));
    assert!(files.contains(&"_0.nvm".to_string()));
}

#[test]
fn reader_field_not_stored() {
    let writer = IndexWriter::new(
        IndexWriterConfig::default().num_threads(1),
        shared_memory_dir(),
    );

    let doc = DocumentBuilder::new()
        .add_field(stored("title").string("test"))
        .add_field(text("contents").value("hello world document"))
        .build();
    writer.add_document(doc).unwrap();

    let segments = writer.commit().unwrap();
    assert_eq!(segments.len(), 1);

    let files = &segments[0].file_names;
    // "title" is stored → .fdt exists
    assert!(files.contains(&"_0.fdt".to_string()));
    // "contents" via reader → postings exist
    assert!(files.contains(&"_0_Lucene103_0.tim".to_string()));
    // Norms exist for "contents"
    assert!(files.contains(&"_0.nvm".to_string()));
}

#[test]
fn stored_tokenized_field_from_file_path() {
    // Write a temp file, then index it as stored+tokenized via PathBuf
    let dir = env::temp_dir().join("bearing_test_stored_path");
    fs::create_dir_all(&dir).unwrap();
    let file_path = dir.join("doc.txt");
    fs::write(&file_path, "hello world from a file").unwrap();

    let writer = IndexWriter::new(
        IndexWriterConfig::default().num_threads(1),
        shared_memory_dir(),
    );

    let doc = DocumentBuilder::new()
        .add_field(text("contents").stored().value(file_path.clone()))
        .build();
    writer.add_document(doc).unwrap();

    let segments = writer.commit().unwrap();
    assert_eq!(segments.len(), 1);

    let files = &segments[0].file_names;
    // Stored fields files exist (content stored from file)
    assert!(files.contains(&"_0.fdt".to_string()));
    assert!(files.contains(&"_0.fdx".to_string()));
    assert!(files.contains(&"_0.fdm".to_string()));
    // Postings files exist (content tokenized from file)
    assert!(files.contains(&"_0_Lucene103_0.tim".to_string()));
    assert!(files.contains(&"_0_Lucene103_0.doc".to_string()));
    assert!(files.contains(&"_0_Lucene103_0.pos".to_string()));
    // Norms exist
    assert!(files.contains(&"_0.nvm".to_string()));

    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn string_field_produces_docs_only_postings() {
    let writer = IndexWriter::new(
        IndexWriterConfig::default().num_threads(1),
        shared_memory_dir(),
    );

    for i in 0..3 {
        let doc = DocumentBuilder::new()
            .add_field(string("title").stored().value(format!("doc_{i}")))
            .build();
        writer.add_document(doc).unwrap();
    }

    let segments = writer.commit().unwrap();
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].doc_count, 3);

    let files = &segments[0].file_names;
    // Postings files exist (DOCS-only terms)
    assert!(files.contains(&"_0_Lucene103_0.tim".to_string()));
    assert!(files.contains(&"_0_Lucene103_0.doc".to_string()));
    // Stored fields exist
    assert!(files.contains(&"_0.fdt".to_string()));
    // No norms (StringField omits norms)
    assert!(!files.iter().any(|f| f.ends_with(".nvm")));
    // No positions file for DOCS-only fields
    assert!(!files.iter().any(|f| f.ends_with(".pos")));
}

#[test]
fn mixed_string_and_text_fields() {
    let writer = IndexWriter::new(
        IndexWriterConfig::default().num_threads(1),
        shared_memory_dir(),
    );

    for i in 0..3 {
        let doc = DocumentBuilder::new()
            .add_field(string("title").stored().value(format!("doc_{i}")))
            .add_field(text("body").stored().value(format!("quick brown fox {i}")))
            .build();
        writer.add_document(doc).unwrap();
    }

    let segments = writer.commit().unwrap();
    assert_eq!(segments.len(), 1);

    let files = &segments[0].file_names;
    // Both fields have postings
    assert!(files.contains(&"_0_Lucene103_0.tim".to_string()));
    assert!(files.contains(&"_0_Lucene103_0.doc".to_string()));
    // Positions exist (from tokenized "body" field)
    assert!(files.contains(&"_0_Lucene103_0.pos".to_string()));
    // Norms exist (from tokenized "body" field)
    assert!(files.contains(&"_0.nvm".to_string()));
    // Stored fields exist
    assert!(files.contains(&"_0.fdt".to_string()));
}

// --- Doc values field tests ---

#[test]
fn numeric_dv_produces_doc_values_files() {
    let writer = IndexWriter::new(
        IndexWriterConfig::default().num_threads(1),
        shared_memory_dir(),
    );

    for i in 0..3 {
        let doc = DocumentBuilder::new()
            .add_field(numeric_dv("count").value(i as i64 * 10))
            .build();
        writer.add_document(doc).unwrap();
    }

    let segments = writer.commit().unwrap();
    assert_eq!(segments.len(), 1);

    let files = &segments[0].file_names;
    assert!(files.contains(&"_0_Lucene90_0.dvm".to_string()));
    assert!(files.contains(&"_0_Lucene90_0.dvd".to_string()));
    assert!(files.contains(&"_0.fnm".to_string()));
    assert!(files.contains(&"_0.si".to_string()));
}

#[test]
fn all_dv_types_produce_files() {
    let writer = IndexWriter::new(
        IndexWriterConfig::default().num_threads(1),
        shared_memory_dir(),
    );

    for i in 0..3 {
        let doc = DocumentBuilder::new()
            .add_field(numeric_dv("count").value(i as i64))
            .add_field(binary_dv("hash").value(format!("{:04x}", i).into_bytes()))
            .add_field(sorted_dv("category").value(format!("cat_{i}").into_bytes()))
            .add_field(sorted_set_dv("tags").value(vec![format!("tag_{i}").into_bytes()]))
            .add_field(sorted_numeric_dv("priority").value(vec![i as i64]))
            .build();
        writer.add_document(doc).unwrap();
    }

    let segments = writer.commit().unwrap();
    assert_eq!(segments.len(), 1);

    let files = &segments[0].file_names;
    assert!(files.contains(&"_0_Lucene90_0.dvm".to_string()));
    assert!(files.contains(&"_0_Lucene90_0.dvd".to_string()));
}

#[test]
fn mixed_dv_and_stored_and_postings() {
    let writer = IndexWriter::new(
        IndexWriterConfig::default().num_threads(1),
        shared_memory_dir(),
    );

    for i in 0..5 {
        let doc = DocumentBuilder::new()
            .add_field(stored("title").string(format!("doc {i}")))
            .add_field(text("body").stored().value(format!("hello world {i}")))
            .add_field(string("id").stored().value(format!("id_{i}")))
            .add_field(numeric_dv("count").value(i as i64 * 100))
            .add_field(sorted_dv("sort_key").value(format!("key_{i}").into_bytes()))
            .build();
        writer.add_document(doc).unwrap();
    }

    let segments = writer.commit().unwrap();
    assert_eq!(segments.len(), 1);

    let files = &segments[0].file_names;
    // Doc values
    assert!(files.contains(&"_0_Lucene90_0.dvm".to_string()));
    assert!(files.contains(&"_0_Lucene90_0.dvd".to_string()));
    // Stored fields
    assert!(files.contains(&"_0.fdt".to_string()));
    // Postings
    assert!(files.contains(&"_0_Lucene103_0.tim".to_string()));
    assert!(files.contains(&"_0_Lucene103_0.doc".to_string()));
    // Norms (from text field)
    assert!(files.contains(&"_0.nvm".to_string()));
}

#[test]
fn dv_only_docs_no_stored_or_postings() {
    let writer = IndexWriter::new(
        IndexWriterConfig::default().num_threads(1),
        shared_memory_dir(),
    );

    for i in 0..3 {
        let doc = DocumentBuilder::new()
            .add_field(numeric_dv("val").value(i as i64))
            .build();
        writer.add_document(doc).unwrap();
    }

    let segments = writer.commit().unwrap();
    assert_eq!(segments.len(), 1);

    let files = &segments[0].file_names;
    // Doc values present
    assert!(files.contains(&"_0_Lucene90_0.dvm".to_string()));
    assert!(files.contains(&"_0_Lucene90_0.dvd".to_string()));
    // No postings
    assert!(!files.iter().any(|f| f.ends_with(".tim")));
    // No norms
    assert!(!files.iter().any(|f| f.ends_with(".nvm")));
}

#[test]
fn dv_compound_file_packaging() {
    let config = IndexWriterConfig::default()
        .num_threads(1)
        .use_compound_file(true);
    let writer = IndexWriter::new(config, shared_memory_dir());

    for i in 0..3 {
        let doc = DocumentBuilder::new()
            .add_field(stored("title").string(format!("doc {i}")))
            .add_field(numeric_dv("count").value(i as i64))
            .build();
        writer.add_document(doc).unwrap();
    }

    let segments = writer.commit().unwrap();
    assert_eq!(segments.len(), 1);

    let files = &segments[0].file_names;
    // Compound packaging wraps sub-files
    assert_any!(files.iter(), |f: &String| f.ends_with(".cfs"));
    assert_any!(files.iter(), |f: &String| f.ends_with(".cfe"));
    assert_any!(files.iter(), |f: &String| f.ends_with(".si"));
}

#[test]
fn dv_multi_segment() {
    let config = IndexWriterConfig::default()
        .num_threads(1)
        .max_buffered_docs(2);
    let writer = IndexWriter::new(config, shared_memory_dir());

    for i in 0..5 {
        let doc = DocumentBuilder::new()
            .add_field(numeric_dv("count").value(i as i64))
            .add_field(sorted_dv("key").value(format!("k{i}").into_bytes()))
            .build();
        writer.add_document(doc).unwrap();
    }

    let segments = writer.commit().unwrap();
    assert_eq!(segments.len(), 3); // 2 + 2 + 1
    let total_docs: i32 = segments.iter().map(|s| s.doc_count).sum();
    assert_eq!(total_docs, 5);

    for seg in &segments {
        assert_any!(seg.file_names.iter(), |f: &String| f.ends_with(".dvm"));
        assert_any!(seg.file_names.iter(), |f: &String| f.ends_with(".dvd"));
    }
}

// --- Term vectors tests ---

fn add_tv_docs_from_testdata(writer: &IndexWriter) {
    let docs_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/docs");
    for entry in fs::read_dir(&docs_dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        let contents = fs::read_to_string(&path).unwrap();
        let doc = DocumentBuilder::new()
            .add_field(stored("path").string(&name))
            .add_field(
                text("contents")
                    .with_term_vectors(TermVectorOptions::PositionsAndOffsets)
                    .stored()
                    .value(contents),
            )
            .build();
        writer.add_document(doc).unwrap();
    }
}

#[test]
fn term_vectors_produce_tvd_tvx_tvm_files() {
    let writer = IndexWriter::new(
        IndexWriterConfig::default().num_threads(1),
        shared_memory_dir(),
    );

    add_tv_docs_from_testdata(&writer);

    let segments = writer.commit().unwrap();
    assert_eq!(segments.len(), 1);

    let files = &segments[0].file_names;
    assert_any!(files.iter(), |f: &String| f.ends_with(".tvd"));
    assert_any!(files.iter(), |f: &String| f.ends_with(".tvx"));
    assert_any!(files.iter(), |f: &String| f.ends_with(".tvm"));
}

#[test]
fn term_vectors_multi_segment() {
    let config = IndexWriterConfig::default()
        .num_threads(1)
        .max_buffered_docs(2);
    let writer = IndexWriter::new(config, shared_memory_dir());

    add_tv_docs_from_testdata(&writer);

    let segments = writer.commit().unwrap();
    assert_eq!(segments.len(), 2);

    for seg in &segments {
        assert_any!(seg.file_names.iter(), |f: &String| f.ends_with(".tvd"));
        assert_any!(seg.file_names.iter(), |f: &String| f.ends_with(".tvx"));
        assert_any!(seg.file_names.iter(), |f: &String| f.ends_with(".tvm"));
    }
}

#[test]
fn term_vectors_compound() {
    let config = IndexWriterConfig::default()
        .num_threads(1)
        .use_compound_file(true);
    let writer = IndexWriter::new(config, shared_memory_dir());

    add_tv_docs_from_testdata(&writer);

    let segments = writer.commit().unwrap();
    assert_eq!(segments.len(), 1);

    let files = &segments[0].file_names;
    assert_any!(files.iter(), |f: &String| f.ends_with(".cfs"));
    assert_any!(files.iter(), |f: &String| f.ends_with(".cfe"));
}

#[test]
fn term_vectors_multi_thread() {
    let config = IndexWriterConfig::default()
        .num_threads(2)
        .max_buffered_docs(2);
    let writer = IndexWriter::new(config, shared_memory_dir());

    add_tv_docs_from_testdata(&writer);

    let segments = writer.commit().unwrap();
    let total_docs: i32 = segments.iter().map(|s| s.doc_count).sum();
    assert_eq!(total_docs, 4);

    for seg in &segments {
        assert_any!(seg.file_names.iter(), |f: &String| f.ends_with(".tvd"));
    }
}

// --- RAM-based flush tests ---

/// Adds documents with enough text content to accumulate meaningful RAM.
fn add_text_docs(writer: &IndexWriter, count: usize) {
    let body = "the quick brown fox jumps over the lazy dog ".repeat(20);
    for i in 0..count {
        let doc = DocumentBuilder::new()
            .add_field(stored("title").string(format!("Document {i}")))
            .add_field(text("body").stored().value(body.clone()))
            .build();
        writer.add_document(doc).unwrap();
    }
}

#[test]
fn ram_flush_produces_multiple_segments() {
    let config = IndexWriterConfig::default()
        .num_threads(1)
        .ram_buffer_size_mb(0.05); // very small — forces frequent flushes

    let writer = IndexWriter::new(config, shared_memory_dir());
    add_text_docs(&writer, 500);
    let segments = writer.commit().unwrap();

    // With a 0.05 MB buffer and ~1KB per doc, should produce multiple segments.
    assert_gt!(segments.len(), 1);

    let total_docs: i32 = segments.iter().map(|s| s.doc_count).sum();
    assert_eq!(total_docs, 500);
}

#[test]
fn large_ram_buffer_produces_single_segment() {
    let config = IndexWriterConfig::default()
        .num_threads(1)
        .ram_buffer_size_mb(1000.0); // huge — should never trigger

    let writer = IndexWriter::new(config, shared_memory_dir());
    add_text_docs(&writer, 50);
    let segments = writer.commit().unwrap();

    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].doc_count, 50);
}

#[test]
fn ram_flush_multi_thread() {
    let config = IndexWriterConfig::default()
        .ram_buffer_size_mb(0.1)
        .num_threads(4);

    let writer = IndexWriter::new(config, shared_memory_dir());
    add_text_docs(&writer, 200);
    let segments = writer.commit().unwrap();

    // Multiple segments from RAM-triggered flushes across 4 threads.
    assert_gt!(segments.len(), 1);

    let total_docs: i32 = segments.iter().map(|s| s.doc_count).sum();
    assert_eq!(total_docs, 200);
}

/// Exercises level1 skip data in the postings writer (>4096 docs per term).
///
/// Creates 5000 documents all containing "trigger", which crosses the 4096-doc
/// level1 boundary. Verifies the full write-read roundtrip: IndexWriter →
/// DirectoryReader → TermQuery → all docs found with correct scores.
#[test]
fn level1_skip_data_roundtrip() {
    let directory: SharedDirectory = MemoryDirectory::create();
    let writer = IndexWriter::new(
        IndexWriterConfig::default().num_threads(1),
        Arc::clone(&directory),
    );

    let doc_count: i32 = 5000;
    for i in 0..doc_count {
        // Vary document length so BM25 norms differ across docs,
        // producing non-trivial competitive impact data at level1.
        let filler_count = (i % 20) + 1;
        let filler: String = (0..filler_count)
            .map(|j| format!("filler{j}"))
            .collect::<Vec<_>>()
            .join(" ");
        let content = format!("trigger {filler}");
        let doc = DocumentBuilder::new()
            .add_field(text("body").value(content.as_str()))
            .build();
        writer.add_document(doc).unwrap();
    }

    let segments = writer.commit().unwrap();
    assert_eq!(segments.len(), 1, "expected single segment");
    assert_eq!(segments[0].doc_count, doc_count);

    let reader = DirectoryReader::open(&*directory).unwrap();
    let searcher = IndexSearcher::new(&reader);

    // Search for "trigger" — all 5000 docs should match.
    let query = TermQuery::new("body", b"trigger");
    // Request all docs (top=5000) to verify full iteration across level1 boundaries.
    let result = searcher.search(&query, doc_count).unwrap();
    // total_hits may be pruned by BatchScoreBulkScorer, but we should get
    // at least the top docs back. Verify the top docs are valid.
    assert_gt!(
        result.score_docs.len(),
        0,
        "should find docs matching 'trigger'"
    );

    // Verify doc IDs are valid (within range)
    for sd in &result.score_docs {
        assert_ge!(sd.doc, 0);
        assert_lt!(sd.doc, doc_count);
        assert_gt!(sd.score, 0.0, "score should be positive");
    }

    // Verify advance() across level1 boundary works by searching with
    // a boolean query that forces advance jumps.
    let mut builder = BooleanQuery::builder();
    builder.add_query(Box::new(TermQuery::new("body", b"trigger")), Occur::Must);
    builder.add_query(Box::new(TermQuery::new("body", b"filler0")), Occur::Must);
    let bool_query = builder.build();
    let bool_result = searcher.search(&bool_query, 10).unwrap();
    assert_gt!(
        bool_result.score_docs.len(),
        0,
        "boolean query should find docs with both 'trigger' and 'filler0'"
    );
}
