// SPDX-License-Identifier: Apache-2.0

//! P0 integration test (SPEC §13): write a real Lucene103 segment with
//! Bearing's public pipeline, open it with `SegmentDirectory`, and check
//! the typed inventory, raw byte access, and CodecUtil framing.

use std::path::Path;

use bearing::prelude::{
    DocumentBuilder, FSDirectory, IndexWriter, IndexWriterConfig, binary_dv, keyword,
    numeric_dv, sorted_dv, sorted_numeric_dv,
};
use lucene_arrow_codec::{DocValuesKind, SegmentDirectory, framing};

const NUM_DOCS: usize = 100;

/// Doc values codec constants, as written by Bearing's
/// `lucene90::doc_values` writer (suffix per Java's PerFieldDocValuesFormat).
const DV_META_CODEC: &str = "Lucene90DocValuesMetadata";
const DV_SUFFIX: &str = "Lucene90_0";
const DV_VERSION: i32 = 0;

fn build_index(path: &Path, compound: bool) {
    let directory = FSDirectory::open(path).unwrap();
    let config = IndexWriterConfig::default()
        .num_threads(1) // one worker => exactly one flushed segment
        .use_compound_file(compound);
    let writer = IndexWriter::new(config, directory);

    for i in 0..NUM_DOCS as i64 {
        let doc = DocumentBuilder::new()
            .add_field(numeric_dv("price").value(i * 10))
            .add_field(sorted_dv("category").value(format!("cat-{}", i % 7).into_bytes()))
            .add_field(sorted_numeric_dv("scores").value(vec![i, i * 2, 42]))
            .add_field(binary_dv("payload").value(vec![i as u8, 0xAB, 0xCD]))
            .add_field(keyword("tag").value(format!("tag-{}", i % 3)))
            .build();
        writer.add_document(doc).unwrap();
    }
    writer.commit().unwrap();
}

fn assert_inventory(dir: &SegmentDirectory, compound: bool) {
    assert_eq!(dir.segments().len(), 1, "expected exactly one segment");
    let seg = &dir.segments()[0];

    assert_eq!(seg.codec, "Lucene103");
    assert_eq!(seg.max_doc, NUM_DOCS as i32);
    assert_eq!(seg.del_count, 0);
    assert_eq!(seg.ord, 0);
    assert_eq!(seg.is_compound, compound);
    assert!(!seg.files.is_empty(), "segment files list must be non-empty");
    let mut sorted = seg.files.clone();
    sorted.sort();
    assert_eq!(seg.files, sorted, "files list must be sorted");

    let dv = |name: &str| dir.segments()[0].field(name).unwrap().doc_values;
    assert_eq!(dv("price"), DocValuesKind::Numeric);
    assert_eq!(dv("category"), DocValuesKind::Sorted);
    assert_eq!(dv("scores"), DocValuesKind::SortedNumeric);
    assert_eq!(dv("payload"), DocValuesKind::Binary);
    // keyword = exact-match indexed + SortedSet doc values.
    let tag = seg.field("tag").unwrap();
    assert_eq!(tag.doc_values, DocValuesKind::SortedSet);
    assert!(tag.indexed);

    let price = seg.field("price").unwrap();
    assert!(!price.indexed);
    assert!(!price.has_norms);
    assert!(!price.has_points);
    assert!(!price.has_vectors);

    // Field numbers are unique.
    let mut numbers: Vec<u32> = seg.fields.iter().map(|f| f.number).collect();
    numbers.sort_unstable();
    numbers.dedup();
    assert_eq!(numbers.len(), seg.fields.len());

    if compound {
        assert!(seg.files.iter().any(|f| f.ends_with(".cfs")));
        assert!(seg.files.iter().any(|f| f.ends_with(".cfe")));
    } else {
        assert!(seg.files.iter().any(|f| f.ends_with(".dvm")));
        assert!(seg.files.iter().any(|f| f.ends_with(".dvd")));
    }
}

/// Fetch the doc-values metadata file's bytes (through `.cfs` when
/// compound) and validate our framing implementation against them.
fn assert_dv_framing(dir: &SegmentDirectory, compound: bool) {
    let seg = &dir.segments()[0];
    let dvm_name = if compound {
        // Inner cfs entries are not in the outer files list; the name is
        // deterministic: <segment>_<per-field suffix>.dvm
        format!("{}_{DV_SUFFIX}.dvm", seg.name)
    } else {
        seg.files
            .iter()
            .find(|f| f.ends_with(".dvm"))
            .expect("segment must have a .dvm file")
            .clone()
    };

    for name in [dvm_name.clone(), dvm_name.replace(".dvm", ".dvd")] {
        let range = dir.open_input(&seg.name, &name).unwrap();
        assert!(!range.is_empty(), "{name} must be non-empty");
        let bytes = range.slice(0, range.len()).expect("host slice available");

        framing::verify_footer(bytes).unwrap();
        let expected_codec = if name.ends_with(".dvm") {
            DV_META_CODEC
        } else {
            "Lucene90DocValuesData"
        };
        let header_len = framing::check_index_header(
            bytes,
            expected_codec,
            DV_VERSION,
            DV_VERSION,
            &seg.id,
            DV_SUFFIX,
        )
        .unwrap();
        assert!(header_len > 0 && header_len < bytes.len());
        // Wrong codec name must be rejected as corrupt.
        assert!(
            framing::check_index_header(bytes, "NotACodec", 0, 0, &seg.id, DV_SUFFIX).is_err()
        );
    }
}

#[test]
fn p0_open_plain_segment() {
    let tmp = tempfile::tempdir().unwrap();
    build_index(tmp.path(), false);

    let dir = SegmentDirectory::open(tmp.path()).unwrap();
    assert_eq!(dir.generation(), 1);
    assert_inventory(&dir, false);
    assert_dv_framing(&dir, false);
}

#[test]
fn p0_open_compound_segment() {
    let tmp = tempfile::tempdir().unwrap();
    build_index(tmp.path(), true);

    let dir = SegmentDirectory::open(tmp.path()).unwrap();
    assert_inventory(&dir, true);
    assert_dv_framing(&dir, true);
}

#[test]
fn open_input_reads_and_bounds_checks() {
    let tmp = tempfile::tempdir().unwrap();
    build_index(tmp.path(), false);

    let dir = SegmentDirectory::open(tmp.path()).unwrap();
    let seg = &dir.segments()[0];
    let dvd = seg.files.iter().find(|f| f.ends_with(".dvd")).unwrap();
    let range = dir.open_input(&seg.name, dvd).unwrap();

    // read_into agrees with slice().
    let mut buf = vec![0u8; 4];
    range
        .read_into(0, 4, lucene_arrow_core::BufferTarget::Host(&mut buf))
        .unwrap();
    assert_eq!(&buf[..], range.slice(0, 4).unwrap());
    // And out-of-bounds fails.
    assert!(
        range
            .read_into(range.len(), 1, lucene_arrow_core::BufferTarget::Host(&mut buf))
            .is_err()
    );
    assert!(range.slice(range.len() - 1, 2).is_none());

    // Unknown segment / unknown file fail cleanly.
    assert!(dir.open_input("_nope", dvd).is_err());
    assert!(dir.open_input(&seg.name, "_0.doesnotexist").is_err());
}

#[test]
fn latest_generation_wins() {
    // Bearing's alpha writer always starts a fresh commit history, so fake
    // a newer commit point by cloning segments_1 -> segments_2. The index
    // header suffix embeds the generation ("1"), so patch it to "2" and
    // recompute the CRC32 footer (it covers everything but the stored CRC).
    let tmp = tempfile::tempdir().unwrap();
    build_index(tmp.path(), false);

    let mut bytes = std::fs::read(tmp.path().join("segments_1")).unwrap();
    // Header: magic(4) + "segments" string(1+8) + version(4) + id(16) +
    // suffix_len(1) + suffix -> the single suffix byte sits at offset 34.
    assert_eq!(bytes[33], 1, "suffix length");
    assert_eq!(bytes[34], b'1', "generation suffix");
    bytes[34] = b'2';
    let crc_at = bytes.len() - 8;
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(&bytes[..crc_at]);
    let crc = u64::from(hasher.finalize()).to_be_bytes();
    bytes[crc_at..].copy_from_slice(&crc);
    std::fs::write(tmp.path().join("segments_2"), &bytes).unwrap();

    let dir = SegmentDirectory::open(tmp.path()).unwrap();
    assert_eq!(dir.generation(), 2);
    assert_eq!(dir.segments()[0].max_doc, NUM_DOCS as i32);
}
