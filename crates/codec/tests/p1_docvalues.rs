// SPDX-License-Identifier: Apache-2.0

//! P1 cross-validation (SPEC §12.4): decode numeric doc values from real
//! Lucene103 segments written by Bearing's pipeline (itself cross-validated
//! against Java Lucene), and prove our encoder emits byte-identical
//! `.dvm`/`.dvd` files for the same input.

use std::path::Path;

use arrow_array::Array;
use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use bearing::prelude::{DocumentBuilder, FSDirectory, IndexWriter, IndexWriterConfig, numeric_dv};

use lucene_arrow_codec::{DocValuesKind, SegmentDirectory};
use lucene_arrow_cpu::decode_numeric;
use lucene_arrow_docvalues::file::DocValuesFileBuilder;
use lucene_arrow_docvalues::read::{DvField, DvKind, plan_doc_values};

const DV_SUFFIX: &str = "Lucene90_0";

/// Per-doc values for several numeric fields; `None` = field absent.
struct Fixture {
    name: &'static str,
    per_doc: Vec<Option<i64>>,
}

fn build_segment(path: &Path, fixtures: &[Fixture]) {
    let directory = FSDirectory::open(path).unwrap();
    let config = IndexWriterConfig::default().num_threads(1).use_compound_file(false);
    let writer = IndexWriter::new(config, directory);
    let num_docs = fixtures[0].per_doc.len();
    for d in 0..num_docs {
        let mut doc = DocumentBuilder::new();
        for f in fixtures {
            if let Some(v) = f.per_doc[d] {
                doc = doc.add_field(numeric_dv(f.name).value(v));
            }
        }
        writer.add_document(doc.build()).unwrap();
    }
    writer.commit().unwrap();
}

fn file_bytes(dir: &SegmentDirectory, seg: &str, ext: &str) -> (String, Vec<u8>) {
    let meta = dir.segment(seg).unwrap();
    let name = meta.files.iter().find(|f| f.ends_with(ext)).unwrap_or_else(|| {
        panic!("no {ext} file in {:?}", meta.files)
    });
    let range = dir.open_input(seg, name).unwrap();
    let bytes = range.slice(0, range.len()).expect("mmap-backed slice").to_vec();
    (name.clone(), bytes)
}

fn dv_fields(dir: &SegmentDirectory, seg: &str) -> Vec<DvField> {
    dir.segment(seg)
        .unwrap()
        .fields
        .iter()
        .filter(|f| f.doc_values != DocValuesKind::None)
        .map(|f| DvField {
            number: f.number as i32,
            name: f.name.clone(),
            kind: match f.doc_values {
                DocValuesKind::Numeric => DvKind::Numeric,
                DocValuesKind::Binary => DvKind::Binary,
                DocValuesKind::Sorted => DvKind::Sorted,
                DocValuesKind::SortedNumeric => DvKind::SortedNumeric,
                DocValuesKind::SortedSet => DvKind::SortedSet,
                DocValuesKind::None => unreachable!(),
            },
            has_skip_index: false,
        })
        .collect()
}

#[test]
fn decode_bearing_written_segment_matches_source_values() {
    let num_docs = 5000usize;
    let fixtures = vec![
        // Dense, gcd-friendly.
        Fixture { name: "price", per_doc: (0..num_docs).map(|i| Some(1000 + i as i64 * 25)).collect() },
        // Dense constant.
        Fixture { name: "flag", per_doc: vec![Some(7); num_docs] },
        // Dense, few distinct values → table encoding.
        Fixture {
            name: "bucket",
            per_doc: (0..num_docs).map(|i| Some([-1_000_000_007i64, 3, 900_719_925_474][i % 3])).collect(),
        },
        // Sparse (~1/6 of docs) → SPARSE DISI.
        Fixture {
            name: "rare",
            per_doc: (0..num_docs)
                .map(|i| if i % 6 == 1 { Some(i as i64 * -13) } else { None })
                .collect(),
        },
        // Sparse-dense (most docs) → DENSE DISI blocks.
        Fixture {
            name: "common",
            per_doc: (0..num_docs)
                .map(|i| if i % 10 != 0 { Some(i as i64) } else { None })
                .collect(),
        },
    ];

    let tmp = tempfile::tempdir().unwrap();
    build_segment(tmp.path(), &fixtures);

    let dir = SegmentDirectory::open(tmp.path()).unwrap();
    assert_eq!(dir.segments().len(), 1);
    let seg_name = dir.segments()[0].name.clone();
    let max_doc = dir.segments()[0].max_doc as u32;
    assert_eq!(max_doc as usize, num_docs);

    let (_, dvm) = file_bytes(&dir, &seg_name, ".dvm");
    let (dvd_name, dvd) = file_bytes(&dir, &seg_name, ".dvd");

    let fields = dv_fields(&dir, &seg_name);
    let plans = plan_doc_values(&dvm, &dvd, &fields, max_doc, &dvd_name).unwrap();
    assert_eq!(plans.plans.len(), fixtures.len(), "one plan per numeric field");
    assert!(plans.skipped.is_empty());

    for fixture in &fixtures {
        let plan = plans
            .plans
            .iter()
            .find(|p| p.column.name == fixture.name)
            .unwrap_or_else(|| panic!("no plan for {}", fixture.name));
        let array = decode_numeric(plan, &dvd).unwrap();
        let ints = array.as_primitive::<Int64Type>();
        assert_eq!(ints.len(), num_docs, "{}", fixture.name);
        for (d, expected) in fixture.per_doc.iter().enumerate() {
            match expected {
                Some(v) => {
                    assert!(ints.is_valid(d), "{} doc {d}", fixture.name);
                    assert_eq!(ints.value(d), *v, "{} doc {d}", fixture.name);
                }
                None => assert!(ints.is_null(d), "{} doc {d}", fixture.name),
            }
        }
    }
}

/// Byte-for-byte: our encoder must reproduce Bearing's `.dvm`/`.dvd`
/// exactly for the same values (Bearing's output is byte-identical to Java
/// Lucene — transitively, so is ours).
#[test]
fn encoder_is_byte_identical_to_bearing() {
    for (label, per_doc) in [
        ("dense-gcd", (0..2000).map(|i| Some(500 + i * 30)).collect::<Vec<_>>()),
        ("dense-table", (0..2000).map(|i| Some([11i64, -22, 1 << 40][i as usize % 3])).collect()),
        ("constant", vec![Some(-3); 999]),
        ("sparse", (0..6000).map(|i| if i % 5 == 0 { Some(i * 7) } else { None }).collect()),
        (
            "sparse-dense-disi",
            (0..80_000).map(|i| if i % 4 != 3 { Some(i) } else { None }).collect(),
        ),
    ] {
        let fixture = Fixture { name: "v", per_doc };
        let tmp = tempfile::tempdir().unwrap();
        build_segment(tmp.path(), std::slice::from_ref(&fixture));

        let dir = SegmentDirectory::open(tmp.path()).unwrap();
        let seg = &dir.segments()[0];
        let (seg_name, seg_id, max_doc) = (seg.name.clone(), seg.id, seg.max_doc as u32);
        let field_number = seg.field("v").unwrap().number as i32;

        let (_, bearing_dvm) = file_bytes(&dir, &seg_name, ".dvm");
        let (_, bearing_dvd) = file_bytes(&dir, &seg_name, ".dvd");

        let (docs, values): (Vec<u32>, Vec<i64>) = fixture
            .per_doc
            .iter()
            .enumerate()
            .filter_map(|(d, v)| v.map(|v| (d as u32, v)))
            .unzip();
        let mut builder = DocValuesFileBuilder::new(&seg_id, DV_SUFFIX);
        builder.add_numeric(field_number, &docs, &values, max_doc).unwrap();
        let (our_dvm, our_dvd) = builder.finish();

        assert_eq!(our_dvd, bearing_dvd, "{label}: .dvd bytes differ");
        assert_eq!(our_dvm, bearing_dvm, "{label}: .dvm bytes differ");
    }
}

/// SORTED write path byte-identity (§10.3): our terms-dict + ords encoder
/// must reproduce Bearing's `.dvm`/`.dvd` exactly (same LZ4, same
/// DirectMonotonic, same policy) — transitively byte-identical to Java.
#[test]
fn sorted_encoder_is_byte_identical_to_bearing() {
    use bearing::prelude::sorted_dv;
    use lucene_arrow_docvalues::write::CpuEncoder;

    for (label, num_docs, absent_mod, num_terms) in [
        ("dense-3-terms", 500usize, 0usize, 3usize),
        ("dense-200-terms", 3000, 0, 200),   // multiple LZ4 blocks
        ("sparse-90-terms", 4000, 7, 90),
    ] {
        let tmp = tempfile::tempdir().unwrap();
        {
            let directory = FSDirectory::open(tmp.path()).unwrap();
            let config = IndexWriterConfig::default().num_threads(1).use_compound_file(false);
            let writer = IndexWriter::new(config, directory);
            for i in 0..num_docs {
                let mut doc = DocumentBuilder::new();
                if absent_mod == 0 || i % absent_mod != 3 {
                    doc = doc.add_field(
                        sorted_dv("cat").value(format!("term-{:05}", i * 13 % num_terms).into_bytes()),
                    );
                } else {
                    // keep the doc non-empty so docids line up
                    doc = doc.add_field(numeric_dv("pad").value(i as i64));
                }
                writer.add_document(doc.build()).unwrap();
            }
            writer.commit().unwrap();
        }
        let dir = SegmentDirectory::open(tmp.path()).unwrap();
        let seg = &dir.segments()[0];
        let (_, bearing_dvm) = file_bytes(&dir, &seg.name, ".dvm");
        let (_, bearing_dvd) = file_bytes(&dir, &seg.name, ".dvd");
        let seg_id = seg.id;
        let cat_number = seg.field("cat").unwrap().number as i32;
        let pad_number = seg.field("pad").map(|f| f.number as i32);
        let max_doc = seg.max_doc as u32;

        // Rebuild the same file pair with our writers (field order = field
        // number order, matching Bearing's flush).
        let mut fields: Vec<(i32, bool)> = vec![(cat_number, true)];
        if let Some(p) = pad_number {
            fields.push((p, false));
        }
        fields.sort_by_key(|f| f.0);

        let mut builder = DocValuesFileBuilder::new(&seg_id, DV_SUFFIX);
        for (number, is_cat) in fields {
            if is_cat {
                let (docs, terms): (Vec<u32>, Vec<Vec<u8>>) = (0..num_docs)
                    .filter(|i| absent_mod == 0 || i % absent_mod != 3)
                    .map(|i| (i as u32, format!("term-{:05}", i * 13 % num_terms).into_bytes()))
                    .unzip();
                let term_refs: Vec<&[u8]> = terms.iter().map(|t| t.as_slice()).collect();
                builder
                    .add_sorted_with(&CpuEncoder, number, &docs, &term_refs, max_doc)
                    .unwrap();
            } else {
                let (docs, values): (Vec<u32>, Vec<i64>) = (0..num_docs)
                    .filter(|i| !(absent_mod == 0 || i % absent_mod != 3))
                    .map(|i| (i as u32, i as i64))
                    .unzip();
                builder.add_numeric(number, &docs, &values, max_doc).unwrap();
            }
        }
        let (our_dvm, our_dvd) = builder.finish();
        assert_eq!(our_dvd, bearing_dvd, "{label}: .dvd differs");
        assert_eq!(our_dvm, bearing_dvm, "{label}: .dvm differs");
    }
}

/// Full-matrix byte-identity: BINARY, multi-valued SORTED_NUMERIC and
/// SORTED_SET writers vs Bearing's pipeline on one segment.
#[test]
fn remaining_dv_writers_byte_identical_to_bearing() {
    use bearing::prelude::{binary_dv, sorted_numeric_dv, sorted_set_dv};
    use lucene_arrow_docvalues::write::CpuEncoder;

    let num_docs = 3000usize;
    let tmp = tempfile::tempdir().unwrap();
    {
        let directory = FSDirectory::open(tmp.path()).unwrap();
        let config = IndexWriterConfig::default().num_threads(1).use_compound_file(false);
        let writer = IndexWriter::new(config, directory);
        for i in 0..num_docs as i64 {
            let mut doc = DocumentBuilder::new();
            // blob: sparse var-length binary
            if i % 3 != 1 {
                doc = doc.add_field(binary_dv("blob").value(vec![i as u8; (i % 5) as usize + 1]));
            }
            // scores: multi sorted_numeric
            doc = doc.add_field(sorted_numeric_dv("scores").value(vec![i * 2, 42, i]));
            // tags: multi sorted_set, 0..3 terms
            let mut terms: Vec<Vec<u8>> = (0..i % 3)
                .map(|j| format!("tag-{:03}", (i + j * 37) % 151).into_bytes())
                .collect();
            terms.push(format!("base-{:02}", i % 41).into_bytes());
            doc = doc.add_field(sorted_set_dv("tags").value(terms));
            writer.add_document(doc.build()).unwrap();
        }
        writer.commit().unwrap();
    }

    let dir = SegmentDirectory::open(tmp.path()).unwrap();
    let seg = &dir.segments()[0];
    let (_, bearing_dvm) = file_bytes(&dir, &seg.name, ".dvm");
    let (_, bearing_dvd) = file_bytes(&dir, &seg.name, ".dvd");
    let max_doc = seg.max_doc as u32;

    // Field-number order drives file order.
    let mut ordered: Vec<(&str, u32)> =
        seg.fields.iter().map(|f| (f.name.as_str(), f.number)).collect();
    ordered.sort_by_key(|f| f.1);

    let mut builder = DocValuesFileBuilder::new(&seg.id, DV_SUFFIX);
    for (name, number) in ordered {
        match name {
            "blob" => {
                let (docs, vals): (Vec<u32>, Vec<Vec<u8>>) = (0..num_docs)
                    .filter(|i| i % 3 != 1)
                    .map(|i| (i as u32, vec![i as u8; i % 5 + 1]))
                    .unzip();
                let refs: Vec<&[u8]> = vals.iter().map(|v| v.as_slice()).collect();
                builder.add_binary(number as i32, &docs, &refs, max_doc).unwrap();
            }
            "scores" => {
                let docs: Vec<u32> = (0..num_docs as u32).collect();
                let vals: Vec<Vec<i64>> =
                    (0..num_docs as i64).map(|i| vec![i * 2, 42, i]).collect();
                builder
                    .add_sorted_numeric_with(&CpuEncoder, number as i32, &docs, &vals, max_doc)
                    .unwrap();
            }
            "tags" => {
                let docs: Vec<u32> = (0..num_docs as u32).collect();
                let terms: Vec<Vec<Vec<u8>>> = (0..num_docs as i64)
                    .map(|i| {
                        let mut t: Vec<Vec<u8>> = (0..i % 3)
                            .map(|j| format!("tag-{:03}", (i + j * 37) % 151).into_bytes())
                            .collect();
                        t.push(format!("base-{:02}", i % 41).into_bytes());
                        t
                    })
                    .collect();
                builder
                    .add_sorted_set_with(&CpuEncoder, number as i32, &docs, &terms, max_doc)
                    .unwrap();
            }
            other => panic!("unexpected field {other}"),
        }
    }
    let (our_dvm, our_dvd) = builder.finish();
    assert_eq!(our_dvd, bearing_dvd, ".dvd differs");
    assert_eq!(our_dvm, bearing_dvm, ".dvm differs");
}
