// SPDX-License-Identifier: Apache-2.0

//! P3 cross-validation: SORTED / SORTED_SET / multi-valued SORTED_NUMERIC
//! decoded from a real Lucene103 segment written by Bearing's pipeline.

use std::path::Path;

use arrow_array::cast::AsArray;
use arrow_array::types::{Int32Type, Int64Type};
use arrow_array::{Array, StringArray};
use bearing::prelude::{
    DocumentBuilder, FSDirectory, IndexWriter, IndexWriterConfig, binary_dv, keyword, numeric_dv,
    sorted_dv, sorted_numeric_dv,
};

use lucene_arrow_codec::{DocValuesKind, SegmentDirectory};
use lucene_arrow_cpu::{decode_multi_numeric, decode_sorted};
use lucene_arrow_docvalues::read::{DvField, DvKind, plan_doc_values};

const NUM_DOCS: usize = 3000;

fn build(path: &Path) {
    let directory = FSDirectory::open(path).unwrap();
    let config = IndexWriterConfig::default().num_threads(1).use_compound_file(false);
    let writer = IndexWriter::new(config, directory);
    for i in 0..NUM_DOCS as i64 {
        let mut doc = DocumentBuilder::new()
            .add_field(numeric_dv("id").value(i))
            .add_field(sorted_numeric_dv("scores").value(vec![i * 2, 42, i]));
        // category: sparse sorted — absent on every 7th doc; >64 distinct
        // values so the terms dict spans multiple LZ4 blocks.
        if i % 7 != 3 {
            doc = doc.add_field(sorted_dv("category").value(format!("cat-{:04}", i % 200).into_bytes()));
        }
        // tag: keyword → SORTED_SET, single-valued.
        doc = doc.add_field(keyword("tag").value(format!("tag-{}", i % 3)));
        // blob: sparse variable-length BINARY.
        if i % 3 != 1 {
            doc = doc.add_field(binary_dv("blob").value(vec![i as u8; (i % 5) as usize + 1]));
        }
        writer.add_document(doc.build()).unwrap();
    }
    writer.commit().unwrap();
}

#[test]
fn sorted_and_multi_shapes_round_trip_through_bearing() {
    let tmp = tempfile::tempdir().unwrap();
    build(tmp.path());

    let dir = SegmentDirectory::open(tmp.path()).unwrap();
    let seg = &dir.segments()[0];
    let fields: Vec<DvField> = seg
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
            has_skip_index: f.has_skip_index,
        })
        .collect();

    let dvm_name = seg.files.iter().find(|f| f.ends_with(".dvm")).unwrap();
    let dvd_name = seg.files.iter().find(|f| f.ends_with(".dvd")).unwrap();
    let dvm = dir.open_input(&seg.name, dvm_name).unwrap();
    let dvd = dir.open_input(&seg.name, dvd_name).unwrap();
    let dvm = dvm.slice(0, dvm.len()).unwrap();
    let dvd = dvd.slice(0, dvd.len()).unwrap();

    let plans = plan_doc_values(dvm, dvd, &fields, NUM_DOCS as u32, dvd_name).unwrap();
    assert_eq!(plans.plans.len(), 1, "id");
    assert_eq!(plans.sorted.len(), 2, "category + tag");
    assert_eq!(plans.multi_numeric.len(), 1, "scores");
    assert_eq!(plans.binary.len(), 1, "blob");
    assert!(plans.skipped.is_empty());

    // --- blob: sparse variable-length BINARY ---
    let blob = &plans.binary[0];
    let array = lucene_arrow_cpu::decode_binary(blob, dvd).unwrap();
    let bin = array.as_binary::<i32>();
    for d in 0..NUM_DOCS {
        if d % 3 != 1 {
            assert!(bin.is_valid(d), "blob doc {d}");
            assert_eq!(bin.value(d), vec![d as u8; d % 5 + 1].as_slice(), "blob doc {d}");
        } else {
            assert!(bin.is_null(d), "blob doc {d}");
        }
    }

    // --- category: sparse SORTED, 200 distinct terms (4 LZ4 blocks) ---
    let cat = plans.sorted.iter().find(|p| p.ords.column.name == "category").unwrap();
    assert!(!cat.set);
    assert_eq!(cat.terms.num_terms, 200);
    let array = decode_sorted(cat, dvd).unwrap();
    let dict = array.as_dictionary::<Int32Type>();
    let terms: &StringArray = dict.values().as_string();
    for d in 0..NUM_DOCS {
        if d % 7 != 3 {
            assert!(dict.is_valid(d), "doc {d}");
            let key = dict.keys().value(d) as usize;
            assert_eq!(terms.value(key), format!("cat-{:04}", d % 200), "doc {d}");
        } else {
            assert!(dict.is_null(d), "doc {d}");
        }
    }

    // --- tag: keyword → SORTED_SET single-valued, dense ---
    let tag = plans.sorted.iter().find(|p| p.ords.column.name == "tag").unwrap();
    assert!(tag.set);
    assert!(tag.addresses.is_none(), "single-valued set");
    assert_eq!(tag.terms.num_terms, 3);
    let array = decode_sorted(tag, dvd).unwrap();
    let dict = array.as_dictionary::<Int32Type>();
    let terms: &StringArray = dict.values().as_string();
    for d in 0..NUM_DOCS {
        let key = dict.keys().value(d) as usize;
        assert_eq!(terms.value(key), format!("tag-{}", d % 3), "doc {d}");
    }

    // --- scores: multi-valued SORTED_NUMERIC → List<Int64>, per-doc sorted ---
    let scores = &plans.multi_numeric[0];
    assert_eq!(scores.values.column.name, "scores");
    let array = decode_multi_numeric(scores, dvd).unwrap();
    let list = array.as_list::<i32>();
    assert_eq!(list.len(), NUM_DOCS);
    let child = list.values().as_primitive::<Int64Type>();
    for d in 0..NUM_DOCS {
        let (start, end) = (list.value_offsets()[d] as usize, list.value_offsets()[d + 1] as usize);
        let got: Vec<i64> = (start..end).map(|i| child.value(i)).collect();
        let mut expected = vec![d as i64 * 2, 42, d as i64];
        expected.sort();
        assert_eq!(got, expected, "doc {d}");
    }
}
