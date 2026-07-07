// SPDX-License-Identifier: Apache-2.0

//! Golden-file tests (SPEC §12.1): decode segments written by **real Java
//! Lucene 10.3.2** (`harness/run.sh golden harness/golden`) and compare
//! against the values the generator recorded in `expected.json`.
//!
//! Skipped (with a note) when `harness/golden` is absent — regenerate with
//! JDK 21 per `harness/README.md`.

use std::path::PathBuf;

use arrow_array::Array;
use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;

use lucene_arrow_codec::{DocValuesKind, SegmentDirectory, SegmentMeta};
use lucene_arrow_cpu::decode_numeric;
use lucene_arrow_docvalues::read::{DocValuesPlans, DvField, DvKind, plan_doc_values};

fn golden_root() -> Option<PathBuf> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../harness/golden");
    if root.join("expected.json").exists() {
        Some(root)
    } else {
        eprintln!("skipping: harness/golden not generated (needs JDK 21, see harness/README.md)");
        None
    }
}

fn expected() -> serde_json::Value {
    let root = golden_root().unwrap();
    serde_json::from_slice(&std::fs::read(root.join("expected.json")).unwrap()).unwrap()
}

fn plan_segment(dir: &SegmentDirectory, seg: &SegmentMeta) -> (DocValuesPlans, Vec<u8>) {
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
            has_skip_index: false,
        })
        .collect();
    let dvm_name = seg.files.iter().find(|f| f.ends_with(".dvm")).expect("dvm file");
    let dvd_name = seg.files.iter().find(|f| f.ends_with(".dvd")).expect("dvd file");
    let dvm = dir.open_input(&seg.name, dvm_name).unwrap();
    let dvd = dir.open_input(&seg.name, dvd_name).unwrap();
    let dvm = dvm.slice(0, dvm.len()).unwrap().to_vec();
    let dvd = dvd.slice(0, dvd.len()).unwrap().to_vec();
    let plans = plan_doc_values(&dvm, &dvd, &fields, seg.max_doc as u32, dvd_name).unwrap();
    (plans, dvd)
}

fn assert_field_matches(plans: &DocValuesPlans, dvd: &[u8], name: &str, expected: &[serde_json::Value]) {
    let plan = plans.plans.iter().find(|p| p.column.name == name).unwrap_or_else(|| {
        panic!("no plan for field {name}");
    });
    let array = decode_numeric(plan, dvd).unwrap();
    let ints = array.as_primitive::<Int64Type>();
    assert_eq!(ints.len(), expected.len(), "{name}: length");
    for (d, exp) in expected.iter().enumerate() {
        match exp.as_i64() {
            Some(v) => {
                assert!(ints.is_valid(d), "{name} doc {d}: expected value, got null");
                assert_eq!(ints.value(d), v, "{name} doc {d}");
            }
            None => assert!(ints.is_null(d), "{name} doc {d}: expected null"),
        }
    }
}

#[test]
fn golden_numerics_all_field_shapes() {
    let Some(root) = golden_root() else { return };
    let exp = expected();
    let dir = SegmentDirectory::open(root.join("numerics")).unwrap();
    assert_eq!(dir.segments().len(), 1);
    let seg = &dir.segments()[0];
    let (plans, dvd) = plan_segment(&dir, seg);
    assert!(plans.skipped.is_empty());

    let fields = exp["numerics"]["fields"].as_object().unwrap();
    assert_eq!(plans.plans.len(), fields.len());
    for (name, values) in fields {
        assert_field_matches(&plans, &dvd, name, values.as_array().unwrap());
    }
}

/// The field Java encodes with multi-block (blockwise) numeric mode — the
/// encoding Bearing never emits, so only real Java segments exercise it.
#[test]
fn golden_multiblock_numeric() {
    let Some(root) = golden_root() else { return };
    let exp = expected();
    let dir = SegmentDirectory::open(root.join("multiblock")).unwrap();
    let seg = &dir.segments()[0];
    let (plans, dvd) = plan_segment(&dir, seg);

    let plan = &plans.plans[0];
    assert!(
        plan.blocks.len() > 1,
        "expected Java to pick multi-block encoding, got {} block(s) — regenerate goldens \
         with a jumpier distribution",
        plan.blocks.len()
    );
    assert_field_matches(&plans, &dvd, "jumpy", exp["multiblock"]["fields"]["jumpy"].as_array().unwrap());
}

/// Deletes: tombstoned docs keep their doc values (positional semantics,
/// SPEC §7.1); the segment metadata must report the delete count.
#[test]
fn golden_deletes_positional_values_and_del_count() {
    let Some(root) = golden_root() else { return };
    let exp = expected();
    let dir = SegmentDirectory::open(root.join("deletes")).unwrap();
    let seg = &dir.segments()[0];
    let deleted = exp["deletes"]["deleted_docids"].as_array().unwrap();
    assert_eq!(seg.del_count as usize, deleted.len(), "del_count");

    let (plans, dvd) = plan_segment(&dir, seg);
    assert_field_matches(&plans, &dvd, "val", exp["deletes"]["fields"]["val"].as_array().unwrap());
}

/// Three segments, one commit: per-segment plans, local docid spaces.
#[test]
fn golden_multisegment_commit() {
    let Some(root) = golden_root() else { return };
    let exp = expected();
    let dir = SegmentDirectory::open(root.join("multisegment")).unwrap();
    let segments = exp["multisegment"]["segments"].as_array().unwrap();
    assert_eq!(dir.segments().len(), segments.len());

    for (seg, seg_expected) in dir.segments().iter().zip(segments) {
        let (plans, dvd) = plan_segment(&dir, seg);
        assert_field_matches(&plans, &dvd, "val", seg_expected.as_array().unwrap());
    }
}

/// `.liv` reader against real Java tombstones: deleted docids must match
/// the generator's record exactly (1 = live on disk).
#[test]
fn golden_live_docs_bitmap() {
    let Some(root) = golden_root() else { return };
    let exp = expected();
    let dir = SegmentDirectory::open(root.join("deletes")).unwrap();
    let seg = &dir.segments()[0];

    let words = dir.live_docs(&seg.name).unwrap().expect("segment has deletes");
    let deleted_expected: Vec<usize> = exp["deletes"]["deleted_docids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let mut deleted = Vec::new();
    for d in 0..seg.max_doc as usize {
        if words[d / 64] >> (d % 64) & 1 == 0 {
            deleted.push(d);
        }
    }
    assert_eq!(deleted, deleted_expected);

    // No-deletes segment → None.
    let dir2 = SegmentDirectory::open(root.join("numerics")).unwrap();
    assert!(dir2.live_docs(&dir2.segments()[0].name).unwrap().is_none());
}
