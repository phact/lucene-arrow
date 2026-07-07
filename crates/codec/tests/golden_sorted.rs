// SPDX-License-Identifier: Apache-2.0

//! Golden dictionary tests against **real Java Lucene 10.3.2** output:
//! three segments (per-segment dictionaries differ), LZ4 term blocks,
//! DirectMonotonic addresses. Value formulas per
//! `GenerateGolden.writeKeywords`; global doc `g = segment_ord * 1000 + d`.
//!
//! Covers both §7.3 dictionary modes implemented so far:
//! `segment` (per-segment dicts) and `global` (OrdinalMap + fused remap).

use arrow_array::cast::AsArray;
use arrow_array::types::{Int32Type, Int64Type};
use arrow_array::{Array, StringArray};

use lucene_arrow_codec::{DocValuesKind, SegmentDirectory, SegmentMeta};
use lucene_arrow_cpu::{decode_multi_numeric, decode_numeric, decode_sorted};
use lucene_arrow_docvalues::read::{DocValuesPlans, DvField, DvKind, plan_doc_values};
use lucene_arrow_docvalues::{ordmap, terms};

const SEG_DOCS: usize = 1000;

fn cat_term(g: usize) -> String {
    format!("cat-{:04}", g * 7 % 501)
}

fn open() -> Option<SegmentDirectory> {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../harness/golden");
    if !root.join("keywords").is_dir() {
        eprintln!("skipping: harness/golden not generated (needs JDK 21)");
        return None;
    }
    Some(SegmentDirectory::open(root.join("keywords")).unwrap())
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
            has_skip_index: f.has_skip_index,
        })
        .collect();
    let dvm_name = seg.files.iter().find(|f| f.ends_with(".dvm")).unwrap();
    let dvd_name = seg.files.iter().find(|f| f.ends_with(".dvd")).unwrap();
    let dvm_r = dir.open_input(&seg.name, dvm_name).unwrap();
    let dvd_r = dir.open_input(&seg.name, dvd_name).unwrap();
    let dvm = dvm_r.slice(0, dvm_r.len()).unwrap().to_vec();
    let dvd = dvd_r.slice(0, dvd_r.len()).unwrap().to_vec();
    let plans = plan_doc_values(&dvm, &dvd, &fields, seg.max_doc as u32, dvd_name).unwrap();
    (plans, dvd)
}

#[test]
fn golden_java_dictionaries_per_segment() {
    let Some(dir) = open() else { return };
    assert_eq!(dir.segments().len(), 3, "flush() every 1000 docs");

    for seg in dir.segments() {
        assert_eq!(seg.max_doc as usize, SEG_DOCS);
        let base = seg.ord * SEG_DOCS;
        let (plans, dvd) = plan_segment(&dir, seg);
        assert!(plans.skipped.is_empty());

        // --- cat: sparse SORTED ---
        let cat = plans.sorted.iter().find(|p| p.ords.column.name == "cat").unwrap();
        let array = decode_sorted(cat, &dvd).unwrap();
        let dict = array.as_dictionary::<Int32Type>();
        let terms_arr: &StringArray = dict.values().as_string();
        for d in 0..SEG_DOCS {
            let g = base + d;
            if g % 5 != 2 {
                assert!(dict.is_valid(d), "cat g{g}");
                assert_eq!(terms_arr.value(dict.keys().value(d) as usize), cat_term(g), "cat g{g}");
            } else {
                assert!(dict.is_null(d), "cat g{g}");
            }
        }

        // --- tags: multi-valued SORTED_SET → List<Dictionary> ---
        let tags = plans.sorted.iter().find(|p| p.ords.column.name == "tags").unwrap();
        let array = decode_sorted(tags, &dvd).unwrap();
        let list = array.as_list::<i32>();
        let child = list.values().as_dictionary::<Int32Type>();
        let child_terms: &StringArray = child.values().as_string();
        for d in 0..SEG_DOCS {
            let g = base + d;
            let mut expected: Vec<String> =
                (0..g % 4).map(|j| format!("tag-{:03}", (g + j * 37) % 211)).collect();
            expected.sort();
            if expected.is_empty() {
                assert!(list.is_null(d), "tags g{g}");
                continue;
            }
            let (s, e) = (list.value_offsets()[d] as usize, list.value_offsets()[d + 1] as usize);
            let got: Vec<String> = (s..e)
                .map(|i| child_terms.value(child.keys().value(i) as usize).to_string())
                .collect();
            assert_eq!(got, expected, "tags g{g}");
        }

        // --- nums: SORTED_NUMERIC → List<Int64> ---
        let nums = &plans.multi_numeric[0];
        let array = decode_multi_numeric(nums, &dvd).unwrap();
        let list = array.as_list::<i32>();
        let child = list.values().as_primitive::<Int64Type>();
        for d in 0..SEG_DOCS {
            let g = base + d;
            let mut expected: Vec<i64> = (0..1 + g % 3)
                .map(|j| match j {
                    0 => g as i64 * 5,
                    1 => g as i64 * 5 - 100,
                    _ => 7,
                })
                .collect();
            expected.sort();
            let (s, e) = (list.value_offsets()[d] as usize, list.value_offsets()[d + 1] as usize);
            let got: Vec<i64> = (s..e).map(|i| child.value(i)).collect();
            assert_eq!(got, expected, "nums g{g}");
        }
    }
}

/// `dict = global` (SPEC §7.3): OrdinalMap across the 3 segments; the
/// remap rides the Table epilogue (`ordmap::apply_remap`), so global keys
/// come straight out of the standard numeric decode.
#[test]
fn golden_java_global_ordinal_map() {
    let Some(dir) = open() else { return };

    let per_seg: Vec<(DocValuesPlans, Vec<u8>)> =
        dir.segments().iter().map(|s| plan_segment(&dir, s)).collect();
    let dicts: Vec<terms::TermsDict> = per_seg
        .iter()
        .map(|(plans, dvd)| {
            let cat = plans.sorted.iter().find(|p| p.ords.column.name == "cat").unwrap();
            terms::materialize(&cat.terms, dvd).unwrap()
        })
        .collect();

    let map = ordmap::build(&dicts.iter().collect::<Vec<_>>()).unwrap();
    // Global dictionary must be the sorted union: every cat term ever used.
    let expected_terms: std::collections::BTreeSet<String> =
        (0..3 * SEG_DOCS).filter(|g| g % 5 != 2).map(cat_term).collect();
    assert_eq!(map.values.len(), expected_terms.len());
    for (i, t) in expected_terms.iter().enumerate() {
        assert_eq!(map.values.term(i), t.as_bytes(), "global ord {i}");
    }

    for (seg_ord, (plans, dvd)) in per_seg.iter().enumerate() {
        let cat = plans.sorted.iter().find(|p| p.ords.column.name == "cat").unwrap();
        let global_plan = ordmap::apply_remap(&cat.ords, &map.remap[seg_ord]).unwrap();
        let array = decode_numeric(&global_plan, dvd).unwrap();
        let keys = array.as_primitive::<Int64Type>();
        for d in 0..SEG_DOCS {
            let g = seg_ord * SEG_DOCS + d;
            if g % 5 != 2 {
                assert!(keys.is_valid(d));
                assert_eq!(
                    map.values.term(keys.value(d) as usize),
                    cat_term(g).as_bytes(),
                    "global key g{g}"
                );
            } else {
                assert!(keys.is_null(d));
            }
        }
    }
}
