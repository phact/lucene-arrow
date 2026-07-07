// SPDX-License-Identifier: Apache-2.0

//! Vector round-trip: encode → plan → CPU decode == original (SPEC §12.2).

use arrow_array::Array;
use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, Int8Type};

use lucene_arrow_cpu::decode_flat_vectors;
use lucene_arrow_vectors::file::VectorsFileBuilder;
use lucene_arrow_vectors::read::{VecField, plan_vectors};
use lucene_arrow_vectors::{Similarity, VectorEncoding};

const SEG_ID: [u8; 16] = *b"segmentid0123456";

fn f32_bytes(vectors: &[Vec<f32>]) -> Vec<u8> {
    vectors.iter().flatten().flat_map(|v| v.to_le_bytes()).collect()
}

fn round_trip_f32(per_doc: &[Option<Vec<f32>>], dim: u32) {
    let max_doc = per_doc.len() as u32;
    let docs: Vec<u32> = per_doc
        .iter()
        .enumerate()
        .filter_map(|(d, v)| v.as_ref().map(|_| d as u32))
        .collect();
    let vectors: Vec<Vec<f32>> = per_doc.iter().flatten().cloned().collect();

    let mut builder = VectorsFileBuilder::new(&SEG_ID, "");
    builder
        .add_field(0, VectorEncoding::Float32, Similarity::Euclidean, dim, &docs, &f32_bytes(&vectors), max_doc)
        .unwrap();
    let (vemf, vec) = builder.finish();

    let fields = [VecField { number: 0, name: "emb".into() }];
    let plans = plan_vectors(&vemf, &fields, max_doc, "_0.vec").unwrap();
    assert_eq!(plans.len(), 1);
    let vp = &plans[0];
    assert_eq!(vp.dim, dim);
    assert_eq!(vp.count, docs.len() as u64);
    assert_eq!(vp.similarity, Similarity::Euclidean);

    let array = decode_flat_vectors(&vp.plan, &vec).unwrap();
    let list = array.as_fixed_size_list();
    assert_eq!(list.len(), per_doc.len());
    let child = list.values().as_primitive::<Float32Type>();
    for (d, expected) in per_doc.iter().enumerate() {
        match expected {
            Some(v) => {
                assert!(list.is_valid(d), "doc {d}");
                let off = d * dim as usize;
                for (k, &expected) in v.iter().enumerate() {
                    assert_eq!(child.value(off + k), expected, "doc {d} dim {k}");
                }
            }
            None => assert!(list.is_null(d), "doc {d} should be null"),
        }
    }
}

fn vecf(seed: usize, dim: u32) -> Vec<f32> {
    (0..dim).map(|k| ((seed * 31 + k as usize * 7) % 1009) as f32 * 0.25 - 100.0).collect()
}

#[test]
fn dense_float_vectors() {
    let per_doc: Vec<Option<Vec<f32>>> = (0..500).map(|d| Some(vecf(d, 64))).collect();
    round_trip_f32(&per_doc, 64);
}

#[test]
fn sparse_float_vectors() {
    let per_doc: Vec<Option<Vec<f32>>> =
        (0..3000).map(|d| if d % 5 == 0 { Some(vecf(d, 32)) } else { None }).collect();
    round_trip_f32(&per_doc, 32);
}

#[test]
fn sparse_dense_disi_vectors() {
    // >4095 vectors within one 65536-doc block → DENSE DISI branch.
    let per_doc: Vec<Option<Vec<f32>>> =
        (0..20_000).map(|d| if d % 3 != 2 { Some(vecf(d, 8)) } else { None }).collect();
    round_trip_f32(&per_doc, 8);
}

#[test]
fn byte_vectors_round_trip() {
    let dim = 16u32;
    let max_doc = 400u32;
    let docs: Vec<u32> = (0..max_doc).filter(|d| d % 2 == 0).collect();
    let payload: Vec<u8> = docs
        .iter()
        .flat_map(|&d| (0..dim).map(move |k| ((d as i32 * 7 + k as i32 * 3) % 256 - 128) as i8 as u8))
        .collect();

    let mut builder = VectorsFileBuilder::new(&SEG_ID, "");
    builder
        .add_field(2, VectorEncoding::Byte, Similarity::DotProduct, dim, &docs, &payload, max_doc)
        .unwrap();
    let (vemf, vec) = builder.finish();

    let fields = [VecField { number: 2, name: "bytes".into() }];
    let plans = plan_vectors(&vemf, &fields, max_doc, "_0.vec").unwrap();
    let vp = &plans[0];
    assert_eq!(vp.encoding, VectorEncoding::Byte);

    let array = decode_flat_vectors(&vp.plan, &vec).unwrap();
    let list = array.as_fixed_size_list();
    let child = list.values().as_primitive::<Int8Type>();
    for d in 0..max_doc as usize {
        if d % 2 == 0 {
            assert!(list.is_valid(d));
            for k in 0..dim as usize {
                let expected = ((d as i32 * 7 + k as i32 * 3) % 256 - 128) as i8;
                assert_eq!(child.value(d * dim as usize + k), expected, "doc {d} dim {k}");
            }
        } else {
            assert!(list.is_null(d));
        }
    }
}

#[test]
fn empty_field_all_null() {
    let per_doc: Vec<Option<Vec<f32>>> = vec![None; 100];
    round_trip_f32(&per_doc, 4);
}
