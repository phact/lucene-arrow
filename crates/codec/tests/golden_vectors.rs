// SPDX-License-Identifier: Apache-2.0

//! Golden vector test: decode flat vectors written by real Java Lucene
//! 10.3.2 (`KnnFloatVectorField`/`KnnByteVectorField` under the default
//! HNSW format — whose *flat* storage delegate writes the `.vemf`/`.vec`
//! we read; the `.vem`/`.vex` graph is ignored by design, SPEC §7.7).
//! Values follow the formula documented in `GenerateGolden.writeVectors`.

use arrow_array::Array;
use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, Int8Type};

use lucene_arrow_codec::SegmentDirectory;
use lucene_arrow_cpu::decode_flat_vectors;
use lucene_arrow_vectors::read::{VecField, plan_vectors};
use lucene_arrow_vectors::{Similarity, VectorEncoding};

#[test]
fn golden_java_flat_vectors() {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../harness/golden");
    if !root.join("vectors").is_dir() {
        eprintln!("skipping: harness/golden not generated (needs JDK 21)");
        return;
    }
    let dir = SegmentDirectory::open(root.join("vectors")).unwrap();
    let seg = &dir.segments()[0];
    let max_doc = seg.max_doc as u32;
    assert_eq!(max_doc, 3000);

    let fields: Vec<VecField> = seg
        .fields
        .iter()
        .filter(|f| f.has_vectors)
        .map(|f| VecField { number: f.number as i32, name: f.name.clone() })
        .collect();
    assert_eq!(fields.len(), 2, "emb + bytes");

    let vemf_name = seg.files.iter().find(|f| f.ends_with(".vemf")).unwrap();
    let vec_name = seg.files.iter().find(|f| f.ends_with(".vec")).unwrap();
    let vemf_r = dir.open_input(&seg.name, vemf_name).unwrap();
    let vec_r = dir.open_input(&seg.name, vec_name).unwrap();
    let vemf = vemf_r.slice(0, vemf_r.len()).unwrap();
    let vec = vec_r.slice(0, vec_r.len()).unwrap();

    let plans = plan_vectors(vemf, &fields, max_doc, vec_name).unwrap();
    assert_eq!(plans.len(), 2);

    // --- emb: float32, dim 64, sparse (doc % 5 != 1), EUCLIDEAN ---
    let emb = plans.iter().find(|p| p.plan.column.name == "emb").unwrap();
    assert_eq!(emb.encoding, VectorEncoding::Float32);
    assert_eq!(emb.similarity, Similarity::Euclidean);
    assert_eq!(emb.dim, 64);
    assert_eq!(emb.count, (0..3000).filter(|d| d % 5 != 1).count() as u64);

    let array = decode_flat_vectors(&emb.plan, vec).unwrap();
    let list = array.as_fixed_size_list();
    assert_eq!(list.len(), 3000);
    let child = list.values().as_primitive::<Float32Type>();
    for d in 0..3000usize {
        if d % 5 != 1 {
            assert!(list.is_valid(d), "doc {d}");
            for k in 0..64usize {
                let expected = ((d * 31 + k * 7) % 1009) as f32 * 0.25 - 100.0;
                assert_eq!(child.value(d * 64 + k), expected, "emb doc {d} dim {k}");
            }
        } else {
            assert!(list.is_null(d), "doc {d} should have no vector");
        }
    }

    // --- bytes: int8, dim 16, dense, DOT_PRODUCT ---
    let bytes = plans.iter().find(|p| p.plan.column.name == "bytes").unwrap();
    assert_eq!(bytes.encoding, VectorEncoding::Byte);
    assert_eq!(bytes.similarity, Similarity::DotProduct);
    assert_eq!(bytes.dim, 16);
    assert_eq!(bytes.count, 3000);

    let array = decode_flat_vectors(&bytes.plan, vec).unwrap();
    let list = array.as_fixed_size_list();
    assert!(list.nulls().is_none(), "dense field has no validity buffer");
    let child = list.values().as_primitive::<Int8Type>();
    for d in 0..3000usize {
        for k in 0..16usize {
            let expected = ((d * 7 + k * 3) % 256) as i32 - 128;
            assert_eq!(child.value(d * 16 + k) as i32, expected, "bytes doc {d} dim {k}");
        }
    }
}
