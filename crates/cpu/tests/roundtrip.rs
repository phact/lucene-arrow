// SPDX-License-Identifier: Apache-2.0

//! P1 round-trip: encode → plan → CPU decode == original (SPEC §12.2).

use arrow_array::Array;
use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;

use lucene_arrow_cpu::decode_numeric;
use lucene_arrow_docvalues::file::DocValuesFileBuilder;
use lucene_arrow_docvalues::read::{DvField, DvKind, plan_doc_values};
use lucene_arrow_docvalues::{consts, direct};
use lucene_arrow_core::cursor::{write_footer, write_index_header};
use lucene_arrow_core::plan::{BlockDecode, Coverage};

const SEG_ID: [u8; 16] = *b"segmentid0123456";

fn field(number: i32) -> DvField {
    DvField { number, name: format!("f{number}"), kind: DvKind::Numeric, has_skip_index: false }
}

/// Encode one numeric field, plan it back, decode it, compare per-doc.
fn round_trip(per_doc: &[Option<i64>]) {
    let max_doc = per_doc.len() as u32;
    let (docs, values): (Vec<u32>, Vec<i64>) = per_doc
        .iter()
        .enumerate()
        .filter_map(|(d, v)| v.map(|v| (d as u32, v)))
        .unzip();

    let mut builder = DocValuesFileBuilder::new(&SEG_ID, "");
    builder.add_numeric(0, &docs, &values, max_doc).unwrap();
    let (dvm, dvd) = builder.finish();

    let plans = plan_doc_values(&dvm, &dvd, &[field(0)], max_doc, "_0.dvd").unwrap();
    assert_eq!(plans.plans.len(), 1, "one numeric field, one plan");
    assert!(plans.skipped.is_empty());

    let array = decode_numeric(&plans.plans[0], &dvd).unwrap();
    assert_eq!(array.len(), per_doc.len());
    let ints = array.as_primitive::<Int64Type>();
    for (d, expected) in per_doc.iter().enumerate() {
        match expected {
            Some(v) => {
                assert!(ints.is_valid(d), "doc {d} should have a value");
                assert_eq!(ints.value(d), *v, "doc {d}");
            }
            None => assert!(ints.is_null(d), "doc {d} should be null"),
        }
    }
}

#[test]
fn dense_delta_values() {
    round_trip(&(0..1000).map(|i| Some(1_000_000 + i * 7 + (i % 13))).collect::<Vec<_>>());
}

#[test]
fn dense_gcd_values() {
    // Multiples of 400 with a large base: GCD encoding kicks in.
    round_trip(&(0..500).map(|i| Some(1_700_000_000 + i * 400)).collect::<Vec<_>>());
}

#[test]
fn dense_table_values() {
    // 3 distinct wide-spread values: table beats delta.
    let vals = [i64::MIN / 4, 0, i64::MAX / 4];
    round_trip(&(0..300).map(|i| Some(vals[i % 3])).collect::<Vec<_>>());
}

#[test]
fn dense_constant_values() {
    round_trip(&vec![Some(42); 128]);
}

#[test]
fn dense_negative_and_wide_values() {
    round_trip(
        &(0..777)
            .map(|i| Some((i as i64 - 388) * 0x0123_4567_89AB + 5))
            .collect::<Vec<_>>(),
    );
}

#[test]
fn sparse_values_use_validity() {
    // Every 7th doc has a value; 3000 docs → one SPARSE DISI block.
    round_trip(
        &(0..3000)
            .map(|d| if d % 7 == 0 { Some(d as i64 * 11) } else { None })
            .collect::<Vec<_>>(),
    );
}

#[test]
fn sparse_dense_disi_block() {
    // >4095 of 65536+ docs carry values → DENSE DISI block with rank table.
    round_trip(
        &(0..70_000)
            .map(|d| if d % 3 != 2 { Some(d as i64) } else { None })
            .collect::<Vec<_>>(),
    );
}

#[test]
fn single_value_column() {
    round_trip(&[Some(-9)]);
}

#[test]
fn all_docs_one_distinct_value_sparse() {
    let mut per_doc = vec![None; 100];
    per_doc[17] = Some(3);
    per_doc[80] = Some(3);
    round_trip(&per_doc);
}

/// Java Lucene emits multi-block ("blockwise") numeric encoding for large
/// jumpy fields; our writer never does, so hand-craft the layout (per
/// `Lucene90DocValuesConsumer.writeValuesMultipleBlocks`) and decode it.
#[test]
fn multi_block_layout_decodes() {
    let block_size = consts::NUMERIC_BLOCK_SIZE;
    let num_docs = (block_size + block_size / 2) as u32; // 1.5 blocks
    let gcd = 10i64;

    // Block 0: base 1000, varied deltas. Block 1: constant 777.
    let block0: Vec<i64> = (0..block_size as i64).map(|i| 1000 + gcd * (i % 500)).collect();
    let block1 = vec![777i64; block_size / 2];

    let mut dvd = Vec::new();
    write_index_header(&mut dvd, consts::DATA_CODEC, consts::VERSION, &SEG_ID, "");
    let values_offset = dvd.len() as u64;

    // Block 0: bpv, blockMin (LE), bufferSize (LE), packed deltas.
    let deltas: Vec<i64> = block0.iter().map(|v| (v - 1000) / gcd).collect();
    let bpv = direct::unsigned_bits_required(499);
    let mut packed = Vec::new();
    direct::pack(&deltas, bpv, &mut packed);
    dvd.push(bpv);
    dvd.extend_from_slice(&1000i64.to_le_bytes());
    dvd.extend_from_slice(&(packed.len() as i32).to_le_bytes());
    let block0_payload = dvd.len() as u64;
    dvd.extend_from_slice(&packed);
    // Block 1: constant.
    dvd.push(0);
    dvd.extend_from_slice(&777i64.to_le_bytes());
    let values_length = dvd.len() as u64 - values_offset;
    write_footer(&mut dvd);

    // Metadata entry.
    let mut dvm = Vec::new();
    write_index_header(&mut dvm, consts::META_CODEC, consts::VERSION, &SEG_ID, "");
    dvm.extend_from_slice(&0i32.to_le_bytes()); // field number
    dvm.push(consts::TYPE_NUMERIC);
    dvm.extend_from_slice(&(-1i64).to_le_bytes()); // all docs have values
    dvm.extend_from_slice(&0i64.to_le_bytes());
    dvm.extend_from_slice(&(-1i16).to_le_bytes());
    dvm.push(0xFF);
    dvm.extend_from_slice(&(num_docs as i64).to_le_bytes()); // numValues
    dvm.extend_from_slice(&consts::MULTI_BLOCK_TABLE_SIZE.to_le_bytes());
    dvm.push(consts::MULTI_BLOCK_BPV);
    dvm.extend_from_slice(&0i64.to_le_bytes()); // min (unused in multi-block)
    dvm.extend_from_slice(&gcd.to_le_bytes());
    dvm.extend_from_slice(&(values_offset as i64).to_le_bytes());
    dvm.extend_from_slice(&(values_length as i64).to_le_bytes());
    dvm.extend_from_slice(&((values_offset + values_length) as i64).to_le_bytes()); // jumpTableOffset
    dvm.extend_from_slice(&(-1i32).to_le_bytes()); // sentinel
    write_footer(&mut dvm);

    let plans = plan_doc_values(&dvm, &dvd, &[field(0)], num_docs, "_0.dvd").unwrap();
    let plan = &plans.plans[0];
    assert_eq!(plan.blocks.len(), 2);
    assert!(matches!(plan.coverage, Coverage::Dense { .. }));
    assert!(
        matches!(plan.blocks[0], BlockDecode::GcdPacked { offset, gcd: g, values, .. }
            if offset == block0_payload && g == gcd && values == block_size as u64)
    );
    assert!(matches!(plan.blocks[1], BlockDecode::DeltaPacked { bit_width: 0, base: 777, .. }));

    let array = decode_numeric(plan, &dvd).unwrap();
    let ints = array.as_primitive::<Int64Type>();
    let expected: Vec<i64> = block0.iter().chain(block1.iter()).copied().collect();
    assert_eq!(ints.values().as_ref(), expected.as_slice());
}

/// Two fields in one file pair, planner walks both.
#[test]
fn two_fields_one_file() {
    let mut builder = DocValuesFileBuilder::new(&SEG_ID, "");
    builder.add_numeric(0, &[0, 1, 2], &[5, 6, 7], 3).unwrap();
    builder.add_numeric(2, &[1], &[99], 3).unwrap();
    let (dvm, dvd) = builder.finish();

    let fields = [field(0), field(2)];
    let plans = plan_doc_values(&dvm, &dvd, &fields, 3, "_0.dvd").unwrap();
    assert_eq!(plans.plans.len(), 2);
    assert!(matches!(plans.plans[0].coverage, Coverage::Dense { .. }));
    assert!(matches!(plans.plans[1].coverage, Coverage::Sparse { .. }));

    let a0 = decode_numeric(&plans.plans[0], &dvd).unwrap();
    assert_eq!(a0.as_primitive::<Int64Type>().values().as_ref(), &[5i64, 6, 7]);
    let a1 = decode_numeric(&plans.plans[1], &dvd).unwrap();
    assert!(a1.as_primitive::<Int64Type>().is_null(0));
    assert_eq!(a1.as_primitive::<Int64Type>().value(1), 99);
    assert!(a1.as_primitive::<Int64Type>().is_null(2));
}
