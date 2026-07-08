// SPDX-License-Identifier: Apache-2.0

//! The zero-copy chunked lane must be byte-identical to the flat lane:
//! encode_numeric_field_dense_chunks(chunks) == encode_numeric_field(concat).

use lucene_arrow_docvalues::write::{
    CpuEncoder, encode_numeric_field_dense_chunks, encode_numeric_field_with,
};

fn check(values: Vec<i64>, splits: &[usize]) {
    let n = values.len() as u32;
    let docs: Vec<u32> = (0..n).collect();
    let flat = encode_numeric_field_with(&CpuEncoder, 7, &docs, &values, n, 123).unwrap();
    let mut chunks: Vec<&[i64]> = Vec::new();
    let mut at = 0;
    for &sp in splits {
        chunks.push(&values[at..at + sp]);
        at += sp;
    }
    chunks.push(&values[at..]);
    let chunked =
        encode_numeric_field_dense_chunks(&CpuEncoder, 7, &chunks, n, 123).unwrap();
    assert_eq!(flat.meta, chunked.meta, "dvm mismatch");
    assert_eq!(flat.data, chunked.data, "dvd mismatch");
}

#[test]
fn chunked_matches_flat_bytes() {
    // gcd-able values (multi-chunk boundaries off word alignment)
    check((0..10_000).map(|i| 1_000_000 + (i % 4096) * 25).collect(), &[1, 4095, 3000]);
    // full-width random (bpv=64 identity path, negative min)
    check(
        (0..5_000).map(|i| (i as i64).wrapping_mul(0x9E37_79B9_7F4A_7C15u64 as i64)).collect(),
        &[1234, 1234],
    );
    // small table (≤256 uniques)
    check((0..8_192).map(|i| (i % 7) * 1000 - 3000).collect(), &[100, 8000]);
    // constant
    check(vec![42; 3000], &[1500]);
    // single chunk degenerate
    check((0..100).map(|i| i * 3).collect(), &[]);
}
