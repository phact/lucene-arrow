// SPDX-License-Identifier: Apache-2.0

//! jVector `OnDiskGraphIndex` serialization (format v5 — what the shipped
//! jVector 4.0.0-beta reader accepts; repo main is at v6).
//!
//! Layout traced from `datastax/jvector` sources (`CommonHeader.write`,
//! `Header.write`, `OnDiskGraphIndexWriter.writeL0Records`,
//! `AbstractGraphIndexWriter.writeFooter`). Everything is **big-endian**
//! (Java `DataOutput`). v1 writes the `INLINE_VECTORS` feature only —
//! full-precision f32 vectors inline per node, exact scoring on read.
//!
//! We emit a single-layer graph (`numLayers = 1`, the Vamana/DiskANN
//! degenerate case of the hierarchical format) fed by the same
//! GPU-built adjacency as our Lucene `.vem`/`.vex` writer.

use lucene_arrow_core::{Error, Result};

pub const MAGIC: i32 = 0xFFFF0D61u32 as i32; // == -62111
pub const FOOTER_MAGIC: i32 = 0x4a564244; // "JVBD"
pub const VERSION: i32 = 5;
const V4_MAX_LAYERS: usize = 32;
const INLINE_VECTORS_BIT: i32 = 1 << 0; // FeatureId ordinal 0

fn put_i32(out: &mut Vec<u8>, v: i32) {
    out.extend_from_slice(&v.to_be_bytes());
}

/// Serialize one single-layer graph + inline f32 vectors as a complete
/// jVector `OnDiskGraphIndex` v5 file. `neighbors[i]` are node i's
/// out-edges (deduped/self-dropped here; order preserved — jVector keeps
/// score-descending order, and proximity order from NN-Descent matches).
pub fn write_index(
    vectors: &[f32],
    dim: usize,
    neighbors: &[Vec<u32>],
    entry_node: u32,
) -> Result<Vec<u8>> {
    let n = neighbors.len();
    if vectors.len() != n * dim {
        return Err(Error::invalid("vectors/neighbors size mismatch"));
    }
    let degree = neighbors.iter().map(|l| l.len()).max().unwrap_or(0).max(1);

    // ---- Header: CommonHeader v4 (288 B) + feature bitset (v3-5) ----
    let mut header = Vec::with_capacity(292);
    put_i32(&mut header, MAGIC);
    put_i32(&mut header, VERSION);
    put_i32(&mut header, n as i32); // layerInfo[0].size
    put_i32(&mut header, dim as i32);
    put_i32(&mut header, entry_node as i32);
    put_i32(&mut header, degree as i32); // layerInfo[0].degree
    put_i32(&mut header, n as i32); // idUpperBound (dense, no holes)
    put_i32(&mut header, 1); // numLayers
    put_i32(&mut header, n as i32); // layer table entry 0: size
    put_i32(&mut header, degree as i32); //                    degree
    for _ in 1..V4_MAX_LAYERS {
        put_i32(&mut header, 0);
        put_i32(&mut header, 0);
    }
    put_i32(&mut header, INLINE_VECTORS_BIT); // feature bitset; headerSize 0

    let mut out = header.clone();

    // ---- Layer-0 dense records ----
    // record = i32 nodeId + dim*f32 vector + i32 count + i32*degree edges
    for (node, nbrs) in neighbors.iter().enumerate() {
        put_i32(&mut out, node as i32);
        for &v in &vectors[node * dim..(node + 1) * dim] {
            out.extend_from_slice(&v.to_be_bytes());
        }
        let mut seen = std::collections::BTreeSet::new();
        let edges: Vec<u32> = nbrs
            .iter()
            .copied()
            .filter(|&x| (x as usize) < n && x as usize != node && seen.insert(x))
            .take(degree)
            .collect();
        put_i32(&mut out, edges.len() as i32);
        for &e in &edges {
            put_i32(&mut out, e as i32);
        }
        for _ in edges.len()..degree {
            put_i32(&mut out, -1);
        }
    }

    // ---- numLayers == 1: no sparse upper layers, no separated features ----

    // ---- Footer (v5): header copy + i64 headerOffset + FOOTER_MAGIC ----
    let header_offset = out.len() as i64;
    out.extend_from_slice(&header);
    out.extend_from_slice(&header_offset.to_be_bytes());
    put_i32(&mut out, FOOTER_MAGIC);
    Ok(out)
}
