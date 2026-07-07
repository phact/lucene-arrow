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
//! [`write_index`] emits a single-layer graph (`numLayers = 1`);
//! [`write_index_multi`] emits a real multi-layer hierarchy (header layer
//! table + dense level-0 records + sparse upper levels) from a parsed
//! HNSW hierarchy — near-native search quality.

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

/// Read the inline f32 vectors out of a jVector `OnDiskGraphIndex` v5 file
/// (big-endian), in ordinal order — the input for a GPU rebuild-merge.
/// Returns `(vectors_flat, dim)`. Reads only the dense level-0 records'
/// inline vector payloads; the graph edges are ignored (a merge rebuilds
/// the graph). Holes (`nodeId == -1`) are skipped.
pub fn read_vectors(bytes: &[u8]) -> Result<(Vec<f32>, usize)> {
    let i32be = |o: usize| -> Result<i32> {
        bytes
            .get(o..o + 4)
            .map(|b| i32::from_be_bytes(b.try_into().unwrap()))
            .ok_or_else(|| Error::corrupt("jvector: truncated"))
    };
    if i32be(0)? != MAGIC {
        return Err(Error::invalid("not a jVector index (bad magic)"));
    }
    let dim = i32be(12)? as usize; // header field 4
    let degree0 = i32be(20)? as usize; // layerInfo[0].degree
    let id_upper = i32be(24)? as usize; // idUpperBound
    // CommonHeader v4 (288) + feature bitset (4). Level-0 records follow.
    const HEADER: usize = 288 + 4;
    let record = 4 + dim * 4 + 4 + degree0 * 4; // nodeId + vector + count + edges
    let mut out = Vec::with_capacity(id_upper * dim);
    for node in 0..id_upper {
        let base = HEADER + node * record;
        if i32be(base)? < 0 {
            continue; // hole
        }
        let voff = base + 4;
        if voff + dim * 4 > bytes.len() {
            return Err(Error::corrupt("jvector: truncated record"));
        }
        for k in 0..dim {
            let o = voff + k * 4;
            out.push(f32::from_be_bytes(bytes[o..o + 4].try_into().unwrap()));
        }
    }
    Ok((out, dim))
}

/// Dedup/self-drop/cap a node's neighbor list, preserving order.
fn clean_edges(nbrs: &[u32], n: usize, node: usize, cap: usize) -> Vec<u32> {
    let mut seen = std::collections::BTreeSet::new();
    nbrs.iter()
        .copied()
        .filter(|&x| (x as usize) < n && x as usize != node && seen.insert(x))
        .take(cap)
        .collect()
}

/// Serialize a **multi-layer** HNSW hierarchy as a jVector `OnDiskGraphIndex`
/// v5 file: CommonHeader with the per-layer (size, degree) table, dense
/// level-0 records (nodeId + inline f32 vector + neighbors padded to the
/// level-0 degree), then sparse upper-level records (nodeId + neighbors
/// padded to that level's degree), then the footer. Layout traced from
/// `AbstractGraphIndexWriter.writeSparseLevels` + `CommonHeader.write`.
pub fn write_index_multi(
    vectors: &[f32],
    dim: usize,
    parsed: &crate::hnsw::HnswParsed,
) -> Result<Vec<u8>> {
    let n = parsed.count;
    if vectors.len() != n * dim {
        return Err(Error::invalid("vectors/graph size mismatch"));
    }
    if parsed.levels.is_empty() || parsed.levels[0].len() != n {
        return Err(Error::invalid("level 0 must hold all nodes"));
    }
    let num_layers = parsed.levels.len();
    // Per-level degree = max neighbor count on that level (records pad to it).
    let level_degrees: Vec<usize> = parsed
        .levels
        .iter()
        .map(|lvl| lvl.iter().map(|(_, nb)| nb.len()).max().unwrap_or(0).max(1))
        .collect();
    let degree0 = level_degrees[0];

    // Dense level-0 adjacency, indexed by node id.
    let mut l0 = vec![Vec::<u32>::new(); n];
    for (id, nb) in &parsed.levels[0] {
        l0[*id as usize] = nb.clone();
    }

    // ---- Header: CommonHeader v4 + feature bitset ----
    let mut header = Vec::new();
    put_i32(&mut header, MAGIC);
    put_i32(&mut header, VERSION);
    put_i32(&mut header, n as i32); // layerInfo[0].size
    put_i32(&mut header, dim as i32);
    put_i32(&mut header, parsed.entry as i32); // entryNode (top level)
    put_i32(&mut header, degree0 as i32); // layerInfo[0].degree
    put_i32(&mut header, n as i32); // idUpperBound
    put_i32(&mut header, num_layers as i32);
    for (lvl, deg) in parsed.levels.iter().zip(&level_degrees) {
        put_i32(&mut header, lvl.len() as i32);
        put_i32(&mut header, *deg as i32);
    }
    for _ in num_layers..V4_MAX_LAYERS {
        put_i32(&mut header, 0);
        put_i32(&mut header, 0);
    }
    put_i32(&mut header, INLINE_VECTORS_BIT);

    let mut out = header.clone();

    // ---- Level-0 dense records ----
    for node in 0..n {
        put_i32(&mut out, node as i32);
        for &v in &vectors[node * dim..(node + 1) * dim] {
            out.extend_from_slice(&v.to_be_bytes());
        }
        let edges = clean_edges(&l0[node], n, node, degree0);
        put_i32(&mut out, edges.len() as i32);
        for &e in &edges {
            put_i32(&mut out, e as i32);
        }
        for _ in edges.len()..degree0 {
            put_i32(&mut out, -1);
        }
    }

    // ---- Sparse upper levels ----
    for (lvl, &deg) in parsed.levels.iter().zip(&level_degrees).skip(1) {
        let mut nodes: Vec<&(u32, Vec<u32>)> = lvl.iter().collect();
        nodes.sort_by_key(|(id, _)| *id);
        for (node, nb) in nodes {
            put_i32(&mut out, *node as i32);
            let edges = clean_edges(nb, n, *node as usize, deg);
            put_i32(&mut out, edges.len() as i32);
            for &e in &edges {
                put_i32(&mut out, e as i32);
            }
            for _ in edges.len()..deg {
                put_i32(&mut out, -1);
            }
        }
    }

    // ---- Footer ----
    let header_offset = out.len() as i64;
    out.extend_from_slice(&header);
    out.extend_from_slice(&header_offset.to_be_bytes());
    put_i32(&mut out, FOOTER_MAGIC);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vectors_round_trip_through_write_then_read() {
        let (n, dim) = (300usize, 16usize);
        let vecs: Vec<f32> = (0..n * dim).map(|i| (i as f32) * 0.5 - 7.0).collect();
        let neighbors: Vec<Vec<u32>> =
            (0..n).map(|i| vec![((i + 1) % n) as u32, ((i + n - 1) % n) as u32]).collect();
        let bytes = write_index(&vecs, dim, &neighbors, 0).unwrap();
        let (read, d) = read_vectors(&bytes).unwrap();
        assert_eq!(d, dim);
        assert_eq!(read, vecs, "inline vectors must round-trip exactly");
    }
}
