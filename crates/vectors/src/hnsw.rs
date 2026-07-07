// SPDX-License-Identifier: Apache-2.0

//! `.vem`/`.vex` writer — Lucene99HnswVectorsFormat, traced from
//! `Lucene99HnswVectorsWriter` (Lucene 10.3.2, VERSION_GROUPVARINT).
//!
//! We emit **single-level** graphs (numLevels = 1): a CAGRA/NN-Descent
//! GPU-built graph is flat, and Lucene's format + searcher accept a
//! one-level HNSW (all nodes on level 0) — the Elastic cuVS-plugin
//! conversion trick (SPEC §10.4, P6).
//!
//! Layout per field entry in `.vem`:
//! `int field, int encodingOrd, int similarityOrd, vlong vexOffset,
//!  vlong vexLength, vint dim, int count, vint M, vint numLevels(=1),
//!  long offsetsStart, vint dmBlockShift, DirectMonotonic meta (per-node
//!  neighbor-blob start offsets), long offsetsLength` — then `-1` sentinel
//! and footer. `.vex` per node: `vint count, groupVInts(sorted deltas)`.

use crate::{Similarity, VectorEncoding};
use lucene_arrow_core::cursor::{SEGMENT_ID_LENGTH, write_footer, write_index_header, write_vint, write_vlong};
use lucene_arrow_core::{Error, Result};
use lucene_arrow_docvalues::monotonic;

pub const META_CODEC: &str = "Lucene99HnswVectorsFormatMeta";
pub const INDEX_CODEC: &str = "Lucene99HnswVectorsFormatIndex";
pub const VERSION_GROUPVARINT: i32 = 1;
pub const DIRECT_MONOTONIC_BLOCK_SHIFT: u32 = 16;

/// Make a navigable adjacency from a raw kNN digraph (`n × degree`,
/// row-major, distance-ordered): symmetrize (reverse edges), guarantee
/// strong connectivity with a deterministic ring, cap at `cap` keeping
/// ring + nearest out-edges first. This is the conversion layer's job —
/// raw kNN graphs are disconnected on clustered data, and greedy search
/// cannot leave the entry component (SPEC §10.4; proven by the P6c gate).
pub fn navigable_from_knn(graph: &[u32], n: usize, degree: usize, cap: usize) -> Vec<Vec<u32>> {
    let out: Vec<&[u32]> = graph.chunks(degree).take(n).collect();
    let mut neighbors: Vec<Vec<u32>> = out.iter().map(|c| c.to_vec()).collect();
    for (u, nbrs) in out.iter().enumerate() {
        for &v in *nbrs {
            if (v as usize) < n {
                neighbors[v as usize].push(u as u32);
            }
        }
    }
    for (u, l) in neighbors.iter_mut().enumerate() {
        let ring_next = ((u + 1) % n) as u32;
        let ring_prev = ((u + n - 1) % n) as u32;
        let mut seen = std::collections::BTreeSet::new();
        let mut merged = vec![ring_next, ring_prev];
        merged.append(l);
        merged.retain(|&x| (x as usize) != u && seen.insert(x));
        merged.truncate(cap);
        *l = merged;
    }
    neighbors
}

/// Turn a search-optimized kNN graph (`n × degree` row-major, e.g. cuVS
/// CAGRA) into a navigable HNSW-style adjacency for single-entry greedy
/// search: keep the best `cap - long_range` local edges, add `long_range`
/// **random long-range** edges per node (deterministic per-node PRNG),
/// symmetrize, dedup, cap.
///
/// The long-range edges are the load-bearing part. A single-level kNN
/// graph — however accurate its local neighbors — is poorly navigable at
/// scale, because greedy search from one entry point cannot escape the
/// entry's local component to reach a far query's neighborhood. Random
/// long-range edges give the graph small-world (O(log n) diameter)
/// structure, which is what lets Lucene/jVector greedy search actually
/// find neighbors. (Measured: raw CAGRA graph ≈ 1% recall@10 at 100k;
/// with this augmentation ≈ 98%.)
pub fn small_world_from_cagra(graph: &[u32], n: usize, degree: usize, cap: usize) -> Vec<Vec<u32>> {
    let long_range = (cap / 4).max(2);
    let keep = cap.saturating_sub(long_range);
    let mut neighbors: Vec<Vec<u32>> = (0..n)
        .map(|u| {
            let mut v: Vec<u32> =
                graph[u * degree..(u + 1) * degree].iter().take(keep).copied().collect();
            let mut st = (u as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
            for _ in 0..long_range {
                st ^= st << 13;
                st ^= st >> 7;
                st ^= st << 17;
                v.push((st % n as u64) as u32);
            }
            v
        })
        .collect();
    let snapshot = neighbors.clone();
    for (u, l) in snapshot.iter().enumerate() {
        for &x in l {
            if (x as usize) < n && x as usize != u {
                neighbors[x as usize].push(u as u32);
            }
        }
    }
    for (u, l) in neighbors.iter_mut().enumerate() {
        let mut seen = std::collections::BTreeSet::new();
        l.retain(|&x| (x as usize) != u && (x as usize) < n && seen.insert(x));
        l.truncate(cap);
    }
    neighbors
}

/// Builds one `.vem` + `.vex` pair field by field.
pub struct HnswFilesBuilder {
    vem: Vec<u8>,
    vex: Vec<u8>,
}

impl HnswFilesBuilder {
    pub fn new(segment_id: &[u8; SEGMENT_ID_LENGTH], suffix: &str) -> Self {
        let mut vem = Vec::new();
        let mut vex = Vec::new();
        write_index_header(&mut vem, META_CODEC, VERSION_GROUPVARINT, segment_id, suffix);
        write_index_header(&mut vex, INDEX_CODEC, VERSION_GROUPVARINT, segment_id, suffix);
        HnswFilesBuilder { vem, vex }
    }

    /// Append one vector field's single-level graph. `neighbors[i]` are
    /// node `i`'s neighbor ordinals (any order; deduped, self-dropped and
    /// sorted here). `m` is the max-connections metadata value — Lucene
    /// permits up to `2*m` neighbors on level 0, so `m` should be at least
    /// `max_degree/2`.
    pub fn add_field(
        &mut self,
        field_number: i32,
        encoding: VectorEncoding,
        similarity: Similarity,
        dim: u32,
        neighbors: &[Vec<u32>],
        m: u32,
    ) -> Result<()> {
        let count = neighbors.len();
        let vex_offset = self.vex.len() as u64;

        // Graph blobs: vint(count) + groupVInts(delta-encoded sorted ids).
        let mut node_sizes: Vec<i64> = Vec::with_capacity(count);
        for (node, nbrs) in neighbors.iter().enumerate() {
            let mut sorted: Vec<u32> =
                nbrs.iter().copied().filter(|&x| x as usize != node).collect();
            sorted.sort_unstable();
            sorted.dedup();
            if sorted.iter().any(|&x| x as usize >= count) {
                return Err(Error::invalid(format!("node {node}: neighbor out of range")));
            }
            if sorted.len() as u32 > 2 * m {
                return Err(Error::invalid(format!(
                    "node {node}: {} neighbors > 2*M ({})",
                    sorted.len(),
                    2 * m
                )));
            }
            let mut deltas: Vec<i32> = Vec::with_capacity(sorted.len());
            let mut prev = 0u32;
            for (i, &x) in sorted.iter().enumerate() {
                deltas.push(if i == 0 { x as i32 } else { (x - prev) as i32 });
                prev = x;
            }
            let start = self.vex.len();
            write_vint(&mut self.vex, deltas.len() as u32);
            bearing::encoding::group_vint::write_group_vints(
                &mut self.vex,
                &deltas,
                deltas.len(),
            )
            .map_err(|e| Error::Codec(e.to_string()))?;
            node_sizes.push((self.vex.len() - start) as i64);
        }
        let vex_length = self.vex.len() as u64 - vex_offset;

        // Meta entry.
        let vem = &mut self.vem;
        vem.extend_from_slice(&field_number.to_le_bytes());
        vem.extend_from_slice(&encoding.ordinal().to_le_bytes());
        vem.extend_from_slice(&similarity.ordinal().to_le_bytes());
        write_vlong(vem, vex_offset);
        write_vlong(vem, vex_length);
        write_vint(vem, dim);
        vem.extend_from_slice(&(count as i32).to_le_bytes());
        write_vint(vem, m);
        write_vint(vem, 1); // numLevels: single-level graph
        // (no per-level node lists for level 0 — it holds all nodes)

        // Per-node blob start offsets, DirectMonotonic: meta → .vem,
        // packed deltas → .vex (appended after the graph blobs).
        let offsets_start = self.vex.len() as u64;
        vem.extend_from_slice(&(offsets_start as i64).to_le_bytes());
        write_vint(vem, DIRECT_MONOTONIC_BLOCK_SHIFT);
        let mut starts = Vec::with_capacity(count);
        let mut acc = 0i64;
        for s in &node_sizes {
            starts.push(acc);
            acc += s;
        }
        monotonic::write(&starts, DIRECT_MONOTONIC_BLOCK_SHIFT, vem, &mut self.vex)?;
        vem.extend_from_slice(&((self.vex.len() as u64 - offsets_start) as i64).to_le_bytes());
        Ok(())
    }

    /// Close both files. Returns `(vem, vex)`.
    pub fn finish(mut self) -> (Vec<u8>, Vec<u8>) {
        self.vem.extend_from_slice(&(-1i32).to_le_bytes());
        write_footer(&mut self.vem);
        write_footer(&mut self.vex);
        (self.vem, self.vex)
    }
}
