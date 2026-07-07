// SPDX-License-Identifier: Apache-2.0

//! `.vem`/`.vex` writer — Lucene99HnswVectorsFormat, traced from
//! `Lucene99HnswVectorsWriter` (Lucene 10.3.2, VERSION_GROUPVARINT).
//!
//! Supports single-level (`add_field`) and **multi-level**
//! (`add_field_multi`) HNSW graphs. Single-level is the Elastic
//! cuVS-plugin trick (all nodes on level 0); multi-level is fed a real
//! HNSW hierarchy (via `parse_hnswlib` of a cuVS `cuvsHnswFromCagra`
//! index) and is ~2× faster to search at near-native recall (SPEC §10.4).
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
    // A small fixed number of long-range shortcuts (cap/4) gives the graph
    // small-world connectivity. NOTE: measured — scaling this *up* with n
    // hurts recall, because the shortcuts trade against local edges within
    // the fixed degree cap, and local edges are what let greedy search pin
    // down the exact nearest. The levers for recall at scale are the search
    // beam (ef) and the degree cap, not the shortcut count.
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

/// A multi-level HNSW graph parsed from a standard hnswlib index file
/// (as produced by `cuvsHnswFromCagra` with hierarchy=CPU). `levels[l]` is
/// `(ordinal, neighbor_ordinals)` for the nodes present on level `l`;
/// `levels[0]` covers all `count` nodes. Feed straight to
/// [`HnswFilesBuilder::add_field_multi`].
pub struct HnswParsed {
    pub count: usize,
    pub m: u32,
    pub levels: Vec<Vec<(u32, Vec<u32>)>>,
}

/// Parse a standard hnswlib `saveIndex` byte image into a multi-level
/// graph. Layout (little-endian): a 96-byte header, then a level-0 block
/// per element (`[u16 count | .. ] + maxM0 u32 links + vector data + u64
/// label`), then per-element upper-level link lists (`u32 byteLen` then
/// `element_levels` blocks of `[u16 count |..] + maxM u32 links`). Internal
/// ids are remapped through each element's stored label.
pub fn parse_hnswlib(b: &[u8]) -> Result<HnswParsed> {
    let need = |o: usize, n: usize| -> Result<()> {
        if o + n > b.len() { Err(Error::corrupt("hnswlib: truncated")) } else { Ok(()) }
    };
    let u64at = |o: usize| u64::from_le_bytes(b[o..o + 8].try_into().unwrap()) as usize;
    let i32at = |o: usize| i32::from_le_bytes(b[o..o + 4].try_into().unwrap());
    let u32at = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
    let u16at = |o: usize| u16::from_le_bytes(b[o..o + 2].try_into().unwrap()) as usize;
    need(96, 0)?;

    let count = u64at(16); // cur_element_count
    let size_per = u64at(24);
    let label_offset = u64at(32);
    let max_level = i32at(48).max(0) as usize;
    let max_m = u64at(56);
    let max_m0 = u64at(64);
    let m = u32at(72);
    const HDR: usize = 96;

    // Level 0 + labels.
    let mut label = vec![0u32; count];
    let mut l0: Vec<Vec<u32>> = Vec::with_capacity(count);
    #[allow(clippy::needless_range_loop)]
    for i in 0..count {
        let e = HDR + i * size_per;
        need(e + size_per, 0)?;
        label[i] = u64at(e + label_offset) as u32;
        let cnt = u16at(e).min(max_m0);
        l0.push((0..cnt).map(|k| u32at(e + 4 + k * 4)).collect());
    }

    // Upper-level link lists.
    let size_links = max_m * 4 + 4;
    let mut pos = HDR + count * size_per;
    let mut elevels = vec![0usize; count];
    let mut upper: Vec<Vec<Vec<u32>>> = vec![Vec::new(); count];
    for i in 0..count {
        need(pos + 4, 0)?;
        let bytes = u32at(pos) as usize;
        pos += 4;
        if bytes > 0 {
            need(pos + bytes, 0)?;
            let n_lv = bytes / size_links;
            elevels[i] = n_lv;
            for lvl in 0..n_lv {
                let blk = pos + lvl * size_links;
                let cnt = u16at(blk).min(max_m);
                upper[i].push((0..cnt).map(|k| u32at(blk + 4 + k * 4)).collect());
            }
            pos += bytes;
        }
    }

    // Remap internal ids → labels; assemble levels.
    let remap = |ids: &[u32]| -> Vec<u32> { ids.iter().map(|&j| label[j as usize]).collect() };
    let mut levels: Vec<Vec<(u32, Vec<u32>)>> = vec![Vec::new(); max_level + 1];
    for i in 0..count {
        levels[0].push((label[i], remap(&l0[i])));
        for lvl in 1..=elevels[i] {
            levels[lvl].push((label[i], remap(&upper[i][lvl - 1])));
        }
    }
    Ok(HnswParsed { count, m, levels })
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

    /// Append one vector field's **single-level** graph. `neighbors[i]` are
    /// node `i`'s neighbor ordinals (any order; deduped, self-dropped, and
    /// capped to `2*m` here). Convenience wrapper over [`add_field_multi`].
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
        let level0: Vec<(u32, Vec<u32>)> =
            neighbors.iter().enumerate().map(|(i, nb)| (i as u32, nb.clone())).collect();
        self.add_field_multi(field_number, encoding, similarity, dim, count, m, &[level0])
    }

    /// Append one vector field's **multi-level** HNSW graph (SPEC §10.4).
    /// `levels[l]` is the list of `(node_id, neighbor_ids)` present on level
    /// `l`; `levels[0]` must cover all `count` nodes. Neighbor ids are global
    /// ordinals on every level (Lucene stores them that way). Level 0 keeps
    /// up to `2*m` neighbors, upper levels up to `m`; each level's nodes are
    /// sorted here. Layout traced from `Lucene99HnswVectorsWriter`
    /// (writeGraph/writeMeta): `.vex` = per-level, per-sorted-node
    /// `vint(count)+groupVInts(sorted deltas)` blobs; `.vem` = numLevels,
    /// then delta-encoded node lists for levels > 0, then a DirectMonotonic
    /// over every blob's start offset (level 0 first).
    #[allow(clippy::too_many_arguments)]
    pub fn add_field_multi(
        &mut self,
        field_number: i32,
        encoding: VectorEncoding,
        similarity: Similarity,
        dim: u32,
        count: usize,
        m: u32,
        levels: &[Vec<(u32, Vec<u32>)>],
    ) -> Result<()> {
        if levels.is_empty() {
            return Err(Error::invalid("hnsw graph needs at least level 0"));
        }
        let vex_offset = self.vex.len() as u64;

        // .vex: blobs in write order — level 0 (sorted), then level 1, ...
        let mut all_sizes: Vec<i64> = Vec::new();
        for (l, level) in levels.iter().enumerate() {
            let max = if l == 0 { (2 * m) as usize } else { m as usize };
            let mut order: Vec<usize> = (0..level.len()).collect();
            order.sort_by_key(|&i| level[i].0);
            for &i in &order {
                let (node, nbrs) = &level[i];
                all_sizes.push(self.write_neighbor_blob(*node, nbrs, count, max)?);
            }
        }
        let vex_length = self.vex.len() as u64 - vex_offset;

        // .vem entry.
        let vem = &mut self.vem;
        vem.extend_from_slice(&field_number.to_le_bytes());
        vem.extend_from_slice(&encoding.ordinal().to_le_bytes());
        vem.extend_from_slice(&similarity.ordinal().to_le_bytes());
        write_vlong(vem, vex_offset);
        write_vlong(vem, vex_length);
        write_vint(vem, dim);
        vem.extend_from_slice(&(count as i32).to_le_bytes());
        write_vint(vem, m);
        write_vint(vem, levels.len() as u32);
        // Per-level node lists for levels > 0 (level 0 holds all nodes):
        // first id absolute, rest delta-encoded.
        for level in &levels[1..] {
            let mut ids: Vec<u32> = level.iter().map(|(id, _)| *id).collect();
            ids.sort_unstable();
            write_vint(vem, ids.len() as u32);
            let mut prev = 0u32;
            for (i, &id) in ids.iter().enumerate() {
                write_vint(vem, if i == 0 { id } else { id - prev });
                prev = id;
            }
        }

        // DirectMonotonic of every blob's cumulative start offset.
        let offsets_start = self.vex.len() as u64;
        vem.extend_from_slice(&(offsets_start as i64).to_le_bytes());
        write_vint(vem, DIRECT_MONOTONIC_BLOCK_SHIFT);
        let mut starts = Vec::with_capacity(all_sizes.len());
        let mut acc = 0i64;
        for s in &all_sizes {
            starts.push(acc);
            acc += s;
        }
        monotonic::write(&starts, DIRECT_MONOTONIC_BLOCK_SHIFT, vem, &mut self.vex)?;
        vem.extend_from_slice(&((self.vex.len() as u64 - offsets_start) as i64).to_le_bytes());
        Ok(())
    }

    /// Write one node's neighbor blob to `.vex`; returns its byte size.
    fn write_neighbor_blob(
        &mut self,
        node: u32,
        nbrs: &[u32],
        count: usize,
        max: usize,
    ) -> Result<i64> {
        let mut sorted: Vec<u32> = nbrs.iter().copied().filter(|&x| x != node).collect();
        sorted.sort_unstable();
        sorted.dedup();
        if sorted.iter().any(|&x| x as usize >= count) {
            return Err(Error::invalid(format!("node {node}: neighbor out of range")));
        }
        sorted.truncate(max);
        let mut deltas: Vec<i32> = Vec::with_capacity(sorted.len());
        let mut prev = 0u32;
        for (i, &x) in sorted.iter().enumerate() {
            deltas.push(if i == 0 { x as i32 } else { (x - prev) as i32 });
            prev = x;
        }
        let start = self.vex.len();
        write_vint(&mut self.vex, deltas.len() as u32);
        bearing::encoding::group_vint::write_group_vints(&mut self.vex, &deltas, deltas.len())
            .map_err(|e| Error::Codec(e.to_string()))?;
        Ok((self.vex.len() - start) as i64)
    }

    /// Close both files. Returns `(vem, vex)`.
    pub fn finish(mut self) -> (Vec<u8>, Vec<u8>) {
        self.vem.extend_from_slice(&(-1i32).to_le_bytes());
        write_footer(&mut self.vem);
        write_footer(&mut self.vex);
        (self.vem, self.vex)
    }
}
