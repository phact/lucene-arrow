// SPDX-License-Identifier: Apache-2.0

//! OrdinalMap: one coherent dictionary across a stream's segments
//! (`lucene.dict = global`, SPEC §7.3, §11.5).
//!
//! Per-segment dictionaries are byte-sorted, so the global dictionary is a
//! k-way merge with duplicate collapse; each segment gets an `i32` remap
//! table (`local ord → global ord`). Cost is proportional to total
//! dictionary size — which is exactly the §7.3 gate: global for small
//! dicts, per-segment replacement for large ones (see the ordmap bench).
//!
//! Applying a remap on decode is a table gather — the same fused epilogue
//! the executors already have (`BlockDecode::Table`), so the GPU path
//! needs no new kernel (SPEC §11.3 "Ordinals: unpack + optional fused
//! gather").

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::terms::TermsDict;
use lucene_arrow_core::plan::{BlockDecode, DecodePlan};
use lucene_arrow_core::{Error, Result};

/// Global dictionary + per-segment remap tables.
#[derive(Debug)]
pub struct OrdinalMap {
    /// Global terms, byte-sorted, deduplicated (Arrow binary layout).
    pub values: TermsDict,
    /// `remap[seg][local_ord] = global_ord`.
    pub remap: Vec<Vec<i32>>,
}

struct HeapEntry<'a> {
    term: &'a [u8],
    seg: usize,
    ord: usize,
}

// Min-heap on (term, seg): BinaryHeap is a max-heap, so reverse.
impl PartialEq for HeapEntry<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.term == other.term && self.seg == other.seg
    }
}
impl Eq for HeapEntry<'_> {}
impl PartialOrd for HeapEntry<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for HeapEntry<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        other.term.cmp(self.term).then(other.seg.cmp(&self.seg))
    }
}

/// Merge per-segment dictionaries into a global one. Segments must be
/// individually sorted (they are, by format construction).
pub fn build(dicts: &[&TermsDict]) -> Result<OrdinalMap> {
    let total: usize = dicts.iter().map(|d| d.len()).sum();
    let total_bytes: usize = dicts.iter().map(|d| d.bytes.len()).sum();

    let mut remap: Vec<Vec<i32>> = dicts.iter().map(|d| vec![0i32; d.len()]).collect();
    let mut values = TermsDict {
        bytes: Vec::with_capacity(total_bytes),
        offsets: Vec::with_capacity(total + 1),
    };
    values.offsets.push(0);

    let mut heap: BinaryHeap<HeapEntry<'_>> = BinaryHeap::with_capacity(dicts.len());
    for (seg, d) in dicts.iter().enumerate() {
        if !d.is_empty() {
            heap.push(HeapEntry { term: d.term(0), seg, ord: 0 });
        }
    }

    let mut last: Option<usize> = None; // start offset of last emitted term
    while let Some(e) = heap.pop() {
        let is_dup = last.is_some_and(|start| {
            &values.bytes[start..values.offsets[values.offsets.len() - 1] as usize] == e.term
        });
        if !is_dup {
            last = Some(values.bytes.len());
            values.bytes.extend_from_slice(e.term);
            values.offsets.push(
                i32::try_from(values.bytes.len())
                    .map_err(|_| Error::unsupported("global dictionary > 2 GiB"))?,
            );
        }
        remap[e.seg][e.ord] = (values.len() - 1) as i32;

        let d = dicts[e.seg];
        if e.ord + 1 < d.len() {
            heap.push(HeapEntry { term: d.term(e.ord + 1), seg: e.seg, ord: e.ord + 1 });
        }
    }

    Ok(OrdinalMap { values, remap })
}

/// Rewrite an ordinal plan so executors emit *global* ordinals: the remap
/// rides the existing table-gather epilogue (CPU and GPU alike), fused
/// into the unpack — no separate remap pass.
pub fn apply_remap(ords: &DecodePlan, remap: &[i32]) -> Result<DecodePlan> {
    let table: Vec<i64> = remap.iter().map(|&g| g as i64).collect();
    let mut plan = ords.clone();
    for block in &mut plan.blocks {
        *block = match *block {
            BlockDecode::Direct { offset, len, bit_width, values } => BlockDecode::Table {
                offset,
                len,
                bit_width,
                table: table.clone(),
                values,
            },
            // Constant ord (single distinct term in segment).
            BlockDecode::DeltaPacked { offset, len, bit_width: 0, base, values } => {
                let global = *table
                    .get(base as usize)
                    .ok_or_else(|| Error::invalid("constant ord out of remap range"))?;
                BlockDecode::DeltaPacked { offset, len, bit_width: 0, base: global, values }
            }
            ref other => {
                return Err(Error::invalid(format!(
                    "ordinal plans are Direct/constant, got {other:?}"
                )));
            }
        };
    }
    Ok(plan)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dict(terms: &[&str]) -> TermsDict {
        let mut d = TermsDict { bytes: Vec::new(), offsets: vec![0] };
        for t in terms {
            d.bytes.extend_from_slice(t.as_bytes());
            d.offsets.push(d.bytes.len() as i32);
        }
        d
    }

    #[test]
    fn merges_with_duplicates() {
        let a = dict(&["ant", "bee", "cat"]);
        let b = dict(&["bee", "dog"]);
        let c = dict(&["ant", "dog", "emu"]);
        let map = build(&[&a, &b, &c]).unwrap();

        let globals: Vec<&[u8]> = (0..map.values.len()).map(|i| map.values.term(i)).collect();
        assert_eq!(globals, [b"ant" as &[u8], b"bee", b"cat", b"dog", b"emu"]);
        assert_eq!(map.remap[0], vec![0, 1, 2]);
        assert_eq!(map.remap[1], vec![1, 3]);
        assert_eq!(map.remap[2], vec![0, 3, 4]);
    }

    #[test]
    fn empty_and_single_segment() {
        let a = dict(&[]);
        let b = dict(&["x"]);
        let map = build(&[&a, &b]).unwrap();
        assert_eq!(map.values.len(), 1);
        assert!(map.remap[0].is_empty());
        assert_eq!(map.remap[1], vec![0]);
    }
}
