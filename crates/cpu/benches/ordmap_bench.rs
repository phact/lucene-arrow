// SPDX-License-Identifier: Apache-2.0

//! OrdinalMap cost sweep — the SPEC §7.3 gate (decision register #3):
//! when is `dict = global` worth building versus per-segment dictionary
//! replacement (which costs ~nothing at plan time but pushes swaps to the
//! client)?
//!
//! Run: `cargo bench -p lucene-arrow-cpu --bench ordmap_bench`

use std::time::Instant;

use lucene_arrow_docvalues::ordmap;
use lucene_arrow_docvalues::terms::TermsDict;

fn synthetic_dict(seg: usize, terms_per_seg: usize, overlap_mod: usize) -> TermsDict {
    // ~50% of terms shared across segments (ids divisible by overlap_mod),
    // rest segment-unique; formatted so byte order == numeric order.
    let mut ids: Vec<u64> = (0..terms_per_seg as u64)
        .map(|i| {
            let id = i * 2;
            if (id as usize).is_multiple_of(overlap_mod) { id } else { id * 10 + seg as u64 }
        })
        .collect();
    ids.sort_unstable();
    ids.dedup();
    let mut d = TermsDict { bytes: Vec::new(), offsets: vec![0] };
    for id in ids {
        d.bytes.extend_from_slice(format!("term-{id:016}").as_bytes());
        d.offsets.push(d.bytes.len() as i32);
    }
    d
}

fn main() {
    println!("OrdinalMap build cost (k-way merge, single thread) — decision register #3");
    println!();
    println!("  segments × terms/seg |  total terms |  build ms | Mterms/s");
    println!("  ---------------------+--------------+-----------+---------");
    for (k, per_seg) in [(4usize, 10_000usize), (4, 100_000), (8, 100_000), (4, 1_000_000), (8, 1_000_000), (16, 1_000_000)] {
        let dicts: Vec<TermsDict> =
            (0..k).map(|s| synthetic_dict(s, per_seg, 4)).collect();
        let refs: Vec<&TermsDict> = dicts.iter().collect();
        let total: usize = dicts.iter().map(|d| d.len()).sum();

        let mut secs = Vec::new();
        for _ in 0..3 {
            let t = Instant::now();
            let map = ordmap::build(&refs).unwrap();
            secs.push(t.elapsed().as_secs_f64());
            std::hint::black_box(&map);
        }
        secs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let t = secs[secs.len() / 2];
        println!(
            "  {k:>9} × {per_seg:>8} | {total:>12} | {:>9.1} | {:>7.1}",
            t * 1e3,
            total as f64 / t / 1e6
        );
    }
    println!();
    println!("lean (SPEC §7.3): global for small dicts (sub-ms to ~10 ms),");
    println!("segment mode once merge cost rivals column decode itself.");
}
