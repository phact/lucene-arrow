// SPDX-License-Identifier: Apache-2.0

//! Inverted-index builder (P9a CPU reference; the GPU sort/aggregate
//! path must produce byte-identical output). Documents in, and out comes
//! the shape Bearing's public `BlockTreeTermsWriter::write_field` wants:
//! sorted terms with per-term (doc, freq) postings, plus per-doc norms
//! (SmallFloat-encoded field length).

use std::collections::HashMap;

use crate::text::{int_to_byte4, tokenize};

/// One field's aggregated postings, CSR over sorted terms.
#[derive(Debug, Default)]
pub struct InvertedField {
    /// Sorted unique terms (flattened + offsets, len = num_terms + 1).
    pub term_bytes: Vec<u8>,
    pub term_offsets: Vec<u64>,
    /// Row spans per term (len = num_terms + 1) into docs/freqs.
    pub row_offsets: Vec<u64>,
    pub docs: Vec<u32>,
    pub freqs: Vec<u32>,
    /// Per-doc norm byte (SmallFloat(field length)), dense in doc order.
    pub norms: Vec<i64>,
    pub sum_total_term_freq: i64,
}

impl InvertedField {
    pub fn num_terms(&self) -> usize {
        self.term_offsets.len().saturating_sub(1)
    }
    pub fn term(&self, ord: usize) -> &[u8] {
        &self.term_bytes[self.term_offsets[ord] as usize..self.term_offsets[ord + 1] as usize]
    }
    pub fn postings(&self, ord: usize) -> (&[u32], &[u32]) {
        let r = self.row_offsets[ord] as usize..self.row_offsets[ord + 1] as usize;
        (&self.docs[r.clone()], &self.freqs[r])
    }
}

/// CPU-reference builder: tokenize + hash-aggregate per doc, sort once at
/// finish. Term ids are assigned first-seen; postings accumulate per id
/// and are permuted into sorted-term order at the end (the same plan the
/// GPU path follows with a device hash + radix sort).
#[derive(Default)]
pub struct IndexBuilder {
    term_ids: HashMap<Vec<u8>, u32>,
    postings: Vec<Vec<(u32, u32)>>, // per term id: (doc, freq)
    norms: Vec<i64>,
    next_doc: u32,
}

impl IndexBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add one document's text (e.g. a markdown file). Returns its docid.
    pub fn add_doc(&mut self, text: &str) -> u32 {
        let doc = self.next_doc;
        self.next_doc += 1;
        let tokens = tokenize(text);
        self.norms.push(int_to_byte4(tokens.len() as i32) as i64);
        let mut counts: HashMap<&str, u32> = HashMap::new();
        for t in &tokens {
            *counts.entry(t.as_str()).or_insert(0) += 1;
        }
        for (term, freq) in counts {
            let id = *self.term_ids.entry(term.as_bytes().to_vec()).or_insert_with(|| {
                self.postings.push(Vec::new());
                (self.postings.len() - 1) as u32
            });
            self.postings[id as usize].push((doc, freq));
        }
        doc
    }

    pub fn num_docs(&self) -> u32 {
        self.next_doc
    }

    /// Sort terms + emit CSR. Postings within a term are already in doc
    /// order (docs are added in order).
    pub fn finish(self) -> InvertedField {
        let mut terms: Vec<(Vec<u8>, u32)> = self.term_ids.into_iter().collect();
        terms.sort_unstable_by(|a, b| a.0.cmp(&b.0));

        let mut out = InvertedField {
            norms: self.norms,
            ..Default::default()
        };
        out.term_offsets.push(0);
        out.row_offsets.push(0);
        for (bytes, id) in terms {
            out.term_bytes.extend_from_slice(&bytes);
            out.term_offsets.push(out.term_bytes.len() as u64);
            for &(doc, freq) in &self.postings[id as usize] {
                out.docs.push(doc);
                out.freqs.push(freq);
                out.sum_total_term_freq += freq as i64;
            }
            out.row_offsets.push(out.docs.len() as u64);
        }
        out
    }
}

/// Parallel builder: chunk docs across threads (zero-alloc tokenizer +
/// `entry_ref` interning, local vocab each), merge vocabs into global
/// sorted-term ords, then bucket-parallel sort+RLE (keys are partitioned
/// by ord range, each bucket sorts independently — no merge step).
/// Output is byte-identical to [`IndexBuilder`] (gated in tests).
pub fn build_parallel(lines: &[&str], threads: usize) -> InvertedField {
    use crate::text::{for_each_token, int_to_byte4};
    let threads = threads.clamp(1, lines.len().max(1));
    let chunk = lines.len().div_ceil(threads);

    struct Local {
        vocab: hashbrown::HashMap<String, u32>,
        keys: Vec<(u32, u32)>, // (local term id, doc)
        norms: Vec<i64>,
        doc_base: u32,
    }
    let locals: Vec<Local> = std::thread::scope(|s| {
        let mut handles = Vec::new();
        for (ti, part) in lines.chunks(chunk).enumerate() {
            let doc_base = (ti * chunk) as u32;
            handles.push(s.spawn(move || {
                let mut vocab: hashbrown::HashMap<String, u32> = hashbrown::HashMap::new();
                let mut keys = Vec::new();
                let mut norms = Vec::new();
                for (i, line) in part.iter().enumerate() {
                    let mut count = 0i32;
                    for_each_token(line, |tok| {
                        count += 1;
                        let next = vocab.len() as u32;
                        let id = *vocab.entry_ref(tok).or_insert(next);
                        keys.push((id, doc_base + i as u32));
                    });
                    norms.push(int_to_byte4(count) as i64);
                }
                Local { vocab, keys, norms, doc_base }
            }));
        }
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    // Global sorted vocab + per-thread local-id → global-ord remaps.
    let mut all_terms: Vec<&str> =
        locals.iter().flat_map(|l| l.vocab.keys().map(|k| k.as_str())).collect();
    all_terms.sort_unstable();
    all_terms.dedup();
    let num_terms = all_terms.len();
    let ord_of: hashbrown::HashMap<&str, u32> =
        all_terms.iter().enumerate().map(|(i, &t)| (t, i as u32)).collect();
    let remaps: Vec<Vec<u32>> = locals
        .iter()
        .map(|l| {
            let mut r = vec![0u32; l.vocab.len()];
            for (term, &id) in &l.vocab {
                r[id as usize] = ord_of[term.as_str()];
            }
            r
        })
        .collect();

    let keys_per_thread: Vec<Vec<u64>> = locals
        .iter()
        .zip(&remaps)
        .map(|(l, remap)| {
            l.keys
                .iter()
                .map(|&(id, doc)| ((remap[id as usize] as u64) << 32) | doc as u64)
                .collect()
        })
        .collect();
    let (docs, freqs, row_offsets, ttf) = keys_to_csr(keys_per_thread, num_terms, threads);
    let mut inv = InvertedField {
        docs,
        freqs,
        row_offsets,
        sum_total_term_freq: ttf,
        ..Default::default()
    };
    inv.term_offsets.push(0);
    for t in &all_terms {
        inv.term_bytes.extend_from_slice(t.as_bytes());
        inv.term_offsets.push(inv.term_bytes.len() as u64);
    }
    inv.norms = {
        let mut norms = vec![0i64; lines.len()];
        for l in &locals {
            norms[l.doc_base as usize..l.doc_base as usize + l.norms.len()]
                .copy_from_slice(&l.norms);
        }
        norms
    };
    inv
}

/// Bucket-parallel sort + RLE + CSR row assembly over `(ord<<32|doc)`
/// keys (any number of unsorted key chunks). Deterministic regardless of
/// chunk composition/order — the shared finisher for the CPU-parallel and
/// GPU ingest paths. Returns `(docs, freqs, row_offsets, sum_ttf)`.
pub fn keys_to_csr(
    key_chunks: Vec<Vec<u64>>,
    num_terms: usize,
    threads: usize,
) -> (Vec<u32>, Vec<u32>, Vec<u64>, i64) {
    const BUCKETS: usize = 128;
    let bucket_of = |ord: u32| -> usize {
        ((ord as u64 * BUCKETS as u64) / num_terms.max(1) as u64).min(BUCKETS as u64 - 1) as usize
    };
    let mut buckets: Vec<Vec<u64>> = std::thread::scope(|s| {
        let handles: Vec<_> = key_chunks
            .iter()
            .map(|chunk| {
                s.spawn(move || {
                    let mut local_buckets: Vec<Vec<u64>> = vec![Vec::new(); BUCKETS];
                    for &key in chunk {
                        local_buckets[bucket_of((key >> 32) as u32)].push(key);
                    }
                    local_buckets
                })
            })
            .collect();
        let per_thread: Vec<Vec<Vec<u64>>> =
            handles.into_iter().map(|h| h.join().unwrap()).collect();
        (0..BUCKETS)
            .map(|b| {
                let cap = per_thread.iter().map(|t| t[b].len()).sum();
                let mut v = Vec::with_capacity(cap);
                for t in &per_thread {
                    v.extend_from_slice(&t[b]);
                }
                v
            })
            .collect()
    });
    std::thread::scope(|s| {
        for chunk in buckets.chunks_mut(BUCKETS.div_ceil(threads)) {
            s.spawn(move || {
                for b in chunk {
                    b.sort_unstable();
                }
            });
        }
    });

    // RLE each bucket (parallel), then stitch CSR sequentially.
    struct BucketOut {
        docs: Vec<u32>,
        freqs: Vec<u32>,
        // (ord, rows-end within this bucket) at each ord boundary
        ord_ends: Vec<(u32, u64)>,
        ttf: i64,
    }
    let outs: Vec<BucketOut> = std::thread::scope(|s| {
        let handles: Vec<_> = buckets
            .iter()
            .map(|keys| {
                s.spawn(move || {
                    let mut o = BucketOut {
                        docs: Vec::new(),
                        freqs: Vec::new(),
                        ord_ends: Vec::new(),
                        ttf: 0,
                    };
                    let mut i = 0usize;
                    while i < keys.len() {
                        let ord = (keys[i] >> 32) as u32;
                        while i < keys.len() && (keys[i] >> 32) as u32 == ord {
                            let key = keys[i];
                            let mut f = 0u32;
                            while i < keys.len() && keys[i] == key {
                                f += 1;
                                i += 1;
                            }
                            o.docs.push(key as u32);
                            o.freqs.push(f);
                            o.ttf += f as i64;
                        }
                        o.ord_ends.push((ord, o.docs.len() as u64));
                    }
                    o
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    let mut docs = Vec::new();
    let mut freqs = Vec::new();
    let mut row_offsets = vec![0u64; num_terms + 1];
    let mut ttf = 0i64;
    let mut base_rows = 0u64;
    for o in &outs {
        for &(ord, end) in &o.ord_ends {
            row_offsets[ord as usize + 1] = base_rows + end;
        }
        docs.extend_from_slice(&o.docs);
        freqs.extend_from_slice(&o.freqs);
        ttf += o.ttf;
        base_rows += o.docs.len() as u64;
    }
    // Fill forward so every ord has a valid span even with empty gaps.
    for t in 1..=num_terms {
        if row_offsets[t] < row_offsets[t - 1] {
            row_offsets[t] = row_offsets[t - 1];
        }
    }
    (docs, freqs, row_offsets, ttf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_sorted_csr_with_correct_freqs_and_norms() {
        let mut b = IndexBuilder::new();
        b.add_doc("the quick brown fox the fox");
        b.add_doc("# The *lazy* dog");
        let f = b.finish();

        let terms: Vec<&[u8]> = (0..f.num_terms()).map(|t| f.term(t)).collect();
        assert_eq!(terms, [&b"brown"[..], b"dog", b"fox", b"lazy", b"quick", b"the"]);
        let (docs, freqs) = f.postings(2); // fox
        assert_eq!((docs, freqs), (&[0u32][..], &[2u32][..]));
        let (docs, freqs) = f.postings(5); // the
        assert_eq!((docs, freqs), (&[0u32, 1][..], &[2u32, 1][..]));
        assert_eq!(f.norms, vec![6, 3]); // 6 and 3 tokens, identity range
        assert_eq!(f.sum_total_term_freq, 9);
    }

    #[test]
    fn parallel_matches_serial_exactly() {
        let lines: Vec<String> = (0..500)
            .map(|i| {
                format!(
                    "# doc{i} alpha w{} w{} common beta w{}",
                    i % 7,
                    i % 31,
                    (i * 13) % 101
                )
            })
            .collect();
        let refs: Vec<&str> = lines.iter().map(|s| s.as_str()).collect();
        let mut b = IndexBuilder::new();
        for l in &refs {
            b.add_doc(l);
        }
        let serial = b.finish();
        for threads in [1, 3, 8] {
            let par = build_parallel(&refs, threads);
            assert_eq!(par.term_bytes, serial.term_bytes, "t={threads}");
            assert_eq!(par.term_offsets, serial.term_offsets, "t={threads}");
            assert_eq!(par.row_offsets, serial.row_offsets, "t={threads}");
            assert_eq!(par.docs, serial.docs, "t={threads}");
            assert_eq!(par.freqs, serial.freqs, "t={threads}");
            assert_eq!(par.norms, serial.norms, "t={threads}");
            assert_eq!(par.sum_total_term_freq, serial.sum_total_term_freq);
        }
    }
}
