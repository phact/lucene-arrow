// SPDX-License-Identifier: Apache-2.0

//! GPU text ingest (P10): tokenize + hash + vocabulary + (term, doc)
//! pair emission on-device, feeding the shared `keys_to_csr` finisher —
//! output byte-identical to `build_parallel` (differential-gated).
//!
//! Hybrid Unicode contract: the kernel owns pure-ASCII tokens (the
//! overwhelming majority in real text). Any maximal run containing a
//! non-ASCII byte is emitted as a *dirty span*; the CPU re-tokenizes
//! those spans with the exact analyzer. Span boundaries are ASCII
//! non-alphanumerics, which the analyzer also always breaks on, so the
//! merged result equals a pure-CPU pass.
//!
//! Vocabulary: open-addressed device table; a slot is one packed u64
//! `(offset<<16 | len)` pointing at the term's first occurrence in the
//! corpus (0 = empty; token lengths are ≥1 so the pack is never 0).
//! Probing compares lowercased bytes, so the table is exact — no
//! reliance on hash uniqueness.

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

use crate::executor::{GpuDecoder, cuda_err};
use lucene_arrow_core::{Error, Result};
use lucene_arrow_postings::build::{InvertedField, keys_to_csr};
use lucene_arrow_postings::text::{for_each_token, int_to_byte4};

const KERNEL: &str = r#"
typedef unsigned int u32; typedef unsigned long long u64;

__device__ inline bool tokenish(unsigned char b) {
    return (b >= '0' && b <= '9') || (b >= 'a' && b <= 'z')
        || (b >= 'A' && b <= 'Z') || b >= 0x80;
}
__device__ inline unsigned char lower(unsigned char b) {
    return (b >= 'A' && b <= 'Z') ? b + 32 : b;
}

// Compare token at [off,len) against stored packed loc, lowercased.
__device__ bool same_term(const unsigned char* text, u64 packed, u64 off, u32 len) {
    u32 slen = (u32)(packed & 0xFFFF);
    if (slen != len) return false;
    u64 soff = packed >> 16;
    for (u32 i = 0; i < len; i++)
        if (lower(text[soff + i]) != lower(text[off + i])) return false;
    return true;
}

extern "C" __global__ void tokenize_hash(
    const unsigned char* __restrict__ text, u64 text_len,
    const u64* __restrict__ doc_starts, u32 num_docs,
    u64* __restrict__ table, u32 table_mask,
    u64* __restrict__ pairs, u64* __restrict__ pair_count,
    u64* __restrict__ dirty, u64* __restrict__ dirty_count, u64 dirty_cap,
    u32* __restrict__ doc_tokens)
{
    const u32 SLICE = 64;
    u64 slice = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    u64 begin = slice * SLICE;
    if (begin >= text_len) return;
    u64 end = min(begin + SLICE, text_len);

    for (u64 i = begin; i < end; i++) {
        unsigned char b = text[i];
        if (!tokenish(b)) continue;
        if (i > 0 && tokenish(text[i - 1])) continue; // not a token start
        // scan the full token (may run past the slice)
        u64 j = i;
        bool dirty_tok = false;
        while (j < text_len && tokenish(text[j])) {
            if (text[j] >= 0x80) dirty_tok = true;
            j++;
        }
        u32 len = (u32)(j - i);
        // doc = upper_bound(doc_starts, i) - 1
        u32 lo = 0, hi = num_docs;
        while (lo < hi) {
            u32 mid = (lo + hi) / 2;
            if (doc_starts[mid] <= i) lo = mid + 1; else hi = mid;
        }
        u32 doc = lo - 1;

        if (dirty_tok) {
            u64 d = atomicAdd(dirty_count, 1);
            if (d < dirty_cap) dirty[d] = (i << 16) | (u64)min(len, (u32)0xFFFF);
            // doc recomputed on CPU from the span offset
            continue;
        }
        if (len > 255) continue; // analyzer drops over-long tokens
        // FNV-1a over lowercased bytes
        u64 h = 1469598103934665603ULL;
        for (u64 k = i; k < j; k++) {
            h ^= (u64)lower(text[k]);
            h *= 1099511628211ULL;
        }
        u64 myloc = (i << 16) | (u64)len;
        u32 slot = (u32)(h & table_mask);
        u32 winner;
        for (;;) {
            u64 old = atomicCAS(&table[slot], 0ULL, myloc);
            if (old == 0ULL || same_term(text, old, i, len)) {
                winner = slot;
                break;
            }
            slot = (slot + 1) & table_mask;
        }
        u64 p = atomicAdd(pair_count, 1);
        pairs[p] = ((u64)winner << 32) | (u64)doc;
        atomicAdd(&doc_tokens[doc], 1);
    }
}

// Emit (slot, packed loc) for occupied vocab slots only.
extern "C" __global__ void compact_table(
    const u64* __restrict__ table, u32 table_cap,
    u64* __restrict__ out_slots, u64* __restrict__ out_locs, u64* __restrict__ out_count)
{
    u32 s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= table_cap) return;
    u64 v = table[s];
    if (v != 0ULL) {
        u64 i = atomicAdd(out_count, 1);
        out_slots[i] = s;
        out_locs[i] = v;
    }
}
"#;

pub struct GpuTextIngest {
    module: std::sync::Arc<cudarc::driver::CudaModule>,
}

/// D2H via a cacheable pinned staging buffer (pageable D2H runs ~2 GB/s;
/// pinned ~14 GB/s — same finding as the H2D PinnedRing).
fn download_pinned_u64(
    stream: &cudarc::driver::CudaStream,
    src: &CudaSlice<u64>,
    n: usize,
) -> Result<Vec<u64>> {
    use cudarc::driver::result as cu;
    use cudarc::driver::DevicePtr;
    if n == 0 {
        return Ok(Vec::new());
    }
    let bytes = n * 8;
    let ptr = unsafe { cu::malloc_host(bytes, 0) }.map_err(cuda_err)? as *mut u64;
    // Safety: freshly allocated pinned block of exactly `bytes`.
    let host: &mut [u64] = unsafe { std::slice::from_raw_parts_mut(ptr, n) };
    let (dptr, _guard) = src.device_ptr(stream);
    unsafe { cu::memcpy_dtoh_async(host, dptr, stream.cu_stream()) }.map_err(cuda_err)?;
    stream.synchronize().map_err(cuda_err)?;
    let out = host.to_vec();
    unsafe {
        let _ = cu::free_host(ptr as *mut std::ffi::c_void);
    }
    Ok(out)
}

#[derive(Debug, Default)]
pub struct GpuIngestStats {
    pub clean_pairs: u64,
    pub dirty_spans: u64,
    pub kernel_ms: f64,
    pub download_ms: f64,
    pub vocab_ms: f64,
    pub dirty_ms: f64,
    pub remap_ms: f64,
    pub csr_ms: f64,
}

impl GpuTextIngest {
    pub fn new(gpu: &GpuDecoder) -> Result<Self> {
        let ptx = cudarc::nvrtc::compile_ptx(KERNEL).map_err(cuda_err)?;
        let module = gpu.stream().context().load_module(ptx).map_err(cuda_err)?;
        Ok(GpuTextIngest { module })
    }

    /// Build an `InvertedField` from a line-per-doc corpus, tokenizing on
    /// the GPU. Byte-identical to `build_parallel` over the same text.
    pub fn build(
        &self,
        gpu: &GpuDecoder,
        corpus: &str,
        threads: usize,
    ) -> Result<(InvertedField, GpuIngestStats)> {
        let bytes = corpus.as_bytes();
        let mut doc_starts: Vec<u64> = vec![0];
        for (i, &b) in bytes.iter().enumerate() {
            if b == b'\n' && i + 1 < bytes.len() {
                doc_starts.push(i as u64 + 1);
            }
        }
        let num_docs = doc_starts.len();

        let stream = gpu.stream().clone();
        let f = self.module.load_function("tokenize_hash").map_err(cuda_err)?;
        let d_text: CudaSlice<u8> = stream.clone_htod(bytes).map_err(cuda_err)?;
        let d_starts: CudaSlice<u64> = stream.clone_htod(&doc_starts).map_err(cuda_err)?;
        // Table: 4M slots (pow2) covers vocabularies into the millions.
        let table_cap: usize = 1 << 22;
        let table_mask = (table_cap - 1) as u32;
        let mut d_table: CudaSlice<u64> = stream.alloc_zeros(table_cap).map_err(cuda_err)?;
        // Worst case one pair per byte/2 (tokens are ≥1 char + separator).
        let pair_cap = bytes.len() / 2 + 16;
        let mut d_pairs: CudaSlice<u64> = stream.alloc_zeros(pair_cap).map_err(cuda_err)?;
        let mut d_pair_count: CudaSlice<u64> = stream.alloc_zeros(1).map_err(cuda_err)?;
        let dirty_cap = (bytes.len() / 8 + 16) as u64;
        let mut d_dirty: CudaSlice<u64> =
            stream.alloc_zeros(dirty_cap as usize).map_err(cuda_err)?;
        let mut d_dirty_count: CudaSlice<u64> = stream.alloc_zeros(1).map_err(cuda_err)?;
        let mut d_doc_tokens: CudaSlice<u32> = stream.alloc_zeros(num_docs).map_err(cuda_err)?;

        let text_len = bytes.len() as u64;
        let n_docs_u32 = num_docs as u32;
        let slices = text_len.div_ceil(64);
        let cfg = LaunchConfig {
            grid_dim: (slices.div_ceil(256) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let t0 = std::time::Instant::now();
        let mut lb = stream.launch_builder(&f);
        lb.arg(&d_text)
            .arg(&text_len)
            .arg(&d_starts)
            .arg(&n_docs_u32)
            .arg(&mut d_table)
            .arg(&table_mask)
            .arg(&mut d_pairs)
            .arg(&mut d_pair_count)
            .arg(&mut d_dirty)
            .arg(&mut d_dirty_count)
            .arg(&dirty_cap)
            .arg(&mut d_doc_tokens);
        // Safety: all buffers sized above; kernel bounds-checks.
        unsafe { lb.launch(cfg) }.map_err(cuda_err)?;
        stream.synchronize().map_err(cuda_err)?;
        let kernel_ms = t0.elapsed().as_secs_f64() * 1e3;

        let mut stats = GpuIngestStats {
            kernel_ms,
            ..Default::default()
        };
        let t = std::time::Instant::now();
        let n_pairs = stream.clone_dtoh(&d_pair_count).map_err(cuda_err)?[0] as usize;
        let n_dirty = stream.clone_dtoh(&d_dirty_count).map_err(cuda_err)?[0];
        if n_dirty > dirty_cap {
            return Err(Error::invalid("dirty-span buffer overflow"));
        }
        // Compact the vocab table on-device: download K entries, not 4M slots.
        let fc = self.module.load_function("compact_table").map_err(cuda_err)?;
        let mut d_slots: CudaSlice<u64> =
            stream.alloc_zeros(table_cap.min(1 << 21)).map_err(cuda_err)?;
        let mut d_locs: CudaSlice<u64> =
            stream.alloc_zeros(table_cap.min(1 << 21)).map_err(cuda_err)?;
        let mut d_occ: CudaSlice<u64> = stream.alloc_zeros(1).map_err(cuda_err)?;
        let tc = table_cap as u32;
        let ccfg = LaunchConfig {
            grid_dim: ((table_cap as u32).div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut cb = stream.launch_builder(&fc);
        cb.arg(&d_table).arg(&tc).arg(&mut d_slots).arg(&mut d_locs).arg(&mut d_occ);
        // Safety: outputs sized to table_cap bound.
        unsafe { cb.launch(ccfg) }.map_err(cuda_err)?;
        stream.synchronize().map_err(cuda_err)?;
        let n_occ = stream.clone_dtoh(&d_occ).map_err(cuda_err)?[0] as usize;

        let pairs_raw = download_pinned_u64(&stream, &d_pairs, n_pairs)?;
        let occ_slots = download_pinned_u64(&stream, &d_slots, n_occ)?;
        let occ_locs = download_pinned_u64(&stream, &d_locs, n_occ)?;
        let dirty = download_pinned_u64(&stream, &d_dirty, n_dirty as usize)?;
        let mut doc_tokens = stream.clone_dtoh(&d_doc_tokens).map_err(cuda_err)?;
        stats.download_ms = t.elapsed().as_secs_f64() * 1e3;
        let t = std::time::Instant::now();

        // Slot → lowercased term bytes (compacted entries only).
        let mut slot_term: hashbrown::HashMap<u32, String> = hashbrown::HashMap::new();
        let mut vocab: hashbrown::HashMap<String, ()> = hashbrown::HashMap::new();
        for (&slot, &packed) in occ_slots.iter().zip(&occ_locs) {
            let off = (packed >> 16) as usize;
            let len = (packed & 0xFFFF) as usize;
            let term: String =
                bytes[off..off + len].iter().map(|b| b.to_ascii_lowercase() as char).collect();
            vocab.insert(term.clone(), ());
            slot_term.insert(slot as u32, term);
        }

        stats.vocab_ms = t.elapsed().as_secs_f64() * 1e3;
        let t = std::time::Instant::now();

        // Dirty spans: exact analyzer on CPU, parallel chunks.
        struct DirtyOut {
            pairs: Vec<(String, u32)>,
            counts: Vec<(u32, u32)>, // (doc, extra tokens)
        }
        let doc_starts_ref = &doc_starts;
        let outs: Vec<DirtyOut> = std::thread::scope(|s| {
            dirty
                .chunks((n_dirty as usize).div_ceil(threads.max(1)).max(1))
                .map(|chunk| {
                    s.spawn(move || {
                        let mut o =
                            DirtyOut { pairs: Vec::new(), counts: Vec::new() };
                        for &packed in chunk {
                            let off = (packed >> 16) as usize;
                            let len = (packed & 0xFFFF) as usize;
                            let doc =
                                (doc_starts_ref.partition_point(|&st| st <= off as u64) - 1)
                                    as u32;
                            let Ok(span) = std::str::from_utf8(&bytes[off..off + len]) else {
                                continue;
                            };
                            let mut n = 0u32;
                            for_each_token(span, |tok| {
                                o.pairs.push((tok.to_string(), doc));
                                n += 1;
                            });
                            if n > 0 {
                                o.counts.push((doc, n));
                            }
                        }
                        o
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|h| h.join().unwrap())
                .collect()
        });
        let mut dirty_pairs: Vec<(String, u32)> = Vec::new();
        for o in outs {
            for (doc, n) in o.counts {
                doc_tokens[doc as usize] += n;
            }
            for p in o.pairs {
                vocab.insert(p.0.clone(), ());
                dirty_pairs.push(p);
            }
        }

        stats.dirty_ms = t.elapsed().as_secs_f64() * 1e3;
        let t = std::time::Instant::now();

        // Global sorted vocab + ord maps.
        let mut all_terms: Vec<&str> = vocab.keys().map(|k| k.as_str()).collect();
        all_terms.sort_unstable();
        let ord_of: hashbrown::HashMap<&str, u32> =
            all_terms.iter().enumerate().map(|(i, &t)| (t, i as u32)).collect();
        let mut slot_ord = vec![u32::MAX; table_cap];
        for (&slot, term) in &slot_term {
            slot_ord[slot as usize] = ord_of[term.as_str()];
        }

        // Remap pairs (slot→ord) in parallel chunks, then shared finisher.
        let pairs = &pairs_raw[..n_pairs];
        let slot_ord_ref = &slot_ord;
        let key_chunks: Vec<Vec<u64>> = std::thread::scope(|s| {
            pairs
                .chunks(n_pairs.div_ceil(threads.max(1)).max(1))
                .map(|c| {
                    s.spawn(move || {
                        c.iter()
                            .map(|&p| {
                                let ord = slot_ord_ref[(p >> 32) as usize] as u64;
                                (ord << 32) | (p & 0xFFFF_FFFF)
                            })
                            .collect::<Vec<u64>>()
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|h| h.join().unwrap())
                .collect()
        });
        let mut chunks = key_chunks;
        chunks.push(
            dirty_pairs
                .iter()
                .map(|(t, d)| ((ord_of[t.as_str()] as u64) << 32) | *d as u64)
                .collect(),
        );

        stats.remap_ms = t.elapsed().as_secs_f64() * 1e3;
        let t = std::time::Instant::now();
        let (docs, freqs, row_offsets, ttf) = keys_to_csr(chunks, all_terms.len(), threads);
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
        inv.norms = doc_tokens.iter().map(|&c| int_to_byte4(c as i32) as i64).collect();
        stats.csr_ms = t.elapsed().as_secs_f64() * 1e3;
        stats.clean_pairs = n_pairs as u64;
        stats.dirty_spans = n_dirty;

        Ok((inv, stats))
    }
}
