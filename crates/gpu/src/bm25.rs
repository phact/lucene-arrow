// SPDX-License-Identifier: Apache-2.0

//! BM25 scoring over the CSR postings relation (P9c, SPEC §7.8 + §11).
//! Exhaustive disjunctive scoring: every posting of every query term
//! contributes `idf · tf / (tf + k1·(1−b+b·dl/avgdl))` to its doc's
//! accumulator — identical math to Lucene's BM25Similarity (norm byte
//! decoded via the SmallFloat table), so scores match Java exactly;
//! only the execution strategy differs (we score all postings, Lucene
//! skips via impacts).

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

use crate::executor::{GpuDecoder, cuda_err};
use lucene_arrow_core::Result;

pub const K1: f32 = 1.2;
pub const B: f32 = 0.75;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct QueryTerm {
    /// Row span of this term in the CSR arrays.
    pub row_start: u64,
    pub row_end: u64,
    /// Precomputed BM25 idf for this term.
    pub idf: f32,
    pub _pad: f32,
}

const KERNEL: &str = r#"
typedef unsigned int u32; typedef unsigned long long u64;

struct QueryTerm { u64 row_start; u64 row_end; float idf; float _pad; };
struct QuerySpan { u32 term_start; u32 term_end; };

// cache[b] = decoded field length for norm byte b (SmallFloat byte4ToInt).
extern "C" __global__ void bm25_score(
    const u32* __restrict__ docs, const u32* __restrict__ freqs,
    const unsigned char* __restrict__ norms, const float* __restrict__ len_cache,
    const QueryTerm* __restrict__ terms, u32 num_terms,
    u64 total_rows, float k1, float b, float avgdl,
    float* __restrict__ scores)
{
    u64 g = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= total_rows) return;
    // locate the query term owning this flattened row index
    u64 off = g;
    u32 t = 0;
    for (; t < num_terms; t++) {
        u64 n = terms[t].row_end - terms[t].row_start;
        if (off < n) break;
        off -= n;
    }
    u64 row = terms[t].row_start + off;
    u32 doc = docs[row];
    float tf = (float)freqs[row];
    float dl = len_cache[norms[doc]];
    float s = terms[t].idf * tf / (tf + k1 * (1.0f - b + b * dl / avgdl));
    atomicAdd(&scores[doc], s);
}

// Batched form: grid.y = query, grid.x covers that query's flattened rows.
// One launch scores the whole batch into scores[q * num_docs + doc] — the
// per-query launch/sync/download overhead was the selective-query
// bottleneck (~200 us/query floor), not the math.
extern "C" __global__ void bm25_score_batch(
    const u32* __restrict__ docs, const u32* __restrict__ freqs,
    const unsigned char* __restrict__ norms, const float* __restrict__ len_cache,
    const QueryTerm* __restrict__ terms, const QuerySpan* __restrict__ queries,
    u32 num_docs, float k1, float b, float avgdl,
    float* __restrict__ scores)
{
    u32 q = blockIdx.y;
    QuerySpan Q = queries[q];
    u64 off = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    u32 t = Q.term_start;
    for (; t < Q.term_end; t++) {
        u64 n = terms[t].row_end - terms[t].row_start;
        if (off < n) break;
        off -= n;
    }
    if (t >= Q.term_end) return; // beyond this query's rows
    u64 row = terms[t].row_start + off;
    u32 doc = docs[row];
    float tf = (float)freqs[row];
    float dl = len_cache[norms[doc]];
    float s = terms[t].idf * tf / (tf + k1 * (1.0f - b + b * dl / avgdl));
    atomicAdd(&scores[(u64)q * num_docs + doc], s);
}

// Device top-k over each query's dense score row: one block per query.
// Each thread keeps an insertion-sorted local top-k of its strided slice
// (docs are disjoint across threads), lists land in shared memory, then k
// rounds of block-parallel argmax select the winners. Download is k pairs
// per query instead of a num_docs-sized float row.
#define MAXK 16
extern "C" __global__ void bm25_topk(
    const float* __restrict__ scores, u32 num_docs, u32 k,
    u32* __restrict__ out_docs, float* __restrict__ out_scores)
{
    u32 q = blockIdx.x;
    const float* s = scores + (u64)q * num_docs;
    float tv[MAXK]; u32 td[MAXK];
    for (u32 i = 0; i < k; i++) { tv[i] = -1e30f; td[i] = 0xffffffffu; }
    for (u32 d = threadIdx.x; d < num_docs; d += blockDim.x) {
        float v = s[d];
        if (v > tv[k - 1]) {
            int i = (int)k - 1;
            while (i > 0 && tv[i - 1] < v) { tv[i] = tv[i - 1]; td[i] = td[i - 1]; i--; }
            tv[i] = v; td[i] = d;
        }
    }
    __shared__ float sv[256 * MAXK];
    __shared__ u32 sd[256 * MAXK];
    for (u32 i = 0; i < k; i++) {
        sv[threadIdx.x * k + i] = tv[i];
        sd[threadIdx.x * k + i] = td[i];
    }
    __syncthreads();
    __shared__ float rv[256];
    __shared__ u32 ri[256];
    u32 total = blockDim.x * k;
    for (u32 round = 0; round < k; round++) {
        float best = -1e30f; u32 besti = 0;
        for (u32 i = threadIdx.x; i < total; i += blockDim.x) {
            if (sv[i] > best) { best = sv[i]; besti = i; }
        }
        rv[threadIdx.x] = best; ri[threadIdx.x] = besti;
        __syncthreads();
        for (u32 step = blockDim.x / 2; step > 0; step >>= 1) {
            if (threadIdx.x < step && rv[threadIdx.x + step] > rv[threadIdx.x]) {
                rv[threadIdx.x] = rv[threadIdx.x + step];
                ri[threadIdx.x] = ri[threadIdx.x + step];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            out_scores[(u64)q * k + round] = rv[0];
            out_docs[(u64)q * k + round] = sd[ri[0]];
            sv[ri[0]] = -1e30f; // consumed
        }
        __syncthreads();
    }
}
"#;

pub struct Bm25Scorer {
    module: std::sync::Arc<cudarc::driver::CudaModule>,
    len_cache: CudaSlice<f32>,
}

impl Bm25Scorer {
    pub fn new(gpu: &GpuDecoder) -> Result<Self> {
        let ptx = cudarc::nvrtc::compile_ptx(KERNEL).map_err(cuda_err)?;
        let module = gpu.stream().context().load_module(ptx).map_err(cuda_err)?;
        // SmallFloat byte4→length lookup, computed once on host.
        let table: Vec<f32> = (0u32..256)
            .map(|b| lucene_arrow_postings::text::byte4_to_int(b as u8) as f32)
            .collect();
        let len_cache = gpu.stream().clone_htod(&table).map_err(cuda_err)?;
        Ok(Bm25Scorer { module, len_cache })
    }

    /// Upload CSR columns once; reusable across queries.
    pub fn upload(
        &self,
        gpu: &GpuDecoder,
        docs: &[u32],
        freqs: &[u32],
        norms_bytes: &[u8],
    ) -> Result<(CudaSlice<u32>, CudaSlice<u32>, CudaSlice<u8>)> {
        let s = gpu.stream();
        Ok((
            s.clone_htod(docs).map_err(cuda_err)?,
            s.clone_htod(freqs).map_err(cuda_err)?,
            s.clone_htod(norms_bytes).map_err(cuda_err)?,
        ))
    }

    /// Score one disjunctive query; returns the dense per-doc score array.
    #[allow(clippy::too_many_arguments)]
    pub fn score(
        &self,
        gpu: &GpuDecoder,
        docs: &CudaSlice<u32>,
        freqs: &CudaSlice<u32>,
        norms: &CudaSlice<u8>,
        terms: &[QueryTerm],
        num_docs: u32,
        avgdl: f32,
    ) -> Result<Vec<f32>> {
        let stream = gpu.stream().clone();
        let f = self.module.load_function("bm25_score").map_err(cuda_err)?;
        let total_rows: u64 = terms.iter().map(|t| t.row_end - t.row_start).sum();
        // Safety: QueryTerm is #[repr(C)] POD.
        let raw: &[u8] = unsafe {
            std::slice::from_raw_parts(
                terms.as_ptr() as *const u8,
                std::mem::size_of_val(terms),
            )
        };
        let terms_dev: CudaSlice<u8> = stream.clone_htod(raw).map_err(cuda_err)?;
        let mut scores: CudaSlice<f32> =
            stream.alloc_zeros(num_docs as usize).map_err(cuda_err)?;
        let n_terms = terms.len() as u32;
        let cfg = LaunchConfig {
            grid_dim: (total_rows.div_ceil(256) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let (k1, b) = (K1, B);
        let mut lb = stream.launch_builder(&f);
        lb.arg(docs)
            .arg(freqs)
            .arg(norms)
            .arg(&self.len_cache)
            .arg(&terms_dev)
            .arg(&n_terms)
            .arg(&total_rows)
            .arg(&k1)
            .arg(&b)
            .arg(&avgdl)
            .arg(&mut scores);
        // Safety: grid bounded by total_rows check in-kernel.
        unsafe { lb.launch(cfg) }.map_err(cuda_err)?;
        stream.clone_dtoh(&scores).map_err(cuda_err)
    }

    /// Score a **batch** of disjunctive queries in one launch and return
    /// each query's top-`k` `(doc, score)` pairs, selected on the device.
    /// This is the shape that fills the GPU: per-query launches pay a
    /// ~200 µs fixed floor (launch + sync + a dense-row download + host
    /// argmax); the batch pays it once. Device memory: `queries.len() ×
    /// num_docs × 4` bytes for the score matrix — chunk the batch if that
    /// doesn't fit.
    #[allow(clippy::too_many_arguments)]
    pub fn score_batch(
        &self,
        gpu: &GpuDecoder,
        docs: &CudaSlice<u32>,
        freqs: &CudaSlice<u32>,
        norms: &CudaSlice<u8>,
        queries: &[Vec<QueryTerm>],
        num_docs: u32,
        avgdl: f32,
        k: usize,
    ) -> Result<Vec<Vec<(u32, f32)>>> {
        use lucene_arrow_core::Error;
        let nq = queries.len();
        if nq == 0 {
            return Ok(Vec::new());
        }
        if !(1..=16).contains(&k) {
            return Err(Error::invalid("bm25 top-k supports 1..=16"));
        }
        if nq > 65_535 {
            return Err(Error::invalid("batch > 65535 queries: chunk it"));
        }
        let stream = gpu.stream().clone();

        // Flatten terms + per-query spans; grid.x sized by the widest query.
        let mut flat: Vec<QueryTerm> = Vec::new();
        let mut spans: Vec<u32> = Vec::with_capacity(nq * 2); // (start, end) pairs
        let mut max_rows = 1u64;
        for q in queries {
            spans.push(flat.len() as u32);
            flat.extend_from_slice(q);
            spans.push(flat.len() as u32);
            max_rows = max_rows.max(q.iter().map(|t| t.row_end - t.row_start).sum());
        }
        // Safety: QueryTerm is #[repr(C)] POD; spans are pairs of u32
        // matching the kernel's QuerySpan.
        let raw_terms: &[u8] = unsafe {
            std::slice::from_raw_parts(flat.as_ptr() as *const u8, std::mem::size_of_val(&flat[..]))
        };
        let d_terms = stream.clone_htod(raw_terms).map_err(cuda_err)?;
        let d_spans = stream.clone_htod(&spans).map_err(cuda_err)?;
        let mut scores: CudaSlice<f32> =
            stream.alloc_zeros(nq * num_docs as usize).map_err(cuda_err)?;

        let f_score = self.module.load_function("bm25_score_batch").map_err(cuda_err)?;
        let cfg = LaunchConfig {
            grid_dim: (max_rows.div_ceil(256) as u32, nq as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let (k1, b) = (K1, B);
        let mut lb = stream.launch_builder(&f_score);
        lb.arg(docs)
            .arg(freqs)
            .arg(norms)
            .arg(&self.len_cache)
            .arg(&d_terms)
            .arg(&d_spans)
            .arg(&num_docs)
            .arg(&k1)
            .arg(&b)
            .arg(&avgdl)
            .arg(&mut scores);
        // Safety: threads beyond a query's rows return via the span check.
        unsafe { lb.launch(cfg) }.map_err(cuda_err)?;

        // Device top-k: one block per query; download k pairs per query.
        let f_topk = self.module.load_function("bm25_topk").map_err(cuda_err)?;
        let mut out_docs: CudaSlice<u32> = stream.alloc_zeros(nq * k).map_err(cuda_err)?;
        let mut out_scores: CudaSlice<f32> = stream.alloc_zeros(nq * k).map_err(cuda_err)?;
        let ku = k as u32;
        let cfg = LaunchConfig {
            grid_dim: (nq as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut lb = stream.launch_builder(&f_topk);
        lb.arg(&scores).arg(&num_docs).arg(&ku).arg(&mut out_docs).arg(&mut out_scores);
        // Safety: shared arrays sized for blockDim 256 × MAXK 16; k ≤ 16.
        unsafe { lb.launch(cfg) }.map_err(cuda_err)?;

        let h_docs = stream.clone_dtoh(&out_docs).map_err(cuda_err)?;
        let h_scores = stream.clone_dtoh(&out_scores).map_err(cuda_err)?;
        stream.synchronize().map_err(cuda_err)?;
        Ok((0..nq)
            .map(|q| {
                (0..k)
                    .map(|i| (h_docs[q * k + i], h_scores[q * k + i]))
                    .filter(|&(d, _)| d != u32::MAX)
                    .collect()
            })
            .collect())
    }
}
