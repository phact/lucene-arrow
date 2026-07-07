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
}
