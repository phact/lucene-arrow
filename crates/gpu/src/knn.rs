// SPDX-License-Identifier: Apache-2.0

//! Exact (flat) KNN over device-resident vectors — the P2 demo shape
//! (SPEC §13 P2, §11.6): decoded `.vec` payload stays on device; scoring
//! is one memory-bound kernel; no HNSW graph anywhere.
//!
//! This is deliberately *not* an ANN library: cuVS/cuBLAS own that job
//! (decision register #14). What this proves is the zero-copy handoff —
//! the same device buffer `vector_payload_device` returns is what a cuVS
//! integration would consume. Scores are computed on GPU; top-k selection
//! happens host-side (k ≪ n; the score dtoh is 4 bytes/vector).

use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use std::sync::Arc;

use crate::executor::{GpuDecoder, cuda_err};
use lucene_arrow_core::{Error, Result};
use lucene_arrow_vectors::Similarity;

const KNN_SRC: &str = r#"
// One thread per (query, vector) pair, queries in blockIdx.y.
// metric: 0 = euclidean (negated squared distance, larger = closer),
//         1 = dot product / max-inner-product,
//         2 = cosine.
extern "C" __global__ void score_flat_f32(
    const float* __restrict__ vectors,   // n × dim, ord-ordered
    const float* __restrict__ queries,   // q × dim
    unsigned long long n,
    int dim,
    int metric,
    float* __restrict__ scores)          // q × n
{
    int qi = blockIdx.y;
    const float* query = queries + (unsigned long long)qi * dim;
    unsigned long long stride = (unsigned long long)blockDim.x * gridDim.x;
    for (unsigned long long v = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
         v < n; v += stride) {
        const float* vec = vectors + v * (unsigned long long)dim;
        float acc = 0.0f, vv = 0.0f;
        if (metric == 0) {
            for (int k = 0; k < dim; k++) {
                float d = vec[k] - query[k];
                acc += d * d;
            }
            acc = -acc;
        } else {
            for (int k = 0; k < dim; k++) {
                acc += vec[k] * query[k];
                vv  += vec[k] * vec[k];
            }
            if (metric == 2) acc = acc * rsqrtf(vv > 0.0f ? vv : 1.0f);
        }
        scores[(unsigned long long)qi * n + v] = acc;
    }
}
"#;

/// One KNN hit: vector ordinal (== ord in the flat storage; map to docid
/// via the column's DISI/ord→doc when sparse) and its score.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hit {
    pub ord: u32,
    pub score: f32,
}

/// Flat exact-KNN scorer over a device-resident f32 vector payload.
pub struct FlatKnn {
    stream: Arc<CudaStream>,
    _module: Arc<CudaModule>,
    score: CudaFunction,
}

impl FlatKnn {
    pub fn new(decoder: &GpuDecoder) -> Result<Self> {
        let stream = decoder.stream().clone();
        let ptx = compile_ptx(KNN_SRC).map_err(cuda_err)?;
        let module = stream.context().load_module(ptx).map_err(cuda_err)?;
        let score = module.load_function("score_flat_f32").map_err(cuda_err)?;
        Ok(FlatKnn { stream, _module: module, score })
    }

    /// Exact top-k for each query against `n` device-resident vectors
    /// (`payload` = ord-ordered f32s, as returned by
    /// `GpuDecoder::vector_payload_device`).
    pub fn search(
        &self,
        payload: &CudaSlice<u8>,
        n: u64,
        dim: u32,
        similarity: Similarity,
        queries: &[f32],
        k: usize,
    ) -> Result<Vec<Vec<Hit>>> {
        if dim == 0 || !queries.len().is_multiple_of(dim as usize) {
            return Err(Error::invalid("queries length must be a multiple of dim"));
        }
        let nq = queries.len() / dim as usize;
        if nq == 0 || n == 0 {
            return Ok(vec![Vec::new(); nq]);
        }
        if payload.len() as u64 != n * dim as u64 * 4 {
            return Err(Error::invalid("payload size != n × dim × 4"));
        }
        let metric: i32 = match similarity {
            Similarity::Euclidean => 0,
            Similarity::DotProduct | Similarity::MaximumInnerProduct => 1,
            Similarity::Cosine => 2,
        };

        let d_queries = self.stream.clone_htod(queries).map_err(cuda_err)?;
        let mut d_scores =
            self.stream.alloc_zeros::<f32>(nq * n as usize).map_err(cuda_err)?;

        let block_dim = 256u32;
        let cfg = LaunchConfig {
            grid_dim: ((n.div_ceil(block_dim as u64) as u32).clamp(1, 4096), nq as u32, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };
        let dim_i = dim as i32;
        let mut launch = self.stream.launch_builder(&self.score);
        launch.arg(payload);
        launch.arg(&d_queries);
        launch.arg(&n);
        launch.arg(&dim_i);
        launch.arg(&metric);
        launch.arg(&mut d_scores);
        unsafe { launch.launch(cfg) }.map_err(cuda_err)?;

        let scores = self.stream.clone_dtoh(&d_scores).map_err(cuda_err)?;
        self.stream.synchronize().map_err(cuda_err)?;

        // Host-side top-k per query (k ≪ n).
        let mut results = Vec::with_capacity(nq);
        for q in 0..nq {
            let row = &scores[q * n as usize..(q + 1) * n as usize];
            let mut hits: Vec<Hit> =
                row.iter().enumerate().map(|(o, &s)| Hit { ord: o as u32, score: s }).collect();
            let k = k.min(hits.len());
            hits.select_nth_unstable_by(k - 1, |a, b| {
                b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
            });
            hits.truncate(k);
            hits.sort_by(|a, b| {
                b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
            });
            results.push(hits);
        }
        Ok(results)
    }
}
