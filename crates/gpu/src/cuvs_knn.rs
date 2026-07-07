// SPDX-License-Identifier: Apache-2.0

//! cuVS-backed ANN (SPEC §11.6, P6a; resolves decision register #14 via
//! the `cuvs` crate against the pixi-provided `libcuvs`).
//!
//! Two engines over the same decoded vectors: brute force (exact — the
//! grown-up version of our `FlatKnn`) and CAGRA (GPU-built graph, the
//! P6 build input). v1 hands host-side copies to cuVS via dlpack tensors;
//! true zero-copy device handoff (wrapping our `CudaSlice` in a dlpack
//! descriptor) is a follow-up — the buffers are already device-resident.

use cuvs::{ManagedTensor, Resources};

fn check(e: cuvs_sys::cuvsError_t) -> Result<()> {
    if e == cuvs_sys::cuvsError_t::CUVS_SUCCESS {
        Ok(())
    } else {
        // Safety: cuVS keeps a thread-local error string.
        let text = unsafe {
            let p = cuvs_sys::cuvsGetLastErrorText();
            if p.is_null() {
                "<no error text>".to_string()
            } else {
                std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
            }
        };
        Err(Error::Codec(format!("cuvs ffi error: {e:?}: {text}")))
    }
}

use crate::knn::Hit;
use lucene_arrow_core::{Error, Result};

fn cuvs_err(e: cuvs::Error) -> Error {
    Error::Codec(format!("cuvs: {e}"))
}

/// One-time cuVS handle.
pub struct CuvsContext {
    res: Resources,
}

impl CuvsContext {
    pub fn new() -> Result<Self> {
        Ok(CuvsContext { res: Resources::new().map_err(cuvs_err)? })
    }

    /// Exact top-k via cuVS brute force.
    pub fn brute_force(
        &self,
        vectors: &[f32],
        dim: usize,
        queries: &[f32],
        k: usize,
    ) -> Result<Vec<Vec<Hit>>> {
        use cuvs::brute_force::Index;
        use cuvs::distance_type::DistanceType;
        let n = vectors.len() / dim;
        let nq = queries.len() / dim;
        let dataset = ndarray::Array2::from_shape_vec((n, dim), vectors.to_vec())
            .map_err(|e| Error::invalid(e.to_string()))?;
        // Brute-force build requires the dataset in device memory.
        let dataset_dev =
            ManagedTensor::from(&dataset).to_device(&self.res).map_err(cuvs_err)?;
        // L2Unexpanded: direct (a-b)² — the expanded form catastrophically
        // cancels for near-duplicate vectors (norms ≫ distances), which an
        // *exact* engine must not do (our FlatKnn caught this).
        let index = Index::build(&self.res, DistanceType::L2Unexpanded, None, dataset_dev)
            .map_err(cuvs_err)?;

        // Brute force reports i64 neighbor ids (CAGRA uses u32).
        let q = ndarray::Array2::from_shape_vec((nq, dim), queries.to_vec())
            .map_err(|e| Error::invalid(e.to_string()))?;
        let q_dev = ManagedTensor::from(&q).to_device(&self.res).map_err(cuvs_err)?;
        let mut neighbors_host = ndarray::Array2::<i64>::zeros((nq, k));
        let neighbors =
            ManagedTensor::from(&neighbors_host).to_device(&self.res).map_err(cuvs_err)?;
        let mut distances_host = ndarray::Array2::<f32>::zeros((nq, k));
        let distances =
            ManagedTensor::from(&distances_host).to_device(&self.res).map_err(cuvs_err)?;
        index.search(&self.res, &q_dev, &neighbors, &distances).map_err(cuvs_err)?;
        neighbors.to_host(&self.res, &mut neighbors_host).map_err(cuvs_err)?;
        distances.to_host(&self.res, &mut distances_host).map_err(cuvs_err)?;
        Ok((0..nq)
            .map(|qi| {
                (0..k)
                    .map(|j| Hit {
                        ord: neighbors_host[[qi, j]] as u32,
                        score: -distances_host[[qi, j]],
                    })
                    .collect()
            })
            .collect())
    }

    /// CAGRA: build the graph on GPU, search it. Returns hits plus wall
    /// times of build and search (the §15 comparison data).
    pub fn cagra(
        &self,
        vectors: &[f32],
        dim: usize,
        queries: &[f32],
        k: usize,
    ) -> Result<(Vec<Vec<Hit>>, std::time::Duration, std::time::Duration)> {
        use cuvs::cagra::{Index, IndexParams, SearchParams};
        let n = vectors.len() / dim;
        let nq = queries.len() / dim;
        let dataset = ndarray::Array2::from_shape_vec((n, dim), vectors.to_vec())
            .map_err(|e| Error::invalid(e.to_string()))?;
        let t = std::time::Instant::now();
        let params = IndexParams::new().map_err(cuvs_err)?;
        let index = Index::build(&self.res, &params, &dataset).map_err(cuvs_err)?;
        let build = t.elapsed();
        let sp = SearchParams::new().map_err(cuvs_err)?;
        let t = std::time::Instant::now();
        let hits = self.search_common(nq, dim, queries, k, |q, nb, ds| {
            index.search(&self.res, &sp, q, nb, ds).map_err(cuvs_err)
        })?;
        Ok((hits, build, t.elapsed()))
    }

    /// GPU k-NN graph via NN-Descent (the algorithm under CAGRA), written
    /// straight into a host `n × degree` u32 matrix — the input for the
    /// Lucene `.vem`/`.vex` writer (P6c).
    pub fn knn_graph(
        &self,
        vectors: &[f32],
        dim: usize,
        degree: usize,
    ) -> Result<(Vec<u32>, std::time::Duration)> {
        use cuvs_sys as ffi;
        let n = vectors.len() / dim;
        let dataset = ndarray::Array2::from_shape_vec((n, dim), vectors.to_vec())
            .map_err(|e| Error::invalid(e.to_string()))?;
        let graph_host = ndarray::Array2::<u32>::zeros((n, degree));

        let t = std::time::Instant::now();
        let dataset_t = ManagedTensor::from(&dataset);
        let graph_t = ManagedTensor::from(&graph_host);
        // Safety: C API with valid handles; graph tensor is host-resident
        // (n × degree u32) exactly as cuvsNNDescentBuild documents.
        unsafe {
            let mut params: ffi::cuvsNNDescentIndexParams_t = std::ptr::null_mut();
            check(ffi::cuvsNNDescentIndexParamsCreate(&mut params))?;
            (*params).graph_degree = degree;
            (*params).intermediate_graph_degree = degree * 3 / 2;
            let mut index: ffi::cuvsNNDescentIndex_t = std::ptr::null_mut();
            check(ffi::cuvsNNDescentIndexCreate(&mut index))?;
            let rc = ffi::cuvsNNDescentBuild(
                self.res.0,
                params,
                dataset_t.as_ptr(),
                std::ptr::null_mut(), // graph fetched via GetGraph below
                index,
            );
            let rc2 = if rc == ffi::cuvsError_t::CUVS_SUCCESS {
                ffi::cuvsNNDescentIndexGetGraph(self.res.0, index, graph_t.as_ptr())
            } else {
                rc
            };
            let _ = ffi::cuvsNNDescentIndexDestroy(index);
            let _ = ffi::cuvsNNDescentIndexParamsDestroy(params);
            check(rc)?;
            check(rc2)?;
        }
        let build = t.elapsed();
        Ok((graph_host.into_raw_vec(), build))
    }

    fn search_common(
        &self,
        nq: usize,
        dim: usize,
        queries: &[f32],
        k: usize,
        run: impl Fn(&ManagedTensor, &ManagedTensor, &ManagedTensor) -> Result<()>,
    ) -> Result<Vec<Vec<Hit>>> {
        let q = ndarray::Array2::from_shape_vec((nq, dim), queries.to_vec())
            .map_err(|e| Error::invalid(e.to_string()))?;
        let q_dev = ManagedTensor::from(&q).to_device(&self.res).map_err(cuvs_err)?;
        let mut neighbors_host = ndarray::Array2::<u32>::zeros((nq, k));
        let neighbors = ManagedTensor::from(&neighbors_host).to_device(&self.res).map_err(cuvs_err)?;
        let mut distances_host = ndarray::Array2::<f32>::zeros((nq, k));
        let distances = ManagedTensor::from(&distances_host).to_device(&self.res).map_err(cuvs_err)?;

        run(&q_dev, &neighbors, &distances)?;

        neighbors.to_host(&self.res, &mut neighbors_host).map_err(cuvs_err)?;
        distances.to_host(&self.res, &mut distances_host).map_err(cuvs_err)?;
        Ok((0..nq)
            .map(|qi| {
                (0..k)
                    .map(|j| Hit {
                        ord: neighbors_host[[qi, j]],
                        // cuVS reports L2 distance; our FlatKnn scores are
                        // negated squared distance — normalize for callers.
                        score: -distances_host[[qi, j]],
                    })
                    .collect()
            })
            .collect())
    }
}
