// SPDX-License-Identifier: Apache-2.0

//! GPU executors (feature `gpu`; SPEC §11).
//!
//! Kernels are NVRTC-compiled at runtime and `libcuda`/`libnvrtc` are
//! dlopen'd, so no CUDA toolchain is needed at build time and the crate
//! builds empty without the feature (SPEC §3.5).
//!
//! Executors must be **bit-identical** to `lucene-arrow-cpu` on every input
//! (SPEC §12.3) — see `tests/differential.rs`.

#[cfg(feature = "gpu")]
pub mod encode;
#[cfg(feature = "gpu")]
mod executor;
#[cfg(feature = "gpu")]
pub mod knn;
#[cfg(feature = "gpu")]
pub mod bm25;
#[cfg(feature = "gpu")]
pub mod postings_gpu;
#[cfg(feature = "gpu")]
pub mod text_ingest;
#[cfg(feature = "cuvs")]
pub mod cuvs_knn;

#[cfg(feature = "gpu")]
pub use executor::{DeviceData, GpuDecoder, PinnedRing};
