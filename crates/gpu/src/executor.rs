// SPDX-License-Identifier: Apache-2.0

//! Fused single-pass numeric decode on the GPU (SPEC §11.3).
//!
//! One kernel launch decodes *all* blocks of a column via a device-side
//! descriptor table (SPEC §11.2 — the cuDF batched-fragment trick; per-block
//! launches are an anti-goal). Every DirectWriter width reduces to "value
//! `i` occupies bits `[i·bpv, (i+1)·bpv)` of a little-endian bitstream"
//! (byte-aligned widths trivially; 1/2/4 pack LSB-first into LE longs;
//! 12/20/28 pack pairs on byte boundaries, so `2·(i/2)·bpv + (i%2)·bpv =
//! i·bpv`), so a single funnel-shift kernel handles them all: two aligned
//! 64-bit loads, shift, mask, fused epilogue (`base + gcd·x` or table
//! gather), one store.
//!
//! The data buffer is uploaded with 8 bytes of slack so the `hi` word load
//! never faults at the tail. DISI validity stays on the CPU for now
//! (bitmap decode is metadata-scale; the value payload is the bulk).

use std::sync::Arc;

use arrow_array::{ArrayRef, Float64Array, Int64Array};
use arrow_buffer::NullBuffer;
use arrow_schema::DataType;
use cudarc::driver::{
    CudaContext, CudaEvent, CudaFunction, CudaModule, CudaStream, DevicePtrMut, DeviceRepr,
    LaunchConfig, PushKernelArg, result as cu,
};
use cudarc::nvrtc::compile_ptx;

use lucene_arrow_core::plan::{BlockDecode, Coverage, DecodePlan};
use lucene_arrow_core::{Error, Result};
use lucene_arrow_docvalues::disi;

const KERNEL_SRC: &str = r#"
struct BlockDesc {
    unsigned long long src_off;   // byte offset of packed payload in data
    unsigned long long dst_off;   // first output value index
    unsigned long long count;     // values in this block
    unsigned long long mask;      // (1<<bpv)-1, ~0 for 64, unused for bpv==0
    long long base;               // epilogue addend (or constant fill)
    long long gcd;                // epilogue multiplier
    int bpv;                      // 0 => constant fill with base
    int table_off;                // >=0 => out = tables[table_off + x]
    int table_len;
    int _pad;
};

// doc_idx: when use_scatter != 0, value i lands at out[doc_idx[dst_off+i]]
// — the SPEC §11.4 fused scatter (sparse columns write straight to their
// doc slots; dead lanes never exist because we only iterate values).
extern "C" __global__ void decode_blocks(
    const unsigned char* __restrict__ data,
    const BlockDesc* __restrict__ descs,
    int n_descs,
    const long long* __restrict__ tables,
    const int* __restrict__ doc_idx,
    int use_scatter,
    long long* __restrict__ out)
{
    for (int di = blockIdx.y; di < n_descs; di += gridDim.y) {
        BlockDesc d = descs[di];
        unsigned long long stride = (unsigned long long)blockDim.x * gridDim.x;
        for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
             i < d.count; i += stride) {
            long long v;
            if (d.bpv == 0) {
                v = d.base;
            } else {
                unsigned long long bit = d.src_off * 8ull + i * (unsigned long long)d.bpv;
                // data is 8-byte aligned (device alloc) and padded, so
                // aligned u64 loads + funnel shift are safe everywhere.
                const unsigned long long* words = (const unsigned long long*)data;
                unsigned long long widx = bit >> 6;
                unsigned int shift = (unsigned int)(bit & 63ull);
                unsigned long long lo = words[widx] >> shift;
                unsigned long long hi = shift ? (words[widx + 1] << (64u - shift)) : 0ull;
                unsigned long long x = (lo | hi) & d.mask;
                if (d.table_off >= 0) {
                    v = tables[d.table_off + (x < (unsigned long long)d.table_len
                                                  ? x : (unsigned long long)(d.table_len - 1))];
                } else {
                    // unsigned arithmetic: defined two's-complement wrap,
                    // matches the CPU executor's wrapping ops exactly.
                    v = (long long)((unsigned long long)d.base
                                    + (unsigned long long)d.gcd * x);
                }
            }
            unsigned long long j = d.dst_off + i;
            out[use_scatter ? (unsigned long long)doc_idx[j] : j] = v;
        }
    }
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct BlockDesc {
    src_off: u64,
    dst_off: u64,
    count: u64,
    mask: u64,
    base: i64,
    gcd: i64,
    bpv: i32,
    table_off: i32,
    table_len: i32,
    _pad: i32,
}

unsafe impl DeviceRepr for BlockDesc {}

/// A CUDA context + compiled decode module. Create once, decode many.
pub struct GpuDecoder {
    stream: Arc<CudaStream>,
    _module: Arc<CudaModule>,
    decode_blocks: CudaFunction,
}

pub(crate) fn cuda_err(e: impl std::fmt::Display) -> Error {
    Error::Codec(format!("cuda: {e}"))
}

impl GpuDecoder {
    /// Fails cleanly when no CUDA device/driver is present — callers use
    /// that as the "skip GPU path" signal (SPEC §3.5).
    pub fn new() -> Result<Self> {
        let ctx = CudaContext::new(0).map_err(cuda_err)?;
        let ptx = compile_ptx(KERNEL_SRC).map_err(cuda_err)?;
        let module = ctx.load_module(ptx).map_err(cuda_err)?;
        let decode_blocks = module.load_function("decode_blocks").map_err(cuda_err)?;
        Ok(GpuDecoder { stream: ctx.default_stream(), _module: module, decode_blocks })
    }

    /// Upload a data file (or extent) once for repeated decodes. Adds the
    /// 8-byte tail slack the kernel's `hi` load needs.
    pub fn upload(&self, bytes: &[u8]) -> Result<DeviceData> {
        let mut padded = Vec::with_capacity(bytes.len() + 8);
        padded.extend_from_slice(bytes);
        padded.extend_from_slice(&[0u8; 8]);
        let dev = self.stream.clone_htod(&padded).map_err(cuda_err)?;
        Ok(DeviceData { dev, len: bytes.len() as u64 })
    }

    /// Decode the packed values of `plan` from device-resident data into a
    /// device buffer, returning it without copying back. The building block
    /// for both [`decode_numeric`](Self::decode_numeric) and the benches
    /// (SPEC §11.0: raw kernel throughput is measured device-to-device).
    pub fn decode_values_device(
        &self,
        plan: &DecodePlan,
        data: &DeviceData,
    ) -> Result<cudarc::driver::CudaSlice<i64>> {
        self.decode_values_device_scattered(plan, data, None)
    }

    /// Like [`decode_values_device`](Self::decode_values_device), but with
    /// an optional fused scatter: `doc_idx[i]` is the output slot of value
    /// `i`, and the output buffer is sized `out_len` (doc count), zeroed
    /// slots meaning "no value" (SPEC §11.4 — the scatter rides the decode
    /// store, no separate pass).
    pub fn decode_values_device_scattered(
        &self,
        plan: &DecodePlan,
        data: &DeviceData,
        scatter: Option<(&cudarc::driver::CudaSlice<i32>, usize)>,
    ) -> Result<cudarc::driver::CudaSlice<i64>> {
        let mut descs = Vec::with_capacity(plan.blocks.len());
        let mut tables: Vec<i64> = Vec::new();
        let mut dst = 0u64;

        for block in &plan.blocks {
            let (offset, len) = block.byte_range();
            if offset + len > data.len {
                return Err(Error::corrupt(format!(
                    "block [{offset}, {}) beyond uploaded data of {} bytes",
                    offset + len,
                    data.len
                )));
            }
            let count = block.value_count();
            let desc = match *block {
                BlockDecode::Direct { offset, bit_width, values, .. } => {
                    desc(offset, dst, values, bit_width, 0, 1, -1, 0)
                }
                BlockDecode::DeltaPacked { offset, bit_width, base, values, .. } => {
                    desc(offset, dst, values, bit_width, base, 1, -1, 0)
                }
                BlockDecode::GcdPacked { offset, bit_width, base, gcd, values, .. } => {
                    desc(offset, dst, values, bit_width, base, gcd, -1, 0)
                }
                BlockDecode::Table { offset, bit_width, ref table, values, .. } => {
                    let table_off = tables.len() as i32;
                    tables.extend_from_slice(table);
                    desc(offset, dst, values, bit_width, 0, 1, table_off, table.len() as i32)
                }
                BlockDecode::Monotonic { .. } | BlockDecode::Ordinals { .. } | BlockDecode::Raw { .. } => {
                    return Err(Error::unsupported("block kind lands with P2/P3"));
                }
            };
            descs.push(desc);
            dst += count;
        }
        if dst != plan.num_values {
            return Err(Error::corrupt("plan value count mismatch"));
        }

        if let Some((idx, out_len)) = scatter
            && ((idx.len() as u64) < plan.num_values || out_len < plan.num_values as usize) {
                return Err(Error::invalid("scatter index/output undersized"));
            }
        let out_len = scatter.map_or(plan.num_values as usize, |(_, n)| n);
        let mut out = self.stream.alloc_zeros::<i64>(out_len.max(1)).map_err(cuda_err)?;
        if descs.is_empty() {
            return Ok(out);
        }
        if tables.is_empty() {
            tables.push(0); // kernel arg must be a live allocation
        }
        let d_descs = self.stream.clone_htod(&descs).map_err(cuda_err)?;
        let d_tables = self.stream.clone_htod(&tables).map_err(cuda_err)?;
        let dummy_idx;
        let (d_idx, use_scatter): (&cudarc::driver::CudaSlice<i32>, i32) = match scatter {
            Some((idx, _)) => (idx, 1),
            None => {
                dummy_idx = self.stream.alloc_zeros::<i32>(1).map_err(cuda_err)?;
                (&dummy_idx, 0)
            }
        };

        let max_count = descs.iter().map(|d| d.count).max().unwrap_or(0);
        let block_dim = 256u32;
        let grid_x = (max_count.div_ceil(block_dim as u64) as u32).clamp(1, 4096);
        let grid_y = (descs.len() as u32).min(65_535);
        let cfg = LaunchConfig {
            grid_dim: (grid_x, grid_y, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_descs = descs.len() as i32;
        let mut launch = self.stream.launch_builder(&self.decode_blocks);
        launch.arg(&data.dev);
        launch.arg(&d_descs);
        launch.arg(&n_descs);
        launch.arg(&d_tables);
        launch.arg(d_idx);
        launch.arg(&use_scatter);
        launch.arg(&mut out);
        unsafe { launch.launch(cfg) }.map_err(cuda_err)?;
        Ok(out)
    }

    /// Full numeric decode: GPU value kernel with fused scatter; the DISI
    /// bitmap itself decodes on the CPU (metadata-scale — the doc-index
    /// build is one popcount walk). Must produce arrays bit-identical to
    /// `lucene_arrow_cpu::decode_numeric`.
    pub fn decode_numeric(&self, plan: &DecodePlan, dvd: &[u8]) -> Result<ArrayRef> {
        let data = self.upload(dvd)?;

        let (num_docs, validity, values) = match &plan.coverage {
            Coverage::Dense { num_docs } => {
                let out = self.decode_values_device(plan, &data)?;
                let values = self.stream.clone_dtoh(&out).map_err(cuda_err)?;
                self.stream.synchronize().map_err(cuda_err)?;
                if values.len() != *num_docs as usize {
                    return Err(Error::corrupt("dense column value/doc count mismatch"));
                }
                (*num_docs, None, values)
            }
            Coverage::Empty { num_docs } => (
                *num_docs,
                Some(vec![0u64; (*num_docs as usize).div_ceil(64)]),
                vec![0i64; *num_docs as usize],
            ),
            Coverage::Sparse { num_docs, disi: d } => {
                let end = d.offset + d.len;
                if end as usize > dvd.len() {
                    return Err(Error::corrupt("DISI range beyond data file"));
                }
                let region = &dvd[d.offset as usize..end as usize];
                let bitmap = disi::decode(region, d.num_values, *num_docs, d.dense_rank_power)?;
                let mut doc_idx = Vec::with_capacity(d.num_values as usize);
                for (w, &word) in bitmap.iter().enumerate() {
                    let mut bits = word;
                    while bits != 0 {
                        doc_idx.push((w * 64 + bits.trailing_zeros() as usize) as i32);
                        bits &= bits - 1;
                    }
                }
                let d_idx = self.stream.clone_htod(&doc_idx).map_err(cuda_err)?;
                let out = self.decode_values_device_scattered(
                    plan,
                    &data,
                    Some((&d_idx, *num_docs as usize)),
                )?;
                let values = self.stream.clone_dtoh(&out).map_err(cuda_err)?;
                self.stream.synchronize().map_err(cuda_err)?;
                (*num_docs, Some(bitmap), values)
            }
        };

        let nulls = validity.map(|words| {
            NullBuffer::new(arrow_buffer::BooleanBuffer::new(
                arrow_buffer::Buffer::from_vec(words),
                0,
                num_docs as usize,
            ))
        });
        match plan.arrow_type {
            DataType::Int64 => Ok(Arc::new(Int64Array::new(values.into(), nulls)) as ArrayRef),
            DataType::Float64 => {
                let floats: Vec<f64> = values.iter().map(|&v| f64::from_bits(v as u64)).collect();
                Ok(Arc::new(Float64Array::new(floats.into(), nulls)) as ArrayRef)
            }
            ref other => Err(Error::unsupported(format!("numeric decode into {other}"))),
        }
    }

    pub fn synchronize(&self) -> Result<()> {
        self.stream.synchronize().map_err(cuda_err)
    }

    /// The stream all executor work runs on (shared by the KNN scorer).
    pub(crate) fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Device-resident copy of a flat-vector plan's payload (the single
    /// `Raw` block, ord-ordered packed vectors). This is the SPEC §11.6
    /// path: no kernel, just DMA — the returned buffer is what gets handed
    /// to cuVS/cuBLAS zero-copy in the P2 demo.
    pub fn vector_payload_device(
        &self,
        plan: &DecodePlan,
        data: &DeviceData,
    ) -> Result<cudarc::driver::CudaSlice<u8>> {
        match plan.blocks.as_slice() {
            [] => self.stream.alloc_zeros::<u8>(0).map_err(cuda_err),
            [BlockDecode::Raw { offset, len }] => {
                if offset + len > data.len {
                    return Err(Error::corrupt("raw block beyond uploaded data"));
                }
                let src = data
                    .dev
                    .slice(*offset as usize..(*offset + *len) as usize);
                let mut dst = self.stream.alloc_zeros::<u8>(*len as usize).map_err(cuda_err)?;
                self.stream.memcpy_dtod(&src, &mut dst).map_err(cuda_err)?;
                Ok(dst)
            }
            _ => Err(Error::invalid("vector plan must be a single Raw block")),
        }
    }

    /// Copy a device buffer back to host (test/verification helper).
    pub fn download(&self, dev: &cudarc::driver::CudaSlice<u8>) -> Result<Vec<u8>> {
        let out = self.stream.clone_dtoh(dev).map_err(cuda_err)?;
        self.stream.synchronize().map_err(cuda_err)?;
        Ok(out)
    }

    /// Upload through a write-combined pinned staging buffer — one-shot
    /// form kept for comparison benches; the ring below is the real path.
    pub fn upload_via_pinned(&self, bytes: &[u8]) -> Result<DeviceData> {
        let ctx = self.stream.context();
        // Safety: fully initialized via copy_from_slice before any read.
        let mut pinned =
            unsafe { ctx.alloc_pinned::<u8>(bytes.len() + 8) }.map_err(cuda_err)?;
        {
            let host = pinned.as_mut_slice().map_err(cuda_err)?;
            host[..bytes.len()].copy_from_slice(bytes);
            host[bytes.len()..].fill(0);
        }
        let dev = self.stream.clone_htod(&pinned).map_err(cuda_err)?;
        self.stream.synchronize().map_err(cuda_err)?;
        Ok(DeviceData { dev, len: bytes.len() as u64 })
    }

    /// Create a pinned staging ring: `depth` cacheable page-locked buffers
    /// of `chunk_bytes` (SPEC §11.2 — always pinned, always async, never
    /// pageable; 2× extent × depth is the intended sizing).
    ///
    /// cudarc's own `alloc_pinned` hardcodes write-combined memory, which
    /// makes the *host-side* memcpy into the buffer the bottleneck
    /// (~3 GB/s measured); plain cacheable pinned memory keeps host copies
    /// at memory speed while DMA runs just as fast — hence the raw driver
    /// calls here.
    pub fn new_pinned_ring(&self, chunk_bytes: usize, depth: usize) -> Result<PinnedRing> {
        let ctx = self.stream.context().clone();
        let mut bufs = Vec::with_capacity(depth);
        for _ in 0..depth {
            // Safety: sized allocation, freed in Drop; flags=0 → cacheable.
            let ptr = unsafe { cu::malloc_host(chunk_bytes, 0) }.map_err(cuda_err)? as *mut u8;
            let event = ctx
                .new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_BLOCKING_SYNC))
                .map_err(cuda_err)?;
            bufs.push((ptr, event));
        }
        Ok(PinnedRing { bufs, chunk_bytes, _ctx: ctx })
    }

    /// Chunked upload through the ring: the host memcpy of chunk `i+1`
    /// overlaps the async DMA of chunk `i` (per-buffer events gate reuse).
    pub fn upload_pipelined(&self, bytes: &[u8], ring: &PinnedRing) -> Result<DeviceData> {
        let mut dev = self.stream.alloc_zeros::<u8>(bytes.len() + 8).map_err(cuda_err)?;
        {
            let (dptr, _sync_on_drop) = dev.device_ptr_mut(&self.stream);
            let depth = ring.bufs.len();
            for (i, chunk) in bytes.chunks(ring.chunk_bytes).enumerate() {
                let (ptr, event) = &ring.bufs[i % depth];
                // Wait until the DMA that last read this buffer finished.
                event.synchronize().map_err(cuda_err)?;
                // Safety: buffer is chunk_bytes long; chunk fits by
                // construction; no DMA in flight per the event sync.
                let staged = unsafe {
                    std::ptr::copy_nonoverlapping(chunk.as_ptr(), *ptr, chunk.len());
                    std::slice::from_raw_parts(*ptr, chunk.len())
                };
                let off = (i * ring.chunk_bytes) as u64;
                // Safety: dst range is within dev (len+8); staged is pinned
                // and outlives the copy (event-gated).
                unsafe { cu::memcpy_htod_async(dptr + off, staged, self.stream.cu_stream()) }
                    .map_err(cuda_err)?;
                event.record(&self.stream).map_err(cuda_err)?;
            }
        }
        self.stream.synchronize().map_err(cuda_err)?;
        Ok(DeviceData { dev, len: bytes.len() as u64 })
    }
}

/// Cacheable pinned staging buffers with per-buffer reuse events.
pub struct PinnedRing {
    bufs: Vec<(*mut u8, CudaEvent)>,
    chunk_bytes: usize,
    _ctx: Arc<CudaContext>,
}

// Safety: the raw pointers are exclusively owned page-locked allocations;
// access is gated by CudaEvents.
unsafe impl Send for PinnedRing {}

impl Drop for PinnedRing {
    fn drop(&mut self) {
        for (ptr, event) in &self.bufs {
            let _ = event.synchronize();
            // Safety: allocated via malloc_host in new_pinned_ring.
            unsafe {
                let _ = cu::free_host(*ptr as *mut std::ffi::c_void);
            }
        }
    }
}

/// Device-resident copy of one data file (or extent), tail-padded.
pub struct DeviceData {
    dev: cudarc::driver::CudaSlice<u8>,
    len: u64,
}

impl DeviceData {
    pub(crate) fn cuda_slice(&self) -> &cudarc::driver::CudaSlice<u8> {
        &self.dev
    }
}

#[allow(clippy::too_many_arguments)]
fn desc(
    src_off: u64,
    dst_off: u64,
    count: u64,
    bpv: u8,
    base: i64,
    gcd: i64,
    table_off: i32,
    table_len: i32,
) -> BlockDesc {
    BlockDesc {
        src_off,
        dst_off,
        count,
        mask: if bpv >= 64 { u64::MAX } else { (1u64 << bpv) - 1 },
        base,
        gcd,
        bpv: bpv as i32,
        table_off,
        table_len,
        _pad: 0,
    }
}
