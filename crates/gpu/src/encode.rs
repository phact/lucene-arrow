// SPDX-License-Identifier: Apache-2.0

//! Encode kernels (SPEC §11.7): GPU stats pass + bit-pack, the inverse
//! funnel shift.
//!
//! Pack strategy: one thread per **output** 64-bit word — each word's bits
//! overlap at most `64/bpv + 2` values; the thread gathers them, applies
//! the `(v - base)/gcd` (or table-index) prologue, and composes the word.
//! No atomics, coalesced stores, bit-identical to the CPU `direct::pack`
//! payload.
//!
//! Stats strategy: grid-stride min/max/gcd-of-deltas folded by atomics,
//! plus **exact** ≤256-distinct detection via a 1024-slot open-addressed
//! global hash — exact so the host-side policy (decision register #6:
//! CPU policy / GPU execute) picks the *same* encoding on either
//! executor, keeping whole files byte-identical.

use std::cell::RefCell;
use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;

use crate::executor::{GpuDecoder, cuda_err};
use lucene_arrow_core::{Error, Result};
use lucene_arrow_docvalues::direct;
use lucene_arrow_docvalues::write::{NumericEncoder, NumericStats};

const TABLE_SLOTS: usize = 1024;

const PACK_SRC: &str = r#"
#define SENTINEL 0x8000000000000000LL  // i64::MIN marks an empty table slot
#define TABLE_SLOTS 1024

__device__ inline long long bin_gcd(long long a, long long b) {
    unsigned long long x = a < 0 ? 0ULL - (unsigned long long)a : (unsigned long long)a;
    unsigned long long y = b < 0 ? 0ULL - (unsigned long long)b : (unsigned long long)b;
    if (x == 0) return (long long)y;
    if (y == 0) return (long long)x;
    int shift = __ffsll((long long)(x | y)) - 1;
    x >>= __ffsll((long long)x) - 1;
    while (y != 0) {
        y >>= __ffsll((long long)y) - 1;
        if (x > y) { unsigned long long t = x; x = y; y = t; }
        y -= x;
    }
    return (long long)(x << shift);
}

// Stats pass (SPEC §11.7): min/max/gcd-of-deltas + exact ≤256-distinct
// detection. Policy stays host-side.
extern "C" __global__ void stats_kernel(
    const long long* __restrict__ values,
    unsigned long long n,
    long long* out_min,              // init i64::MAX
    long long* out_max,              // init i64::MIN
    long long* gcd_accum,            // init 0, CAS-merged
    int* out_of_range,               // any v outside [MIN/2, MAX/2]
    long long* table,                // TABLE_SLOTS, init SENTINEL
    int* table_count,
    int* has_sentinel,               // some value == i64::MIN itself
    int* overflow)
{
    long long first = values[0];
    long long lmin = 0x7FFFFFFFFFFFFFFFLL;
    long long lmax = (long long)SENTINEL;
    long long lgcd = 0;
    unsigned long long stride = (unsigned long long)blockDim.x * gridDim.x;
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        long long v = values[i];
        lmin = v < lmin ? v : lmin;
        lmax = v > lmax ? v : lmax;
        if (v < (long long)0xC000000000000000LL || v > (long long)0x3FFFFFFFFFFFFFFFLL) {
            *out_of_range = 1;
        } else {
            lgcd = bin_gcd(lgcd, v - first);
        }
        if (!(*overflow)) {
            if (v == (long long)SENTINEL) {
                *has_sentinel = 1;
            } else {
                unsigned int h = (unsigned int)((((unsigned long long)v)
                                 * 0x9E3779B97F4A7C15ULL) >> 54) & (TABLE_SLOTS - 1);
                for (int probe = 0; probe < TABLE_SLOTS; probe++) {
                    unsigned long long old = atomicCAS(
                        (unsigned long long*)&table[h],
                        (unsigned long long)SENTINEL,
                        (unsigned long long)v);
                    if (old == (unsigned long long)SENTINEL) {
                        if (atomicAdd(table_count, 1) + 1 > 256) *overflow = 1;
                        break;
                    }
                    if ((long long)old == v) break;
                    h = (h + 1) & (TABLE_SLOTS - 1);
                    if (probe == TABLE_SLOTS - 1) *overflow = 1;
                }
            }
        }
    }
    atomicMin(out_min, lmin);
    atomicMax(out_max, lmax);
    long long cur = *gcd_accum;
    while (true) {
        long long merged = bin_gcd(cur, lgcd);
        if (merged == cur) break;
        long long seen = (long long)atomicCAS((unsigned long long*)gcd_accum,
                                              (unsigned long long)cur,
                                              (unsigned long long)merged);
        if (seen == cur) break;
        cur = seen;
    }
}

extern "C" __global__ void pack_words(
    const long long* __restrict__ values,
    unsigned long long n,           // value count
    int bpv,
    long long base,
    long long gcd,
    const long long* __restrict__ table,  // sorted; table_len == 0 → arithmetic
    int table_len,
    unsigned long long* __restrict__ out,
    unsigned long long n_words)
{
    unsigned long long stride = (unsigned long long)blockDim.x * gridDim.x;
    unsigned long long mask = bpv >= 64 ? ~0ull : ((1ull << bpv) - 1);
    for (unsigned long long w = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
         w < n_words; w += stride) {
        unsigned long long lo_bit = w * 64ull;
        unsigned long long first = lo_bit / (unsigned long long)bpv;
        unsigned long long word = 0;
        for (unsigned long long i = first; i < n; i++) {
            unsigned long long bit = i * (unsigned long long)bpv;
            if (bit >= lo_bit + 64ull) break;
            long long v = values[i];
            unsigned long long x;
            if (table_len > 0) {
                int lo = 0, hi = table_len - 1, idx = 0;
                while (lo <= hi) {
                    int mid = (lo + hi) >> 1;
                    if (table[mid] < v) lo = mid + 1;
                    else { idx = mid; hi = mid - 1; }
                }
                x = (unsigned long long)idx;
            } else {
                // Signed subtract/divide — exactly the CPU writer's
                // arithmetic (real plans never wrap).
                x = (unsigned long long)((v - base) / gcd);
            }
            x &= mask;
            long long shift = (long long)bit - (long long)lo_bit;
            if (shift >= 0) {
                word |= x << shift;
            } else {
                word |= x >> (-shift);
            }
        }
        out[w] = word;
    }
}
"#;

/// GPU stats + bit-pack executor. Implements [`NumericEncoder`], so the
/// doc-values writer runs stats+pack on device under the identical
/// host-side policy.
pub struct GpuPacker {
    stream: Arc<CudaStream>,
    _module: Arc<CudaModule>,
    pack: CudaFunction,
    stats: CudaFunction,
    /// A field encodes as stats-then-pack over the same slice — cache the
    /// upload between the two trait calls.
    cached: RefCell<Option<(usize, usize, CudaSlice<i64>)>>,
}

impl GpuPacker {
    pub fn new(decoder: &GpuDecoder) -> Result<Self> {
        let stream = decoder.stream().clone();
        let ptx = compile_ptx(PACK_SRC).map_err(cuda_err)?;
        let module = stream.context().load_module(ptx).map_err(cuda_err)?;
        let pack = module.load_function("pack_words").map_err(cuda_err)?;
        let stats = module.load_function("stats_kernel").map_err(cuda_err)?;
        Ok(GpuPacker { stream, _module: module, pack, stats, cached: RefCell::new(None) })
    }

    pub fn upload_values(&self, values: &[i64]) -> Result<CudaSlice<i64>> {
        self.stream.clone_htod(values).map_err(cuda_err)
    }

    /// Upload and remember (used by `stats`, which always runs first for
    /// a field). Always uploads fresh — a pointer+len key alone is unsound
    /// (allocator reuse / ABA).
    fn upload_and_cache(&self, values: &[i64]) -> Result<CudaSlice<i64>> {
        let dev = self.upload_values(values)?;
        *self.cached.borrow_mut() =
            Some((values.as_ptr() as usize, values.len(), dev.clone()));
        Ok(dev)
    }

    /// Take the cached upload if it matches; pack is the final per-field
    /// call, so the cache is consumed either way.
    fn take_cached(&self, values: &[i64]) -> Result<CudaSlice<i64>> {
        let key = (values.as_ptr() as usize, values.len());
        if let Some((p, l, dev)) = self.cached.borrow_mut().take()
            && (p, l) == key
        {
            return Ok(dev);
        }
        self.upload_values(values)
    }

    /// Device-side stats (SPEC §11.7 stats pass).
    pub fn stats_device(&self, values: &CudaSlice<i64>, n: u64) -> Result<NumericStats> {
        let d_min = self.stream.clone_htod(&[i64::MAX]).map_err(cuda_err)?;
        let d_max = self.stream.clone_htod(&[i64::MIN]).map_err(cuda_err)?;
        let d_gcd = self.stream.clone_htod(&[0i64]).map_err(cuda_err)?;
        let d_oor = self.stream.alloc_zeros::<i32>(1).map_err(cuda_err)?;
        let d_table = self.stream.clone_htod(&vec![i64::MIN; TABLE_SLOTS]).map_err(cuda_err)?;
        let d_count = self.stream.alloc_zeros::<i32>(1).map_err(cuda_err)?;
        let d_sent = self.stream.alloc_zeros::<i32>(1).map_err(cuda_err)?;
        let d_over = self.stream.alloc_zeros::<i32>(1).map_err(cuda_err)?;

        let block = 256u32;
        let cfg = LaunchConfig {
            grid_dim: ((n.div_ceil(block as u64 * 8) as u32).clamp(1, 4096), 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = self.stream.launch_builder(&self.stats);
        launch.arg(values);
        launch.arg(&n);
        launch.arg(&d_min);
        launch.arg(&d_max);
        launch.arg(&d_gcd);
        launch.arg(&d_oor);
        launch.arg(&d_table);
        launch.arg(&d_count);
        launch.arg(&d_sent);
        launch.arg(&d_over);
        unsafe { launch.launch(cfg) }.map_err(cuda_err)?;

        let min = self.stream.clone_dtoh(&d_min).map_err(cuda_err)?[0];
        let max = self.stream.clone_dtoh(&d_max).map_err(cuda_err)?[0];
        let gcd_raw = self.stream.clone_dtoh(&d_gcd).map_err(cuda_err)?[0];
        let oor = self.stream.clone_dtoh(&d_oor).map_err(cuda_err)?[0];
        let overflow = self.stream.clone_dtoh(&d_over).map_err(cuda_err)?[0];
        let has_sent = self.stream.clone_dtoh(&d_sent).map_err(cuda_err)?[0];
        let slots = self.stream.clone_dtoh(&d_table).map_err(cuda_err)?;
        self.stream.synchronize().map_err(cuda_err)?;

        let gcd = if oor != 0 { 1 } else { gcd_raw };
        let table = if overflow != 0 {
            None
        } else {
            let mut t: Vec<i64> = slots.into_iter().filter(|&v| v != i64::MIN).collect();
            if has_sent != 0 {
                t.push(i64::MIN);
            }
            if t.len() > 256 {
                None
            } else {
                t.sort_unstable();
                Some(t)
            }
        };
        Ok(NumericStats { min, max, gcd, table })
    }

    pub fn pack_device(
        &self,
        values: &CudaSlice<i64>,
        n: u64,
        bpv: u8,
        base: i64,
        gcd: i64,
    ) -> Result<CudaSlice<u64>> {
        self.pack_device_table(values, n, bpv, base, gcd, None)
    }

    pub fn pack_device_table(
        &self,
        values: &CudaSlice<i64>,
        n: u64,
        bpv: u8,
        base: i64,
        gcd: i64,
        table: Option<&[i64]>,
    ) -> Result<CudaSlice<u64>> {
        if !direct::SUPPORTED_BITS_PER_VALUE.contains(&bpv) {
            return Err(Error::invalid(format!("unsupported bpv {bpv}")));
        }
        if gcd == 0 {
            return Err(Error::invalid("gcd must be non-zero"));
        }
        let n_words = (n * bpv as u64).div_ceil(64).max(1);
        let mut out = self.stream.alloc_zeros::<u64>(n_words as usize).map_err(cuda_err)?;
        let table_vec = table.unwrap_or(&[]);
        let table_len = table_vec.len() as i32;
        let d_table = self
            .stream
            .clone_htod(if table_vec.is_empty() { &[0i64][..] } else { table_vec })
            .map_err(cuda_err)?;

        let block = 256u32;
        let cfg = LaunchConfig {
            grid_dim: ((n_words.div_ceil(block as u64) as u32).clamp(1, 65_535), 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let bpv_i = bpv as i32;
        let mut launch = self.stream.launch_builder(&self.pack);
        launch.arg(values);
        launch.arg(&n);
        launch.arg(&bpv_i);
        launch.arg(&base);
        launch.arg(&gcd);
        launch.arg(&d_table);
        launch.arg(&table_len);
        launch.arg(&mut out);
        launch.arg(&n_words);
        unsafe { launch.launch(cfg) }.map_err(cuda_err)?;
        Ok(out)
    }

    fn words_to_payload(&self, words: &CudaSlice<u64>, count: usize, bpv: u8) -> Result<Vec<u8>> {
        let host_words = self.stream.clone_dtoh(words).map_err(cuda_err)?;
        self.stream.synchronize().map_err(cuda_err)?;
        let payload_len = direct::packed_len(count, bpv);
        let mut out = Vec::with_capacity(payload_len + 8);
        for w in &host_words {
            out.extend_from_slice(&w.to_le_bytes());
        }
        // alloc_zeros guarantees zero tail bits, so truncate/resize is
        // exactly DirectWriter's zero padding.
        out.truncate(payload_len);
        out.resize(payload_len, 0);
        Ok(out)
    }

    /// Pack on GPU, return the exact CPU-identical payload bytes.
    pub fn pack_to_host(&self, values: &[i64], bpv: u8, base: i64, gcd: i64) -> Result<Vec<u8>> {
        let dev = self.take_cached(values)?;
        let words = self.pack_device(&dev, values.len() as u64, bpv, base, gcd)?;
        self.words_to_payload(&words, values.len(), bpv)
    }

    pub fn synchronize(&self) -> Result<()> {
        self.stream.synchronize().map_err(cuda_err)
    }
}

impl NumericEncoder for GpuPacker {
    fn stats(&self, values: &[i64]) -> Result<NumericStats> {
        let dev = self.upload_and_cache(values)?;
        self.stats_device(&dev, values.len() as u64)
    }

    fn pack(
        &self,
        values: &[i64],
        bpv: u8,
        base: i64,
        gcd: i64,
        table: Option<&[i64]>,
    ) -> Result<Vec<u8>> {
        let dev = self.take_cached(values)?;
        let words = self.pack_device_table(&dev, values.len() as u64, bpv, base, gcd, table)?;
        self.words_to_payload(&words, values.len(), bpv)
    }
}
