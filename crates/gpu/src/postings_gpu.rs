// SPDX-License-Identifier: Apache-2.0

//! GPU decode of Lucene103 postings doc blocks (SPEC §7.8 + §11).
//!
//! The CPU plans: it walks the terms dict and the level-0/level-1 skip
//! stream to emit one descriptor per 128-doc packed block — crucially the
//! skip data carries each block's *base* (previous last doc id), so every
//! block decodes independently and the whole `.doc` file fans out across
//! the GPU. Kernel strategy: one thread per block running the scalar
//! ForUtil unpack (parallelism comes from tens of thousands of blocks,
//! not intra-block lanes). Tails (<128 docs) stay on CPU.

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

use crate::executor::cuda_err;

use crate::executor::GpuDecoder;
use lucene_arrow_core::cursor::Cursor;
use lucene_arrow_core::{Error, Result};
use lucene_arrow_postings::walk::{FieldTraits, TermMeta, walk_terms};

pub const BLOCK_SIZE: usize = 128;

/// One 128-doc packed block. `bpv` > 0: FOR; == 0: consecutive;
/// < 0: bitset of `-bpv` longs.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DocBlockDesc {
    pub src_off: u64,
    pub dst_off: u64,
    pub base: i32,
    pub bpv: i32,
}

/// Plan every packed doc block of a field. Returns descriptors plus the
/// total number of packed docs (tails are not planned).
pub fn plan_doc_blocks(
    tim: &[u8],
    doc_file: &[u8],
    root_block_fp: u64,
    traits: FieldTraits,
) -> Result<Vec<DocBlockDesc>> {
    fn vint15(c: &mut Cursor) -> Result<i32> {
        let s = c.le_i16()?;
        if s >= 0 { Ok(s as i32) } else { Ok((s as i32 & 0x7FFF) | (c.vint()? << 15)) }
    }
    fn skip_pfor(c: &mut Cursor) -> Result<()> {
        let token = c.u8()? as u32;
        let bpv = token & 0x1F;
        let ex = token >> 5;
        if bpv == 0 {
            c.vlong()?;
            c.skip((ex * 2) as usize)?;
        } else {
            c.skip((bpv as usize) * 16 + (ex * 2) as usize)?;
        }
        Ok(())
    }

    let mut descs = Vec::new();
    let mut dst = 0u64;
    walk_terms(tim, root_block_fp, traits, |_t, df, _ttf, tm: TermMeta| {
        if df < BLOCK_SIZE as u32 {
            return Ok(());
        }
        let mut c = Cursor::at(doc_file, tm.doc_start_fp as usize);
        let mut remaining = df as usize;
        let mut consumed = 0usize;
        let mut prev = -1i32;
        while remaining >= BLOCK_SIZE {
            if consumed.is_multiple_of(4096) && remaining >= 4096 {
                let _ = c.vint()?;
                let _ = c.vlong()?;
                if traits.has_freqs {
                    let span = c.le_i16()? as u16 as usize;
                    c.seek(c.pos() + span)?;
                }
            }
            let nb = c.vlong()? as usize;
            let end = c.pos() + nb;
            let level0_last = prev + vint15(&mut c)?;
            c.seek(end)?;
            let bpv = c.u8()? as i8;
            descs.push(DocBlockDesc {
                src_off: c.pos() as u64,
                dst_off: dst,
                base: prev,
                bpv: bpv as i32,
            });
            if bpv > 0 {
                c.skip(bpv as usize * 16)?;
            } else if bpv < 0 {
                c.skip((-bpv) as usize * 8)?;
            }
            if traits.has_freqs {
                skip_pfor(&mut c)?;
            }
            dst += BLOCK_SIZE as u64;
            prev = level0_last;
            remaining -= BLOCK_SIZE;
            consumed += BLOCK_SIZE;
        }
        Ok(())
    })?;
    Ok(descs)
}

const KERNEL: &str = r#"
typedef unsigned int u32; typedef unsigned long long u64;
typedef long long i64; typedef int i32;

struct Desc { u64 src_off; u64 dst_off; i32 base; i32 bpv; };

__device__ u32 mask_for(u32 prim, u32 b) {
    u32 v = (b >= 32) ? 0xFFFFFFFFu : ((1u << b) - 1u);
    if (prim == 8)  return v * 0x01010101u;
    if (prim == 16) return v * 0x00010001u;
    return v;
}

extern "C" __global__ void decode_doc_blocks(
    const unsigned char* __restrict__ doc, const Desc* __restrict__ descs,
    u32 num_descs, u32* __restrict__ out)
{
    u32 g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= num_descs) return;
    Desc d = descs[g];
    u32* dst = out + d.dst_off;

    if (d.bpv == 0) {                    // consecutive
        for (int i = 0; i < 128; i++) dst[i] = (u32)(d.base + 1 + i);
        return;
    }
    if (d.bpv < 0) {                     // bitset over base+1
        int nl = -d.bpv; int n = 0; i32 base = d.base + 1;
        const u64* words = (const u64*)(doc + d.src_off);
        for (int w = 0; w < nl; w++) {
            u64 word; memcpy(&word, words + w, 8);
            while (word) {
                int bit = __ffsll((long long)word) - 1;
                dst[n++] = (u32)(base + w * 64 + bit);
                word &= word - 1;
            }
        }
        return;
    }

    // FOR: scalar ForUtil unpack (one thread owns the whole block).
    u32 bpv = (u32)d.bpv;
    u32 prim = bpv <= 3 ? 8u : (bpv <= 10 ? 16u : 32u);
    u32 nips = bpv * 4;
    i32 ints[128];
    i32 tmp[128];
    const unsigned char* src = doc + d.src_off;
    for (u32 i = 0; i < nips; i++) {
        u32 w; memcpy(&w, src + i * 4, 4);
        tmp[i] = (i32)w;
    }
    u32 mask = (bpv == prim) ? 0xFFFFFFFFu : mask_for(prim, bpv);
    u32 idx = 0;
    i32 shift = (i32)(prim - bpv);
    for (; shift >= 0; shift -= (i32)bpv)
        for (u32 i = 0; i < nips; i++)
            ints[idx++] = (i32)(((u32)tmp[i] >> shift) & mask);
    u32 num_collapsed = 128u * prim / 32u;
    u32 rem = (u32)(shift + (i32)bpv);
    if (rem > 0 && idx < num_collapsed) {
        u32 mask_full = mask_for(prim, rem);
        u32 ti = 0; u32 remaining_bits = rem;
        while (idx < num_collapsed) {
            i32 b = (i32)bpv - (i32)remaining_bits;
            i32 l = (i32)(((u32)tmp[ti] & mask_for(prim, remaining_bits)) << b);
            ti++;
            while (b >= (i32)rem) {
                b -= (i32)rem;
                l |= (i32)(((u32)tmp[ti] & mask_full) << b);
                ti++;
            }
            if (b > 0) {
                l |= (i32)(((u32)tmp[ti] >> ((i32)rem - b)) & mask_for(prim, (u32)b));
                remaining_bits = rem - (u32)b;
            } else {
                remaining_bits = rem;
            }
            ints[idx++] = l;
        }
    }
    // prefix sums (wrapping is native on GPU ints)
    if (bpv <= 3) {
        i32 s = 0;
        for (int i = 0; i < 32; i++) { s += ints[i]; ints[i] = s; }
        for (int i = 31; i >= 0; i--) {
            i32 l = ints[i];
            ints[i]      = (l >> 24) & 0xFF;
            ints[32 + i] = (l >> 16) & 0xFF;
            ints[64 + i] = (l >> 8) & 0xFF;
            ints[96 + i] = l & 0xFF;
        }
        i32 l0 = d.base, l1 = l0 + ints[31], l2 = l1 + ints[63], l3 = l2 + ints[95];
        for (int i = 0; i < 32; i++) {
            ints[i] += l0; ints[32+i] += l1; ints[64+i] += l2; ints[96+i] += l3;
        }
    } else if (bpv <= 10) {
        i32 s = 0;
        for (int i = 0; i < 64; i++) { s += ints[i]; ints[i] = s; }
        for (int i = 63; i >= 0; i--) {
            i32 l = ints[i];
            ints[i] = (l >> 16) & 0xFFFF;
            ints[64 + i] = l & 0xFFFF;
        }
        i32 l0 = d.base, l1 = d.base + ints[63];
        for (int i = 0; i < 64; i++) { ints[i] += l0; ints[64+i] += l1; }
    } else {
        i32 s = d.base;
        for (int i = 0; i < 128; i++) { s += ints[i]; ints[i] = s; }
    }
    for (int i = 0; i < 128; i++) dst[i] = (u32)ints[i];
}
"#;

/// Decode all planned blocks on-device; returns the docs column
/// (device-resident) with one 128-slot span per descriptor.
pub struct GpuPostings {
    module: std::sync::Arc<cudarc::driver::CudaModule>,
}

impl GpuPostings {
    pub fn new(gpu: &GpuDecoder) -> Result<Self> {
        let ptx = cudarc::nvrtc::compile_ptx(KERNEL)
            .map_err(|e| Error::Codec(format!("nvrtc: {e}")))?;
        let module = gpu.stream().context().load_module(ptx).map_err(cuda_err)?;
        Ok(GpuPostings { module })
    }

    pub fn decode_blocks(
        &self,
        gpu: &GpuDecoder,
        doc_dev: &crate::executor::DeviceData,
        descs: &[DocBlockDesc],
    ) -> Result<CudaSlice<u32>> {
        let doc_dev = doc_dev.cuda_slice();
        let stream = gpu.stream().clone();
        let f = self.module.load_function("decode_doc_blocks").map_err(cuda_err)?;
        // Safety: DocBlockDesc is #[repr(C)] plain-old-data.
        let raw: &[u8] = unsafe {
            std::slice::from_raw_parts(
                descs.as_ptr() as *const u8,
                std::mem::size_of_val(descs),
            )
        };
        let descs_dev: CudaSlice<u8> = stream.clone_htod(raw).map_err(cuda_err)?;
        let n = descs.len() as u32;
        let mut out: CudaSlice<u32> =
            stream.alloc_zeros(descs.len() * BLOCK_SIZE).map_err(cuda_err)?;
        let cfg = LaunchConfig {
            grid_dim: (n.div_ceil(128), 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = stream.launch_builder(&f);
        b.arg(doc_dev).arg(&descs_dev).arg(&n).arg(&mut out);
        // Safety: kernel bounds-checked on num_descs; buffers sized above.
        unsafe { b.launch(cfg) }.map_err(cuda_err)?;
        stream.synchronize().map_err(cuda_err)?;
        Ok(out)
    }

    /// Fetch a decoded docs column back to host.
    pub fn download(&self, gpu: &GpuDecoder, dev: &CudaSlice<u32>) -> Result<Vec<u32>> {
        gpu.stream().clone_dtoh(dev).map_err(cuda_err)
    }
}
