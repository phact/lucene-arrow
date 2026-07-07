// SPDX-License-Identifier: Apache-2.0
//
// Scale test for the GPU rebuild-merge at realistic embedding dimensions.
// Writes SRC source jVector files of N/SRC vectors each (dim DIM), then
// times the merge pipeline: mmap+parallel READ (CPU bswap -> host floats)
// vs the CAGRA rebuild. Prints the read fraction so we can tell whether a
// fused GPU extract (upload raw bytes -> GPU bswap -> CAGRA on-device)
// would actually move the needle.
//
// Env: DIM (768), N (2_000_000), SRC (4). Run with --features cuvs.

#[cfg(feature = "cuvs")]
use std::time::Instant;

#[cfg(feature = "cuvs")]
use lucene_arrow_gpu::GpuDecoder;
#[cfg(feature = "cuvs")]
use lucene_arrow_gpu::cuvs_knn::CuvsContext;
#[cfg(feature = "cuvs")]
use lucene_arrow_vectors::hnsw::parse_hnswlib;
#[cfg(feature = "cuvs")]
use lucene_arrow_vectors::jvector::{l0_layout, read_vectors_file, write_index, write_index_multi};

#[cfg(feature = "cuvs")]
fn env(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn main() {
    #[cfg(not(feature = "cuvs"))]
    {
        eprintln!("needs --features cuvs");
    }
    #[cfg(feature = "cuvs")]
    {
        let dim = env("DIM", 1536);
        let n = env("N", 2_000_000);
        let src = env("SRC", 4);
        let per = n / src;
        let n = per * src;
        let Ok(ctx) = CuvsContext::new() else { return eprintln!("cuVS unavailable") };
        let Ok(gpu) = GpuDecoder::new() else { return eprintln!("no CUDA device") };

        let dir = std::env::temp_dir().join("la_merge_scale");
        std::fs::create_dir_all(&dir).unwrap();
        eprintln!(
            "scale: {src} files × {per} vectors × {dim}d = {n} total ({:.1} GB on disk)",
            (n * dim * 4) as f64 / 1e9
        );

        // --- write source files (setup, not timed as part of the merge) ---
        let mut files = Vec::new();
        for s in 0..src {
            let p = dir.join(format!("src{s}.jvector"));
            if !p.exists() {
                let vecs: Vec<f32> = (0..per * dim)
                    .map(|i| {
                        let h = ((s * per * dim + i) as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                        (h >> 11) as f32 / (1u64 << 53) as f32 * 2.0 - 1.0
                    })
                    .collect();
                let ring: Vec<Vec<u32>> =
                    (0..per).map(|i| vec![((i + 1) % per) as u32]).collect();
                std::fs::write(&p, write_index(&vecs, dim, &ring, 0).unwrap()).unwrap();
            }
            files.push(p);
        }

        // --- READ (cold): mmap + parallel bswap -> host floats, disk pages
        //     faulted in on first touch. This is IO + CPU bswap combined. ---
        let t = Instant::now();
        let mut all: Vec<f32> = Vec::with_capacity(n * dim);
        for p in &files {
            let (v, _) = read_vectors_file(p).unwrap();
            all.extend_from_slice(&v);
        }
        let read_cold_s = t.elapsed().as_secs_f64();

        // --- READ (warm): pages now in the page cache, so this is the pure
        //     CPU-bswap + memory-bandwidth cost — the part a fused GPU
        //     extract actually competes with (disk IO it can't remove without
        //     GPUDirect Storage). ---
        let t = Instant::now();
        let mut warm: Vec<f32> = Vec::with_capacity(n * dim);
        for p in &files {
            let (v, _) = read_vectors_file(p).unwrap();
            warm.extend_from_slice(&v);
        }
        let read_warm_s = t.elapsed().as_secs_f64();
        drop(warm);

        // --- HOST REBUILD: CAGRA -> HNSW (host floats uploaded internally) ---
        let hf = dir.join("merged.hnsw");
        let t = Instant::now();
        ctx.cagra_to_hnswlib(&all, dim, 16, 100, hf.to_str().unwrap()).unwrap();
        let build_host_s = t.elapsed().as_secs_f64();
        let parsed = parse_hnswlib(&std::fs::read(&hf).unwrap()).unwrap();
        let _ = write_index_multi(&all, dim, &parsed).unwrap();
        drop(all); // free ~n*dim*4 host bytes before the fused pass

        // --- FUSED: gather on GPU (upload raw + bswap kernel) then feed CAGRA
        //     a device pointer — no CPU bswap, no host float buffer. ---
        let raws: Vec<Vec<u8>> = files.iter().map(|p| std::fs::read(p).unwrap()).collect();
        let layouts: Vec<(&[u8], usize, usize, usize)> = raws
            .iter()
            .map(|r| {
                let (h, rec, _d, nn) = l0_layout(r).unwrap();
                (r.as_slice(), h, rec, nn)
            })
            .collect();
        let t = Instant::now();
        let dev = gpu.gather_be_f32_multi(&layouts, dim).unwrap();
        let gather_s = t.elapsed().as_secs_f64();
        let hf2 = dir.join("merged_fused.hnsw");
        let t = Instant::now();
        let ptr = gpu.device_ptr_f32(&dev);
        // Safety: dev outlives the call; same primary context; gather synced.
        unsafe { ctx.cagra_to_hnswlib_device(ptr, n, dim, 16, 100, hf2.to_str().unwrap()).unwrap() };
        let build_dev_s = t.elapsed().as_secs_f64();
        drop(dev);

        let gb = (n * dim * 4) as f64 / 1e9;
        let host_total = read_cold_s + build_host_s;
        let fused_total = gather_s + build_dev_s;
        println!();
        println!("merge {n} × {dim}d ({gb:.1} GB of vectors):");
        println!("  [host]  CPU read (mmap+bswap) : {read_cold_s:6.3} s  ({:5.1} GB/s)", gb / read_cold_s);
        println!("          (warm read, cache hot): {read_warm_s:6.3} s  ({:5.1} GB/s)", gb / read_warm_s);
        println!("          CAGRA build+write     : {build_host_s:6.3} s");
        println!("          total                 : {host_total:6.3} s");
        println!("  [fused] GPU gather (upload+bswap): {gather_s:6.3} s  ({:5.1} GB/s)", gb / gather_s);
        println!("          CAGRA build (on device)  : {build_dev_s:6.3} s");
        println!("          total                    : {fused_total:6.3} s");
        println!();
        println!(
            "fused vs host: {:.2}× on the merge; extract {:.3}s → {:.3}s ({:.1}× the read)",
            host_total / fused_total,
            read_cold_s,
            gather_s,
            read_cold_s / gather_s
        );
    }
}
