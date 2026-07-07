// SPDX-License-Identifier: Apache-2.0

//! The P2 demo (SPEC §13): "point at a shard, GPU KNN, no JVM."
//!
//! Opens a real Lucene segment directory (default: the Java-written golden
//! vectors index), plans the flat vector column, DMA's the payload to the
//! 5090, and runs exact KNN — verifying the top-k against a CPU
//! brute-force pass.
//!
//! Usage: cargo run -p lucene-arrow-gpu --features gpu --example gpu_knn \
//!        [segment-dir] [field] [k]

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("build with --features gpu");
}

#[cfg(feature = "gpu")]
fn main() {
    use lucene_arrow_codec::SegmentDirectory;
    use lucene_arrow_core::plan::Coverage;
    use lucene_arrow_docvalues::disi;
    use lucene_arrow_gpu::{GpuDecoder, knn::FlatKnn};
    use lucene_arrow_vectors::read::{VecField, plan_vectors};

    let mut args = std::env::args().skip(1);
    let dir_path = args.next().unwrap_or_else(|| {
        concat!(env!("CARGO_MANIFEST_DIR"), "/../../harness/golden/vectors").to_string()
    });
    let field_name = args.next().unwrap_or_else(|| "emb".to_string());
    let k: usize = args.next().map(|s| s.parse().expect("k")).unwrap_or(5);

    let dir = SegmentDirectory::open(&dir_path).expect("open segment directory");
    let seg = &dir.segments()[0];
    println!("segment {} ({} docs, codec {})", seg.name, seg.max_doc, seg.codec);

    let fields: Vec<VecField> = seg
        .fields
        .iter()
        .filter(|f| f.has_vectors)
        .map(|f| VecField { number: f.number as i32, name: f.name.clone() })
        .collect();
    let vemf_name = seg.files.iter().find(|f| f.ends_with(".vemf")).expect(".vemf");
    let vec_name = seg.files.iter().find(|f| f.ends_with(".vec")).expect(".vec");
    let vemf = dir.open_input(&seg.name, vemf_name).unwrap();
    let vec_file = dir.open_input(&seg.name, vec_name).unwrap();
    let vemf = vemf.slice(0, vemf.len()).unwrap();
    let vec_bytes = vec_file.slice(0, vec_file.len()).unwrap();

    let plans = plan_vectors(vemf, &fields, seg.max_doc as u32, vec_name).unwrap();
    let vp = plans.iter().find(|p| p.plan.column.name == field_name).expect("field");
    println!(
        "field {:?}: {} vectors × dim {}, {:?}, similarity {}",
        field_name,
        vp.count,
        vp.dim,
        vp.encoding,
        vp.similarity.as_str()
    );

    // ord → docid map (identity when dense).
    let ord_to_doc: Vec<u32> = match &vp.plan.coverage {
        Coverage::Dense { .. } => (0..vp.count as u32).collect(),
        Coverage::Sparse { num_docs, disi: d } => {
            let region = &vec_bytes[d.offset as usize..(d.offset + d.len) as usize];
            let bitmap = disi::decode(region, d.num_values, *num_docs, d.dense_rank_power).unwrap();
            let mut docs = Vec::with_capacity(vp.count as usize);
            for (w, &word) in bitmap.iter().enumerate() {
                let mut bits = word;
                while bits != 0 {
                    docs.push(w as u32 * 64 + bits.trailing_zeros());
                    bits &= bits - 1;
                }
            }
            docs
        }
        Coverage::Empty { .. } => Vec::new(),
    };

    // Device path: upload .vec once, DMA payload slice, score on GPU.
    let gpu = GpuDecoder::new().expect("CUDA device");
    let data = gpu.upload(vec_bytes).unwrap();
    let payload = gpu.vector_payload_device(&vp.plan, &data).unwrap();
    let knn = FlatKnn::new(&gpu).unwrap();

    // Query = the stored vector of the first doc that has one (so rank 1
    // must be that doc itself), plus a perturbed copy.
    let host_payload = gpu.download(&payload).unwrap();
    let dim = vp.dim as usize;
    let first: Vec<f32> = host_payload[..dim * 4]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let mut queries = first.clone();
    queries.extend(first.iter().map(|v| v + 0.1));

    let t = std::time::Instant::now();
    let results = knn.search(&payload, vp.count, vp.dim, vp.similarity, &queries, k).unwrap();
    let dt = t.elapsed();

    // CPU verification for query 0.
    let scores_cpu: Vec<f32> = (0..vp.count as usize)
        .map(|o| {
            let v = &host_payload[o * dim * 4..(o + 1) * dim * 4];
            -v.chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .zip(&first)
                .map(|(x, q)| (x - q) * (x - q))
                .sum::<f32>()
        })
        .collect();
    let mut cpu_best: Vec<u32> = (0..vp.count as u32).collect();
    cpu_best.sort_by(|&a, &b| {
        scores_cpu[b as usize].partial_cmp(&scores_cpu[a as usize]).unwrap()
    });
    assert_eq!(
        results[0].iter().map(|h| h.ord).collect::<Vec<_>>(),
        cpu_best[..k].to_vec(),
        "GPU top-k differs from CPU brute force"
    );

    println!("\ntop-{k} per query ({} vectors scored in {dt:?}):", vp.count);
    for (qi, hits) in results.iter().enumerate() {
        print!("  q{qi}:");
        for h in hits {
            print!("  doc {} (score {:.3})", ord_to_doc[h.ord as usize], h.score);
        }
        println!();
    }
    println!("\nGPU top-k == CPU brute force ✓  (no JVM anywhere in this process)");
}
