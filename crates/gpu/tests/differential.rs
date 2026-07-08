// SPDX-License-Identifier: Apache-2.0

//! Differential gate (SPEC §12.3): GPU and CPU executors bit-identical on
//! every plan. Skips gracefully when no CUDA device is present.

#![cfg(feature = "gpu")]

use lucene_arrow_cpu::decode_numeric as cpu_decode;
use lucene_arrow_docvalues::file::DocValuesFileBuilder;
use lucene_arrow_docvalues::read::{DvField, DvKind, plan_doc_values};
use lucene_arrow_gpu::GpuDecoder;

fn decoder() -> Option<GpuDecoder> {
    match GpuDecoder::new() {
        Ok(d) => Some(d),
        Err(e) => {
            eprintln!("skipping GPU differential tests: {e}");
            None
        }
    }
}

fn assert_gpu_matches_cpu(gpu: &GpuDecoder, per_doc: &[Option<i64>]) {
    let max_doc = per_doc.len() as u32;
    let (docs, values): (Vec<u32>, Vec<i64>) = per_doc
        .iter()
        .enumerate()
        .filter_map(|(d, v)| v.map(|v| (d as u32, v)))
        .unzip();
    let mut builder = DocValuesFileBuilder::new(b"segmentid0123456", "");
    builder.add_numeric(0, &docs, &values, max_doc).unwrap();
    let (dvm, dvd) = builder.finish();

    let fields =
        [DvField { number: 0, name: "f".into(), kind: DvKind::Numeric, has_skip_index: false }];
    let plans = plan_doc_values(&dvm, &dvd, &fields, max_doc, "_0.dvd").unwrap();
    let plan = &plans.plans[0];

    let cpu = cpu_decode(plan, &dvd).unwrap();
    let dev = gpu.decode_numeric(plan, &dvd).unwrap();
    assert_eq!(cpu.as_ref(), dev.as_ref(), "GPU array differs from CPU reference");
}

#[test]
fn gpu_matches_cpu_across_shapes() {
    let Some(gpu) = decoder() else { return };

    // Every encoding the writer emits, plus width variety via value ranges.
    let shapes: Vec<Vec<Option<i64>>> = vec![
        (0..10_000).map(Some).collect(),                       // direct
        (0..10_000).map(|i| Some(1_000_000 + i * 25)).collect(),      // gcd
        (0..10_000).map(|i| Some([7i64, -9, 1 << 45][i as usize % 3])).collect(), // table
        vec![Some(42); 5_000],                                        // constant
        (0..10_000).map(|i| Some((i - 5_000) * 0x0123_4567)).collect(), // wide/negative
        (0..30_000).map(|i| if i % 7 == 0 { Some(i * 3) } else { None }).collect(), // sparse
        (0..80_000).map(|i| if i % 4 != 3 { Some(i) } else { None }).collect(), // dense DISI
        (0..1).map(Some).collect(),                                   // single value
    ];
    for per_doc in &shapes {
        assert_gpu_matches_cpu(&gpu, per_doc);
    }

    // Bit-width sweep: max value pinned to each width boundary.
    for bits in [1u32, 2, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 63] {
        let max = if bits == 63 { i64::MAX } else { (1i64 << bits) - 1 };
        let per_doc: Vec<Option<i64>> =
            (0..4_099).map(|i| Some((i * 2_654_435_761i64) & max)).collect();
        assert_gpu_matches_cpu(&gpu, &per_doc);
    }
}

/// Golden segments (real Java Lucene, incl. multi-block) if generated.
#[test]
fn gpu_matches_cpu_on_java_golden_segments() {
    let Some(gpu) = decoder() else { return };
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../harness/golden");
    if !root.join("expected.json").exists() {
        eprintln!("skipping: harness/golden not generated");
        return;
    }

    for case in ["numerics", "multiblock", "deletes", "multisegment"] {
        let dir = lucene_arrow_codec::SegmentDirectory::open(root.join(case)).unwrap();
        for seg in dir.segments() {
            let fields: Vec<DvField> = seg
                .fields
                .iter()
                .filter(|f| f.doc_values == lucene_arrow_codec::DocValuesKind::Numeric)
                .map(|f| DvField {
                    number: f.number as i32,
                    name: f.name.clone(),
                    kind: DvKind::Numeric,
                    has_skip_index: false,
                })
                .collect();
            let dvm_name = seg.files.iter().find(|f| f.ends_with(".dvm")).unwrap();
            let dvd_name = seg.files.iter().find(|f| f.ends_with(".dvd")).unwrap();
            let dvm_r = dir.open_input(&seg.name, dvm_name).unwrap();
            let dvd_r = dir.open_input(&seg.name, dvd_name).unwrap();
            let dvm = dvm_r.slice(0, dvm_r.len()).unwrap();
            let dvd = dvd_r.slice(0, dvd_r.len()).unwrap();

            let plans = plan_doc_values(dvm, dvd, &fields, seg.max_doc as u32, dvd_name).unwrap();
            for plan in &plans.plans {
                let cpu = cpu_decode(plan, dvd).unwrap();
                let dev = gpu.decode_numeric(plan, dvd).unwrap();
                assert_eq!(
                    cpu.as_ref(),
                    dev.as_ref(),
                    "{case}/{}/{}: GPU differs from CPU",
                    seg.name,
                    plan.column.name
                );
            }
        }
    }
}

/// Vector DMA path: device-resident payload must equal the CPU-visible
/// `Raw` range byte-for-byte (SPEC §11.6 — no kernel, just a copy).
#[test]
fn gpu_vector_payload_matches_raw_bytes() {
    let Some(gpu) = decoder() else { return };

    let dim = 64u32;
    let max_doc = 5_000u32;
    let docs: Vec<u32> = (0..max_doc).filter(|d| d % 3 != 1).collect();
    let payload: Vec<u8> = docs
        .iter()
        .flat_map(|&d| {
            (0..dim).flat_map(move |k| {
                (((d as usize * 31 + k as usize * 7) % 1009) as f32 * 0.25 - 100.0).to_le_bytes()
            })
        })
        .collect();

    let mut builder = lucene_arrow_vectors::file::VectorsFileBuilder::new(b"segmentid0123456", "");
    builder
        .add_field(
            0,
            lucene_arrow_vectors::VectorEncoding::Float32,
            lucene_arrow_vectors::Similarity::Euclidean,
            dim,
            &docs,
            &payload,
            max_doc,
        )
        .unwrap();
    let (vemf, vec) = builder.finish();

    let fields = [lucene_arrow_vectors::read::VecField { number: 0, name: "emb".into() }];
    let plans = lucene_arrow_vectors::read::plan_vectors(&vemf, &fields, max_doc, "_0.vec").unwrap();

    let data = gpu.upload(&vec).unwrap();
    let dev_payload = gpu.vector_payload_device(&plans[0].plan, &data).unwrap();
    assert_eq!(gpu.download(&dev_payload).unwrap(), payload);
}

/// Global-ordinal remap (SPEC §7.3 dict=global): the OrdinalMap table
/// rides the Table-gather epilogue — GPU keys must equal CPU keys.
#[test]
fn gpu_global_ordinal_remap_matches_cpu() {
    let Some(gpu) = decoder() else { return };
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../harness/golden");
    if !root.join("keywords").is_dir() {
        eprintln!("skipping: harness/golden not generated");
        return;
    }
    let dir = lucene_arrow_codec::SegmentDirectory::open(root.join("keywords")).unwrap();

    let mut per_seg = Vec::new();
    for seg in dir.segments() {
        let fields: Vec<DvField> = seg
            .fields
            .iter()
            .filter(|f| f.doc_values != lucene_arrow_codec::DocValuesKind::None)
            .map(|f| DvField {
                number: f.number as i32,
                name: f.name.clone(),
                kind: match f.doc_values {
                    lucene_arrow_codec::DocValuesKind::Sorted => DvKind::Sorted,
                    lucene_arrow_codec::DocValuesKind::SortedSet => DvKind::SortedSet,
                    lucene_arrow_codec::DocValuesKind::SortedNumeric => DvKind::SortedNumeric,
                    lucene_arrow_codec::DocValuesKind::Numeric => DvKind::Numeric,
                    lucene_arrow_codec::DocValuesKind::Binary => DvKind::Binary,
                    lucene_arrow_codec::DocValuesKind::None => unreachable!(),
                },
                has_skip_index: f.has_skip_index,
            })
            .collect();
        let dvm_name = seg.files.iter().find(|f| f.ends_with(".dvm")).unwrap();
        let dvd_name = seg.files.iter().find(|f| f.ends_with(".dvd")).unwrap();
        let dvm_r = dir.open_input(&seg.name, dvm_name).unwrap();
        let dvd_r = dir.open_input(&seg.name, dvd_name).unwrap();
        let dvm = dvm_r.slice(0, dvm_r.len()).unwrap().to_vec();
        let dvd = dvd_r.slice(0, dvd_r.len()).unwrap().to_vec();
        let plans = plan_doc_values(&dvm, &dvd, &fields, seg.max_doc as u32, dvd_name).unwrap();
        per_seg.push((plans, dvd));
    }

    let dicts: Vec<_> = per_seg
        .iter()
        .map(|(plans, dvd)| {
            let cat = plans.sorted.iter().find(|p| p.ords.column.name == "cat").unwrap();
            lucene_arrow_docvalues::terms::materialize(&cat.terms, dvd).unwrap()
        })
        .collect();
    let map = lucene_arrow_docvalues::ordmap::build(&dicts.iter().collect::<Vec<_>>()).unwrap();

    for (seg_ord, (plans, dvd)) in per_seg.iter().enumerate() {
        let cat = plans.sorted.iter().find(|p| p.ords.column.name == "cat").unwrap();
        let plan = lucene_arrow_docvalues::ordmap::apply_remap(&cat.ords, &map.remap[seg_ord]).unwrap();
        let cpu = cpu_decode(&plan, dvd).unwrap();
        let dev = gpu.decode_numeric(&plan, dvd).unwrap();
        assert_eq!(cpu.as_ref(), dev.as_ref(), "segment {seg_ord} global keys differ");
    }
}

/// Encode differential (SPEC §11.7 / §6): the GPU packer must produce
/// byte-identical payloads to the CPU `direct::pack` for the same plan,
/// across every width and awkward count.
#[test]
fn gpu_pack_matches_cpu_pack() {
    let Some(gpu) = decoder() else { return };
    let packer = lucene_arrow_gpu::encode::GpuPacker::new(&gpu).unwrap();

    for &bpv in &lucene_arrow_docvalues::direct::SUPPORTED_BITS_PER_VALUE {
        for count in [1usize, 2, 3, 63, 64, 65, 4_099, 100_000] {
            let mask = if bpv == 64 { u64::MAX } else { (1u64 << bpv) - 1 };
            // Real plans never overflow gcd·x + base; keep x small enough.
            let (base, gcd) = if bpv >= 56 { (0i64, 1i64) } else { (12_345i64, 25i64) };
            let x_cap = (i64::MAX as u64 / gcd as u64).saturating_sub(base as u64).min(mask);
            let values: Vec<i64> = (0..count)
                .map(|i| {
                    let x = ((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) & mask).min(x_cap) as i64;
                    base + gcd * ((x as u64 & mask) as i64)
                })
                .collect();

            let mut cpu = Vec::new();
            let encoded: Vec<i64> = values.iter().map(|&v| (v - base) / gcd).collect();
            lucene_arrow_docvalues::direct::pack(&encoded, bpv, &mut cpu);

            let dev = packer.pack_to_host(&values, bpv, base, gcd).unwrap();
            assert_eq!(dev, cpu, "bpv={bpv} count={count}");
        }
    }
}

/// Whole-file encoder differential: segments encoded with the GpuPacker
/// (GPU stats + GPU pack, shared host policy) must be byte-identical to
/// the CPU encoder across every encoding mode.
#[test]
fn gpu_encoder_files_match_cpu_files() {
    let Some(gpu) = decoder() else { return };
    let packer = lucene_arrow_gpu::encode::GpuPacker::new(&gpu).unwrap();
    let cpu = lucene_arrow_docvalues::write::CpuEncoder;

    let shapes: Vec<(&str, Vec<Option<i64>>)> = vec![
        ("gcd", (0..50_000).map(|i| Some(1_000_000 + i * 24)).collect()),
        ("table", (0..50_000).map(|i| Some([3i64, -9, 1 << 40, 77][i as usize % 4])).collect()),
        ("constant", vec![Some(5); 10_000]),
        ("wide", (0..50_000).map(|i| Some((i - 25_000) * 0x0123_4567_89AB)).collect()),
        ("sparse", (0..90_000).map(|i| (i % 3 != 1).then_some(i * 7)).collect()),
        ("with-min", {
            let mut v: Vec<Option<i64>> = (0..1000).map(Some).collect();
            v[500] = Some(i64::MIN); // sentinel edge: forces out-of-range gcd path
            v
        }),
    ];

    for (label, per_doc) in shapes {
        let max_doc = per_doc.len() as u32;
        let (docs, values): (Vec<u32>, Vec<i64>) = per_doc
            .iter()
            .enumerate()
            .filter_map(|(d, v)| v.map(|v| (d as u32, v)))
            .unzip();

        let a = lucene_arrow_docvalues::write::encode_numeric_field_with(
            &cpu, 0, &docs, &values, max_doc, 4096,
        )
        .unwrap();
        let b = lucene_arrow_docvalues::write::encode_numeric_field_with(
            &packer, 0, &docs, &values, max_doc, 4096,
        )
        .unwrap();
        assert_eq!(a.meta, b.meta, "{label}: .dvm entry differs");
        assert_eq!(a.data, b.data, "{label}: .dvd payload differs");
    }
}

/// Zero-copy lane differential: dense-chunked encode on the GPU packer ==
/// flat encode on the CPU reference, whole field bytes (meta + data).
#[test]
fn chunked_dense_encode_matches_cpu_flat() {
    let Ok(gpu) = lucene_arrow_gpu::GpuDecoder::new() else { return };
    let packer = lucene_arrow_gpu::encode::GpuPacker::new(&gpu).unwrap();
    use lucene_arrow_docvalues::write::{
        CpuEncoder, encode_numeric_field_dense_chunks, encode_numeric_field_with,
    };
    let cases: Vec<Vec<i64>> = vec![
        (0..100_000).map(|i| 1_000_000 + (i % 4096) * 25).collect(), // gcd
        (0..80_000).map(|i| (i as i64).wrapping_mul(0x9E37_79B9_7F4A_7C15u64 as i64)).collect(), // 64bpv, neg min
        (0..50_000).map(|i| (i % 5) * 7).collect(),                  // table
        (0..65_536).map(|i| i & 0xFFFFF).collect(),                  // 20-bit
    ];
    for (ci, values) in cases.iter().enumerate() {
        let n = values.len() as u32;
        let docs: Vec<u32> = (0..n).collect();
        let cpu = encode_numeric_field_with(&CpuEncoder, 3, &docs, values, n, 999).unwrap();
        // uneven chunk splits, including a 1-element chunk
        let s1 = 1usize;
        let s2 = values.len() / 3;
        let chunks: Vec<&[i64]> =
            vec![&values[..s1], &values[s1..s1 + s2], &values[s1 + s2..]];
        let gpu_enc =
            encode_numeric_field_dense_chunks(&packer, 3, &chunks, n, 999).unwrap();
        assert_eq!(cpu.meta, gpu_enc.meta, "case {ci}: dvm mismatch");
        assert_eq!(cpu.data, gpu_enc.data, "case {ci}: dvd mismatch");
    }
    eprintln!("gpu chunked dense encode == cpu flat for {} cases", cases.len());
}
