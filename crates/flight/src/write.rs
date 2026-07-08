// SPDX-License-Identifier: Apache-2.0

//! The write job (SPEC §10): Arrow RecordBatches → all-live Lucene
//! segments, cut on thresholds, never mid-batch. v1 subset: the NUMERIC
//! family of the §10.2 acceptance matrix (canonical `Int64`/`Float64`
//! pass-through plus the lossless-auto widenings); every coerced field
//! records `lucene.source_type`; anything else fails at schema time —
//! never mid-stream [CONTRACT].

use std::path::PathBuf;

use arrow_array::cast::AsArray;
use arrow_array::types::*;
use arrow_array::{Array, ArrayRef, RecordBatch};
use arrow_schema::{DataType, TimeUnit};

use crate::request::{FileManifest, SegmentManifest, WriteJob};
use lucene_arrow_codec::writer::{
    WriteField, WrittenSegment, commit_segments, random_segment_id,
};
use lucene_arrow_core::{Error, Result};
use lucene_arrow_docvalues::file::DocValuesFileBuilder;
use lucene_arrow_docvalues::write::{CpuEncoder, NumericEncoder};

/// Default segment cut threshold [DEFAULT] (SPEC §10.1).
pub const DEFAULT_SEGMENT_MAX_DOCS: u64 = 1 << 20;

/// How one Arrow column maps into the numeric doc-values family.
#[derive(Debug, Clone, PartialEq, Eq)]
enum NumericCoercion {
    /// Canonical Int64: store as-is.
    Passthrough,
    /// Lossless integer widening / Boolean as 0/1.
    Widen,
    /// Float64 (canonical): store the IEEE bits as sortable i64 payload
    /// (bit-cast, the read side's `Float64` mapping inverts it).
    FloatBits,
    /// Float32 → Float64 first (lossless), then bit-cast.
    Float32Bits,
    /// Temporal → i64 of its unit; unit+tz preserved in `source_type`.
    TemporalI64,
    /// `Utf8` → SORTED (dict-encode on the fly, §10.2 lossless-auto).
    SortedUtf8,
    /// `Dictionary<Int32, Utf8>` → SORTED (canonical).
    SortedDict,
    /// `Binary` (canonical) / `LargeBinary` (lossless-auto) → BINARY.
    Bytes,
    /// `List<Int64>` → SORTED_NUMERIC (canonical; per-doc order is
    /// Lucene-sorted on disk).
    MultiNumeric,
    /// `List<Utf8>` → SORTED_SET (lossless-auto dict-encode).
    MultiTerms,
    /// `UInt64` → NUMERIC, explicit-only (`lucene.allow_lossy`): values
    /// above `i64::MAX` wrap to negative and lose sort order (§10.2).
    U64Lossy,
    /// `Decimal128` → scaled long, explicit-only (`lucene.scale_factor`;
    /// the ES `scaled_float` convention, §10.2).
    DecimalScaled { factor: i64, scale: i8 },
    /// `FixedSizeList<Float32, d>` → flat vectors + GPU-built HNSW graph
    /// (SPEC §10.4). Requires the `cuvs` build; similarity from
    /// `lucene.vector.similarity` metadata (default euclidean).
    VectorF32 { dim: i32, similarity: u8 },
}

impl NumericCoercion {
    /// Canonical §10.2 shapes pass `strict`; everything else is a
    /// lossless-auto coercion available only under `coercion: "auto"`.
    fn is_canonical(&self, dt: &DataType) -> bool {
        matches!(
            (self, dt),
            (NumericCoercion::Passthrough, _)
                | (NumericCoercion::FloatBits, _)
                | (NumericCoercion::SortedDict, _)
                | (NumericCoercion::Bytes, DataType::Binary)
                | (NumericCoercion::MultiNumeric, _)
                | (NumericCoercion::VectorF32 { .. }, _)
        )
    }
}

struct FieldSpec {
    name: String,
    coercion: NumericCoercion,
    source_type: DataType,
}

/// Streaming accumulator: batches in, committed segments out.
pub struct WriteSession {
    out_dir: PathBuf,
    fields: Vec<FieldSpec>,
    segment_max_docs: u64,
    /// Per field, for the segment being built.
    buffers: Vec<ColBuffer>,
    docs_in_segment: u32,
    written: Vec<WrittenSegment>,
    manifests: Vec<SegmentManifest>,
    encoder: Encoder,
}

/// Stats+pack executor for the session ([DEFAULT] auto → gpu when built
/// with the feature and a device exists; echoed via the manifest).
enum Encoder {
    Cpu,
    #[cfg(feature = "gpu")]
    Gpu(lucene_arrow_gpu::encode::GpuPacker),
}

impl Encoder {
    fn as_dyn(&self) -> &dyn NumericEncoder {
        match self {
            Encoder::Cpu => &CpuEncoder,
            #[cfg(feature = "gpu")]
            Encoder::Gpu(p) => p,
        }
    }
    fn name(&self) -> &'static str {
        match self {
            Encoder::Cpu => "cpu",
            #[cfg(feature = "gpu")]
            Encoder::Gpu(_) => "gpu",
        }
    }
}

/// Accumulated column data for the in-progress segment.
enum ColBuffer {
    Numeric { docs: Vec<u32>, values: Vec<i64> },
    Vectors { docs: Vec<u32>, data: Vec<f32> },
    Sorted { docs: Vec<u32>, terms: Vec<Vec<u8>> },
    Bytes { docs: Vec<u32>, values: Vec<Vec<u8>> },
    MultiNumeric { docs: Vec<u32>, values: Vec<Vec<i64>> },
    MultiTerms { docs: Vec<u32>, terms: Vec<Vec<Vec<u8>>> },
}

impl ColBuffer {
    fn for_coercion(c: &NumericCoercion) -> ColBuffer {
        match c {
            NumericCoercion::SortedUtf8 | NumericCoercion::SortedDict => {
                ColBuffer::Sorted { docs: Vec::new(), terms: Vec::new() }
            }
            NumericCoercion::Bytes => ColBuffer::Bytes { docs: Vec::new(), values: Vec::new() },
            NumericCoercion::MultiNumeric => {
                ColBuffer::MultiNumeric { docs: Vec::new(), values: Vec::new() }
            }
            NumericCoercion::MultiTerms => {
                ColBuffer::MultiTerms { docs: Vec::new(), terms: Vec::new() }
            }
            NumericCoercion::VectorF32 { .. } => {
                ColBuffer::Vectors { docs: Vec::new(), data: Vec::new() }
            }
            _ => ColBuffer::Numeric { docs: Vec::new(), values: Vec::new() },
        }
    }
    fn clear(&mut self) {
        match self {
            ColBuffer::Numeric { docs, values } => {
                docs.clear();
                values.clear();
            }
            ColBuffer::Vectors { docs, data } => {
                docs.clear();
                data.clear();
            }
            ColBuffer::Sorted { docs, terms } => {
                docs.clear();
                terms.clear();
            }
            ColBuffer::Bytes { docs, values } => {
                docs.clear();
                values.clear();
            }
            ColBuffer::MultiNumeric { docs, values } => {
                docs.clear();
                values.clear();
            }
            ColBuffer::MultiTerms { docs, terms } => {
                docs.clear();
                terms.clear();
            }
        }
    }
}

fn resolve_encoder(requested: Option<&str>) -> Result<Encoder> {
    match requested.unwrap_or("auto") {
        "cpu" => Ok(Encoder::Cpu),
        #[cfg(feature = "gpu")]
        "gpu" => {
            let dec = lucene_arrow_gpu::GpuDecoder::new()
                .map_err(|e| Error::unsupported(format!("executor \"gpu\": {e}")))?;
            Ok(Encoder::Gpu(lucene_arrow_gpu::encode::GpuPacker::new(&dec)?))
        }
        #[cfg(feature = "gpu")]
        "auto" => Ok(match lucene_arrow_gpu::GpuDecoder::new()
            .ok()
            .and_then(|d| lucene_arrow_gpu::encode::GpuPacker::new(&d).ok())
        {
            Some(p) => Encoder::Gpu(p),
            None => Encoder::Cpu,
        }),
        #[cfg(not(feature = "gpu"))]
        "gpu" => Err(Error::unsupported(
            "executor \"gpu\": server built without the gpu feature",
        )),
        #[cfg(not(feature = "gpu"))]
        "auto" => Ok(Encoder::Cpu),
        other => Err(Error::invalid(format!("unknown executor {other:?}"))),
    }
}

fn classify(field: &arrow_schema::Field) -> Result<NumericCoercion> {
    use DataType::*;
    let dt = field.data_type();
    let md = field.metadata();
    Ok(match dt {
        UInt64 => {
            if md.get(lucene_arrow_core::meta::ALLOW_LOSSY).map(String::as_str) != Some("true") {
                return Err(Error::invalid(format!(
                    "field {}: UInt64 is lossy (values > i64::MAX unrepresentable) — \
                     set field metadata {} = \"true\" to accept (§10.2, never automatic)",
                    field.name(),
                    lucene_arrow_core::meta::ALLOW_LOSSY
                )));
            }
            return Ok(NumericCoercion::U64Lossy);
        }
        Decimal128(_, scale) => {
            let factor = md
                .get(lucene_arrow_core::meta::SCALE_FACTOR)
                .and_then(|v| v.parse::<i64>().ok())
                .ok_or_else(|| {
                    Error::invalid(format!(
                        "field {}: Decimal128 needs field metadata {} = <i64> \
                         (ES scaled_float convention; lossy, never automatic, §10.2)",
                        field.name(),
                        lucene_arrow_core::meta::SCALE_FACTOR
                    ))
                })?;
            return Ok(NumericCoercion::DecimalScaled { factor, scale: *scale });
        }
        Int64 => NumericCoercion::Passthrough,
        Int8 | Int16 | Int32 | UInt8 | UInt16 | UInt32 | Boolean => NumericCoercion::Widen,
        Float64 => NumericCoercion::FloatBits,
        Float32 => NumericCoercion::Float32Bits,
        Timestamp(_, _) | Date32 | Date64 | Time32(_) | Time64(_) | Duration(_) => {
            NumericCoercion::TemporalI64
        }
        Utf8 | LargeUtf8 => NumericCoercion::SortedUtf8,
        Dictionary(k, v) if **k == Int32 && **v == Utf8 => NumericCoercion::SortedDict,
        Binary | LargeBinary => NumericCoercion::Bytes,
        FixedSizeList(f, d) if *f.data_type() == Float32 => {
            if !cfg!(feature = "cuvs") {
                return Err(Error::invalid(format!(
                    "field {}: vector ingest needs the `cuvs` build (GPU graph \
                     construction); rebuild with --features cuvs (§10.4)",
                    field.name()
                )));
            }
            let similarity = match md.get("lucene.vector.similarity").map(String::as_str) {
                None | Some("euclidean") => 0,
                Some("dot_product") => 1,
                Some("cosine") => 2,
                Some("max_inner_product") => 3,
                Some(other) => {
                    return Err(Error::invalid(format!(
                        "field {}: unknown lucene.vector.similarity {other:?}",
                        field.name()
                    )));
                }
            };
            NumericCoercion::VectorF32 { dim: *d, similarity }
        }
        List(f) if *f.data_type() == Int64 => NumericCoercion::MultiNumeric,
        List(f) if *f.data_type() == Utf8 => NumericCoercion::MultiTerms,
        other => {
            return Err(Error::unsupported(format!(
                "write: no doc-values mapping for {other} (SPEC §10.2); \
                 rejected at schema time"
            )));
        }
    })
}

impl WriteSession {
    pub fn new(job: &WriteJob, schema: &arrow_schema::Schema) -> Result<Self> {
        if job.codec != "Lucene103" {
            return Err(Error::unsupported(format!(
                "codec {:?} (only \"Lucene103\", SPEC §3.2)",
                job.codec
            )));
        }
        let out_dir = PathBuf::from(&job.output_dir);
        std::fs::create_dir_all(&out_dir).map_err(|e| Error::Codec(e.to_string()))?;
        if std::fs::read_dir(&out_dir).map_err(|e| Error::Codec(e.to_string()))?.next().is_some() {
            return Err(Error::invalid(format!(
                "output_dir {:?} is not empty (write jobs never append, SPEC §10.1)",
                job.output_dir
            )));
        }
        let strict = job.coercion.as_deref() == Some("strict");
        let fields = schema
            .fields()
            .iter()
            .map(|f| {
                let coercion = classify(f)?;
                if strict && !coercion.is_canonical(f.data_type()) {
                    return Err(Error::invalid(format!(
                        "coercion \"strict\": {} is {} — not a canonical §10.2 shape",
                        f.name(),
                        f.data_type()
                    )));
                }
                Ok(FieldSpec {
                    name: f.name().clone(),
                    coercion,
                    source_type: f.data_type().clone(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        if fields.is_empty() {
            return Err(Error::invalid("write schema has no fields"));
        }
        let buffers = fields.iter().map(|f| ColBuffer::for_coercion(&f.coercion)).collect();
        Ok(WriteSession {
            out_dir,
            fields,
            segment_max_docs: job.segment_max_docs.unwrap_or(DEFAULT_SEGMENT_MAX_DOCS).max(1),
            buffers,
            docs_in_segment: 0,
            written: Vec::new(),
            manifests: Vec::new(),
            encoder: resolve_encoder(job.executor.as_deref())?,
        })
    }

    /// Docids assigned in arrival order; a batch belongs to exactly one
    /// segment — cuts happen between batches, never mid-batch [CONTRACT].
    pub fn push(&mut self, batch: &RecordBatch) -> Result<()> {
        if self.docs_in_segment as u64 >= self.segment_max_docs {
            self.flush_segment()?;
        }
        let base = self.docs_in_segment;
        for (i, spec) in self.fields.iter().enumerate() {
            let col = batch
                .column_by_name(&spec.name)
                .ok_or_else(|| Error::invalid(format!("batch missing column {:?}", spec.name)))?;
            append_column(col, &spec.coercion, base, &mut self.buffers[i])?;
        }
        self.docs_in_segment += batch.num_rows() as u32;
        Ok(())
    }

    fn flush_segment(&mut self) -> Result<()> {
        if self.docs_in_segment == 0 {
            return Ok(());
        }
        let t = std::time::Instant::now();
        let name = format!("_{}", lucene_arrow_codec::liv::base36(self.written.len() as i64));
        let id = random_segment_id();
        let mut builder = DocValuesFileBuilder::new(&id, "Lucene90_0");
        for (i, buf) in self.buffers.iter().enumerate() {
            match buf {
                ColBuffer::Numeric { docs, values } => builder.add_numeric_with(
                    self.encoder.as_dyn(),
                    i as i32,
                    docs,
                    values,
                    self.docs_in_segment,
                )?,
                ColBuffer::Sorted { docs, terms } => {
                    let refs: Vec<&[u8]> = terms.iter().map(|t| t.as_slice()).collect();
                    builder.add_sorted_with(
                        self.encoder.as_dyn(),
                        i as i32,
                        docs,
                        &refs,
                        self.docs_in_segment,
                    )?
                }
                ColBuffer::Bytes { docs, values } => {
                    let refs: Vec<&[u8]> = values.iter().map(|v| v.as_slice()).collect();
                    builder.add_binary(i as i32, docs, &refs, self.docs_in_segment)?
                }
                ColBuffer::MultiNumeric { docs, values } => builder.add_sorted_numeric_with(
                    self.encoder.as_dyn(),
                    i as i32,
                    docs,
                    values,
                    self.docs_in_segment,
                )?,
                ColBuffer::MultiTerms { docs, terms } => builder.add_sorted_set_with(
                    self.encoder.as_dyn(),
                    i as i32,
                    docs,
                    terms,
                    self.docs_in_segment,
                )?,
                ColBuffer::Vectors { .. } => {} // handled below (not doc values)
            }
        }
        let (dvm, dvd) = builder.finish();
        let write_fields: Vec<WriteField> = self
            .fields
            .iter()
            .enumerate()
            .map(|(i, s)| {
                if let NumericCoercion::VectorF32 { dim, similarity } = s.coercion {
                    return WriteField {
                        name: s.name.clone(),
                        number: i as u32,
                        doc_values_type: 0,
                        vector_dim: dim as u32,
                        vector_encoding: 1,
                        vector_similarity: similarity,
                        index_options: 0,
                    };
                }
                WriteField::doc_values(
                    s.name.clone(),
                    i as u32,
                    match s.coercion {
                        NumericCoercion::SortedUtf8 | NumericCoercion::SortedDict => 3,
                        NumericCoercion::Bytes => 2,
                        NumericCoercion::MultiTerms => 4,
                        NumericCoercion::MultiNumeric => 5,
                        _ => 1,
                    },
                )
            })
            .collect();
        // Vector columns: flat pair + GPU-built graph pair per field
        // (SPEC §10.4). All under the per-field HNSW suffix.
        let mut extra_files: Vec<(String, Vec<u8>)> = Vec::new();
        let suffix = "Lucene99HnswVectorsFormat_0";
        let mut flat: Option<lucene_arrow_vectors::file::VectorsFileBuilder> = None;
        let mut hnsw: Option<lucene_arrow_vectors::hnsw::HnswFilesBuilder> = None;
        for (i, buf) in self.buffers.iter().enumerate() {
            let ColBuffer::Vectors { docs, data } = buf else { continue };
            let NumericCoercion::VectorF32 { dim, similarity } = self.fields[i].coercion else {
                continue;
            };
            let sim = lucene_arrow_vectors::Similarity::from_ordinal(similarity as i32)
                .ok_or_else(|| Error::invalid("bad similarity"))?;
            let payload: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
            flat.get_or_insert_with(|| {
                lucene_arrow_vectors::file::VectorsFileBuilder::new(&id, suffix)
            })
            .add_field(
                i as i32,
                lucene_arrow_vectors::VectorEncoding::Float32,
                sim,
                dim as u32,
                docs,
                &payload,
                self.docs_in_segment,
            )?;
            let parsed = build_graph(data, dim as usize)?;
            hnsw.get_or_insert_with(|| {
                lucene_arrow_vectors::hnsw::HnswFilesBuilder::new(&id, suffix)
            })
            .add_field_multi(
                i as i32,
                lucene_arrow_vectors::VectorEncoding::Float32,
                sim,
                dim as u32,
                parsed.count,
                parsed.m,
                &parsed.levels,
            )?;
        }
        if let (Some(f), Some(h)) = (flat, hnsw) {
            let (vemf, vec) = f.finish();
            let (vem, vex) = h.finish();
            extra_files.push((format!("{name}_{suffix}.vemf"), vemf));
            extra_files.push((format!("{name}_{suffix}.vec"), vec));
            extra_files.push((format!("{name}_{suffix}.vem"), vem));
            extra_files.push((format!("{name}_{suffix}.vex"), vex));
        }
        let extra_refs: Vec<(String, &[u8])> =
            extra_files.iter().map(|(n, b)| (n.clone(), b.as_slice())).collect();
        let seg = lucene_arrow_codec::writer::write_segment_files_ext(
            &self.out_dir,
            &name,
            id,
            &write_fields,
            self.docs_in_segment,
            &dvm,
            &dvd,
            &extra_refs,
        )?;

        let files = seg
            .files
            .iter()
            .map(|f| {
                // The big payloads are still in memory — CRC those without
                // re-reading 100s of MB; framing files are tiny.
                if f.ends_with(".dvd") {
                    Ok(FileManifest {
                        name: f.clone(),
                        bytes: dvd.len() as u64,
                        crc32: crc32fast::hash(&dvd),
                    })
                } else if f.ends_with(".dvm") {
                    Ok(FileManifest {
                        name: f.clone(),
                        bytes: dvm.len() as u64,
                        crc32: crc32fast::hash(&dvm),
                    })
                } else {
                    let bytes = std::fs::read(self.out_dir.join(f))
                        .map_err(|e| Error::Codec(e.to_string()))?;
                    Ok(FileManifest {
                        name: f.clone(),
                        bytes: bytes.len() as u64,
                        crc32: crc32fast::hash(&bytes),
                    })
                }
            })
            .collect::<Result<Vec<_>>>()?;
        self.manifests.push(SegmentManifest {
            name: name.clone(),
            max_doc: self.docs_in_segment as i32,
            files,
            field_stats: self
                .fields
                .iter()
                .map(|s| {
                    (
                        s.name.clone(),
                        serde_json::json!({ "lucene.type":
                                            match s.coercion {
                                                NumericCoercion::SortedUtf8
                                                | NumericCoercion::SortedDict => "sorted",
                                                NumericCoercion::Bytes => "binary",
                                                NumericCoercion::MultiTerms => "sorted_set",
                                                NumericCoercion::MultiNumeric => "sorted_numeric",
                                                NumericCoercion::VectorF32 { .. } => "vector",
                                                _ => "numeric",
                                            },
                                            "lucene.source_type": s.source_type.to_string(),
                                            "lucene.encoder": self.encoder.name() }),
                    )
                })
                .collect(),
            wall_ms: t.elapsed().as_millis() as u64,
        });
        self.written.push(seg);
        self.docs_in_segment = 0;
        for buf in &mut self.buffers {
            buf.clear();
        }
        Ok(())
    }

    /// Flush the tail segment and commit one `segments_N` for the job.
    pub fn finish(mut self) -> Result<Vec<SegmentManifest>> {
        self.flush_segment()?;
        if self.written.is_empty() {
            return Err(Error::invalid("write job received no rows"));
        }
        commit_segments(&self.out_dir, &self.written)?;
        Ok(self.manifests)
    }
}

fn append_column(
    col: &ArrayRef,
    coercion: &NumericCoercion,
    doc_base: u32,
    buffer: &mut ColBuffer,
) -> Result<()> {
    if let ColBuffer::Vectors { docs, data } = buffer {
        let NumericCoercion::VectorF32 { dim, .. } = coercion else {
            return Err(Error::invalid("vector buffer without vector coercion"));
        };
        let a = col
            .as_any()
            .downcast_ref::<arrow_array::FixedSizeListArray>()
            .ok_or_else(|| Error::invalid("expected FixedSizeList column"))?;
        let values = a.values().as_primitive::<arrow_array::types::Float32Type>();
        for i in 0..a.len() {
            if a.is_valid(i) {
                docs.push(doc_base + i as u32);
                let start = i * *dim as usize;
                data.extend_from_slice(&values.values()[start..start + *dim as usize]);
            }
        }
        return Ok(());
    }
    if let ColBuffer::Sorted { docs, terms } = buffer {
        match coercion {
            NumericCoercion::SortedUtf8 => {
                let a = col.as_string::<i32>();
                for i in 0..a.len() {
                    if a.is_valid(i) {
                        docs.push(doc_base + i as u32);
                        terms.push(a.value(i).as_bytes().to_vec());
                    }
                }
            }
            NumericCoercion::SortedDict => {
                let d = col.as_dictionary::<arrow_array::types::Int32Type>();
                let vals = d.values().as_string::<i32>();
                for i in 0..d.len() {
                    if d.is_valid(i) {
                        docs.push(doc_base + i as u32);
                        terms.push(vals.value(d.keys().value(i) as usize).as_bytes().to_vec());
                    }
                }
            }
            other => return Err(Error::invalid(format!("coercion {other:?} for string buffer"))),
        }
        return Ok(());
    }
    if let ColBuffer::Bytes { docs, values } = buffer {
        let a = col.as_binary::<i32>();
        for i in 0..a.len() {
            if a.is_valid(i) {
                docs.push(doc_base + i as u32);
                values.push(a.value(i).to_vec());
            }
        }
        return Ok(());
    }
    if let ColBuffer::MultiNumeric { docs, values } = buffer {
        let l = col.as_list::<i32>();
        let child = l.values().as_primitive::<Int64Type>();
        for i in 0..l.len() {
            if l.is_valid(i) {
                let (s, e) = (l.value_offsets()[i] as usize, l.value_offsets()[i + 1] as usize);
                if s == e {
                    continue; // empty list == field absent (Lucene has no empty doc value)
                }
                docs.push(doc_base + i as u32);
                values.push((s..e).map(|j| child.value(j)).collect());
            }
        }
        return Ok(());
    }
    if let ColBuffer::MultiTerms { docs, terms } = buffer {
        let l = col.as_list::<i32>();
        let child = l.values().as_string::<i32>();
        for i in 0..l.len() {
            if l.is_valid(i) {
                let (s, e) = (l.value_offsets()[i] as usize, l.value_offsets()[i + 1] as usize);
                if s == e {
                    continue;
                }
                docs.push(doc_base + i as u32);
                terms.push((s..e).map(|j| child.value(j).as_bytes().to_vec()).collect());
            }
        }
        return Ok(());
    }
    let ColBuffer::Numeric { docs, values } = buffer else {
        return Err(Error::invalid("buffer/coercion mismatch"));
    };
    let mut emit = |i: usize, v: i64| {
        docs.push(doc_base + i as u32);
        values.push(v);
    };
    macro_rules! prim {
        ($ty:ty, $conv:expr) => {{
            let a = col.as_primitive::<$ty>();
            for i in 0..a.len() {
                if a.is_valid(i) {
                    #[allow(clippy::redundant_closure_call)]
                    emit(i, ($conv)(a.value(i)));
                }
            }
        }};
    }
    match (coercion, col.data_type()) {
        (NumericCoercion::Passthrough, _) => {
            let a = col.as_primitive::<Int64Type>();
            if a.null_count() == 0 {
                // Dense canonical Int64: bulk-copy, no per-value branch.
                values.extend_from_slice(a.values());
                docs.extend(doc_base..doc_base + a.len() as u32);
            } else {
                prim!(Int64Type, |v: i64| v)
            }
        }
        (NumericCoercion::Widen, DataType::Int8) => prim!(Int8Type, |v: i8| v as i64),
        (NumericCoercion::Widen, DataType::Int16) => prim!(Int16Type, |v: i16| v as i64),
        (NumericCoercion::Widen, DataType::Int32) => prim!(Int32Type, |v: i32| v as i64),
        (NumericCoercion::Widen, DataType::UInt8) => prim!(UInt8Type, |v: u8| v as i64),
        (NumericCoercion::Widen, DataType::UInt16) => prim!(UInt16Type, |v: u16| v as i64),
        (NumericCoercion::Widen, DataType::UInt32) => prim!(UInt32Type, |v: u32| v as i64),
        (NumericCoercion::Widen, DataType::Boolean) => {
            let a = col.as_boolean();
            for i in 0..a.len() {
                if a.is_valid(i) {
                    emit(i, a.value(i) as i64);
                }
            }
        }
        (NumericCoercion::U64Lossy, _) => prim!(UInt64Type, |v: u64| v as i64),
        (NumericCoercion::DecimalScaled { factor, scale }, _) => {
            let a = col.as_primitive::<Decimal128Type>();
            let denom: i128 = 10i128.pow(*scale as u32);
            for i in 0..a.len() {
                if a.is_valid(i) {
                    // round(raw · factor / 10^scale), half away from zero —
                    // the ES scaled_float convention.
                    let n = a.value(i) * *factor as i128;
                    let rounded = if n >= 0 { (n + denom / 2) / denom } else { (n - denom / 2) / denom };
                    emit(i, rounded as i64);
                }
            }
        }
        (NumericCoercion::FloatBits, _) => prim!(Float64Type, |v: f64| v.to_bits() as i64),
        (NumericCoercion::Float32Bits, _) => {
            prim!(Float32Type, |v: f32| (v as f64).to_bits() as i64)
        }
        (NumericCoercion::TemporalI64, dt) => match dt {
            DataType::Timestamp(TimeUnit::Second, _) => prim!(TimestampSecondType, |v: i64| v),
            DataType::Timestamp(TimeUnit::Millisecond, _) => {
                prim!(TimestampMillisecondType, |v: i64| v)
            }
            DataType::Timestamp(TimeUnit::Microsecond, _) => {
                prim!(TimestampMicrosecondType, |v: i64| v)
            }
            DataType::Timestamp(TimeUnit::Nanosecond, _) => {
                prim!(TimestampNanosecondType, |v: i64| v)
            }
            DataType::Date32 => prim!(Date32Type, |v: i32| v as i64),
            DataType::Date64 => prim!(Date64Type, |v: i64| v),
            DataType::Time32(TimeUnit::Second) => prim!(Time32SecondType, |v: i32| v as i64),
            DataType::Time32(TimeUnit::Millisecond) => {
                prim!(Time32MillisecondType, |v: i32| v as i64)
            }
            DataType::Time64(TimeUnit::Microsecond) => prim!(Time64MicrosecondType, |v: i64| v),
            DataType::Time64(TimeUnit::Nanosecond) => prim!(Time64NanosecondType, |v: i64| v),
            DataType::Duration(TimeUnit::Second) => prim!(DurationSecondType, |v: i64| v),
            DataType::Duration(TimeUnit::Millisecond) => {
                prim!(DurationMillisecondType, |v: i64| v)
            }
            DataType::Duration(TimeUnit::Microsecond) => {
                prim!(DurationMicrosecondType, |v: i64| v)
            }
            DataType::Duration(TimeUnit::Nanosecond) => prim!(DurationNanosecondType, |v: i64| v),
            other => return Err(Error::unsupported(format!("temporal type {other}"))),
        },
        (c, dt) => {
            return Err(Error::invalid(format!("coercion {c:?} does not accept {dt}")));
        }
    }
    Ok(())
}


/// Build the HNSW graph for one segment's vectors as a multi-level
/// hierarchy. Large segments (`cuvs` build) go CAGRA → cuVS HNSW
/// hierarchy → parse — near-native search quality. Tiny segments use an
/// exact CPU kNN graph wrapped as a single level (no GPU needed).
fn build_graph(data: &[f32], dim: usize) -> Result<lucene_arrow_vectors::hnsw::HnswParsed> {
    use lucene_arrow_vectors::hnsw::HnswParsed;
    let n = data.len() / dim;
    if n == 0 {
        return Ok(HnswParsed { count: 0, m: 1, entry: 0, levels: vec![Vec::new()] });
    }
    let degree = 32usize.min(n.saturating_sub(1)).max(1);
    // Exact CPU graph for small segments (also the no-GPU fallback there),
    // wrapped as a single-level hierarchy.
    if n <= 4096 {
        let mut out: Vec<Vec<u32>> = Vec::with_capacity(n);
        for i in 0..n {
            let vi = &data[i * dim..(i + 1) * dim];
            let mut scored: Vec<(f32, u32)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| {
                    let vj = &data[j * dim..(j + 1) * dim];
                    let d: f32 = vi.iter().zip(vj).map(|(a, b)| (a - b) * (a - b)).sum();
                    (d, j as u32)
                })
                .collect();
            scored.sort_by(|a, b| a.0.total_cmp(&b.0));
            scored.truncate(degree);
            out.push(scored.into_iter().map(|(_, j)| j).collect());
        }
        let flat: Vec<u32> = out.iter().flat_map(|l| l.iter().copied()).collect();
        let single = if out.iter().all(|l| l.len() == degree) {
            lucene_arrow_vectors::hnsw::navigable_from_knn(&flat, n, degree, degree.max(2))
        } else {
            out
        };
        let level0 = single.into_iter().enumerate().map(|(i, nb)| (i as u32, nb)).collect();
        return Ok(HnswParsed {
            count: n,
            m: (degree / 2).max(1) as u32,
            entry: 0,
            levels: vec![level0],
        });
    }
    #[cfg(feature = "cuvs")]
    {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let ctx = lucene_arrow_gpu::cuvs_knn::CuvsContext::new()
            .map_err(|e| Error::invalid(format!("vector ingest needs a GPU at this size: {e}")))?;
        // CAGRA → cuVS HNSW hierarchy (hierarchy=CPU → standard hnswlib) →
        // parse → multi-level graph (near-native recall/QPS; see vectors::hnsw).
        // Serialize lands on /dev/shm when available (the file embeds all
        // vectors) and the parse mmaps it — no disk round-trip, no Vec copy.
        let tmp = lucene_arrow_gpu::cuvs_knn::CuvsContext::hnsw_scratch_dir().join(format!(
            "la_hnsw_{}_{}.bin",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let path = tmp.to_str().ok_or_else(|| Error::invalid("non-utf8 temp path"))?;
        ctx.cagra_to_hnswlib(data, dim, degree / 2, 100, path)?;
        let parsed = lucene_arrow_vectors::hnsw::parse_hnswlib_file(&tmp);
        let _ = std::fs::remove_file(&tmp);
        parsed
    }
    #[cfg(not(feature = "cuvs"))]
    Err(Error::invalid("vector ingest needs the cuvs build for segments > 4096 docs"))
}
