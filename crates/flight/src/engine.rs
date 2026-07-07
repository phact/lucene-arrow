// SPDX-License-Identifier: Apache-2.0

//! The scan engine behind the Flight surface: `ReadRequest` → reconciled
//! schema (§8.2) → segment-scoped RecordBatches (§7.1) with system columns
//! and echoed effective config (§8.3). CPU executors for now; the GPU
//! engine slots in behind the same interface (`executor: "auto"` resolves
//! to `"cpu"` and is echoed as such).

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_array::{ArrayRef, Int32Array, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema, SchemaRef};

use crate::request::{ReadRequest, SegmentSelector};
use lucene_arrow_codec::{DocValuesKind, SegmentDirectory, SegmentMeta};
use lucene_arrow_core::meta;
use lucene_arrow_core::meta::RowMode;
use lucene_arrow_core::{Error, Result};
use lucene_arrow_cpu as cpu;
use lucene_arrow_docvalues::read::{DocValuesPlans, DvField, DvKind, plan_doc_values};
use lucene_arrow_vectors::read::{VecField, plan_vectors};

/// The server-resolved request, echoed back verbatim (§8.3 [CONTRACT]).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ResolvedConfig {
    pub path: String,
    pub segments: Vec<String>,
    pub columns: Vec<String>,
    pub row_mode: String,
    pub dict: BTreeMap<String, String>,
    pub batch_rows: u64,
    pub executor: String,
    pub frame_version: u32,
}

/// Default batch sizing [DEFAULT] (SPEC §7.1).
pub const DEFAULT_BATCH_ROWS: u64 = 128 * 1024;

pub struct ResolvedRead {
    pub dir: SegmentDirectory,
    pub schema: SchemaRef,
    pub segments: Vec<usize>, // ords into dir.segments()
    pub config: ResolvedConfig,
    pub row_mode: RowMode,
    pub use_gpu: bool,
}

/// Process-wide GPU decoder (NVRTC compile once). `None` after a failed
/// probe — the engine then serves `executor: "cpu"` and echoes it.
#[cfg(feature = "gpu")]
fn gpu_decoder() -> Option<&'static lucene_arrow_gpu::GpuDecoder> {
    use std::sync::OnceLock;
    static GPU: OnceLock<Option<lucene_arrow_gpu::GpuDecoder>> = OnceLock::new();
    GPU.get_or_init(|| lucene_arrow_gpu::GpuDecoder::new().ok()).as_ref()
}

fn dv_kind(k: DocValuesKind) -> Option<DvKind> {
    match k {
        DocValuesKind::Numeric => Some(DvKind::Numeric),
        DocValuesKind::Binary => Some(DvKind::Binary),
        DocValuesKind::Sorted => Some(DvKind::Sorted),
        DocValuesKind::SortedNumeric => Some(DvKind::SortedNumeric),
        DocValuesKind::SortedSet => Some(DvKind::SortedSet),
        DocValuesKind::None => None,
    }
}

/// Arrow type a field decodes to (§7.2 shapes). `None` = not scannable
/// yet (BINARY lands with task #14).
fn field_arrow_type(f: &lucene_arrow_codec::FieldMeta, seg_plans: &SegPlans) -> Option<DataType> {
    if f.has_vectors {
        let child = match f.vector_encoding {
            0 => DataType::Int8,
            _ => DataType::Float32,
        };
        return Some(DataType::FixedSizeList(
            Arc::new(Field::new("item", child, false)),
            f.vector_dimension as i32,
        ));
    }
    match f.doc_values {
        DocValuesKind::Numeric => Some(DataType::Int64),
        DocValuesKind::Sorted => Some(DataType::Dictionary(
            Box::new(DataType::Int32),
            Box::new(DataType::Utf8),
        )),
        DocValuesKind::SortedSet => {
            let multi = seg_plans
                .dv
                .sorted
                .iter()
                .find(|p| p.ords.column.name == f.name)
                .is_some_and(|p| p.addresses.is_some());
            let dict = DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8));
            if multi {
                Some(DataType::List(Arc::new(Field::new("item", dict, false))))
            } else {
                Some(dict)
            }
        }
        DocValuesKind::SortedNumeric => {
            let multi =
                seg_plans.dv.multi_numeric.iter().any(|p| p.values.column.name == f.name);
            if multi {
                Some(DataType::List(Arc::new(Field::new("item", DataType::Int64, false))))
            } else {
                Some(DataType::Int64)
            }
        }
        DocValuesKind::Binary => Some(DataType::Binary),
        DocValuesKind::None => None,
    }
}

struct SegPlans {
    dv: DocValuesPlans,
    vectors: Vec<lucene_arrow_vectors::read::VectorPlan>,
    dvd: Vec<u8>,
    vec_data: Vec<u8>,
}

fn plan_segment(dir: &SegmentDirectory, seg: &SegmentMeta) -> Result<SegPlans> {
    let dv_fields: Vec<DvField> = seg
        .fields
        .iter()
        .filter_map(|f| {
            dv_kind(f.doc_values).map(|kind| DvField {
                number: f.number as i32,
                name: f.name.clone(),
                kind,
                has_skip_index: f.has_skip_index,
            })
        })
        .collect();

    let (dv, dvd) = if let Some(dvm_name) = seg.files.iter().find(|f| f.ends_with(".dvm")) {
        let dvd_name = seg
            .files
            .iter()
            .find(|f| f.ends_with(".dvd"))
            .ok_or_else(|| Error::corrupt(".dvm without .dvd"))?;
        let dvm = dir.open_input(&seg.name, dvm_name)?;
        let dvd = dir.open_input(&seg.name, dvd_name)?;
        let dvm = dvm.slice(0, dvm.len()).ok_or_else(|| Error::unsupported("non-mmap source"))?;
        let dvd = dvd
            .slice(0, dvd.len())
            .ok_or_else(|| Error::unsupported("non-mmap source"))?
            .to_vec();
        (plan_doc_values(dvm, &dvd, &dv_fields, seg.max_doc as u32, dvd_name)?, dvd)
    } else {
        (
            DocValuesPlans {
                plans: Vec::new(),
                sorted: Vec::new(),
                multi_numeric: Vec::new(),
                binary: Vec::new(),
                skipped: Vec::new(),
            },
            Vec::new(),
        )
    };

    let (vectors, vec_data) = if let Some(vemf_name) = seg.files.iter().find(|f| f.ends_with(".vemf")) {
        let vec_name = seg
            .files
            .iter()
            .find(|f| f.ends_with(".vec"))
            .ok_or_else(|| Error::corrupt(".vemf without .vec"))?;
        let vec_fields: Vec<VecField> = seg
            .fields
            .iter()
            .filter(|f| f.has_vectors)
            .map(|f| VecField { number: f.number as i32, name: f.name.clone() })
            .collect();
        let vemf = dir.open_input(&seg.name, vemf_name)?;
        let vec_r = dir.open_input(&seg.name, vec_name)?;
        let vemf = vemf.slice(0, vemf.len()).ok_or_else(|| Error::unsupported("non-mmap source"))?;
        let vec_data = vec_r
            .slice(0, vec_r.len())
            .ok_or_else(|| Error::unsupported("non-mmap source"))?
            .to_vec();
        (plan_vectors(vemf, &vec_fields, seg.max_doc as u32, vec_name)?, vec_data)
    } else {
        (Vec::new(), Vec::new())
    };

    Ok(SegPlans { dv, vectors, dvd, vec_data })
}

/// Resolve a request: pick segments, reconcile the schema (§8.2 — absent
/// fields become all-null columns; true type conflicts fail loudly), and
/// echo the effective config (§8.3).
pub fn resolve(request: &ReadRequest) -> Result<ResolvedRead> {
    let row_mode = RowMode::parse(&request.row_mode)
        .ok_or_else(|| Error::invalid("row_mode must be \"compact\" or \"positional\" (no default, SPEC §7.1)"))?;
    if request.frame_version != meta::FRAME_VERSION {
        return Err(Error::unsupported(format!(
            "frame_version {} (server speaks {})",
            request.frame_version,
            meta::FRAME_VERSION
        )));
    }

    let dir = SegmentDirectory::open(&request.path)?;
    let segments: Vec<usize> = match &request.segments {
        SegmentSelector::All => (0..dir.segments().len()).collect(),
        SegmentSelector::Names(names) => names
            .iter()
            .map(|n| {
                dir.segments()
                    .iter()
                    .position(|s| &s.name == n)
                    .ok_or_else(|| Error::invalid(format!("no segment {n:?}")))
            })
            .collect::<Result<_>>()?,
    };


    // Reconcile schema across selected segments (§8.2).
    let mut fields: Vec<Field> = Vec::new();
    let mut seen: BTreeMap<String, DataType> = BTreeMap::new();
    for &ord in &segments {
        let seg = &dir.segments()[ord];
        let plans = plan_segment(&dir, seg)?;
        for f in &seg.fields {
            let Some(dt) = field_arrow_type(f, &plans) else { continue };
            if let Some(want) = request.columns.as_ref()
                && !want.contains(&f.name)
            {
                continue;
            }
            match seen.get(&f.name) {
                None => {
                    seen.insert(f.name.clone(), dt.clone());
                    let mut md = std::collections::HashMap::new();
                    md.insert(meta::FIELD_NAME.to_string(), f.name.clone());
                    fields.push(Field::new(&f.name, dt, true).with_metadata(md));
                }
                Some(prev) if *prev == dt => {}
                Some(prev) => {
                    return Err(Error::invalid(format!(
                        "type conflict for field {:?}: segment {} says {dt}, others {prev} (no coercion, SPEC §8.2)",
                        f.name, seg.name
                    )));
                }
            }
        }
    }

    // System columns (§7.1 [CONTRACT]), deselectable via columns=.
    let want = |name: &str| request.columns.as_ref().is_none_or(|c| c.iter().any(|x| x == name));
    let mut all = Vec::new();
    if want(meta::COL_SEG) {
        all.push(Field::new(meta::COL_SEG, DataType::Int32, false));
    }
    if want(meta::COL_DOC) {
        all.push(Field::new(meta::COL_DOC, DataType::Int32, false));
    }
    if want(meta::COL_GLOBAL_DOC) {
        all.push(Field::new(meta::COL_GLOBAL_DOC, DataType::Int64, false));
    }
    // Positional mode: one row per docid + `_live` (SPEC §7.1). Deletion is
    // never expressed as null — `_live` is non-nullable Boolean.
    if row_mode == RowMode::Positional && want(meta::COL_LIVE) {
        all.push(Field::new(meta::COL_LIVE, DataType::Boolean, false));
    }
    all.extend(fields);
    let schema = Arc::new(Schema::new(all));

    // Executor resolution [DEFAULT: auto → gpu when present] (§8.3: the
    // echoed value is what actually ran, never the request's wish).
    let requested = request.executor.as_deref().unwrap_or("auto");
    let use_gpu = match requested {
        "cpu" => false,
        #[cfg(feature = "gpu")]
        "gpu" => {
            if gpu_decoder().is_none() {
                return Err(Error::unsupported("executor \"gpu\": no CUDA device available"));
            }
            true
        }
        #[cfg(feature = "gpu")]
        "auto" => gpu_decoder().is_some(),
        #[cfg(not(feature = "gpu"))]
        "gpu" => {
            return Err(Error::unsupported(
                "executor \"gpu\": server built without the gpu feature",
            ));
        }
        #[cfg(not(feature = "gpu"))]
        "auto" => false,
        other => return Err(Error::invalid(format!("unknown executor {other:?}"))),
    };

    let config = ResolvedConfig {
        path: request.path.clone(),
        segments: segments.iter().map(|&o| dir.segments()[o].name.clone()).collect(),
        columns: schema.fields().iter().map(|f| f.name().clone()).collect(),
        row_mode: row_mode.as_str().to_string(),
        dict: BTreeMap::from([("*".to_string(), meta::dict_mode::SEGMENT.to_string())]),
        batch_rows: request.batch_rows.unwrap_or(DEFAULT_BATCH_ROWS),
        executor: if use_gpu { "gpu".to_string() } else { "cpu".to_string() },
        frame_version: meta::FRAME_VERSION,
    };

    Ok(ResolvedRead { dir, schema, segments, config, row_mode, use_gpu })
}

/// `DoAction: "stats"` (SPEC §8.1): per-field decode-cost estimates —
/// planner food. Metadata-only (plans, no payload decode).
pub fn stats(path: &str) -> Result<serde_json::Value> {
    let dir = SegmentDirectory::open(path)?;
    let mut segments = Vec::new();
    for seg in dir.segments() {
        let plans = plan_segment(&dir, seg)?;
        let mut fields = Vec::new();
        for p in &plans.dv.plans {
            fields.push(serde_json::json!({
                "field": p.column.name, "shape": "numeric",
                "num_values": p.num_values, "payload_bytes": p.payload_bytes(),
                "blocks": p.blocks.len(),
            }));
        }
        for p in &plans.dv.sorted {
            fields.push(serde_json::json!({
                "field": p.ords.column.name,
                "shape": if p.addresses.is_some() { "sorted_set_multi" } else { "sorted" },
                "num_values": p.ords.num_values,
                "payload_bytes": p.ords.payload_bytes(),
                "num_terms": p.terms.num_terms,
                "terms_bytes": p.terms.terms_len,
            }));
        }
        for p in &plans.dv.multi_numeric {
            fields.push(serde_json::json!({
                "field": p.values.column.name, "shape": "sorted_numeric_multi",
                "num_values": p.values.num_values,
                "payload_bytes": p.values.payload_bytes(),
            }));
        }
        for p in &plans.dv.binary {
            fields.push(serde_json::json!({
                "field": p.column.name, "shape": "binary",
                "num_values": p.num_docs_with_field, "payload_bytes": p.data_len,
            }));
        }
        for p in &plans.vectors {
            fields.push(serde_json::json!({
                "field": p.plan.column.name, "shape": "vector",
                "num_values": p.count, "dim": p.dim,
                "payload_bytes": p.count * p.dim as u64 * p.encoding.width() as u64,
            }));
        }
        segments.push(serde_json::json!({
            "segment": seg.name, "max_doc": seg.max_doc, "del_count": seg.del_count,
            "fields": fields,
        }));
    }
    Ok(serde_json::json!({ "path": path, "segments": segments }))
}

/// Decode one segment into full-length columns, then slice into
/// `batch_rows`-sized RecordBatches (slices are zero-copy).
pub fn segment_batches(read: &ResolvedRead, seg_ord: usize, doc_base: i64) -> Result<Vec<RecordBatch>> {
    let seg = &read.dir.segments()[seg_ord];
    let plans = plan_segment(&read.dir, seg)?;
    let num_docs = seg.max_doc as usize;
    let live = read.dir.live_docs(&seg.name)?;
    let live_at = |d: usize| live.as_ref().is_none_or(|w| w[d / 64] >> (d % 64) & 1 == 1);

    let mut columns: Vec<ArrayRef> = Vec::with_capacity(read.schema.fields().len());
    for field in read.schema.fields() {
        let name = field.name().as_str();
        let array: ArrayRef = match name {
            _ if name == meta::COL_SEG => {
                Arc::new(Int32Array::from(vec![seg.ord as i32; num_docs]))
            }
            _ if name == meta::COL_DOC => Arc::new(Int32Array::from_iter_values(0..num_docs as i32)),
            _ if name == meta::COL_GLOBAL_DOC => {
                Arc::new(Int64Array::from_iter_values(doc_base..doc_base + num_docs as i64))
            }
            _ if name == meta::COL_LIVE => {
                Arc::new(arrow_array::BooleanArray::from_iter((0..num_docs).map(|d| Some(live_at(d)))))
            }
            _ => decode_field(&plans, name, num_docs, field.data_type(), read.use_gpu)?,
        };
        columns.push(array);
    }

    let mut batch = RecordBatch::try_new(read.schema.clone(), columns)?;
    // Compact mode drops tombstones server-side (SPEC §7.1).
    if read.row_mode == RowMode::Compact && live.is_some() {
        let mask = arrow_array::BooleanArray::from_iter((0..num_docs).map(|d| Some(live_at(d))));
        batch = arrow_select::filter::filter_record_batch(&batch, &mask)?;
    }
    let num_docs = batch.num_rows();
    let rows_per = read.config.batch_rows.max(1) as usize;
    let mut out = Vec::with_capacity(num_docs.div_ceil(rows_per));
    let mut off = 0;
    while off < num_docs {
        let len = rows_per.min(num_docs - off);
        out.push(batch.slice(off, len));
        off += len;
    }
    Ok(out)
}

fn decode_field(
    plans: &SegPlans,
    name: &str,
    num_docs: usize,
    expected: &DataType,
    use_gpu: bool,
) -> Result<ArrayRef> {
    if let Some(plan) = plans.dv.plans.iter().find(|p| p.column.name == name) {
        // Numeric columns are the bulk payload — GPU when resolved so.
        // (Dictionary/list/binary shapes stay CPU until their kernels land.)
        #[cfg(feature = "gpu")]
        if use_gpu && let Some(gpu) = gpu_decoder() {
            return gpu.decode_numeric(plan, &plans.dvd);
        }
        let _ = use_gpu;
        return cpu::decode_numeric(plan, &plans.dvd);
    }
    let _ = use_gpu;
    if let Some(plan) = plans.dv.sorted.iter().find(|p| p.ords.column.name == name) {
        return cpu::decode_sorted(plan, &plans.dvd);
    }
    if let Some(plan) = plans.dv.multi_numeric.iter().find(|p| p.values.column.name == name) {
        return cpu::decode_multi_numeric(plan, &plans.dvd);
    }
    if let Some(plan) = plans.dv.binary.iter().find(|p| p.column.name == name) {
        return cpu::decode_binary(plan, &plans.dvd);
    }
    if let Some(plan) = plans.vectors.iter().find(|p| p.plan.column.name == name) {
        return cpu::decode_flat_vectors(&plan.plan, &plans.vec_data);
    }
    // Field absent in this segment → all-null column (§8.2).
    Ok(arrow_array::new_null_array(expected, num_docs))
}

/// SPEC §7.8: the postings relation — one batch stream per segment,
/// schema `term: Dictionary(Int32, Binary) | doc: UInt32 | freq: UInt32`.
/// Raw positional truth: deleted docs are NOT filtered (join `_live`
/// from the doc-values relation when needed).
pub fn postings_batches(
    dir: &lucene_arrow_codec::SegmentDirectory,
    seg: &lucene_arrow_codec::SegmentMeta,
    field_name: &str,
    batch_rows: usize,
) -> Result<Vec<RecordBatch>> {
    use arrow_array::builder::BinaryBuilder;
    use arrow_array::{DictionaryArray, UInt32Array};
    use lucene_arrow_postings::walk::FieldTraits;

    let field = seg
        .field(field_name)
        .ok_or_else(|| Error::invalid(format!("field {field_name} not found")))?;
    if !field.indexed {
        return Err(Error::invalid(format!("{field_name} has no postings")));
    }
    let traits = FieldTraits {
        has_freqs: field.index_options >= 2,
        has_positions: field.index_options >= 3,
        has_offsets: field.index_options >= 4,
    };

    let read = |ext: &str| -> Result<Vec<u8>> {
        let name = seg
            .files
            .iter()
            .find(|f| f.ends_with(ext))
            .ok_or_else(|| Error::invalid(format!("no {ext} in segment {}", seg.name)))?;
        let input = dir.open_input(&seg.name, name)?;
        input
            .slice(0, input.len())
            .map(|b| b.to_vec())
            .ok_or_else(|| Error::invalid(format!("cannot slice {name}")))
    };
    let tmd = read(".tmd")?;
    let tim = read(".tim")?;
    let tip = read(".tip")?;
    let doc = read(".doc")?;

    let metas = lucene_arrow_postings::parse_tmd(&tmd, |n| {
        seg.fields.iter().find(|f| f.number == n).map(|f| f.index_options >= 2).unwrap_or(false)
    })?;
    let meta = metas
        .iter()
        .find(|m| m.field_number == field.number)
        .ok_or_else(|| Error::invalid(format!("{field_name} not in terms dict")))?;
    let tip_slice = &tip[meta.index_start_fp as usize..meta.index_end_fp as usize];
    let root = lucene_arrow_postings::root_block(tip_slice, meta.trie_root_fp)?;
    let csr = lucene_arrow_postings::coo::read_csr(&tim, &doc, root.fp, traits)?;

    // Shared dictionary values (the sorted term set).
    let mut values = BinaryBuilder::new();
    for t in 0..csr.num_terms() {
        values.append_value(csr.term(t));
    }
    let values = std::sync::Arc::new(values.finish());

    let ords = csr.term_ords();
    let schema = std::sync::Arc::new(arrow_schema::Schema::new(vec![
        arrow_schema::Field::new(
            "term",
            arrow_schema::DataType::Dictionary(
                Box::new(arrow_schema::DataType::Int32),
                Box::new(arrow_schema::DataType::Binary),
            ),
            false,
        ),
        arrow_schema::Field::new("doc", arrow_schema::DataType::UInt32, false),
        arrow_schema::Field::new("freq", arrow_schema::DataType::UInt32, false),
    ]));

    let mut out = Vec::new();
    let n = csr.num_rows();
    let mut start = 0usize;
    while start < n {
        let len = batch_rows.min(n - start);
        let keys = arrow_array::Int32Array::from_iter_values(
            ords[start..start + len].iter().map(|&o| o as i32),
        );
        let term = DictionaryArray::try_new(keys, values.clone())
            .map_err(|e| Error::Codec(e.to_string()))?;
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                std::sync::Arc::new(term),
                std::sync::Arc::new(UInt32Array::from_iter_values(
                    csr.docs[start..start + len].iter().copied(),
                )),
                std::sync::Arc::new(UInt32Array::from_iter_values(
                    csr.freqs[start..start + len].iter().copied(),
                )),
            ],
        )
        .map_err(|e| Error::Codec(e.to_string()))?;
        out.push(batch);
        start += len;
    }
    Ok(out)
}
