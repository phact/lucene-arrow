// SPDX-License-Identifier: Apache-2.0

//! `DoAction: "hydrate"` (SPEC §7.4): stored fields for `(_seg, _doc)`
//! pairs, returned as one Arrow IPC-encoded RecordBatch. Row-oriented,
//! LZ4-block-compressed data whose analytic role is hydrating *filtered*
//! results — hence an RPC keyed by row address, never a scan column.
//! (GPU nvCOMP path is decision register #8; CPU LZ4 until that gate.)

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_array::builder::{BinaryBuilder, Float64Builder, Int64Builder, StringBuilder};
use arrow_array::{ArrayRef, Int32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};

use crate::request::HydrateRequest;
use lucene_arrow_codec::{SegmentDirectory, StoredVal};
use lucene_arrow_core::{Error, Result};

/// Run a hydrate request and IPC-encode the resulting batch.
pub fn hydrate_ipc(request: &HydrateRequest) -> Result<Vec<u8>> {
    let batch = hydrate(request)?;
    let mut out = Vec::new();
    {
        let mut writer = arrow_ipc::writer::StreamWriter::try_new(&mut out, &batch.schema())?;
        writer.write(&batch)?;
        writer.finish()?;
    }
    Ok(out)
}

/// Column values arrive typed per doc; a column's type is fixed by its
/// first non-null value (stored fields carry their type per value).
enum ColBuf {
    Unknown(usize),          // nulls seen so far
    Str(StringBuilder),
    Bytes(BinaryBuilder),
    I64(Int64Builder),
    F64(Float64Builder),
}

pub fn hydrate(request: &HydrateRequest) -> Result<RecordBatch> {
    let dir = SegmentDirectory::open(&request.path)?;

    // Group pair indices by segment so each reader opens once, preserving
    // the caller's pair order in the output.
    let mut by_seg: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for (i, &(seg, _)) in request.pairs.iter().enumerate() {
        by_seg.entry(seg).or_default().push(i);
    }

    // row → (field name → value)
    let mut rows: Vec<BTreeMap<String, StoredVal>> =
        vec![BTreeMap::new(); request.pairs.len()];
    for (&seg_ord, idxs) in &by_seg {
        let seg = dir
            .segments()
            .get(seg_ord as usize)
            .ok_or_else(|| Error::invalid(format!("no segment ordinal {seg_ord}")))?;
        let docs: Vec<u32> = idxs.iter().map(|&i| request.pairs[i].1).collect();
        let field_name = |num: u32| -> Option<&str> {
            seg.fields.iter().find(|f| f.number == num).map(|f| f.name.as_str())
        };
        for (row_vals, &i) in dir.stored_documents(&seg.name, &docs)?.into_iter().zip(idxs) {
            for (num, val) in row_vals {
                if let Some(name) = field_name(num)
                    && request.columns.iter().any(|c| c == name)
                {
                    rows[i].insert(name.to_string(), val);
                }
            }
        }
    }

    // Build columns: _seg, _doc, then requested stored columns.
    let n = request.pairs.len();
    let mut names: Vec<String> = vec!["_seg".into(), "_doc".into()];
    let mut arrays: Vec<ArrayRef> = vec![
        Arc::new(Int32Array::from_iter_values(request.pairs.iter().map(|p| p.0 as i32))),
        Arc::new(Int32Array::from_iter_values(request.pairs.iter().map(|p| p.1 as i32))),
    ];

    for col in &request.columns {
        let mut buf = ColBuf::Unknown(0);
        for row in &rows {
            let v = row.get(col);
            // Promote Unknown on first typed value.
            if let ColBuf::Unknown(nulls) = buf
                && let Some(v) = v
            {
                let mut promoted = match v {
                    StoredVal::Str(_) => ColBuf::Str(StringBuilder::new()),
                    StoredVal::Bytes(_) => ColBuf::Bytes(BinaryBuilder::new()),
                    StoredVal::I64(_) => ColBuf::I64(Int64Builder::new()),
                    StoredVal::F64(_) => ColBuf::F64(Float64Builder::new()),
                };
                for _ in 0..nulls {
                    match &mut promoted {
                        ColBuf::Str(b) => b.append_null(),
                        ColBuf::Bytes(b) => b.append_null(),
                        ColBuf::I64(b) => b.append_null(),
                        ColBuf::F64(b) => b.append_null(),
                        ColBuf::Unknown(_) => unreachable!(),
                    }
                }
                buf = promoted;
            }
            match (&mut buf, v) {
                (ColBuf::Unknown(nulls), None) => *nulls += 1,
                (ColBuf::Str(b), Some(StoredVal::Str(s))) => b.append_value(s),
                (ColBuf::Str(b), None) => b.append_null(),
                (ColBuf::Bytes(b), Some(StoredVal::Bytes(x))) => b.append_value(x),
                (ColBuf::Bytes(b), None) => b.append_null(),
                (ColBuf::I64(b), Some(StoredVal::I64(x))) => b.append_value(*x),
                (ColBuf::I64(b), None) => b.append_null(),
                (ColBuf::F64(b), Some(StoredVal::F64(x))) => b.append_value(*x),
                (ColBuf::F64(b), None) => b.append_null(),
                (_, Some(other)) => {
                    return Err(Error::invalid(format!(
                        "column {col:?} mixes stored types (saw {other:?})"
                    )));
                }
            }
        }
        let array: ArrayRef = match buf {
            ColBuf::Unknown(_) => arrow_array::new_null_array(&DataType::Utf8, n),
            ColBuf::Str(mut b) => Arc::new(b.finish()),
            ColBuf::Bytes(mut b) => Arc::new(b.finish()),
            ColBuf::I64(mut b) => Arc::new(b.finish()),
            ColBuf::F64(mut b) => Arc::new(b.finish()),
        };
        names.push(col.clone());
        arrays.push(array);
    }

    let fields: Vec<Field> = names
        .iter()
        .zip(&arrays)
        .map(|(name, a)| {
            Field::new(name, a.data_type().clone(), name != "_seg" && name != "_doc")
        })
        .collect();
    Ok(RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays)?)
}
