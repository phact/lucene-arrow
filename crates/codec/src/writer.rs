// SPDX-License-Identifier: Apache-2.0

//! Segment assembly (SPEC §10.5): our encoders' payloads + the container
//! files a complete Lucene segment needs. `.dvd`/`.dvm` come from
//! `lucene-arrow-docvalues` (byte-identical to Java); this module emits
//! `.fnm` (inverse of our parser), `.si`, and the empty stored-fields trio
//! (`.fdt`/`.fdx`/`.fdm` — Lucene opens stored fields unconditionally, so
//! they must exist even when nothing is stored); `segments_N` goes through
//! Bearing's public `SegmentInfos` commit. Acceptance: Java `CheckIndex`
//! passes on every output segment (SPEC §10.5, gated in tests).

use std::collections::BTreeMap;
use std::path::Path;

use bearing::encoding::lz4;
use bearing::index::segment::{FlushedSegment, SegmentId};
use bearing::index::segment_infos::SegmentInfos;
use bearing::store::FSDirectory;

use lucene_arrow_core::cursor::{write_footer, write_index_header, write_vint};
use lucene_arrow_core::{Error, Result};

/// The Lucene version stamped into `.si` — matches the codec pin (§3.2).
pub const LUCENE_VERSION: (i32, i32, i32) = (10, 3, 2);

/// One field entry for `.fnm` writing (subset our writer emits: doc-values
/// only fields, no postings/points/vectors yet).
#[derive(Debug, Clone)]
pub struct WriteField {
    pub name: String,
    pub number: u32,
    /// On-disk doc-values byte (0=NONE, 1=NUMERIC, 2=BINARY, 3=SORTED,
    /// 4=SORTED_SET, 5=SORTED_NUMERIC).
    pub doc_values_type: u8,
    /// Vector shape (0 dim = no vectors). Encoding/similarity are Lucene
    /// ordinals; vector fields get PerFieldKnnVectorsFormat routing
    /// attributes for Lucene99HnswVectorsFormat.
    pub vector_dim: u32,
    pub vector_encoding: u8,
    pub vector_similarity: u8,
    /// Lucene index-options byte (0 NONE … 2 DOCS_AND_FREQS). Indexed
    /// fields get PerFieldPostingsFormat routing attributes and norms.
    pub index_options: u8,
}

impl WriteField {
    pub fn doc_values(name: impl Into<String>, number: u32, doc_values_type: u8) -> Self {
        WriteField {
            name: name.into(),
            number,
            doc_values_type,
            vector_dim: 0,
            vector_encoding: 0,
            vector_similarity: 0,
            index_options: 0,
        }
    }
}

/// Lucene94FieldInfos writer — exact inverse of `crate::fnm::parse` for
/// the doc-values-only subset.
pub fn write_fnm(fields: &[WriteField], segment_id: &[u8; 16]) -> Vec<u8> {
    let mut out = Vec::new();
    write_index_header(&mut out, crate::fnm::CODEC_NAME, crate::fnm::FORMAT_CURRENT, segment_id, "");
    write_vint(&mut out, fields.len() as u32);
    for f in fields {
        write_vint(&mut out, f.name.len() as u32);
        out.extend_from_slice(f.name.as_bytes());
        write_vint(&mut out, f.number);
        out.push(0); // fieldBits: nothing stored/omitted/soft-deleted
        out.push(f.index_options);
        out.push(f.doc_values_type);
        out.push(0); // docValuesSkipIndexType = NONE
        out.extend_from_slice(&(-1i64).to_le_bytes()); // docValuesGen
        // Attributes: per-field DV format routing (required — the reader
        // silently ignores the field without them).
        let mut attrs: BTreeMap<&str, &str> = BTreeMap::new();
        if f.doc_values_type != 0 {
            attrs.insert("PerFieldDocValuesFormat.format", "Lucene90");
            attrs.insert("PerFieldDocValuesFormat.suffix", "0");
        }
        if f.vector_dim > 0 {
            attrs.insert("PerFieldKnnVectorsFormat.format", "Lucene99HnswVectorsFormat");
            attrs.insert("PerFieldKnnVectorsFormat.suffix", "0");
        }
        if f.index_options > 0 {
            attrs.insert("PerFieldPostingsFormat.format", "Lucene103");
            attrs.insert("PerFieldPostingsFormat.suffix", "0");
        }
        write_vint(&mut out, attrs.len() as u32);
        for (k, v) in attrs {
            write_vint(&mut out, k.len() as u32);
            out.extend_from_slice(k.as_bytes());
            write_vint(&mut out, v.len() as u32);
            out.extend_from_slice(v.as_bytes());
        }
        write_vint(&mut out, 0); // pointDimensionCount
        write_vint(&mut out, f.vector_dim);
        out.push(if f.vector_dim > 0 { f.vector_encoding } else { 1 });
        out.push(f.vector_similarity);
    }
    write_footer(&mut out);
    out
}

/// Lucene99SegmentInfoFormat writer (codec name "Lucene90SegmentInfo").
pub fn write_si(
    segment_id: &[u8; 16],
    max_doc: u32,
    files: &[String],
    diagnostics: &BTreeMap<String, String>,
    attributes: &BTreeMap<String, String>,
) -> Vec<u8> {
    let mut out = Vec::new();
    write_index_header(&mut out, "Lucene90SegmentInfo", 0, segment_id, "");
    let (maj, min, bug) = LUCENE_VERSION;
    out.extend_from_slice(&maj.to_le_bytes());
    out.extend_from_slice(&min.to_le_bytes());
    out.extend_from_slice(&bug.to_le_bytes());
    out.push(1); // hasMinVersion
    out.extend_from_slice(&maj.to_le_bytes());
    out.extend_from_slice(&min.to_le_bytes());
    out.extend_from_slice(&bug.to_le_bytes());
    out.extend_from_slice(&(max_doc as i32).to_le_bytes());
    out.push(0xFF); // isCompoundFile = NO (-1)
    out.push(0xFF); // hasBlocks = NO
    let write_map = |out: &mut Vec<u8>, m: &BTreeMap<String, String>| {
        write_vint(out, m.len() as u32);
        for (k, v) in m {
            write_vint(out, k.len() as u32);
            out.extend_from_slice(k.as_bytes());
            write_vint(out, v.len() as u32);
            out.extend_from_slice(v.as_bytes());
        }
    };
    write_map(&mut out, diagnostics);
    write_vint(&mut out, files.len() as u32);
    for f in files {
        write_vint(&mut out, f.len() as u32);
        out.extend_from_slice(f.as_bytes());
    }
    write_map(&mut out, attributes);
    write_vint(&mut out, 0); // numSortFields
    write_footer(&mut out);
    out
}

// Stored-fields constants (Lucene90, BEST_SPEED — see Bearing's writer).
const CHUNK_SIZE: i32 = 10 * 8 * 1024;
const MAX_DOCS_PER_CHUNK: u32 = 1024;
const FDX_BLOCK_SHIFT: u32 = 10;

/// Empty stored-fields trio: every doc has zero stored fields. Lucene
/// opens stored fields unconditionally, so these are mandatory framing.
pub fn write_empty_stored_fields(num_docs: u32, segment_id: &[u8; 16]) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut fdt = Vec::new();
    let mut fdx = Vec::new();
    let mut fdm = Vec::new();
    write_index_header(&mut fdt, "Lucene90StoredFieldsFastData", 1, segment_id, "");
    write_index_header(&mut fdx, "Lucene90FieldsIndexIdx", 0, segment_id, "");
    write_index_header(&mut fdm, "Lucene90FieldsIndexMeta", 1, segment_id, "");
    write_vint(&mut fdm, CHUNK_SIZE as u32);

    // Chunks of ≤1024 empty docs; the last one is a forced ("dirty") flush.
    let mut doc_counts = Vec::new();
    let mut start_pointers = Vec::new();
    let mut ht = lz4::FastHashTable::new();
    let empty_lz4 = lz4::compress_reuse(&[], &mut ht);
    let mut doc_base = 0u32;
    while doc_base < num_docs {
        let n = (num_docs - doc_base).min(MAX_DOCS_PER_CHUNK);
        let dirty = doc_base + n == num_docs; // final forced flush
        start_pointers.push(fdt.len() as i64);
        doc_counts.push(n);
        write_vint(&mut fdt, doc_base);
        write_vint(&mut fdt, (n << 2) | u32::from(dirty));
        // numStoredFields then lengths — all zeros (save_ints layout:
        // single value → bare vint; else marker 0 + vint).
        for _ in 0..2 {
            if n == 1 {
                write_vint(&mut fdt, 0);
            } else {
                fdt.push(0);
                write_vint(&mut fdt, 0);
            }
        }
        // LZ4-with-preset-dict framing of the empty doc buffer.
        write_vint(&mut fdt, 0); // dictLength
        write_vint(&mut fdt, 0); // blockLength
        write_vint(&mut fdt, empty_lz4.len() as u32);
        fdt.extend_from_slice(&empty_lz4);
        doc_base += n;
    }
    let max_pointer = fdt.len() as i64;
    let num_chunks = doc_counts.len() as i64;
    let (num_dirty_chunks, num_dirty_docs) = if num_docs == 0 {
        (0i64, 0i64)
    } else {
        (1, *doc_counts.last().expect("non-empty") as i64)
    };

    // FieldsIndexWriter.finish
    fdm.extend_from_slice(&(num_docs as i32).to_le_bytes());
    fdm.extend_from_slice(&(FDX_BLOCK_SHIFT as i32).to_le_bytes());
    fdm.extend_from_slice(&((num_chunks + 1) as i32).to_le_bytes());
    fdm.extend_from_slice(&(fdx.len() as i64).to_le_bytes());
    let mut docs_seq = Vec::with_capacity(doc_counts.len() + 1);
    let mut acc = 0i64;
    docs_seq.push(0);
    for &c in &doc_counts {
        acc += c as i64;
        docs_seq.push(acc);
    }
    lucene_arrow_docvalues::monotonic::write(&docs_seq, FDX_BLOCK_SHIFT, &mut fdm, &mut fdx)
        .expect("monotone by construction");
    fdm.extend_from_slice(&(fdx.len() as i64).to_le_bytes());
    let mut fp_seq = start_pointers;
    fp_seq.push(max_pointer);
    lucene_arrow_docvalues::monotonic::write(&fp_seq, FDX_BLOCK_SHIFT, &mut fdm, &mut fdx)
        .expect("monotone by construction");
    fdm.extend_from_slice(&(fdx.len() as i64).to_le_bytes());
    write_footer(&mut fdx);

    fdm.extend_from_slice(&max_pointer.to_le_bytes());
    let vlong = |out: &mut Vec<u8>, v: i64| {
        lucene_arrow_core::cursor::write_vlong(out, v as u64)
    };
    vlong(&mut fdm, num_chunks);
    vlong(&mut fdm, num_dirty_chunks);
    vlong(&mut fdm, num_dirty_docs);
    write_footer(&mut fdm);
    write_footer(&mut fdt);
    (fdt, fdx, fdm)
}

/// Pseudo-random 16-byte segment id (uniqueness, not cryptography).
pub fn random_segment_id() -> [u8; 16] {
    use std::hash::{BuildHasher, Hasher, RandomState};
    let mut id = [0u8; 16];
    let s = RandomState::new();
    let mut h1 = s.build_hasher();
    h1.write_u64(std::process::id() as u64);
    let mut h2 = s.build_hasher();
    h2.write_u128(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0),
    );
    id[..8].copy_from_slice(&h1.finish().to_le_bytes());
    id[8..].copy_from_slice(&h2.finish().to_le_bytes());
    id
}

/// One written (not yet committed) segment, ready for [`commit_segments`].
#[derive(Debug, Clone)]
pub struct WrittenSegment {
    pub name: String,
    pub id: [u8; 16],
    pub num_docs: u32,
    pub files: Vec<String>,
}

/// Commit one or more written segments as a single `segments_N` point
/// (via Bearing's public `SegmentInfos` writer). Segment order = docid
/// base order (SPEC §7.1).
pub fn commit_segments(dir_path: &Path, segments: &[WrittenSegment]) -> Result<()> {
    let io = |e: std::io::Error| Error::Codec(e.to_string());
    let directory = FSDirectory::open(dir_path).map_err(io)?;
    let mut infos = SegmentInfos::new();
    for seg in segments {
        infos.add(FlushedSegment {
            segment_id: SegmentId { name: seg.name.clone(), id: seg.id },
            doc_count: seg.num_docs as i32,
            file_names: seg.files.clone(),
        });
    }
    infos.commit(&*directory).map_err(io)?;
    Ok(())
}

/// Write one segment's files (no commit): pre-encoded doc-values pair +
/// our `.fnm`/`.si`/empty-stored-fields framing.
pub fn write_segment_files(
    dir_path: &Path,
    segment_name: &str,
    segment_id: [u8; 16],
    fields: &[WriteField],
    num_docs: u32,
    dvm: &[u8],
    dvd: &[u8],
) -> Result<WrittenSegment> {
    write_segment_files_ext(dir_path, segment_name, segment_id, fields, num_docs, dvm, dvd, &[])
}

/// [`write_segment_files_ext`] plus names of files ALREADY on disk in
/// `dir_path` (e.g. postings written by Bearing's writer) to include in
/// the segment's file list.
#[allow(clippy::too_many_arguments)]
pub fn write_segment_files_full(
    dir_path: &Path,
    segment_name: &str,
    segment_id: [u8; 16],
    fields: &[WriteField],
    num_docs: u32,
    dvm: &[u8],
    dvd: &[u8],
    extra: &[(String, &[u8])],
    existing: &[String],
) -> Result<WrittenSegment> {
    let mut seg = write_segment_files_ext(
        dir_path, segment_name, segment_id, fields, num_docs, dvm, dvd, extra,
    )?;
    for name in existing {
        seg.files.push(name.clone());
    }
    seg.files.sort();
    rewrite_si(dir_path, segment_name, segment_id, num_docs, &seg)?;
    Ok(seg)
}

/// [`write_segment_files`] plus extra per-segment files (vector formats
/// etc.) as `(full_file_name, bytes)` pairs.
#[allow(clippy::too_many_arguments)]
pub fn write_segment_files_ext(
    dir_path: &Path,
    segment_name: &str,
    segment_id: [u8; 16],
    fields: &[WriteField],
    num_docs: u32,
    dvm: &[u8],
    dvd: &[u8],
    extra: &[(String, &[u8])],
) -> Result<WrittenSegment> {
    let io = |e: std::io::Error| Error::Codec(e.to_string());
    let write = |name: &str, bytes: &[u8]| -> Result<()> {
        std::fs::write(dir_path.join(name), bytes).map_err(io)
    };

    let dvd_name = format!("{segment_name}_Lucene90_0.dvd");
    let dvm_name = format!("{segment_name}_Lucene90_0.dvm");
    let fnm_name = format!("{segment_name}.fnm");
    let fdt_name = format!("{segment_name}.fdt");
    let fdx_name = format!("{segment_name}.fdx");
    let fdm_name = format!("{segment_name}.fdm");
    let si_name = format!("{segment_name}.si");

    write(&dvd_name, dvd)?;
    write(&dvm_name, dvm)?;
    write(&fnm_name, &write_fnm(fields, &segment_id))?;
    let (fdt, fdx, fdm) = write_empty_stored_fields(num_docs, &segment_id);
    write(&fdt_name, &fdt)?;
    write(&fdx_name, &fdx)?;
    write(&fdm_name, &fdm)?;

    let mut files = vec![
        dvd_name, dvm_name, fnm_name, fdt_name, fdx_name, fdm_name, si_name.clone(),
    ];
    for (name, bytes) in extra {
        write(name, bytes)?;
        files.push(name.clone());
    }
    let diagnostics = BTreeMap::from([
        ("source".to_string(), "flush".to_string()),
        ("lucene.version".to_string(), "10.3.2".to_string()),
    ]);
    let attributes = BTreeMap::from([(
        "Lucene90StoredFieldsFormat.mode".to_string(),
        "BEST_SPEED".to_string(),
    )]);
    write(
        &si_name,
        &write_si(&segment_id, num_docs, &files, &diagnostics, &attributes),
    )?;

    Ok(WrittenSegment {
        name: segment_name.to_string(),
        id: segment_id,
        num_docs,
        files,
    })
}

fn rewrite_si(
    dir_path: &Path,
    segment_name: &str,
    segment_id: [u8; 16],
    num_docs: u32,
    seg: &WrittenSegment,
) -> Result<()> {
    let diagnostics = BTreeMap::from([
        ("source".to_string(), "flush".to_string()),
        ("lucene.version".to_string(), "10.3.2".to_string()),
    ]);
    let attributes = BTreeMap::from([(
        "Lucene90StoredFieldsFormat.mode".to_string(),
        "BEST_SPEED".to_string(),
    )]);
    std::fs::write(
        dir_path.join(format!("{segment_name}.si")),
        write_si(&segment_id, num_docs, &seg.files, &diagnostics, &attributes),
    )?;
    Ok(())
}

/// Single-segment convenience: write files + commit in one call.
pub fn write_segment_commit(
    dir_path: &Path,
    segment_name: &str,
    segment_id: [u8; 16],
    fields: &[WriteField],
    num_docs: u32,
    dvm: &[u8],
    dvd: &[u8],
) -> Result<Vec<String>> {
    let seg = write_segment_files(dir_path, segment_name, segment_id, fields, num_docs, dvm, dvd)?;
    commit_segments(dir_path, std::slice::from_ref(&seg))?;
    Ok(seg.files)
}
