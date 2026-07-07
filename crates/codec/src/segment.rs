// SPDX-License-Identifier: Apache-2.0

//! Typed segment inventory over Bearing's Lucene103 parsers (SPEC §5, §13 P0).
//!
//! [`SegmentDirectory::open`] walks a Lucene segment directory the way a
//! reader does: latest `segments_N` commit → per-segment `.si` → `.fnm`
//! (through the `.cfs` compound directory when the segment is compound),
//! and materializes [`SegmentMeta`]/[`FieldMeta`]. The codec is asserted to
//! be the v1 pin, [`PINNED_CODEC`] (SPEC §3.2); anything else is
//! [`Error::Unsupported`].
//!
//! [`open_input`](SegmentDirectory::open_input) is the downstream byte
//! door: it resolves a segment file (compound or not) to a
//! [`ByteRange`], which is how the docvalues/vectors crates get their
//! `.dvm`/`.dvd`/`.vec` bytes without knowing about `.cfs`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use bearing::codecs::lucene94::field_infos_format;
use bearing::codecs::lucene99::segment_info_format;
use bearing::document::DocValuesType;
use bearing::index::segment_infos;
use bearing::store::{
    CompoundDirectory, Directory, FSDirectory, FileBacking, SharedDirectory,
};
use lucene_arrow_core::{BufferTarget, ByteRange, Error, Result};

/// The one codec v1 reads (SPEC §3.2): Lucene 10.x / OpenSearch 3.x.
pub const PINNED_CODEC: &str = "Lucene103";

/// Doc-values type of one field, decoupled from Bearing's enum so
/// downstream crates never import Bearing types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DocValuesKind {
    None,
    Numeric,
    Binary,
    Sorted,
    SortedNumeric,
    SortedSet,
}

impl DocValuesKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            DocValuesKind::None => "none",
            DocValuesKind::Numeric => "numeric",
            DocValuesKind::Binary => "binary",
            DocValuesKind::Sorted => "sorted",
            DocValuesKind::SortedNumeric => "sorted_numeric",
            DocValuesKind::SortedSet => "sorted_set",
        }
    }
}

impl std::fmt::Display for DocValuesKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl From<DocValuesType> for DocValuesKind {
    fn from(dv: DocValuesType) -> Self {
        match dv {
            DocValuesType::None => DocValuesKind::None,
            DocValuesType::Numeric => DocValuesKind::Numeric,
            DocValuesType::Binary => DocValuesKind::Binary,
            DocValuesType::Sorted => DocValuesKind::Sorted,
            DocValuesType::SortedNumeric => DocValuesKind::SortedNumeric,
            DocValuesType::SortedSet => DocValuesKind::SortedSet,
        }
    }
}

/// A stored-field value, decoupled from Bearing's enum.
#[derive(Debug, Clone, PartialEq)]
pub enum StoredVal {
    Str(String),
    Bytes(Vec<u8>),
    I64(i64),
    F64(f64),
}

impl From<bearing::document::StoredValue> for StoredVal {
    fn from(v: bearing::document::StoredValue) -> Self {
        use bearing::document::StoredValue as S;
        match v {
            S::String(s) => StoredVal::Str(s),
            S::Bytes(b) => StoredVal::Bytes(b),
            S::Int(i) => StoredVal::I64(i as i64),
            S::Long(l) => StoredVal::I64(l),
            S::Float(f) => StoredVal::F64(f as f64),
            S::Double(d) => StoredVal::F64(d),
        }
    }
}

/// One field of one segment, from `.fnm` (SPEC §7.5: the schema source).
#[derive(Debug, Clone)]
pub struct FieldMeta {
    pub name: String,
    pub number: u32,
    pub doc_values: DocValuesKind,
    /// True when the field has postings (index options != NONE).
    pub indexed: bool,
    /// Lucene index-options byte (0 NONE, 1 DOCS, 2 +FREQS, 3 +POSITIONS,
    /// 4 +OFFSETS) from our own `.fnm` parse.
    pub index_options: u8,
    pub has_norms: bool,
    pub has_points: bool,
    pub has_vectors: bool,
    /// Vector shape from our own `.fnm` parse (Bearing discards these):
    /// 0 when the field has no vectors; encoding/similarity are Lucene
    /// ordinals (see `lucene_arrow_vectors`).
    pub vector_dimension: u32,
    pub vector_encoding: u8,
    pub vector_similarity: u8,
    /// Doc-values skip index present — changes `.dvm` entry layout
    /// (Lucene 10 `DocValuesSkipIndexType != NONE`).
    pub has_skip_index: bool,
    pub attributes: BTreeMap<String, String>,
}

/// One segment of a commit: `segments_N` entry joined with its `.si`
/// and `.fnm`.
#[derive(Debug, Clone)]
pub struct SegmentMeta {
    /// Position within the commit (docid-base ordering, SPEC §7.1).
    pub ord: usize,
    /// Segment name, e.g. `_0`.
    pub name: String,
    /// 16-byte segment id; every per-segment file header repeats it.
    pub id: [u8; 16],
    /// Codec name; always [`PINNED_CODEC`] once `open` succeeds.
    pub codec: String,
    pub max_doc: i32,
    pub del_count: i32,
    /// Deletes generation (-1 = none); names the `.liv` file.
    pub del_gen: i64,
    pub is_compound: bool,
    /// Files belonging to the segment (sorted). For compound segments this
    /// is the outer view (`.si`/`.cfs`/`.cfe`), not the `.cfs` contents.
    pub files: Vec<String>,
    pub fields: Vec<FieldMeta>,
}

impl SegmentMeta {
    pub fn field(&self, name: &str) -> Option<&FieldMeta> {
        self.fields.iter().find(|f| f.name == name)
    }
}

/// An opened Lucene segment directory: parsed inventory + byte access.
pub struct SegmentDirectory {
    path: PathBuf,
    dir: SharedDirectory,
    generation: i64,
    segments: Vec<SegmentMeta>,
}

/// Stringify a Bearing/io failure at the crate boundary (core does not
/// depend on bearing).
fn codec_err(e: std::io::Error) -> Error {
    Error::Codec(e.to_string())
}

/// The v1 codec assertion (SPEC §3.2): anything but Lucene103 is out of scope.
fn ensure_pinned_codec(segment: &str, codec: &str) -> Result<()> {
    if codec == PINNED_CODEC {
        Ok(())
    } else {
        Err(Error::unsupported(format!(
            "segment {segment} uses codec {codec:?}; only {PINNED_CODEC:?} is supported (SPEC §3.2)"
        )))
    }
}

/// Files of a compound segment that still live outside the `.cfs`:
/// the compound pair itself, the `.si`, and any live-docs/updates gens.
fn stored_outside_cfs(file_name: &str) -> bool {
    [".si", ".cfs", ".cfe", ".liv"]
        .iter()
        .any(|ext| file_name.ends_with(ext))
}

impl SegmentDirectory {
    /// Opens the latest commit (`segments_N` with the highest generation)
    /// of a Lucene segment directory and parses the full inventory.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if !path.is_dir() {
            return Err(Error::invalid(format!(
                "not a directory: {}",
                path.display()
            )));
        }
        let dir: SharedDirectory = FSDirectory::open(path).map_err(codec_err)?;

        let files = dir.list_all().map_err(codec_err)?;
        let segments_file =
            segment_infos::get_last_commit_segments_file_name(&files).map_err(|e| {
                Error::invalid(format!(
                    "{}: not a Lucene segment directory ({e})",
                    path.display()
                ))
            })?;
        // Verifies the segments_N CRC internally.
        let infos = segment_infos::read(&*dir, &segments_file).map_err(codec_err)?;

        let mut segments = Vec::with_capacity(infos.segments.len());
        for (ord, entry) in infos.segments.iter().enumerate() {
            ensure_pinned_codec(&entry.name, &entry.codec)?;

            let si = segment_info_format::read(&*dir, &entry.name, &entry.id)
                .map_err(codec_err)?;

            let fnm_name = format!("{}.fnm", si.name);
            let (field_infos, fnm_bytes) = if si.is_compound_file {
                let cfs =
                    CompoundDirectory::open(&*dir, &si.name, &si.id).map_err(codec_err)?;
                let fi = field_infos_format::read(&cfs, &si, "").map_err(codec_err)?;
                let bytes = cfs.open_file(&fnm_name).map_err(codec_err)?.as_bytes().to_vec();
                (fi, bytes)
            } else {
                let fi = field_infos_format::read(&*dir, &si, "").map_err(codec_err)?;
                let bytes = dir.open_file(&fnm_name).map_err(codec_err)?.as_bytes().to_vec();
                (fi, bytes)
            };
            // Our own parse fills what Bearing discards (vector shape,
            // skip-index flag) — and cross-checks the shared fields.
            let fnm_fields = crate::fnm::parse(&fnm_bytes)?;

            let mut fields: Vec<FieldMeta> = field_infos
                .iter()
                .map(|fi| {
                    let own = fnm_fields.iter().find(|f| f.number == fi.number());
                    let own = own.ok_or_else(|| {
                        Error::corrupt(format!(
                            "field {} in Bearing's .fnm parse but not ours",
                            fi.number()
                        ))
                    })?;
                    debug_assert_eq!(own.name, fi.name());
                    Ok(FieldMeta {
                        name: fi.name().to_string(),
                        number: fi.number(),
                        doc_values: fi.doc_values_type().into(),
                        indexed: fi.is_indexed(),
                        index_options: own.index_options,
                        has_norms: fi.has_norms(),
                        has_points: fi.has_point_values(),
                        has_vectors: own.vector_dimension > 0,
                        vector_dimension: own.vector_dimension,
                        vector_encoding: own.vector_encoding,
                        vector_similarity: own.vector_similarity,
                        has_skip_index: own.doc_values_skip_index,
                        attributes: fi
                            .attributes()
                            .iter()
                            .map(|(k, v)| (k.clone(), v.clone()))
                            .collect(),
                    })
                })
                .collect::<Result<_>>()?;
            fields.sort_by_key(|f| f.number);

            let mut seg_files: Vec<String> = si.files.iter().cloned().collect();
            seg_files.sort();

            segments.push(SegmentMeta {
                ord,
                name: si.name.clone(),
                id: si.id,
                codec: entry.codec.clone(),
                max_doc: si.max_doc,
                del_count: entry.del_count,
                del_gen: entry.del_gen,
                is_compound: si.is_compound_file,
                files: seg_files,
                fields,
            });
        }

        Ok(SegmentDirectory {
            path: path.to_path_buf(),
            dir,
            generation: infos.generation,
            segments,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Generation `N` of the `segments_N` commit that was opened.
    pub fn generation(&self) -> i64 {
        self.generation
    }

    /// Segments of the commit, in docid-base order.
    pub fn segments(&self) -> &[SegmentMeta] {
        &self.segments
    }

    pub fn segment(&self, name: &str) -> Option<&SegmentMeta> {
        self.segments.iter().find(|s| s.name == name)
    }

    /// Opens the raw bytes of one file of one segment, resolving through
    /// the `.cfs` compound directory when needed. This is how downstream
    /// crates fetch `.dvm`/`.dvd`/`.vec` payloads (SPEC §5, §9).
    /// Live-docs bitmap (1 = live) for a segment, `None` when it has no
    /// deletes. Words are LSB-first, `ceil(max_doc/64)` long.
    pub fn live_docs(&self, segment: &str) -> Result<Option<Vec<u64>>> {
        let meta = self
            .segment(segment)
            .ok_or_else(|| Error::invalid(format!("no segment {segment:?} in commit")))?;
        if meta.del_gen < 0 || meta.del_count == 0 {
            return Ok(None);
        }
        let name = crate::liv::file_name(&meta.name, meta.del_gen);
        let range = self.open_input(segment, &name)?;
        let bytes = range
            .slice(0, range.len())
            .ok_or_else(|| Error::unsupported("non-mmap source for .liv"))?;
        crate::liv::read_live_docs(bytes, meta.max_doc as u32, &meta.id, meta.del_gen, meta.del_count)
            .map(Some)
    }

    /// Fetch stored fields for a batch of docs in one segment (SPEC §7.4
    /// hydration: row-oriented, block-compressed — never in the scan
    /// stream). The reader is opened once per call; callers batch pairs.
    pub fn stored_documents(
        &self,
        segment: &str,
        docs: &[u32],
    ) -> Result<Vec<Vec<(u32, StoredVal)>>> {
        use bearing::codecs::lucene90::stored_fields_reader::StoredFieldsReader;
        let meta = self
            .segment(segment)
            .ok_or_else(|| Error::invalid(format!("no segment {segment:?} in commit")))?;
        let mut out = Vec::with_capacity(docs.len());
        let mut read_all = |reader: &mut StoredFieldsReader| -> Result<()> {
            for &d in docs {
                if d >= meta.max_doc as u32 {
                    return Err(Error::invalid(format!("doc {d} >= max_doc {}", meta.max_doc)));
                }
                let fields = reader.document(d).map_err(codec_err)?;
                out.push(
                    fields
                        .into_iter()
                        .map(|f| (f.field_number, StoredVal::from(f.value)))
                        .collect(),
                );
            }
            Ok(())
        };
        if meta.is_compound {
            let cfs = CompoundDirectory::open(&*self.dir, &meta.name, &meta.id)
                .map_err(codec_err)?;
            let mut reader = StoredFieldsReader::open(&cfs, &meta.name, "", &meta.id)
                .map_err(codec_err)?;
            read_all(&mut reader)?;
        } else {
            let mut reader = StoredFieldsReader::open(&*self.dir, &meta.name, "", &meta.id)
                .map_err(codec_err)?;
            read_all(&mut reader)?;
        }
        Ok(out)
    }

    pub fn open_input(&self, segment: &str, file_name: &str) -> Result<Arc<dyn ByteRange>> {
        let meta = self.segment(segment).ok_or_else(|| {
            Error::invalid(format!("no segment {segment:?} in commit"))
        })?;
        let backing = if meta.is_compound && !stored_outside_cfs(file_name) {
            let cfs = CompoundDirectory::open(&*self.dir, &meta.name, &meta.id)
                .map_err(codec_err)?;
            cfs.open_file(file_name).map_err(codec_err)?
        } else {
            self.dir.open_file(file_name).map_err(codec_err)?
        };
        Ok(Arc::new(BackingRange { backing }))
    }
}

/// [`ByteRange`] over a Bearing [`FileBacking`] (mmap, owned bytes, or a
/// zero-copy `.cfs` sub-slice).
struct BackingRange {
    backing: FileBacking,
}

impl ByteRange for BackingRange {
    fn len(&self) -> u64 {
        self.backing.as_bytes().len() as u64
    }

    fn read_into(&self, offset: u64, len: u64, dst: BufferTarget<'_>) -> Result<()> {
        let end = offset
            .checked_add(len)
            .ok_or_else(|| Error::invalid("offset + len overflows u64"))?;
        if end > self.len() {
            return Err(Error::corrupt(format!(
                "read [{offset}, {end}) beyond file of {} bytes",
                self.len()
            )));
        }
        if dst.capacity() < len {
            return Err(Error::invalid(format!(
                "destination capacity {} < requested {len}",
                dst.capacity()
            )));
        }
        let src = &self.backing.as_bytes()[offset as usize..end as usize];
        match dst {
            BufferTarget::Host(out) => out[..len as usize].copy_from_slice(src),
            BufferTarget::Device { .. } => {
                return Err(Error::unsupported(
                    "codec BackingRange cannot fill device memory; use the gpu source",
                ));
            }
        }
        Ok(())
    }

    fn slice(&self, offset: u64, len: u64) -> Option<&[u8]> {
        let end = offset.checked_add(len)?;
        let bytes = self.backing.as_bytes();
        if end > bytes.len() as u64 {
            return None;
        }
        Some(&bytes[offset as usize..end as usize])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinned_codec_gate() {
        assert!(ensure_pinned_codec("_0", "Lucene103").is_ok());
        let err = ensure_pinned_codec("_0", "Lucene99").unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)), "got {err:?}");
    }

    #[test]
    fn compound_outer_files() {
        assert!(stored_outside_cfs("_0.si"));
        assert!(stored_outside_cfs("_0.cfs"));
        assert!(stored_outside_cfs("_0.cfe"));
        assert!(stored_outside_cfs("_0_1.liv"));
        assert!(!stored_outside_cfs("_0_Lucene90_0.dvm"));
        assert!(!stored_outside_cfs("_0.fnm"));
    }
}
