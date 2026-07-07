// SPDX-License-Identifier: Apache-2.0

//! Norms reader for the Lucene90 norms format.
//!
//! Reads `.nvm` (metadata) and `.nvd` (data) files written by the norms writer.
//! Metadata is read eagerly during construction; norm values are read lazily from
//! the `.nvd` data file on demand.

use std::collections::HashMap;
use std::fmt;
use std::io;

use log::debug;

use crate::codecs::codec_file_handle::{CodecFileHandle, IndexFile};
use crate::codecs::codec_headers;
use crate::codecs::lucene90::indexed_disi::IndexedDISI;
use crate::index::doc_values_iterators::{DocValuesIterator, NumericDocValues};
use crate::index::pipeline::segment_accumulator::PerFieldNormsData;
use crate::index::{FieldInfo, FieldInfos};
use crate::search::DocIdSetIterator;
use crate::search::doc_id_set_iterator::NO_MORE_DOCS;
use crate::store::{Directory, FileBacking, IndexInput};

/// Per-field norms metadata entry read from `.nvm`.
#[derive(Clone)]
struct NormsEntry {
    /// Offset of the IndexedDISI bitset in `.nvd`, or -1 (ALL) / -2 (EMPTY).
    docs_with_field_offset: i64,
    /// Byte length of the IndexedDISI bitset in `.nvd`.
    docs_with_field_length: i64,
    /// Number of jump table entries for the IndexedDISI.
    jump_table_entry_count: i16,
    /// Dense rank power for the IndexedDISI.
    dense_rank_power: u8,
    /// Number of documents that have a norm value for this field.
    num_docs_with_field: i32,
    /// Bytes per norm value: 0 (constant), 1, 2, 4, or 8.
    bytes_per_norm: u8,
    /// If `bytes_per_norm == 0`: the constant norm value.
    /// Otherwise: byte offset into `.nvd` where norm values start.
    norms_offset: i64,
}

/// Produces per-field normalization values.
///
/// Both file-backed readers and in-memory buffered producers implement this trait.
pub trait NormsProducer: fmt::Debug {
    /// Returns a [`NumericDocValues`] iterator for the given field.
    ///
    /// Each call returns a **fresh** iterator positioned before the first document.
    /// Returns `None` if the field has no norms.
    fn get_norms(
        &self,
        field_info: &FieldInfo,
    ) -> io::Result<Option<Box<dyn NumericDocValues + '_>>>;
}

/// Reads norms for a segment from `.nvm` / `.nvd` files.
///
/// Opens files during construction. Metadata entries are read eagerly and stored
/// as an immutable slice; the `.nvd` data file handle is kept open for lazy value
/// reads via [`get_norms`](NormsProducer::get_norms).
pub struct NormsReader {
    /// Per-field metadata indexed by field number. `None` for fields without norms.
    entries: Box<[Option<NormsEntry>]>,
    /// Maximum document ID + 1 for this segment.
    max_doc: i32,
    /// Owned bytes of the `.nvd` data file.
    data: FileBacking,
}

impl fmt::Debug for NormsReader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NormsReader")
            .field("entries", &self.entries.len())
            .field("max_doc", &self.max_doc)
            .finish()
    }
}

impl NormsReader {
    /// Opens norms files (`.nvm`, `.nvd`) for the given segment.
    ///
    /// Reads all metadata eagerly from `.nvm`; the `.nvd` data file is opened but
    /// no data is read until [`get_norms`](NormsProducer::get_norms) is called.
    pub fn open(
        directory: &dyn Directory,
        segment_name: &str,
        segment_suffix: &str,
        segment_id: &[u8; codec_headers::ID_LENGTH],
        field_infos: &FieldInfos,
        max_doc: i32,
    ) -> io::Result<Self> {
        let nvm = CodecFileHandle::open(
            directory,
            IndexFile::NormsMeta,
            segment_name,
            segment_id,
            segment_suffix,
        )?;
        let entries = read_fields(&mut nvm.body(), field_infos)?;

        let nvd = CodecFileHandle::open(
            directory,
            IndexFile::NormsData,
            segment_name,
            segment_id,
            segment_suffix,
        )?;
        if nvm.version() != nvd.version() {
            return Err(io::Error::other(format!(
                "format version mismatch: meta={}, data={}",
                nvm.version(),
                nvd.version()
            )));
        }

        debug!(
            "norms_reader: opened {} entries for segment {segment_name}",
            entries.len()
        );

        Ok(Self {
            entries,
            max_doc,
            data: nvd.into_backing(),
        })
    }

    /// Returns a [`NumericDocValues`] for the given field, or `None` if absent.
    ///
    /// The returned iterator borrows from `&self.data`; its lifetime is tied to
    /// this reader. The [`NormsProducer`] trait impl delegates here.
    pub fn get_norms(
        &self,
        field_info: &FieldInfo,
    ) -> io::Result<Option<Box<dyn NumericDocValues + '_>>> {
        let entry = match self.entry(field_info.number()) {
            Some(e) => e,
            None => return Ok(None),
        };

        let data_input = IndexInput::unnamed(self.data.as_bytes());

        if entry.docs_with_field_offset == -2 {
            // EMPTY: no documents have norms for this field
            return Ok(None);
        } else if entry.docs_with_field_offset == -1 {
            // DENSE: every document has a norm
            if entry.bytes_per_norm == 0 {
                return Ok(Some(Box::new(DenseNormsIterator {
                    doc: -1,
                    max_doc: self.max_doc,
                    slice: None,
                    bytes_per_norm: 0,
                    constant_value: entry.norms_offset,
                })));
            }
            let data_length = entry.num_docs_with_field as usize * entry.bytes_per_norm as usize;
            let slice = data_input.view("norms data", entry.norms_offset as usize, data_length)?;
            return Ok(Some(Box::new(DenseNormsIterator {
                doc: -1,
                max_doc: self.max_doc,
                slice: Some(slice),
                bytes_per_norm: entry.bytes_per_norm,
                constant_value: 0,
            })));
        }

        // SPARSE: use IndexedDISI to check presence and get ordinal
        let disi = IndexedDISI::new(
            &data_input,
            entry.docs_with_field_offset as usize,
            entry.docs_with_field_length as usize,
            entry.jump_table_entry_count,
            entry.dense_rank_power,
            entry.num_docs_with_field as i64,
        )?;

        if entry.bytes_per_norm == 0 {
            return Ok(Some(Box::new(SparseNormsIterator {
                disi,
                slice: None,
                bytes_per_norm: 0,
                constant_value: entry.norms_offset,
            })));
        }
        let data_length = entry.num_docs_with_field as usize * entry.bytes_per_norm as usize;
        let slice = data_input.view("norms data", entry.norms_offset as usize, data_length)?;
        Ok(Some(Box::new(SparseNormsIterator {
            disi,
            slice: Some(slice),
            bytes_per_norm: entry.bytes_per_norm,
            constant_value: 0,
        })))
    }

    /// Returns the number of documents that have norms for the given field.
    ///
    /// Returns `None` if no norms entry exists for this field number.
    pub fn num_docs_with_field(&self, field_number: u32) -> Option<i32> {
        self.entry(field_number).map(|e| e.num_docs_with_field)
    }

    /// Returns the norms entry for a field number, or `None` if absent.
    fn entry(&self, field_number: u32) -> Option<&NormsEntry> {
        self.entries
            .get(field_number as usize)
            .and_then(|opt| opt.as_ref())
    }
}

impl NormsProducer for NormsReader {
    fn get_norms(
        &self,
        field_info: &FieldInfo,
    ) -> io::Result<Option<Box<dyn NumericDocValues + '_>>> {
        self.get_norms(field_info)
    }
}

// ---------------------------------------------------------------------------
// BufferedNormsProducer — in-memory norms from the indexing pipeline
// ---------------------------------------------------------------------------

/// Per-field norms data borrowed from the accumulator.
#[derive(Debug)]
struct BufferedFieldNorms<'a> {
    docs: &'a [i32],
    values: &'a [i64],
}

/// In-memory [`NormsProducer`] borrowing from indexing pipeline buffers.
///
/// Each call to [`get_norms`](NormsProducer::get_norms) returns a fresh iterator
/// over the borrowed data, allowing the writer to make multiple passes.
#[derive(Debug)]
pub struct BufferedNormsProducer<'a> {
    /// Per-field norms data indexed by field number.
    fields: Vec<Option<BufferedFieldNorms<'a>>>,
}

impl<'a> BufferedNormsProducer<'a> {
    /// Creates a new buffered producer borrowing from accumulated norms data.
    pub fn new(norms: &'a HashMap<u32, PerFieldNormsData>) -> Self {
        let max_field = norms.keys().max().map_or(0, |&k| k as usize + 1);
        let mut fields = Vec::with_capacity(max_field);
        fields.resize_with(max_field, || None);

        for (&field_number, data) in norms {
            fields[field_number as usize] = Some(BufferedFieldNorms {
                docs: &data.docs,
                values: &data.values,
            });
        }

        Self { fields }
    }

    /// Returns the field numbers that have norms, sorted.
    pub fn field_numbers(&self) -> Vec<u32> {
        self.fields
            .iter()
            .enumerate()
            .filter_map(|(i, f)| f.as_ref().map(|_| i as u32))
            .collect()
    }
}

impl NormsProducer for BufferedNormsProducer<'_> {
    fn get_norms(
        &self,
        field_info: &FieldInfo,
    ) -> io::Result<Option<Box<dyn NumericDocValues + '_>>> {
        let field_data = match self.fields.get(field_info.number() as usize) {
            Some(Some(data)) => data,
            _ => return Ok(None),
        };

        if field_data.docs.is_empty() {
            return Ok(None);
        }

        Ok(Some(Box::new(BufferedNorms {
            docs: field_data.docs,
            values: field_data.values,
            pos: -1,
        })))
    }
}

/// Forward iterator over in-memory buffered norms.
#[derive(Debug)]
struct BufferedNorms<'a> {
    docs: &'a [i32],
    values: &'a [i64],
    /// Current position in the docs/values arrays, or -1 if not started.
    pos: i32,
}

impl DocIdSetIterator for BufferedNorms<'_> {
    fn doc_id(&self) -> i32 {
        if self.pos < 0 {
            -1
        } else if (self.pos as usize) < self.docs.len() {
            self.docs[self.pos as usize]
        } else {
            NO_MORE_DOCS
        }
    }

    fn next_doc(&mut self) -> io::Result<i32> {
        self.pos += 1;
        if (self.pos as usize) < self.docs.len() {
            Ok(self.docs[self.pos as usize])
        } else {
            self.pos = self.docs.len() as i32;
            Ok(NO_MORE_DOCS)
        }
    }

    fn advance(&mut self, target: i32) -> io::Result<i32> {
        loop {
            let doc = self.next_doc()?;
            if doc >= target || doc == NO_MORE_DOCS {
                return Ok(doc);
            }
        }
    }

    fn cost(&self) -> i64 {
        self.docs.len() as i64
    }
}

impl DocValuesIterator for BufferedNorms<'_> {
    fn advance_exact(&mut self, target: i32) -> io::Result<bool> {
        // Binary search for the target doc
        match self.docs.binary_search(&target) {
            Ok(idx) => {
                self.pos = idx as i32;
                Ok(true)
            }
            Err(_) => Ok(false),
        }
    }
}

impl NumericDocValues for BufferedNorms<'_> {
    fn long_value(&self) -> io::Result<i64> {
        Ok(self.values[self.pos as usize])
    }
}

// ---------------------------------------------------------------------------
// DenseNormsIterator — all documents have a norm value
// ---------------------------------------------------------------------------

/// Dense norms iterator where every document has a norm value.
///
/// For `bytes_per_norm > 0`, reads values via random access into the `.nvd`
/// data slice. For `bytes_per_norm == 0`, returns a constant value.
struct DenseNormsIterator<'a> {
    doc: i32,
    max_doc: i32,
    slice: Option<IndexInput<'a>>,
    bytes_per_norm: u8,
    constant_value: i64,
}

impl fmt::Debug for DenseNormsIterator<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DenseNormsIterator")
            .field("doc", &self.doc)
            .field("max_doc", &self.max_doc)
            .field("bytes_per_norm", &self.bytes_per_norm)
            .finish()
    }
}

impl DocIdSetIterator for DenseNormsIterator<'_> {
    fn doc_id(&self) -> i32 {
        self.doc
    }

    fn next_doc(&mut self) -> io::Result<i32> {
        self.advance(self.doc + 1)
    }

    fn advance(&mut self, target: i32) -> io::Result<i32> {
        if target >= self.max_doc {
            self.doc = NO_MORE_DOCS;
        } else {
            self.doc = target;
        }
        Ok(self.doc)
    }

    fn cost(&self) -> i64 {
        self.max_doc as i64
    }
}

impl DocValuesIterator for DenseNormsIterator<'_> {
    fn advance_exact(&mut self, target: i32) -> io::Result<bool> {
        self.doc = target;
        Ok(true)
    }
}

impl NumericDocValues for DenseNormsIterator<'_> {
    fn long_value(&self) -> io::Result<i64> {
        if self.bytes_per_norm == 0 {
            return Ok(self.constant_value);
        }
        let slice = self.slice.as_ref().unwrap();
        match self.bytes_per_norm {
            1 => Ok(slice.read_byte_at(self.doc as usize)? as i8 as i64),
            2 => Ok(slice.read_le_short_at((self.doc as usize) << 1)? as i64),
            4 => Ok(slice.read_le_int_at((self.doc as usize) << 2)? as i64),
            8 => slice.read_le_long_at((self.doc as usize) << 3),
            _ => Err(io::Error::other(format!(
                "invalid bytes_per_norm: {}",
                self.bytes_per_norm
            ))),
        }
    }
}

// ---------------------------------------------------------------------------
// SparseNormsIterator — only some documents have a norm value
// ---------------------------------------------------------------------------

/// Sparse norms iterator backed by an [`IndexedDISI`].
///
/// `advance_exact` delegates to the DISI to check document presence.
/// `long_value` reads from the data slice at the DISI's ordinal index.
struct SparseNormsIterator<'a> {
    disi: IndexedDISI<'a>,
    slice: Option<IndexInput<'a>>,
    bytes_per_norm: u8,
    constant_value: i64,
}

impl fmt::Debug for SparseNormsIterator<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SparseNormsIterator")
            .field("bytes_per_norm", &self.bytes_per_norm)
            .finish()
    }
}

impl DocIdSetIterator for SparseNormsIterator<'_> {
    fn doc_id(&self) -> i32 {
        self.disi.doc_id()
    }

    fn next_doc(&mut self) -> io::Result<i32> {
        match self.disi.next_doc()? {
            Some(doc) => Ok(doc),
            None => Ok(NO_MORE_DOCS),
        }
    }

    fn advance(&mut self, target: i32) -> io::Result<i32> {
        match self.disi.advance(target)? {
            Some(doc) => Ok(doc),
            None => Ok(NO_MORE_DOCS),
        }
    }

    fn cost(&self) -> i64 {
        self.disi.cost()
    }
}

impl DocValuesIterator for SparseNormsIterator<'_> {
    fn advance_exact(&mut self, target: i32) -> io::Result<bool> {
        self.disi.advance_exact(target)
    }
}

impl NumericDocValues for SparseNormsIterator<'_> {
    fn long_value(&self) -> io::Result<i64> {
        if self.bytes_per_norm == 0 {
            return Ok(self.constant_value);
        }
        let index = self.disi.index() as usize;
        let slice = self.slice.as_ref().unwrap();
        match self.bytes_per_norm {
            1 => Ok(slice.read_byte_at(index)? as i8 as i64),
            2 => Ok(slice.read_le_short_at(index << 1)? as i64),
            4 => Ok(slice.read_le_int_at(index << 2)? as i64),
            8 => slice.read_le_long_at(index << 3),
            _ => Err(io::Error::other(format!(
                "invalid bytes_per_norm: {}",
                self.bytes_per_norm
            ))),
        }
    }
}

/// Reads all norms metadata entries from the `.nvm` file.
///
/// Returns a `Box<[Option<NormsEntry>]>` indexed by field number.
fn read_fields(
    meta: &mut IndexInput<'_>,
    field_infos: &FieldInfos,
) -> io::Result<Box<[Option<NormsEntry>]>> {
    let mut entries: Vec<Option<NormsEntry>> = vec![None; field_infos.len()];

    loop {
        let field_number = meta.read_le_int()?;
        if field_number == -1 {
            break; // EOF marker
        }

        let field_number = field_number as u32;
        let info = field_infos
            .field_info_by_number(field_number)
            .ok_or_else(|| io::Error::other(format!("invalid field number: {field_number}")))?;

        if !info.has_norms() {
            return Err(io::Error::other(format!(
                "invalid field: {} (no norms)",
                info.name()
            )));
        }

        let docs_with_field_offset = meta.read_le_long()?;
        let docs_with_field_length = meta.read_le_long()?;
        let jump_table_entry_count = meta.read_le_short()?;
        let dense_rank_power = meta.read_byte()?;
        let num_docs_with_field = meta.read_le_int()?;
        let bytes_per_norm = meta.read_byte()?;

        match bytes_per_norm {
            0 | 1 | 2 | 4 | 8 => {}
            _ => {
                return Err(io::Error::other(format!(
                    "invalid bytesPerNorm: {bytes_per_norm}, field: {}",
                    info.name()
                )));
            }
        }

        let norms_offset = meta.read_le_long()?;

        entries[field_number as usize] = Some(NormsEntry {
            docs_with_field_offset,
            docs_with_field_length,
            jump_table_entry_count,
            dense_rank_power,
            num_docs_with_field,
            bytes_per_norm,
            norms_offset,
        });
    }

    Ok(entries.into_boxed_slice())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codecs::lucene90::norms;
    use crate::document::{DocValuesType, IndexOptions};
    use crate::index::field_infos::PointDimensionConfig;
    use crate::index::{FieldInfo, FieldInfos};
    use crate::store::{MemoryDirectory, SharedDirectory};
    use assertables::*;

    fn make_field_info(name: &str, number: u32, has_norms: bool) -> FieldInfo {
        FieldInfo::new(
            name.to_string(),
            number,
            false,
            !has_norms,
            IndexOptions::DocsAndFreqsAndPositions,
            DocValuesType::None,
            PointDimensionConfig::default(),
        )
    }

    fn test_directory() -> SharedDirectory {
        MemoryDirectory::create()
    }

    /// Writes norms using the production path and opens a reader.
    fn write_and_read(
        fields: &[(&str, u32, &[i64], &[i32])],
        field_infos: &FieldInfos,
        num_docs: i32,
    ) -> NormsReader {
        let segment_id = [0u8; 16];
        let dir = test_directory();

        let mut norms_map = HashMap::new();
        let mut infos = Vec::new();
        for &(name, number, values, docs) in fields {
            norms_map.insert(
                number,
                PerFieldNormsData {
                    field_name: name.to_string(),
                    docs: docs.to_vec(),
                    values: values.to_vec(),
                },
            );
            infos.push(make_field_info(name, number, true));
        }
        infos.sort_by_key(|f| f.number());
        let producer = BufferedNormsProducer::new(&norms_map);
        let info_refs: Vec<&FieldInfo> = infos.iter().collect();

        norms::write(&dir, "_0", "", &segment_id, &info_refs, &producer, num_docs).unwrap();

        NormsReader::open(&dir, "_0", "", &segment_id, field_infos, num_docs).unwrap()
    }

    /// Helper: get a single norm value using get_norms + advance_exact + long_value.
    fn get_norm(reader: &NormsReader, fi: &FieldInfo, doc_id: i32) -> io::Result<Option<i64>> {
        let mut norms = match reader.get_norms(fi)? {
            Some(n) => n,
            None => return Ok(None),
        };
        if norms.advance_exact(doc_id)? {
            Ok(Some(norms.long_value()?))
        } else {
            Ok(None)
        }
    }

    // Ported from org.apache.lucene.codecs.lucene90.TestLucene90NormsFormat

    #[test]
    fn test_all_constant() {
        let field_infos = FieldInfos::new(vec![make_field_info("contents", 0, true)]);
        let fi = field_infos.field_info_by_number(0).unwrap();
        let reader = write_and_read(
            &[("contents", 0, &[42, 42, 42], &[0, 1, 2])],
            &field_infos,
            3,
        );

        assert_eq!(reader.num_docs_with_field(0), Some(3));
        for doc in 0..3 {
            assert_eq!(get_norm(&reader, fi, doc).unwrap(), Some(42), "doc {doc}");
        }
    }

    #[test]
    fn test_all_variable_1byte() {
        let field_infos = FieldInfos::new(vec![make_field_info("contents", 0, true)]);
        let fi = field_infos.field_info_by_number(0).unwrap();
        let reader = write_and_read(
            &[("contents", 0, &[12, 8, 10], &[0, 1, 2])],
            &field_infos,
            3,
        );

        assert_eq!(reader.num_docs_with_field(0), Some(3));
        assert_eq!(get_norm(&reader, fi, 0).unwrap(), Some(12));
        assert_eq!(get_norm(&reader, fi, 1).unwrap(), Some(8));
        assert_eq!(get_norm(&reader, fi, 2).unwrap(), Some(10));
    }

    #[test]
    fn test_all_variable_2byte() {
        let field_infos = FieldInfos::new(vec![make_field_info("f", 0, true)]);
        let fi = field_infos.field_info_by_number(0).unwrap();
        let reader = write_and_read(
            &[("f", 0, &[1000, -500, 32000], &[0, 1, 2])],
            &field_infos,
            3,
        );

        assert_eq!(get_norm(&reader, fi, 0).unwrap(), Some(1000));
        assert_eq!(get_norm(&reader, fi, 1).unwrap(), Some(-500));
        assert_eq!(get_norm(&reader, fi, 2).unwrap(), Some(32000));
    }

    #[test]
    fn test_all_variable_4byte() {
        let field_infos = FieldInfos::new(vec![make_field_info("f", 0, true)]);
        let fi = field_infos.field_info_by_number(0).unwrap();
        let reader = write_and_read(&[("f", 0, &[100_000, -100_000], &[0, 1])], &field_infos, 2);

        assert_eq!(get_norm(&reader, fi, 0).unwrap(), Some(100_000));
        assert_eq!(get_norm(&reader, fi, 1).unwrap(), Some(-100_000));
    }

    #[test]
    fn test_all_variable_8byte() {
        let field_infos = FieldInfos::new(vec![make_field_info("f", 0, true)]);
        let fi = field_infos.field_info_by_number(0).unwrap();
        let reader = write_and_read(&[("f", 0, &[i64::MAX, i64::MIN], &[0, 1])], &field_infos, 2);

        assert_eq!(get_norm(&reader, fi, 0).unwrap(), Some(i64::MAX));
        assert_eq!(get_norm(&reader, fi, 1).unwrap(), Some(i64::MIN));
    }

    #[test]
    fn test_empty_pattern() {
        let field_infos = FieldInfos::new(vec![make_field_info("contents", 0, true)]);
        let fi = field_infos.field_info_by_number(0).unwrap();
        let reader = write_and_read(&[("contents", 0, &[], &[])], &field_infos, 3);

        assert_eq!(reader.num_docs_with_field(0), Some(0));
        assert_none!(get_norm(&reader, fi, 0).unwrap());
        assert_none!(get_norm(&reader, fi, 1).unwrap());
    }

    #[test]
    fn test_sparse_variable() {
        let field_infos = FieldInfos::new(vec![make_field_info("contents", 0, true)]);
        let fi = field_infos.field_info_by_number(0).unwrap();
        let reader = write_and_read(&[("contents", 0, &[12, 8], &[1, 3])], &field_infos, 5);

        assert_eq!(reader.num_docs_with_field(0), Some(2));
        assert_none!(get_norm(&reader, fi, 0).unwrap());
        assert_eq!(get_norm(&reader, fi, 1).unwrap(), Some(12));
        assert_none!(get_norm(&reader, fi, 2).unwrap());
        assert_eq!(get_norm(&reader, fi, 3).unwrap(), Some(8));
        assert_none!(get_norm(&reader, fi, 4).unwrap());
    }

    #[test]
    fn test_sparse_constant() {
        let field_infos = FieldInfos::new(vec![make_field_info("title", 0, true)]);
        let fi = field_infos.field_info_by_number(0).unwrap();
        let reader = write_and_read(&[("title", 0, &[42, 42, 42], &[0, 2, 4])], &field_infos, 5);

        assert_eq!(reader.num_docs_with_field(0), Some(3));
        assert_eq!(get_norm(&reader, fi, 0).unwrap(), Some(42));
        assert_none!(get_norm(&reader, fi, 1).unwrap());
        assert_eq!(get_norm(&reader, fi, 2).unwrap(), Some(42));
        assert_none!(get_norm(&reader, fi, 3).unwrap());
        assert_eq!(get_norm(&reader, fi, 4).unwrap(), Some(42));
    }

    #[test]
    fn test_multiple_fields_mixed_patterns() {
        let field_infos = FieldInfos::new(vec![
            make_field_info("alpha", 0, true),
            make_field_info("beta", 1, true),
        ]);
        let fi = field_infos.field_info_by_number(0).unwrap();
        let fi1 = field_infos.field_info_by_number(1).unwrap();
        let reader = write_and_read(
            &[
                ("alpha", 0, &[5, 5, 5], &[0, 1, 2]),
                ("beta", 1, &[10, 20], &[0, 2]),
            ],
            &field_infos,
            3,
        );

        // alpha: ALL constant
        assert_eq!(reader.num_docs_with_field(0), Some(3));
        assert_eq!(get_norm(&reader, fi, 0).unwrap(), Some(5));
        assert_eq!(get_norm(&reader, fi, 1).unwrap(), Some(5));
        assert_eq!(get_norm(&reader, fi, 2).unwrap(), Some(5));

        // beta: SPARSE variable
        assert_eq!(reader.num_docs_with_field(1), Some(2));
        assert_eq!(get_norm(&reader, fi1, 0).unwrap(), Some(10));
        assert_none!(get_norm(&reader, fi1, 1).unwrap());
        assert_eq!(get_norm(&reader, fi1, 2).unwrap(), Some(20));
    }

    #[test]
    fn test_nonexistent_field() {
        let field_infos = FieldInfos::new(vec![make_field_info("contents", 0, true)]);
        let reader = write_and_read(&[("contents", 0, &[10], &[0])], &field_infos, 1);

        assert_none!(reader.num_docs_with_field(99));
    }

    #[test]
    fn test_negative_norm_values() {
        let field_infos = FieldInfos::new(vec![make_field_info("f", 0, true)]);
        let fi = field_infos.field_info_by_number(0).unwrap();
        let reader = write_and_read(
            &[("f", 0, &[-128, -1, 0, 127], &[0, 1, 2, 3])],
            &field_infos,
            4,
        );

        assert_eq!(get_norm(&reader, fi, 0).unwrap(), Some(-128));
        assert_eq!(get_norm(&reader, fi, 1).unwrap(), Some(-1));
        assert_eq!(get_norm(&reader, fi, 2).unwrap(), Some(0));
        assert_eq!(get_norm(&reader, fi, 3).unwrap(), Some(127));
    }
}
