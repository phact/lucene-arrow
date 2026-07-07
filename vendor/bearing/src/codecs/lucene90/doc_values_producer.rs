// SPDX-License-Identifier: Apache-2.0

//! Doc values producer trait and metadata reader for the Lucene90 doc values format.
//!
//! The [`DocValuesProducer`] trait abstracts access to per-field doc values,
//! allowing both file-backed readers and in-memory buffered producers.
//!
//! [`DocValuesReader`] reads `.dvm` (metadata) and `.dvd` (data) files written
//! by [`super::doc_values`]. Metadata is read eagerly during construction; value
//! data is read lazily from the `.dvd` data file on demand.

use std::fmt;
use std::io;

use log::debug;

use crate::codecs::codec_file_handle::{CodecFileHandle, IndexFile};
use crate::codecs::codec_headers;
use crate::codecs::lucene90::doc_values::{
    BINARY, DIRECT_MONOTONIC_BLOCK_SHIFT, NUMERIC, SORTED, SORTED_NUMERIC, SORTED_SET,
};
use crate::index::doc_values_iterators::{
    BinaryDocValues, NumericDocValues, SortedDocValues, SortedNumericDocValues, SortedSetDocValues,
};
use crate::index::{FieldInfo, FieldInfos};
use crate::store::{Directory, FileBacking, IndexInput};

// ---------------------------------------------------------------------------
// Entry types — one per doc values type, stored eagerly in memory
// ---------------------------------------------------------------------------

/// Per-field doc values metadata.
///
/// Stores only the document count for now. Additional fields (offsets, encoding
/// params) will be added when lazy value reads are implemented.
#[derive(Clone, Copy)]
struct DocValuesEntry {
    num_docs_with_field: i32,
}

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// Produces per-field doc values.
///
/// Both file-backed readers and in-memory buffered producers implement this
/// trait, allowing codec writers to accept data from either source.
pub trait DocValuesProducer: fmt::Debug {
    /// Returns a [`NumericDocValues`] iterator for the given field.
    fn get_numeric(
        &self,
        field_info: &FieldInfo,
    ) -> io::Result<Option<Box<dyn NumericDocValues + '_>>>;

    /// Returns a [`BinaryDocValues`] iterator for the given field.
    fn get_binary(
        &self,
        field_info: &FieldInfo,
    ) -> io::Result<Option<Box<dyn BinaryDocValues + '_>>>;

    /// Returns a [`SortedDocValues`] iterator for the given field.
    fn get_sorted(
        &self,
        field_info: &FieldInfo,
    ) -> io::Result<Option<Box<dyn SortedDocValues + '_>>>;

    /// Returns a [`SortedNumericDocValues`] iterator for the given field.
    fn get_sorted_numeric(
        &self,
        field_info: &FieldInfo,
    ) -> io::Result<Option<Box<dyn SortedNumericDocValues + '_>>>;

    /// Returns a [`SortedSetDocValues`] iterator for the given field.
    fn get_sorted_set(
        &self,
        field_info: &FieldInfo,
    ) -> io::Result<Option<Box<dyn SortedSetDocValues + '_>>>;
}

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------

/// Reads doc values metadata for a segment.
///
/// Opens `.dvm` and `.dvd` files during construction. All metadata is read
/// eagerly from `.dvm`; the `.dvd` data file handle is kept open for future
/// lazy value reads.
pub struct DocValuesReader {
    /// Per-field metadata indexed by field number. `None` for fields without doc values.
    entries: Box<[Option<DocValuesEntry>]>,
    /// Owned bytes of the `.dvd` data file for lazy value reads.
    #[expect(dead_code)]
    data: FileBacking,
}

impl fmt::Debug for DocValuesReader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DocValuesReader")
            .field(
                "entries",
                &self.entries.iter().filter(|e| e.is_some()).count(),
            )
            .finish()
    }
}

impl DocValuesReader {
    /// Opens doc values files (`.dvm`, `.dvd`) for the given segment.
    pub fn open(
        directory: &dyn Directory,
        segment_name: &str,
        segment_suffix: &str,
        segment_id: &[u8; codec_headers::ID_LENGTH],
        field_infos: &FieldInfos,
    ) -> io::Result<Self> {
        let dvm = CodecFileHandle::open(
            directory,
            IndexFile::DocValuesMeta,
            segment_name,
            segment_id,
            segment_suffix,
        )?;
        let entries = read_fields(&mut dvm.body(), field_infos)?;

        let dvd = CodecFileHandle::open(
            directory,
            IndexFile::DocValuesData,
            segment_name,
            segment_id,
            segment_suffix,
        )?;
        if dvm.version() != dvd.version() {
            return Err(io::Error::other(format!(
                "format version mismatch: meta={}, data={}",
                dvm.version(),
                dvd.version()
            )));
        }

        debug!(
            "doc_values_reader: opened {} entries for segment {segment_name}",
            entries.iter().filter(|e| e.is_some()).count()
        );

        Ok(Self {
            entries,
            data: dvd.into_backing(),
        })
    }

    /// Returns the number of documents that have values for the given field.
    pub fn num_docs_with_field(&self, field_number: u32) -> Option<i32> {
        self.entries
            .get(field_number as usize)
            .and_then(|opt| opt.as_ref())
            .map(|e| e.num_docs_with_field)
    }
}

impl DocValuesProducer for DocValuesReader {
    fn get_numeric(
        &self,
        _field_info: &FieldInfo,
    ) -> io::Result<Option<Box<dyn NumericDocValues + '_>>> {
        todo!("disk-backed doc values reading for merge path")
    }

    fn get_binary(
        &self,
        _field_info: &FieldInfo,
    ) -> io::Result<Option<Box<dyn BinaryDocValues + '_>>> {
        todo!("disk-backed doc values reading for merge path")
    }

    fn get_sorted(
        &self,
        _field_info: &FieldInfo,
    ) -> io::Result<Option<Box<dyn SortedDocValues + '_>>> {
        todo!("disk-backed doc values reading for merge path")
    }

    fn get_sorted_numeric(
        &self,
        _field_info: &FieldInfo,
    ) -> io::Result<Option<Box<dyn SortedNumericDocValues + '_>>> {
        todo!("disk-backed doc values reading for merge path")
    }

    fn get_sorted_set(
        &self,
        _field_info: &FieldInfo,
    ) -> io::Result<Option<Box<dyn SortedSetDocValues + '_>>> {
        todo!("disk-backed doc values reading for merge path")
    }
}

// ---------------------------------------------------------------------------
// Metadata parsing
// ---------------------------------------------------------------------------

/// Reads all doc values metadata entries from the `.dvm` file.
fn read_fields(
    meta: &mut IndexInput<'_>,
    field_infos: &FieldInfos,
) -> io::Result<Box<[Option<DocValuesEntry>]>> {
    let mut entries: Vec<Option<DocValuesEntry>> = vec![None; field_infos.len()];

    loop {
        let field_number = meta.read_le_int()?;
        if field_number == -1 {
            break;
        }

        let field_number = field_number as u32;
        let info = field_infos
            .field_info_by_number(field_number)
            .ok_or_else(|| io::Error::other(format!("invalid field number: {field_number}")))?;

        let type_byte = meta.read_byte()?;

        // Skip-index metadata must be read BEFORE type-specific metadata
        if info.doc_values_skip_index_type != 0 {
            read_doc_values_skipper_meta(meta)?;
        }

        let entry = match type_byte {
            NUMERIC => read_numeric(meta)?,
            BINARY => read_binary(meta)?,
            SORTED => read_sorted(meta)?,
            SORTED_SET => read_sorted_set(meta)?,
            SORTED_NUMERIC => read_sorted_numeric(meta)?,
            _ => {
                return Err(io::Error::other(format!(
                    "invalid doc values type: {type_byte}"
                )));
            }
        };

        entries[field_number as usize] = Some(entry);
    }

    Ok(entries.into_boxed_slice())
}

/// Reads doc values skipper metadata to keep the stream in sync.
///
/// Java stores these values in a `DocValuesSkipperEntry` for use by
/// `getDocValuesSkipper()`. We read and discard them for now since skip
/// index queries are not yet implemented.
fn read_doc_values_skipper_meta(meta: &mut IndexInput<'_>) -> io::Result<()> {
    meta.read_le_long()?; // offset
    meta.read_le_long()?; // length
    meta.read_le_long()?; // maxValue
    meta.read_le_long()?; // minValue
    meta.read_le_int()?; // docCount
    meta.read_le_int()?; // maxDocId
    Ok(())
}

/// Skips the docs-with-field block written by `write_values()`.
fn skip_docs_with_field(meta: &mut IndexInput<'_>) -> io::Result<()> {
    meta.read_le_long()?; // offset
    meta.read_le_long()?; // length
    meta.read_le_short()?; // jump_table_entry_count
    meta.read_byte()?; // dense_rank_power
    Ok(())
}

/// Reads the `write_values()` block, returning `num_values`.
/// Skips all encoding metadata (table, bpv, min, gcd, offsets).
fn read_values_num_values(meta: &mut IndexInput<'_>) -> io::Result<i64> {
    skip_docs_with_field(meta)?;

    let num_values = meta.read_le_long()?;

    let table_size = meta.read_le_int()?;
    if table_size > 0 {
        meta.skip_bytes(table_size as usize * 8)?; // table entries
    }

    meta.read_byte()?; // bits_per_value
    meta.read_le_long()?; // min_value
    meta.read_le_long()?; // gcd
    meta.read_le_long()?; // values_offset
    meta.read_le_long()?; // values_length
    meta.read_le_long()?; // jump_table_offset

    Ok(num_values)
}

fn read_numeric(meta: &mut IndexInput<'_>) -> io::Result<DocValuesEntry> {
    let num_values = read_values_num_values(meta)?;
    Ok(DocValuesEntry {
        num_docs_with_field: num_values as i32,
    })
}

fn read_binary(meta: &mut IndexInput<'_>) -> io::Result<DocValuesEntry> {
    meta.read_le_long()?; // data_offset
    meta.read_le_long()?; // data_length
    skip_docs_with_field(meta)?;
    let num_docs_with_field = meta.read_le_int()?;
    let min_length = meta.read_le_int()?;
    let max_length = meta.read_le_int()?;

    if max_length > min_length {
        skip_direct_monotonic_addresses(meta, num_docs_with_field as i64 + 1)?;
    }

    Ok(DocValuesEntry {
        num_docs_with_field,
    })
}

fn read_sorted(meta: &mut IndexInput<'_>) -> io::Result<DocValuesEntry> {
    let num_values = read_values_num_values(meta)?;
    skip_terms_dict(meta)?;
    Ok(DocValuesEntry {
        num_docs_with_field: num_values as i32,
    })
}

fn read_sorted_numeric(meta: &mut IndexInput<'_>) -> io::Result<DocValuesEntry> {
    let num_values = read_values_num_values(meta)?;
    let num_docs_with_field = meta.read_le_int()?;

    if num_values > num_docs_with_field as i64 {
        skip_direct_monotonic_addresses(meta, num_docs_with_field as i64 + 1)?;
    }

    Ok(DocValuesEntry {
        num_docs_with_field,
    })
}

fn read_sorted_set(meta: &mut IndexInput<'_>) -> io::Result<DocValuesEntry> {
    let is_multi_valued = meta.read_byte()?;

    if is_multi_valued == 0 {
        // Single-valued: like SORTED
        let num_values = read_values_num_values(meta)?;
        skip_terms_dict(meta)?;
        Ok(DocValuesEntry {
            num_docs_with_field: num_values as i32,
        })
    } else {
        // Multi-valued: like SORTED_NUMERIC + terms dict
        let num_values = read_values_num_values(meta)?;
        let num_docs_with_field = meta.read_le_int()?;

        if num_values > num_docs_with_field as i64 {
            skip_direct_monotonic_addresses(meta, num_docs_with_field as i64 + 1)?;
        }

        skip_terms_dict(meta)?;

        Ok(DocValuesEntry {
            num_docs_with_field,
        })
    }
}

// ---------------------------------------------------------------------------
// Skip helpers — advance past metadata we don't need yet
// ---------------------------------------------------------------------------

/// Skips DirectMonotonicReader metadata blocks in the meta stream.
///
/// Each block is 21 bytes: min(i64) + avgInc(i32) + offset(i64) + bpv(u8).
fn skip_direct_monotonic_meta_blocks(
    meta: &mut IndexInput<'_>,
    num_values: i64,
    block_shift: u32,
) -> io::Result<()> {
    let block_size = 1i64 << block_shift;
    let num_blocks = (num_values + block_size - 1) / block_size;
    // 21 bytes per block: i64(8) + i32(4) + i64(8) + u8(1)
    meta.skip_bytes(num_blocks as usize * 21)?;
    Ok(())
}

/// Skips DirectMonotonic addresses metadata: offset, blockShift vint,
/// DM meta blocks, and length.
fn skip_direct_monotonic_addresses(meta: &mut IndexInput<'_>, num_values: i64) -> io::Result<()> {
    let _addresses_offset = meta.read_le_long()?;
    let block_shift = meta.read_vint()? as u32;
    skip_direct_monotonic_meta_blocks(meta, num_values, block_shift)?;
    let _addresses_length = meta.read_le_long()?;
    Ok(())
}

/// Skips the terms dictionary metadata written by `add_terms_dict()`.
fn skip_terms_dict(meta: &mut IndexInput<'_>) -> io::Result<()> {
    let num_terms = meta.read_vlong()?;
    let block_shift = meta.read_le_int()? as u32;

    // Terms block addresses: DirectMonotonic meta blocks
    let terms_block_size = 1i64 << block_shift;
    let num_term_blocks = (num_terms + terms_block_size - 1) / terms_block_size;
    // DM writer adds entries at block boundaries, so num DM values = num_term_blocks
    skip_direct_monotonic_meta_blocks(meta, num_term_blocks, DIRECT_MONOTONIC_BLOCK_SHIFT)?;

    let _max_term_length = meta.read_le_int()?;
    let _max_block_length = meta.read_le_int()?;
    let _terms_data_offset = meta.read_le_long()?;
    let _terms_data_length = meta.read_le_long()?;
    let _terms_addresses_offset = meta.read_le_long()?;
    let _terms_addresses_length = meta.read_le_long()?;

    // Reverse index
    let reverse_index_shift = meta.read_le_int()? as u32;
    let reverse_block_size = 1i64 << reverse_index_shift;
    let num_reverse_blocks = (num_terms + reverse_block_size - 1) / reverse_block_size;
    // Reverse index DM writer adds entries at reverse index boundaries + final entry
    skip_direct_monotonic_meta_blocks(meta, num_reverse_blocks + 1, DIRECT_MONOTONIC_BLOCK_SHIFT)?;

    let _reverse_index_offset = meta.read_le_long()?;
    let _reverse_index_length = meta.read_le_long()?;
    let _reverse_addresses_offset = meta.read_le_long()?;
    let _reverse_addresses_length = meta.read_le_long()?;

    Ok(())
}

// ---------------------------------------------------------------------------
// BufferedDocValuesProducer — in-memory doc values from the indexing pipeline
// ---------------------------------------------------------------------------

use std::collections::{BTreeSet, HashMap};

use crate::codecs::lucene90::doc_values::{
    BinaryDocValue, DocValuesAccumulator, DocValuesFieldData, NumericDocValue, SortedDocValue,
    SortedNumericDocValue, SortedSetDocValue,
};
use crate::index::doc_values_iterators::DocValuesIterator;
use crate::search::DocIdSetIterator;
use crate::search::doc_id_set_iterator::NO_MORE_DOCS;

/// Per-field buffered doc values data borrowed from the accumulator.
enum BufferedFieldDocValues<'a> {
    Numeric(&'a [NumericDocValue]),
    Binary(&'a [BinaryDocValue]),
    Sorted(&'a [SortedDocValue]),
    SortedNumeric(&'a [SortedNumericDocValue]),
    SortedSet(&'a [SortedSetDocValue]),
}

/// In-memory [`DocValuesProducer`] borrowing from indexing pipeline buffers.
///
/// Each call to a `get_*` method returns a fresh iterator over the borrowed
/// data, allowing the writer to make multiple passes.
#[derive(Debug)]
pub struct BufferedDocValuesProducer<'a> {
    /// Per-field doc values data indexed by field number.
    fields: Vec<Option<BufferedFieldDocValues<'a>>>,
}

impl fmt::Debug for BufferedFieldDocValues<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Numeric(v) => write!(f, "Numeric({})", v.len()),
            Self::Binary(v) => write!(f, "Binary({})", v.len()),
            Self::Sorted(v) => write!(f, "Sorted({})", v.len()),
            Self::SortedNumeric(v) => write!(f, "SortedNumeric({})", v.len()),
            Self::SortedSet(v) => write!(f, "SortedSet({})", v.len()),
        }
    }
}

impl<'a> BufferedDocValuesProducer<'a> {
    /// Creates a new buffered producer borrowing from accumulated doc values data.
    pub(crate) fn new(fields_data: &'a [DocValuesFieldData]) -> Self {
        let max_field = fields_data
            .iter()
            .map(|f| f.number as usize + 1)
            .max()
            .unwrap_or(0);
        let mut fields = Vec::with_capacity(max_field);
        fields.resize_with(max_field, || None);

        for field in fields_data {
            let buffered = match &field.doc_values {
                DocValuesAccumulator::Numeric(vals) => BufferedFieldDocValues::Numeric(vals),
                DocValuesAccumulator::Binary(vals) => BufferedFieldDocValues::Binary(vals),
                DocValuesAccumulator::Sorted(vals) => BufferedFieldDocValues::Sorted(vals),
                DocValuesAccumulator::SortedNumeric(vals) => {
                    BufferedFieldDocValues::SortedNumeric(vals)
                }
                DocValuesAccumulator::SortedSet(vals) => BufferedFieldDocValues::SortedSet(vals),
            };
            fields[field.number as usize] = Some(buffered);
        }

        Self { fields }
    }
}

impl DocValuesProducer for BufferedDocValuesProducer<'_> {
    fn get_numeric(
        &self,
        field_info: &FieldInfo,
    ) -> io::Result<Option<Box<dyn NumericDocValues + '_>>> {
        match self.fields.get(field_info.number() as usize) {
            Some(Some(BufferedFieldDocValues::Numeric(vals))) if !vals.is_empty() => {
                Ok(Some(Box::new(BufferedNumericDV {
                    entries: vals,
                    pos: -1,
                })))
            }
            _ => Ok(None),
        }
    }

    fn get_binary(
        &self,
        field_info: &FieldInfo,
    ) -> io::Result<Option<Box<dyn BinaryDocValues + '_>>> {
        match self.fields.get(field_info.number() as usize) {
            Some(Some(BufferedFieldDocValues::Binary(vals))) if !vals.is_empty() => {
                Ok(Some(Box::new(BufferedBinaryDV {
                    entries: vals,
                    pos: -1,
                })))
            }
            _ => Ok(None),
        }
    }

    fn get_sorted(
        &self,
        field_info: &FieldInfo,
    ) -> io::Result<Option<Box<dyn SortedDocValues + '_>>> {
        match self.fields.get(field_info.number() as usize) {
            Some(Some(BufferedFieldDocValues::Sorted(vals))) if !vals.is_empty() => {
                Ok(Some(Box::new(BufferedSortedDV::new(vals))))
            }
            _ => Ok(None),
        }
    }

    fn get_sorted_numeric(
        &self,
        field_info: &FieldInfo,
    ) -> io::Result<Option<Box<dyn SortedNumericDocValues + '_>>> {
        match self.fields.get(field_info.number() as usize) {
            Some(Some(BufferedFieldDocValues::SortedNumeric(vals))) if !vals.is_empty() => {
                Ok(Some(Box::new(BufferedSortedNumericDV::new(vals))))
            }
            _ => Ok(None),
        }
    }

    fn get_sorted_set(
        &self,
        field_info: &FieldInfo,
    ) -> io::Result<Option<Box<dyn SortedSetDocValues + '_>>> {
        match self.fields.get(field_info.number() as usize) {
            Some(Some(BufferedFieldDocValues::SortedSet(vals))) if !vals.is_empty() => {
                Ok(Some(Box::new(BufferedSortedSetDV::new(vals))))
            }
            _ => Ok(None),
        }
    }
}

// --- Buffered iterator: Numeric ---

struct BufferedNumericDV<'a> {
    entries: &'a [NumericDocValue],
    pos: i32,
}

impl fmt::Debug for BufferedNumericDV<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BufferedNumericDV")
            .field("entries", &self.entries.len())
            .field("pos", &self.pos)
            .finish()
    }
}

impl DocIdSetIterator for BufferedNumericDV<'_> {
    fn doc_id(&self) -> i32 {
        if self.pos < 0 {
            -1
        } else if (self.pos as usize) < self.entries.len() {
            self.entries[self.pos as usize].doc_id
        } else {
            NO_MORE_DOCS
        }
    }

    fn next_doc(&mut self) -> io::Result<i32> {
        self.pos += 1;
        Ok(self.doc_id())
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
        self.entries.len() as i64
    }
}

impl DocValuesIterator for BufferedNumericDV<'_> {
    fn advance_exact(&mut self, target: i32) -> io::Result<bool> {
        match self.entries.binary_search_by_key(&target, |e| e.doc_id) {
            Ok(idx) => {
                self.pos = idx as i32;
                Ok(true)
            }
            Err(_) => Ok(false),
        }
    }
}

impl NumericDocValues for BufferedNumericDV<'_> {
    fn long_value(&self) -> io::Result<i64> {
        Ok(self.entries[self.pos as usize].value)
    }
}

// --- Buffered iterator: Binary ---

struct BufferedBinaryDV<'a> {
    entries: &'a [BinaryDocValue],
    pos: i32,
}

impl fmt::Debug for BufferedBinaryDV<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BufferedBinaryDV")
            .field("entries", &self.entries.len())
            .field("pos", &self.pos)
            .finish()
    }
}

impl DocIdSetIterator for BufferedBinaryDV<'_> {
    fn doc_id(&self) -> i32 {
        if self.pos < 0 {
            -1
        } else if (self.pos as usize) < self.entries.len() {
            self.entries[self.pos as usize].doc_id
        } else {
            NO_MORE_DOCS
        }
    }

    fn next_doc(&mut self) -> io::Result<i32> {
        self.pos += 1;
        Ok(self.doc_id())
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
        self.entries.len() as i64
    }
}

impl DocValuesIterator for BufferedBinaryDV<'_> {
    fn advance_exact(&mut self, target: i32) -> io::Result<bool> {
        match self.entries.binary_search_by_key(&target, |e| e.doc_id) {
            Ok(idx) => {
                self.pos = idx as i32;
                Ok(true)
            }
            Err(_) => Ok(false),
        }
    }
}

impl BinaryDocValues for BufferedBinaryDV<'_> {
    fn binary_value(&self) -> io::Result<&[u8]> {
        Ok(&self.entries[self.pos as usize].value)
    }
}

// --- Buffered iterator: Sorted ---

struct BufferedSortedDV<'a> {
    entries: &'a [SortedDocValue],
    sorted_terms: Vec<&'a [u8]>,
    ord_map: HashMap<&'a [u8], i32>,
    pos: i32,
}

impl fmt::Debug for BufferedSortedDV<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BufferedSortedDV")
            .field("entries", &self.entries.len())
            .field("pos", &self.pos)
            .finish()
    }
}

impl<'a> BufferedSortedDV<'a> {
    fn new(entries: &'a [SortedDocValue]) -> Self {
        let mut unique_terms: BTreeSet<&[u8]> = BTreeSet::new();
        for entry in entries {
            unique_terms.insert(&entry.value);
        }

        let mut ord_map = HashMap::with_capacity(unique_terms.len());
        let sorted_terms: Vec<&[u8]> = unique_terms
            .into_iter()
            .enumerate()
            .map(|(i, term)| {
                ord_map.insert(term, i as i32);
                term
            })
            .collect();

        Self {
            entries,
            sorted_terms,
            ord_map,
            pos: -1,
        }
    }
}

impl DocIdSetIterator for BufferedSortedDV<'_> {
    fn doc_id(&self) -> i32 {
        if self.pos < 0 {
            -1
        } else if (self.pos as usize) < self.entries.len() {
            self.entries[self.pos as usize].doc_id
        } else {
            NO_MORE_DOCS
        }
    }

    fn next_doc(&mut self) -> io::Result<i32> {
        self.pos += 1;
        Ok(self.doc_id())
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
        self.entries.len() as i64
    }
}

impl DocValuesIterator for BufferedSortedDV<'_> {
    fn advance_exact(&mut self, target: i32) -> io::Result<bool> {
        match self.entries.binary_search_by_key(&target, |e| e.doc_id) {
            Ok(idx) => {
                self.pos = idx as i32;
                Ok(true)
            }
            Err(_) => Ok(false),
        }
    }
}

impl SortedDocValues for BufferedSortedDV<'_> {
    fn ord_value(&self) -> io::Result<i32> {
        let value = &self.entries[self.pos as usize].value;
        Ok(self.ord_map[value.as_slice()])
    }

    fn lookup_ord(&self, ord: i32) -> io::Result<&[u8]> {
        Ok(self.sorted_terms[ord as usize])
    }

    fn value_count(&self) -> i32 {
        self.sorted_terms.len() as i32
    }
}

// --- Buffered iterator: SortedNumeric ---

struct BufferedSortedNumericDV<'a> {
    entries: &'a [SortedNumericDocValue],
    pos: i32,
    sorted_values: Vec<Vec<i64>>,
    value_idx: usize,
}

impl fmt::Debug for BufferedSortedNumericDV<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BufferedSortedNumericDV")
            .field("entries", &self.entries.len())
            .field("pos", &self.pos)
            .finish()
    }
}

impl<'a> BufferedSortedNumericDV<'a> {
    fn new(entries: &'a [SortedNumericDocValue]) -> Self {
        let sorted_values: Vec<Vec<i64>> = entries
            .iter()
            .map(|entry| {
                let mut v = entry.values.clone();
                v.sort();
                v
            })
            .collect();

        Self {
            entries,
            pos: -1,
            sorted_values,
            value_idx: 0,
        }
    }
}

impl DocIdSetIterator for BufferedSortedNumericDV<'_> {
    fn doc_id(&self) -> i32 {
        if self.pos < 0 {
            -1
        } else if (self.pos as usize) < self.entries.len() {
            self.entries[self.pos as usize].doc_id
        } else {
            NO_MORE_DOCS
        }
    }

    fn next_doc(&mut self) -> io::Result<i32> {
        self.pos += 1;
        self.value_idx = 0;
        Ok(self.doc_id())
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
        self.entries.len() as i64
    }
}

impl DocValuesIterator for BufferedSortedNumericDV<'_> {
    fn advance_exact(&mut self, target: i32) -> io::Result<bool> {
        match self.entries.binary_search_by_key(&target, |e| e.doc_id) {
            Ok(idx) => {
                self.pos = idx as i32;
                self.value_idx = 0;
                Ok(true)
            }
            Err(_) => Ok(false),
        }
    }
}

impl SortedNumericDocValues for BufferedSortedNumericDV<'_> {
    fn doc_value_count(&self) -> i32 {
        self.sorted_values[self.pos as usize].len() as i32
    }

    fn next_value(&mut self) -> io::Result<i64> {
        let val = self.sorted_values[self.pos as usize][self.value_idx];
        self.value_idx += 1;
        Ok(val)
    }
}

// --- Buffered iterator: SortedSet ---

struct BufferedSortedSetDV<'a> {
    entries: &'a [SortedSetDocValue],
    sorted_terms: Vec<&'a [u8]>,
    /// Per-doc ordinal lists (sorted, deduped).
    doc_ords: Vec<Vec<i64>>,
    pos: i32,
    ord_idx: usize,
}

impl fmt::Debug for BufferedSortedSetDV<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BufferedSortedSetDV")
            .field("entries", &self.entries.len())
            .field("pos", &self.pos)
            .finish()
    }
}

impl<'a> BufferedSortedSetDV<'a> {
    fn new(entries: &'a [SortedSetDocValue]) -> Self {
        let mut unique_terms: BTreeSet<&[u8]> = BTreeSet::new();
        for entry in entries {
            for v in &entry.values {
                unique_terms.insert(v);
            }
        }

        let mut ord_map: HashMap<&[u8], i64> = HashMap::with_capacity(unique_terms.len());
        let sorted_terms: Vec<&[u8]> = unique_terms
            .into_iter()
            .enumerate()
            .map(|(i, term)| {
                ord_map.insert(term, i as i64);
                term
            })
            .collect();

        let doc_ords: Vec<Vec<i64>> = entries
            .iter()
            .map(|entry| {
                let mut ords: Vec<i64> =
                    entry.values.iter().map(|v| ord_map[v.as_slice()]).collect();
                ords.sort();
                ords.dedup();
                ords
            })
            .collect();

        Self {
            entries,
            sorted_terms,
            doc_ords,
            pos: -1,
            ord_idx: 0,
        }
    }
}

impl DocIdSetIterator for BufferedSortedSetDV<'_> {
    fn doc_id(&self) -> i32 {
        if self.pos < 0 {
            -1
        } else if (self.pos as usize) < self.entries.len() {
            self.entries[self.pos as usize].doc_id
        } else {
            NO_MORE_DOCS
        }
    }

    fn next_doc(&mut self) -> io::Result<i32> {
        self.pos += 1;
        self.ord_idx = 0;
        Ok(self.doc_id())
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
        self.entries.len() as i64
    }
}

impl DocValuesIterator for BufferedSortedSetDV<'_> {
    fn advance_exact(&mut self, target: i32) -> io::Result<bool> {
        match self.entries.binary_search_by_key(&target, |e| e.doc_id) {
            Ok(idx) => {
                self.pos = idx as i32;
                self.ord_idx = 0;
                Ok(true)
            }
            Err(_) => Ok(false),
        }
    }
}

impl SortedSetDocValues for BufferedSortedSetDV<'_> {
    fn doc_value_count(&self) -> i32 {
        self.doc_ords[self.pos as usize].len() as i32
    }

    fn next_ord(&mut self) -> io::Result<i64> {
        let ord = self.doc_ords[self.pos as usize][self.ord_idx];
        self.ord_idx += 1;
        Ok(ord)
    }

    fn lookup_ord(&self, ord: i64) -> io::Result<&[u8]> {
        Ok(self.sorted_terms[ord as usize])
    }

    fn value_count(&self) -> i64 {
        self.sorted_terms.len() as i64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codecs::lucene90::doc_values::{
        self, BinaryDocValue, DocValuesAccumulator, DocValuesFieldData, NumericDocValue,
        SortedDocValue, SortedNumericDocValue, SortedSetDocValue,
    };
    use crate::document::{DocValuesType, IndexOptions};
    use crate::index::{FieldInfo, FieldInfos};
    use crate::store::{MemoryDirectory, SharedDirectory};
    use crate::test_util;
    use assertables::*;

    fn make_field_info(name: &str, number: u32, dv_type: DocValuesType) -> FieldInfo {
        test_util::make_field_info(name, number, true, IndexOptions::None, dv_type)
    }

    fn test_directory() -> SharedDirectory {
        MemoryDirectory::create()
    }

    fn make_dv_numeric(
        name: &str,
        number: u32,
        values: Vec<NumericDocValue>,
    ) -> DocValuesFieldData {
        DocValuesFieldData {
            name: name.to_string(),
            number,
            doc_values_type: DocValuesType::Numeric,
            doc_values: DocValuesAccumulator::Numeric(values),
        }
    }

    fn make_dv_binary(name: &str, number: u32, values: Vec<BinaryDocValue>) -> DocValuesFieldData {
        DocValuesFieldData {
            name: name.to_string(),
            number,
            doc_values_type: DocValuesType::Binary,
            doc_values: DocValuesAccumulator::Binary(values),
        }
    }

    fn make_dv_sorted(name: &str, number: u32, values: Vec<SortedDocValue>) -> DocValuesFieldData {
        DocValuesFieldData {
            name: name.to_string(),
            number,
            doc_values_type: DocValuesType::Sorted,
            doc_values: DocValuesAccumulator::Sorted(values),
        }
    }

    fn make_dv_sorted_numeric(
        name: &str,
        number: u32,
        values: Vec<SortedNumericDocValue>,
    ) -> DocValuesFieldData {
        DocValuesFieldData {
            name: name.to_string(),
            number,
            doc_values_type: DocValuesType::SortedNumeric,
            doc_values: DocValuesAccumulator::SortedNumeric(values),
        }
    }

    fn make_dv_sorted_set(
        name: &str,
        number: u32,
        values: Vec<SortedSetDocValue>,
    ) -> DocValuesFieldData {
        DocValuesFieldData {
            name: name.to_string(),
            number,
            doc_values_type: DocValuesType::SortedSet,
            doc_values: DocValuesAccumulator::SortedSet(values),
        }
    }

    fn nv(doc_id: i32, value: i64) -> NumericDocValue {
        NumericDocValue { doc_id, value }
    }

    fn bv(doc_id: i32, value: Vec<u8>) -> BinaryDocValue {
        BinaryDocValue { doc_id, value }
    }

    fn sv(doc_id: i32, value: Vec<u8>) -> SortedDocValue {
        SortedDocValue { doc_id, value }
    }

    fn snv(doc_id: i32, values: Vec<i64>) -> SortedNumericDocValue {
        SortedNumericDocValue { doc_id, values }
    }

    fn ssv(doc_id: i32, values: Vec<Vec<u8>>) -> SortedSetDocValue {
        SortedSetDocValue { doc_id, values }
    }

    /// Writes doc values via BufferedDocValuesProducer and opens a reader.
    fn write_and_read(
        field_infos: &FieldInfos,
        fields: &[DocValuesFieldData],
        num_docs: i32,
        suffix: &str,
    ) -> DocValuesReader {
        let segment_id = [0u8; 16];
        let dir = test_directory();

        let producer = BufferedDocValuesProducer::new(fields);
        let fi_refs: Vec<&FieldInfo> = (0..field_infos.len())
            .filter_map(|i| field_infos.field_info_by_number(i as u32))
            .collect();

        doc_values::write(
            &dir,
            "_0",
            suffix,
            &segment_id,
            &fi_refs,
            &producer,
            num_docs,
        )
        .unwrap();
        DocValuesReader::open(&dir, "_0", suffix, &segment_id, field_infos).unwrap()
    }

    #[test]
    fn test_numeric_all_docs() {
        let fi = make_field_info("count", 0, DocValuesType::Numeric);
        let field_infos = FieldInfos::new(vec![fi]);
        let fields = [make_dv_numeric(
            "count",
            0,
            vec![nv(0, 10), nv(1, 20), nv(2, 30)],
        )];

        let reader = write_and_read(&field_infos, &fields, 3, "Lucene90_0");
        assert_eq!(reader.num_docs_with_field(0), Some(3));
    }

    #[test]
    fn test_numeric_sparse() {
        let fi = make_field_info("count", 0, DocValuesType::Numeric);
        let field_infos = FieldInfos::new(vec![fi]);
        let fields = [make_dv_numeric("count", 0, vec![nv(1, 10), nv(3, 20)])];

        let reader = write_and_read(&field_infos, &fields, 5, "Lucene90_0");
        assert_eq!(reader.num_docs_with_field(0), Some(2));
    }

    #[test]
    fn test_numeric_empty() {
        let fi = make_field_info("count", 0, DocValuesType::Numeric);
        let field_infos = FieldInfos::new(vec![fi]);
        let fields = [make_dv_numeric("count", 0, vec![])];

        let reader = write_and_read(&field_infos, &fields, 5, "Lucene90_0");
        assert_eq!(reader.num_docs_with_field(0), Some(0));
    }

    #[test]
    fn test_binary_all_docs() {
        let fi = make_field_info("hash", 0, DocValuesType::Binary);
        let field_infos = FieldInfos::new(vec![fi]);
        let fields = [make_dv_binary(
            "hash",
            0,
            vec![
                bv(0, b"abc".to_vec()),
                bv(1, b"def".to_vec()),
                bv(2, b"ghi".to_vec()),
            ],
        )];

        let reader = write_and_read(&field_infos, &fields, 3, "Lucene90_0");
        assert_eq!(reader.num_docs_with_field(0), Some(3));
    }

    #[test]
    fn test_binary_variable_length() {
        let fi = make_field_info("data", 0, DocValuesType::Binary);
        let field_infos = FieldInfos::new(vec![fi]);
        let fields = [make_dv_binary(
            "data",
            0,
            vec![
                bv(0, b"short".to_vec()),
                bv(1, b"a longer value here".to_vec()),
            ],
        )];

        let reader = write_and_read(&field_infos, &fields, 2, "Lucene90_0");
        assert_eq!(reader.num_docs_with_field(0), Some(2));
    }

    #[test]
    fn test_binary_sparse() {
        let fi = make_field_info("data", 0, DocValuesType::Binary);
        let field_infos = FieldInfos::new(vec![fi]);
        let fields = [make_dv_binary(
            "data",
            0,
            vec![bv(1, b"abc".to_vec()), bv(3, b"def".to_vec())],
        )];

        let reader = write_and_read(&field_infos, &fields, 5, "Lucene90_0");
        assert_eq!(reader.num_docs_with_field(0), Some(2));
    }

    #[test]
    fn test_sorted() {
        let fi = make_field_info("category", 0, DocValuesType::Sorted);
        let field_infos = FieldInfos::new(vec![fi]);
        let fields = [make_dv_sorted(
            "category",
            0,
            vec![
                sv(0, b"alpha".to_vec()),
                sv(1, b"beta".to_vec()),
                sv(2, b"alpha".to_vec()),
            ],
        )];

        let reader = write_and_read(&field_infos, &fields, 3, "Lucene90_0");
        assert_eq!(reader.num_docs_with_field(0), Some(3));
    }

    #[test]
    fn test_sorted_numeric_single_valued() {
        let fi = make_field_info("priority", 0, DocValuesType::SortedNumeric);
        let field_infos = FieldInfos::new(vec![fi]);
        let fields = [make_dv_sorted_numeric(
            "priority",
            0,
            vec![snv(0, vec![100]), snv(1, vec![200]), snv(2, vec![300])],
        )];

        let reader = write_and_read(&field_infos, &fields, 3, "Lucene90_0");
        assert_eq!(reader.num_docs_with_field(0), Some(3));
    }

    #[test]
    fn test_sorted_numeric_multi_valued() {
        let fi = make_field_info("tags", 0, DocValuesType::SortedNumeric);
        let field_infos = FieldInfos::new(vec![fi]);
        let fields = [make_dv_sorted_numeric(
            "tags",
            0,
            vec![snv(0, vec![1, 2, 3]), snv(1, vec![4]), snv(2, vec![5, 6])],
        )];

        let reader = write_and_read(&field_infos, &fields, 3, "Lucene90_0");
        assert_eq!(reader.num_docs_with_field(0), Some(3));
    }

    #[test]
    fn test_sorted_numeric_sparse() {
        let fi = make_field_info("tags", 0, DocValuesType::SortedNumeric);
        let field_infos = FieldInfos::new(vec![fi]);
        let fields = [make_dv_sorted_numeric(
            "tags",
            0,
            vec![snv(1, vec![10, 20]), snv(3, vec![30])],
        )];

        let reader = write_and_read(&field_infos, &fields, 5, "Lucene90_0");
        assert_eq!(reader.num_docs_with_field(0), Some(2));
    }

    #[test]
    fn test_sorted_set_single_valued() {
        let fi = make_field_info("tag", 0, DocValuesType::SortedSet);
        let field_infos = FieldInfos::new(vec![fi]);
        let fields = [make_dv_sorted_set(
            "tag",
            0,
            vec![
                ssv(0, vec![b"a".to_vec()]),
                ssv(1, vec![b"b".to_vec()]),
                ssv(2, vec![b"c".to_vec()]),
            ],
        )];

        let reader = write_and_read(&field_infos, &fields, 3, "Lucene90_0");
        assert_eq!(reader.num_docs_with_field(0), Some(3));
    }

    #[test]
    fn test_sorted_set_multi_valued() {
        let fi = make_field_info("tags", 0, DocValuesType::SortedSet);
        let field_infos = FieldInfos::new(vec![fi]);
        let fields = [make_dv_sorted_set(
            "tags",
            0,
            vec![
                ssv(0, vec![b"a".to_vec(), b"b".to_vec()]),
                ssv(1, vec![b"c".to_vec()]),
                ssv(2, vec![b"a".to_vec(), b"d".to_vec()]),
            ],
        )];

        let reader = write_and_read(&field_infos, &fields, 3, "Lucene90_0");
        assert_eq!(reader.num_docs_with_field(0), Some(3));
    }

    #[test]
    fn test_multiple_fields_mixed_types() {
        let fi_num = make_field_info("count", 0, DocValuesType::Numeric);
        let fi_bin = make_field_info("hash", 1, DocValuesType::Binary);
        let fi_sn = make_field_info("priority", 2, DocValuesType::SortedNumeric);
        let field_infos = FieldInfos::new(vec![fi_num, fi_bin, fi_sn]);
        let fields = [
            make_dv_numeric("count", 0, vec![nv(0, 10), nv(1, 20), nv(2, 30)]),
            make_dv_binary(
                "hash",
                1,
                vec![bv(0, b"abc".to_vec()), bv(1, b"def".to_vec())],
            ),
            make_dv_sorted_numeric("priority", 2, vec![snv(0, vec![1]), snv(2, vec![3])]),
        ];

        let reader = write_and_read(&field_infos, &fields, 3, "Lucene90_0");
        assert_eq!(reader.num_docs_with_field(0), Some(3));
        assert_eq!(reader.num_docs_with_field(1), Some(2));
        assert_eq!(reader.num_docs_with_field(2), Some(2));
    }

    #[test]
    fn test_nonexistent_field() {
        let fi = make_field_info("count", 0, DocValuesType::Numeric);
        let field_infos = FieldInfos::new(vec![fi]);
        let fields = [make_dv_numeric("count", 0, vec![nv(0, 10)])];

        let reader = write_and_read(&field_infos, &fields, 1, "Lucene90_0");
        assert_none!(reader.num_docs_with_field(99));
    }

    // --- BufferedDocValuesProducer tests ---

    fn make_fi(name: &str, number: u32, dv_type: DocValuesType) -> FieldInfo {
        make_field_info(name, number, dv_type)
    }

    #[test]
    fn buffered_numeric_iteration() {
        let fields = [make_dv_numeric(
            "n",
            0,
            vec![nv(0, 10), nv(2, 30), nv(4, 50)],
        )];
        let producer = BufferedDocValuesProducer::new(&fields);
        let fi = make_fi("n", 0, DocValuesType::Numeric);

        let mut iter = producer.get_numeric(&fi).unwrap().unwrap();
        assert_eq!(iter.next_doc().unwrap(), 0);
        assert_eq!(iter.long_value().unwrap(), 10);
        assert_eq!(iter.next_doc().unwrap(), 2);
        assert_eq!(iter.long_value().unwrap(), 30);
        assert_eq!(iter.next_doc().unwrap(), 4);
        assert_eq!(iter.long_value().unwrap(), 50);
        assert_eq!(iter.next_doc().unwrap(), NO_MORE_DOCS);
    }

    #[test]
    fn buffered_numeric_advance_exact() {
        let fields = [make_dv_numeric(
            "n",
            0,
            vec![nv(0, 10), nv(2, 30), nv(4, 50)],
        )];
        let producer = BufferedDocValuesProducer::new(&fields);
        let fi = make_fi("n", 0, DocValuesType::Numeric);

        let mut iter = producer.get_numeric(&fi).unwrap().unwrap();
        assert!(iter.advance_exact(2).unwrap());
        assert_eq!(iter.long_value().unwrap(), 30);
        assert!(!iter.advance_exact(3).unwrap());
        assert!(iter.advance_exact(4).unwrap());
        assert_eq!(iter.long_value().unwrap(), 50);
    }

    #[test]
    fn buffered_binary_iteration() {
        let fields = [make_dv_binary(
            "b",
            0,
            vec![bv(0, b"abc".to_vec()), bv(1, b"def".to_vec())],
        )];
        let producer = BufferedDocValuesProducer::new(&fields);
        let fi = make_fi("b", 0, DocValuesType::Binary);

        let mut iter = producer.get_binary(&fi).unwrap().unwrap();
        assert!(iter.advance_exact(0).unwrap());
        assert_eq!(iter.binary_value().unwrap(), b"abc");
        assert!(iter.advance_exact(1).unwrap());
        assert_eq!(iter.binary_value().unwrap(), b"def");
        assert!(!iter.advance_exact(2).unwrap());
    }

    #[test]
    fn buffered_sorted_ordinals() {
        let fields = [make_dv_sorted(
            "s",
            0,
            vec![
                sv(0, b"beta".to_vec()),
                sv(1, b"alpha".to_vec()),
                sv(2, b"beta".to_vec()),
            ],
        )];
        let producer = BufferedDocValuesProducer::new(&fields);
        let fi = make_fi("s", 0, DocValuesType::Sorted);

        let mut iter = producer.get_sorted(&fi).unwrap().unwrap();
        assert_eq!(iter.value_count(), 2); // alpha=0, beta=1

        assert!(iter.advance_exact(0).unwrap());
        assert_eq!(iter.ord_value().unwrap(), 1); // "beta"
        assert_eq!(iter.lookup_ord(0).unwrap(), b"alpha");
        assert_eq!(iter.lookup_ord(1).unwrap(), b"beta");

        assert!(iter.advance_exact(1).unwrap());
        assert_eq!(iter.ord_value().unwrap(), 0); // "alpha"
    }

    #[test]
    fn buffered_sorted_numeric_values() {
        let fields = [make_dv_sorted_numeric(
            "sn",
            0,
            vec![snv(0, vec![30, 10, 20]), snv(2, vec![5])],
        )];
        let producer = BufferedDocValuesProducer::new(&fields);
        let fi = make_fi("sn", 0, DocValuesType::SortedNumeric);

        let mut iter = producer.get_sorted_numeric(&fi).unwrap().unwrap();
        assert!(iter.advance_exact(0).unwrap());
        assert_eq!(iter.doc_value_count(), 3);
        assert_eq!(iter.next_value().unwrap(), 10); // sorted
        assert_eq!(iter.next_value().unwrap(), 20);
        assert_eq!(iter.next_value().unwrap(), 30);

        assert!(!iter.advance_exact(1).unwrap());

        assert!(iter.advance_exact(2).unwrap());
        assert_eq!(iter.doc_value_count(), 1);
        assert_eq!(iter.next_value().unwrap(), 5);
    }

    #[test]
    fn buffered_sorted_set_ordinals() {
        let fields = [make_dv_sorted_set(
            "ss",
            0,
            vec![
                ssv(0, vec![b"b".to_vec(), b"a".to_vec()]),
                ssv(1, vec![b"c".to_vec()]),
            ],
        )];
        let producer = BufferedDocValuesProducer::new(&fields);
        let fi = make_fi("ss", 0, DocValuesType::SortedSet);

        let mut iter = producer.get_sorted_set(&fi).unwrap().unwrap();
        assert_eq!(iter.value_count(), 3); // a=0, b=1, c=2
        assert_eq!(iter.lookup_ord(0).unwrap(), b"a");
        assert_eq!(iter.lookup_ord(1).unwrap(), b"b");
        assert_eq!(iter.lookup_ord(2).unwrap(), b"c");

        assert!(iter.advance_exact(0).unwrap());
        assert_eq!(iter.doc_value_count(), 2);
        assert_eq!(iter.next_ord().unwrap(), 0); // "a" (sorted)
        assert_eq!(iter.next_ord().unwrap(), 1); // "b"

        assert!(iter.advance_exact(1).unwrap());
        assert_eq!(iter.doc_value_count(), 1);
        assert_eq!(iter.next_ord().unwrap(), 2); // "c"
    }

    #[test]
    fn buffered_empty_field_returns_none() {
        let fields = [make_dv_numeric("n", 0, vec![])];
        let producer = BufferedDocValuesProducer::new(&fields);
        let fi = make_fi("n", 0, DocValuesType::Numeric);
        assert_none!(producer.get_numeric(&fi).unwrap());
    }

    #[test]
    fn buffered_missing_field_returns_none() {
        let fields = [make_dv_numeric("n", 0, vec![nv(0, 10)])];
        let producer = BufferedDocValuesProducer::new(&fields);
        let fi = make_fi("other", 5, DocValuesType::Numeric);
        assert_none!(producer.get_numeric(&fi).unwrap());
    }

    #[test]
    fn buffered_fresh_iterator_each_call() {
        let fields = [make_dv_numeric("n", 0, vec![nv(0, 10), nv(1, 20)])];
        let producer = BufferedDocValuesProducer::new(&fields);
        let fi = make_fi("n", 0, DocValuesType::Numeric);

        // First iterator: advance to end
        let mut iter1 = producer.get_numeric(&fi).unwrap().unwrap();
        assert_eq!(iter1.next_doc().unwrap(), 0);
        assert_eq!(iter1.next_doc().unwrap(), 1);
        assert_eq!(iter1.next_doc().unwrap(), NO_MORE_DOCS);

        // Second iterator: starts fresh
        let mut iter2 = producer.get_numeric(&fi).unwrap().unwrap();
        assert_eq!(iter2.next_doc().unwrap(), 0);
        assert_eq!(iter2.long_value().unwrap(), 10);
    }
}
