// SPDX-License-Identifier: Apache-2.0

//! Stored fields reader for the Lucene90 compressing stored fields format.
//!
//! Reads `.fdt` (field data), `.fdx` (index), and `.fdm` (metadata) files
//! produced by the Lucene90 compressing stored fields writer.

use std::io;
use std::str;

use crate::codecs::codec_file_handle::{CodecFileHandle, IndexFile};
use crate::codecs::codec_headers;
use crate::codecs::lucene90::stored_fields::{
    DAY, DAY_ENCODING, HOUR, HOUR_ENCODING, SECOND, SECOND_ENCODING, TYPE_BITS, TYPE_BYTE_ARR,
    TYPE_NUMERIC_DOUBLE, TYPE_NUMERIC_FLOAT, TYPE_NUMERIC_INT, TYPE_NUMERIC_LONG, TYPE_STRING,
};
use crate::codecs::packed_readers::DirectMonotonicReader;
use crate::document::StoredValue;
use crate::encoding::lz4;
use crate::encoding::zigzag;
use crate::store::{Directory, FileBacking, IndexInput};

const STORED_FIELDS_INTS_BLOCK_SIZE: usize = 128;

/// Reads the fields index from `.fdm` metadata and `.fdx` index files.
///
/// Maintains two [`DirectMonotonicReader`]s:
/// - `docs`: maps chunk index to starting doc ID
/// - `start_pointers`: maps chunk index to file pointer in `.fdt`
///
/// The readers own per-block metadata only; packed data is borrowed from the
/// caller-supplied `.fdx` bytes at lookup time.
pub(crate) struct FieldsIndexReader {
    docs: DirectMonotonicReader,
    start_pointers: DirectMonotonicReader,
    /// Number of entries in the index (includes sentinel).
    pub(crate) num_chunks: u32,
}

impl FieldsIndexReader {
    /// Loads the fields index metadata from the meta stream.
    pub(crate) fn open(meta_input: &mut IndexInput<'_>) -> io::Result<Self> {
        let _num_docs = meta_input.read_le_int()?;
        let block_shift = meta_input.read_le_int()? as u32;
        let num_chunks = meta_input.read_le_int()? as u32;
        let docs_start_pointer = meta_input.read_le_long()? as u64;

        let docs = DirectMonotonicReader::load_with_shift(
            meta_input,
            num_chunks,
            docs_start_pointer,
            block_shift,
        )?;

        let start_pointers_start = meta_input.read_le_long()? as u64;

        let start_pointers = DirectMonotonicReader::load_with_shift(
            meta_input,
            num_chunks,
            start_pointers_start,
            block_shift,
        )?;

        let _start_pointers_end = meta_input.read_le_long()?;
        let _max_pointer = meta_input.read_le_long()?;

        Ok(Self {
            docs,
            start_pointers,
            num_chunks,
        })
    }

    /// Returns the chunk index containing the given doc ID.
    fn block_id(&self, doc_id: u32, fdx: &[u8]) -> io::Result<u32> {
        // Binary search: find the last chunk whose starting doc ID <= doc_id
        let mut lo = 0u32;
        let mut hi = self.num_chunks;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let mid_doc = self.docs.get(mid as u64, fdx)? as u32;
            if mid_doc <= doc_id {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        // lo is the first chunk with starting doc > doc_id, so the answer is lo - 1
        if lo == 0 {
            return Err(io::Error::other(format!(
                "doc {doc_id} not found in any chunk"
            )));
        }
        Ok(lo - 1)
    }

    /// Returns the file pointer in `.fdt` where the given chunk starts.
    fn block_start_pointer(&self, block: u32, fdx: &[u8]) -> io::Result<u64> {
        Ok(self.start_pointers.get(block as u64, fdx)? as u64)
    }
}

/// A decoded stored field with its field number and value.
pub struct StoredField {
    /// The field number (matches the field's position in FieldInfos).
    pub field_number: u32,
    /// The stored value.
    pub value: StoredValue,
}

/// Cached state for the most recently loaded block.
///
/// Mirrors Java's `BlockState` in `Lucene90CompressingStoredFieldsReader`.
/// Caches block metadata and decompressed data so that consecutive reads
/// within the same block skip seeking, metadata parsing, and decompression.
struct BlockState {
    /// First document ID in this block.
    doc_base: u32,
    /// Number of documents in this block (0 = no block loaded).
    chunk_docs: u32,
    /// Number of stored fields per document.
    num_stored_fields: Box<[i64]>,
    /// Cumulative byte offsets within the decompressed data (length = chunk_docs + 1).
    offsets: Box<[i64]>,
    /// Full decompressed chunk data.
    decompressed: Box<[u8]>,
}

impl BlockState {
    /// Creates an empty block state that matches no document.
    fn new() -> Self {
        Self {
            doc_base: 0,
            chunk_docs: 0,
            num_stored_fields: Box::new([]),
            offsets: Box::new([]),
            decompressed: Box::new([]),
        }
    }

    /// Returns `true` if `doc_id` is within the currently loaded block.
    fn contains(&self, doc_id: u32) -> bool {
        doc_id >= self.doc_base && doc_id < self.doc_base + self.chunk_docs
    }

    /// Extracts and decodes one document's stored fields from the cached block.
    fn document(&self, doc_id: u32) -> io::Result<Vec<StoredField>> {
        let index = (doc_id - self.doc_base) as usize;
        let doc_offset = self.offsets[index] as usize;
        let doc_length = self.offsets[index + 1] as usize - doc_offset;
        let num_fields = self.num_stored_fields[index] as usize;
        let doc_data = &self.decompressed[doc_offset..doc_offset + doc_length];
        decode_fields(doc_data, num_fields)
    }
}

/// Reads stored fields from a segment.
///
/// Opens `.fdt`, `.fdx`, and `.fdm` files and provides document-level
/// access to stored field values. Caches the most recently loaded block
/// so that consecutive reads within the same chunk avoid redundant I/O
/// and decompression.
pub struct StoredFieldsReader {
    fdt: FileBacking,
    fdx: FileBacking,
    index_reader: FieldsIndexReader,
    chunk_size: i32,
    state: BlockState,
}

impl StoredFieldsReader {
    /// Opens a stored fields reader for the given segment.
    ///
    /// 1. Open `.fdt` (data), validate header and footer — keep backing
    /// 2. Open `.fdx` (index), validate header and footer — keep backing
    /// 3. Open `.fdm` (meta), verify full CRC over file bytes, parse metadata
    /// 4. Validate dirty chunk invariants
    pub fn open(
        directory: &dyn Directory,
        segment_name: &str,
        segment_suffix: &str,
        segment_id: &[u8; codec_headers::ID_LENGTH],
    ) -> io::Result<Self> {
        let fdt = CodecFileHandle::open(
            directory,
            IndexFile::StoredFieldsData,
            segment_name,
            segment_id,
            segment_suffix,
        )?;
        let fdx = CodecFileHandle::open(
            directory,
            IndexFile::StoredFieldsIndex,
            segment_name,
            segment_id,
            segment_suffix,
        )?;
        let fdm = CodecFileHandle::open(
            directory,
            IndexFile::StoredFieldsMeta,
            segment_name,
            segment_id,
            segment_suffix,
        )?;

        let mut meta = fdm.body();
        let chunk_size = meta.read_vint()?;
        let index_reader = FieldsIndexReader::open(&mut meta)?;

        // Read and validate dirty chunk counts
        let num_chunks = meta.read_vlong()?;
        let num_dirty_chunks = meta.read_vlong()?;
        let num_dirty_docs = meta.read_vlong()?;

        if num_dirty_chunks > num_chunks {
            return Err(io::Error::other(format!(
                "invalid numDirtyChunks: dirty={num_dirty_chunks} total={num_chunks}"
            )));
        }
        if (num_dirty_chunks == 0) != (num_dirty_docs == 0) {
            return Err(io::Error::other(format!(
                "dirty chunks/docs mismatch: dirtyChunks={num_dirty_chunks} dirtyDocs={num_dirty_docs}"
            )));
        }
        if num_dirty_docs < num_dirty_chunks {
            return Err(io::Error::other(format!(
                "numDirtyDocs < numDirtyChunks: dirtyDocs={num_dirty_docs} dirtyChunks={num_dirty_chunks}"
            )));
        }

        Ok(Self {
            fdt: fdt.into_backing(),
            fdx: fdx.into_backing(),
            index_reader,
            chunk_size,
            state: BlockState::new(),
        })
    }

    /// Reads all stored fields for the given document.
    ///
    /// **TODO: Replace with visitor pattern before production use.** Java's
    /// equivalent is `document(int docID, StoredFieldVisitor visitor)` which
    /// allows selective field loading and early termination. This method
    /// eagerly decodes all fields. Currently only used in tests.
    pub fn document(&mut self, doc_id: u32) -> io::Result<Vec<StoredField>> {
        if !self.state.contains(doc_id) {
            self.reset_state(doc_id)?;
        }
        self.state.document(doc_id)
    }

    /// Loads block metadata and decompressed data for the block containing `doc_id`.
    ///
    /// On failure, invalidates the cached state so stale data is never reused.
    fn reset_state(&mut self, doc_id: u32) -> io::Result<()> {
        // Invalidate first — if anything below fails, state stays empty
        self.state.chunk_docs = 0;

        let block = self.index_reader.block_id(doc_id, self.fdx.as_bytes())?;
        let start_pointer = self
            .index_reader
            .block_start_pointer(block, self.fdx.as_bytes())?;

        let mut stream = IndexInput::unnamed(self.fdt.as_bytes());
        stream.seek(start_pointer as usize)?;

        // Read chunk header
        let doc_base = stream.read_vint()? as u32;
        let token = stream.read_vint()? as u32;
        let chunk_docs = token >> 2;
        let sliced = (token & 1) != 0;

        if chunk_docs == 0 {
            return Err(io::Error::other("chunk with 0 docs"));
        }

        let doc_in_chunk = doc_id - doc_base;
        if doc_in_chunk >= chunk_docs {
            return Err(io::Error::other(format!(
                "doc {doc_id} not in chunk (base={doc_base}, docs={chunk_docs})"
            )));
        }

        // Read numStoredFields and lengths arrays
        let (num_stored_fields, offsets) = if chunk_docs == 1 {
            let nsf = stream.read_vint()?;
            let length = stream.read_vint()?;
            (vec![nsf as i64], vec![0i64, length as i64])
        } else {
            let mut nsf = vec![0i64; chunk_docs as usize];
            read_stored_fields_ints(&mut stream, chunk_docs as usize, &mut nsf)?;

            let mut lengths = vec![0i64; chunk_docs as usize + 1];
            read_stored_fields_ints(&mut stream, chunk_docs as usize, &mut lengths[1..])?;
            // Convert lengths to cumulative offsets
            for i in 1..=chunk_docs as usize {
                lengths[i] += lengths[i - 1];
            }
            (nsf, lengths)
        };

        let total_length = *offsets.last().unwrap() as usize;

        // Decompress the chunk data
        let decompressed = decompress_chunk(&mut stream, self.chunk_size, total_length, sliced)?;

        // Store in cached state
        self.state = BlockState {
            doc_base,
            chunk_docs,
            num_stored_fields: num_stored_fields.into_boxed_slice(),
            offsets: offsets.into_boxed_slice(),
            decompressed: decompressed.into_boxed_slice(),
        };

        Ok(())
    }
}

/// Decompresses the chunk data starting at the current stream position.
fn decompress_chunk(
    stream: &mut IndexInput<'_>,
    chunk_size: i32,
    total_length: usize,
    sliced: bool,
) -> io::Result<Vec<u8>> {
    if !sliced {
        // Single LZ4 block with preset dictionary
        decompress_lz4_with_dict(stream, total_length)
    } else {
        // Multiple slices, each up to chunk_size bytes
        let chunk_size = chunk_size as usize;
        let mut result = Vec::with_capacity(total_length);
        let mut remaining = total_length;

        while remaining > 0 {
            let block_len = remaining.min(chunk_size);
            let block_data = decompress_lz4_with_dict(stream, block_len)?;
            result.extend_from_slice(&block_data);
            remaining -= block_len;
        }
        Ok(result)
    }
}

/// Decompresses a single LZ4-with-preset-dict block from the fields stream.
///
/// Format: all compressed lengths first, then all compressed data.
fn decompress_lz4_with_dict(
    stream: &mut IndexInput<'_>,
    decompressed_length: usize,
) -> io::Result<Vec<u8>> {
    let dict_length = stream.read_vint()? as usize;
    let block_length = stream.read_vint()? as usize;

    // Read all compressed lengths first
    let mut compressed_lengths = Vec::new();

    // Dictionary compressed length
    let dict_compressed_len = stream.read_vint()? as usize;
    compressed_lengths.push(dict_compressed_len);

    // Sub-block compressed lengths
    if block_length > 0 {
        let data_length = decompressed_length.saturating_sub(dict_length);
        let num_sub_blocks = data_length.div_ceil(block_length);
        for _ in 0..num_sub_blocks {
            compressed_lengths.push(stream.read_vint()? as usize);
        }
    }

    // Now read all compressed data
    // Decompress dictionary
    let mut dict_compressed = vec![0u8; dict_compressed_len];
    stream.read_bytes(&mut dict_compressed)?;
    let dict = if dict_length > 0 {
        lz4::decompress(&dict_compressed, dict_length)?
    } else {
        Vec::new()
    };

    // Decompress sub-blocks
    if block_length == 0 {
        // No sub-blocks — dictionary is the entire data
        return Ok(dict);
    }

    // Result includes dictionary bytes followed by decompressed sub-blocks
    let mut result = Vec::with_capacity(decompressed_length);
    result.extend_from_slice(&dict);
    let mut data_start = dict_length;

    for &comp_len in &compressed_lengths[1..] {
        let block_decompressed = (decompressed_length - data_start).min(block_length);
        let mut compressed = vec![0u8; comp_len];
        stream.read_bytes(&mut compressed)?;

        let block_data = lz4::decompress_with_prefix(&compressed, block_decompressed, &dict)?;
        result.extend_from_slice(&block_data);
        data_start += block_decompressed;
    }

    Ok(result)
}

/// Decodes stored fields from decompressed data.
fn decode_fields(data: &[u8], num_fields: usize) -> io::Result<Vec<StoredField>> {
    let mut fields = Vec::with_capacity(num_fields);
    let mut reader = IndexInput::unnamed(data);

    for _ in 0..num_fields {
        let info_and_bits = reader.read_vlong()?;
        let field_number = (info_and_bits as u64 >> TYPE_BITS) as u32;
        let type_code = info_and_bits as u64 & ((1 << TYPE_BITS) - 1);

        let value = match type_code {
            TYPE_STRING => {
                let len = reader.read_vint()? as usize;
                let s = str::from_utf8(reader.read_slice(len)?)
                    .map_err(|e| io::Error::other(format!("invalid utf-8: {e}")))?;
                StoredValue::String(s.to_string())
            }
            TYPE_BYTE_ARR => {
                let len = reader.read_vint()? as usize;
                StoredValue::Bytes(reader.read_slice(len)?.to_vec())
            }
            TYPE_NUMERIC_INT => StoredValue::Int(reader.read_zint()?),
            TYPE_NUMERIC_FLOAT => StoredValue::Float(read_zfloat(&mut reader)?),
            TYPE_NUMERIC_LONG => StoredValue::Long(read_tlong(&mut reader)?),
            TYPE_NUMERIC_DOUBLE => StoredValue::Double(read_zdouble(&mut reader)?),
            _ => {
                return Err(io::Error::other(format!(
                    "unknown stored field type: {type_code}"
                )));
            }
        };

        fields.push(StoredField {
            field_number,
            value,
        });
    }

    Ok(fields)
}

// ============================================================
// StoredFieldsInts reading
// ============================================================

/// Reads `count` integers from the input using StoredFieldsInts encoding.
fn read_stored_fields_ints(
    input: &mut IndexInput<'_>,
    count: usize,
    values: &mut [i64],
) -> io::Result<()> {
    let bpv = input.read_byte()?;
    match bpv {
        0 => {
            let v = input.read_vint()? as i64;
            values[..count].fill(v);
        }
        8 => read_ints_8(input, count, values)?,
        16 => read_ints_16(input, count, values)?,
        32 => read_ints_32(input, count, values)?,
        _ => {
            return Err(io::Error::other(format!(
                "unsupported bpv in StoredFieldsInts: {bpv}"
            )));
        }
    }
    Ok(())
}

fn read_ints_8(input: &mut IndexInput<'_>, count: usize, values: &mut [i64]) -> io::Result<()> {
    let mut k = 0;
    while k + STORED_FIELDS_INTS_BLOCK_SIZE <= count {
        // Read 16 LE longs, each packing 8 values
        for i in 0..16 {
            let l = input.read_le_long()? as u64;
            values[k + i] = ((l >> 56) & 0xFF) as i64;
            values[k + 16 + i] = ((l >> 48) & 0xFF) as i64;
            values[k + 32 + i] = ((l >> 40) & 0xFF) as i64;
            values[k + 48 + i] = ((l >> 32) & 0xFF) as i64;
            values[k + 64 + i] = ((l >> 24) & 0xFF) as i64;
            values[k + 80 + i] = ((l >> 16) & 0xFF) as i64;
            values[k + 96 + i] = ((l >> 8) & 0xFF) as i64;
            values[k + 112 + i] = (l & 0xFF) as i64;
        }
        k += STORED_FIELDS_INTS_BLOCK_SIZE;
    }
    while k < count {
        values[k] = input.read_byte()? as i64;
        k += 1;
    }
    Ok(())
}

fn read_ints_16(input: &mut IndexInput<'_>, count: usize, values: &mut [i64]) -> io::Result<()> {
    let mut k = 0;
    while k + STORED_FIELDS_INTS_BLOCK_SIZE <= count {
        for i in 0..32 {
            let l = input.read_le_long()? as u64;
            values[k + i] = ((l >> 48) & 0xFFFF) as i64;
            values[k + 32 + i] = ((l >> 32) & 0xFFFF) as i64;
            values[k + 64 + i] = ((l >> 16) & 0xFFFF) as i64;
            values[k + 96 + i] = (l & 0xFFFF) as i64;
        }
        k += STORED_FIELDS_INTS_BLOCK_SIZE;
    }
    while k < count {
        values[k] = (input.read_le_short()? as u16) as i64;
        k += 1;
    }
    Ok(())
}

fn read_ints_32(input: &mut IndexInput<'_>, count: usize, values: &mut [i64]) -> io::Result<()> {
    let mut k = 0;
    while k + STORED_FIELDS_INTS_BLOCK_SIZE <= count {
        for i in 0..64 {
            let l = input.read_le_long()? as u64;
            values[k + i] = (l >> 32) as i64;
            values[k + 64 + i] = (l & 0xFFFFFFFF) as i64;
        }
        k += STORED_FIELDS_INTS_BLOCK_SIZE;
    }
    while k < count {
        values[k] = input.read_le_int()? as i64;
        k += 1;
    }
    Ok(())
}

// ============================================================
// Field value decoders
// ============================================================

/// Reads a variable-length float (ZFloat encoding).
fn read_zfloat(input: &mut IndexInput<'_>) -> io::Result<f32> {
    let header = input.read_byte()? as u32;
    if header == 0xFF {
        // Negative float: 4 bytes follow
        let bits = input.read_le_int()? as u32;
        Ok(f32::from_bits(bits))
    } else if header >= 0x80 {
        // Small integer [-1..125]
        Ok((header as i32 - 0x80 - 1) as f32)
    } else {
        // Positive float: header is the high byte, 3 more bytes follow
        let b1 = input.read_le_short()? as u16 as u32;
        let b2 = input.read_byte()? as u32;
        let bits = (header << 24) | (b1 << 8) | b2;
        Ok(f32::from_bits(bits))
    }
}

/// Reads a timestamp-compressed long (TLong encoding).
fn read_tlong(input: &mut IndexInput<'_>) -> io::Result<i64> {
    let header = input.read_byte()?;
    let time_encoding = header & 0xC0;
    let mut zig_zag = (header as u64 & 0x1F) as i64;

    if (header & 0x20) != 0 {
        // Upper bits follow
        let upper = input.read_vlong()?;
        zig_zag |= upper << 5;
    }

    let val = zigzag::decode_i64(zig_zag);

    match time_encoding {
        0x00 => Ok(val),
        SECOND_ENCODING => Ok(val * SECOND),
        HOUR_ENCODING => Ok(val * HOUR),
        DAY_ENCODING => Ok(val * DAY),
        _ => Err(io::Error::other(format!(
            "unknown time encoding: {time_encoding:#x}"
        ))),
    }
}

/// Reads a variable-length double (ZDouble encoding).
fn read_zdouble(input: &mut IndexInput<'_>) -> io::Result<f64> {
    let header = input.read_byte()? as u32;
    if header == 0xFF {
        // Negative double: 8 bytes follow
        let bits = input.read_le_long()? as u64;
        Ok(f64::from_bits(bits))
    } else if header == 0xFE {
        // Float-representable: 4 bytes follow
        let float_bits = input.read_le_int()? as u32;
        Ok(f32::from_bits(float_bits) as f64)
    } else if header >= 0x80 {
        // Small integer [-1..124]
        Ok((header as i32 - 0x80 - 1) as f64)
    } else {
        // Positive double: header is the high byte, 7 more bytes follow
        let b4 = input.read_le_int()? as u32 as u64;
        let b2 = input.read_le_short()? as u16 as u64;
        let b1 = input.read_byte()? as u64;
        let bits = ((header as u64) << 56) | (b4 << 24) | (b2 << 8) | b1;
        Ok(f64::from_bits(bits))
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;
    use std::sync::Arc;

    use super::*;
    use crate::document::StoredValue;
    use crate::document::{Document, DocumentBuilder};
    use crate::index::config::IndexWriterConfig;
    use crate::index::field::stored;
    use crate::index::segment_infos;
    use crate::index::writer::IndexWriter;
    use crate::store::{MemoryDirectory, SharedDirectory};
    use assertables::*;

    /// Write a simple index and read stored fields back via StoredFieldsReader.
    fn write_and_read_stored(docs: Vec<Document>) -> (SharedDirectory, Vec<Vec<StoredField>>) {
        let num_docs = docs.len();
        let config = IndexWriterConfig::default().num_threads(1);
        let directory: SharedDirectory = MemoryDirectory::create();
        let writer = IndexWriter::new(config, Arc::clone(&directory));
        for doc in docs {
            writer.add_document(doc).unwrap();
        }
        writer.commit().unwrap();

        // Find the segment info to get segment name and ID
        let files = directory.list_all().unwrap();
        let segments_file = files
            .iter()
            .find(|f| f.starts_with("segments_"))
            .expect("no segments file");
        let infos = segment_infos::read(&*directory, segments_file).unwrap();
        let seg = &infos.segments[0];

        let mut reader = StoredFieldsReader::open(&*directory, &seg.name, "", &seg.id).unwrap();

        let mut results = Vec::new();
        for doc_id in 0..num_docs {
            results.push(reader.document(doc_id as u32).unwrap());
        }

        (directory, results)
    }

    #[test]
    fn test_round_trip_string_and_int() {
        let doc = DocumentBuilder::new()
            .add_field(stored("title").string("Hello World"))
            .add_field(stored("count").int(42))
            .build();

        let doc2 = DocumentBuilder::new()
            .add_field(stored("title").string("Second Doc"))
            .add_field(stored("count").int(99))
            .build();

        let (_, results) = write_and_read_stored(vec![doc, doc2]);

        // Doc 0
        let fields = &results[0];
        assert!(
            fields
                .iter()
                .any(|f| matches!(&f.value, StoredValue::String(s) if s == "Hello World"))
        );
        assert!(
            fields
                .iter()
                .any(|f| matches!(&f.value, StoredValue::Int(42)))
        );

        // Doc 1
        let fields1 = &results[1];
        assert!(
            fields1
                .iter()
                .any(|f| matches!(&f.value, StoredValue::String(s) if s == "Second Doc"))
        );
        assert!(
            fields1
                .iter()
                .any(|f| matches!(&f.value, StoredValue::Int(99)))
        );
    }

    #[test]
    fn test_round_trip_all_types() {
        let doc = DocumentBuilder::new()
            .add_field(stored("s").string("text"))
            .add_field(stored("i").int(123))
            .add_field(stored("l").long(456789))
            .add_field(stored("f").float(3.125))
            .add_field(stored("d").double(2.7))
            .add_field(stored("b").bytes(vec![1, 2, 3]))
            .build();

        let (_, results) = write_and_read_stored(vec![doc]);
        let fields = &results[0];

        assert!(
            fields
                .iter()
                .any(|f| matches!(&f.value, StoredValue::String(s) if s == "text")),
            "missing string"
        );
        assert!(
            fields
                .iter()
                .any(|f| matches!(&f.value, StoredValue::Int(123))),
            "missing int"
        );
        assert!(
            fields
                .iter()
                .any(|f| matches!(&f.value, StoredValue::Long(456789))),
            "missing long"
        );
        assert!(
            fields
                .iter()
                .any(|f| matches!(&f.value, StoredValue::Float(v) if (*v - 3.125).abs() < 0.001)),
            "missing float"
        );
        assert!(
            fields
                .iter()
                .any(|f| matches!(&f.value, StoredValue::Double(v) if (*v - 2.7).abs() < 0.001)),
            "missing double"
        );
        assert!(
            fields
                .iter()
                .any(|f| matches!(&f.value, StoredValue::Bytes(b) if b == &[1, 2, 3])),
            "missing bytes"
        );
    }

    #[test]
    fn test_read_zfloat_small_int() {
        // Small int encoding: header = 0x80 + (1 + val)
        // val=0 -> header=0x81
        let data = [0x81u8];
        let mut reader = IndexInput::unnamed(&data);
        assert_in_delta!(read_zfloat(&mut reader).unwrap(), 0.0, 0.001);

        // val=42 -> header=0x80+43=0xAB
        let data = [0xABu8];
        let mut reader = IndexInput::unnamed(&data);
        assert_in_delta!(read_zfloat(&mut reader).unwrap(), 42.0, 0.001);

        // val=-1 -> header=0x80+0=0x80
        let data = [0x80u8];
        let mut reader = IndexInput::unnamed(&data);
        assert_in_delta!(read_zfloat(&mut reader).unwrap(), -1.0, 0.001);
    }

    #[test]
    fn test_read_zdouble_small_int() {
        // val=0 -> header=0x81
        let data = [0x81u8];
        let mut reader = IndexInput::unnamed(&data);
        assert_in_delta!(read_zdouble(&mut reader).unwrap(), 0.0, 0.001);

        // val=-1 -> header=0x80
        let data = [0x80u8];
        let mut reader = IndexInput::unnamed(&data);
        assert_in_delta!(read_zdouble(&mut reader).unwrap(), -1.0, 0.001);
    }

    #[test]
    fn test_read_tlong_no_encoding() {
        // val=5, no time encoding, no upper bits
        // zigzag(5) = 10, header = 0x00 | 10 = 0x0A
        let data = [0x0Au8];
        let mut reader = IndexInput::unnamed(&data);
        assert_eq!(read_tlong(&mut reader).unwrap(), 5);
    }

    #[test]
    fn test_read_tlong_second_encoding() {
        // val=5000 ms = 5 seconds, SECOND_ENCODING
        // val/SECOND = 5, zigzag(5) = 10
        // header = SECOND_ENCODING | 10 = 0x40 | 0x0A = 0x4A
        let data = [0x4Au8];
        let mut reader = IndexInput::unnamed(&data);
        assert_eq!(read_tlong(&mut reader).unwrap(), 5000);
    }

    #[test]
    fn test_stored_fields_ints_uniform() {
        // All values equal: marker=0, VInt=42
        let data = [0x00u8, 42];
        let mut reader = IndexInput::unnamed(&data);
        let mut values = vec![0i64; 4];
        read_stored_fields_ints(&mut reader, 4, &mut values).unwrap();
        assert_eq!(values, vec![42, 42, 42, 42]);
    }

    #[test]
    fn test_stored_fields_ints_8bit() {
        // 3 values (< 128 block size), 8-bit packing, remainder path
        let data = [8u8, 10, 20, 30]; // marker=8, then 3 bytes
        let mut reader = IndexInput::unnamed(&data);
        let mut values = vec![0i64; 3];
        read_stored_fields_ints(&mut reader, 3, &mut values).unwrap();
        assert_eq!(values, vec![10, 20, 30]);
    }

    // --- Write-then-read round-trip helpers using the writer functions ---

    use crate::codecs::lucene90::stored_fields;
    use crate::store::memory::MemoryIndexOutput;

    /// Write a float with write_zfloat, then read it back with read_zfloat.
    fn zfloat_round_trip(val: f32) -> f32 {
        let mut out = MemoryIndexOutput::new("test".to_string());
        stored_fields::write_zfloat_for_test(&mut out, val).unwrap();
        let bytes = out.bytes();
        let mut reader = IndexInput::unnamed(bytes);
        read_zfloat(&mut reader).unwrap()
    }

    /// Write a double with write_zdouble, then read it back with read_zdouble.
    fn zdouble_round_trip(val: f64) -> f64 {
        let mut out = MemoryIndexOutput::new("test".to_string());
        stored_fields::write_zdouble_for_test(&mut out, val).unwrap();
        let bytes = out.bytes();
        let mut reader = IndexInput::unnamed(bytes);
        read_zdouble(&mut reader).unwrap()
    }

    /// Write a long with write_tlong, then read it back with read_tlong.
    fn tlong_round_trip(val: i64) -> i64 {
        let mut out = MemoryIndexOutput::new("test".to_string());
        stored_fields::write_tlong_for_test(&mut out, val).unwrap();
        let bytes = out.bytes();
        let mut reader = IndexInput::unnamed(bytes);
        read_tlong(&mut reader).unwrap()
    }

    /// Write ints with save_ints, then read them back with read_stored_fields_ints.
    fn stored_ints_round_trip(values: &[i32]) -> Vec<i64> {
        let mut out = MemoryIndexOutput::new("test".to_string());
        stored_fields::save_ints_for_test(values, values.len(), &mut out).unwrap();
        let bytes = out.bytes();
        let mut reader = IndexInput::unnamed(bytes);
        let mut result = vec![0i64; values.len()];
        if values.len() == 1 {
            result[0] = reader.read_vint().unwrap() as i64;
        } else {
            read_stored_fields_ints(&mut reader, values.len(), &mut result).unwrap();
        }
        result
    }

    // --- zfloat tests ---

    #[test]
    fn test_read_zfloat_negative() {
        // Negative float: header=0xFF, then LE int of float bits
        let val = -42.5f32;
        assert_in_delta!(zfloat_round_trip(val), val, 0.001);
    }

    #[test]
    fn test_read_zfloat_positive_non_integer() {
        // Positive non-integer float: header is high byte of float bits
        let val = 3.125f32;
        assert_in_delta!(zfloat_round_trip(val), val, 0.001);
    }

    #[test]
    fn test_read_zfloat_large_positive() {
        assert_in_delta!(zfloat_round_trip(1_000_000.0), 1_000_000.0, 1.0);
    }

    #[test]
    fn test_read_zfloat_boundary_values() {
        // val=125 is the max small int
        assert_in_delta!(zfloat_round_trip(125.0), 125.0, 0.001);
        // val=126 should NOT use small int path (> 0x7D=125)
        assert_in_delta!(zfloat_round_trip(126.0), 126.0, 0.001);
    }

    // --- zdouble tests ---

    #[test]
    fn test_read_zdouble_negative() {
        let val = -99.99;
        assert_in_delta!(zdouble_round_trip(val), val, 0.001);
    }

    #[test]
    fn test_read_zdouble_float_representable() {
        // A value that can be exactly represented as f32 uses the 0xFE path
        let val = 3.25f32 as f64;
        assert_in_delta!(zdouble_round_trip(val), val, 0.001);
    }

    #[test]
    fn test_read_zdouble_positive_non_integer() {
        // A value that cannot be represented as f32 uses the positive double path
        let val = PI; // high precision, not float-representable
        assert_in_delta!(zdouble_round_trip(val), val, 1e-10);
    }

    #[test]
    fn test_read_zdouble_large_negative() {
        assert_in_delta!(zdouble_round_trip(-1e15), -1e15, 1.0);
    }

    // --- tlong tests ---

    #[test]
    fn test_read_tlong_hour_encoding() {
        // 2 hours = 7200000 ms
        let val = 2 * HOUR;
        assert_eq!(tlong_round_trip(val), val);
    }

    #[test]
    fn test_read_tlong_day_encoding() {
        // 3 days = 259200000 ms
        let val = 3 * DAY;
        assert_eq!(tlong_round_trip(val), val);
    }

    #[test]
    fn test_read_tlong_upper_bits() {
        // Large value that requires upper bits (zigzag value > 31)
        // val=100 (not divisible by SECOND), zigzag(100)=200, needs upper bits
        let val = 100i64;
        assert_eq!(tlong_round_trip(val), val);
    }

    #[test]
    fn test_read_tlong_negative() {
        assert_eq!(tlong_round_trip(-5000), -5000);
    }

    #[test]
    fn test_read_tlong_large_timestamp() {
        // Typical epoch millis
        let val = 1_700_000_000_000i64;
        assert_eq!(tlong_round_trip(val), val);
    }

    // --- StoredFieldsInts tests ---

    #[test]
    fn test_stored_fields_ints_16bit_round_trip() {
        let values: Vec<i32> = (0..5).map(|i| 256 + i * 100).collect();
        let result = stored_ints_round_trip(&values);
        for (i, &v) in values.iter().enumerate() {
            assert_eq!(result[i], v as i64, "mismatch at index {i}");
        }
    }

    #[test]
    fn test_stored_fields_ints_32bit_round_trip() {
        let values: Vec<i32> = (0..5).map(|i| 70000 + i * 10000).collect();
        let result = stored_ints_round_trip(&values);
        for (i, &v) in values.iter().enumerate() {
            assert_eq!(result[i], v as i64, "mismatch at index {i}");
        }
    }

    #[test]
    fn test_stored_fields_ints_8bit_block_path() {
        // 130 values triggers the block path (128 values per block + 2 remainder)
        let values: Vec<i32> = (0..130).map(|i| i % 200).collect();
        let result = stored_ints_round_trip(&values);
        for (i, &v) in values.iter().enumerate() {
            assert_eq!(result[i], v as i64, "mismatch at index {i}");
        }
    }

    #[test]
    fn test_stored_fields_ints_16bit_block_path() {
        // 130 values with values > 255 triggers 16-bit block path
        let values: Vec<i32> = (0..130).map(|i| 300 + i).collect();
        let result = stored_ints_round_trip(&values);
        for (i, &v) in values.iter().enumerate() {
            assert_eq!(result[i], v as i64, "mismatch at index {i}");
        }
    }

    #[test]
    fn test_stored_fields_ints_32bit_block_path() {
        // 130 values with values > 65535 triggers 32-bit block path
        let values: Vec<i32> = (0..130).map(|i| 70000 + i * 1000).collect();
        let result = stored_ints_round_trip(&values);
        for (i, &v) in values.iter().enumerate() {
            assert_eq!(result[i], v as i64, "mismatch at index {i}");
        }
    }

    #[test]
    fn test_block_state_cache_sequential_reads() {
        // Write several small docs — they'll all land in the same chunk.
        // Reading them sequentially should hit the cache after the first.
        let mut docs = Vec::new();
        for i in 0..5 {
            let doc = DocumentBuilder::new()
                .add_field(stored("name").string(format!("doc_{i}")))
                .add_field(stored("idx").int(i))
                .build();
            docs.push(doc);
        }

        let config = IndexWriterConfig::default().num_threads(1);
        let directory: SharedDirectory = MemoryDirectory::create();
        let writer = IndexWriter::new(config, Arc::clone(&directory));
        for doc in docs {
            writer.add_document(doc).unwrap();
        }
        writer.commit().unwrap();

        let files = directory.list_all().unwrap();
        let segments_file = files.iter().find(|f| f.starts_with("segments_")).unwrap();
        let infos = segment_infos::read(&*directory, segments_file).unwrap();
        let seg = &infos.segments[0];

        let mut reader = StoredFieldsReader::open(&*directory, &seg.name, "", &seg.id).unwrap();

        // First read loads the block
        let fields0 = reader.document(0).unwrap();
        assert_eq!(fields0.len(), 2);
        assert!(reader.state.contains(0));

        // Subsequent reads should hit the cache (same block)
        for i in 1u32..5 {
            assert!(
                reader.state.contains(i),
                "doc {i} should be in cached block"
            );
            let fields = reader.document(i).unwrap();
            assert_eq!(fields.len(), 2);
            // Verify the stored int value matches
            let idx_field = fields.iter().find(|f| f.field_number == 1).unwrap();
            assert_matches!(idx_field.value, StoredValue::Int(v) if v == i as i32);
        }
    }

    #[test]
    fn test_block_state_invalidated_on_new_block() {
        // Write docs with large stored fields to force multiple chunks.
        // Chunk size is 81920 bytes, so ~40KB per doc should force 2+ blocks.
        let big_string: String = "x".repeat(45_000);

        let mut docs = Vec::new();
        for i in 0..4 {
            let doc = DocumentBuilder::new()
                .add_field(stored("data").string(big_string.clone()))
                .add_field(stored("idx").int(i))
                .build();
            docs.push(doc);
        }

        let config = IndexWriterConfig::default().num_threads(1);
        let directory: SharedDirectory = MemoryDirectory::create();
        let writer = IndexWriter::new(config, Arc::clone(&directory));
        for doc in docs {
            writer.add_document(doc).unwrap();
        }
        writer.commit().unwrap();

        let files = directory.list_all().unwrap();
        let segments_file = files.iter().find(|f| f.starts_with("segments_")).unwrap();
        let infos = segment_infos::read(&*directory, segments_file).unwrap();
        let seg = &infos.segments[0];

        let mut reader = StoredFieldsReader::open(&*directory, &seg.name, "", &seg.id).unwrap();

        // Read all docs and verify values — this exercises cache invalidation
        // as docs should span multiple blocks
        for i in 0u32..4 {
            let fields = reader.document(i).unwrap();
            let idx_field = fields.iter().find(|f| f.field_number == 1).unwrap();
            assert_matches!(idx_field.value, StoredValue::Int(v) if v == i as i32);
        }
    }
}
