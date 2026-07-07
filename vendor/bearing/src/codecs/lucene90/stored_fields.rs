// SPDX-License-Identifier: Apache-2.0
//! Stored fields writer with LZ4 block compression.

use std::cmp;
use std::fmt;
use std::io;
use std::io::Write;

use log::debug;

use crate::codecs::packed_writers::DirectMonotonicWriter;
use crate::codecs::{codec_footers, codec_headers};
use crate::document::StoredValue;
use crate::encoding::lz4;
use crate::encoding::write_encoding::WriteEncoding;
use crate::encoding::zigzag;
use crate::index::index_file_names;
use crate::store::{DataOutput, Directory, IndexOutput, VecOutput};

// File extensions
pub(crate) const FIELDS_EXTENSION: &str = "fdt";
pub(crate) const INDEX_EXTENSION: &str = "fdx";
pub(crate) const META_EXTENSION: &str = "fdm";

// Codec names and versions
pub(crate) const FORMAT_NAME: &str = "Lucene90StoredFieldsFastData";
pub(crate) const INDEX_CODEC_NAME_IDX: &str = "Lucene90FieldsIndexIdx";
pub(crate) const INDEX_CODEC_NAME_META: &str = "Lucene90FieldsIndexMeta";
pub(crate) const FDT_VERSION: i32 = 1;
pub(crate) const FDX_VERSION: i32 = 0;
pub(crate) const FDM_VERSION: i32 = 1;

// Compression parameters (BEST_SPEED mode)
pub(crate) const CHUNK_SIZE: i32 = 10 * 8 * 1024; // 81920 bytes
pub(crate) const BLOCK_SHIFT: u32 = 10;
const MAX_DOCS_PER_CHUNK: i32 = 1024;
const NUM_SUB_BLOCKS: usize = 10;
const DICT_SIZE_FACTOR: usize = 2;
const LZ4_MAX_DISTANCE: usize = 1 << 16;

// Type codes for stored field values
pub(crate) const TYPE_STRING: u64 = 0x00;
pub(crate) const TYPE_BYTE_ARR: u64 = 0x01;
pub(crate) const TYPE_NUMERIC_INT: u64 = 0x02;
pub(crate) const TYPE_NUMERIC_FLOAT: u64 = 0x03;
pub(crate) const TYPE_NUMERIC_LONG: u64 = 0x04;
pub(crate) const TYPE_NUMERIC_DOUBLE: u64 = 0x05;
pub(crate) const TYPE_BITS: u32 = 3;

// Timestamp compression constants for writeTLong
pub(crate) const SECOND: i64 = 1000;
pub(crate) const HOUR: i64 = 60 * 60 * SECOND;
pub(crate) const DAY: i64 = 24 * HOUR;
pub(crate) const SECOND_ENCODING: u8 = 0x40;
pub(crate) const HOUR_ENCODING: u8 = 0x80;
pub(crate) const DAY_ENCODING: u8 = 0xC0;

// ============================================================
// FieldsIndexWriter — accumulates per-chunk index entries
// ============================================================

/// Accumulates per-chunk doc counts and file pointers during indexing,
/// then writes the fields index (.fdx) and metadata (.fdm) at finish time.
struct FieldsIndexWriter {
    total_docs: i32,
    total_chunks: i32,
    previous_fp: i64,
    doc_counts: Vec<i32>,
    start_pointers: Vec<i64>,
}

impl FieldsIndexWriter {
    fn new() -> Self {
        Self {
            total_docs: 0,
            total_chunks: 0,
            previous_fp: 0,
            doc_counts: Vec::new(),
            start_pointers: Vec::new(),
        }
    }

    fn write_index(&mut self, num_docs: i32, start_pointer: i64) {
        assert!(start_pointer >= self.previous_fp);
        self.doc_counts.push(num_docs);
        self.start_pointers.push(start_pointer);
        self.previous_fp = start_pointer;
        self.total_docs += num_docs;
        self.total_chunks += 1;
    }

    fn finish(
        &self,
        num_docs: i32,
        max_pointer: i64,
        fdx: &mut dyn IndexOutput,
        fdm: &mut dyn IndexOutput,
    ) -> io::Result<()> {
        if num_docs != self.total_docs {
            return Err(io::Error::other(format!(
                "Expected {} docs, but got {}",
                num_docs, self.total_docs
            )));
        }

        fdm.write_le_int(num_docs)?;
        fdm.write_le_int(BLOCK_SHIFT as i32)?;
        fdm.write_le_int(self.total_chunks + 1)?;

        fdm.write_le_long(fdx.file_pointer() as i64)?;

        let mut docs_writer = DirectMonotonicWriter::new(BLOCK_SHIFT);
        let mut doc = 0i64;
        docs_writer.add(doc);
        for &count in &self.doc_counts {
            doc += count as i64;
            docs_writer.add(doc);
        }
        docs_writer.finish(fdm, fdx)?;

        fdm.write_le_long(fdx.file_pointer() as i64)?;

        let mut fp_writer = DirectMonotonicWriter::new(BLOCK_SHIFT);
        for &pointer in &self.start_pointers {
            fp_writer.add(pointer);
        }
        fp_writer.add(max_pointer);
        fp_writer.finish(fdm, fdx)?;

        fdm.write_le_long(fdx.file_pointer() as i64)?;

        Ok(())
    }
}

// ============================================================
// StoredFieldsWriter — streaming trait
// ============================================================

/// Trait for a codec-level stored fields writer that receives streaming data.
///
/// The lifecycle is: `start_document` -> N x `write_field` -> `finish_document`,
/// repeated for each document. Then `finish` and `close`.
pub(crate) trait StoredFieldsWriter {
    /// Called before writing the stored fields of a document.
    fn start_document(&mut self) -> io::Result<()>;

    /// Writes a single stored field value.
    fn write_field(&mut self, field_number: u32, value: &StoredValue) -> io::Result<()>;

    /// Called when a document and all its fields have been added.
    fn finish_document(&mut self) -> io::Result<()>;

    /// Called after all documents have been written.
    fn finish(&mut self, num_docs: i32) -> io::Result<()>;
}

// ============================================================
// Lucene90StoredFieldsWriter — streaming stored fields writer
// ============================================================

/// Streaming stored fields writer for the Lucene90 compressing format.
///
/// Implements [`StoredFieldsWriter`] for incremental document processing.
/// Fields are buffered per-chunk and compressed with LZ4 when the chunk is full.
pub(crate) struct Lucene90StoredFieldsWriter {
    fields_stream: Option<Box<dyn IndexOutput>>,
    fdx: Option<Box<dyn IndexOutput>>,
    fdm: Option<Box<dyn IndexOutput>>,

    index_writer: FieldsIndexWriter,

    buffered_docs: Vec<u8>,
    num_stored_fields: Vec<i32>,
    end_offsets: Vec<i32>,
    num_buffered_docs: i32,
    num_stored_fields_in_doc: i32,

    doc_base: i32,
    num_chunks: i64,
    num_dirty_chunks: i64,
    num_dirty_docs: i64,

    chunk_size: i32,
    max_docs_per_chunk: i32,

    /// Reusable LZ4 hash table for compression.
    lz4_ht: lz4::FastHashTable,
}

impl fmt::Debug for Lucene90StoredFieldsWriter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Lucene90StoredFieldsWriter")
            .field("doc_base", &self.doc_base)
            .field("num_buffered_docs", &self.num_buffered_docs)
            .field("buffered_bytes", &self.buffered_docs.len())
            .field("num_chunks", &self.num_chunks)
            .finish()
    }
}

impl Lucene90StoredFieldsWriter {
    /// Creates a new streaming stored fields writer.
    ///
    /// Opens the .fdt, .fdx, and .fdm output files and writes their headers.
    pub(crate) fn new(
        directory: &dyn Directory,
        segment_name: &str,
        segment_suffix: &str,
        segment_id: &[u8; 16],
    ) -> io::Result<Self> {
        let fdt_name =
            index_file_names::segment_file_name(segment_name, segment_suffix, FIELDS_EXTENSION);
        let fdx_name =
            index_file_names::segment_file_name(segment_name, segment_suffix, INDEX_EXTENSION);
        let fdm_name =
            index_file_names::segment_file_name(segment_name, segment_suffix, META_EXTENSION);

        debug!(
            "Lucene90StoredFieldsWriter: opening {fdt_name}, {fdx_name}, {fdm_name} \
             for segment={segment_name:?}"
        );

        let mut fields_stream = directory.create_output(&fdt_name)?;
        let mut fdx = directory.create_output(&fdx_name)?;
        let mut fdm = directory.create_output(&fdm_name)?;

        codec_headers::write_index_header(
            &mut *fields_stream,
            FORMAT_NAME,
            FDT_VERSION,
            segment_id,
            segment_suffix,
        )?;
        codec_headers::write_index_header(
            &mut *fdx,
            INDEX_CODEC_NAME_IDX,
            FDX_VERSION,
            segment_id,
            segment_suffix,
        )?;
        codec_headers::write_index_header(
            &mut *fdm,
            INDEX_CODEC_NAME_META,
            FDM_VERSION,
            segment_id,
            segment_suffix,
        )?;

        fdm.write_vint(CHUNK_SIZE)?;

        Ok(Self {
            fields_stream: Some(fields_stream),
            fdx: Some(fdx),
            fdm: Some(fdm),
            index_writer: FieldsIndexWriter::new(),
            buffered_docs: Vec::new(),
            num_stored_fields: Vec::with_capacity(16),
            end_offsets: Vec::with_capacity(16),
            num_buffered_docs: 0,
            num_stored_fields_in_doc: 0,
            doc_base: 0,
            num_chunks: 0,
            num_dirty_chunks: 0,
            num_dirty_docs: 0,
            chunk_size: CHUNK_SIZE,
            max_docs_per_chunk: MAX_DOCS_PER_CHUNK,
            lz4_ht: lz4::FastHashTable::new(),
        })
    }

    /// Returns the names of the three output files (.fdt, .fdx, .fdm).
    pub(crate) fn file_names(segment_name: &str, segment_suffix: &str) -> Vec<String> {
        vec![
            index_file_names::segment_file_name(segment_name, segment_suffix, FIELDS_EXTENSION),
            index_file_names::segment_file_name(segment_name, segment_suffix, INDEX_EXTENSION),
            index_file_names::segment_file_name(segment_name, segment_suffix, META_EXTENSION),
        ]
    }

    fn trigger_flush(&self) -> bool {
        self.buffered_docs.len() >= self.chunk_size as usize
            || self.num_buffered_docs >= self.max_docs_per_chunk
    }

    fn flush_chunk(&mut self, force: bool) -> io::Result<()> {
        assert!(self.trigger_flush() != force);
        self.num_chunks += 1;
        if force {
            self.num_dirty_chunks += 1;
            self.num_dirty_docs += self.num_buffered_docs as i64;
        }
        self.index_writer.write_index(
            self.num_buffered_docs,
            self.fields_stream.as_ref().unwrap().file_pointer() as i64,
        );

        let num = self.num_buffered_docs as usize;
        for i in (1..num).rev() {
            self.end_offsets[i] -= self.end_offsets[i - 1];
            assert!(self.end_offsets[i] >= 0);
        }

        let sliced = self.buffered_docs.len() >= 2 * self.chunk_size as usize;
        let dirty_chunk = force;
        self.write_header(sliced, dirty_chunk)?;

        if sliced {
            let capacity = self.buffered_docs.len();
            let mut compressed = 0;
            while compressed < capacity {
                let l = cmp::min(self.chunk_size as usize, capacity - compressed);
                compress_lz4_preset_dict_reuse(
                    &self.buffered_docs[compressed..compressed + l],
                    &mut **self.fields_stream.as_mut().unwrap(),
                    &mut self.lz4_ht,
                )?;
                compressed += self.chunk_size as usize;
            }
        } else {
            compress_lz4_preset_dict_reuse(
                &self.buffered_docs,
                &mut **self.fields_stream.as_mut().unwrap(),
                &mut self.lz4_ht,
            )?;
        }

        self.doc_base += self.num_buffered_docs;
        self.num_buffered_docs = 0;
        self.buffered_docs.clear();
        self.num_stored_fields.clear();
        self.end_offsets.clear();

        Ok(())
    }

    fn write_header(&mut self, sliced: bool, dirty_chunk: bool) -> io::Result<()> {
        let sliced_bit: i32 = if sliced { 1 } else { 0 };
        let dirty_bit: i32 = if dirty_chunk { 2 } else { 0 };

        self.fields_stream
            .as_mut()
            .unwrap()
            .write_vint(self.doc_base)?;
        self.fields_stream
            .as_mut()
            .unwrap()
            .write_vint((self.num_buffered_docs << 2) | dirty_bit | sliced_bit)?;

        let num = self.num_buffered_docs as usize;
        save_ints(
            &self.num_stored_fields,
            num,
            &mut **self.fields_stream.as_mut().unwrap(),
        )?;

        save_ints(
            &self.end_offsets,
            num,
            &mut **self.fields_stream.as_mut().unwrap(),
        )?;

        Ok(())
    }

    fn serialize_field(&mut self, field_number: u32, value: &StoredValue) -> io::Result<()> {
        let type_code = match value {
            StoredValue::String(_) => TYPE_STRING,
            StoredValue::Int(_) => TYPE_NUMERIC_INT,
            StoredValue::Long(_) => TYPE_NUMERIC_LONG,
            StoredValue::Float(_) => TYPE_NUMERIC_FLOAT,
            StoredValue::Double(_) => TYPE_NUMERIC_DOUBLE,
            StoredValue::Bytes(_) => TYPE_BYTE_ARR,
        };
        let info_and_bits = ((field_number as u64) << TYPE_BITS) | type_code;
        VecOutput(&mut self.buffered_docs).write_vlong(info_and_bits as i64)?;

        match value {
            StoredValue::String(s) => {
                VecOutput(&mut self.buffered_docs).write_string(s)?;
            }
            StoredValue::Int(i) => {
                VecOutput(&mut self.buffered_docs).write_zint(*i)?;
            }
            StoredValue::Long(l) => {
                write_tlong(&mut VecOutput(&mut self.buffered_docs), *l)?;
            }
            StoredValue::Float(f) => {
                write_zfloat(&mut VecOutput(&mut self.buffered_docs), *f)?;
            }
            StoredValue::Double(d) => {
                write_zdouble(&mut VecOutput(&mut self.buffered_docs), *d)?;
            }
            StoredValue::Bytes(b) => {
                VecOutput(&mut self.buffered_docs).write_vint(b.len() as i32)?;
                VecOutput(&mut self.buffered_docs).write_all(b)?;
            }
        }
        Ok(())
    }
}

impl StoredFieldsWriter for Lucene90StoredFieldsWriter {
    fn start_document(&mut self) -> io::Result<()> {
        Ok(())
    }

    fn write_field(&mut self, field_number: u32, value: &StoredValue) -> io::Result<()> {
        self.num_stored_fields_in_doc += 1;
        self.serialize_field(field_number, value)
    }

    fn finish_document(&mut self) -> io::Result<()> {
        self.num_stored_fields.push(self.num_stored_fields_in_doc);
        self.num_stored_fields_in_doc = 0;
        self.end_offsets.push(self.buffered_docs.len() as i32);
        self.num_buffered_docs += 1;
        if self.trigger_flush() {
            self.flush_chunk(false)?;
        }
        Ok(())
    }

    fn finish(&mut self, num_docs: i32) -> io::Result<()> {
        if self.num_buffered_docs > 0 {
            self.flush_chunk(true)?;
        } else {
            assert!(self.buffered_docs.is_empty());
        }
        if self.doc_base != num_docs {
            return Err(io::Error::other(format!(
                "Wrote {} docs, finish called with num_docs={}",
                self.doc_base, num_docs
            )));
        }

        let max_pointer = self.fields_stream.as_ref().unwrap().file_pointer() as i64;

        self.index_writer.finish(
            num_docs,
            max_pointer,
            &mut **self.fdx.as_mut().unwrap(),
            &mut **self.fdm.as_mut().unwrap(),
        )?;

        codec_footers::write_footer(&mut **self.fdx.as_mut().unwrap())?;

        self.fdm.as_mut().unwrap().write_le_long(max_pointer)?;

        debug!(
            "Lucene90StoredFieldsWriter: num_chunks={}, num_dirty_chunks={}, \
             num_dirty_docs={}",
            self.num_chunks, self.num_dirty_chunks, self.num_dirty_docs
        );
        self.fdm.as_mut().unwrap().write_vlong(self.num_chunks)?;
        self.fdm
            .as_mut()
            .unwrap()
            .write_vlong(self.num_dirty_chunks)?;
        self.fdm
            .as_mut()
            .unwrap()
            .write_vlong(self.num_dirty_docs)?;

        codec_footers::write_footer(&mut **self.fdm.as_mut().unwrap())?;
        codec_footers::write_footer(&mut **self.fields_stream.as_mut().unwrap())?;

        assert!(self.buffered_docs.is_empty());

        Ok(())
    }
}

// ============================================================
// LZ4 with preset dictionary compression
// ============================================================

/// Compress data using LZ4 with preset dictionary format, reusing an existing hash table.
fn compress_lz4_preset_dict_reuse(
    data: &[u8],
    mut out: &mut dyn DataOutput,
    lz4_ht: &mut lz4::FastHashTable,
) -> io::Result<()> {
    let len = data.len();
    let dict_length = cmp::min(LZ4_MAX_DISTANCE, len / (NUM_SUB_BLOCKS * DICT_SIZE_FACTOR));
    let block_length = if len <= dict_length {
        0
    } else {
        (len - dict_length).div_ceil(NUM_SUB_BLOCKS)
    };

    out.write_vint(dict_length as i32)?;
    out.write_vint(block_length as i32)?;

    let mut compressed_parts: Vec<Vec<u8>> = Vec::new();

    let dict_compressed = lz4::compress_reuse(&data[..dict_length], lz4_ht);
    out.write_vint(dict_compressed.len() as i32)?;
    compressed_parts.push(dict_compressed);

    if block_length > 0 {
        let mut buffer = Vec::with_capacity(dict_length + block_length);
        buffer.extend_from_slice(&data[..dict_length]);

        let mut start = dict_length;
        while start < len {
            let l = cmp::min(block_length, len - start);
            buffer.truncate(dict_length);
            buffer.extend_from_slice(&data[start..start + l]);

            let block_compressed =
                lz4::compress_with_dictionary_reuse(&buffer, dict_length, lz4_ht);
            out.write_vint(block_compressed.len() as i32)?;
            compressed_parts.push(block_compressed);

            start += block_length;
        }
    }

    for part in &compressed_parts {
        out.write_all(part)?;
    }

    Ok(())
}

// ============================================================
// StoredFieldsInts encoding
// ============================================================

/// Writes integers using StoredFieldsInts encoding.
/// For 1 value: writes a single VInt.
/// For multiple values: uses StoredFieldsInts.writeInts format.
fn save_ints(values: &[i32], count: usize, mut out: &mut dyn DataOutput) -> io::Result<()> {
    if count == 1 {
        out.write_vint(values[0])?;
    } else {
        write_stored_fields_ints(values, count, out)?;
    }
    Ok(())
}

fn write_stored_fields_ints(
    values: &[i32],
    count: usize,
    mut out: &mut dyn DataOutput,
) -> io::Result<()> {
    // Check if all values are equal
    let all_equal = values[1..count].iter().all(|&v| v == values[0]);

    if all_equal {
        out.write_byte(0)?;
        out.write_vint(values[0])?;
    } else {
        let max = values[..count].iter().map(|&v| v as u32).max().unwrap_or(0);

        if max <= 0xFF {
            out.write_byte(8)?;
            write_ints_8(values, count, out)?;
        } else if max <= 0xFFFF {
            out.write_byte(16)?;
            write_ints_16(values, count, out)?;
        } else {
            out.write_byte(32)?;
            write_ints_32(values, count, out)?;
        }
    }
    Ok(())
}

/// Write 8-bit packed integers in 128-element blocks.
fn write_ints_8(values: &[i32], count: usize, out: &mut dyn DataOutput) -> io::Result<()> {
    let block_size = 128;
    let mut k = 0;
    while k + block_size <= count {
        for i in 0..16 {
            let l = ((values[k + i] as u64) << 56)
                | ((values[k + 16 + i] as u64) << 48)
                | ((values[k + 32 + i] as u64) << 40)
                | ((values[k + 48 + i] as u64) << 32)
                | ((values[k + 64 + i] as u64) << 24)
                | ((values[k + 80 + i] as u64) << 16)
                | ((values[k + 96 + i] as u64) << 8)
                | (values[k + 112 + i] as u64);
            out.write_le_long(l as i64)?;
        }
        k += block_size;
    }
    // Remainder
    while k < count {
        out.write_byte(values[k] as u8)?;
        k += 1;
    }
    Ok(())
}

/// Write 16-bit packed integers in 128-element blocks.
fn write_ints_16(values: &[i32], count: usize, out: &mut dyn DataOutput) -> io::Result<()> {
    let block_size = 128;
    let mut k = 0;
    while k + block_size <= count {
        for i in 0..32 {
            let l = ((values[k + i] as u64) << 48)
                | ((values[k + 32 + i] as u64) << 32)
                | ((values[k + 64 + i] as u64) << 16)
                | (values[k + 96 + i] as u64);
            out.write_le_long(l as i64)?;
        }
        k += block_size;
    }
    while k < count {
        out.write_le_short(values[k] as i16)?;
        k += 1;
    }
    Ok(())
}

/// Write 32-bit packed integers in 128-element blocks.
fn write_ints_32(values: &[i32], count: usize, out: &mut dyn DataOutput) -> io::Result<()> {
    let block_size = 128;
    let mut k = 0;
    while k + block_size <= count {
        for i in 0..64 {
            let l = ((values[k + i] as u64) << 32) | (values[k + 64 + i] as u64);
            out.write_le_long(l as i64)?;
        }
        k += block_size;
    }
    while k < count {
        out.write_le_int(values[k])?;
        k += 1;
    }
    Ok(())
}

// ============================================================
// Timestamp-aware long encoding (writeTLong)
// ============================================================

/// Writes a long in a variable-length format optimized for timestamps.
fn write_tlong(mut out: &mut dyn DataOutput, l: i64) -> io::Result<()> {
    let mut val = l;
    let header_base: u8;

    if val % SECOND != 0 {
        header_base = 0;
    } else if val % DAY == 0 {
        header_base = DAY_ENCODING;
        val /= DAY;
    } else if val % HOUR == 0 {
        header_base = HOUR_ENCODING;
        val /= HOUR;
    } else {
        header_base = SECOND_ENCODING;
        val /= SECOND;
    }

    let zig_zag = zigzag::encode_i64(val);
    let mut header = header_base | ((zig_zag as u8) & 0x1F);
    let upper_bits = ((zig_zag as u64) >> 5) as i64;

    if upper_bits != 0 {
        header |= 0x20;
    }
    out.write_byte(header)?;
    if upper_bits != 0 {
        out.write_vlong(upper_bits)?;
    }

    Ok(())
}

// ============================================================
// Float/double compression (writeZFloat / writeZDouble)
// ============================================================

/// Negative zero float bits, used to avoid compressing -0.0f as a small integer.
const NEGATIVE_ZERO_FLOAT: i32 = f32::to_bits(-0.0_f32) as i32; // 0x80000000
/// Negative zero double bits, used to avoid compressing -0.0 as a small integer.
const NEGATIVE_ZERO_DOUBLE: i64 = f64::to_bits(-0.0_f64) as i64; // 0x8000000000000000

/// Writes a float in a variable-length format. Writes between 1 and 5 bytes.
fn write_zfloat(out: &mut dyn DataOutput, f: f32) -> io::Result<()> {
    let int_val = f as i32;
    let float_bits = f32::to_bits(f) as i32;

    if f == int_val as f32 && (-1..=0x7D).contains(&int_val) && float_bits != NEGATIVE_ZERO_FLOAT {
        // Small integer value [-1..125]: single byte
        out.write_byte((0x80 | (1 + int_val)) as u8)?;
    } else if ((float_bits as u32) >> 31) == 0 {
        // Other positive floats: 4 bytes
        out.write_byte((float_bits >> 24) as u8)?;
        out.write_le_short(((float_bits as u32) >> 8) as i16)?;
        out.write_byte(float_bits as u8)?;
    } else {
        // Other negative floats: 5 bytes
        out.write_byte(0xFF)?;
        out.write_le_int(float_bits)?;
    }
    Ok(())
}

/// Writes a double in a variable-length format. Writes between 1 and 9 bytes.
fn write_zdouble(out: &mut dyn DataOutput, d: f64) -> io::Result<()> {
    let int_val = d as i32;
    let double_bits = f64::to_bits(d) as i64;

    if d == int_val as f64 && (-1..=0x7C).contains(&int_val) && double_bits != NEGATIVE_ZERO_DOUBLE
    {
        // Small integer value [-1..124]: single byte
        out.write_byte((0x80 | (int_val + 1)) as u8)?;
    } else if d == (d as f32) as f64 {
        // d has an accurate float representation: 5 bytes
        out.write_byte(0xFE)?;
        out.write_le_int(f32::to_bits(d as f32) as i32)?;
    } else if ((double_bits as u64) >> 63) == 0 {
        // Other positive doubles: 8 bytes
        out.write_byte((double_bits >> 56) as u8)?;
        out.write_le_int(((double_bits as u64) >> 24) as i32)?;
        out.write_le_short(((double_bits as u64) >> 8) as i16)?;
        out.write_byte(double_bits as u8)?;
    } else {
        // Other negative doubles: 9 bytes
        out.write_byte(0xFF)?;
        out.write_le_long(double_bits)?;
    }
    Ok(())
}

// ============================================================
// Test-only exports for round-trip testing from stored_fields_reader
// ============================================================

#[cfg(test)]
pub(crate) fn write_zfloat_for_test(out: &mut dyn DataOutput, f: f32) -> io::Result<()> {
    write_zfloat(out, f)
}

#[cfg(test)]
pub(crate) fn write_zdouble_for_test(out: &mut dyn DataOutput, d: f64) -> io::Result<()> {
    write_zdouble(out, d)
}

#[cfg(test)]
pub(crate) fn write_tlong_for_test(out: &mut dyn DataOutput, l: i64) -> io::Result<()> {
    write_tlong(out, l)
}

#[cfg(test)]
pub(crate) fn save_ints_for_test(
    values: &[i32],
    count: usize,
    out: &mut dyn DataOutput,
) -> io::Result<()> {
    save_ints(values, count, out)
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use assertables::*;

    use super::*;
    use crate::store::memory::MemoryIndexOutput;
    use crate::store::{MemoryDirectory, SharedDirectory};

    fn test_directory() -> SharedDirectory {
        MemoryDirectory::create()
    }

    /// Helper: write stored fields using the streaming writer.
    fn write_with_streaming(
        dir: &dyn Directory,
        segment_name: &str,
        segment_suffix: &str,
        segment_id: &[u8; 16],
        docs: &[Vec<(u32, StoredValue)>],
    ) -> Vec<String> {
        let mut writer =
            Lucene90StoredFieldsWriter::new(dir, segment_name, segment_suffix, segment_id).unwrap();
        for fields in docs {
            writer.start_document().unwrap();
            for (field_number, value) in fields {
                writer.write_field(*field_number, value).unwrap();
            }
            writer.finish_document().unwrap();
        }
        let num_docs = docs.len() as i32;
        writer.finish(num_docs).unwrap();
        Lucene90StoredFieldsWriter::file_names(segment_name, segment_suffix)
    }

    // -- Streaming writer tests --

    #[test]
    fn streaming_write_basic() {
        let docs = vec![
            vec![(0, StoredValue::String("path/doc1.txt".to_string()))],
            vec![(0, StoredValue::String("path/doc2.txt".to_string()))],
            vec![(0, StoredValue::String("path/doc3.txt".to_string()))],
        ];

        let segment_id = [0u8; 16];
        let dir = test_directory();
        let names = write_with_streaming(&dir, "_0", "", &segment_id, &docs);
        assert_len_eq_x!(&names, 3);

        assert_eq!(names[0], "_0.fdt");
        assert_eq!(names[1], "_0.fdx");
        assert_eq!(names[2], "_0.fdm");

        let fdt_data = dir.read_file(&names[0]).unwrap();
        let fdx_data = dir.read_file(&names[1]).unwrap();
        let fdm_data = dir.read_file(&names[2]).unwrap();

        assert_not_empty!(fdt_data);
        assert_not_empty!(fdx_data);
        assert_not_empty!(fdm_data);

        // .fdt header magic
        assert_eq!(&fdt_data[0..4], &[0x3f, 0xd7, 0x6c, 0x17]);
        // .fdt footer magic
        let footer_start = fdt_data.len() - 16;
        assert_eq!(
            &fdt_data[footer_start..footer_start + 4],
            &[0xc0, 0x28, 0x93, 0xe8]
        );
        // .fdx and .fdm header magic
        assert_eq!(&fdx_data[0..4], &[0x3f, 0xd7, 0x6c, 0x17]);
        assert_eq!(&fdm_data[0..4], &[0x3f, 0xd7, 0x6c, 0x17]);
    }

    #[test]
    fn streaming_write_empty() {
        let segment_id = [0u8; 16];
        let dir = test_directory();
        let names = write_with_streaming(&dir, "_0", "", &segment_id, &[]);
        assert_len_eq_x!(&names, 3);

        for name in &names {
            assert_not_empty!(dir.read_file(name).unwrap());
        }
    }

    #[test]
    fn streaming_write_with_suffix() {
        let docs = vec![vec![(0, StoredValue::String("hello".to_string()))]];

        let segment_id = [1u8; 16];
        let dir = test_directory();
        let names = write_with_streaming(&dir, "_0", "0", &segment_id, &docs);

        assert_eq!(names[0], "_0_0.fdt");
        assert_eq!(names[1], "_0_0.fdx");
        assert_eq!(names[2], "_0_0.fdm");
    }

    #[test]
    fn streaming_write_all_value_types() {
        let docs = vec![vec![
            (0, StoredValue::String("hello".to_string())),
            (1, StoredValue::Int(42)),
            (2, StoredValue::Long(1000)),
            (3, StoredValue::Float(1.5)),
            (4, StoredValue::Double(9.87)),
            (5, StoredValue::Bytes(vec![1, 2, 3])),
        ]];

        let segment_id = [0u8; 16];
        let dir = test_directory();
        let names = write_with_streaming(&dir, "_0", "", &segment_id, &docs);
        assert_len_eq_x!(&names, 3);
        for name in &names {
            assert_not_empty!(dir.read_file(name).unwrap());
        }
    }

    /// Verifies .fdm layout matches the FieldsIndexReader read sequence.
    #[test]
    fn streaming_fdm_layout_matches_fields_index_reader() {
        let docs = vec![
            vec![(0, StoredValue::String("doc1".to_string()))],
            vec![(0, StoredValue::String("doc2".to_string()))],
            vec![(0, StoredValue::String("doc3".to_string()))],
        ];

        let segment_id = [0u8; 16];
        let dir = test_directory();
        let names = write_with_streaming(&dir, "_0", "", &segment_id, &docs);
        let fdm = dir.read_file(&names[2]).unwrap();
        let fdx = dir.read_file(&names[1]).unwrap();

        let hdr_len = codec_headers::index_header_length(INDEX_CODEC_NAME_META, "");
        let mut pos = hdr_len;

        // chunkSize (VInt)
        assert_eq!(fdm[pos] & 0x80, 0x80, "chunkSize should be multi-byte VInt");
        let mut vint_bytes = 0;
        while fdm[pos + vint_bytes] & 0x80 != 0 {
            vint_bytes += 1;
        }
        vint_bytes += 1;
        pos += vint_bytes;

        // numDocs = 3
        let num_docs = i32::from_le_bytes(fdm[pos..pos + 4].try_into().unwrap());
        assert_eq!(num_docs, 3, "numDocs mismatch");
        pos += 4;

        // blockShift = BLOCK_SHIFT
        let block_shift = i32::from_le_bytes(fdm[pos..pos + 4].try_into().unwrap());
        assert_eq!(block_shift, BLOCK_SHIFT as i32, "blockShift mismatch");
        pos += 4;

        // numChunks = totalChunks + 1 = 2
        let num_chunks = i32::from_le_bytes(fdm[pos..pos + 4].try_into().unwrap());
        assert_eq!(num_chunks, 2, "numChunks should be totalChunks+1=2");
        pos += 4;

        // docsStartPointer
        let docs_start = i64::from_le_bytes(fdm[pos..pos + 8].try_into().unwrap());
        let fdx_hdr_len = codec_headers::index_header_length(INDEX_CODEC_NAME_IDX, "") as i64;
        assert_eq!(
            docs_start, fdx_hdr_len,
            "docsStartPointer should point right after .fdx header"
        );
        pos += 8;

        // DMW metadata for docs: 1 block
        let dmw_block_bytes = 8 + 4 + 8 + 1; // min(Long) + avgInc(Int) + offset(Long) + bpv(Byte)
        pos += dmw_block_bytes;

        // docsEndPointer
        let docs_end = i64::from_le_bytes(fdm[pos..pos + 8].try_into().unwrap());
        assert_ge!(docs_end, docs_start);
        pos += 8;

        // DMW metadata for filePointers
        pos += dmw_block_bytes;

        // startPointersEndPointer
        let sp_end = i64::from_le_bytes(fdm[pos..pos + 8].try_into().unwrap());
        assert_ge!(sp_end, docs_end);
        pos += 8;

        // maxPointer
        let max_pointer = i64::from_le_bytes(fdm[pos..pos + 8].try_into().unwrap());
        assert_gt!(max_pointer, 0);
        pos += 8;

        // numChunks (VLong) = 1
        assert_eq!(fdm[pos], 1, "trailing numChunks VLong should be 1");
        pos += 1;

        // numDirtyChunks (VLong) = 1
        assert_eq!(fdm[pos], 1, "numDirtyChunks VLong should be 1");
        pos += 1;

        // numDirtyDocs (VLong) = 3
        assert_eq!(fdm[pos], 3, "numDirtyDocs VLong should be 3");
        pos += 1;

        // Remaining bytes should be the 16-byte footer
        let remaining = fdm.len() - pos;
        assert_eq!(
            remaining, 16,
            "expected 16-byte footer, got {remaining} bytes"
        );

        // .fdx data regions are consistent with pointers
        let fdx_footer_start = fdx.len() - 16;
        assert_le!(sp_end as usize, fdx_footer_start);
    }

    // -- Encoding tests --

    #[test]
    fn test_write_tlong_zero() {
        let mut buf = Vec::new();
        write_tlong(&mut VecOutput(&mut buf), 0).unwrap();
        assert_eq!(buf, [0xC0]);
    }

    #[test]
    fn test_write_tlong_day_precision() {
        let mut buf = Vec::new();
        write_tlong(&mut VecOutput(&mut buf), DAY).unwrap();
        assert_eq!(buf, [0xC2]);
    }

    #[test]
    fn test_write_tlong_hour_precision() {
        let mut buf = Vec::new();
        write_tlong(&mut VecOutput(&mut buf), HOUR).unwrap();
        assert_eq!(buf, [0x82]);
    }

    #[test]
    fn test_write_tlong_second_precision() {
        let mut buf = Vec::new();
        write_tlong(&mut VecOutput(&mut buf), 5000).unwrap();
        assert_eq!(buf, [0x4A]);
    }

    #[test]
    fn test_write_tlong_uncompressed() {
        let mut buf = Vec::new();
        write_tlong(&mut VecOutput(&mut buf), 123).unwrap();
        assert_eq!(buf[0], 0x36);
        assert_eq!(buf[1], 0x07);
    }

    #[test]
    fn test_write_tlong_negative() {
        let mut buf = Vec::new();
        write_tlong(&mut VecOutput(&mut buf), -1).unwrap();
        assert_eq!(buf, [0x01]);
    }

    #[test]
    fn test_write_tlong_millisecond_precision() {
        let mut buf = Vec::new();
        write_tlong(&mut VecOutput(&mut buf), 60_000).unwrap();
        assert_eq!(buf, [0x78, 0x03]);
    }

    #[test]
    fn test_stored_fields_ints_all_equal() {
        let mut out = MemoryIndexOutput::new("test".to_string());
        write_stored_fields_ints(&[5, 5, 5], 3, &mut out).unwrap();
        let bytes = out.bytes();
        assert_eq!(bytes[0], 0);
        assert_eq!(bytes[1], 5);
    }

    #[test]
    fn test_stored_fields_ints_8bit() {
        let mut out = MemoryIndexOutput::new("test".to_string());
        write_stored_fields_ints(&[1, 2, 3], 3, &mut out).unwrap();
        let bytes = out.bytes();
        assert_eq!(bytes[0], 8);
        assert_eq!(bytes[1], 1);
        assert_eq!(bytes[2], 2);
        assert_eq!(bytes[3], 3);
    }

    #[test]
    fn test_stored_fields_ints_16bit() {
        let values: Vec<i32> = vec![256, 1000, 65535, 300];
        let mut out = MemoryIndexOutput::new("test".to_string());
        write_stored_fields_ints(&values, 4, &mut out).unwrap();
        assert_eq!(out.bytes()[0], 16);
    }

    #[test]
    fn test_save_ints_single() {
        let mut out = MemoryIndexOutput::new("test".to_string());
        save_ints(&[42], 1, &mut out).unwrap();
        assert_eq!(out.bytes()[0], 42);
    }

    #[test]
    fn test_write_ints_16_block_and_remainder() {
        let values: Vec<i32> = (0..140).map(|i| (i % 0xFFFF) + 1).collect();
        let mut out = MemoryIndexOutput::new("test".to_string());
        write_ints_16(&values, 140, &mut out).unwrap();
        assert_eq!(out.bytes().len(), 256 + 24);
    }

    #[test]
    fn test_write_ints_32_block_and_remainder() {
        let values: Vec<i32> = (0..130).map(|i| 0x10000 + i).collect();
        let mut out = MemoryIndexOutput::new("test".to_string());
        write_ints_32(&values, 130, &mut out).unwrap();
        assert_eq!(out.bytes().len(), 512 + 8);
    }

    #[test]
    fn test_compress_lz4_preset_dict_reuse_small() {
        let data = b"hello world from stored fields";
        let mut out = MemoryIndexOutput::new("test".to_string());
        let mut ht = lz4::FastHashTable::new();
        compress_lz4_preset_dict_reuse(data, &mut out, &mut ht).unwrap();
        assert_not_empty!(out.bytes());
    }

    #[test]
    fn test_write_zfloat_small_integer() {
        let mut buf = Vec::new();
        write_zfloat(&mut VecOutput(&mut buf), 0.0).unwrap();
        assert_eq!(buf, [0x81]);

        buf.clear();
        write_zfloat(&mut VecOutput(&mut buf), 1.0).unwrap();
        assert_eq!(buf, [0x82]);

        buf.clear();
        write_zfloat(&mut VecOutput(&mut buf), -1.0).unwrap();
        assert_eq!(buf, [0x80]);

        buf.clear();
        write_zfloat(&mut VecOutput(&mut buf), 125.0).unwrap();
        assert_eq!(buf, [0xFE]);
    }

    #[test]
    fn test_write_zfloat_positive() {
        let mut buf = Vec::new();
        write_zfloat(&mut VecOutput(&mut buf), 1.5).unwrap();
        assert_len_eq_x!(&buf, 4);
    }

    #[test]
    fn test_write_zfloat_negative() {
        let mut buf = Vec::new();
        write_zfloat(&mut VecOutput(&mut buf), -1.5).unwrap();
        assert_len_eq_x!(&buf, 5);
        assert_eq!(buf[0], 0xFF);
    }

    #[test]
    fn test_write_zfloat_negative_zero() {
        let mut buf = Vec::new();
        write_zfloat(&mut VecOutput(&mut buf), -0.0).unwrap();
        assert_len_eq_x!(&buf, 5);
        assert_eq!(buf[0], 0xFF);
    }

    #[test]
    fn test_write_zdouble_small_integer() {
        let mut buf = Vec::new();
        write_zdouble(&mut VecOutput(&mut buf), 0.0).unwrap();
        assert_eq!(buf, [0x81]);

        buf.clear();
        write_zdouble(&mut VecOutput(&mut buf), 124.0).unwrap();
        assert_eq!(buf, [0xFD]);
    }

    #[test]
    fn test_write_zdouble_float_representable() {
        let mut buf = Vec::new();
        write_zdouble(&mut VecOutput(&mut buf), 1.5).unwrap();
        assert_len_eq_x!(&buf, 5);
        assert_eq!(buf[0], 0xFE);
    }

    #[test]
    fn test_write_zdouble_positive() {
        let mut buf = Vec::new();
        let val = 1.0000000000000002_f64;
        write_zdouble(&mut VecOutput(&mut buf), val).unwrap();
        assert_len_eq_x!(&buf, 8);
    }

    #[test]
    fn test_write_zdouble_negative() {
        let mut buf = Vec::new();
        let val = -1.0000000000000002_f64;
        write_zdouble(&mut VecOutput(&mut buf), val).unwrap();
        assert_len_eq_x!(&buf, 9);
        assert_eq!(buf[0], 0xFF);
    }

    #[test]
    fn test_write_zdouble_negative_zero() {
        let mut buf = Vec::new();
        write_zdouble(&mut VecOutput(&mut buf), -0.0).unwrap();
        assert_len_eq_x!(&buf, 5);
        assert_eq!(buf[0], 0xFE);
    }
}
