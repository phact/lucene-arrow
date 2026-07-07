// SPDX-License-Identifier: Apache-2.0
//! Field infos format (.fnm) writer and reader for per-field metadata.

use std::collections::HashMap;
use std::io;

use log::debug;

use crate::codecs::codec_file_handle::{CodecFileHandle, IndexFile};
use crate::codecs::{codec_footers, codec_headers};
use crate::document::{DocValuesType, IndexOptions};
use crate::encoding::write_encoding::WriteEncoding;
use crate::index::index_file_names;
use crate::index::{FieldInfo, FieldInfos, PointDimensionConfig, SegmentInfo};
use crate::store::Directory;

const CODEC_NAME: &str = "Lucene94FieldInfos";
const FORMAT_CURRENT: i32 = 2; // FORMAT_DOCVALUE_SKIPPER
const EXTENSION: &str = "fnm";

// Field bit flags
const STORE_TERMVECTOR: u8 = 0b0000_0001;
const OMIT_NORMS: u8 = 0b0000_0010;
const STORE_PAYLOADS: u8 = 0b0000_0100;
const SOFT_DELETES_FIELD: u8 = 0b0000_1000;
const PARENT_FIELD_FIELD: u8 = 0b0001_0000;

/// Per-field metadata for writing the .fnm file.
#[derive(Debug, Clone, mem_dbg::MemSize)]
pub(crate) struct FieldInfosFieldData {
    /// Field name.
    pub name: String,
    /// Field number (unique within a segment).
    pub number: u32,
    /// Whether this field stores term vectors.
    pub store_term_vectors: bool,
    /// Whether this field has norms.
    pub has_norms: bool,
    /// Index options as a raw byte (0=None, 1=Docs, 2=DocsAndFreqs, etc.).
    pub index_options: u8,
    /// Doc values type for this field.
    pub doc_values_type: DocValuesType,
    /// Number of point dimensions (0 if not a point field).
    pub point_dimension_count: u32,
    /// Number of point index dimensions.
    pub point_index_dimension_count: u32,
    /// Bytes per point dimension value.
    pub point_num_bytes: u32,
}

/// Writes the .fnm file for a segment. Returns the file name written.
pub(crate) fn write(
    directory: &dyn Directory,
    segment_name: &str,
    segment_suffix: &str,
    segment_id: &[u8; 16],
    fields: &[FieldInfosFieldData],
) -> io::Result<String> {
    let file_name = index_file_names::segment_file_name(segment_name, segment_suffix, EXTENSION);
    let mut output = directory.create_output(&file_name)?;

    codec_headers::write_index_header(
        &mut *output,
        CODEC_NAME,
        FORMAT_CURRENT,
        segment_id,
        segment_suffix,
    )?;

    output.write_vint(fields.len() as i32)?;

    for fi in fields {
        debug!(
            "field_infos: field={:?} #{}, has_norms={}, index_options={}",
            fi.name, fi.number, fi.has_norms, fi.index_options
        );

        // Field name
        output.write_string(&fi.name)?;

        // Field number
        output.write_vint(fi.number as i32)?;

        // Field bits
        let mut bits: u8 = 0;
        if fi.store_term_vectors {
            bits |= STORE_TERMVECTOR;
        }
        if !fi.has_norms {
            bits |= OMIT_NORMS;
        }
        output.write_byte(bits)?;

        // Index options
        output.write_byte(fi.index_options)?;

        // Doc values type — custom encoding, NOT enum ordinal
        output.write_byte(doc_values_byte(fi.doc_values_type))?;

        // Doc values skip index type: 0 = NONE
        output.write_byte(0)?;

        // Doc values gen: -1 (no per-gen doc values)
        output.write_le_long(-1)?;

        // Attributes: per-field format metadata
        let mut attrs = HashMap::new();
        if fi.index_options > 0 {
            attrs.insert(
                "PerFieldPostingsFormat.format".to_string(),
                "Lucene103".to_string(),
            );
            attrs.insert("PerFieldPostingsFormat.suffix".to_string(), "0".to_string());
        }
        if fi.doc_values_type != DocValuesType::None {
            attrs.insert(
                "PerFieldDocValuesFormat.format".to_string(),
                "Lucene90".to_string(),
            );
            attrs.insert(
                "PerFieldDocValuesFormat.suffix".to_string(),
                "0".to_string(),
            );
        }
        output.write_map_of_strings(&attrs)?;

        // Point dimensions
        output.write_vint(fi.point_dimension_count as i32)?;
        if fi.point_dimension_count > 0 {
            output.write_vint(fi.point_index_dimension_count as i32)?;
            output.write_vint(fi.point_num_bytes as i32)?;
        }

        // Vector dimension: 0
        output.write_vint(0)?;

        // Vector encoding: 0 = BYTE
        output.write_byte(0)?;

        // Vector similarity: 0 = EUCLIDEAN
        output.write_byte(0)?;
    }

    codec_footers::write_footer(&mut *output)?;

    Ok(file_name)
}

/// Reads a `.fnm` (field infos) file from `directory`.
///
/// Validates the codec header, reads per-field metadata, and verifies the footer checksum.
pub fn read(
    directory: &dyn Directory,
    segment_info: &SegmentInfo,
    segment_suffix: &str,
) -> io::Result<FieldInfos> {
    let handle = CodecFileHandle::open(
        directory,
        IndexFile::FieldInfos,
        &segment_info.name,
        &segment_info.id,
        segment_suffix,
    )?;
    let mut input = handle.body();

    let num_fields = input.read_vint()?;
    if num_fields < 0 {
        return Err(io::Error::other(format!(
            "invalid field count: {num_fields}"
        )));
    }

    let mut fields = Vec::with_capacity(num_fields as usize);

    for _ in 0..num_fields {
        let name = input.read_string()?;
        let number = input.read_vint()? as u32;

        // Field bits
        let bits = input.read_byte()?;
        let store_term_vector = bits & STORE_TERMVECTOR != 0;
        let omit_norms = bits & OMIT_NORMS != 0;
        let store_payloads = bits & STORE_PAYLOADS != 0;
        let soft_deletes_field = bits & SOFT_DELETES_FIELD != 0;
        let is_parent_field = bits & PARENT_FIELD_FIELD != 0;

        // Index options
        let index_options = byte_to_index_options(input.read_byte()?)?;

        // Doc values type
        let doc_values_type = byte_to_doc_values_type(input.read_byte()?)?;

        // Doc values skip index type (FORMAT >= 2): 0 = NONE, 1 = RANGE
        let dv_skip_index_type = input.read_byte()?;

        // Doc values gen
        let dv_gen = input.read_le_long()?;

        // Attributes
        let attributes = input.read_map_of_strings()?;

        // Point dimensions
        let dimension_count = input.read_vint()? as u32;
        let point_config = if dimension_count != 0 {
            let index_dimension_count = input.read_vint()? as u32;
            let num_bytes = input.read_vint()? as u32;
            PointDimensionConfig {
                dimension_count,
                index_dimension_count,
                num_bytes,
            }
        } else {
            PointDimensionConfig::default()
        };

        // Vector dimension (read and discard — not supported yet)
        let _vector_dimension = input.read_vint()?;
        // Vector encoding (byte)
        let _vector_encoding = input.read_byte()?;
        // Vector similarity (byte)
        let _vector_similarity = input.read_byte()?;

        let mut fi = FieldInfo::new(
            name,
            number,
            store_term_vector,
            omit_norms,
            index_options,
            doc_values_type,
            point_config,
        );
        fi.store_payloads = store_payloads;
        fi.soft_deletes_field = soft_deletes_field;
        fi.is_parent_field = is_parent_field;
        fi.doc_values_skip_index_type = dv_skip_index_type;
        fi.dv_gen = dv_gen;
        for (k, v) in attributes {
            fi.put_attribute(k, v);
        }

        fields.push(fi);
    }

    debug!("field_infos: read {} fields", fields.len());

    Ok(FieldInfos::new(fields))
}

/// Encodes DocValuesType to the byte used in .fnm files.
/// NOTE: This is NOT the same as the enum ordinal!
/// SORTED_SET=4, SORTED_NUMERIC=5 in .fnm (swapped vs Java enum ordinals).
fn doc_values_byte(dvt: DocValuesType) -> u8 {
    match dvt {
        DocValuesType::None => 0,
        DocValuesType::Numeric => 1,
        DocValuesType::Binary => 2,
        DocValuesType::Sorted => 3,
        DocValuesType::SortedSet => 4,
        DocValuesType::SortedNumeric => 5,
    }
}

fn byte_to_index_options(b: u8) -> io::Result<IndexOptions> {
    match b {
        0 => Ok(IndexOptions::None),
        1 => Ok(IndexOptions::Docs),
        2 => Ok(IndexOptions::DocsAndFreqs),
        3 => Ok(IndexOptions::DocsAndFreqsAndPositions),
        4 => Ok(IndexOptions::DocsAndFreqsAndPositionsAndOffsets),
        _ => Err(io::Error::other(format!("invalid index options byte: {b}"))),
    }
}

fn byte_to_doc_values_type(b: u8) -> io::Result<DocValuesType> {
    match b {
        0 => Ok(DocValuesType::None),
        1 => Ok(DocValuesType::Numeric),
        2 => Ok(DocValuesType::Binary),
        3 => Ok(DocValuesType::Sorted),
        4 => Ok(DocValuesType::SortedSet),
        5 => Ok(DocValuesType::SortedNumeric),
        _ => Err(io::Error::other(format!(
            "invalid doc values type byte: {b}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{MemoryDirectory, SharedDirectory};

    fn test_directory() -> SharedDirectory {
        MemoryDirectory::create()
    }

    fn index_options_byte(opts: IndexOptions) -> u8 {
        match opts {
            IndexOptions::None => 0,
            IndexOptions::Docs => 1,
            IndexOptions::DocsAndFreqs => 2,
            IndexOptions::DocsAndFreqsAndPositions => 3,
            IndexOptions::DocsAndFreqsAndPositionsAndOffsets => 4,
        }
    }

    const SEGMENT_NAME: &str = "_0";
    const SEGMENT_ID: [u8; 16] = [0u8; 16];

    fn stored_only(name: &str, number: u32) -> FieldInfosFieldData {
        FieldInfosFieldData {
            name: name.to_string(),
            number,
            store_term_vectors: false,
            has_norms: false,
            index_options: 0,
            doc_values_type: DocValuesType::None,
            point_dimension_count: 0,
            point_index_dimension_count: 0,
            point_num_bytes: 0,
        }
    }

    fn indexed_with_norms(name: &str, number: u32) -> FieldInfosFieldData {
        FieldInfosFieldData {
            name: name.to_string(),
            number,
            store_term_vectors: false,
            has_norms: true,
            index_options: 3, // DocsAndFreqsAndPositions
            doc_values_type: DocValuesType::None,
            point_dimension_count: 0,
            point_index_dimension_count: 0,
            point_num_bytes: 0,
        }
    }

    /// Helper to create a SegmentInfo for read() calls in roundtrip tests.
    fn make_test_segment() -> SegmentInfo {
        SegmentInfo::new(
            SEGMENT_NAME.to_string(),
            3,
            true,
            SEGMENT_ID,
            HashMap::new(),
            HashMap::new(),
        )
    }

    // --- Write-side tests ---

    #[test]
    fn write_produces_fnm_file() {
        let dir = test_directory();
        let fields = vec![stored_only("title", 0), stored_only("body", 1)];
        let name = write(&dir, SEGMENT_NAME, "", &SEGMENT_ID, &fields).unwrap();
        assert_eq!(name, "_0.fnm");

        let data = dir.read_file(&name).unwrap();
        // Header magic
        assert_eq!(&data[0..4], &[0x3f, 0xd7, 0x6c, 0x17]);
        // Footer magic
        let footer_start = data.len() - 16;
        assert_eq!(
            &data[footer_start..footer_start + 4],
            &[0xc0, 0x28, 0x93, 0xe8]
        );
    }

    #[test]
    fn write_encodes_field_count_and_names() {
        let dir = test_directory();
        let fields = vec![stored_only("title", 0), stored_only("body", 1)];
        write(&dir, SEGMENT_NAME, "", &SEGMENT_ID, &fields).unwrap();

        let data = dir.read_file("_0.fnm").unwrap();

        // Header = 4(magic) + 1+18(codec "Lucene94FieldInfos") + 4(version) + 16(id) + 1(suffix) = 44
        let offset = 44;

        // Field count = 2 (VInt)
        assert_eq!(data[offset], 2);

        // First field name: VInt length(5) + "title"
        assert_eq!(data[offset + 1], 5); // length
        assert_eq!(&data[offset + 2..offset + 7], b"title");

        // File should be substantial (2 fields with all metadata)
        assert_gt!(data.len(), 80);
    }

    #[test]
    fn stored_only_field_has_omit_norms() {
        let dir = test_directory();
        let fields = vec![stored_only("f", 0)];
        write(&dir, SEGMENT_NAME, "", &SEGMENT_ID, &fields).unwrap();

        let data = dir.read_file("_0.fnm").unwrap();

        // Header(44) + field_count(1) + name_len(1) + "f"(1) + field_number(1) = 48
        let bits_offset = 48;

        // bits byte: OMIT_NORMS = 0x02 (stored-ness is not in the bits byte)
        assert_eq!(data[bits_offset], 0b0000_0010);

        // index options byte: NONE = 0
        assert_eq!(data[bits_offset + 1], 0);

        // doc values type byte: NONE = 0
        assert_eq!(data[bits_offset + 2], 0);
    }

    #[test]
    fn indexed_field_with_norms() {
        let dir = test_directory();
        let fields = vec![indexed_with_norms("body", 0)];
        write(&dir, SEGMENT_NAME, "", &SEGMENT_ID, &fields).unwrap();

        let data = dir.read_file("_0.fnm").unwrap();

        // Header(44) + field_count(1) + name_len(1) + "body"(4) + field_number(1) = 51
        let bits_offset = 51;

        // bits byte: no OMIT_NORMS = 0x00
        assert_eq!(data[bits_offset], 0b0000_0000);

        // index options byte: 3 = DocsAndFreqsAndPositions
        assert_eq!(data[bits_offset + 1], 3);
    }

    #[test]
    fn indexed_field_has_postings_format_attributes() {
        let dir = test_directory();
        let fields = vec![indexed_with_norms("body", 0)];
        write(&dir, SEGMENT_NAME, "", &SEGMENT_ID, &fields).unwrap();

        let data = dir.read_file("_0.fnm").unwrap();
        let content = String::from_utf8_lossy(&data);

        assert!(content.contains("PerFieldPostingsFormat.format"));
        assert!(content.contains("Lucene103"));
        assert!(content.contains("PerFieldPostingsFormat.suffix"));
    }

    #[test]
    fn stored_only_field_has_no_postings_format_attributes() {
        let dir = test_directory();
        let fields = vec![stored_only("title", 0)];
        write(&dir, SEGMENT_NAME, "", &SEGMENT_ID, &fields).unwrap();

        let data = dir.read_file("_0.fnm").unwrap();
        let content = String::from_utf8_lossy(&data);

        assert!(!content.contains("PerFieldPostingsFormat"));
    }

    #[test]
    fn dv_only_field_writes_correct_type_byte() {
        let dir = test_directory();
        let fields = vec![FieldInfosFieldData {
            name: "count".to_string(),
            number: 0,
            store_term_vectors: false,
            has_norms: false,
            index_options: 0,
            doc_values_type: DocValuesType::Numeric,
            point_dimension_count: 0,
            point_index_dimension_count: 0,
            point_num_bytes: 0,
        }];
        write(&dir, SEGMENT_NAME, "", &SEGMENT_ID, &fields).unwrap();

        let data = dir.read_file("_0.fnm").unwrap();

        // Header(44) + field_count(1) + name_len(1) + "count"(5) + field_number(1) = 52
        let bits_offset = 52;

        // bits byte: OMIT_NORMS
        assert_eq!(data[bits_offset], 0b0000_0010);
        // index options: NONE
        assert_eq!(data[bits_offset + 1], 0);
        // doc values type: Numeric = 1
        assert_eq!(data[bits_offset + 2], 1);
    }

    #[test]
    fn dv_sorted_set_writes_byte_4() {
        let dir = test_directory();
        let fields = vec![FieldInfosFieldData {
            name: "tags".to_string(),
            number: 0,
            store_term_vectors: false,
            has_norms: false,
            index_options: 0,
            doc_values_type: DocValuesType::SortedSet,
            point_dimension_count: 0,
            point_index_dimension_count: 0,
            point_num_bytes: 0,
        }];
        write(&dir, SEGMENT_NAME, "", &SEGMENT_ID, &fields).unwrap();

        let data = dir.read_file("_0.fnm").unwrap();

        // Header(44) + field_count(1) + name_len(1) + "tags"(4) + field_number(1) = 51
        let bits_offset = 51;

        // doc values type: SortedSet = 4 in .fnm format
        assert_eq!(data[bits_offset + 2], 4);
    }

    #[test]
    fn dv_sorted_numeric_writes_byte_5() {
        let dir = test_directory();
        let fields = vec![FieldInfosFieldData {
            name: "vals".to_string(),
            number: 0,
            store_term_vectors: false,
            has_norms: false,
            index_options: 0,
            doc_values_type: DocValuesType::SortedNumeric,
            point_dimension_count: 0,
            point_index_dimension_count: 0,
            point_num_bytes: 0,
        }];
        write(&dir, SEGMENT_NAME, "", &SEGMENT_ID, &fields).unwrap();

        let data = dir.read_file("_0.fnm").unwrap();

        // Header(44) + field_count(1) + name_len(1) + "vals"(4) + field_number(1) = 51
        let bits_offset = 51;

        // doc values type: SortedNumeric = 5 in .fnm format
        assert_eq!(data[bits_offset + 2], 5);
    }

    #[test]
    fn dv_field_has_doc_values_format_attributes() {
        let dir = test_directory();
        let fields = vec![FieldInfosFieldData {
            name: "count".to_string(),
            number: 0,
            store_term_vectors: false,
            has_norms: false,
            index_options: 0,
            doc_values_type: DocValuesType::SortedNumeric,
            point_dimension_count: 0,
            point_index_dimension_count: 0,
            point_num_bytes: 0,
        }];
        write(&dir, SEGMENT_NAME, "", &SEGMENT_ID, &fields).unwrap();

        let data = dir.read_file("_0.fnm").unwrap();
        let content = String::from_utf8_lossy(&data);

        assert!(content.contains("PerFieldDocValuesFormat.format"));
        assert!(content.contains("Lucene90"));
        assert!(content.contains("PerFieldDocValuesFormat.suffix"));
        // No postings format for DV-only field
        assert!(!content.contains("PerFieldPostingsFormat"));
    }

    #[test]
    fn dv_type_none_has_no_dv_attributes() {
        let dir = test_directory();
        let fields = vec![stored_only("title", 0)];
        write(&dir, SEGMENT_NAME, "", &SEGMENT_ID, &fields).unwrap();

        let data = dir.read_file("_0.fnm").unwrap();
        let content = String::from_utf8_lossy(&data);

        assert!(!content.contains("PerFieldDocValuesFormat"));
    }

    #[test]
    fn test_doc_values_byte_encoding() {
        assert_eq!(doc_values_byte(DocValuesType::None), 0);
        assert_eq!(doc_values_byte(DocValuesType::Numeric), 1);
        assert_eq!(doc_values_byte(DocValuesType::Binary), 2);
        assert_eq!(doc_values_byte(DocValuesType::Sorted), 3);
        assert_eq!(doc_values_byte(DocValuesType::SortedSet), 4);
        assert_eq!(doc_values_byte(DocValuesType::SortedNumeric), 5);
    }

    #[test]
    fn test_index_options_byte_encoding() {
        assert_eq!(index_options_byte(IndexOptions::None), 0);
        assert_eq!(index_options_byte(IndexOptions::Docs), 1);
        assert_eq!(index_options_byte(IndexOptions::DocsAndFreqs), 2);
        assert_eq!(
            index_options_byte(IndexOptions::DocsAndFreqsAndPositions),
            3
        );
        assert_eq!(
            index_options_byte(IndexOptions::DocsAndFreqsAndPositionsAndOffsets),
            4
        );
    }

    // --- Read round-trip tests ---

    #[test]
    fn test_read_roundtrip_single_field() {
        let dir = test_directory();
        let si = make_test_segment();
        let fields = vec![FieldInfosFieldData {
            name: "test".to_string(),
            number: 0,
            store_term_vectors: false,
            has_norms: true,
            index_options: 3, // DocsAndFreqsAndPositions
            doc_values_type: DocValuesType::None,
            point_dimension_count: 0,
            point_index_dimension_count: 0,
            point_num_bytes: 0,
        }];
        write(&dir, SEGMENT_NAME, "", &SEGMENT_ID, &fields).unwrap();

        let read_fis = read(&dir, &si, "").unwrap();

        assert_eq!(read_fis.len(), 1);
        let f = &read_fis.iter().next().unwrap();
        assert_eq!(f.name(), "test");
        assert_eq!(f.number(), 0);
        assert_eq!(f.index_options(), IndexOptions::DocsAndFreqsAndPositions);
        assert_eq!(f.doc_values_type(), DocValuesType::None);
        assert!(!f.store_term_vector());
        assert!(!f.omit_norms());
    }

    #[test]
    fn test_read_roundtrip_multiple_fields() {
        let dir = test_directory();
        let si = make_test_segment();

        let fields = vec![
            FieldInfosFieldData {
                name: "path".to_string(),
                number: 0,
                store_term_vectors: false,
                has_norms: false,
                index_options: 1, // Docs
                doc_values_type: DocValuesType::SortedSet,
                point_dimension_count: 0,
                point_index_dimension_count: 0,
                point_num_bytes: 0,
            },
            FieldInfosFieldData {
                name: "body".to_string(),
                number: 1,
                store_term_vectors: true,
                has_norms: true,
                index_options: 3, // DocsAndFreqsAndPositions
                doc_values_type: DocValuesType::None,
                point_dimension_count: 0,
                point_index_dimension_count: 0,
                point_num_bytes: 0,
            },
            FieldInfosFieldData {
                name: "location".to_string(),
                number: 2,
                store_term_vectors: false,
                has_norms: false,
                index_options: 0, // None
                doc_values_type: DocValuesType::None,
                point_dimension_count: 2,
                point_index_dimension_count: 2,
                point_num_bytes: 4,
            },
        ];
        write(&dir, SEGMENT_NAME, "", &SEGMENT_ID, &fields).unwrap();

        let read_fis = read(&dir, &si, "").unwrap();

        assert_eq!(read_fis.len(), 3);
        let read_fields: Vec<_> = read_fis.iter().collect();

        assert_eq!(read_fields[0].name(), "path");
        assert!(read_fields[0].omit_norms());
        assert_eq!(read_fields[0].doc_values_type(), DocValuesType::SortedSet);

        assert_eq!(read_fields[1].name(), "body");
        assert!(read_fields[1].store_term_vector());
        assert_eq!(
            read_fields[1].index_options(),
            IndexOptions::DocsAndFreqsAndPositions
        );

        assert_eq!(read_fields[2].name(), "location");
        assert_eq!(read_fields[2].point_config().dimension_count, 2);
        assert_eq!(read_fields[2].point_config().num_bytes, 4);
    }

    #[test]
    fn test_read_roundtrip_with_attributes() {
        let dir = test_directory();
        let si = make_test_segment();
        // Indexed field with doc values — write() generates postings + DV attributes
        let fields = vec![FieldInfosFieldData {
            name: "test".to_string(),
            number: 0,
            store_term_vectors: false,
            has_norms: true,
            index_options: 1, // Docs
            doc_values_type: DocValuesType::Numeric,
            point_dimension_count: 0,
            point_index_dimension_count: 0,
            point_num_bytes: 0,
        }];
        write(&dir, SEGMENT_NAME, "", &SEGMENT_ID, &fields).unwrap();

        let read_fis = read(&dir, &si, "").unwrap();
        let f = read_fis.iter().next().unwrap();
        assert_eq!(
            f.get_attribute("PerFieldPostingsFormat.format").unwrap(),
            "Lucene103"
        );
        assert_eq!(
            f.get_attribute("PerFieldDocValuesFormat.format").unwrap(),
            "Lucene90"
        );
    }
}
