// SPDX-License-Identifier: Apache-2.0
//! Segment info format (.si) writer and reader for segment-level metadata.

use std::collections::{HashMap, HashSet};
use std::io;

use log::debug;

use crate::codecs::codec_file_handle::{CodecFileHandle, IndexFile};
use crate::codecs::{codec_footers, codec_headers};
use crate::encoding::write_encoding::WriteEncoding;
use crate::index::SegmentInfo;
use crate::index::index_file_names;
use crate::store::Directory;

const CODEC_NAME: &str = "Lucene90SegmentInfo";
const VERSION_CURRENT: i32 = 0;
const EXTENSION: &str = "si";

const LUCENE_MAJOR: i32 = 10;
const LUCENE_MINOR: i32 = 3;
const LUCENE_BUGFIX: i32 = 2;

const SI_YES: u8 = 1;
const SI_NO: u8 = 0xFF; // -1 as byte

/// Segment-level metadata for writing the .si file.
#[derive(Debug)]
pub(crate) struct SegmentInfoFieldData {
    /// Segment name (e.g. "_0").
    pub name: String,
    /// Number of documents in the segment.
    pub max_doc: i32,
    /// Whether this segment is stored as a compound file (.cfs/.cfe).
    pub is_compound_file: bool,
    /// 16-byte unique segment ID.
    pub id: [u8; 16],
    /// Diagnostic metadata (e.g. source=flush).
    pub diagnostics: HashMap<String, String>,
    /// Segment attributes (e.g. stored fields compression mode).
    pub attributes: HashMap<String, String>,
    /// Whether this segment has blocks.
    pub has_blocks: bool,
}

/// Writes the .si file for a segment. Returns the file name written.
pub(crate) fn write(
    directory: &dyn Directory,
    segment_info: &SegmentInfoFieldData,
    files: &[String],
) -> io::Result<String> {
    let file_name = index_file_names::segment_file_name(&segment_info.name, "", EXTENSION);
    let mut output = directory.create_output(&file_name)?;

    debug!(
        "segment_info: segment={:?}, maxDoc={}, compound={}, files={}",
        segment_info.name,
        segment_info.max_doc,
        segment_info.is_compound_file,
        files.len()
    );

    codec_headers::write_index_header(
        &mut *output,
        CODEC_NAME,
        VERSION_CURRENT,
        &segment_info.id,
        "",
    )?;

    // Lucene version (LE ints)
    output.write_le_int(LUCENE_MAJOR)?;
    output.write_le_int(LUCENE_MINOR)?;
    output.write_le_int(LUCENE_BUGFIX)?;

    // minVersion: present + same as main version
    output.write_byte(1)?;
    output.write_le_int(LUCENE_MAJOR)?;
    output.write_le_int(LUCENE_MINOR)?;
    output.write_le_int(LUCENE_BUGFIX)?;

    // maxDoc
    output.write_le_int(segment_info.max_doc)?;

    // isCompoundFile
    output.write_byte(if segment_info.is_compound_file {
        SI_YES
    } else {
        SI_NO
    })?;

    // hasBlocks
    output.write_byte(if segment_info.has_blocks {
        SI_YES
    } else {
        SI_NO
    })?;

    // diagnostics
    output.write_map_of_strings(&segment_info.diagnostics)?;

    // files
    output.write_set_of_strings(files)?;

    // attributes
    output.write_map_of_strings(&segment_info.attributes)?;

    // numSortFields = 0
    output.write_vint(0)?;

    codec_footers::write_footer(&mut *output)?;

    Ok(file_name)
}

/// Reads a `.si` (segment info) file from `directory`.
///
/// Validates the codec header, reads segment metadata, and verifies the footer checksum.
pub fn read(
    directory: &dyn Directory,
    segment_name: &str,
    segment_id: &[u8; codec_headers::ID_LENGTH],
) -> io::Result<SegmentInfo> {
    let handle = CodecFileHandle::open(
        directory,
        IndexFile::SegmentInfo,
        segment_name,
        segment_id,
        "",
    )?;
    let mut input = handle.body();

    // Lucene version (LE ints)
    let _major = input.read_le_int()?;
    let _minor = input.read_le_int()?;
    let _bugfix = input.read_le_int()?;

    // Min version
    let has_min_version = input.read_byte()?;
    match has_min_version {
        0 => {} // no min version
        1 => {
            let _min_major = input.read_le_int()?;
            let _min_minor = input.read_le_int()?;
            let _min_bugfix = input.read_le_int()?;
        }
        _ => {
            return Err(io::Error::other(format!(
                "illegal hasMinVersion value: {has_min_version}"
            )));
        }
    }

    // Max doc
    let max_doc = input.read_le_int()?;
    if max_doc < 0 {
        return Err(io::Error::other(format!("invalid docCount: {max_doc}")));
    }

    // Compound file flag
    let is_compound_file = input.read_byte()? == SI_YES;

    // Has blocks flag
    let has_blocks_byte = input.read_byte()?;
    let has_blocks = has_blocks_byte == SI_YES;

    // Diagnostics
    let diagnostics = input.read_map_of_strings()?;

    // Files
    let files_vec = input.read_set_of_strings()?;
    let files: HashSet<String> = files_vec.into_iter().collect();

    // Attributes
    let attributes = input.read_map_of_strings()?;

    // Sort fields (must be 0 — sorted segments not supported)
    let num_sort_fields = input.read_vint()?;
    if num_sort_fields != 0 {
        return Err(io::Error::other(format!(
            "index sort not supported, got {num_sort_fields} sort fields"
        )));
    }

    let mut si = SegmentInfo::new(
        segment_name.to_string(),
        max_doc,
        is_compound_file,
        *segment_id,
        diagnostics,
        attributes,
    );
    si.has_blocks = has_blocks;
    si.files = files;

    debug!(
        "segment_info: read segment={segment_name}, maxDoc={max_doc}, compound={is_compound_file}"
    );

    Ok(si)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{MemoryDirectory, SharedDirectory};

    fn test_directory() -> SharedDirectory {
        MemoryDirectory::create()
    }

    const SEGMENT_NAME: &str = "_0";
    const SEGMENT_ID: [u8; 16] = [0u8; 16];

    fn make_segment_info() -> SegmentInfoFieldData {
        let mut diagnostics = HashMap::new();
        diagnostics.insert("source".to_string(), "flush".to_string());
        SegmentInfoFieldData {
            name: SEGMENT_NAME.to_string(),
            max_doc: 3,
            is_compound_file: false,
            id: SEGMENT_ID,
            diagnostics,
            attributes: HashMap::new(),
            has_blocks: false,
        }
    }

    // --- Write-side tests ---

    #[test]
    fn write_produces_si_file() {
        let dir = test_directory();
        let si = make_segment_info();
        let files = vec!["_0.fdt".to_string(), "_0.fdx".to_string()];
        let name = write(&dir, &si, &files).unwrap();
        assert_eq!(name, "_0.si");

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
    fn write_encodes_version_and_maxdoc() {
        let dir = test_directory();
        let si = make_segment_info();
        write(&dir, &si, &[]).unwrap();

        let data = dir.read_file("_0.si").unwrap();

        // After index header: codec="Lucene90SegmentInfo"(19 chars), version=0
        // Header = 4(magic) + 1+19(codec) + 4(version) + 16(id) + 1(suffix len) = 45
        let offset = 45;

        // Lucene version: major=10 (LE int)
        assert_eq!(data[offset], 10);
        assert_eq!(data[offset + 1], 0);
        assert_eq!(data[offset + 2], 0);
        assert_eq!(data[offset + 3], 0);

        // After version(12) + hasMinVersion(1) + minVersion(12) = 25 more bytes
        let maxdoc_offset = offset + 25;

        // maxDoc=3 (LE int)
        assert_eq!(data[maxdoc_offset], 3);
        assert_eq!(data[maxdoc_offset + 1], 0);
        assert_eq!(data[maxdoc_offset + 2], 0);
        assert_eq!(data[maxdoc_offset + 3], 0);

        // isCompoundFile = NO = 0xFF
        assert_eq!(data[maxdoc_offset + 4], 0xFF);

        // hasBlocks = NO = 0xFF
        assert_eq!(data[maxdoc_offset + 5], 0xFF);
    }

    #[test]
    fn write_compound_file_flag() {
        let dir = test_directory();
        let mut si = make_segment_info();
        si.is_compound_file = true;
        write(&dir, &si, &[]).unwrap();

        let data = dir.read_file("_0.si").unwrap();

        // After header (45) + version (12) + hasMinVersion(1) + minVersion(12) + maxDoc(4) = 74
        let flag_offset = 74;

        // isCompoundFile = YES = 1
        assert_eq!(data[flag_offset], 1);

        // hasBlocks = NO = 0xFF
        assert_eq!(data[flag_offset + 1], 0xFF);
    }

    // --- Read round-trip tests ---

    #[test]
    fn test_read_roundtrip() {
        let dir = test_directory();
        let si = make_segment_info();
        let files = vec!["_0.cfs".to_string(), "_0.cfe".to_string()];
        write(&dir, &si, &files).unwrap();

        let read_si = read(&dir, SEGMENT_NAME, &SEGMENT_ID).unwrap();

        assert_eq!(read_si.name, si.name);
        assert_eq!(read_si.max_doc, si.max_doc);
        assert_eq!(read_si.is_compound_file, si.is_compound_file);
        assert_eq!(read_si.id, si.id);
        assert_eq!(read_si.diagnostics, si.diagnostics);
        assert_eq!(read_si.has_blocks, si.has_blocks);
        assert_len_eq_x!(&read_si.files, files.len());
        for f in &files {
            assert_contains!(read_si.files, f);
        }
    }

    #[test]
    fn test_read_roundtrip_compound() {
        let dir = test_directory();
        let mut si = make_segment_info();
        si.is_compound_file = true;
        let files = vec!["_0.cfs".to_string(), "_0.cfe".to_string()];
        write(&dir, &si, &files).unwrap();

        let read_si = read(&dir, SEGMENT_NAME, &SEGMENT_ID).unwrap();

        assert!(read_si.is_compound_file);
        assert_eq!(read_si.max_doc, 3);
    }

    #[test]
    fn test_read_roundtrip_non_compound() {
        let dir = test_directory();
        let si = make_segment_info();
        let files = vec!["_0.fnm".to_string(), "_0.fdt".to_string()];
        write(&dir, &si, &files).unwrap();

        let read_si = read(&dir, SEGMENT_NAME, &SEGMENT_ID).unwrap();

        assert!(!read_si.is_compound_file);
        assert_eq!(read_si.max_doc, 3);
    }

    #[test]
    fn test_read_roundtrip_with_attributes() {
        let dir = test_directory();
        let mut si = make_segment_info();
        si.attributes
            .insert("custom_key".to_string(), "custom_val".to_string());
        write(&dir, &si, &[]).unwrap();

        let read_si = read(&dir, SEGMENT_NAME, &SEGMENT_ID).unwrap();

        assert_eq!(read_si.attributes.get("custom_key").unwrap(), "custom_val");
    }

    #[test]
    fn test_read_wrong_segment_id() {
        let dir = test_directory();
        let si = make_segment_info();
        write(&dir, &si, &[]).unwrap();

        let wrong_id = [0xFFu8; 16];
        assert_err!(read(&dir, SEGMENT_NAME, &wrong_id));
    }
}
