// SPDX-License-Identifier: Apache-2.0
//! Compound file format (.cfs/.cfe) writer that merges segment files into a single file.

use std::io;

use log::debug;

use crate::codecs::{codec_footers, codec_headers};
use crate::encoding::write_encoding::WriteEncoding;
use crate::index::index_file_names;
use crate::store::memory::MemoryIndexOutput;
use crate::store::{DataOutput, IndexOutput, SegmentFile};

const ENTRIES_EXTENSION: &str = "cfe";
const DATA_CODEC: &str = "Lucene90CompoundData";
const ENTRY_CODEC: &str = "Lucene90CompoundEntries";
const VERSION_START: i32 = 0;
const VERSION_CURRENT: i32 = VERSION_START;

/// Writes a compound file (.cfs) to the provided output and returns the entry table (.cfe)
/// as a [`SegmentFile`].
///
/// Takes a list of [`SegmentFile`]s representing all files for a segment,
/// and concatenates them into the compound data output with an entry table.
/// Files are sorted by size ascending so smaller files pack into pages together.
///
/// The `.cfs` data is streamed directly to `cfs_out`, avoiding in-memory buffering
/// of the entire compound file.
pub(crate) fn write_to(
    segment_name: &str,
    segment_id: &[u8; 16],
    files: &[SegmentFile],
    cfs_out: &mut dyn IndexOutput,
) -> io::Result<SegmentFile> {
    let entries_file = index_file_names::segment_file_name(segment_name, "", ENTRIES_EXTENSION);
    let mut entries = MemoryIndexOutput::new(entries_file);

    // Write headers
    codec_headers::write_index_header(cfs_out, DATA_CODEC, VERSION_CURRENT, segment_id, "")?;
    codec_headers::write_index_header(&mut entries, ENTRY_CODEC, VERSION_CURRENT, segment_id, "")?;

    // Write file count
    entries.write_vint(files.len() as i32)?;

    // Sort files by size ascending (smallest first)
    let mut sorted: Vec<(usize, usize)> = files
        .iter()
        .enumerate()
        .map(|(i, f)| (i, f.data.len()))
        .collect();
    sorted.sort_by_key(|&(_, len)| len);

    debug!("compound: writing {} files into .cfs/.cfe", files.len());

    for &(idx, _) in &sorted {
        let f = &files[idx];
        let filename = &f.name;
        let file_bytes = &f.data;
        let file_len = file_bytes.len();

        if file_len < codec_footers::FOOTER_LENGTH {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("file too small to contain footer: {filename}"),
            ));
        }

        // Align to 8-byte boundary
        let start_offset = cfs_out.align_file_pointer(8)?;

        // Copy all bytes except the 16-byte footer
        let body_len = file_len - codec_footers::FOOTER_LENGTH;
        cfs_out.write_all(&file_bytes[..body_len])?;

        // Extract original checksum from footer (last 8 bytes, BE long)
        let checksum = i64::from_be_bytes(file_bytes[file_len - 8..file_len].try_into().unwrap());

        // Write custom footer with original checksum (not the compound file's running CRC)
        cfs_out.write_be_int(codec_footers::FOOTER_MAGIC)?;
        cfs_out.write_be_int(0)?;
        cfs_out.write_be_long(checksum)?;

        let end_offset = cfs_out.file_pointer();
        let length = end_offset - start_offset;

        // Write entry to .cfe
        let stripped = index_file_names::strip_segment_name(filename);
        entries.write_string(stripped)?;
        entries.write_le_long(start_offset as i64)?;
        entries.write_le_long(length as i64)?;

        debug!(
            "compound: file={filename} stripped={stripped} offset={start_offset} length={length}"
        );
    }

    // Write footers
    codec_footers::write_footer(cfs_out)?;
    codec_footers::write_footer(&mut entries)?;

    Ok(entries.into_inner())
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    const DATA_EXTENSION: &str = "cfs";

    /// Writes a compound file (.cfs) and entry table (.cfe) from the given segment files,
    /// returning both as in-memory [`SegmentFile`]s.
    ///
    /// Convenience wrapper for tests only.
    fn write(
        segment_name: &str,
        segment_id: &[u8; 16],
        files: &[SegmentFile],
    ) -> io::Result<Vec<SegmentFile>> {
        let data_file = index_file_names::segment_file_name(segment_name, "", DATA_EXTENSION);
        let mut data = MemoryIndexOutput::new(data_file);
        let cfe = write_to(segment_name, segment_id, files, &mut data)?;
        Ok(vec![data.into_inner(), cfe])
    }
    use super::*;
    use crate::codecs::codec_footers::{FOOTER_LENGTH, FOOTER_MAGIC};
    use crate::codecs::codec_headers::CODEC_MAGIC;
    use crate::store::memory::MemoryIndexOutput;
    use crate::test_util::TestDataReader;

    /// Creates a fake segment file with a proper index header, body data, and footer.
    fn make_test_file(name: &str, segment_id: &[u8; 16], body: &[u8]) -> SegmentFile {
        let mut out = MemoryIndexOutput::new(name.to_string());
        codec_headers::write_index_header(&mut out, "TestCodec", 1, segment_id, "").unwrap();
        out.write_all(body).unwrap();
        codec_footers::write_footer(&mut out).unwrap();
        out.into_inner()
    }

    /// Returns the index header length for a given codec name and empty suffix.
    fn index_header_len(codec: &str) -> usize {
        codec_headers::index_header_length(codec, "")
    }

    // Ported from org.apache.lucene.codecs.lucene90.TestLucene90CompoundFormat

    #[test]
    fn test_compound_single_file() {
        let segment_id = [0xABu8; 16];
        let file = make_test_file("_0.fnm", &segment_id, b"field data here");
        let files = vec![file];

        let result = write("_0", &segment_id, &files).unwrap();
        assert_len_eq_x!(&result, 2);

        let cfs = &result[0];
        let cfe = &result[1];

        assert_eq!(cfs.name, "_0.cfs");
        assert_eq!(cfe.name, "_0.cfe");

        // .cfs should start with a valid index header
        let mut r = TestDataReader::new(&cfs.data, 0);
        assert_eq!(r.read_be_int(), CODEC_MAGIC);

        // .cfe should start with a valid index header
        let mut r = TestDataReader::new(&cfe.data, 0);
        assert_eq!(r.read_be_int(), CODEC_MAGIC);

        // .cfs should end with a valid footer
        let mut r = TestDataReader::new(&cfs.data, cfs.data.len() - FOOTER_LENGTH);
        assert_eq!(r.read_be_int(), FOOTER_MAGIC);
        assert_eq!(r.read_be_int(), 0); // algorithm

        // .cfe should end with a valid footer
        let mut r = TestDataReader::new(&cfe.data, cfe.data.len() - FOOTER_LENGTH);
        assert_eq!(r.read_be_int(), FOOTER_MAGIC);
        assert_eq!(r.read_be_int(), 0); // algorithm
    }

    #[test]
    fn test_compound_alignment() {
        let segment_id = [0x01u8; 16];
        // Create files of different sizes to test alignment
        let file1 = make_test_file("_0.fnm", &segment_id, b"abc"); // small
        let file2 = make_test_file("_0.fdx", &segment_id, b"defgh"); // slightly larger

        let files = vec![file1, file2];
        let result = write("_0", &segment_id, &files).unwrap();
        let cfe = &result[1];

        // Parse .cfe to get offsets
        let mut r = TestDataReader::new(&cfe.data, index_header_len(ENTRY_CODEC));
        let file_count = r.read_vint();
        assert_eq!(file_count, 2);

        for _ in 0..file_count {
            let _name = r.read_string();
            let offset = r.read_le_long() as u64;
            let _length = r.read_le_long();

            // Each file must start at an 8-byte aligned offset
            assert_eq!(offset % 8, 0, "file offset {offset} is not 8-byte aligned");
        }
    }

    #[test]
    fn test_compound_entry_table() {
        let segment_id = [0x42u8; 16];
        let file1 = make_test_file("_0.fnm", &segment_id, b"field info");
        let file2 = make_test_file("_0_Lucene90_0.dvd", &segment_id, b"doc values data");
        let file1_total_len = file1.data.len();
        let file2_total_len = file2.data.len();

        let files = vec![file1, file2];
        let result = write("_0", &segment_id, &files).unwrap();
        let cfe = &result[1];

        // Parse .cfe entries
        let mut r = TestDataReader::new(&cfe.data, index_header_len(ENTRY_CODEC));
        let file_count = r.read_vint();
        assert_eq!(file_count, 2);

        // Files are sorted by size — file1 ("field info" = 10 bytes body) is smaller
        // than file2 ("doc values data" = 15 bytes body)
        let name1 = r.read_string();
        let offset1 = r.read_le_long() as u64;
        let length1 = r.read_le_long() as u64;

        let name2 = r.read_string();
        let offset2 = r.read_le_long() as u64;
        let length2 = r.read_le_long() as u64;

        // Stripped filenames
        assert_eq!(name1, ".fnm");
        assert_eq!(name2, "_Lucene90_0.dvd");

        // Lengths must match original file sizes (same content is preserved)
        assert_eq!(length1 as usize, file1_total_len);
        assert_eq!(length2 as usize, file2_total_len);

        // Second file must start after first file ends
        assert_ge!(offset2, offset1 + length1);
    }

    #[test]
    fn test_compound_preserves_checksums() {
        let segment_id = [0xFFu8; 16];
        let file = make_test_file("_0.fnm", &segment_id, b"test body data");
        let original_bytes = file.data.clone();

        // Extract original checksum from the file's footer
        let orig_len = original_bytes.len();
        let original_checksum =
            i64::from_be_bytes(original_bytes[orig_len - 8..orig_len].try_into().unwrap());

        let files = vec![file];
        let result = write("_0", &segment_id, &files).unwrap();
        let cfs = &result[0];

        // Parse .cfe to find the file's location in .cfs
        let cfe = &result[1];
        let mut r = TestDataReader::new(&cfe.data, index_header_len(ENTRY_CODEC));
        let _count = r.read_vint();
        let _name = r.read_string();
        let offset = r.read_le_long() as usize;
        let length = r.read_le_long() as usize;

        // Extract the embedded file's footer checksum from within .cfs
        let embedded_end = offset + length;
        let embedded_footer_start = embedded_end - FOOTER_LENGTH;
        let mut r = TestDataReader::new(&cfs.data, embedded_footer_start);
        assert_eq!(r.read_be_int(), FOOTER_MAGIC);
        assert_eq!(r.read_be_int(), 0);
        let embedded_checksum = r.read_be_long();

        // The embedded checksum must match the original file's checksum
        assert_eq!(embedded_checksum, original_checksum);
    }

    #[test]
    fn test_compound_size_ordering() {
        let segment_id = [0x33u8; 16];
        // Create files with distinctly different body sizes.
        // Put the largest first in the input to verify sorting.
        let large = make_test_file("_0.fdt", &segment_id, &[0xAA; 100]); // largest
        let medium = make_test_file("_0.fdx", &segment_id, &[0xBB; 50]); // medium
        let small = make_test_file("_0.fnm", &segment_id, &[0xCC; 10]); // smallest

        let files = vec![large, medium, small];
        let result = write("_0", &segment_id, &files).unwrap();
        let cfe = &result[1];

        // Parse entries — should be ordered smallest to largest
        let mut r = TestDataReader::new(&cfe.data, index_header_len(ENTRY_CODEC));
        let file_count = r.read_vint();
        assert_eq!(file_count, 3);

        let mut lengths = Vec::new();
        let mut names = Vec::new();
        for _ in 0..file_count {
            let name = r.read_string();
            let _offset = r.read_le_long();
            let length = r.read_le_long();
            names.push(name);
            lengths.push(length);
        }

        // Smallest file (.fnm with 10 bytes body) should come first
        assert_eq!(names[0], ".fnm");
        assert_eq!(names[1], ".fdx");
        assert_eq!(names[2], ".fdt");

        // Lengths should be in ascending order
        assert_le!(lengths[0], lengths[1]);
        assert_le!(lengths[1], lengths[2]);
    }
}
