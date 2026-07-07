// SPDX-License-Identifier: Apache-2.0
//! Segment infos reader and writer for the segments_N commit point file.

use std::collections::HashMap;
use std::io;

use log::debug;

use crate::codecs::codec_footers::{FOOTER_LENGTH, verify_checksum};
use crate::codecs::codec_headers::check_header;
use crate::codecs::{codec_footers, codec_headers};
use crate::encoding::write_encoding::WriteEncoding;
use crate::index::index_file_names;
use crate::index::segment::FlushedSegment;
use crate::store::{Directory, IndexInput};
use crate::util::string_helper;

/// Codec name for the segments_N file header.
const CODEC_NAME: &str = "segments";

/// Format version: VERSION_86 = 10 (Lucene 8.6+).
const VERSION_CURRENT: i32 = 10;

/// Lucene version 10.3.2 — written into segments_N as the index version.
const LUCENE_VERSION_MAJOR: i32 = 10;
const LUCENE_VERSION_MINOR: i32 = 3;
const LUCENE_VERSION_BUGFIX: i32 = 2;

/// The codec name written for each segment entry.
/// For the Lucene103 codec target, this is "Lucene103".
const SEGMENT_CODEC_NAME: &str = "Lucene103";

/// A raw segment entry parsed from a `segments_N` file.
///
/// Contains only what's stored in the `segments_N` file itself — no data from
/// `.si` or `.fnm` files. The caller is responsible for reading those via the
/// appropriate codec format readers.
#[derive(Debug)]
pub struct SegmentEntry {
    /// Segment name (e.g., "_0").
    pub name: String,
    /// Segment ID (16 bytes).
    pub id: [u8; codec_headers::ID_LENGTH],
    /// Codec name (e.g., "Lucene103").
    pub codec: String,
    /// Delete generation.
    pub del_gen: i64,
    /// Number of deleted documents.
    pub del_count: i32,
    /// Field infos generation.
    pub field_infos_gen: i64,
    /// Doc values generation.
    pub doc_values_gen: i64,
    /// Number of soft-deleted documents.
    pub soft_del_count: i32,
    /// Segment commit info ID (optional).
    pub sci_id: Option<[u8; codec_headers::ID_LENGTH]>,
}

/// Result of reading a `segments_N` file.
///
/// Contains only the data stored in the `segments_N` file. Per-segment metadata
/// (`.si`, `.fnm`) must be read separately by the caller using the codec name
/// from each [`SegmentEntry`].
pub struct SegmentInfosRead {
    /// The raw segment entries in this commit.
    pub segments: Box<[SegmentEntry]>,
    /// The commit generation (from the filename suffix).
    pub generation: i64,
    /// The segment infos version (monotonically increasing).
    pub version: i64,
    /// The segment name counter.
    pub counter: i64,
    /// User data written with this commit.
    pub user_data: HashMap<String, String>,
}

/// Parses the generation number from a `segments_N` filename.
///
/// Ported from `SegmentInfos.generationFromSegmentsFileName`.
///
/// - `"segments"` → generation 0
/// - `"segments_1"` → generation 1
/// - `"segments_a"` → generation 10 (base-36)
/// - `"segments_10"` → generation 36
pub fn generation_from_segments_file_name(file_name: &str) -> io::Result<i64> {
    if file_name == "segments" {
        return Ok(0);
    }
    let suffix = file_name.strip_prefix("segments_").ok_or_else(|| {
        io::Error::other(format!("fileName \"{file_name}\" is not a segments file"))
    })?;
    i64::from_str_radix(suffix, 36)
        .map_err(|e| io::Error::other(format!("invalid generation in {file_name}: {e}")))
}

/// Returns the `segments_N` filename for the most recent commit generation.
///
/// Ported from `SegmentInfos.getLastCommitSegmentsFileName`.
/// Parses each `segments_N` generation as base-36 and picks the numeric max.
pub fn get_last_commit_segments_file_name(files: &[String]) -> io::Result<String> {
    let mut max_generation: i64 = -1;
    for file in files {
        if file.starts_with("segments_") {
            let generation = generation_from_segments_file_name(file)?;
            if generation > max_generation {
                max_generation = generation;
            }
        }
    }
    if max_generation == -1 {
        return Err(io::Error::other("no segments_N file found in directory"));
    }
    let suffix = index_file_names::radix36(max_generation as u64);
    Ok(format!("segments_{suffix}"))
}

/// Reads a `segments_N` file from `directory`.
///
/// Returns only the data stored in the `segments_N` file. Does NOT read
/// per-segment `.si` or `.fnm` files — the caller should use the codec name
/// from each [`SegmentEntry`] to dispatch to the appropriate format readers.
pub fn read(directory: &dyn Directory, segment_file_name: &str) -> io::Result<SegmentInfosRead> {
    let generation = generation_from_segments_file_name(segment_file_name)?;
    let expected_suffix = index_file_names::radix36(generation as u64);

    let backing = directory.open_file(segment_file_name)?;
    verify_checksum(backing.as_bytes())?;

    let bytes = backing.as_bytes();
    let prefix_len = bytes.len() - FOOTER_LENGTH;
    let mut input = IndexInput::new(segment_file_name, &bytes[..prefix_len]);

    // The segments_N file has a random ID we don't know ahead of time,
    // so use check_header (not check_index_header) then read the ID and suffix manually.
    check_header(&mut input, CODEC_NAME, VERSION_CURRENT, VERSION_CURRENT)?;

    // Read segment infos ID (16 bytes) — we discover it here
    let mut _id = [0u8; codec_headers::ID_LENGTH];
    input.read_bytes(&mut _id)?;

    // Read and validate suffix (should match generation in base-36)
    let suffix_len = input.read_byte()? as usize;
    let mut suffix_bytes = vec![0u8; suffix_len];
    input.read_bytes(&mut suffix_bytes)?;
    let suffix = String::from_utf8(suffix_bytes).map_err(|e| io::Error::other(e.to_string()))?;
    if suffix != expected_suffix {
        return Err(io::Error::other(format!(
            "segments suffix mismatch: expected {expected_suffix:?}, got {suffix:?}"
        )));
    }

    // Lucene version (VInts)
    let _major = input.read_vint()?;
    let _minor = input.read_vint()?;
    let _bugfix = input.read_vint()?;

    // Index created version major
    let _index_created_version = input.read_vint()?;

    // Segment infos version (BE long)
    let version = input.read_be_long()?;

    // Counter (VLong)
    let counter = input.read_vlong()?;

    // Number of segments (BE int)
    let num_segments = input.read_be_int()?;
    if num_segments < 0 {
        return Err(io::Error::other(format!(
            "invalid segment count: {num_segments}"
        )));
    }

    // Min segment version (only present if segments > 0)
    if num_segments > 0 {
        let _min_major = input.read_vint()?;
        let _min_minor = input.read_vint()?;
        let _min_bugfix = input.read_vint()?;
    }

    // Per-segment entries
    let mut segments = Vec::with_capacity(num_segments as usize);
    for _ in 0..num_segments {
        let seg_name = input.read_string()?;

        let mut seg_id = [0u8; codec_headers::ID_LENGTH];
        input.read_bytes(&mut seg_id)?;

        let codec_name = input.read_string()?;

        let del_gen = input.read_be_long()?;
        let del_count = input.read_be_int()?;
        let field_infos_gen = input.read_be_long()?;
        let doc_values_gen = input.read_be_long()?;
        let soft_del_count = input.read_be_int()?;

        let sci_id = match input.read_byte()? {
            1 => {
                let mut id = [0u8; codec_headers::ID_LENGTH];
                input.read_bytes(&mut id)?;
                Some(id)
            }
            0 => None,
            marker => {
                return Err(io::Error::other(format!("invalid SCI ID marker: {marker}")));
            }
        };

        // Field infos files (set of strings)
        let _field_infos_files = input.read_set_of_strings()?;

        // Doc values updates files
        let num_dv_fields = input.read_be_int()?;
        for _ in 0..num_dv_fields {
            let _field_number = input.read_be_int()?;
            let _files = input.read_set_of_strings()?;
        }

        segments.push(SegmentEntry {
            name: seg_name,
            id: seg_id,
            codec: codec_name,
            del_gen,
            del_count,
            field_infos_gen,
            doc_values_gen,
            soft_del_count,
            sci_id,
        });
    }

    // User data
    let user_data = input.read_map_of_strings()?;

    debug!(
        "segment_infos: read {segment_file_name}, version={version}, \
         counter={counter}, num_segments={num_segments}"
    );

    Ok(SegmentInfosRead {
        segments: segments.into_boxed_slice(),
        generation,
        version,
        counter,
        user_data,
    })
}

/// Writes a `segments_N` file for flushed segments to the directory.
///
/// Returns the filename written (e.g., "segments_1").
fn write_flushed_segments(
    directory: &dyn Directory,
    segments: &[FlushedSegment],
    generation: u64,
) -> io::Result<String> {
    let gen_suffix = index_file_names::radix36(generation);
    let pending_name = format!("pending_segments_{gen_suffix}");
    let final_name = format!("segments_{gen_suffix}");
    let id = string_helper::random_id();

    let counter = segments
        .iter()
        .map(|s| {
            // Parse the segment number from the name (e.g., "_0" → 0, "_a" → 10)
            let num_str = s.segment_id.name.trim_start_matches('_');
            u64::from_str_radix(num_str, 36).unwrap_or(0)
        })
        .max()
        .map(|m| m + 1)
        .unwrap_or(0) as i64;

    let version = generation as i64;
    let num_segments = segments.len() as i32;

    debug!(
        "segment_infos: writing {final_name}, version={version}, counter={counter}, num_segments={num_segments}"
    );

    let mut output = directory.create_output(&pending_name)?;

    // Index header
    codec_headers::write_index_header(&mut *output, CODEC_NAME, VERSION_CURRENT, &id, &gen_suffix)?;

    // Lucene version
    output.write_vint(LUCENE_VERSION_MAJOR)?;
    output.write_vint(LUCENE_VERSION_MINOR)?;
    output.write_vint(LUCENE_VERSION_BUGFIX)?;

    // Index created version major
    output.write_vint(LUCENE_VERSION_MAJOR)?;

    // Segment infos version (BE long)
    output.write_be_long(version)?;

    // Counter (VLong)
    output.write_vlong(counter)?;

    // Number of segments (BE int)
    output.write_be_int(num_segments)?;

    // Min segment version (only if segments > 0)
    if !segments.is_empty() {
        output.write_vint(LUCENE_VERSION_MAJOR)?;
        output.write_vint(LUCENE_VERSION_MINOR)?;
        output.write_vint(LUCENE_VERSION_BUGFIX)?;
    }

    // Per-segment entries
    for seg in segments {
        // Segment name
        output.write_string(&seg.segment_id.name)?;

        // Segment ID (16 bytes)
        output.write_all(&seg.segment_id.id)?;

        // Codec name
        output.write_string(SEGMENT_CODEC_NAME)?;

        // Del gen (BE long) — -1 for fresh segment
        output.write_be_long(-1)?;

        // Del count (BE int) — 0
        output.write_be_int(0)?;

        // Field infos gen (BE long) — -1
        output.write_be_long(-1)?;

        // Doc values gen (BE long) — -1
        output.write_be_long(-1)?;

        // Soft del count (BE int) — 0
        output.write_be_int(0)?;

        // SCI ID: present (1) + 16 random bytes
        let sci_id = string_helper::random_id();
        output.write_byte(1)?;
        output.write_all(&sci_id)?;

        // Field infos files (empty set)
        output.write_set_of_strings(&[])?;

        // Doc values updates files (empty: BE int 0)
        output.write_be_int(0)?;
    }

    // User data (empty map)
    output.write_map_of_strings(&HashMap::new())?;

    // Footer
    codec_footers::write_footer(&mut *output)?;

    // Flush the output before syncing
    drop(output);

    // Sync and rename
    directory.sync(&[&pending_name])?;
    directory.rename(&pending_name, &final_name)?;

    Ok(final_name)
}

/// Collects flushed segments and writes the `segments_N` commit point.
///
/// Holds the list of segments that make up the index and the generation
/// counter that increments on each commit. Writing is atomic: the file
/// is written to a temp name and renamed.
#[derive(Default)]
pub struct SegmentInfos {
    /// Flushed segments to include in the next commit.
    segments: Vec<FlushedSegment>,
    /// Generation counter — the `N` in `segments_N`.
    generation: u64,
}

impl SegmentInfos {
    /// Creates an empty instance at generation 0.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a flushed segment to be included in the next commit.
    pub fn add(&mut self, segment: FlushedSegment) {
        self.segments.push(segment);
    }

    /// Writes `segments_N` to the directory and increments the generation.
    ///
    /// The file is written to a pending name, synced, and atomically renamed.
    pub fn commit(&mut self, directory: &dyn Directory) -> io::Result<String> {
        self.generation += 1;
        write_flushed_segments(directory, &self.segments, self.generation)
    }

    /// Returns the current generation number.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Returns the segments in this commit.
    pub fn segments(&self) -> &[FlushedSegment] {
        &self.segments
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{MemoryDirectory, SharedDirectory};

    // --- Read round-trip tests ---

    #[test]
    fn test_read_roundtrip_empty() {
        let dir = test_shared_directory();
        let name = write_flushed_segments(&dir, &[], 1).unwrap();

        let guard = &*dir;
        let result = read(guard, &name).unwrap();
        assert_is_empty!(&result.segments);
        assert_eq!(result.version, 1);
        assert_eq!(result.counter, 0);
        assert_is_empty!(&result.user_data);
    }

    #[test]
    fn test_read_roundtrip_single_segment() {
        let dir = test_shared_directory();
        let segments = vec![make_flushed_segment("_0", 3)];
        let name = write_flushed_segments(&dir, &segments, 1).unwrap();

        let guard = &*dir;
        let result = read(guard, &name).unwrap();

        assert_len_eq_x!(&result.segments, 1);
        assert_eq!(result.segments[0].name, "_0");
        assert_eq!(result.segments[0].id, [0xABu8; 16]);
        assert_eq!(result.segments[0].codec, "Lucene103");
        assert_some!(&result.segments[0].sci_id);
        assert_eq!(result.version, 1);
        assert_eq!(result.counter, 1);
    }

    // --- generation_from_segments_file_name tests ---

    #[test]
    fn test_generation_bare_segments() {
        assert_eq!(generation_from_segments_file_name("segments").unwrap(), 0);
    }

    #[test]
    fn test_generation_single_digit() {
        assert_eq!(generation_from_segments_file_name("segments_1").unwrap(), 1);
        assert_eq!(generation_from_segments_file_name("segments_9").unwrap(), 9);
    }

    #[test]
    fn test_generation_base36_letters() {
        assert_eq!(
            generation_from_segments_file_name("segments_a").unwrap(),
            10
        );
        assert_eq!(
            generation_from_segments_file_name("segments_z").unwrap(),
            35
        );
    }

    #[test]
    fn test_generation_base36_multi_char() {
        assert_eq!(
            generation_from_segments_file_name("segments_10").unwrap(),
            36
        );
        assert_eq!(
            generation_from_segments_file_name("segments_1a").unwrap(),
            46
        );
    }

    #[test]
    fn test_generation_invalid_filename() {
        assert!(generation_from_segments_file_name("_0.cfs").is_err());
        assert!(generation_from_segments_file_name("not_segments").is_err());
    }

    // --- get_last_commit_segments_file_name tests ---

    #[test]
    fn test_last_commit_single_file() {
        let files = vec!["segments_1".to_string()];
        assert_eq!(
            get_last_commit_segments_file_name(&files).unwrap(),
            "segments_1"
        );
    }

    #[test]
    fn test_last_commit_numeric_max_not_lexicographic() {
        // segments_10 = gen 36, segments_z = gen 35
        // Lexicographic would pick segments_z; numeric picks segments_10
        let files = vec!["segments_z".to_string(), "segments_10".to_string()];
        assert_eq!(
            get_last_commit_segments_file_name(&files).unwrap(),
            "segments_10"
        );
    }

    #[test]
    fn test_last_commit_ignores_non_segments() {
        let files = vec![
            "_0.cfs".to_string(),
            "_0.si".to_string(),
            "segments_3".to_string(),
            "write.lock".to_string(),
        ];
        assert_eq!(
            get_last_commit_segments_file_name(&files).unwrap(),
            "segments_3"
        );
    }

    #[test]
    fn test_last_commit_no_segments_files() {
        let files = vec!["_0.cfs".to_string(), "write.lock".to_string()];
        assert!(get_last_commit_segments_file_name(&files).is_err());
    }

    #[test]
    fn test_last_commit_empty() {
        let files: Vec<String> = vec![];
        assert!(get_last_commit_segments_file_name(&files).is_err());
    }

    #[test]
    fn test_last_commit_multiple_generations() {
        let files = vec![
            "segments_1".to_string(),
            "segments_5".to_string(),
            "segments_3".to_string(),
        ];
        assert_eq!(
            get_last_commit_segments_file_name(&files).unwrap(),
            "segments_5"
        );
    }

    // --- write_flushed_segments tests ---

    use crate::index::segment::SegmentId;

    fn test_shared_directory() -> SharedDirectory {
        MemoryDirectory::create()
    }

    fn make_flushed_segment(name: &str, doc_count: i32) -> FlushedSegment {
        FlushedSegment {
            segment_id: SegmentId {
                name: name.to_string(),
                id: [0xABu8; 16],
            },
            doc_count,
            file_names: vec![format!("{name}.fdt")],
        }
    }

    #[test]
    fn write_flushed_empty_segments() {
        let dir = test_shared_directory();
        let name = write_flushed_segments(&dir, &[], 1).unwrap();
        assert_eq!(name, "segments_1");

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
    fn write_flushed_single_segment_structure() {
        let dir = test_shared_directory();
        let segments = vec![make_flushed_segment("_0", 5)];
        let name = write_flushed_segments(&dir, &segments, 1).unwrap();
        assert_eq!(name, "segments_1");

        let data = dir.read_file(&name).unwrap();

        // Header magic
        assert_eq!(&data[0..4], &[0x3f, 0xd7, 0x6c, 0x17]);

        // Footer magic
        let footer_start = data.len() - 16;
        assert_eq!(
            &data[footer_start..footer_start + 4],
            &[0xc0, 0x28, 0x93, 0xe8]
        );

        // After header: codec="segments"(8 chars)
        // Header = 4(magic) + 1+8(codec) + 4(version) + 16(id) + 1+1(suffix "1") = 35
        let offset = 35;

        // Lucene version: 10, 3, 2 as VInts
        assert_eq!(data[offset], 10);
        assert_eq!(data[offset + 1], 3);
        assert_eq!(data[offset + 2], 2);

        // Index created version major
        assert_eq!(data[offset + 3], 10);

        // File should be substantial (header + version + segment entry + footer)
        assert_gt!(data.len(), 80);
    }

    #[test]
    fn write_flushed_generation_suffix_base36() {
        let dir = test_shared_directory();
        let name = write_flushed_segments(&dir, &[], 36).unwrap();
        assert_eq!(name, "segments_10");
    }

    // --- SegmentInfos (commit wrapper) tests ---

    #[test]
    fn commit_writes_segments_file() {
        let dir = test_shared_directory();
        let mut si = SegmentInfos::new();
        si.add(make_flushed_segment("_0", 3));

        let name = si.commit(&dir).unwrap();
        assert_eq!(name, "segments_1");
        assert_eq!(si.generation(), 1);

        let data = dir.read_file(&name).unwrap();
        // Header magic
        assert_eq!(&data[0..4], &[0x3f, 0xd7, 0x6c, 0x17]);
    }

    #[test]
    fn commit_increments_generation() {
        let dir = test_shared_directory();
        let mut si = SegmentInfos::new();
        si.add(make_flushed_segment("_0", 1));

        si.commit(&dir).unwrap();
        assert_eq!(si.generation(), 1);

        si.add(make_flushed_segment("_1", 2));
        let name = si.commit(&dir).unwrap();
        assert_eq!(name, "segments_2");
        assert_eq!(si.generation(), 2);
    }
}
