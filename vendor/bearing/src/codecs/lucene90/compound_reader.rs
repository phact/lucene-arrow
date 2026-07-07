// SPDX-License-Identifier: Apache-2.0

//! Compound file reader — opens `.cfs`/`.cfe` files as a read-only [`Directory`].
//!
//! Ported from `Lucene90CompoundReader`.

use std::collections::HashMap;
use std::io;

use crate::codecs::codec_file_handle::{CodecFileHandle, IndexFile};
use crate::codecs::codec_footers::FOOTER_LENGTH;
use crate::codecs::codec_headers;
use crate::index::index_file_names;
use crate::store::{Directory, FileBacking, IndexInput, IndexOutput};

const DATA_EXTENSION: &str = "cfs";
const DATA_CODEC: &str = "Lucene90CompoundData";

struct FileEntry {
    offset: u64,
    length: u64,
}

/// A read-only [`Directory`] that reads files from a compound file (`.cfs`/`.cfe`).
///
/// Parses the `.cfe` entry table on construction and holds a borrowed reference
/// to the parent [`Directory`]. Each [`Directory::open_file`] call re-maps the
/// parent `.cfs` and returns a [`FileBacking::MmapSlice`] with the entry's
/// offset and length.
pub struct CompoundDirectory<'a> {
    parent: &'a dyn Directory,
    segment_name: String,
    data_file_name: String,
    entries: HashMap<String, FileEntry>,
    #[expect(dead_code)]
    version: i32,
}

impl<'a> CompoundDirectory<'a> {
    /// Opens a compound directory for the given segment.
    ///
    /// Reads the `.cfe` entry table from `directory` and validates the `.cfs`
    /// data file's header, footer, and length. Retains a reference to
    /// `directory` for on-demand sub-file access.
    pub fn open(
        directory: &'a dyn Directory,
        segment_name: &str,
        segment_id: &[u8; codec_headers::ID_LENGTH],
    ) -> io::Result<Self> {
        let (version, entries) = read_entries(directory, segment_name, segment_id)?;

        // Find the last FileEntry (largest offset+length) and add footer length
        let expected_length = entries
            .values()
            .map(|e| e.offset + e.length)
            .max()
            .unwrap_or(codec_headers::index_header_length(DATA_CODEC, "") as u64)
            + FOOTER_LENGTH as u64;

        let data_file_name = index_file_names::segment_file_name(segment_name, "", DATA_EXTENSION);

        // Validate .cfs header, footer, and length once at open time.
        // The handle is dropped at the end of this block; each subsequent
        // sub-file open re-maps the parent via `self.parent.open_file`.
        let cfs_handle = CodecFileHandle::open(
            directory,
            IndexFile::CompoundData,
            segment_name,
            segment_id,
            "",
        )?;
        if cfs_handle.version() != version {
            return Err(io::Error::other(format!(
                "compound version mismatch: cfe={version}, cfs={}",
                cfs_handle.version()
            )));
        }
        cfs_handle.verify_length(expected_length as i64)?;

        Ok(Self {
            parent: directory,
            segment_name: segment_name.to_string(),
            data_file_name,
            entries,
            version,
        })
    }
}

impl Directory for CompoundDirectory<'_> {
    fn create_output(&self, _name: &str) -> io::Result<Box<dyn IndexOutput>> {
        Err(io::Error::other(
            "CompoundDirectory is read-only: cannot create output",
        ))
    }

    fn open_file(&self, name: &str) -> io::Result<FileBacking> {
        let id = index_file_names::strip_segment_name(name);
        let entry = self.entries.get(id).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!(
                    "No sub-file with id {id} found in compound file \
                     (fileName={name} files: {:?})",
                    self.entries.keys().collect::<Vec<_>>()
                ),
            )
        })?;
        let offset = entry.offset as usize;
        let length = entry.length as usize;
        let parent_backing = self.parent.open_file(&self.data_file_name)?;
        match parent_backing {
            FileBacking::Mmap(mmap) => Ok(FileBacking::MmapSlice {
                mmap,
                offset,
                length,
            }),
            FileBacking::Owned(v) => Ok(FileBacking::Owned(v[offset..offset + length].to_vec())),
            FileBacking::MmapSlice { .. } => {
                Err(io::Error::other("nested compound files are not supported"))
            }
        }
    }

    fn list_all(&self) -> io::Result<Vec<String>> {
        let names: Vec<String> = self
            .entries
            .keys()
            .map(|k| format!("{}{k}", self.segment_name))
            .collect();
        Ok(names)
    }

    fn file_length(&self, name: &str) -> io::Result<u64> {
        let id = index_file_names::strip_segment_name(name);
        let entry = self.entries.get(id).ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotFound, format!("file not found: {name}"))
        })?;
        Ok(entry.length)
    }

    fn delete_file(&self, _name: &str) -> io::Result<()> {
        Err(io::Error::other(
            "CompoundDirectory is read-only: cannot delete",
        ))
    }

    fn rename(&self, _source: &str, _dest: &str) -> io::Result<()> {
        Err(io::Error::other(
            "CompoundDirectory is read-only: cannot rename",
        ))
    }

    fn read_file(&self, name: &str) -> io::Result<Vec<u8>> {
        Ok(self.open_file(name)?.as_bytes().to_vec())
    }
}

/// Reads the `.cfe` entry table. Returns `(version, entries)`.
fn read_entries(
    directory: &dyn Directory,
    segment_name: &str,
    segment_id: &[u8; codec_headers::ID_LENGTH],
) -> io::Result<(i32, HashMap<String, FileEntry>)> {
    let handle = CodecFileHandle::open(
        directory,
        IndexFile::CompoundEntries,
        segment_name,
        segment_id,
        "",
    )?;
    let version = handle.version();
    let mut input = handle.body();
    let mapping = read_mapping(&mut input)?;
    Ok((version, mapping))
}

/// Reads the entry mapping from the `.cfe` stream.
fn read_mapping(input: &mut IndexInput<'_>) -> io::Result<HashMap<String, FileEntry>> {
    let num_entries = input.read_vint()?;
    let mut mapping = HashMap::with_capacity(num_entries as usize);

    for _ in 0..num_entries {
        let id = input.read_string()?;
        let offset = input.read_le_long()? as u64;
        let length = input.read_le_long()? as u64;

        if mapping
            .insert(id.clone(), FileEntry { offset, length })
            .is_some()
        {
            return Err(io::Error::other(format!(
                "Duplicate cfs entry id={id} in CFS"
            )));
        }
    }

    Ok(mapping)
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;
    use crate::codecs::codec_footers;
    use crate::codecs::lucene90::compound;
    use crate::store::SharedDirectory;
    use crate::store::memory::MemoryIndexOutput;
    use crate::store::{MemoryDirectory, SegmentFile};
    use assertables::*;

    fn make_test_file(name: &str, segment_id: &[u8; 16], body: &[u8]) -> SegmentFile {
        let mut out = MemoryIndexOutput::new(name.to_string());
        codec_headers::write_index_header(&mut out, "TestCodec", 1, segment_id, "").unwrap();
        out.write_all(body).unwrap();
        codec_footers::write_footer(&mut out).unwrap();
        out.into_inner()
    }

    fn write_compound(
        segment_name: &str,
        segment_id: &[u8; 16],
        files: &[SegmentFile],
    ) -> (SegmentFile, SegmentFile) {
        let cfs_name = index_file_names::segment_file_name(segment_name, "", "cfs");
        let mut cfs_out = MemoryIndexOutput::new(cfs_name);
        let cfe = compound::write_to(segment_name, segment_id, files, &mut cfs_out).unwrap();
        (cfs_out.into_inner(), cfe)
    }

    fn setup_compound_files(
        segment_name: &str,
        segment_id: &[u8; 16],
        files: &[SegmentFile],
    ) -> SharedDirectory {
        let (cfs, cfe) = write_compound(segment_name, segment_id, files);
        let dir = MemoryDirectory::create();
        dir.write_file(&cfs.name, &cfs.data).unwrap();
        dir.write_file(&cfe.name, &cfe.data).unwrap();
        dir
    }

    #[test]
    fn test_list_all() {
        let seg_id = [0xABu8; 16];
        let files = vec![
            make_test_file("_0.fnm", &seg_id, b"field data"),
            make_test_file("_0.fdt", &seg_id, b"stored data"),
        ];
        let dir = setup_compound_files("_0", &seg_id, &files);
        let compound_dir = CompoundDirectory::open(&dir, "_0", &seg_id).unwrap();

        let listed = compound_dir.list_all().unwrap();
        assert_len_eq_x!(&listed, 2);
        assert_contains!(listed, &"_0.fnm".to_string());
        assert_contains!(listed, &"_0.fdt".to_string());
    }

    #[test]
    fn test_open_file_strips_segment_name() {
        let seg_id = [0xABu8; 16];
        let files = vec![make_test_file("_0.fnm", &seg_id, b"data")];
        let dir = setup_compound_files("_0", &seg_id, &files);
        let compound_dir = CompoundDirectory::open(&dir, "_0", &seg_id).unwrap();

        // Both full name and stripped name resolve via strip_segment_name
        assert!(compound_dir.open_file("_0.fnm").is_ok());
        assert!(compound_dir.open_file(".fnm").is_ok());
    }

    #[test]
    fn test_file_length() {
        let seg_id = [0xABu8; 16];
        let body = b"test body";
        let files = vec![make_test_file("_0.fnm", &seg_id, body)];
        let dir = setup_compound_files("_0", &seg_id, &files);
        let compound_dir = CompoundDirectory::open(&dir, "_0", &seg_id).unwrap();

        let len = compound_dir.file_length("_0.fnm").unwrap();
        let expected = codec_headers::index_header_length("TestCodec", "") as u64
            + body.len() as u64
            + codec_footers::FOOTER_LENGTH as u64;
        assert_eq!(len, expected);
    }

    #[test]
    fn test_open_file_missing() {
        let seg_id = [0xABu8; 16];
        let files = vec![make_test_file("_0.fnm", &seg_id, b"data")];
        let dir = setup_compound_files("_0", &seg_id, &files);
        let compound_dir = CompoundDirectory::open(&dir, "_0", &seg_id).unwrap();

        assert!(compound_dir.open_file("_0.xxx").is_err());
    }

    #[test]
    fn test_read_only_operations() {
        let seg_id = [0xABu8; 16];
        let files = vec![make_test_file("_0.fnm", &seg_id, b"data")];
        let dir = setup_compound_files("_0", &seg_id, &files);
        let compound_dir = CompoundDirectory::open(&dir, "_0", &seg_id).unwrap();

        assert!(compound_dir.create_output("test").is_err());
        assert!(compound_dir.delete_file("test").is_err());
        assert!(compound_dir.rename("a", "b").is_err());
    }

    #[test]
    fn test_multiple_files() {
        let seg_id = [0xABu8; 16];
        let files = vec![
            make_test_file("_0.fnm", &seg_id, b"field info data"),
            make_test_file("_0.fdt", &seg_id, b"stored fields data here"),
            make_test_file("_0.nvd", &seg_id, b"norms"),
        ];
        let dir = setup_compound_files("_0", &seg_id, &files);
        let compound_dir = CompoundDirectory::open(&dir, "_0", &seg_id).unwrap();

        let listed = compound_dir.list_all().unwrap();
        assert_len_eq_x!(&listed, 3);

        for name in ["_0.fnm", "_0.fdt", "_0.nvd"] {
            let backing = compound_dir.open_file(name).unwrap();
            let mut input = IndexInput::new(name, backing.as_bytes());
            let magic = input.read_be_int().unwrap();
            assert_eq!(magic, codec_headers::CODEC_MAGIC, "bad magic for {name}");
        }
    }

    #[test]
    fn test_read_file() {
        let seg_id = [0xABu8; 16];
        let files = vec![make_test_file("_0.fnm", &seg_id, b"body data")];
        let dir = setup_compound_files("_0", &seg_id, &files);
        let compound_dir = CompoundDirectory::open(&dir, "_0", &seg_id).unwrap();

        let data = compound_dir.read_file("_0.fnm").unwrap();
        assert_not_empty!(data);
        // First 4 bytes should be codec magic
        assert_eq!(&data[..4], &[0x3F, 0xD7, 0x6C, 0x17]);
    }

    #[test]
    fn test_truncated_cfs_detected() {
        let seg_id = [0xABu8; 16];
        let files = vec![make_test_file("_0.fnm", &seg_id, b"data")];
        let (cfs, cfe) = write_compound("_0", &seg_id, &files);

        let dir = MemoryDirectory::create();
        // Truncate the .cfs file
        let truncated = &cfs.data[..cfs.data.len() - 4];
        dir.write_file(&cfs.name, truncated).unwrap();
        dir.write_file(&cfe.name, &cfe.data).unwrap();

        let result = CompoundDirectory::open(&dir, "_0", &seg_id);
        assert!(result.is_err());
    }
}
