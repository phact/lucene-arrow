// SPDX-License-Identifier: Apache-2.0

//! Filesystem-backed [`Directory`] implementations.
//!
//! [`FSDirectory`] uses file handles for reads (like Java's `NIOFSDirectory`).
//! [`MmapDirectory`] uses memory-mapped I/O
//! (like Java's `MMapDirectory`) and is preferred for read-heavy workloads.
//! `FSDirectory::open()` returns an `MmapDirectory`.

use std::fs;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::mmap::MmapDirectory;
use crate::store::checksum::CRC32;
use crate::store::{DataOutput, Directory, IndexOutput};

// ============================================================
// Shared filesystem helpers used by both FSDirectory and MmapDirectory
// ============================================================

pub(crate) fn fs_create_output(dir_path: &Path, name: &str) -> io::Result<Box<dyn IndexOutput>> {
    let file_path = dir_path.join(name);
    let file = File::create(&file_path)?;
    let writer = BufWriter::new(file);
    Ok(Box::new(FSIndexOutput::new(name.to_string(), writer)))
}

pub(crate) fn fs_list_all(dir_path: &Path) -> io::Result<Vec<String>> {
    let mut names: Vec<String> = fs::read_dir(dir_path)?
        .filter_map(|entry| {
            entry.ok().and_then(|e| {
                let ft = e.file_type().ok()?;
                if ft.is_file() {
                    e.file_name().into_string().ok()
                } else {
                    None
                }
            })
        })
        .collect();
    names.sort();
    Ok(names)
}

pub(crate) fn fs_file_length(dir_path: &Path, name: &str) -> io::Result<u64> {
    let file_path = dir_path.join(name);
    let meta = fs::metadata(&file_path)?;
    Ok(meta.len())
}

pub(crate) fn fs_delete_file(dir_path: &Path, name: &str) -> io::Result<()> {
    fs::remove_file(dir_path.join(name))
}

pub(crate) fn fs_rename(dir_path: &Path, source: &str, dest: &str) -> io::Result<()> {
    fs::rename(dir_path.join(source), dir_path.join(dest))
}

pub(crate) fn fs_read_file(dir_path: &Path, name: &str) -> io::Result<Vec<u8>> {
    fs::read(dir_path.join(name))
}

pub(crate) fn fs_write_file(dir_path: &Path, name: &str, data: &[u8]) -> io::Result<()> {
    fs::write(dir_path.join(name), data)
}

pub(crate) fn fs_sync(dir_path: &Path, names: &[&str]) -> io::Result<()> {
    for name in names {
        let file = File::open(dir_path.join(name))?;
        file.sync_all()?;
    }
    Ok(())
}

pub(crate) fn fs_sync_meta_data(dir_path: &Path) -> io::Result<()> {
    let dir = File::open(dir_path)?;
    dir.sync_all()
}

// ============================================================
// FSDirectory — file-handle-based reads
// ============================================================

/// Filesystem directory using file handles for reads.
///
/// For read-heavy workloads (queries), prefer
/// [`MmapDirectory`] which avoids syscalls.
/// `FSDirectory::open()` returns an `MmapDirectory` by default, matching
/// Java's `FSDirectory.open()` behavior.
pub struct FSDirectory {
    path: PathBuf,
}

impl FSDirectory {
    /// Opens (or creates) a filesystem directory at the given path.
    ///
    /// Returns an [`MmapDirectory`], matching
    /// Java where `FSDirectory.open()` returns `MMapDirectory` on 64-bit.
    pub fn open(path: &Path) -> io::Result<super::SharedDirectory> {
        MmapDirectory::create(path)
    }

    /// Opens a filesystem directory with file-handle-based reads.
    ///
    /// Use this only when mmap is not desired. For most read workloads,
    /// prefer `FSDirectory::open()` which returns an `MmapDirectory`.
    pub fn open_with_file_handles(path: &Path) -> io::Result<super::SharedDirectory> {
        fs::create_dir_all(path)?;
        Ok(Arc::new(Self {
            path: path.to_path_buf(),
        }))
    }

    /// Returns the filesystem path of this directory.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Directory for FSDirectory {
    fn create_output(&self, name: &str) -> io::Result<Box<dyn IndexOutput>> {
        fs_create_output(&self.path, name)
    }

    fn list_all(&self) -> io::Result<Vec<String>> {
        fs_list_all(&self.path)
    }

    fn file_length(&self, name: &str) -> io::Result<u64> {
        fs_file_length(&self.path, name)
    }

    fn delete_file(&self, name: &str) -> io::Result<()> {
        fs_delete_file(&self.path, name)
    }

    fn rename(&self, source: &str, dest: &str) -> io::Result<()> {
        fs_rename(&self.path, source, dest)
    }

    fn read_file(&self, name: &str) -> io::Result<Vec<u8>> {
        fs_read_file(&self.path, name)
    }

    fn write_file(&self, name: &str, data: &[u8]) -> io::Result<()> {
        fs_write_file(&self.path, name, data)
    }

    fn sync(&self, names: &[&str]) -> io::Result<()> {
        fs_sync(&self.path, names)
    }

    fn sync_meta_data(&self) -> io::Result<()> {
        fs_sync_meta_data(&self.path)
    }
}

/// Filesystem-backed IndexOutput wrapping a `BufWriter<File>` with CRC32 tracking.
pub struct FSIndexOutput {
    name: String,
    writer: Option<BufWriter<File>>,
    crc: CRC32,
    position: u64,
}

impl FSIndexOutput {
    fn new(name: String, writer: BufWriter<File>) -> Self {
        Self {
            name,
            writer: Some(writer),
            crc: CRC32::new(),
            position: 0,
        }
    }

    fn writer(&mut self) -> &mut BufWriter<File> {
        self.writer
            .as_mut()
            .expect("writer already consumed by into_inner")
    }

    /// Flushes the BufWriter and returns the underlying File.
    pub fn into_inner(mut self) -> io::Result<File> {
        let mut writer = self.writer.take().expect("writer already consumed");
        writer.flush()?;
        writer.into_inner().map_err(io::Error::other)
    }
}

impl Drop for FSIndexOutput {
    fn drop(&mut self) {
        if let Some(ref mut writer) = self.writer {
            let _ = writer.flush();
        }
    }
}

impl io::Write for FSIndexOutput {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.writer().write_all(buf)?;
        self.crc.update(buf);
        self.position += buf.len() as u64;
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        if let Some(ref mut writer) = self.writer {
            writer.flush()
        } else {
            Ok(())
        }
    }
}

impl DataOutput for FSIndexOutput {
    fn write_byte(&mut self, b: u8) -> io::Result<()> {
        self.writer().write_all(&[b])?;
        self.crc.update_byte(b);
        self.position += 1;
        Ok(())
    }
}

impl IndexOutput for FSIndexOutput {
    fn name(&self) -> &str {
        &self.name
    }

    fn file_pointer(&self) -> u64 {
        self.position
    }

    fn checksum(&self) -> u64 {
        self.crc.value()
    }
}

#[cfg(test)]
mod tests {
    use std::env;
    use std::process;

    use std::io::Write;

    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        let dir = env::temp_dir().join(format!("rustlucene_test_fs_{name}_{}", process::id()));
        // Clean up any leftover from a previous run
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn test_fs_directory_write_and_list() {
        let dir_path = temp_dir("write_and_list");
        let dir = FSDirectory::open(&dir_path).unwrap();
        let _cleanup = DirCleanup(&dir_path);

        dir.write_file("beta.bin", b"hello").unwrap();
        dir.write_file("alpha.bin", b"world").unwrap();

        let files = dir.list_all().unwrap();
        assert_eq!(files, vec!["alpha.bin", "beta.bin"]);
    }

    #[test]
    fn test_fs_directory_file_length() {
        let dir_path = temp_dir("file_length");
        let dir = FSDirectory::open(&dir_path).unwrap();
        let _cleanup = DirCleanup(&dir_path);

        dir.write_file("test.bin", b"hello world").unwrap();
        assert_eq!(dir.file_length("test.bin").unwrap(), 11);
    }

    #[test]
    fn test_fs_directory_delete_file() {
        let dir_path = temp_dir("delete_file");
        let dir = FSDirectory::open(&dir_path).unwrap();
        let _cleanup = DirCleanup(&dir_path);

        dir.write_file("test.bin", b"data").unwrap();
        assert_ok!(dir.delete_file("test.bin"));
        assert_err!(dir.delete_file("test.bin"));
    }

    #[test]
    fn test_fs_directory_rename() {
        let dir_path = temp_dir("rename");
        let dir = FSDirectory::open(&dir_path).unwrap();
        let _cleanup = DirCleanup(&dir_path);

        dir.write_file("old.bin", b"data").unwrap();
        dir.rename("old.bin", "new.bin").unwrap();

        assert_err!(dir.file_length("old.bin"));
        assert_eq!(dir.file_length("new.bin").unwrap(), 4);
    }

    #[test]
    fn test_fs_directory_create_output() {
        let dir_path = temp_dir("create_output");
        let dir = FSDirectory::open(&dir_path).unwrap();
        let _cleanup = DirCleanup(&dir_path);

        {
            let mut out = dir.create_output("test.bin").unwrap();
            out.write_all(b"hello").unwrap();
            assert_eq!(out.file_pointer(), 5);
            assert_eq!(out.name(), "test.bin");
            // CRC32 of "hello" = 0x3610A686
            assert_eq!(out.checksum(), 0x3610A686);
        }

        // Verify file contents on disk
        let contents = fs::read(dir_path.join("test.bin")).unwrap();
        assert_eq!(contents, b"hello");
    }

    #[test]
    fn test_fs_output_into_inner() {
        let dir_path = temp_dir("into_inner");
        fs::create_dir_all(&dir_path).unwrap();
        let _cleanup = DirCleanup(&dir_path);

        let file = File::create(dir_path.join("into_inner.bin")).unwrap();
        let writer = BufWriter::new(file);
        let mut fs_out = FSIndexOutput::new("into_inner.bin".to_string(), writer);
        fs_out.write_all(b"test data").unwrap();
        let file = fs_out.into_inner().unwrap();
        drop(file);

        // Verify file contents on disk
        let contents = fs::read(dir_path.join("into_inner.bin")).unwrap();
        assert_eq!(contents, b"test data");
    }

    #[test]
    fn test_fs_directory_read_file() {
        let dir_path = temp_dir("read_file");
        let dir = FSDirectory::open(&dir_path).unwrap();
        let _cleanup = DirCleanup(&dir_path);

        dir.write_file("test.bin", b"hello world").unwrap();
        let data = dir.read_file("test.bin").unwrap();
        assert_eq!(data, b"hello world");

        assert_err!(dir.read_file("nonexistent.bin"));
    }

    #[test]
    fn test_fs_directory_sync() {
        let dir_path = temp_dir("sync");
        let dir = FSDirectory::open(&dir_path).unwrap();
        let _cleanup = DirCleanup(&dir_path);

        dir.write_file("sync_test.bin", b"data").unwrap();
        dir.sync(&["sync_test.bin"]).unwrap();
    }

    #[test]
    fn test_fs_directory_sync_meta_data() {
        let dir_path = temp_dir("sync_meta");
        let dir = FSDirectory::open(&dir_path).unwrap();
        let _cleanup = DirCleanup(&dir_path);

        dir.sync_meta_data().unwrap();
    }

    #[test]
    fn test_fs_output_write_byte() {
        let dir_path = temp_dir("write_byte");
        let dir = FSDirectory::open(&dir_path).unwrap();
        let _cleanup = DirCleanup(&dir_path);

        {
            let mut out = dir.create_output("byte_test.bin").unwrap();
            out.write_byte(0x42).unwrap();
            out.write_byte(0x43).unwrap();
            assert_eq!(out.file_pointer(), 2);
        }

        let contents = fs::read(dir_path.join("byte_test.bin")).unwrap();
        assert_eq!(contents, &[0x42, 0x43]);
    }

    #[test]
    fn test_fs_output_checksum() {
        let dir_path = temp_dir("checksum");
        let dir = FSDirectory::open(&dir_path).unwrap();
        let _cleanup = DirCleanup(&dir_path);

        let mut out = dir.create_output("checksum.bin").unwrap();
        out.write_all(b"test").unwrap();

        // Verify checksum matches manually computed CRC32
        let mut crc = CRC32::new();
        crc.update(b"test");
        assert_eq!(out.checksum(), crc.value());
    }

    #[test]
    fn test_fs_directory_open_file_returns_owned_with_correct_bytes() {
        use crate::store::FileBacking;

        let dir_path = temp_dir("open_file");
        let dir = FSDirectory::open_with_file_handles(&dir_path).unwrap();
        let _cleanup = DirCleanup(&dir_path);

        dir.write_file("backing.bin", b"hello fs backing").unwrap();
        let backing = dir.open_file("backing.bin").unwrap();

        assert_matches!(backing, FileBacking::Owned(_));
        assert_eq!(backing.as_bytes(), b"hello fs backing");
        assert_len_eq_x!(backing, 16);
    }

    #[test]
    fn test_fs_directory_open_file_missing() {
        let dir_path = temp_dir("open_file_missing");
        let dir = FSDirectory::open_with_file_handles(&dir_path).unwrap();
        let _cleanup = DirCleanup(&dir_path);

        assert_err!(dir.open_file("nonexistent.bin"));
    }

    /// RAII helper to clean up temp directories after tests.
    struct DirCleanup<'a>(&'a Path);

    impl<'a> Drop for DirCleanup<'a> {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(self.0);
        }
    }
}
