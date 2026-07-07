// SPDX-License-Identifier: Apache-2.0

//! Memory-mapped [`Directory`] implementation.
//!
//! Files are mapped into the process address space once at open time. All
//! subsequent reads — sequential, sliced, and random-access — are zero-copy
//! array indexing with no syscall overhead. This matches Java's `MMapDirectory`,
//! the default `Directory` for read-heavy workloads in Lucene.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use memmap2::Mmap;

use crate::store::fs::{
    fs_create_output, fs_delete_file, fs_file_length, fs_list_all, fs_read_file, fs_rename,
    fs_sync, fs_sync_meta_data, fs_write_file,
};
use crate::store::{Directory, FileBacking, IndexOutput};

/// Memory-mapped filesystem directory.
///
/// Uses `memmap2::Mmap` to map index files into the process address space.
/// All reads are zero-copy array indexing — no file handles, no seeks, no
/// syscalls. This is the preferred `Directory` for queries and is returned
/// by [`FSDirectory::open()`](super::FSDirectory::open).
pub struct MmapDirectory {
    path: PathBuf,
}

impl MmapDirectory {
    /// Opens (or creates) a memory-mapped directory at the given path.
    pub fn create(path: &Path) -> io::Result<super::SharedDirectory> {
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

impl Directory for MmapDirectory {
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

    fn open_file(&self, name: &str) -> io::Result<FileBacking> {
        let file_path = self.path.join(name);
        let file = fs::File::open(&file_path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        Ok(FileBacking::Mmap(mmap))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mmap_open_file_returns_mmap_with_correct_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let mmap_dir = MmapDirectory::create(dir.path()).unwrap();
        mmap_dir
            .write_file("backing.bin", b"hello mmap backing")
            .unwrap();

        let backing = mmap_dir.open_file("backing.bin").unwrap();

        assert_matches!(backing, FileBacking::Mmap(_));
        assert_eq!(backing.as_bytes(), b"hello mmap backing");
        assert_len_eq_x!(backing, 18);
    }

    #[test]
    fn test_mmap_open_file_missing() {
        let dir = tempfile::tempdir().unwrap();
        let mmap_dir = MmapDirectory::create(dir.path()).unwrap();

        assert_err!(mmap_dir.open_file("nonexistent.bin"));
    }
}
