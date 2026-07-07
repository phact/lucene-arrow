// SPDX-License-Identifier: Apache-2.0

//! Storage abstraction layer: directories, file backing, and I/O primitives.
//!
//! The [`Directory`] trait abstracts file storage. [`FSDirectory`] opens a
//! filesystem directory; by default it returns an [`MmapDirectory`] for
//! zero-copy reads. [`MemoryDirectory`] holds files in memory (for tests).
//! [`CompoundDirectory`] exposes a `.cfs` compound file's sub-files as a
//! read-only directory.
//!
//! # Write side
//!
//! [`DataOutput`] and [`IndexOutput`] define the byte-level writing interface
//! used by codec writers.
//!
//! # Read side
//!
//! Codec readers open files through [`Directory::open_file`], which returns a
//! [`FileBacking`] owning the file's bytes — either a memory-mapped region
//! ([`FileBacking::Mmap`] / [`FileBacking::MmapSlice`]) or an owned byte vector
//! ([`FileBacking::Owned`]). Readers then construct positioned views over the
//! backing's bytes for sequential and random access.

pub mod checksum;
pub mod data_output;
pub(super) mod file_backing;
pub mod fs;
pub(super) mod index_input;
pub mod index_output;
pub mod memory;
pub mod mmap;

pub use checksum::CRC32;
pub use data_output::{DataOutput, VecOutput};
pub use file_backing::FileBacking;
pub use fs::FSDirectory;
pub(crate) use index_input::IndexInput;
pub use index_output::IndexOutput;
pub use memory::MemoryDirectory;
pub use mmap::MmapDirectory;

// Re-export CompoundDirectory — a read-only Directory for compound files (.cfs/.cfe)
pub use crate::codecs::lucene90::compound_reader::CompoundDirectory;

use std::io;
use std::sync::Arc;

/// A named in-memory file produced by codec writers during indexing.
#[derive(Clone, Debug)]
pub(crate) struct SegmentFile {
    pub(crate) name: String,
    pub(crate) data: Vec<u8>,
}

/// A shared, thread-safe [`Directory`] reference.
///
/// Functions that need shared ownership of a directory (struct fields,
/// cross-thread handoffs) take `SharedDirectory`. Functions that just
/// borrow a directory for the duration of a call take `&dyn Directory`.
///
/// ```no_run
/// use bearing::store::FSDirectory;
///
/// let directory = FSDirectory::open(std::path::Path::new("/tmp/my-index")).unwrap();
/// ```
pub type SharedDirectory = Arc<dyn Directory>;

/// Trait for a directory that can create and manage index files.
///
/// All methods take `&self`. Implementations use interior mutability where
/// needed (e.g., `RwLock` for in-memory directories). Filesystem-backed
/// directories delegate to OS calls that are inherently thread-safe.
pub trait Directory: Send + Sync {
    /// Creates a new output file with the given name.
    fn create_output(&self, name: &str) -> io::Result<Box<dyn IndexOutput>>;

    /// Lists all files in this directory.
    fn list_all(&self) -> io::Result<Vec<String>>;

    /// Returns the byte length of a file.
    fn file_length(&self, name: &str) -> io::Result<u64>;

    /// Deletes a file.
    fn delete_file(&self, name: &str) -> io::Result<()>;

    /// Renames a file. Used for atomic commit of segments_N.
    fn rename(&self, source: &str, dest: &str) -> io::Result<()>;

    /// Reads the raw bytes of a file into memory.
    fn read_file(&self, name: &str) -> io::Result<Vec<u8>>;

    /// Opens a file as a [`FileBacking`].
    ///
    /// Returns the bytes of the named file as either an owned `Vec<u8>` or a
    /// memory-mapped region, depending on the implementation. The default impl
    /// reads the file into memory via [`read_file`](Self::read_file); mmap-backed
    /// directories override this to return a zero-copy `Mmap`.
    fn open_file(&self, name: &str) -> io::Result<FileBacking> {
        Ok(FileBacking::Owned(self.read_file(name)?))
    }

    /// Writes complete byte contents to a file in this directory.
    /// Default implementation uses `create_output` + `write_bytes`.
    fn write_file(&self, name: &str, data: &[u8]) -> io::Result<()> {
        let mut out = self.create_output(name)?;
        out.write_all(data)?;
        Ok(())
    }

    /// Ensures that any writes to the named files are moved to stable storage.
    fn sync(&self, names: &[&str]) -> io::Result<()> {
        let _ = names;
        Ok(())
    }

    /// Ensures that directory metadata (e.g., new file entries) is persisted.
    fn sync_meta_data(&self) -> io::Result<()> {
        Ok(())
    }
}

/// Blanket impl so `&SharedDirectory` (`&Arc<dyn Directory>`) can be used
/// anywhere `&dyn Directory` is expected, without manual `&*` deref.
impl Directory for Arc<dyn Directory> {
    fn create_output(&self, name: &str) -> io::Result<Box<dyn IndexOutput>> {
        (**self).create_output(name)
    }
    fn list_all(&self) -> io::Result<Vec<String>> {
        (**self).list_all()
    }
    fn file_length(&self, name: &str) -> io::Result<u64> {
        (**self).file_length(name)
    }
    fn delete_file(&self, name: &str) -> io::Result<()> {
        (**self).delete_file(name)
    }
    fn rename(&self, source: &str, dest: &str) -> io::Result<()> {
        (**self).rename(source, dest)
    }
    fn read_file(&self, name: &str) -> io::Result<Vec<u8>> {
        (**self).read_file(name)
    }
    fn open_file(&self, name: &str) -> io::Result<FileBacking> {
        (**self).open_file(name)
    }
    fn write_file(&self, name: &str, data: &[u8]) -> io::Result<()> {
        (**self).write_file(name, data)
    }
    fn sync(&self, names: &[&str]) -> io::Result<()> {
        (**self).sync(names)
    }
    fn sync_meta_data(&self) -> io::Result<()> {
        (**self).sync_meta_data()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::index_output::align_offset;

    // --- DataOutput tests ---

    #[test]
    fn test_write_le_int() {
        let mut buf = Vec::new();
        VecOutput(&mut buf).write_le_int(0x04030201_i32).unwrap();
        assert_eq!(buf, [0x01, 0x02, 0x03, 0x04]);
    }

    #[test]
    fn test_write_le_long() {
        let mut buf = Vec::new();
        VecOutput(&mut buf)
            .write_le_long(0x0807060504030201_i64)
            .unwrap();
        assert_eq!(buf, [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]);
    }

    #[test]
    fn test_write_le_short() {
        let mut buf = Vec::new();
        VecOutput(&mut buf).write_le_short(0x0201_i16).unwrap();
        assert_eq!(buf, [0x01, 0x02]);
    }

    #[test]
    fn test_write_be_int() {
        let mut buf = Vec::new();
        VecOutput(&mut buf).write_be_int(0x04030201_i32).unwrap();
        assert_eq!(buf, [0x04, 0x03, 0x02, 0x01]);
    }

    #[test]
    fn test_write_be_long() {
        let mut buf = Vec::new();
        VecOutput(&mut buf)
            .write_be_long(0x0807060504030201_i64)
            .unwrap();
        assert_eq!(buf, [0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01]);
    }

    #[test]
    fn test_align_offset() {
        assert_eq!(align_offset(0, 8), 0);
        assert_eq!(align_offset(1, 8), 8);
        assert_eq!(align_offset(7, 8), 8);
        assert_eq!(align_offset(8, 8), 8);
        assert_eq!(align_offset(9, 8), 16);
    }
}
