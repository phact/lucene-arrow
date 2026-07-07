// SPDX-License-Identifier: Apache-2.0

//! `SegmentSource` — bandwidth-agnostic IO (SPEC §3.4, §9).
//!
//! Decode/encode executors never know whether bytes arrived via mmap, a
//! pinned bounce buffer, or true GPUDirect Storage. They see a
//! [`BufferTarget`], full stop. `MmapSource` is the default host-memory
//! implementation; `KvikioSource` lives in the `gpu` crate behind the `gpu`
//! feature and implements the same traits.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::error::{Error, Result};

/// Destination for a byte-range fetch: host slice or raw device pointer.
///
/// The device variant is deliberately not feature-gated — plans and sources
/// are transport-agnostic by contract; only implementations that *fill*
/// device memory need CUDA.
pub enum BufferTarget<'a> {
    Host(&'a mut [u8]),
    /// CUDA device pointer + capacity in bytes. The impl decides DMA vs
    /// bounce; callers guarantee the allocation outlives the call.
    Device { ptr: u64, capacity: u64 },
}

impl BufferTarget<'_> {
    pub fn capacity(&self) -> u64 {
        match self {
            BufferTarget::Host(s) => s.len() as u64,
            BufferTarget::Device { capacity, .. } => *capacity,
        }
    }
}

/// One openable file within a segment directory.
pub trait ByteRange: Send + Sync {
    /// Total length of the file in bytes.
    fn len(&self) -> u64;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Fetch `[offset, offset + len)` into `dst`. The impl decides DMA vs
    /// bounce vs mmap-copy. Errors if the range is out of bounds or `dst`
    /// is too small.
    fn read_into(&self, offset: u64, len: u64, dst: BufferTarget<'_>) -> Result<()>;

    /// Zero-copy view of `[offset, offset + len)` in host memory, when the
    /// source can offer one (mmap can; network/GDS sources return `None`).
    /// Executors use this to skip the bounce copy on the CPU path.
    fn slice(&self, _offset: u64, _len: u64) -> Option<&[u8]> {
        None
    }
}

/// Opens files of one segment directory (or compound-file view of one).
pub trait SegmentSource: Send + Sync {
    fn open(&self, file: &str) -> Result<Arc<dyn ByteRange>>;

    /// Files visible in this source (segment-directory listing).
    fn list(&self) -> Result<Vec<String>>;
}

/// Default source: mmap over a local directory. No GPU anywhere.
pub struct MmapSource {
    dir: PathBuf,
}

impl MmapSource {
    pub fn new(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        if !dir.is_dir() {
            return Err(Error::invalid(format!("not a directory: {}", dir.display())));
        }
        Ok(MmapSource { dir })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }
}

struct MmapRange {
    map: memmap2::Mmap,
}

impl ByteRange for MmapRange {
    fn len(&self) -> u64 {
        self.map.len() as u64
    }

    fn read_into(&self, offset: u64, len: u64, dst: BufferTarget<'_>) -> Result<()> {
        let end = offset
            .checked_add(len)
            .ok_or_else(|| Error::invalid("offset + len overflows u64"))?;
        if end > self.len() {
            return Err(Error::corrupt(format!(
                "read [{offset}, {end}) beyond file of {} bytes",
                self.len()
            )));
        }
        if dst.capacity() < len {
            return Err(Error::invalid(format!(
                "destination capacity {} < requested {len}",
                dst.capacity()
            )));
        }
        let src = &self.map[offset as usize..end as usize];
        match dst {
            BufferTarget::Host(out) => out[..len as usize].copy_from_slice(src),
            BufferTarget::Device { .. } => {
                return Err(Error::unsupported(
                    "MmapSource cannot fill device memory; use KvikioSource (gpu feature)",
                ));
            }
        }
        Ok(())
    }

    fn slice(&self, offset: u64, len: u64) -> Option<&[u8]> {
        let end = offset.checked_add(len)?;
        if end > self.len() {
            return None;
        }
        Some(&self.map[offset as usize..end as usize])
    }
}

impl SegmentSource for MmapSource {
    fn open(&self, file: &str) -> Result<Arc<dyn ByteRange>> {
        let path = self.dir.join(file);
        let f = File::open(&path)?;
        // Safety: standard mmap caveat — the mapping is UB if the file is
        // truncated concurrently. Lucene segment files are write-once, so a
        // segment directory not under active IndexWriter mutation is safe.
        let map = unsafe { memmap2::Mmap::map(&f)? };
        Ok(Arc::new(MmapRange { map }))
    }

    fn list(&self) -> Result<Vec<String>> {
        let mut names = Vec::new();
        for entry in std::fs::read_dir(&self.dir)? {
            let entry = entry?;
            if entry.file_type()?.is_file()
                && let Some(name) = entry.file_name().to_str()
            {
                names.push(name.to_string());
            }
        }
        names.sort();
        Ok(names)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mmap_source_reads_ranges() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("_0.dvd"), b"0123456789").unwrap();

        let src = MmapSource::new(dir.path()).unwrap();
        assert_eq!(src.list().unwrap(), vec!["_0.dvd".to_string()]);

        let range = src.open("_0.dvd").unwrap();
        assert_eq!(range.len(), 10);

        let mut buf = [0u8; 4];
        range.read_into(3, 4, BufferTarget::Host(&mut buf)).unwrap();
        assert_eq!(&buf, b"3456");
        assert_eq!(range.slice(3, 4).unwrap(), b"3456");

        assert!(range.read_into(8, 4, BufferTarget::Host(&mut buf)).is_err());
    }
}
