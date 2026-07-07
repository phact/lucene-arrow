// SPDX-License-Identifier: Apache-2.0

//! Shared byte arena and variable-length slice allocation for postings data.
//!
//! [`ByteBlockPool`] manages a contiguous byte buffer. [`ByteSlicePool`]
//! allocates variable-length slices within the pool, growing through defined levels
//! with forwarding addresses. [`ByteSliceReader`] provides cursor-based read access
//! over the slice chain.

use std::io;

use mem_dbg::MemSize;

use crate::encoding::varint;

// ---------------------------------------------------------------------------
// ByteBlockPool
// ---------------------------------------------------------------------------

/// Contiguous byte arena for postings and term data.
///
/// Backed by a single flat `Vec<u8>`. All offsets are direct indices.
/// `data.len()` is the write frontier — all bytes beyond it are unallocated.
#[derive(Debug)]
pub struct ByteBlockPool {
    /// The backing storage. Callers access directly via `pool.data[offset]`.
    pub(crate) data: Vec<u8>,
}

impl MemSize for ByteBlockPool {
    fn mem_size_rec(
        &self,
        _flags: mem_dbg::SizeFlags,
        _refs: &mut mem_dbg::HashMap<usize, usize>,
    ) -> usize {
        // Report used bytes, not capacity.
        self.data.len()
    }
}

/// Growth chunk size — capacity grows in 32KB increments instead of doubling.
const GROWTH_CHUNK: usize = 32 * 1024;

impl ByteBlockPool {
    /// Creates a new pool with the given initial capacity.
    pub fn new(initial_capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(initial_capacity),
        }
    }

    /// Allocates `n` zeroed bytes and returns the start offset.
    ///
    /// Grows capacity in fixed chunks to keep it close to actual usage,
    /// avoiding the 2x overhead of Vec's default doubling strategy.
    pub fn alloc(&mut self, n: usize) -> usize {
        let offset = self.data.len();
        let needed = offset + n;
        if needed > self.data.capacity() {
            let grow = n.max(GROWTH_CHUNK);
            self.data.reserve_exact(grow);
        }
        self.data.resize(needed, 0);
        offset
    }

    /// Resets the pool, clearing all data but preserving allocated capacity.
    pub fn reset(&mut self) {
        self.data.clear();
    }
}

// ---------------------------------------------------------------------------
// ByteSlicePool
// ---------------------------------------------------------------------------

/// Sizes for each slice growth level.
const LEVEL_SIZE_ARRAY: [usize; 10] = [5, 14, 20, 30, 40, 40, 80, 80, 120, 200];

/// Next level index for each level (last level stays at max).
const NEXT_LEVEL_ARRAY: [usize; 10] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 9];

/// Size of the first slice level.
pub const FIRST_LEVEL_SIZE: usize = LEVEL_SIZE_ARRAY[0];

/// Allocates variable-length byte slices within a [`ByteBlockPool`].
///
/// Slices grow through predefined levels. Each slice ends with a level marker
/// byte (upper nibble = 0x10, lower nibble = level index). When a slice is
/// exhausted, a forwarding address (LE i32) is written in place of the last
/// 4 bytes, pointing to the next slice.
pub struct ByteSlicePool;

impl ByteSlicePool {
    /// Allocate a new initial slice of `size` bytes. Returns the global offset
    /// where the slice starts.
    pub fn new_slice(pool: &mut ByteBlockPool, size: usize) -> usize {
        let upto = pool.alloc(size);
        // Write level 0 marker at end of slice
        pool.data[upto + size - 1] = 0x10;
        upto
    }

    /// Grow a slice to its next level. Returns `(data_start, usable_length)` of
    /// the new slice, where `data_start` is the global offset and `usable_length`
    /// is bytes available for data (excluding the forwarding address reservation
    /// and level marker).
    ///
    /// `upto` is the global offset of the end-of-slice marker in the old slice.
    pub fn alloc_known_size_slice(pool: &mut ByteBlockPool, upto: usize) -> (usize, usize) {
        let level = (pool.data[upto] & 0x0F) as usize;
        let new_level = NEXT_LEVEL_ARRAY[level];
        let new_size = LEVEL_SIZE_ARRAY[new_level];

        let new_upto = pool.alloc(new_size);

        // Save the 3 bytes about to be overwritten by the forwarding address
        let mut temp = [0u8; 3];
        temp.copy_from_slice(&pool.data[upto - 3..upto]);

        // Write forwarding address (LE i32) at end of old slice
        let addr_bytes = (new_upto as i32).to_le_bytes();
        pool.data[upto - 3..upto + 1].copy_from_slice(&addr_bytes);

        // Copy the 3 saved bytes to the start of the new slice
        pool.data[new_upto..new_upto + 3].copy_from_slice(&temp);

        // Write new level marker at end of new slice
        pool.data[new_upto + new_size - 1] = 0x10 | new_level as u8;

        // Data starts after the 3 copied bytes; usable length excludes level marker
        (new_upto + 3, new_size - 3)
    }

    /// Convenience: grow a slice and return just the new data start offset.
    pub fn alloc_slice(pool: &mut ByteBlockPool, upto: usize) -> usize {
        Self::alloc_known_size_slice(pool, upto).0
    }
}

// ---------------------------------------------------------------------------
// ByteSliceReader
// ---------------------------------------------------------------------------

/// Read cursor that follows the forwarding chain of byte slices.
///
/// Borrows the [`ByteBlockPool`] immutably for its lifetime. Created after
/// all writes are complete.
pub struct ByteSliceReader<'a> {
    pool: &'a ByteBlockPool,
    /// Current global read position.
    upto: usize,
    /// Global limit (exclusive) — data bytes only, excludes forwarding address.
    limit: usize,
    /// Current slice level (used to determine next level on forwarding).
    level: usize,
    /// Global end offset of all data across all slices.
    end_index: usize,
}

impl<'a> ByteSliceReader<'a> {
    /// Create a reader over slices starting at `start_index` (global offset of
    /// first slice) through `end_index` (global offset of last written byte + 1).
    pub fn new(pool: &'a ByteBlockPool, start_index: usize, end_index: usize) -> Self {
        debug_assert!(end_index >= start_index);

        let limit = if start_index + FIRST_LEVEL_SIZE >= end_index {
            // Single slice — limit is the end
            end_index
        } else {
            // Multiple slices — first slice data ends 4 bytes before the end
            // (last 4 bytes are forwarding address + level marker)
            start_index + FIRST_LEVEL_SIZE - 4
        };

        Self {
            pool,
            upto: start_index,
            limit,
            level: 0,
            end_index,
        }
    }

    /// Returns true if all bytes have been read.
    pub fn eof(&self) -> bool {
        self.upto == self.end_index
    }

    /// Read `dest.len()` bytes into `dest`.
    pub fn read_bytes(&mut self, dest: &mut [u8]) {
        let mut offset = 0;
        let mut remaining = dest.len();
        while remaining > 0 {
            let available = self.limit - self.upto;
            if available < remaining {
                dest[offset..offset + available]
                    .copy_from_slice(&self.pool.data[self.upto..self.upto + available]);
                offset += available;
                remaining -= available;
                self.next_slice();
            } else {
                dest[offset..offset + remaining]
                    .copy_from_slice(&self.pool.data[self.upto..self.upto + remaining]);
                self.upto += remaining;
                break;
            }
        }
    }

    /// Reads a variable-length integer from the current position.
    pub fn read_vint(&mut self) -> io::Result<i32> {
        varint::read_vint(self)
    }

    /// Follow the forwarding address to the next slice.
    fn next_slice(&mut self) {
        // Read LE i32 forwarding address from the limit position
        let bytes: [u8; 4] = self.pool.data[self.limit..self.limit + 4]
            .try_into()
            .unwrap();
        let next_index = i32::from_le_bytes(bytes) as usize;

        self.level = NEXT_LEVEL_ARRAY[self.level];
        let new_size = LEVEL_SIZE_ARRAY[self.level];

        self.upto = next_index;

        if next_index + new_size >= self.end_index {
            // Final slice
            self.limit = self.end_index;
        } else {
            // Not final — reserve 4 bytes for forwarding address
            self.limit = self.upto + new_size - 4;
        }
    }
}

impl io::Read for ByteSliceReader<'_> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.eof() {
            return Ok(0);
        }
        let len = buf.len().min(self.end_index - self.upto);
        if len == 0 {
            return Ok(0);
        }
        self.read_bytes(&mut buf[..len]);
        Ok(len)
    }
}

#[cfg(test)]
mod tests {
    use std::io::Read;

    use super::*;
    use assertables::*;

    #[test]
    fn test_alloc_and_read() {
        let mut pool = ByteBlockPool::new(1024);

        let offset = pool.alloc(10);
        assert_eq!(offset, 0);
        assert_eq!(pool.data.len(), 10);

        pool.data[0] = 0xAB;
        pool.data[9] = 0xCD;
        assert_eq!(pool.data[0], 0xAB);
        assert_eq!(pool.data[9], 0xCD);
    }

    #[test]
    fn test_append_and_read_roundtrip() {
        let mut pool = ByteBlockPool::new(1024);

        let data: Vec<u8> = (0..=255).collect();
        let offset = pool.data.len();
        pool.data.extend_from_slice(&data);

        assert_eq!(&pool.data[offset..offset + 256], &data[..]);
    }

    #[test]
    fn test_large_data() {
        let mut pool = ByteBlockPool::new(1024);

        let total = 100_000;
        let offset = pool.alloc(total);

        for i in 0..total {
            pool.data[offset + i] = (i % 256) as u8;
        }
        for i in 0..total {
            assert_eq!(pool.data[offset + i], (i % 256) as u8);
        }
    }

    #[test]
    fn test_reset_preserves_capacity() {
        let mut pool = ByteBlockPool::new(1024);
        pool.alloc(50_000);
        let cap_before = pool.data.capacity();

        pool.reset();

        assert_eq!(pool.data.len(), 0);
        assert_eq!(pool.data.capacity(), cap_before);
    }

    #[test]
    fn test_new_slice_writes_level_marker() {
        let mut pool = ByteBlockPool::new(1024);

        let offset = ByteSlicePool::new_slice(&mut pool, FIRST_LEVEL_SIZE);
        assert_eq!(offset, 0);
        // Level marker at end of slice: 0x10 = level 0
        assert_eq!(pool.data[FIRST_LEVEL_SIZE - 1], 0x10);
    }

    #[test]
    fn test_level_progression() {
        let mut pool = ByteBlockPool::new(1024);

        let offset = ByteSlicePool::new_slice(&mut pool, FIRST_LEVEL_SIZE);

        // Write 1 data byte (first slice has 1 usable byte before marker)
        pool.data[offset] = 0x42;

        // Now grow: upto points to the level marker
        let (data_start, usable) =
            ByteSlicePool::alloc_known_size_slice(&mut pool, offset + FIRST_LEVEL_SIZE - 1);

        // Level 1 has size 14, usable = 14 - 3 = 11
        assert_eq!(usable, 11);
        assert_gt!(data_start, 0);
    }

    #[test]
    fn test_forwarding_chain() {
        let mut pool = ByteBlockPool::new(4096);

        let _offset = ByteSlicePool::new_slice(&mut pool, FIRST_LEVEL_SIZE);

        let mut upto = FIRST_LEVEL_SIZE - 1; // level marker position

        for (expected_level, &new_size) in LEVEL_SIZE_ARRAY.iter().enumerate().skip(1).take(5) {
            let new_upto = ByteSlicePool::alloc_slice(&mut pool, upto);
            // The new slice should have the correct level marker at its end
            let marker_pos = new_upto + new_size - 3 - 1;
            assert_eq!(pool.data[marker_pos] & 0x0F, expected_level as u8);
            upto = marker_pos;
        }
    }

    #[test]
    fn test_reader_via_io_read() {
        let data: Vec<u8> = (0..150).map(|i| (i * 7 + 3) as u8).collect();

        let mut pool = ByteBlockPool::new(4096);

        let upto_start = ByteSlicePool::new_slice(&mut pool, FIRST_LEVEL_SIZE);
        let mut upto = upto_start;

        for &b in &data {
            if (pool.data[upto] & 16) != 0 {
                upto = ByteSlicePool::alloc_slice(&mut pool, upto);
            }
            pool.data[upto] = b;
            upto += 1;
        }

        let end = upto;

        // Read all via io::Read
        let mut reader = ByteSliceReader::new(&pool, upto_start, end);
        let mut result = Vec::new();
        reader.read_to_end(&mut result).unwrap();
        assert_eq!(data, result);
    }

    #[test]
    fn test_reader_eof_empty_data() {
        let mut pool = ByteBlockPool::new(1024);

        let upto_start = ByteSlicePool::new_slice(&mut pool, FIRST_LEVEL_SIZE);
        let end = upto_start;

        let reader = ByteSliceReader::new(&pool, upto_start, end);
        assert!(reader.eof());
    }

    #[test]
    fn test_multiple_allocs() {
        let mut pool = ByteBlockPool::new(32);
        let a = pool.alloc(10);
        let b = pool.alloc(20);
        assert_eq!(a, 0);
        assert_eq!(b, 10);
        assert_eq!(pool.data.len(), 30);
    }
}
