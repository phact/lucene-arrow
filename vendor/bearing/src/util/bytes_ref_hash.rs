// SPDX-License-Identifier: Apache-2.0

//! Hash map for deduplicating byte sequences into compact integer IDs.
//!
//! Byte data is stored in a contiguous pool (length-prefixed), and the hash
//! table maps byte sequences to sequential integer IDs using open addressing
//! with linear probing. High bits of the hash code are stored alongside the
//! ID for fast rejection during probing.

use std::fmt;

use crate::util::byte_block_pool::ByteBlockPool;

/// Hash map that interns byte sequences into an external [`ByteBlockPool`] and
/// assigns sequential integer IDs.
///
/// Each unique byte sequence gets an ID starting at 0. The bytes are stored
/// length-prefixed in the pool (1 byte for lengths < 128, 2 bytes otherwise).
/// The pool is owned externally and passed to methods that need it.
#[derive(mem_dbg::MemSize)]
pub struct BytesRefHash {
    bytes_start: Vec<i32>,
    hash_size: usize,
    hash_half_size: usize,
    hash_mask: i32,
    high_mask: i32,
    count: usize,
    last_count: i32,
    ids: Vec<i32>,
}

impl BytesRefHash {
    /// Creates a new `BytesRefHash` with the given initial capacity.
    ///
    /// The pool is owned externally and passed to methods that need it.
    ///
    /// # Panics
    /// Panics if `capacity` is not a positive power of two.
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "capacity must be greater than 0");
        assert!(
            capacity.is_power_of_two(),
            "capacity must be a power of two, got {capacity}"
        );
        let hash_size = capacity;
        let hash_mask = (hash_size - 1) as i32;
        Self {
            bytes_start: Vec::new(),
            hash_size,
            hash_half_size: hash_size >> 1,
            hash_mask,
            high_mask: !hash_mask,
            count: 0,
            last_count: -1,
            ids: vec![-1i32; hash_size],
        }
    }

    /// Returns the number of unique byte sequences in this hash.
    pub fn size(&self) -> usize {
        self.count
    }

    /// Returns the bytes for the given ID.
    ///
    /// # Panics
    /// Panics if `bytes_id` is out of range.
    pub fn get<'a>(&self, pool: &'a ByteBlockPool, bytes_id: usize) -> &'a [u8] {
        let start = self.bytes_start[bytes_id] as usize;
        Self::read_bytes_at_pool(pool, start)
    }

    /// Adds a byte sequence. Returns the new ID (>= 0) if the bytes are new,
    /// or `-(existing_id) - 1` if already present.
    pub fn add(&mut self, pool: &mut ByteBlockPool, bytes: &[u8]) -> i32 {
        let hashcode = do_hash(bytes);
        let hash_pos = self.find_hash(pool, bytes, hashcode);
        let e = self.ids[hash_pos];

        if e == -1 {
            // new entry
            if self.count >= self.bytes_start.len() {
                self.bytes_start
                    .resize(grow_size(self.bytes_start.len().max(1)), 0);
            }
            self.bytes_start[self.count] = Self::add_bytes_to_pool(pool, bytes);
            let new_id = self.count as i32;
            self.count += 1;
            assert!(self.ids[hash_pos] == -1);
            self.ids[hash_pos] = new_id | (hashcode & self.high_mask);

            if self.count == self.hash_half_size {
                self.rehash(Some(pool), 2 * self.hash_size, true);
            }
            return new_id;
        }
        let existing_id = e & self.hash_mask;
        -(existing_id + 1)
    }

    /// Returns the `bytes_start` offset for the given ID.
    pub fn byte_start(&self, bytes_id: usize) -> i32 {
        assert!(bytes_id < self.count);
        self.bytes_start[bytes_id]
    }

    /// Compacts the IDs array and returns a slice of term IDs in arbitrary order.
    /// Valid IDs are at indices `0..size()`.
    ///
    /// This is a destructive operation. [`clear`](Self::clear) must be called
    /// before reusing.
    pub fn compact(&mut self) -> &[i32] {
        let mut upto = 0;
        for i in 0..self.hash_size {
            if self.ids[i] != -1 {
                self.ids[upto] = self.ids[i] & self.hash_mask;
                if upto < i {
                    self.ids[i] = -1;
                }
                upto += 1;
            }
        }
        assert_eq!(upto, self.count);
        self.last_count = self.count as i32;
        &self.ids[..self.count]
    }

    /// Returns term IDs sorted by the referenced byte values (lexicographic).
    ///
    /// This is a destructive operation. [`clear`](Self::clear) must be called
    /// before reusing.
    pub fn sort(&mut self, pool: &ByteBlockPool) -> Vec<i32> {
        let compact = self.compact().to_vec();
        let mut sorted = compact;
        sorted.sort_by(|&a, &b| {
            let a_bytes = self.get(pool, a as usize);
            let b_bytes = self.get(pool, b as usize);
            a_bytes.cmp(b_bytes)
        });
        sorted
    }

    /// Clears the hash for reuse.
    pub fn clear(&mut self) {
        self.last_count = self.count as i32;
        self.count = 0;
        self.bytes_start.clear();
        if self.last_count != -1 && self.shrink(self.last_count as usize) {
            return;
        }
        self.ids.fill(-1);
    }

    // --- Internal methods ---

    fn find_hash(&self, pool: &ByteBlockPool, bytes: &[u8], hashcode: i32) -> usize {
        let mut code = hashcode;
        let mut hash_pos = (code & self.hash_mask) as usize;
        let mut e = self.ids[hash_pos];
        let high_bits = hashcode & self.high_mask;

        while e != -1
            && ((e & self.high_mask) != high_bits
                || !Self::pool_bytes_equal(
                    pool,
                    self.bytes_start[(e & self.hash_mask) as usize],
                    bytes,
                ))
        {
            code = code.wrapping_add(1);
            hash_pos = (code & self.hash_mask) as usize;
            e = self.ids[hash_pos];
        }

        hash_pos
    }

    fn rehash(&mut self, pool: Option<&ByteBlockPool>, new_size: usize, hash_on_data: bool) {
        let new_mask = (new_size - 1) as i32;
        let new_high_mask = !new_mask;
        let mut new_hash = vec![-1i32; new_size];

        for i in 0..self.hash_size {
            let e0 = self.ids[i];
            if e0 != -1 {
                let e0 = e0 & self.hash_mask;
                let (hashcode, mut code);
                if hash_on_data {
                    let start = self.bytes_start[e0 as usize] as usize;
                    let bytes = Self::read_bytes_at_pool(
                        pool.expect("pool required when hash_on_data is true"),
                        start,
                    );
                    hashcode = do_hash(bytes);
                    code = hashcode;
                } else {
                    code = self.bytes_start[e0 as usize];
                    hashcode = 0;
                }

                let mut hash_pos = (code & new_mask) as usize;
                assert!(hash_pos < new_size);

                while new_hash[hash_pos] != -1 {
                    code = code.wrapping_add(1);
                    hash_pos = (code & new_mask) as usize;
                }

                new_hash[hash_pos] = e0 | (hashcode & new_high_mask);
            }
        }

        self.hash_mask = new_mask;
        self.high_mask = new_high_mask;
        self.ids = new_hash;
        self.hash_size = new_size;
        self.hash_half_size = new_size / 2;
    }

    fn shrink(&mut self, target_size: usize) -> bool {
        let mut new_size = self.hash_size;
        while new_size >= 8 && new_size / 4 > target_size {
            new_size /= 2;
        }
        if new_size != self.hash_size {
            self.hash_size = new_size;
            self.ids = vec![-1i32; new_size];
            self.hash_half_size = new_size / 2;
            self.hash_mask = (new_size - 1) as i32;
            self.high_mask = !self.hash_mask;
            true
        } else {
            false
        }
    }

    /// Adds bytes to the pool with length prefix. Returns the start offset.
    fn add_bytes_to_pool(pool: &mut ByteBlockPool, bytes: &[u8]) -> i32 {
        let length = bytes.len();
        let len_bytes = if length < 128 { 1 } else { 2 };
        let start = pool.alloc(len_bytes + length);

        if length < 128 {
            pool.data[start] = length as u8;
            pool.data[start + 1..start + 1 + length].copy_from_slice(bytes);
        } else {
            let encoded = (length as u16) | 0x8000;
            pool.data[start] = (encoded >> 8) as u8;
            pool.data[start + 1] = encoded as u8;
            pool.data[start + 2..start + 2 + length].copy_from_slice(bytes);
        }

        start as i32
    }

    /// Reads the length-prefixed bytes at the given pool offset.
    pub fn read_bytes_at_pool(pool: &ByteBlockPool, start: usize) -> &[u8] {
        if (pool.data[start] & 0x80) == 0 {
            let length = pool.data[start] as usize;
            &pool.data[start + 1..start + 1 + length]
        } else {
            let length =
                (((pool.data[start] as usize) << 8) | (pool.data[start + 1] as usize)) & 0x7FFF;
            &pool.data[start + 2..start + 2 + length]
        }
    }

    /// Compares the bytes at `start` in the pool with `other`.
    fn pool_bytes_equal(pool: &ByteBlockPool, start: i32, other: &[u8]) -> bool {
        Self::read_bytes_at_pool(pool, start as usize) == other
    }
}

impl BytesRefHash {
    /// Adds an arbitrary int offset instead of a byte sequence.
    ///
    /// Used by term vectors, which don't redundantly store term bytes — they
    /// reference the bytes already stored by the postings `BytesRefHash`. The
    /// offset is used as both the stored value (`bytes_start[id] = offset`) and
    /// the hash code. Rehash uses `hash_on_data=false` for this mode.
    pub fn add_by_pool_offset(&mut self, offset: i32) -> i32 {
        let mut code = offset;
        let mut hash_pos = (offset & self.hash_mask) as usize;
        let mut e = self.ids[hash_pos];

        // Conflict; use linear probe to find an open slot (see LUCENE-5604):
        while e != -1 && self.bytes_start[e as usize] != offset {
            code = code.wrapping_add(1);
            hash_pos = (code & self.hash_mask) as usize;
            e = self.ids[hash_pos];
        }
        if e == -1 {
            // new entry
            if self.count >= self.bytes_start.len() {
                self.bytes_start
                    .resize(grow_size(self.bytes_start.len().max(1)), 0);
            }
            e = self.count as i32;
            self.count += 1;
            self.bytes_start[e as usize] = offset;
            assert!(self.ids[hash_pos] == -1);
            self.ids[hash_pos] = e;

            if self.count == self.hash_half_size {
                self.rehash(None, 2 * self.hash_size, false);
            }
            return e;
        }
        -(e + 1)
    }
}

#[cfg(test)]
impl BytesRefHash {
    /// Looks up a byte sequence. Returns the ID if found, or -1 if not present.
    pub fn find(&self, pool: &ByteBlockPool, bytes: &[u8]) -> i32 {
        let hashcode = do_hash(bytes);
        let id = self.ids[self.find_hash(pool, bytes, hashcode)];
        if id == -1 { -1 } else { id & self.hash_mask }
    }
}

impl fmt::Debug for BytesRefHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BytesRefHash")
            .field("count", &self.count)
            .field("hash_size", &self.hash_size)
            .finish()
    }
}

/// Computes the hash of a byte slice using MurmurHash3 x86 32-bit.
pub fn do_hash(bytes: &[u8]) -> i32 {
    murmurhash3_x86_32(bytes, GOOD_FAST_HASH_SEED)
}

/// Seed for hashing. Fixed value for deterministic behavior.
const GOOD_FAST_HASH_SEED: i32 = 0;

/// MurmurHash3 x86 32-bit implementation.
///
/// Matches Java's `StringHelper.murmurhash3_x86_32`.
fn murmurhash3_x86_32(data: &[u8], seed: i32) -> i32 {
    let c1: i32 = 0xcc9e2d51_u32 as i32;
    let c2: i32 = 0x1b873593_u32 as i32;

    let mut h1 = seed;
    let len = data.len();
    let rounded_end = len & !3; // round down to 4 byte block

    let mut i = 0;
    while i < rounded_end {
        // little endian load order
        let k1_raw = i32::from_le_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]]);
        let mut k1 = k1_raw;
        k1 = k1.wrapping_mul(c1);
        k1 = k1.rotate_left(15);
        k1 = k1.wrapping_mul(c2);

        h1 ^= k1;
        h1 = h1.rotate_left(13);
        h1 = h1.wrapping_mul(5).wrapping_add(0xe6546b64_u32 as i32);
        i += 4;
    }

    // tail
    let mut k1: i32;
    match len & 3 {
        3 => {
            k1 = (data[rounded_end + 2] as i32 & 0xff) << 16;
            k1 |= (data[rounded_end + 1] as i32 & 0xff) << 8;
            k1 |= data[rounded_end] as i32 & 0xff;
            k1 = k1.wrapping_mul(c1);
            k1 = k1.rotate_left(15);
            k1 = k1.wrapping_mul(c2);
            h1 ^= k1;
        }
        2 => {
            k1 = (data[rounded_end + 1] as i32 & 0xff) << 8;
            k1 |= data[rounded_end] as i32 & 0xff;
            k1 = k1.wrapping_mul(c1);
            k1 = k1.rotate_left(15);
            k1 = k1.wrapping_mul(c2);
            h1 ^= k1;
        }
        1 => {
            k1 = data[rounded_end] as i32 & 0xff;
            k1 = k1.wrapping_mul(c1);
            k1 = k1.rotate_left(15);
            k1 = k1.wrapping_mul(c2);
            h1 ^= k1;
        }
        _ => {}
    }

    // finalization
    h1 ^= len as i32;

    // fmix
    h1 ^= (h1 as u32 >> 16) as i32;
    h1 = h1.wrapping_mul(0x85ebca6b_u32 as i32);
    h1 ^= (h1 as u32 >> 13) as i32;
    h1 = h1.wrapping_mul(0xc2b2ae35_u32 as i32);
    h1 ^= (h1 as u32 >> 16) as i32;

    h1
}

/// Computes a grow size (approximately 1.5x).
fn grow_size(current: usize) -> usize {
    current + (current >> 1).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use assertables::*;

    fn make_pool() -> ByteBlockPool {
        ByteBlockPool::new(32 * 1024)
    }

    #[test]
    fn test_add_and_dedup() {
        let mut pool = make_pool();
        let mut hash = BytesRefHash::new(16);
        let id0 = hash.add(&mut pool, b"hello");
        assert_eq!(id0, 0);
        let id1 = hash.add(&mut pool, b"world");
        assert_eq!(id1, 1);

        // Adding same bytes returns negative
        let dup = hash.add(&mut pool, b"hello");
        assert_eq!(dup, -1); // -(id)-1 = -(0)-1 = -1

        let dup2 = hash.add(&mut pool, b"world");
        assert_eq!(dup2, -2); // -(1)-1 = -2
    }

    #[test]
    fn test_get_bytes() {
        let mut pool = make_pool();
        let mut hash = BytesRefHash::new(16);
        hash.add(&mut pool, b"foo");
        hash.add(&mut pool, b"bar");
        hash.add(&mut pool, b"baz");

        assert_eq!(hash.get(&pool, 0), b"foo");
        assert_eq!(hash.get(&pool, 1), b"bar");
        assert_eq!(hash.get(&pool, 2), b"baz");
    }

    #[test]
    fn test_size() {
        let mut pool = make_pool();
        let mut hash = BytesRefHash::new(16);
        assert_eq!(hash.size(), 0);
        hash.add(&mut pool, b"a");
        assert_eq!(hash.size(), 1);
        hash.add(&mut pool, b"b");
        assert_eq!(hash.size(), 2);
        // Duplicate doesn't increase size
        hash.add(&mut pool, b"a");
        assert_eq!(hash.size(), 2);
    }

    #[test]
    fn test_many_entries_triggers_rehash() {
        let mut pool = make_pool();
        let mut hash = BytesRefHash::new(16);
        // Add enough entries to trigger multiple rehashes
        for i in 0i32..1000 {
            let key = format!("key_{i:04}");
            let id = hash.add(&mut pool, key.as_bytes());
            assert_eq!(id, i);
        }
        assert_eq!(hash.size(), 1000);

        // Verify all can be found via duplicate add
        for i in 0i32..1000 {
            let key = format!("key_{i:04}");
            let dup = hash.add(&mut pool, key.as_bytes());
            assert_eq!(dup, -(i + 1));
        }
    }

    #[test]
    fn test_sort_order() {
        let mut pool = make_pool();
        let mut hash = BytesRefHash::new(16);
        hash.add(&mut pool, b"cherry");
        hash.add(&mut pool, b"apple");
        hash.add(&mut pool, b"banana");

        let sorted = hash.sort(&pool);
        assert_len_eq_x!(&sorted, 3);

        // Verify sorted order by checking the bytes
        assert_eq!(hash.get(&pool, sorted[0] as usize), b"apple");
        assert_eq!(hash.get(&pool, sorted[1] as usize), b"banana");
        assert_eq!(hash.get(&pool, sorted[2] as usize), b"cherry");
    }

    #[test]
    fn test_compact() {
        let mut pool = make_pool();
        let mut hash = BytesRefHash::new(16);
        hash.add(&mut pool, b"x");
        hash.add(&mut pool, b"y");
        hash.add(&mut pool, b"z");

        let compact = hash.compact().to_vec();
        assert_len_eq_x!(&compact, 3);

        // Should contain IDs 0, 1, 2 in some order
        let mut sorted = compact;
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2]);
    }

    #[test]
    fn test_clear_and_reuse() {
        let mut pool = make_pool();
        let mut hash = BytesRefHash::new(16);
        hash.add(&mut pool, b"first");
        hash.add(&mut pool, b"second");
        assert_eq!(hash.size(), 2);

        hash.clear();
        assert_eq!(hash.size(), 0);

        // Can add new entries after clear (re-adding "first" returns 0, not duplicate)
        let id = hash.add(&mut pool, b"first");
        assert_eq!(id, 0);
        assert_eq!(hash.size(), 1);
    }

    #[test]
    fn test_empty_bytes() {
        let mut pool = make_pool();
        let mut hash = BytesRefHash::new(16);
        let id = hash.add(&mut pool, b"");
        assert_eq!(id, 0);
        assert_eq!(hash.get(&pool, 0), b"");
    }

    #[test]
    fn test_long_bytes() {
        let mut pool = make_pool();
        let mut hash = BytesRefHash::new(16);
        // Test bytes longer than 127 (requires 2-byte length encoding)
        let long_bytes = vec![b'x'; 200];
        let id = hash.add(&mut pool, &long_bytes);
        assert_eq!(id, 0);
        assert_eq!(hash.get(&pool, 0), long_bytes.as_slice());
    }

    #[test]
    fn test_murmurhash3_empty() {
        let h = murmurhash3_x86_32(b"", 0);
        // Known value for empty input with seed 0
        assert_eq!(h, 0);
    }

    #[test]
    fn test_murmurhash3_deterministic() {
        let h1 = murmurhash3_x86_32(b"test", 42);
        let h2 = murmurhash3_x86_32(b"test", 42);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_murmurhash3_different_inputs() {
        let h1 = murmurhash3_x86_32(b"abc", 0);
        let h2 = murmurhash3_x86_32(b"abd", 0);
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_byte_start() {
        let mut pool = make_pool();
        let mut hash = BytesRefHash::new(16);
        hash.add(&mut pool, b"hello");
        let start = hash.byte_start(0);
        assert_ge!(start, 0);
    }
}
