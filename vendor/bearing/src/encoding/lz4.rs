// SPDX-License-Identifier: Apache-2.0
//! LZ4 block compression for stored fields and blocktree suffix data.

use std::io;
use std::io::Read;

const MIN_MATCH: usize = 4;
const LAST_LITERALS: usize = 5;
const MEMORY_USAGE: usize = 14;
const MAX_DISTANCE: usize = 1 << 16; // 65536
const HASH_LOG_HC: usize = 15;
const HASH_TABLE_SIZE_HC: usize = 1 << HASH_LOG_HC;
const MAX_ATTEMPTS: usize = 256;
const MASK: usize = MAX_DISTANCE - 1;

/// Compute hash log for the fast compressor based on input length.
/// Matches Java's `FastCompressionHashTable.reset()`.
fn fast_hash_log(len: usize) -> usize {
    let bits_per_offset = if len.saturating_sub(LAST_LITERALS) < (1 << 16) {
        16_usize
    } else {
        32_usize
    };
    let bits_per_offset_log = 32 - ((bits_per_offset - 1) as u32).leading_zeros() as usize;
    MEMORY_USAGE + 3 - bits_per_offset_log
}

/// Reusable hash table for the fast LZ4 compressor.
///
/// Matches Java's `LZ4.FastCompressionHashTable`: the table is sized dynamically
/// based on input length, and is intentionally **not** cleared between calls.
/// Stale entries are harmless because the compression loop validates matches.
/// Reusing across calls within the same context (e.g., multiple blocks of a terms
/// dictionary) matches Java's compression behavior byte-for-byte.
pub struct FastHashTable {
    table: Vec<u32>,
    hash_log: usize,
}

impl Default for FastHashTable {
    fn default() -> Self {
        Self::new()
    }
}

impl FastHashTable {
    /// Create a new empty hash table.
    pub fn new() -> Self {
        Self {
            table: Vec::new(),
            hash_log: 0,
        }
    }

    /// Reset for a new compression of `len` bytes. Grows the table if needed
    /// but does NOT clear existing entries, matching Java's behavior.
    fn reset(&mut self, len: usize) {
        let hash_log = fast_hash_log(len);
        let required = 1 << hash_log;
        if self.table.len() < required {
            self.table = vec![0u32; required];
        }
        self.hash_log = hash_log;
    }

    /// Pre-populate with dictionary positions.
    fn init_dictionary(&mut self, buffer: &[u8], dict_len: usize) {
        for i in 0..dict_len {
            if i + 4 <= buffer.len() {
                let h = hash4(buffer, i, self.hash_log);
                self.table[h] = i as u32;
            }
        }
    }
}

/// LZ4 compress a block of data.
/// Returns the compressed bytes.
pub fn compress(input: &[u8]) -> Vec<u8> {
    let mut ht = FastHashTable::new();
    compress_reuse(input, &mut ht)
}

/// LZ4 compress a block of data, reusing a hash table across calls.
pub fn compress_reuse(input: &[u8], ht: &mut FastHashTable) -> Vec<u8> {
    if input.is_empty() {
        return Vec::new();
    }
    if input.len() < MIN_MATCH + LAST_LITERALS {
        return encode_literals_only(input);
    }

    ht.reset(input.len());
    compress_inner(input, 0, ht.hash_log, &mut ht.table)
}

/// Shared compression loop used by both `compress` and `compress_with_dictionary`.
/// `buffer` is the full input (possibly prefixed by dictionary bytes).
/// `data_start` is the byte offset where actual data begins (0 for plain compress,
/// dict_len for dictionary compress).
/// `hash_table` must be pre-sized to `1 << hash_log` and pre-populated if using a dictionary.
fn compress_inner(
    buffer: &[u8],
    data_start: usize,
    hash_log: usize,
    hash_table: &mut [u32],
) -> Vec<u8> {
    let data_len = buffer.len() - data_start;
    let mut output = Vec::with_capacity(data_len);

    let mut anchor = data_start;
    let mut ip = data_start;
    let limit = buffer.len() - LAST_LITERALS;
    let match_limit = limit - MIN_MATCH;

    while ip < match_limit {
        let h = hash4(buffer, ip, hash_log);
        let ref_pos = hash_table[h] as usize;
        hash_table[h] = ip as u32;

        if ref_pos < ip
            && ip - ref_pos < MAX_DISTANCE
            && buffer[ref_pos] == buffer[ip]
            && buffer[ref_pos + 1] == buffer[ip + 1]
            && buffer[ref_pos + 2] == buffer[ip + 2]
            && buffer[ref_pos + 3] == buffer[ip + 3]
        {
            let match_distance = ip - ref_pos;

            // Extend match forward — cap at `limit` to preserve LAST_LITERALS bytes
            let mut match_len = MIN_MATCH;
            while ip + match_len < limit && buffer[ref_pos + match_len] == buffer[ip + match_len] {
                match_len += 1;
            }

            let literal_len = ip - anchor;
            encode_sequence(
                &mut output,
                buffer,
                anchor,
                literal_len,
                match_distance,
                match_len,
            );

            anchor = ip + match_len;
            ip = anchor;
        } else {
            ip += 1;
        }
    }

    // Encode remaining literals
    let remaining = buffer.len() - anchor;
    encode_last_literals(&mut output, buffer, anchor, remaining);

    output
}

/// Hash 4 bytes at position `pos` using the given hash log.
fn hash4(data: &[u8], pos: usize, hash_log: usize) -> usize {
    let v = u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
    ((v.wrapping_mul(2654435761)) >> (32 - hash_log)) as usize
}

/// Encode a sequence: literal_len literals + match(distance, match_len).
fn encode_sequence(
    output: &mut Vec<u8>,
    input: &[u8],
    literal_start: usize,
    literal_len: usize,
    match_distance: usize,
    match_len: usize,
) {
    // Token byte: [literal_len:4][match_len-4:4]
    let lit_code = literal_len.min(15);
    let match_code = (match_len - MIN_MATCH).min(15);
    output.push(((lit_code << 4) | match_code) as u8);

    // Extended literal length
    if literal_len >= 15 {
        let mut remaining = literal_len - 15;
        while remaining >= 255 {
            output.push(255);
            remaining -= 255;
        }
        output.push(remaining as u8);
    }

    // Literal bytes
    output.extend_from_slice(&input[literal_start..literal_start + literal_len]);

    // Match distance (LE u16)
    output.push(match_distance as u8);
    output.push((match_distance >> 8) as u8);

    // Extended match length
    if match_len - MIN_MATCH >= 15 {
        let mut remaining = match_len - MIN_MATCH - 15;
        while remaining >= 255 {
            output.push(255);
            remaining -= 255;
        }
        output.push(remaining as u8);
    }
}

/// Encode remaining literals at the end of the block (no match follows).
fn encode_last_literals(output: &mut Vec<u8>, input: &[u8], start: usize, len: usize) {
    let lit_code = len.min(15);
    output.push((lit_code << 4) as u8);

    if len >= 15 {
        let mut remaining = len - 15;
        while remaining >= 255 {
            output.push(255);
            remaining -= 255;
        }
        output.push(remaining as u8);
    }

    output.extend_from_slice(&input[start..start + len]);
}

/// Encode input as all-literals (no matches possible).
fn encode_literals_only(input: &[u8]) -> Vec<u8> {
    let mut output = Vec::with_capacity(input.len() + 10);
    encode_last_literals(&mut output, input, 0, input.len());
    output
}

/// LZ4 compress with a preset dictionary.
/// `buffer[0..dict_len]` is the dictionary, `buffer[dict_len..]` is the data to compress.
/// Returns compressed bytes for the data portion only.
#[cfg(test)]
pub fn compress_with_dictionary(buffer: &[u8], dict_len: usize) -> Vec<u8> {
    let mut ht = FastHashTable::new();
    compress_with_dictionary_reuse(buffer, dict_len, &mut ht)
}

/// LZ4 compress with a preset dictionary, reusing a hash table across calls.
/// `buffer[0..dict_len]` is the dictionary, `buffer[dict_len..]` is the data to compress.
/// Returns compressed bytes for the data portion only.
pub fn compress_with_dictionary_reuse(
    buffer: &[u8],
    dict_len: usize,
    ht: &mut FastHashTable,
) -> Vec<u8> {
    let data_len = buffer.len() - dict_len;

    if data_len == 0 {
        return Vec::new();
    }
    if data_len < MIN_MATCH + LAST_LITERALS {
        return encode_literals_only(&buffer[dict_len..]);
    }

    ht.reset(buffer.len());
    ht.init_dictionary(buffer, dict_len);

    compress_inner(buffer, dict_len, ht.hash_log, &mut ht.table)
}

/// Hash function for the high-compression hash table (HASH_LOG_HC = 15 bits).
fn hash_hc(v: u32) -> usize {
    (v.wrapping_mul(2654435761u32) >> (32 - HASH_LOG_HC)) as usize
}

/// Read a native-endian 32-bit int from a byte slice at the given position.
fn read_int(data: &[u8], pos: usize) -> u32 {
    u32::from_ne_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]])
}

/// Count common bytes between `data[o1..]` and `data[o2..]` up to `limit`.
fn common_bytes(data: &[u8], o1: usize, o2: usize, limit: usize) -> usize {
    let mut len = 0;
    while o1 + len < limit && o2 + len < limit && data[o1 + len] == data[o2 + len] {
        len += 1;
    }
    len
}

/// High-compression hash table for LZ4 that stores up to 256 occurrences of
/// 4-byte sequences within a 64KB window. Produces better compression than the
/// standard fast hash table at the cost of slower compression speed.
pub struct HighCompressionHashTable {
    hash_table: Vec<i32>,
    chain_table: Vec<u16>,
    base: usize,
    next: usize,
    end: usize,
}

impl Default for HighCompressionHashTable {
    fn default() -> Self {
        Self::new()
    }
}

impl HighCompressionHashTable {
    /// Create a new high-compression hash table.
    pub fn new() -> Self {
        Self {
            hash_table: vec![-1; HASH_TABLE_SIZE_HC],
            chain_table: vec![0xFFFF; MAX_DISTANCE],
            base: 0,
            next: 0,
            end: 0,
        }
    }

    /// Reset the hash table for compressing `data[off..off+len]`.
    fn reset(&mut self, off: usize, len: usize) {
        if self.end - self.base < self.chain_table.len() {
            // Previous compression was on < 64KB — only reset relevant chain entries
            let start_offset = self.base & MASK;
            let end_offset = if self.end == 0 {
                0
            } else {
                ((self.end - 1) & MASK) + 1
            };
            if start_offset < end_offset {
                self.chain_table[start_offset..end_offset].fill(0xFFFF);
            } else {
                self.chain_table[..end_offset].fill(0xFFFF);
                self.chain_table[start_offset..].fill(0xFFFF);
            }
        } else {
            self.hash_table.fill(-1);
            self.chain_table.fill(0xFFFF);
        }
        self.base = off;
        self.next = off;
        self.end = off + len;
    }

    /// Add a position to the hash table.
    fn add_hash(&mut self, data: &[u8], off: usize) {
        let v = read_int(data, off);
        let h = hash_hc(v);
        let mut delta = off as i64 - self.hash_table[h] as i64;
        if delta <= 0 || delta >= MAX_DISTANCE as i64 {
            delta = (MAX_DISTANCE - 1) as i64;
        }
        self.chain_table[off & MASK] = delta as u16;
        self.hash_table[h] = off as i32;
    }

    /// Find a match for the 4-byte sequence at `off`. Advances internal cursor.
    /// Returns the matched position or `None`.
    fn get(&mut self, data: &[u8], off: usize) -> Option<usize> {
        // Advance cursor, hashing skipped positions
        while self.next < off {
            self.add_hash(data, self.next);
            self.next += 1;
        }

        let v = read_int(data, off);
        let h = hash_hc(v);

        let mut attempts = 0;
        let mut ref_pos = self.hash_table[h];

        if ref_pos >= off as i32 {
            // Remainder from a previous call
            return None;
        }

        let min = if off >= MAX_DISTANCE - 1 {
            (off - MAX_DISTANCE + 1).max(self.base)
        } else {
            self.base
        };

        while ref_pos >= min as i32 && attempts < MAX_ATTEMPTS {
            if read_int(data, ref_pos as usize) == v {
                return Some(ref_pos as usize);
            }
            let delta = self.chain_table[ref_pos as usize & MASK] as usize;
            ref_pos -= delta as i32;
            // If delta was 0xFFFF (end of chain), ref_pos will go negative and loop ends
            attempts += 1;
        }
        None
    }

    /// Find a previous match in the chain from `off`.
    fn previous(&self, data: &[u8], off: usize, attempts: &mut usize) -> Option<usize> {
        let v = read_int(data, off);
        let mut ref_pos = off as i64 - (self.chain_table[off & MASK] as u64 & 0xFFFF) as i64;

        while ref_pos >= self.base as i64 && *attempts < MAX_ATTEMPTS {
            if read_int(data, ref_pos as usize) == v {
                return Some(ref_pos as usize);
            }
            let delta = self.chain_table[ref_pos as usize & MASK] as u64 & 0xFFFF;
            ref_pos -= delta as i64;
            *attempts += 1;
        }
        None
    }
}

/// LZ4 compress using the high-compression hash table.
///
/// Produces the same LZ4 block format as [`compress`] but finds better matches,
/// resulting in smaller output. The hash table is reusable across calls.
pub fn compress_high(input: &[u8], ht: &mut HighCompressionHashTable) -> Vec<u8> {
    if input.is_empty() {
        return Vec::new();
    }
    if input.len() < MIN_MATCH + LAST_LITERALS {
        return encode_literals_only(input);
    }

    let len = input.len();
    let mut output = Vec::with_capacity(len);

    ht.reset(0, len);

    let limit = len - LAST_LITERALS;
    let match_limit = limit - MIN_MATCH;
    let mut off = 0;
    let mut anchor = 0;

    'main: while off <= limit {
        // Find a match
        loop {
            if off >= match_limit {
                break 'main;
            }
            if let Some(ref_pos) = ht.get(input, off) {
                // Found a match — compute length
                let match_len =
                    MIN_MATCH + common_bytes(input, ref_pos + MIN_MATCH, off + MIN_MATCH, limit);

                // Try to find a better match via the chain
                let mut attempts = 0;
                let mut best_ref = ref_pos;
                let mut best_len = match_len;

                let mut r_opt = ht.previous(input, ref_pos, &mut attempts);
                let min = if off >= MAX_DISTANCE - 1 {
                    off - MAX_DISTANCE + 1
                } else {
                    0
                };
                while let Some(r) = r_opt {
                    if r < min {
                        break;
                    }
                    let r_match_len =
                        MIN_MATCH + common_bytes(input, r + MIN_MATCH, off + MIN_MATCH, limit);
                    if r_match_len > best_len {
                        best_ref = r;
                        best_len = r_match_len;
                    }
                    r_opt = ht.previous(input, r, &mut attempts);
                }

                // Encode the sequence
                let literal_len = off - anchor;
                let match_distance = off - best_ref;
                encode_sequence(
                    &mut output,
                    input,
                    anchor,
                    literal_len,
                    match_distance,
                    best_len,
                );
                off += best_len;
                anchor = off;
                break;
            }
            off += 1;
        }
    }

    // Last literals
    let remaining = len - anchor;
    encode_last_literals(&mut output, input, anchor, remaining);

    output
}

/// LZ4 decompress a block. `dest_len` is the expected uncompressed size.
pub fn decompress(compressed: &[u8], dest_len: usize) -> io::Result<Vec<u8>> {
    let mut output = Vec::with_capacity(dest_len);
    let mut ip = 0;

    while ip < compressed.len() && output.len() < dest_len {
        let token = compressed[ip] as usize;
        ip += 1;

        // Literal length
        let mut literal_len = token >> 4;
        if literal_len == 15 {
            loop {
                if ip >= compressed.len() {
                    return Err(io::Error::other("truncated literal length"));
                }
                let b = compressed[ip] as usize;
                ip += 1;
                literal_len += b;
                if b < 255 {
                    break;
                }
            }
        }

        // Copy literals
        if ip + literal_len > compressed.len() {
            return Err(io::Error::other("literal bytes out of bounds"));
        }
        output.extend_from_slice(&compressed[ip..ip + literal_len]);
        ip += literal_len;

        if output.len() >= dest_len {
            break;
        }

        // Match distance (LE u16)
        if ip + 2 > compressed.len() {
            return Err(io::Error::other("truncated match distance"));
        }
        let distance = compressed[ip] as usize | ((compressed[ip + 1] as usize) << 8);
        ip += 2;

        if distance == 0 || distance > output.len() {
            return Err(io::Error::other("invalid match distance"));
        }

        // Match length
        let mut match_len = (token & 0x0F) + MIN_MATCH;
        if (token & 0x0F) == 15 {
            loop {
                if ip >= compressed.len() {
                    return Err(io::Error::other("truncated match length"));
                }
                let b = compressed[ip] as usize;
                ip += 1;
                match_len += b;
                if b < 255 {
                    break;
                }
            }
        }

        // Copy match (may overlap)
        let match_start = output.len() - distance;
        for i in 0..match_len {
            let b = output[match_start + i];
            output.push(b);
        }
    }

    Ok(output)
}

/// LZ4 decompress with a preset dictionary prefix.
///
/// The dictionary bytes are placed at the start of the output buffer so that
/// backward references in the compressed data can reach into the dictionary.
/// Only the decompressed data (after the dictionary) is returned.
pub fn decompress_with_prefix(
    compressed: &[u8],
    dest_len: usize,
    prefix: &[u8],
) -> io::Result<Vec<u8>> {
    let mut output = Vec::from(prefix);
    let target_len = prefix.len() + dest_len;
    let mut ip = 0;

    while ip < compressed.len() && output.len() < target_len {
        let token = compressed[ip] as usize;
        ip += 1;

        // Literal length
        let mut literal_len = token >> 4;
        if literal_len == 15 {
            loop {
                if ip >= compressed.len() {
                    return Err(io::Error::other("truncated literal length"));
                }
                let b = compressed[ip] as usize;
                ip += 1;
                literal_len += b;
                if b < 255 {
                    break;
                }
            }
        }

        // Copy literals
        if ip + literal_len > compressed.len() {
            return Err(io::Error::other("literal bytes out of bounds"));
        }
        output.extend_from_slice(&compressed[ip..ip + literal_len]);
        ip += literal_len;

        if output.len() >= target_len {
            break;
        }

        // Match distance (LE u16)
        if ip + 2 > compressed.len() {
            return Err(io::Error::other("truncated match distance"));
        }
        let distance = compressed[ip] as usize | ((compressed[ip + 1] as usize) << 8);
        ip += 2;

        if distance == 0 || distance > output.len() {
            return Err(io::Error::other("invalid match distance"));
        }

        // Match length
        let mut match_len = (token & 0x0F) + MIN_MATCH;
        if (token & 0x0F) == 15 {
            loop {
                if ip >= compressed.len() {
                    return Err(io::Error::other("truncated match length"));
                }
                let b = compressed[ip] as usize;
                ip += 1;
                match_len += b;
                if b < 255 {
                    break;
                }
            }
        }

        // Copy match (may overlap, may reference prefix/dictionary)
        let match_start = output.len() - distance;
        for i in 0..match_len {
            let b = output[match_start + i];
            output.push(b);
        }
    }

    Ok(output[prefix.len()..].to_vec())
}

/// LZ4 decompress from a streaming reader.
///
/// Reads compressed bytes directly from the input until `dest_len` output
/// bytes are produced. The LZ4 format is self-delimiting so no compressed
/// length needs to be known in advance.
pub fn decompress_from_reader(reader: &mut dyn Read, dest_len: usize) -> io::Result<Vec<u8>> {
    let mut output = Vec::with_capacity(dest_len);

    while output.len() < dest_len {
        let mut byte = [0u8; 1];
        reader.read_exact(&mut byte)?;
        let token = byte[0] as usize;

        // Literal length
        let mut literal_len = token >> 4;
        if literal_len == 15 {
            loop {
                reader.read_exact(&mut byte)?;
                let b = byte[0] as usize;
                literal_len += b;
                if b < 255 {
                    break;
                }
            }
        }

        // Copy literals
        let mut literal_buf = vec![0u8; literal_len];
        reader.read_exact(&mut literal_buf)?;
        output.extend_from_slice(&literal_buf);

        if output.len() >= dest_len {
            break;
        }

        // Match distance (LE u16)
        let mut dist_buf = [0u8; 2];
        reader.read_exact(&mut dist_buf)?;
        let distance = dist_buf[0] as usize | ((dist_buf[1] as usize) << 8);

        if distance == 0 || distance > output.len() {
            return Err(io::Error::other("invalid match distance"));
        }

        // Match length
        let mut match_len = (token & 0x0F) + MIN_MATCH;
        if (token & 0x0F) == 15 {
            loop {
                reader.read_exact(&mut byte)?;
                let b = byte[0] as usize;
                match_len += b;
                if b < 255 {
                    break;
                }
            }
        }

        // Copy match (may overlap)
        let match_start = output.len() - distance;
        for i in 0..match_len {
            let b = output[match_start + i];
            output.push(b);
        }
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compress_decompress_simple() {
        let input = b"hello world hello world hello world!";
        let compressed = compress(input);
        let decompressed = decompress(&compressed, input.len()).unwrap();
        assert_eq!(&decompressed, input);
    }

    #[test]
    fn test_compress_decompress_short() {
        let input = b"hi";
        let compressed = compress(input);
        let decompressed = decompress(&compressed, input.len()).unwrap();
        assert_eq!(&decompressed, input);
    }

    #[test]
    fn test_compress_decompress_empty() {
        let input = b"";
        let compressed = compress(input);
        assert_is_empty!(compressed);
    }

    #[test]
    fn test_compress_decompress_repetitive() {
        let input: Vec<u8> = "abcdefgh".repeat(100).into_bytes();
        let compressed = compress(&input);
        // Repetitive data should compress well
        assert_lt!(compressed.len(), input.len());
        let decompressed = decompress(&compressed, input.len()).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_compress_decompress_longer_text() {
        let input = b"The quick brown fox jumps over the lazy dog. \
                       The quick brown fox jumps over the lazy dog again. \
                       And once more the quick brown fox jumps over the lazy dog.";
        let compressed = compress(input);
        let decompressed = decompress(&compressed, input.len()).unwrap();
        assert_eq!(&decompressed[..], &input[..]);
    }

    #[test]
    fn test_compress_all_same_byte() {
        let input = vec![0xAA; 1000];
        let compressed = compress(&input);
        assert_lt!(compressed.len(), input.len());
        let decompressed = decompress(&compressed, input.len()).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_compress_with_dictionary_no_dict() {
        // With dict_len=0, should behave like regular compress
        let input = b"hello world hello world hello world!";
        let buffer = input.to_vec();
        let compressed = compress_with_dictionary(&buffer, 0);
        let decompressed = decompress(&compressed, input.len()).unwrap();
        assert_eq!(&decompressed, input);
    }

    #[test]
    fn test_compress_with_dictionary_small() {
        // Dict + data
        let dict = b"the quick brown fox ";
        let data = b"the quick brown fox jumps over";
        let mut buffer = Vec::new();
        buffer.extend_from_slice(dict);
        buffer.extend_from_slice(data);

        let compressed = compress_with_dictionary(&buffer, dict.len());
        // Decompress: need dict bytes in output buffer first, then decompress data
        let output = Vec::from(&dict[..]);
        let decompressed = test_decompress_with_prefix(&compressed, data.len(), &output);
        assert_eq!(&decompressed, data);
    }

    #[test]
    fn test_compress_with_dictionary_short_data() {
        // Very short data that's all literals
        let dict = b"abcd";
        let data = b"xyz";
        let mut buffer = Vec::new();
        buffer.extend_from_slice(dict);
        buffer.extend_from_slice(data);

        let compressed = compress_with_dictionary(&buffer, dict.len());
        let decompressed = decompress(&compressed, data.len()).unwrap();
        assert_eq!(&decompressed, data);
    }

    /// Regression test: when dict+data is small (e.g. 27 bytes) and a match near the end of
    /// the data references the dictionary, the compressor must NOT consume all remaining bytes
    /// in the match — at least LAST_LITERALS (5) bytes must remain for the final-literals token.
    /// Otherwise, a spurious 0x00 byte is emitted, causing stream misalignment in the
    /// LZ4WithPresetDict decompressor.
    /// Ported from the bug exposed by Java's LZ4WithPresetDictDecompressor failing with
    /// "arraycopy: source index -999 out of bounds for byte[32]".
    #[test]
    fn test_compress_with_dictionary_last_literals_preserved() {
        // Simulate the stored fields scenario: 9-byte dict + 18-byte block where
        // the block data ends with bytes that match the dictionary prefix.
        // Dict = [0x00, 0x3C, '/', 'h', 'o', 'm', 'e', '/', 'r']
        // Block data contains "...XX" + [0x00, 0x3C, '/', 'h', 'o', 'm', 'e', '/']
        // which matches dict[0..8], potentially consuming all remaining bytes.
        let dict: &[u8] = &[0x00, 0x3C, b'/', b'h', b'o', b'm', b'e', b'/', b'r'];
        let data: &[u8] = &[
            b'n', b'o', b'l', b'o', b'g', b'y', b'.', b't', b'x', b't', 0x00, 0x3C, b'/', b'h',
            b'o', b'm', b'e', b'/',
        ];
        let mut buffer = Vec::new();
        buffer.extend_from_slice(dict);
        buffer.extend_from_slice(data);

        let compressed = compress_with_dictionary(&buffer, dict.len());

        // Verify the compressed output ends with a valid last-literals token
        // (i.e., no spurious 0x00 byte from encode_last_literals with len=0).
        // The last token must encode at least LAST_LITERALS (5) literal bytes.
        assert_not_empty!(compressed);

        // Decompress with dictionary prefix and verify round-trip
        let decompressed = test_decompress_with_prefix(&compressed, data.len(), dict);
        assert_eq!(
            &decompressed,
            data,
            "round-trip failed: compressed {} bytes",
            compressed.len()
        );

        // Verify the compressed data is exactly consumed by decompression
        // (no spurious trailing bytes that would cause stream misalignment)
        let mut consumed = 0;
        let mut produced = 0;
        while consumed < compressed.len() && produced < data.len() {
            let token = compressed[consumed] as usize;
            consumed += 1;
            let mut literal_len = token >> 4;
            if literal_len == 15 {
                loop {
                    let b = compressed[consumed] as usize;
                    consumed += 1;
                    literal_len += b;
                    if b < 255 {
                        break;
                    }
                }
            }
            consumed += literal_len;
            produced += literal_len;
            if produced >= data.len() {
                break;
            }
            consumed += 2; // match distance
            let mut match_len = (token & 0x0F) + MIN_MATCH;
            if (token & 0x0F) == 15 {
                loop {
                    let b = compressed[consumed] as usize;
                    consumed += 1;
                    match_len += b;
                    if b < 255 {
                        break;
                    }
                }
            }
            produced += match_len;
        }
        assert_eq!(
            consumed,
            compressed.len(),
            "compressed data has {} unconsumed trailing bytes (stream misalignment bug)",
            compressed.len() - consumed
        );
    }

    /// Test helper that unwraps the public decompress_with_prefix.
    fn test_decompress_with_prefix(compressed: &[u8], dest_len: usize, prefix: &[u8]) -> Vec<u8> {
        super::decompress_with_prefix(compressed, dest_len, prefix).unwrap()
    }

    #[test]
    fn test_compress_decompress_large_input() {
        // >64KB input exercises the 32-bit offset path in fast_hash_log
        let mut input = Vec::with_capacity(70_000);
        // Non-repeating pattern followed by repeating data
        for i in 0u32..17500 {
            input.extend_from_slice(&i.to_le_bytes());
        }
        // Add repeating section for compression
        let pattern = b"compress me please! ";
        for _ in 0..50 {
            input.extend_from_slice(pattern);
        }
        assert_gt!(input.len(), 65536);
        let compressed = compress(&input);
        let decompressed = decompress(&compressed, input.len()).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_compress_high_large_input() {
        // >64KB exercises HC reset for large previous compression
        // and the distance limit branches in get() and previous()
        let mut input = Vec::with_capacity(70_000);
        // Create data with matches at various distances
        let chunk = b"the quick brown fox jumps over the lazy dog and ";
        for _ in 0..1500 {
            input.extend_from_slice(chunk);
        }
        assert_gt!(input.len(), 65536);

        let mut ht = HighCompressionHashTable::new();
        let compressed = compress_high(&input, &mut ht);
        let decompressed = decompress(&compressed, input.len()).unwrap();
        assert_eq!(decompressed, input);

        // Compress again — exercises reset with large previous end-base
        let compressed2 = compress_high(&input, &mut ht);
        let decompressed2 = decompress(&compressed2, input.len()).unwrap();
        assert_eq!(decompressed2, input);
    }

    #[test]
    fn test_compress_with_dictionary_empty_data() {
        // Empty data with non-empty dictionary (line 244)
        let dict = b"some dictionary content";
        let compressed = compress_with_dictionary(dict, dict.len());
        assert_is_empty!(compressed);
    }

    #[test]
    fn test_compress_decompress_long_literal_run() {
        // Input with a long non-repeating section (>255 bytes of literals)
        // followed by repeating data to get a match.
        let mut input = Vec::new();
        // 300 bytes of incrementing data (won't match anything)
        for i in 0..300u16 {
            input.push((i % 251) as u8); // prime modulus avoids accidental 4-byte repeats
        }
        // Then repeating data to exercise match encoding
        let repeat = b"ABCDEFGHIJKLMNOP";
        for _ in 0..20 {
            input.extend_from_slice(repeat);
        }
        let compressed = compress(&input);
        let decompressed = decompress(&compressed, input.len()).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_compress_decompress_long_trailing_literals() {
        // Input where the final unmatched section is >=270 bytes,
        // exercising the while remaining >= 255 loop in encode_last_literals.
        // We need LAST_LITERALS (5) + 270 = 275 trailing non-matching bytes
        // minimum. Use 300 to be safe.
        // The trailing section must be truly unmatchable — no 4-byte sequence
        // can repeat within MAX_DISTANCE (64KB).
        let mut input = Vec::new();
        // Short repeating prefix to seed the hash table
        let pattern = b"ABCDEFGHIJKLMNOP";
        for _ in 0..3 {
            input.extend_from_slice(pattern);
        }
        // 300 bytes where every 4-byte window is unique: use a counter
        // encoded big-endian so no 4-byte subsequence repeats.
        for i in 0u32..75 {
            input.extend_from_slice(&(i + 0x80000000).to_be_bytes());
        }
        assert_ge!(input.len(), 48 + 300); // pattern + unique
        let compressed = compress(&input);
        let decompressed = decompress(&compressed, input.len()).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_compress_decompress_long_match() {
        // Input with a very long match (>19 bytes, exercises extended match length)
        let pattern = b"the quick brown fox ";
        let mut input = Vec::new();
        input.extend_from_slice(pattern);
        // Add unrelated filler so the hash table gets populated
        input.extend_from_slice(b"ZZZZ filler text!! ");
        // Repeat the pattern — will produce a long match
        for _ in 0..5 {
            input.extend_from_slice(pattern);
        }
        let compressed = compress(&input);
        let decompressed = decompress(&compressed, input.len()).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_compress_high_simple() {
        let input = b"hello world hello world hello world!";
        let mut ht = HighCompressionHashTable::new();
        let compressed = compress_high(input, &mut ht);
        let decompressed = decompress(&compressed, input.len()).unwrap();
        assert_eq!(&decompressed, input);
    }

    #[test]
    fn test_compress_high_short() {
        let input = b"hi";
        let mut ht = HighCompressionHashTable::new();
        let compressed = compress_high(input, &mut ht);
        let decompressed = decompress(&compressed, input.len()).unwrap();
        assert_eq!(&decompressed, input);
    }

    #[test]
    fn test_compress_high_empty() {
        let mut ht = HighCompressionHashTable::new();
        let compressed = compress_high(b"", &mut ht);
        assert_is_empty!(compressed);
    }

    #[test]
    fn test_compress_high_repetitive() {
        let input: Vec<u8> = "abcdefgh".repeat(100).into_bytes();
        let mut ht = HighCompressionHashTable::new();
        let compressed = compress_high(&input, &mut ht);
        assert_lt!(compressed.len(), input.len());
        let decompressed = decompress(&compressed, input.len()).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_compress_high_better_than_fast() {
        // High compression should generally match or beat fast compression
        let input: Vec<u8> = "the quick brown fox jumps over the lazy dog "
            .repeat(50)
            .into_bytes();
        let fast_compressed = compress(&input);
        let mut ht = HighCompressionHashTable::new();
        let high_compressed = compress_high(&input, &mut ht);

        // Both must decompress correctly
        let fast_decompressed = decompress(&fast_compressed, input.len()).unwrap();
        let high_decompressed = decompress(&high_compressed, input.len()).unwrap();
        assert_eq!(fast_decompressed, input);
        assert_eq!(high_decompressed, input);

        // High compression should be at least as good
        assert_le!(high_compressed.len(), fast_compressed.len());
    }

    #[test]
    fn test_compress_high_reuse_hash_table() {
        let mut ht = HighCompressionHashTable::new();

        // Compress multiple inputs reusing the same hash table
        for _ in 0..5 {
            let input: Vec<u8> = "pattern repeated pattern repeated pattern "
                .repeat(20)
                .into_bytes();
            let compressed = compress_high(&input, &mut ht);
            let decompressed = decompress(&compressed, input.len()).unwrap();
            assert_eq!(decompressed, input);
        }
    }

    #[test]
    fn test_compress_high_all_same_byte() {
        let input = vec![0xAA; 1000];
        let mut ht = HighCompressionHashTable::new();
        let compressed = compress_high(&input, &mut ht);
        assert_lt!(compressed.len(), input.len());
        let decompressed = decompress(&compressed, input.len()).unwrap();
        assert_eq!(decompressed, input);
    }
}
