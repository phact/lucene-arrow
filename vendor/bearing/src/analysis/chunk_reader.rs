// SPDX-License-Identifier: Apache-2.0

//! Streaming UTF-8 chunk reader using `utf8-zero`.

use std::io::{self, BufReader, Read};

use utf8_zero::BufReadDecoder;

/// Default buffer capacity for the underlying `BufReader` (8 KB).
const DEFAULT_CAPACITY: usize = 8192;

/// Streams valid UTF-8 chunks from a reader.
///
/// Each call to [`next_chunk`](Utf8ChunkReader::next_chunk) returns the next
/// chunk of valid UTF-8 text, sized by the underlying `BufReader` buffer
/// (default 8 KB). Chunks are guaranteed to contain only complete UTF-8
/// codepoints — multi-byte sequences are never split across chunks.
pub(crate) struct Utf8ChunkReader {
    decoder: BufReadDecoder<BufReader<Box<dyn Read + Send>>>,
}

impl Utf8ChunkReader {
    /// Creates a new chunk reader with the default 8 KB buffer.
    pub(crate) fn new(reader: Box<dyn Read + Send>) -> Self {
        Self::with_capacity(DEFAULT_CAPACITY, reader)
    }

    /// Creates a new chunk reader with a custom buffer capacity.
    ///
    /// Smaller capacities are useful for testing chunk boundary behavior.
    pub(crate) fn with_capacity(capacity: usize, reader: Box<dyn Read + Send>) -> Self {
        let buf_reader = BufReader::with_capacity(capacity, reader);
        Self {
            decoder: BufReadDecoder::new(buf_reader),
        }
    }

    /// Returns the next UTF-8 chunk, or `None` at EOF.
    pub(crate) fn next_chunk(&mut self) -> io::Result<Option<String>> {
        match self.decoder.next_strict() {
            Some(Ok(chunk)) => Ok(Some(chunk.to_string())),
            Some(Err(e)) => Err(io::Error::new(io::ErrorKind::InvalidData, e.to_string())),
            None => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assertables::*;

    fn reader_from(s: &str) -> Box<dyn Read + Send> {
        Box::new(io::Cursor::new(s.as_bytes().to_vec()))
    }

    #[test]
    fn empty_input_returns_none() {
        let mut reader = Utf8ChunkReader::new(reader_from(""));
        let chunk = reader.next_chunk().unwrap();
        assert_none!(&chunk);
    }

    #[test]
    fn small_input_returns_one_chunk() {
        let mut reader = Utf8ChunkReader::new(reader_from("hello world"));
        let chunk = reader.next_chunk().unwrap();
        assert_eq!(chunk.unwrap(), "hello world");
        let chunk = reader.next_chunk().unwrap();
        assert_none!(&chunk);
    }

    #[test]
    fn large_input_returns_multiple_chunks() {
        let input: String = "abcdefgh".repeat(2000); // 16KB > 8KB buffer
        let mut reader = Utf8ChunkReader::new(reader_from(&input));
        let mut reassembled = String::new();
        let mut chunk_count = 0;
        while let Some(chunk) = reader.next_chunk().unwrap() {
            reassembled.push_str(&chunk);
            chunk_count += 1;
        }
        assert_eq!(reassembled, input);
        assert_gt!(chunk_count, 1);
    }

    #[test]
    fn multibyte_chars_not_split() {
        // U+00E9 (é) is 2 bytes, U+2019 (') is 3 bytes, U+1F600 (😀) is 4 bytes
        let input = "é".repeat(5000); // ~10KB of 2-byte chars
        let mut reader = Utf8ChunkReader::new(reader_from(&input));
        let mut reassembled = String::new();
        while let Some(chunk) = reader.next_chunk().unwrap() {
            reassembled.push_str(&chunk);
        }
        assert_eq!(reassembled, input);
    }

    #[test]
    fn tiny_capacity_forces_many_chunks() {
        let input = "hello world, this is a test";
        let mut reader = Utf8ChunkReader::with_capacity(4, reader_from(input));
        let mut reassembled = String::new();
        let mut chunk_count = 0;
        while let Some(chunk) = reader.next_chunk().unwrap() {
            assert_le!(chunk.len(), 4);
            reassembled.push_str(&chunk);
            chunk_count += 1;
        }
        assert_eq!(reassembled, input);
        assert_gt!(chunk_count, 5);
    }

    #[test]
    fn all_multibyte_with_small_capacity() {
        // Each '😀' is 4 bytes. With capacity 4, each chunk should be exactly one emoji.
        let input = "😀😀😀😀😀";
        let mut reader = Utf8ChunkReader::with_capacity(4, reader_from(input));
        let mut reassembled = String::new();
        while let Some(chunk) = reader.next_chunk().unwrap() {
            reassembled.push_str(&chunk);
        }
        assert_eq!(reassembled, input);
    }
}
