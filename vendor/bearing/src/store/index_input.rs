// SPDX-License-Identifier: Apache-2.0

//! `IndexInput<'a>` — read-path struct over `Cursor<&'a [u8]>`.
//!
//! Borrows its bytes with a caller-supplied lifetime `'a` and owns its own
//! cursor position. All read methods are inherent on this struct.
//!
//! Every byte offset and length in the public API is `usize`, matching the
//! underlying `&[u8]` indexing type.

use std::collections::HashMap;
use std::fmt;
use std::io;
use std::io::BufRead;
use std::io::Cursor;

use crate::encoding::string;
use crate::encoding::varint;

/// Sentinel name used by [`IndexInput::unnamed`] when the caller has no
/// meaningful label to attach (e.g. tests, transient parsing buffers). Only
/// surfaces in the [`fmt::Debug`] output.
const UNNAMED: &str = "<unnamed>";

pub(crate) struct IndexInput<'a> {
    name: String,
    cursor: Cursor<&'a [u8]>,
}

impl<'a> IndexInput<'a> {
    /// Constructs a new input named `name` over `bytes`. Position starts at 0.
    pub(crate) fn new(name: impl Into<String>, bytes: &'a [u8]) -> Self {
        Self {
            name: name.into(),
            cursor: Cursor::new(bytes),
        }
    }

    /// Constructs a new input over `bytes` without a meaningful name.
    /// Equivalent to [`new`](Self::new) with the [`UNNAMED`] sentinel.
    pub(crate) fn unnamed(bytes: &'a [u8]) -> Self {
        Self::new(UNNAMED, bytes)
    }

    // ---------- identity / cursor state ----------

    pub(crate) fn length(&self) -> usize {
        self.cursor.get_ref().len()
    }

    pub(crate) fn position(&self) -> usize {
        self.cursor.position() as usize
    }

    /// Moves the cursor to `pos`. Seeking exactly to `length()` is allowed;
    /// seeking past the end is an error.
    pub(crate) fn seek(&mut self, pos: usize) -> io::Result<()> {
        if pos > self.length() {
            return Err(io::Error::other(format!(
                "seek past end: pos={pos} length={}",
                self.length()
            )));
        }
        self.cursor.set_position(pos as u64);
        Ok(())
    }

    // ---------- fixed-width reads (little-endian, data path) ----------

    #[inline]
    pub(crate) fn read_byte(&mut self) -> io::Result<u8> {
        let mut buf = [0u8; 1];
        self.read_bytes(&mut buf)?;
        Ok(buf[0])
    }

    #[inline]
    pub(crate) fn read_le_short(&mut self) -> io::Result<i16> {
        let mut buf = [0u8; 2];
        self.read_bytes(&mut buf)?;
        Ok(i16::from_le_bytes(buf))
    }

    #[inline]
    pub(crate) fn read_le_int(&mut self) -> io::Result<i32> {
        let mut buf = [0u8; 4];
        self.read_bytes(&mut buf)?;
        Ok(i32::from_le_bytes(buf))
    }

    #[inline]
    pub(crate) fn read_le_long(&mut self) -> io::Result<i64> {
        let mut buf = [0u8; 8];
        self.read_bytes(&mut buf)?;
        Ok(i64::from_le_bytes(buf))
    }

    // ---------- fixed-width reads (big-endian, codec headers/footers) ----------

    #[inline]
    pub(crate) fn read_be_int(&mut self) -> io::Result<i32> {
        let mut buf = [0u8; 4];
        self.read_bytes(&mut buf)?;
        Ok(i32::from_be_bytes(buf))
    }

    #[inline]
    pub(crate) fn read_be_long(&mut self) -> io::Result<i64> {
        let mut buf = [0u8; 8];
        self.read_bytes(&mut buf)?;
        Ok(i64::from_be_bytes(buf))
    }

    /// Returns a mutable reference to the underlying cursor. Intended for
    /// codec-level callers that invoke `crate::encoding::*` decoders (which
    /// take `&mut Cursor<&[u8]>` directly) without introducing per-helper
    /// wrappers on [`IndexInput`].
    #[inline]
    pub(crate) fn cursor_mut(&mut self) -> &mut Cursor<&'a [u8]> {
        &mut self.cursor
    }

    // ---------- variable-length integer reads ----------

    #[inline]
    pub(crate) fn read_vint(&mut self) -> io::Result<i32> {
        varint::read_vint_cursor(&mut self.cursor)
    }

    #[inline]
    pub(crate) fn read_vlong(&mut self) -> io::Result<i64> {
        varint::read_vlong(&mut self.cursor)
    }

    #[inline]
    pub(crate) fn read_zint(&mut self) -> io::Result<i32> {
        varint::read_zint(&mut self.cursor)
    }

    // ---------- string / collection reads ----------

    #[inline]
    pub(crate) fn read_string(&mut self) -> io::Result<String> {
        string::read_string(&mut self.cursor)
    }

    #[inline]
    pub(crate) fn read_set_of_strings(&mut self) -> io::Result<Vec<String>> {
        string::read_set_of_strings(&mut self.cursor)
    }

    #[inline]
    pub(crate) fn read_map_of_strings(&mut self) -> io::Result<HashMap<String, String>> {
        string::read_map_of_strings(&mut self.cursor)
    }

    // ---------- bulk reads ----------

    /// Copies the next `dst.len()` bytes into `dst`.
    #[inline]
    pub(crate) fn read_bytes(&mut self, dst: &mut [u8]) -> io::Result<()> {
        let buf = self.cursor.fill_buf()?;
        if buf.len() < dst.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("read_bytes: need {} have {}", dst.len(), buf.len()),
            ));
        }
        dst.copy_from_slice(&buf[..dst.len()]);
        self.cursor.consume(dst.len());
        Ok(())
    }

    /// Returns a zero-copy borrow of the next `len` bytes with lifetime `'a`
    /// and advances the cursor. The returned slice outlives the input.
    #[inline]
    pub(crate) fn read_slice(&mut self, len: usize) -> io::Result<&'a [u8]> {
        let pos = self.position();
        let full: &'a [u8] = self.cursor.get_ref();
        let end = pos
            .checked_add(len)
            .ok_or_else(|| io::Error::other("read_slice: length overflow"))?;
        if end > full.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("read_slice past end: end={end} length={}", full.len()),
            ));
        }
        let out = &full[pos..end];
        self.cursor.set_position(end as u64);
        Ok(out)
    }

    /// Advances the cursor by `n` bytes without copying.
    pub(crate) fn skip_bytes(&mut self, n: usize) -> io::Result<()> {
        let pos = self.position();
        let end = pos
            .checked_add(n)
            .ok_or_else(|| io::Error::other("skip_bytes: length overflow"))?;
        if end > self.length() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("skip_bytes past end: end={end} length={}", self.length()),
            ));
        }
        self.cursor.set_position(end as u64);
        Ok(())
    }

    // ---------- derived view ----------

    /// Returns a new `IndexInput<'a>` borrowing a range of this input's
    /// bytes. The returned input has a fresh cursor at position 0. Does not
    /// mutate `self`.
    pub(crate) fn view(
        &self,
        name: impl Into<String>,
        offset: usize,
        length: usize,
    ) -> io::Result<IndexInput<'a>> {
        let full: &'a [u8] = self.cursor.get_ref();
        let end = offset
            .checked_add(length)
            .ok_or_else(|| io::Error::other("view: offset + length overflow"))?;
        if end > full.len() {
            return Err(io::Error::other(format!(
                "view out of range: offset={offset} length={length} source={}",
                full.len()
            )));
        }
        Ok(IndexInput::new(name, &full[offset..end]))
    }

    // ---------- absolute-position reads ----------

    /// Reads a single byte at absolute position `pos`. Does not mutate the
    /// cursor.
    #[inline]
    pub(crate) fn read_byte_at(&self, pos: usize) -> io::Result<u8> {
        self.cursor.get_ref().get(pos).copied().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("read_byte_at {pos} of {}", self.length()),
            )
        })
    }

    #[inline]
    pub(crate) fn read_le_short_at(&self, pos: usize) -> io::Result<i16> {
        Ok(i16::from_le_bytes(self.bytes_at(pos)?))
    }

    #[inline]
    pub(crate) fn read_le_int_at(&self, pos: usize) -> io::Result<i32> {
        Ok(i32::from_le_bytes(self.bytes_at(pos)?))
    }

    #[inline]
    pub(crate) fn read_le_long_at(&self, pos: usize) -> io::Result<i64> {
        Ok(i64::from_le_bytes(self.bytes_at(pos)?))
    }

    fn bytes_at<const N: usize>(&self, pos: usize) -> io::Result<[u8; N]> {
        let full = self.cursor.get_ref();
        let end = pos
            .checked_add(N)
            .ok_or_else(|| io::Error::other("read_at: offset + length overflow"))?;
        full.get(pos..end)
            .and_then(|s| s.try_into().ok())
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!("read_at {pos}..{end} of {}", full.len()),
                )
            })
    }
}

impl fmt::Debug for IndexInput<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IndexInput")
            .field("name", &self.name)
            .field("length", &self.length())
            .field("position", &self.position())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoding::string::{write_map_of_strings, write_set_of_strings, write_string};
    use crate::encoding::varint::{write_vint, write_vlong, write_zint};

    fn input_over(bytes: &[u8]) -> IndexInput<'_> {
        IndexInput::unnamed(bytes)
    }

    // ---------- identity / initial state ----------

    #[test]
    fn new_initial_state() {
        let input = IndexInput::new("foo", &[1u8, 2, 3, 4, 5][..]);
        assert_eq!(input.length(), 5);
        assert_eq!(input.position(), 0);
    }

    #[test]
    fn empty_input_state() {
        let input = IndexInput::new("empty", &[][..]);
        assert_eq!(input.length(), 0);
        assert_eq!(input.position(), 0);
    }

    #[test]
    fn debug_shows_name_length_position_not_bytes() {
        let input = IndexInput::new("foo", &[0u8; 1024][..]);
        let rendered = format!("{input:?}");
        assert_contains!(&rendered, "foo");
        assert_contains!(&rendered, "1024");
        assert_contains!(&rendered, "position");
        assert_not_contains!(&rendered, "0, 0, 0");
    }

    // ---------- seek ----------

    #[test]
    fn seek_to_valid_position() {
        let mut input = input_over(&[1, 2, 3, 4]);
        input.seek(2).unwrap();
        assert_eq!(input.position(), 2);
    }

    #[test]
    fn seek_past_end_errors() {
        let mut input = input_over(&[1, 2, 3, 4]);
        assert_err!(input.seek(5));
    }

    #[test]
    fn seek_to_length_allowed_but_read_errors() {
        let mut input = input_over(&[1, 2]);
        input.seek(2).unwrap();
        assert_eq!(input.position(), 2);
        assert_err!(input.read_byte());
    }

    #[test]
    fn seek_then_read() {
        let mut input = input_over(&[10, 20, 30, 40]);
        input.seek(2).unwrap();
        assert_eq!(input.read_byte().unwrap(), 30);
        assert_eq!(input.position(), 3);
    }

    // ---------- fixed-width LE ----------

    #[test]
    fn read_byte_roundtrip() {
        let mut input = input_over(&[42]);
        assert_eq!(input.read_byte().unwrap(), 42);
        assert_eq!(input.position(), 1);
    }

    #[test]
    fn read_byte_past_end_errors() {
        let mut input = input_over(&[]);
        assert_err!(input.read_byte());
    }

    #[test]
    fn read_le_short_roundtrip() {
        let bytes = 0x1234i16.to_le_bytes();
        let mut input = input_over(&bytes);
        assert_eq!(input.read_le_short().unwrap(), 0x1234);
        assert_eq!(input.position(), 2);
    }

    #[test]
    fn read_le_short_past_end_errors() {
        let mut input = input_over(&[0x34]);
        assert_err!(input.read_le_short());
    }

    #[test]
    fn read_le_int_roundtrip() {
        let bytes = 0x1234_5678i32.to_le_bytes();
        let mut input = input_over(&bytes);
        assert_eq!(input.read_le_int().unwrap(), 0x1234_5678);
        assert_eq!(input.position(), 4);
    }

    #[test]
    fn read_le_int_past_end_errors() {
        let mut input = input_over(&[0, 0, 0]);
        assert_err!(input.read_le_int());
    }

    #[test]
    fn read_le_long_roundtrip() {
        let bytes = 0x1234_5678_9ABC_DEF0i64.to_le_bytes();
        let mut input = input_over(&bytes);
        assert_eq!(input.read_le_long().unwrap(), 0x1234_5678_9ABC_DEF0);
        assert_eq!(input.position(), 8);
    }

    #[test]
    fn read_le_long_past_end_errors() {
        let mut input = input_over(&[0u8; 7]);
        assert_err!(input.read_le_long());
    }

    // ---------- fixed-width BE ----------

    #[test]
    fn read_be_int_roundtrip() {
        let bytes = 0x1234_5678i32.to_be_bytes();
        let mut input = input_over(&bytes);
        assert_eq!(input.read_be_int().unwrap(), 0x1234_5678);
        assert_eq!(input.position(), 4);
    }

    #[test]
    fn read_be_int_past_end_errors() {
        let mut input = input_over(&[0, 0, 0]);
        assert_err!(input.read_be_int());
    }

    #[test]
    fn read_be_long_roundtrip() {
        let bytes = 0x1234_5678_9ABC_DEF0i64.to_be_bytes();
        let mut input = input_over(&bytes);
        assert_eq!(input.read_be_long().unwrap(), 0x1234_5678_9ABC_DEF0);
        assert_eq!(input.position(), 8);
    }

    #[test]
    fn read_be_long_past_end_errors() {
        let mut input = input_over(&[0u8; 7]);
        assert_err!(input.read_be_long());
    }

    // ---------- variable-length (smoke — heavy coverage in varint module) ----------

    #[test]
    fn read_vint_delegates() {
        let mut buf = Vec::new();
        write_vint(&mut buf, 16384).unwrap();
        let mut input = input_over(&buf);
        assert_eq!(input.read_vint().unwrap(), 16384);
        assert_eq!(input.position(), buf.len());
    }

    #[test]
    fn read_vlong_delegates() {
        let mut buf = Vec::new();
        write_vlong(&mut buf, i64::MAX).unwrap();
        let mut input = input_over(&buf);
        assert_eq!(input.read_vlong().unwrap(), i64::MAX);
    }

    #[test]
    fn read_zint_delegates() {
        let mut buf = Vec::new();
        write_zint(&mut buf, -42).unwrap();
        let mut input = input_over(&buf);
        assert_eq!(input.read_zint().unwrap(), -42);
    }

    // ---------- strings / collections (smoke — heavy coverage in string module) ----------

    #[test]
    fn read_string_delegates() {
        let mut buf = Vec::new();
        write_string(&mut buf, "hello").unwrap();
        let mut input = input_over(&buf);
        assert_eq!(input.read_string().unwrap(), "hello");
    }

    #[test]
    fn read_set_of_strings_delegates() {
        let set = vec!["a".to_string(), "b".to_string()];
        let mut buf = Vec::new();
        write_set_of_strings(&mut buf, &set).unwrap();
        let mut input = input_over(&buf);
        assert_eq!(input.read_set_of_strings().unwrap(), set);
    }

    #[test]
    fn read_map_of_strings_delegates() {
        let mut map = HashMap::new();
        map.insert("k".to_string(), "v".to_string());
        let mut buf = Vec::new();
        write_map_of_strings(&mut buf, &map).unwrap();
        let mut input = input_over(&buf);
        assert_eq!(input.read_map_of_strings().unwrap(), map);
    }

    // ---------- bulk ----------

    #[test]
    fn read_bytes_fills_buffer() {
        let mut input = input_over(&[10, 20, 30, 40, 50]);
        let mut dst = [0u8; 3];
        input.read_bytes(&mut dst).unwrap();
        assert_eq!(dst, [10, 20, 30]);
        assert_eq!(input.position(), 3);
    }

    #[test]
    fn read_bytes_past_end_errors() {
        let mut input = input_over(&[1, 2]);
        let mut dst = [0u8; 3];
        assert_err!(input.read_bytes(&mut dst));
    }

    #[test]
    fn read_slice_returns_borrowed_bytes_with_outer_lifetime() {
        let source = [10u8, 20, 30, 40, 50];
        let slice = {
            let mut input = IndexInput::unnamed(&source[..]);
            let s = input.read_slice(3).unwrap();
            assert_eq!(input.position(), 3);
            s
        };
        // slice outlived `input`; lifetime is tied to `source`
        assert_eq!(slice, &[10, 20, 30]);
    }

    #[test]
    fn read_slice_past_end_errors() {
        let mut input = input_over(&[1, 2]);
        assert_err!(input.read_slice(3));
    }

    #[test]
    fn skip_bytes_advances_position() {
        let mut input = input_over(&[1, 2, 3, 4, 5]);
        input.skip_bytes(2).unwrap();
        assert_eq!(input.position(), 2);
        assert_eq!(input.read_byte().unwrap(), 3);
    }

    #[test]
    fn skip_bytes_past_end_errors() {
        let mut input = input_over(&[1, 2]);
        assert_err!(input.skip_bytes(3));
    }

    // ---------- view ----------

    #[test]
    fn view_valid_bounds() {
        let input = input_over(&[1, 2, 3, 4, 5]);
        let mut sub = input.view("sub", 1, 3).unwrap();
        assert_eq!(sub.length(), 3);
        assert_eq!(sub.position(), 0);
        assert_eq!(sub.read_byte().unwrap(), 2);
        assert_eq!(sub.read_byte().unwrap(), 3);
        assert_eq!(sub.read_byte().unwrap(), 4);
    }

    #[test]
    fn view_offset_out_of_range_errors() {
        let input = input_over(&[1, 2, 3]);
        assert_err!(input.view("sub", 5, 1));
    }

    #[test]
    fn view_length_out_of_range_errors() {
        let input = input_over(&[1, 2, 3]);
        assert_err!(input.view("sub", 1, 5));
    }

    #[test]
    fn view_nested() {
        let input = input_over(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let sub = input.view("sub", 2, 5).unwrap();
        let mut sub_sub = sub.view("sub-sub", 1, 3).unwrap();
        assert_eq!(sub_sub.length(), 3);
        assert_eq!(sub_sub.read_byte().unwrap(), 4);
        assert_eq!(sub_sub.read_byte().unwrap(), 5);
        assert_eq!(sub_sub.read_byte().unwrap(), 6);
    }

    #[test]
    fn view_does_not_mutate_parent() {
        let mut input = input_over(&[1, 2, 3, 4, 5]);
        input.seek(3).unwrap();
        let sub = input.view("sub", 1, 3).unwrap();
        assert_eq!(sub.position(), 0);
        assert_eq!(input.position(), 3);
    }

    // ---------- absolute-position reads ----------

    #[test]
    fn read_byte_at_valid() {
        let input = input_over(&[10, 20, 30]);
        assert_eq!(input.read_byte_at(0).unwrap(), 10);
        assert_eq!(input.read_byte_at(2).unwrap(), 30);
    }

    #[test]
    fn read_byte_at_past_end_errors() {
        let input = input_over(&[10, 20, 30]);
        assert_err!(input.read_byte_at(3));
    }

    #[test]
    fn read_le_short_at_valid() {
        let bytes = [0u8, 0, 0x34, 0x12, 0, 0];
        let input = input_over(&bytes);
        assert_eq!(input.read_le_short_at(2).unwrap(), 0x1234);
    }

    #[test]
    fn read_le_short_at_past_end_errors() {
        let input = input_over(&[0, 0, 0]);
        assert_err!(input.read_le_short_at(2));
    }

    #[test]
    fn read_le_int_at_valid() {
        let mut bytes = [0u8; 10];
        bytes[4..8].copy_from_slice(&0x1234_5678i32.to_le_bytes());
        let input = input_over(&bytes);
        assert_eq!(input.read_le_int_at(4).unwrap(), 0x1234_5678);
    }

    #[test]
    fn read_le_int_at_past_end_errors() {
        let input = input_over(&[0u8; 5]);
        assert_err!(input.read_le_int_at(3));
    }

    #[test]
    fn read_le_long_at_valid() {
        let mut bytes = [0u8; 16];
        bytes[4..12].copy_from_slice(&0x1234_5678_9ABC_DEF0i64.to_le_bytes());
        let input = input_over(&bytes);
        assert_eq!(input.read_le_long_at(4).unwrap(), 0x1234_5678_9ABC_DEF0);
    }

    #[test]
    fn read_le_long_at_past_end_errors() {
        let input = input_over(&[0u8; 10]);
        assert_err!(input.read_le_long_at(5));
    }

    #[test]
    fn absolute_reads_do_not_mutate_position() {
        let mut input = input_over(&[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]);
        input.seek(2).unwrap();
        let before = input.position();
        let _ = input.read_byte_at(5).unwrap();
        let _ = input.read_le_short_at(4).unwrap();
        let _ = input.read_le_int_at(0).unwrap();
        let _ = input.read_le_long_at(0).unwrap();
        assert_eq!(input.position(), before);
    }
}
