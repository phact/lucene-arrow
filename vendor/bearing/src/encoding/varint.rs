// SPDX-License-Identifier: Apache-2.0

//! Variable-length integer encoding/decoding (VInt, VLong).
//!
//! Uses 7 bits per byte with the high bit as a continuation flag.
//! Small values encode in fewer bytes, making them space-efficient for
//! fields that are typically small (e.g., string lengths, doc deltas).
//!
//! Writers take `&mut dyn Write`. Readers come in two flavors:
//!
//! Cursor-based ([`read_vint_cursor`], [`read_vlong`], [`read_zint`]) take
//! `&mut Cursor<&[u8]>` and use slice access via [`BufRead::fill_buf`] +
//! [`BufRead::consume`]. They are preferred on the read path — codec readers
//! hold an `IndexInput` which exposes a `Cursor`.
//!
//! Read-based ([`read_vint`]) takes `&mut dyn Read` for callers that cannot
//! expose a `Cursor<&[u8]>` — today `ByteSliceReader` on the indexing path and
//! the Read-based `encoding::string`/`pfor`/etc. decoders that call
//! `read_vint` internally.

use std::io;
use std::io::BufRead;
use std::io::Cursor;
use std::io::Read;
use std::io::Write;

use crate::encoding::zigzag;

const VINT_MAX_BYTES: usize = 5;
const VLONG_MAX_BYTES: usize = 10;

/// Writes a variable-length integer (1-5 bytes).
///
/// Uses 7 bits per byte. High bit = continuation.
pub fn write_vint(out: &mut dyn Write, i: i32) -> io::Result<()> {
    let mut val = i as u32;
    while (val & !0x7F) != 0 {
        out.write_all(&[((val & 0x7F) | 0x80) as u8])?;
        val >>= 7;
    }
    out.write_all(&[val as u8])
}

/// Reads a variable-length integer (1-5 bytes) from a [`Read`]-based source.
///
/// Prefer [`read_vint_cursor`] when the caller already holds a
/// `Cursor<&[u8]>`; the Cursor form avoids per-byte `read_exact` dispatch.
pub fn read_vint(reader: &mut dyn Read) -> io::Result<i32> {
    let mut buf = [0u8; 1];
    let mut result = 0i32;
    let mut shift = 0;
    loop {
        reader.read_exact(&mut buf)?;
        let b = buf[0] as i32;
        result |= (b & 0x7F) << shift;
        if b & 0x80 == 0 {
            break;
        }
        shift += 7;
    }
    Ok(result)
}

/// Reads a variable-length integer (1-5 bytes) directly from a
/// `Cursor<&[u8]>`, advancing its position by the bytes consumed on success.
/// Cursor state is undefined on error; the caller is expected to abort the
/// surrounding read.
pub fn read_vint_cursor(cursor: &mut Cursor<&[u8]>) -> io::Result<i32> {
    let buf = cursor.fill_buf()?;
    let mut result: i32 = 0;
    let mut shift = 0;
    for (i, &b) in buf.iter().take(VINT_MAX_BYTES).enumerate() {
        result |= ((b & 0x7F) as i32) << shift;
        if b & 0x80 == 0 {
            cursor.consume(i + 1);
            return Ok(result);
        }
        shift += 7;
    }
    Err(truncated_or_malformed(buf.len(), VINT_MAX_BYTES, "VInt"))
}

/// Writes a variable-length long (1-9 bytes).
///
/// Uses 7 bits per byte. High bit = continuation.
pub fn write_vlong(out: &mut dyn Write, i: i64) -> io::Result<()> {
    let mut val = i as u64;
    while (val & !0x7F) != 0 {
        out.write_all(&[((val & 0x7F) | 0x80) as u8])?;
        val >>= 7;
    }
    out.write_all(&[val as u8])
}

/// Reads a variable-length long (1-10 bytes) directly from a
/// `Cursor<&[u8]>`, advancing its position by the bytes consumed on success.
/// Cursor state is undefined on error; the caller is expected to abort the
/// surrounding read.
pub fn read_vlong(cursor: &mut Cursor<&[u8]>) -> io::Result<i64> {
    let buf = cursor.fill_buf()?;
    let mut result: i64 = 0;
    let mut shift = 0;
    for (i, &b) in buf.iter().take(VLONG_MAX_BYTES).enumerate() {
        result |= ((b & 0x7F) as i64) << shift;
        if b & 0x80 == 0 {
            cursor.consume(i + 1);
            return Ok(result);
        }
        shift += 7;
    }
    Err(truncated_or_malformed(buf.len(), VLONG_MAX_BYTES, "VLong"))
}

/// Writes a zigzag-encoded variable-length int.
pub fn write_zint(out: &mut dyn Write, i: i32) -> io::Result<()> {
    write_vint(out, zigzag::encode_i32(i))
}

/// Reads a zigzag-encoded variable-length int directly from a
/// `Cursor<&[u8]>`.
pub fn read_zint(cursor: &mut Cursor<&[u8]>) -> io::Result<i32> {
    Ok(zigzag::decode_i32(read_vint_cursor(cursor)?))
}

/// Writes a zigzag-encoded variable-length long.
pub fn write_zlong(out: &mut dyn Write, i: i64) -> io::Result<()> {
    let encoded = zigzag::encode_i64(i);
    write_signed_vlong(out, encoded)
}

/// Writes a variable-length long that may be negative (used by [`write_zlong`]).
pub fn write_signed_vlong(out: &mut dyn Write, mut i: i64) -> io::Result<()> {
    while (i & !0x7Fi64) != 0 {
        out.write_all(&[((i & 0x7F) | 0x80) as u8])?;
        i = ((i as u64) >> 7) as i64;
    }
    out.write_all(&[i as u8])
}

fn truncated_or_malformed(buf_len: usize, limit: usize, label: &str) -> io::Error {
    if buf_len < limit {
        io::Error::new(io::ErrorKind::UnexpectedEof, format!("truncated {label}"))
    } else {
        io::Error::other(format!("{label} exceeds {limit} bytes"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---------- write_vint ----------

    #[test]
    fn test_write_vint() {
        let mut buf = Vec::new();
        write_vint(&mut buf, 0).unwrap();
        assert_eq!(buf, [0x00]);

        let mut buf = Vec::new();
        write_vint(&mut buf, 127).unwrap();
        assert_eq!(buf, [0x7F]);

        let mut buf = Vec::new();
        write_vint(&mut buf, 128).unwrap();
        assert_eq!(buf, [0x80, 0x01]);

        let mut buf = Vec::new();
        write_vint(&mut buf, 16383).unwrap();
        assert_eq!(buf, [0xFF, 0x7F]);

        let mut buf = Vec::new();
        write_vint(&mut buf, 16384).unwrap();
        assert_eq!(buf, [0x80, 0x80, 0x01]);
    }

    #[test]
    fn test_write_vint_sizes() {
        let mut buf = Vec::new();
        write_vint(&mut buf, 0).unwrap();
        assert_len_eq_x!(&buf, 1);

        buf.clear();
        write_vint(&mut buf, 127).unwrap();
        assert_len_eq_x!(&buf, 1);

        buf.clear();
        write_vint(&mut buf, 128).unwrap();
        assert_len_eq_x!(&buf, 2);
    }

    // ---------- read_vint (Read-based) ----------

    #[test]
    fn test_read_vint_roundtrip() {
        let test_values = [0, 1, 127, 128, 255, 256, 16383, 16384, 0x7FFF_FFFF, -1];
        for &val in &test_values {
            let mut buf = Vec::new();
            write_vint(&mut buf, val).unwrap();
            let mut cursor = &buf[..];
            let decoded = read_vint(&mut cursor).unwrap();
            assert_eq!(decoded, val, "round-trip failed for {val}");
            assert_is_empty!(cursor);
        }
    }

    // ---------- read_vint_cursor ----------

    #[test]
    fn vint_cursor_roundtrip_values() {
        for &value in &[0i32, 1, 127, 128, 255, 16383, 16384, i32::MAX, -1] {
            let mut buf = Vec::new();
            write_vint(&mut buf, value).unwrap();
            let mut cursor = Cursor::new(&buf[..]);
            assert_eq!(
                read_vint_cursor(&mut cursor).unwrap(),
                value,
                "value = {value}"
            );
            assert_eq!(
                cursor.position() as usize,
                buf.len(),
                "cursor fully consumed for {value}"
            );
        }
    }

    #[test]
    fn vint_cursor_empty_input_errors() {
        let mut cursor = Cursor::new(&[][..]);
        assert_err!(read_vint_cursor(&mut cursor));
    }

    #[test]
    fn vint_cursor_truncated_continuation_errors() {
        let mut cursor = Cursor::new(&[0x80, 0x80][..]);
        assert_err!(read_vint_cursor(&mut cursor));
    }

    #[test]
    fn vint_cursor_exceeds_max_bytes_errors() {
        let mut cursor = Cursor::new(&[0x80, 0x80, 0x80, 0x80, 0x80, 0x80][..]);
        assert_err!(read_vint_cursor(&mut cursor));
    }

    #[test]
    fn vint_cursor_consecutive_reads_advance_position() {
        let mut buf = Vec::new();
        write_vint(&mut buf, 42).unwrap();
        write_vint(&mut buf, 128).unwrap();
        write_vint(&mut buf, i32::MAX).unwrap();
        let mut cursor = Cursor::new(&buf[..]);
        assert_eq!(read_vint_cursor(&mut cursor).unwrap(), 42);
        assert_eq!(read_vint_cursor(&mut cursor).unwrap(), 128);
        assert_eq!(read_vint_cursor(&mut cursor).unwrap(), i32::MAX);
        assert_eq!(cursor.position() as usize, buf.len());
    }

    // ---------- write_vlong ----------

    #[test]
    fn test_write_vlong() {
        let mut buf = Vec::new();
        write_vlong(&mut buf, 0).unwrap();
        assert_eq!(buf, [0x00]);

        let mut buf = Vec::new();
        write_vlong(&mut buf, 128).unwrap();
        assert_eq!(buf, [0x80, 0x01]);
    }

    // ---------- read_vlong (Cursor-based) ----------

    #[test]
    fn vlong_cursor_roundtrip_values() {
        for &value in &[0i64, 1, 127, 128, 255, 16383, 16384, i64::MAX] {
            let mut buf = Vec::new();
            write_vlong(&mut buf, value).unwrap();
            let mut cursor = Cursor::new(&buf[..]);
            assert_eq!(read_vlong(&mut cursor).unwrap(), value, "value = {value}");
            assert_eq!(
                cursor.position() as usize,
                buf.len(),
                "cursor fully consumed for {value}"
            );
        }
    }

    #[test]
    fn vlong_cursor_empty_input_errors() {
        let mut cursor = Cursor::new(&[][..]);
        assert_err!(read_vlong(&mut cursor));
    }

    #[test]
    fn vlong_cursor_truncated_continuation_errors() {
        let mut cursor = Cursor::new(&[0x80, 0x80, 0x80][..]);
        assert_err!(read_vlong(&mut cursor));
    }

    #[test]
    fn vlong_cursor_exceeds_max_bytes_errors() {
        let mut cursor = Cursor::new(&[0x80; 11][..]);
        assert_err!(read_vlong(&mut cursor));
    }

    #[test]
    fn vlong_cursor_consecutive_reads_advance_position() {
        let mut buf = Vec::new();
        write_vlong(&mut buf, 42).unwrap();
        write_vlong(&mut buf, 128).unwrap();
        write_vlong(&mut buf, i64::MAX).unwrap();
        let mut cursor = Cursor::new(&buf[..]);
        assert_eq!(read_vlong(&mut cursor).unwrap(), 42);
        assert_eq!(read_vlong(&mut cursor).unwrap(), 128);
        assert_eq!(read_vlong(&mut cursor).unwrap(), i64::MAX);
        assert_eq!(cursor.position() as usize, buf.len());
    }

    // ---------- write_zint ----------

    #[test]
    fn test_write_zint() {
        let mut buf = Vec::new();
        write_zint(&mut buf, 0).unwrap();
        assert_eq!(buf, [0x00]);

        let mut buf = Vec::new();
        write_zint(&mut buf, -1).unwrap();
        assert_eq!(buf, [0x01]);

        let mut buf = Vec::new();
        write_zint(&mut buf, 1).unwrap();
        assert_eq!(buf, [0x02]);

        let mut buf = Vec::new();
        write_zint(&mut buf, i32::MIN).unwrap();
        assert_eq!(buf, [0xFF, 0xFF, 0xFF, 0xFF, 0x0F]);

        let mut buf = Vec::new();
        write_zint(&mut buf, i32::MAX).unwrap();
        assert_eq!(buf, [0xFE, 0xFF, 0xFF, 0xFF, 0x0F]);
    }

    // ---------- read_zint (Cursor-based) ----------

    #[test]
    fn zint_cursor_roundtrip_values() {
        for &value in &[0i32, 1, -1, 127, -128, i32::MIN, i32::MAX, 42, -42] {
            let mut buf = Vec::new();
            write_zint(&mut buf, value).unwrap();
            let mut cursor = Cursor::new(&buf[..]);
            assert_eq!(read_zint(&mut cursor).unwrap(), value, "value = {value}");
            assert_eq!(
                cursor.position() as usize,
                buf.len(),
                "cursor fully consumed for {value}"
            );
        }
    }

    #[test]
    fn zint_cursor_propagates_vint_error() {
        let mut cursor = Cursor::new(&[][..]);
        assert_err!(read_zint(&mut cursor));
    }

    // ---------- write_zlong / write_signed_vlong ----------

    #[test]
    fn test_write_zlong() {
        let mut buf = Vec::new();
        write_zlong(&mut buf, 0).unwrap();
        assert_eq!(buf, [0x00]);

        let mut buf = Vec::new();
        write_zlong(&mut buf, -1).unwrap();
        assert_eq!(buf, [0x01]);

        let mut buf = Vec::new();
        write_zlong(&mut buf, 1).unwrap();
        assert_eq!(buf, [0x02]);

        let mut buf = Vec::new();
        write_zlong(&mut buf, i64::MIN).unwrap();
        assert_len_eq_x!(&buf, 10);
        for &b in &buf[..9] {
            assert_eq!(b, 0xFF);
        }
        assert_eq!(buf[9], 0x01);

        let mut buf = Vec::new();
        write_zlong(&mut buf, i64::MAX).unwrap();
        assert_len_eq_x!(&buf, 10);
        assert_eq!(buf[0], 0xFE);
        for &b in &buf[1..9] {
            assert_eq!(b, 0xFF);
        }
        assert_eq!(buf[9], 0x01);
    }
}
