// SPDX-License-Identifier: Apache-2.0

//! VInt-prefixed string encoding/decoding.
//!
//! Strings are encoded as a VInt byte length followed by the UTF-8 bytes.
//! Collections (sets, maps) are encoded as a VInt count followed by each element.
//!
//! Writers take `&mut dyn Write`. Readers take `&mut Cursor<&[u8]>`.
//! Cursor state is undefined on error; the caller is expected to abort the
//! surrounding read.

use std::collections::HashMap;
use std::io;
use std::io::Cursor;
use std::io::Read;
use std::io::Write;

use crate::encoding::varint;

/// Writes a string as VInt-encoded byte length followed by UTF-8 bytes.
pub fn write_string(out: &mut dyn Write, s: &str) -> io::Result<()> {
    let bytes = s.as_bytes();
    let len = i32::try_from(bytes.len())
        .map_err(|_| io::Error::other("string length exceeds i32::MAX"))?;
    varint::write_vint(out, len)?;
    out.write_all(bytes)
}

/// Reads a VInt-prefixed UTF-8 string from a cursor.
pub fn read_string(cursor: &mut Cursor<&[u8]>) -> io::Result<String> {
    let len = varint::read_vint_cursor(cursor)?;
    let len = usize::try_from(len).map_err(|_| io::Error::other("negative string length"))?;
    let mut buf = vec![0u8; len];
    cursor.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| io::Error::other(format!("invalid UTF-8: {e}")))
}

/// Writes a set of strings: VInt count followed by each string.
pub fn write_set_of_strings(out: &mut dyn Write, set: &[String]) -> io::Result<()> {
    let count =
        i32::try_from(set.len()).map_err(|_| io::Error::other("set size exceeds i32::MAX"))?;
    varint::write_vint(out, count)?;
    for s in set {
        write_string(out, s)?;
    }
    Ok(())
}

/// Reads a VInt count followed by that many UTF-8 strings.
pub fn read_set_of_strings(cursor: &mut Cursor<&[u8]>) -> io::Result<Vec<String>> {
    let count = read_count(cursor, "set")?;
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        out.push(read_string(cursor)?);
    }
    Ok(out)
}

/// Writes a map of strings: VInt count followed by key-value pairs.
pub fn write_map_of_strings(out: &mut dyn Write, map: &HashMap<String, String>) -> io::Result<()> {
    let count =
        i32::try_from(map.len()).map_err(|_| io::Error::other("map size exceeds i32::MAX"))?;
    varint::write_vint(out, count)?;
    for (k, v) in map {
        write_string(out, k)?;
        write_string(out, v)?;
    }
    Ok(())
}

/// Reads a VInt count followed by that many key/value UTF-8 string pairs.
pub fn read_map_of_strings(cursor: &mut Cursor<&[u8]>) -> io::Result<HashMap<String, String>> {
    let count = read_count(cursor, "map")?;
    let mut out = HashMap::with_capacity(count);
    for _ in 0..count {
        let key = read_string(cursor)?;
        let value = read_string(cursor)?;
        out.insert(key, value);
    }
    Ok(out)
}

fn read_count(cursor: &mut Cursor<&[u8]>, label: &str) -> io::Result<usize> {
    let count = varint::read_vint_cursor(cursor)?;
    usize::try_from(count).map_err(|_| io::Error::other(format!("negative {label} count")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoding::varint::write_vint;

    fn encode<F>(f: F) -> Vec<u8>
    where
        F: FnOnce(&mut Vec<u8>) -> io::Result<()>,
    {
        let mut buf = Vec::new();
        f(&mut buf).unwrap();
        buf
    }

    // ---------- write_string ----------

    #[test]
    fn test_write_string() {
        let mut buf = Vec::new();
        write_string(&mut buf, "hello").unwrap();
        // VInt(5) = 0x05, then "hello" as UTF-8
        assert_eq!(buf, [0x05, b'h', b'e', b'l', b'l', b'o']);
    }

    #[test]
    fn test_write_set_of_strings_empty() {
        let mut buf = Vec::new();
        write_set_of_strings(&mut buf, &[]).unwrap();
        assert_eq!(buf, [0x00]);
    }

    #[test]
    fn test_write_set_of_strings() {
        let mut buf = Vec::new();
        let set = vec!["hello".to_string(), "world".to_string()];
        write_set_of_strings(&mut buf, &set).unwrap();
        let mut expected = vec![0x02];
        expected.extend_from_slice(&[0x05, b'h', b'e', b'l', b'l', b'o']);
        expected.extend_from_slice(&[0x05, b'w', b'o', b'r', b'l', b'd']);
        assert_eq!(buf, expected);
    }

    #[test]
    fn test_write_map_of_strings_empty() {
        let mut buf = Vec::new();
        let map = HashMap::new();
        write_map_of_strings(&mut buf, &map).unwrap();
        assert_eq!(buf, [0x00]);
    }

    #[test]
    fn test_write_map_of_strings() {
        let mut buf = Vec::new();
        let mut map = HashMap::new();
        map.insert("key".to_string(), "val".to_string());
        write_map_of_strings(&mut buf, &map).unwrap();
        assert_eq!(buf[0], 0x01);
        assert_eq!(&buf[1..], &[0x03, b'k', b'e', b'y', 0x03, b'v', b'a', b'l']);
    }

    // ---------- read_string ----------

    #[test]
    fn string_roundtrip_values() {
        for s in &["", "a", "hello", "hello world", "\u{00e9}\u{00e8}"] {
            let buf = encode(|b| write_string(b, s));
            let mut cursor = Cursor::new(&buf[..]);
            assert_eq!(read_string(&mut cursor).unwrap(), *s, "value = {s:?}");
            assert_eq!(
                cursor.position() as usize,
                buf.len(),
                "cursor fully consumed for {s:?}"
            );
        }
    }

    #[test]
    fn string_empty_input_errors() {
        let mut cursor = Cursor::new(&[][..]);
        assert_err!(read_string(&mut cursor));
    }

    #[test]
    fn string_negative_length_errors() {
        let buf = encode(|b| write_vint(b, -1));
        let mut cursor = Cursor::new(&buf[..]);
        assert_err!(read_string(&mut cursor));
    }

    #[test]
    fn string_truncated_body_errors() {
        let buf = [0x05u8, b'a', b'b', b'c'];
        let mut cursor = Cursor::new(&buf[..]);
        assert_err!(read_string(&mut cursor));
    }

    #[test]
    fn string_invalid_utf8_errors() {
        let buf = [0x02u8, 0xFF, 0xFE];
        let mut cursor = Cursor::new(&buf[..]);
        assert_err!(read_string(&mut cursor));
    }

    #[test]
    fn string_consecutive_reads_advance_cursor() {
        let mut buf = Vec::new();
        write_string(&mut buf, "alpha").unwrap();
        write_string(&mut buf, "beta").unwrap();
        write_string(&mut buf, "gamma").unwrap();
        let mut cursor = Cursor::new(&buf[..]);
        assert_eq!(read_string(&mut cursor).unwrap(), "alpha");
        assert_eq!(read_string(&mut cursor).unwrap(), "beta");
        assert_eq!(read_string(&mut cursor).unwrap(), "gamma");
        assert_eq!(cursor.position() as usize, buf.len());
    }

    // ---------- read_set_of_strings ----------

    #[test]
    fn set_roundtrip_empty() {
        let buf = encode(|b| write_set_of_strings(b, &[]));
        let mut cursor = Cursor::new(&buf[..]);
        let decoded = read_set_of_strings(&mut cursor).unwrap();
        assert_is_empty!(&decoded);
        assert_eq!(cursor.position() as usize, buf.len());
    }

    #[test]
    fn set_roundtrip_single() {
        let set = vec!["only".to_string()];
        let buf = encode(|b| write_set_of_strings(b, &set));
        let mut cursor = Cursor::new(&buf[..]);
        assert_eq!(read_set_of_strings(&mut cursor).unwrap(), set);
        assert_eq!(cursor.position() as usize, buf.len());
    }

    #[test]
    fn set_roundtrip_multiple() {
        let set = vec!["one".to_string(), "two".to_string(), "three".to_string()];
        let buf = encode(|b| write_set_of_strings(b, &set));
        let mut cursor = Cursor::new(&buf[..]);
        assert_eq!(read_set_of_strings(&mut cursor).unwrap(), set);
        assert_eq!(cursor.position() as usize, buf.len());
    }

    #[test]
    fn set_empty_input_errors() {
        let mut cursor = Cursor::new(&[][..]);
        assert_err!(read_set_of_strings(&mut cursor));
    }

    #[test]
    fn set_negative_count_errors() {
        let buf = encode(|b| write_vint(b, -1));
        let mut cursor = Cursor::new(&buf[..]);
        assert_err!(read_set_of_strings(&mut cursor));
    }

    #[test]
    fn set_truncated_mid_element_errors() {
        let mut buf = Vec::new();
        write_vint(&mut buf, 3).unwrap();
        write_string(&mut buf, "one").unwrap();
        buf.push(0x05);
        buf.extend_from_slice(b"ab");
        let mut cursor = Cursor::new(&buf[..]);
        assert_err!(read_set_of_strings(&mut cursor));
    }

    // ---------- read_map_of_strings ----------

    #[test]
    fn map_roundtrip_empty() {
        let buf = encode(|b| write_map_of_strings(b, &HashMap::new()));
        let mut cursor = Cursor::new(&buf[..]);
        let decoded = read_map_of_strings(&mut cursor).unwrap();
        assert_is_empty!(&decoded);
        assert_eq!(cursor.position() as usize, buf.len());
    }

    #[test]
    fn map_roundtrip_single() {
        let mut map = HashMap::new();
        map.insert("key".to_string(), "val".to_string());
        let buf = encode(|b| write_map_of_strings(b, &map));
        let mut cursor = Cursor::new(&buf[..]);
        assert_eq!(read_map_of_strings(&mut cursor).unwrap(), map);
        assert_eq!(cursor.position() as usize, buf.len());
    }

    #[test]
    fn map_roundtrip_multiple() {
        let mut map = HashMap::new();
        map.insert("k1".to_string(), "v1".to_string());
        map.insert("k2".to_string(), "v2".to_string());
        map.insert("k3".to_string(), "v3".to_string());
        let buf = encode(|b| write_map_of_strings(b, &map));
        let mut cursor = Cursor::new(&buf[..]);
        assert_eq!(read_map_of_strings(&mut cursor).unwrap(), map);
        assert_eq!(cursor.position() as usize, buf.len());
    }

    #[test]
    fn map_empty_input_errors() {
        let mut cursor = Cursor::new(&[][..]);
        assert_err!(read_map_of_strings(&mut cursor));
    }

    #[test]
    fn map_negative_count_errors() {
        let buf = encode(|b| write_vint(b, -1));
        let mut cursor = Cursor::new(&buf[..]);
        assert_err!(read_map_of_strings(&mut cursor));
    }

    #[test]
    fn map_truncated_mid_value_errors() {
        let mut buf = Vec::new();
        write_vint(&mut buf, 2).unwrap();
        write_string(&mut buf, "k1").unwrap();
        write_string(&mut buf, "v1").unwrap();
        write_string(&mut buf, "k2").unwrap();
        buf.push(0x04);
        buf.push(b'x');
        let mut cursor = Cursor::new(&buf[..]);
        assert_err!(read_map_of_strings(&mut cursor));
    }
}
