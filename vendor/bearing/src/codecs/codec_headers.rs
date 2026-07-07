// SPDX-License-Identifier: Apache-2.0

//! Codec header read + write helpers and the segment-ID type alias.
//!
//! Owns everything related to the codec header wire format:
//!   - [`CODEC_MAGIC`] / [`ID_LENGTH`] constants
//!   - [`write_header`] / [`write_index_header`] (write side)
//!   - [`check_header`] / [`check_index_header`] (read side, against [`IndexInput`])
//!   - [`header_length`] / [`index_header_length`] (sizing helpers used by both)

use std::io;

use log::debug;

use crate::encoding::write_encoding::WriteEncoding;
use crate::store::{DataOutput, IndexInput};

/// Magic number written at the start of every codec header (big-endian).
pub(crate) const CODEC_MAGIC: i32 = 0x3FD76C17_u32 as i32;

/// Length of the segment ID in bytes.
pub(crate) const ID_LENGTH: usize = 16;

/// Reads and validates a codec header, returning the version.
///
/// Checks, in order:
///   1. Magic matches [`CODEC_MAGIC`].
///   2. Codec name matches `codec`.
///   3. Version is in `[min_version, max_version]`.
pub(crate) fn check_header(
    input: &mut IndexInput<'_>,
    codec: &str,
    min_version: i32,
    max_version: i32,
) -> io::Result<i32> {
    let actual_magic = input.read_be_int()?;
    if actual_magic != CODEC_MAGIC {
        return Err(io::Error::other(format!(
            "codec header mismatch: expected 0x{CODEC_MAGIC:08X}, got 0x{actual_magic:08X}"
        )));
    }

    let actual_codec = input.read_string()?;
    if actual_codec != codec {
        return Err(io::Error::other(format!(
            "codec mismatch: expected {codec:?}, got {actual_codec:?}"
        )));
    }

    let version = input.read_be_int()?;
    if version < min_version || version > max_version {
        return Err(io::Error::other(format!(
            "version {version} out of range [{min_version}, {max_version}] for codec {codec:?}"
        )));
    }

    Ok(version)
}

/// Reads and validates an index header (header + segment ID + suffix), returning the version.
pub(crate) fn check_index_header(
    input: &mut IndexInput<'_>,
    codec: &str,
    min_version: i32,
    max_version: i32,
    expected_id: &[u8; ID_LENGTH],
    expected_suffix: &str,
) -> io::Result<i32> {
    let version = check_header(input, codec, min_version, max_version)?;

    let mut actual_id = [0u8; ID_LENGTH];
    input.read_bytes(&mut actual_id)?;
    if actual_id != *expected_id {
        return Err(io::Error::other(format!(
            "segment ID mismatch: expected {expected_id:02x?}, got {actual_id:02x?}"
        )));
    }

    let suffix_len = input.read_byte()? as usize;
    let mut suffix_bytes = vec![0u8; suffix_len];
    input.read_bytes(&mut suffix_bytes)?;
    let actual_suffix =
        String::from_utf8(suffix_bytes).map_err(|e| io::Error::other(e.to_string()))?;
    if actual_suffix != expected_suffix {
        return Err(io::Error::other(format!(
            "suffix mismatch: expected {expected_suffix:?}, got {actual_suffix:?}"
        )));
    }

    Ok(version)
}

/// Writes a codec header.
///
/// Format (all big-endian):
///   - 4 bytes: `CODEC_MAGIC` (BE int)
///   - N bytes: codec name (VInt length + UTF-8 string)
///   - 4 bytes: version (BE int)
///
/// Returns the number of bytes written (= 9 + codec.len()).
pub(crate) fn write_header(
    mut out: &mut dyn DataOutput,
    codec: &str,
    version: i32,
) -> io::Result<usize> {
    validate_codec_name(codec)?;
    out.write_be_int(CODEC_MAGIC)?;
    out.write_string(codec)?;
    out.write_be_int(version)?;
    Ok(header_length(codec))
}

/// Writes an index header (header + segment ID + suffix).
///
/// Format:
///   - header (`write_header`)
///   - 16 bytes: segment ID
///   - 1 byte: suffix length
///   - N bytes: suffix bytes
///
/// Returns the number of bytes written.
pub(crate) fn write_index_header(
    out: &mut dyn DataOutput,
    codec: &str,
    version: i32,
    id: &[u8; ID_LENGTH],
    suffix: &str,
) -> io::Result<usize> {
    write_header(out, codec, version)?;
    out.write_all(id)?;
    let suffix_bytes = suffix.as_bytes();
    if suffix_bytes.len() > 255 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("suffix too long: {}", suffix_bytes.len()),
        ));
    }
    out.write_byte(suffix_bytes.len() as u8)?;
    out.write_all(suffix_bytes)?;
    debug!(
        "write_index_header: codec={codec:?}, version={version}, suffix={suffix:?}, id={id:02x?}"
    );
    Ok(index_header_length(codec, suffix))
}

/// Returns the byte length of a codec header for the given codec name.
/// 4 (magic) + 1+ (vint string length) + codec.len() + 4 (version).
/// For ASCII codec names < 128 chars, the VInt is 1 byte.
pub(crate) fn header_length(codec: &str) -> usize {
    4 + vint_size(codec.len() as u32) + codec.len() + 4
}

/// Returns the byte length of an index header.
pub(crate) fn index_header_length(codec: &str, suffix: &str) -> usize {
    header_length(codec) + ID_LENGTH + 1 + suffix.len()
}

/// Returns the number of bytes needed to encode a value as a VInt.
fn vint_size(mut val: u32) -> usize {
    let mut size = 1;
    while val > 0x7F {
        val >>= 7;
        size += 1;
    }
    size
}

/// Validates that a codec name is simple ASCII and not too long.
fn validate_codec_name(codec: &str) -> io::Result<()> {
    if codec.len() >= 128 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("codec name too long: {}", codec.len()),
        ));
    }
    if !codec.bytes().all(|b| b.is_ascii_graphic() || b == b' ') {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "codec name must be simple ASCII",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use assertables::*;

    /// Builds a valid codec header: magic + vint length + name + version.
    fn build_header(codec: &str, version: i32) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&CODEC_MAGIC.to_be_bytes());
        // VInt of length 1: codec name length is < 128 in our tests
        let name_bytes = codec.as_bytes();
        assert_len_lt_x!(name_bytes, 128);
        buf.push(name_bytes.len() as u8);
        buf.extend_from_slice(name_bytes);
        buf.extend_from_slice(&version.to_be_bytes());
        buf
    }

    /// Builds a valid index header: header + 16-byte ID + 1-byte suffix length + suffix.
    fn build_index_header(
        codec: &str,
        version: i32,
        id: &[u8; ID_LENGTH],
        suffix: &str,
    ) -> Vec<u8> {
        let mut buf = build_header(codec, version);
        buf.extend_from_slice(id);
        let suffix_bytes = suffix.as_bytes();
        assert_len_le_x!(suffix_bytes, 255);
        buf.push(suffix_bytes.len() as u8);
        buf.extend_from_slice(suffix_bytes);
        buf
    }

    #[test]
    fn check_header_returns_version_on_match() {
        let bytes = build_header("MyCodec", 7);
        let mut input = IndexInput::unnamed(&bytes);
        assert_ok_eq_x!(check_header(&mut input, "MyCodec", 5, 10), 7);
    }

    #[test]
    fn check_header_min_max_boundaries() {
        let bytes = build_header("MyCodec", 5);
        let mut input = IndexInput::unnamed(&bytes);
        assert_ok!(check_header(&mut input, "MyCodec", 5, 10));

        let bytes = build_header("MyCodec", 10);
        let mut input = IndexInput::unnamed(&bytes);
        assert_ok!(check_header(&mut input, "MyCodec", 5, 10));
    }

    #[test]
    fn check_header_bad_magic() {
        let mut bytes = build_header("MyCodec", 7);
        bytes[0] ^= 0xFF;
        let mut input = IndexInput::unnamed(&bytes);
        assert_err!(check_header(&mut input, "MyCodec", 5, 10));
    }

    #[test]
    fn check_header_wrong_codec_name() {
        let bytes = build_header("MyCodec", 7);
        let mut input = IndexInput::unnamed(&bytes);
        assert_err!(check_header(&mut input, "OtherCodec", 5, 10));
    }

    #[test]
    fn check_header_version_below_min() {
        let bytes = build_header("MyCodec", 4);
        let mut input = IndexInput::unnamed(&bytes);
        assert_err!(check_header(&mut input, "MyCodec", 5, 10));
    }

    #[test]
    fn check_header_version_above_max() {
        let bytes = build_header("MyCodec", 11);
        let mut input = IndexInput::unnamed(&bytes);
        assert_err!(check_header(&mut input, "MyCodec", 5, 10));
    }

    #[test]
    fn check_index_header_returns_version_on_match() {
        let id = [0xABu8; ID_LENGTH];
        let bytes = build_index_header("MyCodec", 7, &id, "suffix1");
        let mut input = IndexInput::unnamed(&bytes);
        let version = check_index_header(&mut input, "MyCodec", 5, 10, &id, "suffix1").unwrap();
        assert_eq!(version, 7);
    }

    #[test]
    fn check_index_header_id_mismatch() {
        let id = [0xABu8; ID_LENGTH];
        let other = [0xCDu8; ID_LENGTH];
        let bytes = build_index_header("MyCodec", 7, &id, "suffix1");
        let mut input = IndexInput::unnamed(&bytes);
        assert_err!(check_index_header(
            &mut input, "MyCodec", 5, 10, &other, "suffix1"
        ));
    }

    #[test]
    fn check_index_header_suffix_mismatch() {
        let id = [0xABu8; ID_LENGTH];
        let bytes = build_index_header("MyCodec", 7, &id, "suffix1");
        let mut input = IndexInput::unnamed(&bytes);
        assert_err!(check_index_header(
            &mut input, "MyCodec", 5, 10, &id, "suffix2"
        ));
    }

    #[test]
    fn check_index_header_empty_suffix() {
        let id = [0u8; ID_LENGTH];
        let bytes = build_index_header("MyCodec", 7, &id, "");
        let mut input = IndexInput::unnamed(&bytes);
        assert_ok!(check_index_header(&mut input, "MyCodec", 5, 10, &id, ""));
    }

    // Write-side tests

    use crate::store::memory::MemoryIndexOutput;

    #[test]
    fn test_header_length() {
        assert_eq!(header_length("FooBar"), 9 + 6);
    }

    #[test]
    fn test_write_header() {
        let mut out = MemoryIndexOutput::new("test".to_string());
        let len = write_header(&mut out, "FooBar", 5).unwrap();
        let bytes = out.bytes();

        assert_eq!(len, 15);
        assert_len_eq_x!(&bytes, 15);

        // Magic (BE): 0x3fd76c17
        assert_eq!(bytes[0], 0x3f);
        assert_eq!(bytes[1], 0xd7);
        assert_eq!(bytes[2], 0x6c);
        assert_eq!(bytes[3], 0x17);

        // String: VInt(6) = 0x06, then "FooBar"
        assert_eq!(bytes[4], 6);
        assert_eq!(&bytes[5..11], b"FooBar");

        // Version (BE): 5
        assert_eq!(bytes[11], 0);
        assert_eq!(bytes[12], 0);
        assert_eq!(bytes[13], 0);
        assert_eq!(bytes[14], 5);
    }

    #[test]
    fn test_write_index_header() {
        let mut out = MemoryIndexOutput::new("test".to_string());
        let id = [1u8; 16];
        let len = write_index_header(&mut out, "FooBar", 5, &id, "xyz").unwrap();
        let bytes = out.bytes();

        // header(15) + 16(id) + 1(suffix len) + 3(suffix) = 35
        assert_eq!(len, 35);
        assert_len_eq_x!(&bytes, 35);

        // ID starts at byte 15
        assert_eq!(&bytes[15..31], &[1u8; 16]);

        // Suffix length at byte 31
        assert_eq!(bytes[31], 3);

        // Suffix at bytes 32..35
        assert_eq!(&bytes[32..35], b"xyz");
    }

    #[test]
    fn test_validate_codec_name_empty() {
        // Empty codec name is valid in Java
        let mut out = MemoryIndexOutput::new("test".to_string());
        assert_ok!(write_header(&mut out, "", 0));
    }

    #[test]
    fn test_validate_codec_name_too_long() {
        let long_name: String = "a".repeat(128);
        let mut out = MemoryIndexOutput::new("test".to_string());
        assert_err!(write_header(&mut out, &long_name, 0));
    }

    #[test]
    fn test_validate_codec_name_non_ascii() {
        let mut out = MemoryIndexOutput::new("test".to_string());
        assert_err!(write_header(&mut out, "bad\x01name", 0));
    }

    #[test]
    fn test_index_header_suffix_too_long() {
        let mut out = MemoryIndexOutput::new("test".to_string());
        let id = [0u8; 16];
        let long_suffix: String = "x".repeat(256);
        assert_err!(write_index_header(&mut out, "Test", 1, &id, &long_suffix));
    }

    #[test]
    fn test_vint_size_multi_byte() {
        // header_length uses vint_size internally. A codec name of length 128+
        // would need 2 vint bytes, but validate_codec_name rejects >= 128.
        // So test via header_length with a 127-char name (vint = 1 byte).
        let name = "a".repeat(127);
        assert_eq!(header_length(&name), 4 + 1 + 127 + 4);
    }
}
