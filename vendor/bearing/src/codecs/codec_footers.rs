// SPDX-License-Identifier: Apache-2.0

//! Codec footer read + write helpers.
//!
//! Owns everything related to the codec footer wire format:
//!   - [`FOOTER_MAGIC`] / [`FOOTER_LENGTH`] constants
//!   - [`write_footer`] (write side)
//!   - [`verify_checksum`] / [`retrieve_checksum`] / [`retrieve_checksum_with_length`] (read side)
//!
//! Every codec-framed file ends with a 16-byte footer:
//!   - 4 bytes: footer magic (big-endian [`FOOTER_MAGIC`])
//!   - 4 bytes: algorithm ID (big-endian `0` = zlib CRC32)
//!   - 8 bytes: stored CRC32 (big-endian) over every preceding byte

use std::io;

use log::debug;

use crate::store::IndexOutput;
use crate::store::checksum::CRC32;

/// Footer length in bytes: 4 (magic) + 4 (algorithm ID) + 8 (stored CRC).
pub(crate) const FOOTER_LENGTH: usize = 16;

/// Footer magic (big-endian `i32`). Bitwise NOT of the codec header magic
/// (`0x3FD76C17`).
pub(crate) const FOOTER_MAGIC: i32 = !(0x3FD76C17_u32 as i32);

/// Writes the codec footer: magic (BE), algorithm ID 0 (BE), CRC32 (BE long).
/// The CRC32 covers all bytes written to the output before the footer, plus
/// the first 8 bytes of the footer itself (magic + algorithm).
pub(crate) fn write_footer(out: &mut dyn IndexOutput) -> io::Result<()> {
    out.write_be_int(FOOTER_MAGIC)?;
    out.write_be_int(0)?; // algorithm ID = 0 (zlib crc32)
    let checksum = out.checksum();
    debug!("write_footer: checksum=0x{checksum:08x}");
    out.write_be_long(checksum as i64)?;
    Ok(())
}

/// Verifies the trailing codec footer of `bytes`.
///
/// Checks, in order:
///   1. `bytes.len() >= FOOTER_LENGTH`.
///   2. Footer magic matches [`FOOTER_MAGIC`].
///   3. Algorithm ID is `0` (zlib CRC32).
///   4. The stored CRC equals a CRC32 computed over `bytes[..bytes.len() - 8]`.
///
/// Returns `Ok(())` on success. Returns the first failing check as `Err`.
pub(crate) fn verify_checksum(bytes: &[u8]) -> io::Result<()> {
    if bytes.len() < FOOTER_LENGTH {
        return Err(io::Error::other(format!(
            "file too short for footer: {} < {FOOTER_LENGTH}",
            bytes.len()
        )));
    }

    let footer_start = bytes.len() - FOOTER_LENGTH;
    let magic = read_be_i32(&bytes[footer_start..footer_start + 4]);
    if magic != FOOTER_MAGIC {
        return Err(io::Error::other(format!(
            "footer magic mismatch: expected 0x{:08X}, got 0x{:08X}",
            FOOTER_MAGIC as u32, magic as u32,
        )));
    }

    let algorithm_id = read_be_i32(&bytes[footer_start + 4..footer_start + 8]);
    if algorithm_id != 0 {
        return Err(io::Error::other(format!(
            "unsupported checksum algorithm: {algorithm_id}"
        )));
    }

    let stored_crc = read_be_i64(&bytes[footer_start + 8..footer_start + 16]) as u64;

    let mut crc = CRC32::new();
    crc.update(&bytes[..bytes.len() - 8]);
    let computed = crc.value();

    if computed != stored_crc {
        return Err(io::Error::other(format!(
            "checksum mismatch: stored=0x{stored_crc:016X}, computed=0x{computed:016X}",
        )));
    }

    Ok(())
}

/// Reads the stored CRC from the trailing footer without recomputing it.
///
/// Validates, in order:
///   1. `bytes.len() >= FOOTER_LENGTH`.
///   2. Footer magic matches [`FOOTER_MAGIC`].
///   3. Algorithm ID is `0` (zlib CRC32).
///   4. The stored CRC fits in 32 bits (high 32 bits are zero).
///
/// Returns the stored CRC value on success. This is a cheap O(1) sanity
/// check on the footer — it does NOT compute the CRC over the file body.
/// Use [`verify_checksum`] for full integrity validation.
pub(crate) fn retrieve_checksum(bytes: &[u8]) -> io::Result<i64> {
    if bytes.len() < FOOTER_LENGTH {
        return Err(io::Error::other(format!(
            "misplaced codec footer (file truncated?): length={} but footerLength=={FOOTER_LENGTH}",
            bytes.len()
        )));
    }

    let footer_start = bytes.len() - FOOTER_LENGTH;
    let magic = read_be_i32(&bytes[footer_start..footer_start + 4]);
    if magic != FOOTER_MAGIC {
        return Err(io::Error::other(format!(
            "codec footer mismatch (file truncated?): actual footer=0x{:08X} vs expected footer=0x{:08X}",
            magic as u32, FOOTER_MAGIC as u32,
        )));
    }

    let algorithm_id = read_be_i32(&bytes[footer_start + 4..footer_start + 8]);
    if algorithm_id != 0 {
        return Err(io::Error::other(format!(
            "codec footer mismatch: unknown algorithmID: {algorithm_id}"
        )));
    }

    let checksum = read_be_i64(&bytes[footer_start + 8..footer_start + 16]);
    if (checksum as u64) & 0xFFFF_FFFF_0000_0000 != 0 {
        return Err(io::Error::other(format!(
            "illegal CRC-32 checksum: {checksum}"
        )));
    }

    Ok(checksum)
}

/// Reads the stored CRC from the footer, also verifying that the file length
/// matches `expected_length`.
pub(crate) fn retrieve_checksum_with_length(bytes: &[u8], expected_length: i64) -> io::Result<i64> {
    if expected_length < FOOTER_LENGTH as i64 {
        return Err(io::Error::other(
            "expectedLength cannot be less than the footer length",
        ));
    }
    let actual = bytes.len() as i64;
    if actual < expected_length {
        return Err(io::Error::other(format!(
            "truncated file: length={actual} but expectedLength=={expected_length}"
        )));
    } else if actual > expected_length {
        return Err(io::Error::other(format!(
            "file too long: length={actual} but expectedLength=={expected_length}"
        )));
    }
    retrieve_checksum(bytes)
}

fn read_be_i32(bytes: &[u8]) -> i32 {
    let arr: [u8; 4] = bytes.try_into().expect("slice of length 4");
    i32::from_be_bytes(arr)
}

fn read_be_i64(bytes: &[u8]) -> i64 {
    let arr: [u8; 8] = bytes.try_into().expect("slice of length 8");
    i64::from_be_bytes(arr)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a valid codec-framed buffer: `data` followed by a correct footer.
    fn with_valid_footer(data: &[u8]) -> Vec<u8> {
        let mut buf = Vec::with_capacity(data.len() + FOOTER_LENGTH);
        buf.extend_from_slice(data);
        buf.extend_from_slice(&FOOTER_MAGIC.to_be_bytes());
        buf.extend_from_slice(&0i32.to_be_bytes());
        let mut crc = CRC32::new();
        crc.update(&buf);
        buf.extend_from_slice(&(crc.value() as i64).to_be_bytes());
        buf
    }

    #[test]
    fn valid_footer_over_nonempty_data() {
        let bytes = with_valid_footer(b"hello world");
        verify_checksum(&bytes).unwrap();
    }

    #[test]
    fn valid_footer_over_empty_data() {
        let bytes = with_valid_footer(b"");
        assert_len_eq_x!(&bytes, FOOTER_LENGTH);
        verify_checksum(&bytes).unwrap();
    }

    #[test]
    fn too_short_errors() {
        assert_err!(verify_checksum(&[]));
        assert_err!(verify_checksum(&[0u8; 15]));
    }

    #[test]
    fn bad_magic_errors() {
        let mut bytes = with_valid_footer(b"hello");
        let footer_start = bytes.len() - FOOTER_LENGTH;
        bytes[footer_start] ^= 0xFF;
        assert_err!(verify_checksum(&bytes));
    }

    #[test]
    fn bad_algorithm_errors() {
        let mut bytes = with_valid_footer(b"hello");
        let footer_start = bytes.len() - FOOTER_LENGTH;
        bytes[footer_start + 4] = 0xFF;
        assert_err!(verify_checksum(&bytes));
    }

    #[test]
    fn wrong_stored_crc_errors() {
        let mut bytes = with_valid_footer(b"hello");
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;
        assert_err!(verify_checksum(&bytes));
    }

    #[test]
    fn corrupted_data_errors() {
        let mut bytes = with_valid_footer(b"hello");
        bytes[0] ^= 0xFF;
        assert_err!(verify_checksum(&bytes));
    }

    #[test]
    fn retrieve_checksum_returns_stored_value() {
        let bytes = with_valid_footer(b"hello world");
        let mut crc = CRC32::new();
        crc.update(&bytes[..bytes.len() - 8]);
        let expected = crc.value() as i64;
        assert_eq!(retrieve_checksum(&bytes).unwrap(), expected);
    }

    #[test]
    fn retrieve_checksum_too_short_errors() {
        assert_err!(retrieve_checksum(&[]));
        assert_err!(retrieve_checksum(&[0u8; 15]));
    }

    #[test]
    fn retrieve_checksum_bad_magic_errors() {
        let mut bytes = with_valid_footer(b"hello");
        let footer_start = bytes.len() - FOOTER_LENGTH;
        bytes[footer_start] ^= 0xFF;
        assert_err!(retrieve_checksum(&bytes));
    }

    #[test]
    fn retrieve_checksum_bad_algorithm_errors() {
        let mut bytes = with_valid_footer(b"hello");
        let footer_start = bytes.len() - FOOTER_LENGTH;
        bytes[footer_start + 4] = 0xFF;
        assert_err!(retrieve_checksum(&bytes));
    }

    #[test]
    fn retrieve_checksum_does_not_verify_crc() {
        // Corrupt body bytes — retrieve_checksum should still succeed because
        // it does not recompute the CRC. This documents the intentional
        // difference from `verify_checksum`.
        let mut bytes = with_valid_footer(b"hello");
        bytes[0] ^= 0xFF;
        assert_ok!(retrieve_checksum(&bytes));
    }

    #[test]
    fn retrieve_checksum_with_length_matches() {
        let bytes = with_valid_footer(b"hello world");
        assert_ok!(retrieve_checksum_with_length(&bytes, bytes.len() as i64));
    }

    #[test]
    fn retrieve_checksum_with_length_too_short() {
        let bytes = with_valid_footer(b"hello world");
        assert_err!(retrieve_checksum_with_length(
            &bytes,
            bytes.len() as i64 + 1
        ));
    }

    #[test]
    fn retrieve_checksum_with_length_too_long() {
        let bytes = with_valid_footer(b"hello world");
        assert_err!(retrieve_checksum_with_length(
            &bytes,
            bytes.len() as i64 - 1
        ));
    }

    #[test]
    fn retrieve_checksum_with_length_below_footer_errors() {
        let bytes = with_valid_footer(b"hello");
        assert_err!(retrieve_checksum_with_length(&bytes, 8));
    }

    // Write-side tests

    use std::io::Write;

    use crate::store::DataOutput;
    use crate::store::memory::MemoryIndexOutput;

    #[test]
    fn test_write_footer() {
        let mut out = MemoryIndexOutput::new("test".to_string());

        // Write some data first
        out.write_all(b"hello").unwrap();

        write_footer(&mut out).unwrap();
        let bytes = out.bytes();

        // Total: 5 (data) + 16 (footer) = 21
        assert_len_eq_x!(&bytes, 21);

        // Footer magic (BE): ~0x3fd76c17 = 0xC02893E8
        let footer_start = 5;
        assert_eq!(bytes[footer_start], 0xc0);
        assert_eq!(bytes[footer_start + 1], 0x28);
        assert_eq!(bytes[footer_start + 2], 0x93);
        assert_eq!(bytes[footer_start + 3], 0xe8);

        // Algorithm ID (BE): 0
        assert_eq!(&bytes[footer_start + 4..footer_start + 8], &[0, 0, 0, 0]);

        // CRC32 is a BE long (8 bytes) — upper 32 bits should be 0
        assert_eq!(&bytes[footer_start + 8..footer_start + 12], &[0, 0, 0, 0]);
    }

    #[test]
    fn test_footer_covers_preceding_bytes() {
        // The CRC in the footer covers all bytes up to and including
        // the first 8 bytes of the footer (magic + algorithm ID)
        let mut out = MemoryIndexOutput::new("test".to_string());
        out.write_all(b"test data").unwrap();

        let checksum_before_crc = {
            let mut out2 = MemoryIndexOutput::new("test2".to_string());
            out2.write_all(b"test data").unwrap();
            out2.write_be_int(FOOTER_MAGIC).unwrap();
            out2.write_be_int(0).unwrap();
            out2.checksum()
        };

        write_footer(&mut out).unwrap();
        let bytes = out.bytes();

        let footer_crc_offset = bytes.len() - 8;
        let written_crc = u64::from_be_bytes(
            bytes[footer_crc_offset..footer_crc_offset + 8]
                .try_into()
                .unwrap(),
        );

        assert_eq!(written_crc, checksum_before_crc);
    }
}
