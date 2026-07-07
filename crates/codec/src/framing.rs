// SPDX-License-Identifier: Apache-2.0

//! Minimal CodecUtil framing over `&[u8]` (SPEC §3.3, §13 P0).
//!
//! Bearing keeps its header/footer helpers `pub(crate)`, so the small
//! subset downstream crates need to validate raw file bytes (`.dvm`/`.dvd`
//! slices feeding decode plans, SPEC §6) is reimplemented here against
//! Lucene's `CodecUtil` wire format:
//!
//! - index header: BE i32 magic, Lucene string (vint length + UTF-8) codec
//!   name, BE i32 version, 16-byte object id, 1-byte suffix length + suffix.
//! - footer (16 bytes): BE i32 footer magic, BE i32 algorithm id (0 =
//!   zlib CRC32), BE u64 CRC32 over every preceding byte — i.e. everything
//!   except the final 8 stored-checksum bytes.

use lucene_arrow_core::{Error, Result};

/// Magic at the start of every codec header (big-endian i32).
pub const CODEC_MAGIC: i32 = 0x3FD76C17;

/// Footer magic: bitwise NOT of [`CODEC_MAGIC`] (`0xC02893E8` as u32).
pub const FOOTER_MAGIC: i32 = !CODEC_MAGIC;

/// Footer length in bytes: 4 (magic) + 4 (algorithm id) + 8 (stored CRC).
pub const FOOTER_LENGTH: usize = 16;

/// Length of a segment/object id in bytes.
pub const ID_LENGTH: usize = 16;

/// Bounds-checked big-endian cursor over a byte slice.
struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).filter(|&e| e <= self.bytes.len());
        match end {
            Some(end) => {
                let out = &self.bytes[self.pos..end];
                self.pos = end;
                Ok(out)
            }
            None => Err(Error::corrupt(format!(
                "truncated header: need {n} bytes at offset {}, file is {} bytes",
                self.pos,
                self.bytes.len()
            ))),
        }
    }

    fn byte(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn be_i32(&mut self) -> Result<i32> {
        let b = self.take(4)?;
        Ok(i32::from_be_bytes([b[0], b[1], b[2], b[3]]))
    }

    /// Lucene vint: 7 bits per byte, MSB = continuation, max 5 bytes.
    fn vint(&mut self) -> Result<u32> {
        let mut value: u64 = 0;
        for shift in (0..35).step_by(7) {
            let b = self.byte()?;
            value |= u64::from(b & 0x7F) << shift;
            if b & 0x80 == 0 {
                return u32::try_from(value)
                    .map_err(|_| Error::corrupt(format!("vint {value} overflows u32")));
            }
        }
        Err(Error::corrupt("vint longer than 5 bytes"))
    }

    /// Lucene string: vint byte-length + UTF-8 bytes.
    fn string(&mut self) -> Result<String> {
        let len = self.vint()? as usize;
        let bytes = self.take(len)?;
        String::from_utf8(bytes.to_vec())
            .map_err(|e| Error::corrupt(format!("non-UTF-8 codec string: {e}")))
    }
}

/// Parses and validates a `CodecUtil.writeIndexHeader` prefix of `bytes`.
///
/// Checks, in order: magic, codec name, version range, 16-byte object id,
/// suffix. Returns the header length in bytes — the offset where the file
/// body starts.
pub fn check_index_header(
    bytes: &[u8],
    expected_codec_name: &str,
    min_version: i32,
    max_version: i32,
    expected_segment_id: &[u8; ID_LENGTH],
    expected_suffix: &str,
) -> Result<usize> {
    let mut cur = Cursor { bytes, pos: 0 };

    let magic = cur.be_i32()?;
    if magic != CODEC_MAGIC {
        return Err(Error::corrupt(format!(
            "codec magic mismatch: expected 0x{CODEC_MAGIC:08X}, got 0x{magic:08X}"
        )));
    }

    let codec = cur.string()?;
    if codec != expected_codec_name {
        return Err(Error::corrupt(format!(
            "codec name mismatch: expected {expected_codec_name:?}, got {codec:?}"
        )));
    }

    let version = cur.be_i32()?;
    if version < min_version || version > max_version {
        return Err(Error::corrupt(format!(
            "version {version} out of range [{min_version}, {max_version}] \
             for codec {expected_codec_name:?}"
        )));
    }

    let id = cur.take(ID_LENGTH)?;
    if id != expected_segment_id {
        return Err(Error::corrupt(format!(
            "object id mismatch: expected {expected_segment_id:02x?}, got {id:02x?}"
        )));
    }

    let suffix_len = cur.byte()? as usize;
    let suffix_bytes = cur.take(suffix_len)?;
    if suffix_bytes != expected_suffix.as_bytes() {
        return Err(Error::corrupt(format!(
            "segment suffix mismatch: expected {expected_suffix:?}, got {:?}",
            String::from_utf8_lossy(suffix_bytes)
        )));
    }

    Ok(cur.pos)
}

/// Verifies the 16-byte CodecUtil footer at the end of `bytes`.
///
/// The CRC32 covers everything except the final 8 stored-checksum bytes
/// (so the footer magic and algorithm id are themselves checksummed).
pub fn verify_footer(bytes: &[u8]) -> Result<()> {
    if bytes.len() < FOOTER_LENGTH {
        return Err(Error::corrupt(format!(
            "file too short for codec footer: {} < {FOOTER_LENGTH} bytes",
            bytes.len()
        )));
    }
    let footer = &bytes[bytes.len() - FOOTER_LENGTH..];

    let magic = i32::from_be_bytes([footer[0], footer[1], footer[2], footer[3]]);
    if magic != FOOTER_MAGIC {
        return Err(Error::corrupt(format!(
            "footer magic mismatch: expected 0x{:08X}, got 0x{:08X}",
            FOOTER_MAGIC as u32, magic as u32
        )));
    }

    let algorithm = i32::from_be_bytes([footer[4], footer[5], footer[6], footer[7]]);
    if algorithm != 0 {
        return Err(Error::corrupt(format!(
            "unknown checksum algorithm id: {algorithm} (only 0 = zlib CRC32)"
        )));
    }

    let stored = u64::from_be_bytes([
        footer[8], footer[9], footer[10], footer[11], footer[12], footer[13], footer[14],
        footer[15],
    ]);
    if stored & 0xFFFF_FFFF_0000_0000 != 0 {
        return Err(Error::corrupt(format!(
            "stored checksum 0x{stored:016X} does not fit in 32 bits"
        )));
    }

    let mut hasher = crc32fast::Hasher::new();
    hasher.update(&bytes[..bytes.len() - 8]);
    let computed = u64::from(hasher.finalize());
    if computed != stored {
        return Err(Error::corrupt(format!(
            "checksum mismatch: stored=0x{stored:08X}, computed=0x{computed:08X}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const ID: [u8; ID_LENGTH] = *b"0123456789abcdef";

    /// Builds a synthetic CodecUtil-framed file: header + payload + footer.
    fn frame(codec: &str, version: i32, suffix: &str, payload: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&CODEC_MAGIC.to_be_bytes());
        assert!(codec.len() < 0x80, "test codec names stay single-byte vint");
        out.push(codec.len() as u8);
        out.extend_from_slice(codec.as_bytes());
        out.extend_from_slice(&version.to_be_bytes());
        out.extend_from_slice(&ID);
        out.push(suffix.len() as u8);
        out.extend_from_slice(suffix.as_bytes());
        out.extend_from_slice(payload);
        out.extend_from_slice(&FOOTER_MAGIC.to_be_bytes());
        out.extend_from_slice(&0i32.to_be_bytes());
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(&out);
        out.extend_from_slice(&u64::from(hasher.finalize()).to_be_bytes());
        out
    }

    #[test]
    fn header_and_footer_round_trip() {
        let bytes = frame("TestCodec", 3, "sfx", b"payload");
        let header_len = check_index_header(&bytes, "TestCodec", 0, 3, &ID, "sfx").unwrap();
        // magic + (1 + 9) name + version + id + (1 + 3) suffix
        assert_eq!(header_len, 4 + 10 + 4 + ID_LENGTH + 4);
        assert_eq!(&bytes[header_len..header_len + 7], b"payload");
        verify_footer(&bytes).unwrap();
    }

    #[test]
    fn header_mismatches_are_corrupt() {
        let bytes = frame("TestCodec", 3, "", b"");
        let is_corrupt = |r: Result<usize>| {
            matches!(r, Err(lucene_arrow_core::Error::Corrupt(_)))
        };
        assert!(is_corrupt(check_index_header(&bytes, "Other", 0, 3, &ID, "")));
        assert!(is_corrupt(check_index_header(&bytes, "TestCodec", 4, 9, &ID, "")));
        assert!(is_corrupt(check_index_header(
            &bytes,
            "TestCodec",
            0,
            3,
            &[0u8; ID_LENGTH],
            ""
        )));
        assert!(is_corrupt(check_index_header(&bytes, "TestCodec", 0, 3, &ID, "x")));
        let mut bad_magic = bytes.clone();
        bad_magic[0] ^= 0xFF;
        assert!(is_corrupt(check_index_header(&bad_magic, "TestCodec", 0, 3, &ID, "")));
    }

    #[test]
    fn footer_detects_corruption() {
        let mut bytes = frame("TestCodec", 0, "", b"data data data");
        verify_footer(&bytes).unwrap();

        // Flip one payload bit: CRC must fail.
        let payload_at = bytes.len() - FOOTER_LENGTH - 3;
        bytes[payload_at] ^= 0x01;
        assert!(verify_footer(&bytes).is_err());
        bytes[payload_at] ^= 0x01;

        // Corrupt the footer magic.
        let magic_at = bytes.len() - FOOTER_LENGTH;
        bytes[magic_at] ^= 0xFF;
        assert!(verify_footer(&bytes).is_err());
        bytes[magic_at] ^= 0xFF;

        // Stored checksum with high bits set.
        let hi_at = bytes.len() - 8;
        bytes[hi_at] = 0x01;
        assert!(verify_footer(&bytes).is_err());

        assert!(verify_footer(&bytes[..FOOTER_LENGTH - 1]).is_err());
    }
}
