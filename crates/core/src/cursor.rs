// SPDX-License-Identifier: Apache-2.0

//! Little cursor over Lucene-encoded bytes, plus CodecUtil framing.
//!
//! Lucene data payloads are little-endian (`DataOutput.writeInt/Long/Short`);
//! only CodecUtil header/footer ints are big-endian. This module is the one
//! place both conventions live. Metadata parsing is CPU-side and sequential
//! (the cuIO pattern, SPEC §5) — this cursor is that parser's substrate.

use crate::error::{Error, Result};

/// CodecUtil constants (Lucene `CodecUtil.java`).
pub const CODEC_MAGIC: u32 = 0x3FD7_6C17;
pub const FOOTER_MAGIC: u32 = !CODEC_MAGIC; // 0xC028_93E8
pub const FOOTER_LENGTH: usize = 16;
pub const SEGMENT_ID_LENGTH: usize = 16;

/// Sequential reader over a byte slice with Lucene primitive decoders.
#[derive(Debug, Clone)]
pub struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        Cursor { bytes, pos: 0 }
    }

    pub fn at(bytes: &'a [u8], pos: usize) -> Self {
        Cursor { bytes, pos }
    }

    pub fn pos(&self) -> usize {
        self.pos
    }

    pub fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.pos)
    }

    pub fn seek(&mut self, pos: usize) -> Result<()> {
        if pos > self.bytes.len() {
            return Err(Error::corrupt(format!("seek {pos} beyond {} bytes", self.bytes.len())));
        }
        self.pos = pos;
        Ok(())
    }

    pub fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.remaining() < n {
            return Err(Error::corrupt(format!(
                "need {n} bytes at offset {}, have {}",
                self.pos,
                self.remaining()
            )));
        }
        let out = &self.bytes[self.pos..self.pos + n];
        self.pos += n;
        Ok(out)
    }

    pub fn skip(&mut self, n: usize) -> Result<()> {
        self.take(n).map(|_| ())
    }

    pub fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    pub fn le_i16(&mut self) -> Result<i16> {
        Ok(i16::from_le_bytes(self.take(2)?.try_into().expect("len checked")))
    }

    pub fn le_i32(&mut self) -> Result<i32> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().expect("len checked")))
    }

    pub fn le_i64(&mut self) -> Result<i64> {
        Ok(i64::from_le_bytes(self.take(8)?.try_into().expect("len checked")))
    }

    pub fn be_i32(&mut self) -> Result<i32> {
        Ok(i32::from_be_bytes(self.take(4)?.try_into().expect("len checked")))
    }

    pub fn be_u32(&mut self) -> Result<u32> {
        Ok(u32::from_be_bytes(self.take(4)?.try_into().expect("len checked")))
    }

    pub fn be_u64(&mut self) -> Result<u64> {
        Ok(u64::from_be_bytes(self.take(8)?.try_into().expect("len checked")))
    }

    /// Lucene VInt: 7-bit groups, low group first, high bit = continuation.
    pub fn vint(&mut self) -> Result<i32> {
        let mut value: u32 = 0;
        for shift in (0..35).step_by(7) {
            let b = self.u8()?;
            value |= ((b & 0x7F) as u32) << shift;
            if b & 0x80 == 0 {
                return Ok(value as i32);
            }
        }
        Err(Error::corrupt("vint longer than 5 bytes"))
    }

    pub fn vlong(&mut self) -> Result<i64> {
        let mut value: u64 = 0;
        for shift in (0..70).step_by(7) {
            let b = self.u8()?;
            value |= ((b & 0x7F) as u64) << shift;
            if b & 0x80 == 0 {
                return Ok(value as i64);
            }
        }
        Err(Error::corrupt("vlong longer than 10 bytes"))
    }

    /// Lucene writeString: VInt byte length + UTF-8 bytes.
    pub fn string(&mut self) -> Result<String> {
        let len = self.vint()?;
        if len < 0 {
            return Err(Error::corrupt("negative string length"));
        }
        let bytes = self.take(len as usize)?;
        String::from_utf8(bytes.to_vec()).map_err(|e| Error::corrupt(format!("bad utf8: {e}")))
    }
}

/// Parsed CodecUtil index header (`CodecUtil.writeIndexHeader`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexHeader {
    pub codec: String,
    pub version: i32,
    pub segment_id: [u8; SEGMENT_ID_LENGTH],
    pub suffix: String,
    /// Byte length of the header; payload starts here.
    pub length: usize,
}

/// Parse and validate an index header at the start of `bytes`.
pub fn read_index_header(
    bytes: &[u8],
    expected_codec: &str,
    min_version: i32,
    max_version: i32,
) -> Result<IndexHeader> {
    let mut c = Cursor::new(bytes);
    let magic = c.be_u32()?;
    if magic != CODEC_MAGIC {
        return Err(Error::corrupt(format!("bad codec magic {magic:#010x}")));
    }
    let codec = c.string()?;
    if codec != expected_codec {
        return Err(Error::corrupt(format!("codec name {codec:?}, expected {expected_codec:?}")));
    }
    let version = c.be_i32()?;
    if version < min_version || version > max_version {
        return Err(Error::unsupported(format!(
            "{codec} version {version} outside [{min_version}, {max_version}]"
        )));
    }
    let segment_id: [u8; SEGMENT_ID_LENGTH] =
        c.take(SEGMENT_ID_LENGTH)?.try_into().expect("len checked");
    let suffix_len = c.u8()? as usize;
    let suffix_bytes = c.take(suffix_len)?;
    let suffix = String::from_utf8(suffix_bytes.to_vec())
        .map_err(|e| Error::corrupt(format!("bad suffix utf8: {e}")))?;
    Ok(IndexHeader { codec, version, segment_id, suffix, length: c.pos() })
}

/// Verify the 16-byte CodecUtil footer (magic, algorithm 0, CRC32 of
/// everything before the stored checksum).
pub fn verify_footer(bytes: &[u8]) -> Result<()> {
    if bytes.len() < FOOTER_LENGTH {
        return Err(Error::corrupt("file shorter than codec footer"));
    }
    let mut c = Cursor::at(bytes, bytes.len() - FOOTER_LENGTH);
    let magic = c.be_u32()?;
    if magic != FOOTER_MAGIC {
        return Err(Error::corrupt(format!("bad footer magic {magic:#010x}")));
    }
    let algorithm = c.be_i32()?;
    if algorithm != 0 {
        return Err(Error::corrupt(format!("unknown checksum algorithm {algorithm}")));
    }
    let stored = c.be_u64()?;
    if stored & 0xFFFF_FFFF_0000_0000 != 0 {
        return Err(Error::corrupt(format!("checksum {stored:#018x} exceeds 32 bits")));
    }
    let actual = crc32fast::hash(&bytes[..bytes.len() - 8]) as u64;
    if actual != stored {
        return Err(Error::corrupt(format!("checksum mismatch: stored {stored:#x}, actual {actual:#x}")));
    }
    Ok(())
}

// --- Write side -------------------------------------------------------------

/// Append a Lucene VInt.
pub fn write_vint(out: &mut Vec<u8>, mut v: u32) {
    while v & !0x7F != 0 {
        out.push((v & 0x7F) as u8 | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

/// Append a Lucene VLong (must be non-negative, as Lucene requires).
pub fn write_vlong(out: &mut Vec<u8>, mut v: u64) {
    while v & !0x7F != 0 {
        out.push((v & 0x7F) as u8 | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

/// Append `CodecUtil.writeIndexHeader` bytes.
pub fn write_index_header(
    out: &mut Vec<u8>,
    codec: &str,
    version: i32,
    segment_id: &[u8; SEGMENT_ID_LENGTH],
    suffix: &str,
) {
    out.extend_from_slice(&CODEC_MAGIC.to_be_bytes());
    write_vint(out, codec.len() as u32);
    out.extend_from_slice(codec.as_bytes());
    out.extend_from_slice(&version.to_be_bytes());
    out.extend_from_slice(segment_id);
    debug_assert!(suffix.len() < 256);
    out.push(suffix.len() as u8);
    out.extend_from_slice(suffix.as_bytes());
}

/// Append the CodecUtil footer: magic, algorithm 0, CRC32 of everything
/// written so far including the magic+algorithm words.
pub fn write_footer(out: &mut Vec<u8>) {
    out.extend_from_slice(&FOOTER_MAGIC.to_be_bytes());
    out.extend_from_slice(&0i32.to_be_bytes());
    let crc = crc32fast::hash(out) as u64;
    out.extend_from_slice(&crc.to_be_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_and_footer_round_trip() {
        let id = *b"0123456789abcdef";
        let mut file = Vec::new();
        write_index_header(&mut file, "TestCodec", 2, &id, "sfx");
        file.extend_from_slice(b"payload");
        write_footer(&mut file);

        let header = read_index_header(&file, "TestCodec", 0, 3).unwrap();
        assert_eq!(header.version, 2);
        assert_eq!(header.segment_id, id);
        assert_eq!(header.suffix, "sfx");
        assert_eq!(&file[header.length..header.length + 7], b"payload");
        verify_footer(&file).unwrap();

        let mut corrupted = file.clone();
        let n = corrupted.len();
        corrupted[n - 20] ^= 1; // flip a payload bit
        assert!(verify_footer(&corrupted).is_err());

        assert!(read_index_header(&file, "OtherCodec", 0, 3).is_err());
        assert!(read_index_header(&file, "TestCodec", 3, 3).is_err());
    }

    #[test]
    fn vint_matches_lucene_layout() {
        let mut out = Vec::new();
        write_vint(&mut out, 300);
        assert_eq!(out, vec![0xAC, 0x02]);
        let mut c = Cursor::new(&out);
        assert_eq!(c.vint().unwrap(), 300);
    }
}
