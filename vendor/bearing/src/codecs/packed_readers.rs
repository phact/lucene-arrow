// SPDX-License-Identifier: Apache-2.0

//! Packed integer readers for data written by [`super::packed_writers`].

use std::io;

use crate::store::IndexInput;

/// Reads bit-packed integers written by [`super::packed_writers::DirectWriter`].
///
/// Values are stored LSB-first in little-endian byte order. Supported
/// bits-per-value: 1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64.
pub struct DirectReader<'a> {
    input: IndexInput<'a>,
    bits_per_value: u32,
    offset: u64,
}

impl<'a> DirectReader<'a> {
    /// Creates a reader over `bytes` starting at `offset` with the given bpv.
    pub fn new(bytes: &'a [u8], bits_per_value: u32, offset: u64) -> Self {
        Self {
            input: IndexInput::unnamed(bytes),
            bits_per_value,
            offset,
        }
    }

    /// Reads the value at the given logical index.
    pub fn get(&mut self, index: u64) -> io::Result<i64> {
        let bpv = self.bits_per_value;
        match bpv {
            0 => Ok(0),
            1 | 2 | 4 => self.get_sub_byte(index, bpv),
            8 => self.get_byte_aligned(index, 1, 0xFF),
            12 => self.get_odd(index, bpv, 0xFFF),
            16 => self.get_byte_aligned(index, 2, 0xFFFF),
            20 => self.get_odd(index, bpv, 0xFFFFF),
            24 => self.get_byte_aligned(index, 3, 0xFFFFFF),
            28 => self.get_odd(index, bpv, 0xFFFFFFF),
            32 => self.get_byte_aligned(index, 4, 0xFFFFFFFF),
            40 => self.get_byte_aligned(index, 5, 0xFF_FFFFFFFF),
            48 => self.get_byte_aligned(index, 6, 0xFFFF_FFFFFFFF),
            56 => self.get_byte_aligned(index, 7, 0xFFFFFF_FFFFFFFF),
            64 => self.get_byte_aligned(index, 8, u64::MAX),
            _ => Err(io::Error::other(format!(
                "unsupported bits per value: {bpv}"
            ))),
        }
    }

    /// Sub-byte bpv (1, 2, 4): multiple values packed into each byte.
    fn get_sub_byte(&mut self, index: u64, bpv: u32) -> io::Result<i64> {
        let values_per_byte = 8 / bpv;
        let byte_offset = index / values_per_byte as u64;
        let bit_offset = (index % values_per_byte as u64) * bpv as u64;
        let mask = (1u64 << bpv) - 1;

        self.input.seek((self.offset + byte_offset) as usize)?;
        let b = self.input.read_byte()? as u64;
        Ok(((b >> bit_offset) & mask) as i64)
    }

    /// Byte-aligned bpv (8, 16, 24, 32, 40, 48, 56, 64).
    fn get_byte_aligned(&mut self, index: u64, bytes_per_value: u64, mask: u64) -> io::Result<i64> {
        let pos = self.offset + index * bytes_per_value;
        self.input.seek(pos as usize)?;
        let mut buf = [0u8; 8];
        self.input
            .read_bytes(&mut buf[..bytes_per_value as usize])?;
        let raw = u64::from_le_bytes(buf);
        Ok((raw & mask) as i64)
    }

    /// Odd bpv (12, 20, 28): pairs of values packed together.
    fn get_odd(&mut self, index: u64, bpv: u32, mask: u64) -> io::Result<i64> {
        let byte_offset = (index * bpv as u64) / 8;
        let shift = (index & 1) * 4;
        let bytes_to_read = if bpv <= 16 { 4usize } else { 8 };

        self.input.seek((self.offset + byte_offset) as usize)?;
        let mut buf = [0u8; 8];
        let read_len = bytes_to_read.min(self.input.length() - self.input.position());
        self.input.read_bytes(&mut buf[..read_len])?;
        let raw = u64::from_le_bytes(buf);
        Ok(((raw >> shift) & mask) as i64)
    }
}

/// Per-block metadata for [`DirectMonotonicReader`].
struct BlockMeta {
    min: i64,
    avg: f32,
    bpv: u8,
    /// Relative offset within the data region where this block's packed values begin.
    offset: u64,
}

/// Reads monotonically-increasing sequences written by
/// [`super::packed_writers::DirectMonotonicWriter`].
///
/// Each block stores a minimum, average increment, and bit-packed deltas.
/// Values are reconstructed as `min + avg * index_within_block + delta`.
///
/// The reader owns its per-block metadata but does not own the data bytes —
/// callers pass the data slice at read time via [`get`](Self::get). This
/// matches the Layer 3 principle that codec readers do not hold cursor state.
pub(crate) struct DirectMonotonicReader {
    block_shift: u32,
    base_data_offset: u64,
    blocks: Box<[BlockMeta]>,
}

impl DirectMonotonicReader {
    /// Loads metadata from `meta`. `base_data_offset` is the absolute offset
    /// within the caller-supplied data slice where per-block data begins.
    pub(crate) fn load_with_shift(
        meta: &mut IndexInput<'_>,
        num_values: u32,
        base_data_offset: u64,
        block_shift: u32,
    ) -> io::Result<Self> {
        let block_size = 1u32 << block_shift;
        let num_blocks = (num_values as usize).div_ceil(block_size as usize);
        let mut blocks = Vec::with_capacity(num_blocks);

        for _ in 0..num_blocks {
            let min = meta.read_le_long()?;
            let avg_bits = meta.read_le_int()? as u32;
            let avg = f32::from_bits(avg_bits);
            let offset = meta.read_le_long()? as u64;
            let bpv = meta.read_byte()?;

            blocks.push(BlockMeta {
                min,
                avg,
                bpv,
                offset,
            });
        }

        Ok(Self {
            block_shift,
            base_data_offset,
            blocks: blocks.into_boxed_slice(),
        })
    }

    /// Reads the value at the given logical index, borrowing packed bytes from `data`.
    pub(crate) fn get(&self, index: u64, data: &[u8]) -> io::Result<i64> {
        let block = (index >> self.block_shift) as usize;
        let block_index = index & ((1u64 << self.block_shift) - 1);
        let meta = &self.blocks[block];
        let delta = if meta.bpv == 0 {
            0
        } else {
            let start = (self.base_data_offset + meta.offset) as usize;
            let mut reader = DirectReader::new(&data[start..], meta.bpv as u32, 0);
            reader.get(block_index)?
        };
        Ok(meta.min + (meta.avg as f64 * block_index as f64) as i64 + delta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codecs::packed_writers::DirectWriter;
    use crate::store::memory::MemoryIndexOutput;
    use crate::store::{DataOutput, IndexOutput};

    /// Helper to write values with DirectWriter and read them back with DirectReader.
    fn round_trip(values: &[i64], bpv: u32) {
        let mut out = MemoryIndexOutput::new("test".to_string());
        let mut writer = DirectWriter::new(bpv);
        for &v in values {
            writer.add(v);
        }
        writer.finish(&mut out).unwrap();

        let mut reader = DirectReader::new(out.bytes(), bpv, 0);

        for (i, &expected) in values.iter().enumerate() {
            let actual = reader.get(i as u64).unwrap();
            assert_eq!(actual, expected, "mismatch at index {i} for bpv {bpv}");
        }
    }

    #[test]
    fn test_direct_reader_bpv1() {
        round_trip(&[1, 0, 1, 1, 0, 0, 1, 0], 1);
    }

    #[test]
    fn test_direct_reader_bpv2() {
        round_trip(&[0, 1, 2, 3, 3, 2, 1, 0], 2);
    }

    #[test]
    fn test_direct_reader_bpv4() {
        round_trip(&[0, 5, 10, 15, 1, 14, 7, 8], 4);
    }

    #[test]
    fn test_direct_reader_bpv8() {
        round_trip(&[0, 127, 255, 1, 42], 8);
    }

    #[test]
    fn test_direct_reader_bpv12() {
        round_trip(&[0x123, 0x456, 0x789, 0xABC], 12);
    }

    #[test]
    fn test_direct_reader_bpv16() {
        round_trip(&[0, 0x1234, 0xABCD, 0xFFFF], 16);
    }

    #[test]
    fn test_direct_reader_bpv20() {
        round_trip(&[0, 0x12345, 0xABCDE, 0xFFFFF], 20);
    }

    #[test]
    fn test_direct_reader_bpv24() {
        round_trip(&[0, 0x123456, 0xABCDEF, 0xFFFFFF], 24);
    }

    #[test]
    fn test_direct_reader_bpv28() {
        round_trip(&[0, 0x1234567, 0xABCDEF0, 0xFFFFFFF], 28);
    }

    #[test]
    fn test_direct_reader_bpv32() {
        round_trip(&[0, 0x12345678, 0xFFFFFFFF], 32);
    }

    #[test]
    fn test_direct_reader_bpv40() {
        round_trip(&[0, 0x12_3456_7890, 0xFF_FFFFFFFF], 40);
    }

    #[test]
    fn test_direct_reader_bpv48() {
        round_trip(&[0, 0x1234_5678_9ABC, 0xFFFF_FFFFFFFF], 48);
    }

    #[test]
    fn test_direct_reader_bpv56() {
        round_trip(&[0, 0x12_3456_7890_ABCD, 0xFFFFFF_FFFFFFFF], 56);
    }

    #[test]
    fn test_direct_reader_bpv64() {
        round_trip(&[0, i64::MAX, 1, 0x1234_5678_9ABC_DEF0u64 as i64], 64);
    }

    #[test]
    fn test_direct_reader_bpv0() {
        let mut reader = DirectReader::new(&[], 0, 0);
        assert_eq!(reader.get(0).unwrap(), 0);
        assert_eq!(reader.get(100).unwrap(), 0);
    }

    #[test]
    fn test_direct_reader_single_value() {
        for bpv in [1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64] {
            round_trip(&[1], bpv);
        }
    }

    #[test]
    fn test_direct_reader_with_offset() {
        let mut out = MemoryIndexOutput::new("test".to_string());
        for i in 0..10u8 {
            out.write_byte(i).unwrap();
        }
        let offset = out.file_pointer();

        let mut writer = DirectWriter::new(8);
        writer.add(42);
        writer.add(99);
        writer.add(255);
        writer.finish(&mut out).unwrap();

        let mut reader = DirectReader::new(out.bytes(), 8, offset);

        assert_eq!(reader.get(0).unwrap(), 42);
        assert_eq!(reader.get(1).unwrap(), 99);
        assert_eq!(reader.get(2).unwrap(), 255);
    }

    // --- DirectMonotonicReader tests ---

    use crate::codecs::packed_writers::DirectMonotonicWriter;

    /// Write monotonic values with DirectMonotonicWriter, read back with DirectMonotonicReader.
    fn monotonic_round_trip(values: &[i64], block_shift: u32) {
        let mut writer = DirectMonotonicWriter::new(block_shift);
        for &v in values {
            writer.add(v);
        }

        let mut meta_out = MemoryIndexOutput::new("meta".to_string());
        let mut data_out = MemoryIndexOutput::new("data".to_string());
        let _num_blocks = writer.finish(&mut meta_out, &mut data_out).unwrap();

        let meta_bytes = meta_out.bytes().to_vec();
        let data_bytes = data_out.bytes().to_vec();
        let mut meta_input = IndexInput::unnamed(&meta_bytes);

        let num_values = values.len() as u32;
        let reader =
            DirectMonotonicReader::load_with_shift(&mut meta_input, num_values, 0, block_shift)
                .unwrap();

        for (i, &expected) in values.iter().enumerate() {
            let actual = reader.get(i as u64, &data_bytes).unwrap();
            assert_eq!(
                actual, expected,
                "mismatch at index {i}: expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn test_monotonic_reader_simple() {
        monotonic_round_trip(&[0, 10, 20, 30], 2);
    }

    #[test]
    fn test_monotonic_reader_constant() {
        monotonic_round_trip(&[42, 42, 42, 42], 2);
    }

    #[test]
    fn test_monotonic_reader_multiple_blocks() {
        // block_shift=1 means block_size=2, so 5 values = 3 blocks
        monotonic_round_trip(&[0, 100, 200, 300, 400], 1);
    }

    #[test]
    fn test_monotonic_reader_single_value() {
        monotonic_round_trip(&[999], 2);
    }

    #[test]
    fn test_monotonic_reader_large_values() {
        monotonic_round_trip(&[0, 1_000_000, 2_000_000, 3_000_000, 10_000_000], 2);
    }

    #[test]
    fn test_monotonic_reader_irregular_increments() {
        monotonic_round_trip(&[0, 1, 100, 101, 10000, 10001, 10002, 10003], 2);
    }

    #[test]
    fn test_monotonic_reader_two_values() {
        monotonic_round_trip(&[0, 1], 1);
    }

    #[test]
    fn test_monotonic_reader_large_block() {
        // All values in one block (block_shift=10 = 1024 values per block)
        let values: Vec<i64> = (0..100).map(|i| i * 7).collect();
        monotonic_round_trip(&values, 10);
    }
}
