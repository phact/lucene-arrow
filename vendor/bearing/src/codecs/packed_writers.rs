// SPDX-License-Identifier: Apache-2.0

//! Packed integer writers that accumulate values and serialize using
//! the encoding functions in [`crate::encoding::packed`].

use std::io;

use crate::encoding::packed::{
    BPV_SHIFT, MIN_VALUE_EQUALS_0, pack_msb, packed_bits_required, packed_max_value,
    unsigned_bits_required, write_block_packed_vlong,
};
use crate::encoding::zigzag;
use crate::store::{DataOutput, IndexOutput};

/// Writes bit-packed integers using LSB-first (little-endian) bit ordering.
pub struct DirectWriter {
    bits_per_value: u32,
    values: Vec<i64>,
}

impl DirectWriter {
    /// Creates a new writer with the given bits-per-value.
    pub fn new(bits_per_value: u32) -> Self {
        Self {
            bits_per_value,
            values: Vec::new(),
        }
    }

    /// Adds a value to the writer.
    pub fn add(&mut self, value: i64) {
        self.values.push(value);
    }

    /// Writes all accumulated values as bit-packed data, then padding bytes.
    pub fn finish(&self, output: &mut dyn DataOutput) -> io::Result<()> {
        if self.bits_per_value == 0 {
            return Ok(());
        }

        let bpv = self.bits_per_value;
        let up_to = self.values.len();

        if (bpv & 7) == 0 {
            // bpv is a multiple of 8: 8, 16, 24, 32, 40, 48, 56, 64
            let bytes_per_value = (bpv / 8) as usize;
            for i in 0..up_to {
                let v = self.values[i] as u64;
                let le = v.to_le_bytes();
                output.write_all(&le[..bytes_per_value])?;
            }
        } else if bpv < 8 {
            // bpv is 1, 2, or 4: pack values LSB-first into LE longs
            let values_per_long = (64 / bpv) as usize;
            let mut i = 0;
            while i < up_to {
                let mut packed: u64 = 0;
                for j in 0..values_per_long {
                    if i + j < up_to {
                        packed |= (self.values[i + j] as u64) << (bpv * j as u32);
                    }
                }
                let remaining = (up_to - i).min(values_per_long);
                let bytes_needed = (remaining * bpv as usize).div_ceil(8);
                output.write_all(&packed.to_le_bytes()[..bytes_needed])?;
                i += values_per_long;
            }
        } else {
            // bpv is 12, 20, or 28: write pairs LSB-first
            let num_bytes_for_2 = (bpv * 2 / 8) as usize;
            let mut i = 0;
            while i < up_to {
                let l1 = self.values[i] as u64;
                let l2 = if i + 1 < up_to {
                    self.values[i + 1] as u64
                } else {
                    0
                };
                let merged = l1 | (l2 << bpv);
                if bpv <= 16 {
                    let le = (merged as u32).to_le_bytes();
                    output.write_all(&le[..num_bytes_for_2])?;
                } else {
                    let le = merged.to_le_bytes();
                    output.write_all(&le[..num_bytes_for_2])?;
                }
                i += 2;
            }
        }

        // Add padding bytes for fast I/O reads
        let padding_bits = if bpv > 32 {
            64 - bpv
        } else if bpv > 16 {
            32 - bpv
        } else if bpv > 8 {
            16 - bpv
        } else {
            0
        };
        let padding_bytes = padding_bits.div_ceil(8);
        for _ in 0..padding_bytes {
            output.write_byte(0)?;
        }

        Ok(())
    }
}

/// Writes monotonically-increasing sequences of longs with delta compression.
///
/// Data is split into blocks. For each block:
/// - Meta: min value (VLong), avg increment (Int as float bits), offset (VLong), bits per value (Byte)
/// - Data: bit-packed deltas from expected linear values
pub struct DirectMonotonicWriter {
    block_shift: u32,
    values: Vec<i64>,
}

impl DirectMonotonicWriter {
    /// Creates a new writer with the given block shift.
    pub fn new(block_shift: u32) -> Self {
        Self {
            block_shift,
            values: Vec::new(),
        }
    }

    /// Adds a value to the writer.
    pub fn add(&mut self, value: i64) {
        self.values.push(value);
    }

    /// Writes metadata to meta_output and data to data_output.
    /// Returns the number of blocks written.
    pub fn finish(
        &self,
        meta_output: &mut dyn IndexOutput,
        data_output: &mut dyn IndexOutput,
    ) -> io::Result<u32> {
        let block_size = 1usize << self.block_shift;
        let num_blocks = self.values.len().div_ceil(block_size);
        let base_data_pointer = data_output.file_pointer() as i64;

        for block_idx in 0..num_blocks {
            let start = block_idx * block_size;
            let end = (start + block_size).min(self.values.len());
            let count = end - start;

            let mut buffer: Vec<i64> = self.values[start..end].to_vec();

            let avg_inc = (buffer[count - 1] - buffer[0]) as f64 / (count - 1).max(1) as f64;
            let avg_inc_f = avg_inc as f32;

            let mut min = i64::MAX;
            for (i, val) in buffer.iter_mut().enumerate().take(count) {
                let expected = (avg_inc_f * i as f32) as i64;
                *val -= expected;
                min = min.min(*val);
            }

            let mut max_delta: i64 = 0;
            for val in buffer.iter_mut().take(count) {
                *val -= min;
                max_delta |= *val;
            }

            meta_output.write_le_long(min)?;
            meta_output.write_le_int(f32::to_bits(avg_inc_f) as i32)?;
            meta_output.write_le_long(data_output.file_pointer() as i64 - base_data_pointer)?;
            if max_delta == 0 {
                meta_output.write_byte(0)?;
            } else {
                let bits_per_value = unsigned_bits_required(max_delta);
                let mut writer = DirectWriter::new(bits_per_value);
                for &val in buffer.iter().take(count) {
                    writer.add(val);
                }
                writer.finish(data_output)?;
                meta_output.write_byte(bits_per_value as u8)?;
            }
        }

        Ok(num_blocks as u32)
    }
}

/// Writes large sequences of longs using block-level delta compression.
///
/// Values are split into fixed-size blocks. For each block, the minimum value is subtracted
/// and the deltas are packed MSB-first using the minimum number of bits required. Each block
/// has a 1–10 byte header encoding the bits-per-value and optional minimum.
pub(crate) struct BlockPackedWriter {
    block_size: usize,
    values: Vec<i64>,
    off: usize,
    ord: u64,
    finished: bool,
}

impl BlockPackedWriter {
    /// Creates a new writer with the given block size, which must be a multiple of 64.
    pub(crate) fn new(block_size: usize) -> Self {
        assert!(
            block_size >= 64 && block_size.is_multiple_of(64),
            "block_size must be a multiple of 64, got {block_size}"
        );
        Self {
            block_size,
            values: vec![0i64; block_size],
            off: 0,
            ord: 0,
            finished: false,
        }
    }

    /// Appends a value, flushing the current block if full.
    pub(crate) fn add(&mut self, output: &mut dyn DataOutput, value: i64) -> io::Result<()> {
        assert!(!self.finished, "already finished");
        if self.off == self.block_size {
            self.flush(output)?;
        }
        self.values[self.off] = value;
        self.off += 1;
        self.ord += 1;
        Ok(())
    }

    /// Flushes any remaining values and marks the writer as finished.
    pub(crate) fn finish(&mut self, output: &mut dyn DataOutput) -> io::Result<()> {
        assert!(!self.finished, "already finished");
        if self.off > 0 {
            self.flush(output)?;
        }
        self.finished = true;
        Ok(())
    }

    /// Returns the number of values added so far.
    #[cfg(test)]
    pub(crate) fn ord(&self) -> u64 {
        self.ord
    }

    /// Resets the writer for reuse with a new output.
    pub(crate) fn reset(&mut self) {
        self.off = 0;
        self.ord = 0;
        self.finished = false;
    }

    fn flush(&mut self, output: &mut dyn DataOutput) -> io::Result<()> {
        assert!(self.off > 0);

        let mut min = i64::MAX;
        let mut max = i64::MIN;
        for i in 0..self.off {
            min = min.min(self.values[i]);
            max = max.max(self.values[i]);
        }

        let delta = max.wrapping_sub(min);
        let bpv = if delta == 0 {
            0
        } else {
            packed_bits_required(delta)
        };

        if bpv == 64 {
            min = 0;
        } else if min > 0 {
            min = 0i64.max(max - packed_max_value(bpv));
        }

        let token = ((bpv << BPV_SHIFT)
            | if min == 0 {
                MIN_VALUE_EQUALS_0 as u32
            } else {
                0
            }) as u8;
        output.write_byte(token)?;

        if min != 0 {
            write_block_packed_vlong(output, zigzag::encode_i64(min) - 1)?;
        }

        if bpv > 0 {
            if min != 0 {
                for i in 0..self.off {
                    self.values[i] -= min;
                }
            }
            for i in self.off..self.block_size {
                self.values[i] = 0;
            }

            let packed = pack_msb(&self.values, self.off, bpv);
            output.write_all(&packed)?;
        }

        self.off = 0;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoding::packed::BPV_SHIFT;
    use crate::store::memory::MemoryIndexOutput;

    #[test]
    fn test_direct_writer_bpv8_lsb_first() {
        let mut out = MemoryIndexOutput::new("test".to_string());
        let mut w = DirectWriter::new(8);
        w.add(0x12);
        w.add(0x34);
        w.add(0xAB);
        w.finish(&mut out).unwrap();
        assert_eq!(out.bytes(), &[0x12, 0x34, 0xAB]);
    }

    #[test]
    fn test_direct_writer_bpv16() {
        let mut out = MemoryIndexOutput::new("test".to_string());
        let mut w = DirectWriter::new(16);
        w.add(0x1234);
        w.add(0xABCD);
        w.finish(&mut out).unwrap();
        assert_eq!(out.bytes(), &[0x34, 0x12, 0xCD, 0xAB]);
    }

    #[test]
    fn test_direct_writer_bpv1_lsb_first() {
        let mut out = MemoryIndexOutput::new("test".to_string());
        let mut w = DirectWriter::new(1);
        w.add(1);
        w.add(0);
        w.add(1);
        for _ in 0..61 {
            w.add(0);
        }
        w.finish(&mut out).unwrap();
        assert_eq!(out.bytes()[0], 0x05);
    }

    #[test]
    fn test_direct_writer_bpv4_lsb_first() {
        let mut out = MemoryIndexOutput::new("test".to_string());
        let mut w = DirectWriter::new(4);
        w.add(0xA);
        w.add(0xB);
        for _ in 0..14 {
            w.add(0);
        }
        w.finish(&mut out).unwrap();
        assert_eq!(out.bytes()[0], 0xBA);
        for byte in &out.bytes()[1..8] {
            assert_eq!(*byte, 0);
        }
    }

    #[test]
    fn test_direct_writer_bpv12_pairs() {
        let mut out = MemoryIndexOutput::new("test".to_string());
        let mut w = DirectWriter::new(12);
        w.add(0x123);
        w.add(0x456);
        w.finish(&mut out).unwrap();
        let bytes = out.bytes();
        assert_eq!(bytes[0], 0x23);
        assert_eq!(bytes[1], 0x61);
        assert_eq!(bytes[2], 0x45);
        assert_len_eq_x!(&bytes, 4);
        assert_eq!(bytes[3], 0);
    }

    #[test]
    fn test_direct_writer_bpv32_padding() {
        let mut out = MemoryIndexOutput::new("test".to_string());
        let mut w = DirectWriter::new(32);
        w.add(1);
        w.finish(&mut out).unwrap();
        assert_eq!(out.bytes(), &[1, 0, 0, 0]);
    }

    #[test]
    fn test_direct_writer_bpv24_padding() {
        let mut out = MemoryIndexOutput::new("test".to_string());
        let mut w = DirectWriter::new(24);
        w.add(0x010203);
        w.finish(&mut out).unwrap();
        assert_eq!(out.bytes(), &[0x03, 0x02, 0x01, 0x00]);
    }

    #[test]
    fn test_direct_writer_bpv40_padding() {
        let mut out = MemoryIndexOutput::new("test".to_string());
        let mut w = DirectWriter::new(40);
        w.add(0x01);
        w.finish(&mut out).unwrap();
        assert_len_eq_x!(out.bytes(), 5 + 3);
    }

    #[test]
    fn test_direct_writer_bpv0() {
        let mut out = MemoryIndexOutput::new("test".to_string());
        let w = DirectWriter::new(0);
        w.finish(&mut out).unwrap();
        assert_is_empty!(out.bytes());
    }

    #[test]
    fn test_direct_monotonic_writer_simple() {
        let mut writer = DirectMonotonicWriter::new(2);
        writer.add(0);
        writer.add(10);
        writer.add(20);
        writer.add(30);

        let mut meta = MemoryIndexOutput::new("meta".to_string());
        let mut data = MemoryIndexOutput::new("data".to_string());

        let blocks = writer.finish(&mut meta, &mut data).unwrap();
        assert_eq!(blocks, 1);
        assert_len_eq_x!(&meta.bytes(), 21);
    }

    #[test]
    fn test_direct_monotonic_writer_multiple_blocks() {
        let mut writer = DirectMonotonicWriter::new(1);
        writer.add(0);
        writer.add(100);
        writer.add(200);
        writer.add(300);
        writer.add(400);

        let mut meta = MemoryIndexOutput::new("meta".to_string());
        let mut data = MemoryIndexOutput::new("data".to_string());

        let blocks = writer.finish(&mut meta, &mut data).unwrap();
        assert_eq!(blocks, 3);
    }

    #[test]
    fn test_direct_monotonic_writer_constant() {
        let mut writer = DirectMonotonicWriter::new(2);
        writer.add(42);
        writer.add(42);
        writer.add(42);
        writer.add(42);

        let mut meta = MemoryIndexOutput::new("meta".to_string());
        let mut data = MemoryIndexOutput::new("data".to_string());

        writer.finish(&mut meta, &mut data).unwrap();
        assert_is_empty!(data.bytes());
    }

    #[test]
    fn test_block_packed_writer_all_same() {
        let mut out = MemoryIndexOutput::new("test".to_string());
        let mut w = BlockPackedWriter::new(64);
        for _ in 0..64 {
            w.add(&mut out, 42).unwrap();
        }
        w.finish(&mut out).unwrap();
        assert_len_eq_x!(&out.bytes(), 2);
        assert_eq!(out.bytes()[0], 0x00);
        assert_eq!(out.bytes()[1], 83);
    }

    #[test]
    fn test_block_packed_writer_all_zeros() {
        let mut out = MemoryIndexOutput::new("test".to_string());
        let mut w = BlockPackedWriter::new(64);
        for _ in 0..64 {
            w.add(&mut out, 0).unwrap();
        }
        w.finish(&mut out).unwrap();
        assert_len_eq_x!(&out.bytes(), 1);
        assert_eq!(out.bytes()[0], 0x01);
    }

    #[test]
    fn test_block_packed_writer_sequential() {
        let mut out = MemoryIndexOutput::new("test".to_string());
        let mut w = BlockPackedWriter::new(64);
        for i in 0..64 {
            w.add(&mut out, i).unwrap();
        }
        w.finish(&mut out).unwrap();
        let bytes = out.bytes();
        assert_eq!(bytes[0], 0x0D);
        assert_len_eq_x!(bytes, 1 + 48);
    }

    #[test]
    fn test_block_packed_writer_partial_block() {
        let mut out = MemoryIndexOutput::new("test".to_string());
        let mut w = BlockPackedWriter::new(64);
        for i in 0..10 {
            w.add(&mut out, i).unwrap();
        }
        w.finish(&mut out).unwrap();
        let bytes = out.bytes();
        assert_eq!(bytes[0], 9);
        assert_eq!(bytes.len(), 1 + 5);
    }

    #[test]
    fn test_block_packed_writer_multiple_blocks() {
        let mut out = MemoryIndexOutput::new("test".to_string());
        let mut w = BlockPackedWriter::new(64);
        for i in 0..100 {
            w.add(&mut out, i).unwrap();
        }
        w.finish(&mut out).unwrap();
        assert_eq!(w.ord(), 100);

        let bytes = out.bytes();
        assert_eq!(bytes[0], 13);
        let block2_token = bytes[49];
        let block2_bpv = (block2_token >> BPV_SHIFT) as u32;
        assert_eq!(block2_bpv, 6);
        assert_eq!(block2_token & 1, 0);
    }

    #[test]
    fn test_block_packed_writer_delta_with_min() {
        let mut out = MemoryIndexOutput::new("test".to_string());
        let mut w = BlockPackedWriter::new(64);
        for i in 0..4 {
            w.add(&mut out, 1000 + i).unwrap();
        }
        w.finish(&mut out).unwrap();
        let bytes = out.bytes();
        let token = bytes[0];
        let bpv = (token >> BPV_SHIFT) as u32;
        assert_eq!(bpv, 2);
        assert_eq!(token & 1, 0);
    }

    #[test]
    fn test_block_packed_writer_bpv64() {
        let mut out = MemoryIndexOutput::new("test".to_string());
        let mut w = BlockPackedWriter::new(64);
        w.add(&mut out, i64::MIN).unwrap();
        w.add(&mut out, i64::MAX).unwrap();
        for _ in 2..64 {
            w.add(&mut out, 0).unwrap();
        }
        w.finish(&mut out).unwrap();
        let bytes = out.bytes();
        assert_eq!(bytes[0], 129u8);
        assert_eq!(bytes.len(), 1 + 512);
    }

    #[test]
    fn test_block_packed_writer_ord() {
        let mut out = MemoryIndexOutput::new("test".to_string());
        let mut w = BlockPackedWriter::new(64);
        assert_eq!(w.ord(), 0);
        for i in 0..10 {
            w.add(&mut out, i).unwrap();
        }
        assert_eq!(w.ord(), 10);
        w.finish(&mut out).unwrap();
        assert_eq!(w.ord(), 10);
    }

    #[test]
    fn test_block_packed_writer_reset() {
        let mut out = MemoryIndexOutput::new("test".to_string());
        let mut w = BlockPackedWriter::new(64);
        for i in 0..10 {
            w.add(&mut out, i).unwrap();
        }
        w.finish(&mut out).unwrap();

        w.reset();
        assert_eq!(w.ord(), 0);
        let mut out2 = MemoryIndexOutput::new("test2".to_string());
        for i in 0..5 {
            w.add(&mut out2, i).unwrap();
        }
        w.finish(&mut out2).unwrap();
        assert_eq!(w.ord(), 5);
    }

    // Cross-validation tests: expected bytes from Java Lucene's BlockPackedWriter

    #[test]
    fn test_block_packed_writer_java_64x42() {
        let mut out = MemoryIndexOutput::new("test".to_string());
        let mut w = BlockPackedWriter::new(64);
        for _ in 0..64 {
            w.add(&mut out, 42).unwrap();
        }
        w.finish(&mut out).unwrap();
        #[rustfmt::skip]
        let expected: &[u8] = &[0x00, 0x53];
        assert_eq!(out.bytes(), expected);
    }

    #[test]
    fn test_block_packed_writer_java_0_to_63() {
        let mut out = MemoryIndexOutput::new("test".to_string());
        let mut w = BlockPackedWriter::new(64);
        for i in 0..64 {
            w.add(&mut out, i).unwrap();
        }
        w.finish(&mut out).unwrap();
        #[rustfmt::skip]
        let expected: &[u8] = &[
            0x0D, 0x00, 0x10, 0x83, 0x10, 0x51, 0x87, 0x20,
            0x92, 0x8B, 0x30, 0xD3, 0x8F, 0x41, 0x14, 0x93,
            0x51, 0x55, 0x97, 0x61, 0x96, 0x9B, 0x71, 0xD7,
            0x9F, 0x82, 0x18, 0xA3, 0x92, 0x59, 0xA7, 0xA2,
            0x9A, 0xAB, 0xB2, 0xDB, 0xAF, 0xC3, 0x1C, 0xB3,
            0xD3, 0x5D, 0xB7, 0xE3, 0x9E, 0xBB, 0xF3, 0xDF,
            0xBF,
        ];
        assert_eq!(out.bytes(), expected);
    }

    #[test]
    fn test_block_packed_writer_java_0_to_99() {
        let mut out = MemoryIndexOutput::new("test".to_string());
        let mut w = BlockPackedWriter::new(64);
        for i in 0..100 {
            w.add(&mut out, i).unwrap();
        }
        w.finish(&mut out).unwrap();
        #[rustfmt::skip]
        let expected: &[u8] = &[
            0x0D, 0x00, 0x10, 0x83, 0x10, 0x51, 0x87, 0x20,
            0x92, 0x8B, 0x30, 0xD3, 0x8F, 0x41, 0x14, 0x93,
            0x51, 0x55, 0x97, 0x61, 0x96, 0x9B, 0x71, 0xD7,
            0x9F, 0x82, 0x18, 0xA3, 0x92, 0x59, 0xA7, 0xA2,
            0x9A, 0xAB, 0xB2, 0xDB, 0xAF, 0xC3, 0x1C, 0xB3,
            0xD3, 0x5D, 0xB7, 0xE3, 0x9E, 0xBB, 0xF3, 0xDF,
            0xBF, 0x0C, 0x47, 0x71, 0xD7, 0x9F, 0x82, 0x18,
            0xA3, 0x92, 0x59, 0xA7, 0xA2, 0x9A, 0xAB, 0xB2,
            0xDB, 0xAF, 0xC3, 0x1C, 0xB3, 0xD3, 0x5D, 0xB7,
            0xE3, 0x9E, 0xBB, 0xF3, 0xDF, 0xBF,
        ];
        assert_eq!(out.bytes(), expected);
    }

    #[test]
    fn test_block_packed_writer_java_1000_to_1063() {
        let mut out = MemoryIndexOutput::new("test".to_string());
        let mut w = BlockPackedWriter::new(64);
        for i in 0..64 {
            w.add(&mut out, 1000 + i).unwrap();
        }
        w.finish(&mut out).unwrap();
        #[rustfmt::skip]
        let expected: &[u8] = &[
            0x0C, 0xCF, 0x0F, 0x00, 0x10, 0x83, 0x10, 0x51,
            0x87, 0x20, 0x92, 0x8B, 0x30, 0xD3, 0x8F, 0x41,
            0x14, 0x93, 0x51, 0x55, 0x97, 0x61, 0x96, 0x9B,
            0x71, 0xD7, 0x9F, 0x82, 0x18, 0xA3, 0x92, 0x59,
            0xA7, 0xA2, 0x9A, 0xAB, 0xB2, 0xDB, 0xAF, 0xC3,
            0x1C, 0xB3, 0xD3, 0x5D, 0xB7, 0xE3, 0x9E, 0xBB,
            0xF3, 0xDF, 0xBF,
        ];
        assert_eq!(out.bytes(), expected);
    }

    #[test]
    fn test_block_packed_writer_java_0_to_9() {
        let mut out = MemoryIndexOutput::new("test".to_string());
        let mut w = BlockPackedWriter::new(64);
        for i in 0..10 {
            w.add(&mut out, i).unwrap();
        }
        w.finish(&mut out).unwrap();
        #[rustfmt::skip]
        let expected: &[u8] = &[0x09, 0x01, 0x23, 0x45, 0x67, 0x89];
        assert_eq!(out.bytes(), expected);
    }

    #[test]
    fn test_block_packed_writer_java_64x0() {
        let mut out = MemoryIndexOutput::new("test".to_string());
        let mut w = BlockPackedWriter::new(64);
        for _ in 0..64 {
            w.add(&mut out, 0).unwrap();
        }
        w.finish(&mut out).unwrap();
        #[rustfmt::skip]
        let expected: &[u8] = &[0x01];
        assert_eq!(out.bytes(), expected);
    }

    #[test]
    fn test_direct_monotonic_f32_precision() {
        // Regression test: DirectMonotonicWriter must compute expected values
        // using f32 arithmetic (matching Java's float*float), not f64.
        //
        // With values [0, 258, 535, 791]:
        //   avgInc as f32 = 263.6666564941406
        //   f32(avgInc) * f32(3) rounds to 791.0  → expected[3] = 791
        //   f64(avgInc) * f64(3) = 790.9999...    → expected[3] = 790 (WRONG)
        //
        // The incorrect f64 path produces a different delta for value[3],
        // which corrupts the packed output and makes Java's reader fail.
        let mut dm = DirectMonotonicWriter::new(16);
        dm.add(0);
        dm.add(258);
        dm.add(535);
        dm.add(791);

        let mut meta = MemoryIndexOutput::new("meta".to_string());
        let mut data = MemoryIndexOutput::new("data".to_string());
        dm.finish(&mut meta, &mut data).unwrap();

        // The meta should contain: min(i64), avgInc bits(i32), offset(i64), bitsReq(u8)
        let meta_bytes = meta.bytes().to_vec();
        // min = -5 (i64 LE)
        assert_eq!(
            i64::from_le_bytes(meta_bytes[0..8].try_into().unwrap()),
            -5,
            "min should be -5"
        );

        // The data should contain the packed deltas [5, 0, 13, 5] at 4 bits each.
        // With f32 precision: deltas = [5, 0, 13, 5] (value 3: 791-791=0, +5=5)
        // With f64 precision: deltas = [5, 0, 13, 6] (value 3: 791-790=1, +5=6)  WRONG
        //
        // Packed at 4 bits LE: byte0 = 5|(0<<4) = 0x05, byte1 = 13|(5<<4) = 0x5d
        let data_bytes = data.bytes();
        assert_eq!(data_bytes.len(), 2, "4 values at 4 bits = 2 bytes");
        assert_eq!(data_bytes[0], 0x05, "byte 0: deltas[0]=5, deltas[1]=0");
        assert_eq!(
            data_bytes[1], 0x5d,
            "byte 1: deltas[2]=13, deltas[3]=5 (f32 precision required)"
        );
    }
}
