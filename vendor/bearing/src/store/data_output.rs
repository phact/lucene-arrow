// SPDX-License-Identifier: Apache-2.0

//! Byte-level writing primitives.
//!
//! [`DataOutput`] defines the interface for writing raw bytes to an output
//! stream. Encoding-aware methods (VInt, strings, etc.) are provided by
//! [`WriteEncoding`](crate::encoding::write_encoding::WriteEncoding), which is
//! blanket-implemented for all [`Write`] types.
//!
//! [`VecOutput`] is a convenience adapter for writing into a `Vec<u8>`.

use std::io;
use std::io::Write;

/// Trait for writing raw bytes to an output stream.
///
/// Extends [`Write`] with a single-byte write and convenience methods for
/// fixed-width integers in little-endian and big-endian byte order. All
/// implementors must also implement [`Write`].
pub trait DataOutput: Write {
    /// Writes a single byte.
    fn write_byte(&mut self, b: u8) -> io::Result<()>;

    /// Writes a 32-bit integer in **little-endian** byte order.
    fn write_le_int(&mut self, i: i32) -> io::Result<()> {
        self.write_all(&i.to_le_bytes())
    }

    /// Writes a 16-bit short in **little-endian** byte order.
    fn write_le_short(&mut self, i: i16) -> io::Result<()> {
        self.write_all(&i.to_le_bytes())
    }

    /// Writes a 64-bit long in **little-endian** byte order.
    fn write_le_long(&mut self, i: i64) -> io::Result<()> {
        self.write_all(&i.to_le_bytes())
    }

    /// Writes a 32-bit integer in **big-endian** byte order.
    /// Used by CodecUtil for headers/footers.
    fn write_be_int(&mut self, i: i32) -> io::Result<()> {
        self.write_all(&i.to_be_bytes())
    }

    /// Writes a 64-bit long in **big-endian** byte order.
    /// Used by CodecUtil for headers/footers.
    fn write_be_long(&mut self, i: i64) -> io::Result<()> {
        self.write_all(&i.to_be_bytes())
    }
}

/// A [`DataOutput`] adapter for writing directly into a `Vec<u8>`.
///
/// Use this instead of defining ad-hoc wrapper structs in individual codec modules.
/// The inner `Vec<u8>` is borrowed mutably, so you can inspect it after writing.
pub struct VecOutput<'a>(pub &'a mut Vec<u8>);

impl Write for VecOutput<'_> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl DataOutput for VecOutput<'_> {
    fn write_byte(&mut self, b: u8) -> io::Result<()> {
        self.0.push(b);
        Ok(())
    }
}
