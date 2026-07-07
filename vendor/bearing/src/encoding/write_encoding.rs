// SPDX-License-Identifier: Apache-2.0

//! Extension trait providing encoding-aware write methods on any [`Write`].
//!
//! [`WriteEncoding`] is blanket-implemented for all [`Write`] types, adding
//! methods like [`write_vint`](WriteEncoding::write_vint) and
//! [`write_string`](WriteEncoding::write_string) that encode Lucene's wire formats.

use std::collections::HashMap;
use std::io;
use std::io::Write;

use crate::encoding::group_vint;
use crate::encoding::string;
use crate::encoding::varint;

/// Encoding-aware write methods available on any [`Write`] type.
pub trait WriteEncoding: Write {
    /// Writes a variable-length integer (1-5 bytes). High bit = continuation.
    fn write_vint(&mut self, i: i32) -> io::Result<()>;

    /// Writes a variable-length long (1-9 bytes). High bit = continuation.
    fn write_vlong(&mut self, i: i64) -> io::Result<()>;

    /// Writes a zigzag-encoded variable-length int.
    fn write_zint(&mut self, i: i32) -> io::Result<()>;

    /// Writes a zigzag-encoded variable-length long.
    fn write_zlong(&mut self, i: i64) -> io::Result<()>;

    /// Writes a variable-length long that may be negative.
    fn write_signed_vlong(&mut self, i: i64) -> io::Result<()>;

    /// Writes a string as VInt-encoded byte length followed by UTF-8 bytes.
    fn write_string(&mut self, s: &str) -> io::Result<()>;

    /// Writes a set of strings: VInt count followed by each string.
    fn write_set_of_strings(&mut self, set: &[String]) -> io::Result<()>;

    /// Writes a map of strings: VInt count followed by key-value pairs.
    fn write_map_of_strings(&mut self, map: &HashMap<String, String>) -> io::Result<()>;

    /// Writes integers using group-varint encoding.
    fn write_group_vints(&mut self, values: &[i32], limit: usize) -> io::Result<()>;
}

impl<T: Write> WriteEncoding for T {
    fn write_vint(&mut self, i: i32) -> io::Result<()> {
        varint::write_vint(self, i)
    }

    fn write_vlong(&mut self, i: i64) -> io::Result<()> {
        varint::write_vlong(self, i)
    }

    fn write_zint(&mut self, i: i32) -> io::Result<()> {
        varint::write_zint(self, i)
    }

    fn write_zlong(&mut self, i: i64) -> io::Result<()> {
        varint::write_zlong(self, i)
    }

    fn write_signed_vlong(&mut self, i: i64) -> io::Result<()> {
        varint::write_signed_vlong(self, i)
    }

    fn write_string(&mut self, s: &str) -> io::Result<()> {
        string::write_string(self, s)
    }

    fn write_set_of_strings(&mut self, set: &[String]) -> io::Result<()> {
        string::write_set_of_strings(self, set)
    }

    fn write_map_of_strings(&mut self, map: &HashMap<String, String>) -> io::Result<()> {
        string::write_map_of_strings(self, map)
    }

    fn write_group_vints(&mut self, values: &[i32], limit: usize) -> io::Result<()> {
        group_vint::write_group_vints(self, values, limit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store;
    use crate::store::DataOutput;
    use crate::store::memory::MemoryIndexOutput;

    /// Verifies the blanket impl works on a concrete type (Vec<u8>).
    #[test]
    fn test_vint_roundtrip_on_vec() {
        let mut buf = Vec::new();
        buf.write_vint(16384).unwrap();
        let mut input = store::IndexInput::unnamed(&buf);
        assert_eq!(input.read_vint().unwrap(), 16384);
    }

    /// Verifies the blanket impl works on a DataOutput implementor.
    #[test]
    fn test_vlong_on_data_output() {
        let mut out = MemoryIndexOutput::new("test".into());
        out.write_vlong(123456789).unwrap();
        let mut input = store::IndexInput::unnamed(out.bytes());
        assert_eq!(input.read_vlong().unwrap(), 123456789);
    }

    /// Verifies the blanket impl works on a trait object (&mut dyn DataOutput).
    #[test]
    fn test_string_on_dyn_data_output() {
        let mut out = MemoryIndexOutput::new("test".into());
        let mut dyn_out: &mut dyn DataOutput = &mut out;
        dyn_out.write_string("hello").unwrap();
        let mut input = store::IndexInput::unnamed(out.bytes());
        assert_eq!(input.read_string().unwrap(), "hello");
    }

    /// Verifies set/map of strings roundtrip.
    #[test]
    fn test_set_of_strings_roundtrip() {
        let set = vec!["alpha".to_string(), "beta".to_string()];
        let mut buf = Vec::new();
        buf.write_set_of_strings(&set).unwrap();
        let mut input = store::IndexInput::unnamed(&buf);
        assert_eq!(input.read_set_of_strings().unwrap(), set);
    }
}
