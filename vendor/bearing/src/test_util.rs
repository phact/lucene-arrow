// SPDX-License-Identifier: Apache-2.0

use std::str;

use crate::document::{DocValuesType, IndexOptions};
use crate::index::{FieldInfo, PointDimensionConfig};

/// Creates a [`FieldInfo`] with common test defaults (no term vectors, no payloads, no points).
///
/// Covers the three common test patterns:
/// - Doc values fields: `(name, number, true, IndexOptions::None, dv_type)`
/// - Norms fields: `(name, number, !has_norms, IndexOptions::DocsAndFreqsAndPositions, DocValuesType::None)`
/// - Indexed fields: `(name, number, false, index_opts, DocValuesType::None)`
pub fn make_field_info(
    name: &str,
    number: u32,
    omit_norms: bool,
    index_options: IndexOptions,
    doc_values_type: DocValuesType,
) -> FieldInfo {
    FieldInfo::new(
        name.to_string(),
        number,
        false,
        omit_norms,
        index_options,
        doc_values_type,
        PointDimensionConfig::default(),
    )
}

/// A test helper that reads binary values from a byte slice, advancing a position cursor.
///
/// Supports both big-endian and little-endian read methods, matching the dual
/// endianness used throughout the Lucene format.
pub struct TestDataReader<'a> {
    pub data: &'a [u8],
    pub pos: usize,
}

impl<'a> TestDataReader<'a> {
    pub fn new(data: &'a [u8], pos: usize) -> Self {
        Self { data, pos }
    }

    // -- Single byte --

    pub fn read_byte(&mut self) -> u8 {
        let v = self.data[self.pos];
        self.pos += 1;
        v
    }

    // -- Big-endian reads (codec headers/footers) --

    pub fn read_be_int(&mut self) -> i32 {
        let bytes: [u8; 4] = self.data[self.pos..self.pos + 4].try_into().unwrap();
        self.pos += 4;
        i32::from_be_bytes(bytes)
    }

    pub fn read_be_long(&mut self) -> i64 {
        let bytes: [u8; 8] = self.data[self.pos..self.pos + 8].try_into().unwrap();
        self.pos += 8;
        i64::from_be_bytes(bytes)
    }

    // -- Little-endian reads (data fields) --

    pub fn read_le_short(&mut self) -> i16 {
        let v = i16::from_le_bytes(self.data[self.pos..self.pos + 2].try_into().unwrap());
        self.pos += 2;
        v
    }

    pub fn read_le_int(&mut self) -> i32 {
        let v = i32::from_le_bytes(self.data[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        v
    }

    pub fn read_le_long(&mut self) -> i64 {
        let v = i64::from_le_bytes(self.data[self.pos..self.pos + 8].try_into().unwrap());
        self.pos += 8;
        v
    }

    // -- Variable-length reads --

    pub fn read_vint(&mut self) -> i32 {
        let mut result: u32 = 0;
        let mut shift = 0;
        loop {
            let b = self.data[self.pos] as u32;
            self.pos += 1;
            result |= (b & 0x7F) << shift;
            if b & 0x80 == 0 {
                break;
            }
            shift += 7;
        }
        result as i32
    }

    pub fn read_vlong(&mut self) -> i64 {
        let mut result: u64 = 0;
        let mut shift = 0;
        loop {
            let b = self.data[self.pos] as u64;
            self.pos += 1;
            result |= (b & 0x7F) << shift;
            if b & 0x80 == 0 {
                break;
            }
            shift += 7;
        }
        result as i64
    }

    // -- String --

    pub fn read_string(&mut self) -> String {
        let len = self.read_vint() as usize;
        let s = str::from_utf8(&self.data[self.pos..self.pos + len])
            .unwrap()
            .to_string();
        self.pos += len;
        s
    }
}
