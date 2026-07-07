// SPDX-License-Identifier: Apache-2.0

//! Data encoding and decoding algorithms.
//!
//! This module centralizes all encoding/decoding logic used by codecs, index
//! writers, and index readers. Algorithms include variable-length integers,
//! zigzag encoding, packed integers, and compression.
//!
//! Writers take `&mut dyn Write`. Readers take `&mut Cursor<&[u8]>` directly,
//! operating on the cursor's underlying slice for speed on the read path.
//! The one exception is [`varint::read_vint`], which retains a `&mut dyn Read`
//! form for `ByteSliceReader` on the indexing path and for internal callers
//! in other encoding modules.

pub mod geo;
pub mod group_vint;
pub mod lowercase_ascii;
pub mod lz4;
pub mod packed;
pub mod pfor;
pub mod range;
pub mod sortable_bytes;
pub mod string;
pub mod varint;
pub mod write_encoding;
pub mod zigzag;
