// SPDX-License-Identifier: Apache-2.0

//! Postings relation (SPEC §7.8): full-enumeration reader for the
//! Lucene103 block-tree terms dictionary — `.tmd` field metadata, a
//! one-node `.tip` trie read (root only), and a sequential in-order walk
//! of `.tim` blocks. No FST traversal: sub-blocks are recursed in entry
//! order and floor chains are followed physically via the
//! `isLastInFloor` bit.
//!
//! Byte layouts traced from Bearing's `codecs/lucene103` readers (which
//! are cross-validated against Java Lucene 10.3): `blocktree_reader.rs`
//! (.tmd), `trie_reader.rs` (.tip node), `segment_terms_enum_frame.rs`
//! (.tim block sections, stats RLE, term-metadata delta chains).

use lucene_arrow_core::cursor::Cursor;
use lucene_arrow_core::{Error, Result};

pub mod build;
pub mod coo;
pub mod doc;
pub mod pfor;
pub mod segment;
pub mod text;
pub mod walk;

pub const TERMS_META_CODEC: &str = "BlockTreeTermsMeta";
pub const TERMS_CODEC: &str = "BlockTreeTermsDict";
pub const POSTINGS_BLOCK_SIZE: u32 = 128;

/// Per-field terms metadata from `.tmd`.
#[derive(Debug, Clone)]
pub struct TermsFieldMeta {
    pub field_number: u32,
    pub num_terms: u64,
    pub sum_total_term_freq: i64,
    pub sum_doc_freq: i64,
    pub doc_count: u32,
    pub min_term: Vec<u8>,
    pub max_term: Vec<u8>,
    /// Root trie node file pointer within the field's `.tip` slice.
    pub trie_root_fp: u64,
    pub index_start_fp: u64,
    pub index_end_fp: u64,
}

fn skip_index_header(c: &mut Cursor) -> Result<()> {
    let magic = c.be_u32()?;
    if magic != 0x3FD76C17 {
        return Err(Error::invalid(format!("bad header magic {magic:#x}")));
    }
    let _name = c.string()?;
    let _version = c.be_i32()?;
    c.skip(16)?; // segment id
    let suffix_len = c.u8()? as usize;
    c.skip(suffix_len)?;
    Ok(())
}

/// Parse `.tmd`: per-field metadata. `has_freqs[field]` must reflect the
/// field's IndexOptions (DOCS-only fields fold sumDocFreq into sumTTF).
pub fn parse_tmd(
    tmd: &[u8],
    has_freqs: impl Fn(u32) -> bool,
) -> Result<Vec<TermsFieldMeta>> {
    let mut c = Cursor::at(tmd, 0);
    skip_index_header(&mut c)?; // BlockTreeTermsMeta
    skip_index_header(&mut c)?; // embedded Lucene103PostingsWriterTerms
    let block_size = c.vint()? as u32;
    if block_size != POSTINGS_BLOCK_SIZE {
        return Err(Error::invalid(format!("postings block size {block_size}")));
    }
    let num_fields = c.vint()? as u32;
    let mut fields = Vec::with_capacity(num_fields as usize);
    for _ in 0..num_fields {
        let field_number = c.vint()? as u32;
        let num_terms = c.vlong()? as u64;
        let sum_total_term_freq = c.vlong()?;
        let sum_doc_freq =
            if has_freqs(field_number) { c.vlong()? } else { sum_total_term_freq };
        let doc_count = c.vint()? as u32;
        let min_len = c.vint()? as usize;
        let min_term = c.take(min_len)?.to_vec();
        let max_len = c.vint()? as usize;
        let max_term = c.take(max_len)?.to_vec();
        let index_start_fp = c.vlong()? as u64;
        let trie_root_fp = c.vlong()? as u64;
        let index_end_fp = c.vlong()? as u64;
        fields.push(TermsFieldMeta {
            field_number,
            num_terms,
            sum_total_term_freq,
            sum_doc_freq,
            doc_count,
            min_term,
            max_term,
            trie_root_fp,
            index_start_fp,
            index_end_fp,
        });
    }
    Ok(fields)
}

/// Root-block location decoded from the field's `.tip` trie root node.
#[derive(Debug, Clone, Copy)]
pub struct RootBlock {
    pub fp: u64,
    pub has_terms: bool,
    pub is_floor: bool,
}

const BYTES_MASK: [u64; 8] = [
    0xFF,
    0xFFFF,
    0xFF_FFFF,
    0xFFFF_FFFF,
    0xFF_FFFF_FFFF,
    0xFFFF_FFFF_FFFF,
    0xFF_FFFF_FFFF_FFFF,
    0xFFFF_FFFF_FFFF_FFFF,
];

fn le_long_at(tip: &[u8], pos: usize) -> Result<u64> {
    // Reads up to 8 bytes little-endian, zero-padded past EOF (the trie
    // reader over-reads by design; Bearing slices with padding).
    let mut buf = [0u8; 8];
    let avail = tip.len().saturating_sub(pos).min(8);
    buf[..avail].copy_from_slice(&tip[pos..pos + avail]);
    Ok(u64::from_le_bytes(buf))
}

/// Decode the trie ROOT node (only its output — child navigation is not
/// needed for a full walk). Ported from Bearing `trie_reader.rs::load`.
pub fn root_block(tip_field_slice: &[u8], root_fp: u64) -> Result<RootBlock> {
    const SIGN_NO_CHILDREN: u32 = 0x00;
    const SIGN_SINGLE_CHILD_WITHOUT_OUTPUT: u32 = 0x02;
    const SIGN_MULTI_CHILDREN: u32 = 0x03;
    const LEAF_HAS_TERMS: u32 = 1 << 5;
    const LEAF_HAS_FLOOR: u32 = 1 << 6;
    const NON_LEAF_HAS_TERMS: u64 = 1 << 1;
    const NON_LEAF_HAS_FLOOR: u64 = 1 << 0;

    let fp = root_fp as usize;
    let flags_long = le_long_at(tip_field_slice, fp)?;
    let flags = flags_long as u32;
    match flags & 0x03 {
        SIGN_NO_CHILDREN => {
            let nb = ((flags >> 2) & 0x07) as usize;
            let out = if nb <= 6 {
                (flags_long >> 8) & BYTES_MASK[nb]
            } else {
                le_long_at(tip_field_slice, fp + 1)?
            };
            Ok(RootBlock {
                fp: out,
                has_terms: flags & LEAF_HAS_TERMS != 0,
                is_floor: flags & LEAF_HAS_FLOOR != 0,
            })
        }
        SIGN_MULTI_CHILDREN => {
            if flags & 0x20 == 0 {
                return Err(Error::invalid("trie root has no output"));
            }
            let nb = ((flags >> 6) & 0x07) as usize;
            let l = if nb <= 4 {
                flags_long >> 24
            } else {
                le_long_at(tip_field_slice, fp + 3)?
            };
            let enc = l & BYTES_MASK[nb];
            Ok(RootBlock {
                fp: enc >> 2,
                has_terms: enc & NON_LEAF_HAS_TERMS != 0,
                is_floor: enc & NON_LEAF_HAS_FLOOR != 0,
            })
        }
        sign => {
            if sign == SIGN_SINGLE_CHILD_WITHOUT_OUTPUT {
                return Err(Error::invalid("trie root has no output"));
            }
            // SIGN_SINGLE_CHILD_WITH_OUTPUT
            let child_nb = ((flags >> 2) & 0x07) as usize;
            let out_nb = ((flags >> 5) & 0x07) as usize;
            let offset = fp + child_nb + 3;
            let enc = le_long_at(tip_field_slice, offset)? & BYTES_MASK[out_nb];
            Ok(RootBlock {
                fp: enc >> 2,
                has_terms: enc & NON_LEAF_HAS_TERMS != 0,
                is_floor: enc & NON_LEAF_HAS_FLOOR != 0,
            })
        }
    }
}
