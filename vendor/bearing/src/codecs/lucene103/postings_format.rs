// SPDX-License-Identifier: Apache-2.0
//! Postings format constants and term state for the Lucene 10.3 codec.

use std::io;

use crate::codecs::codec_headers;
use crate::codecs::codec_headers::check_index_header;
use crate::store::IndexInput;

// --- Postings format constants ---

pub const BLOCK_SIZE: usize = 128;
pub const LEVEL1_FACTOR: usize = 32;
pub const LEVEL1_NUM_DOCS: usize = LEVEL1_FACTOR * BLOCK_SIZE;
pub const LEVEL1_MASK: usize = LEVEL1_NUM_DOCS - 1;

pub const VERSION_START: i32 = 0;
pub const VERSION_CURRENT: i32 = VERSION_START;

// Postings file extensions
pub const META_EXTENSION: &str = "psm";
pub const DOC_EXTENSION: &str = "doc";
pub const POS_EXTENSION: &str = "pos";
pub const PAY_EXTENSION: &str = "pay";
// Postings codec names
pub const TERMS_CODEC: &str = "Lucene103PostingsWriterTerms";
pub const META_CODEC: &str = "Lucene103PostingsWriterMeta";
pub const DOC_CODEC: &str = "Lucene103PostingsWriterDoc";
pub const POS_CODEC: &str = "Lucene103PostingsWriterPos";
pub const PAY_CODEC: &str = "Lucene103PostingsWriterPay";

// BlockTree terms dict constants
pub const TERMS_EXTENSION: &str = "tim";
pub const TERMS_INDEX_EXTENSION: &str = "tip";
pub const TERMS_META_EXTENSION: &str = "tmd";

pub const TERMS_CODEC_NAME: &str = "BlockTreeTermsDict";
pub const TERMS_INDEX_CODEC_NAME: &str = "BlockTreeTermsIndex";
pub const TERMS_META_CODEC_NAME: &str = "BlockTreeTermsMeta";

pub const BLOCKTREE_VERSION_START: i32 = 0;
pub const BLOCKTREE_VERSION_CURRENT: i32 = BLOCKTREE_VERSION_START;

pub const DEFAULT_MIN_BLOCK_SIZE: usize = 25;
pub const DEFAULT_MAX_BLOCK_SIZE: usize = 48;

/// Validates the embedded postings header inside `.tmd`. The .tmd outer
/// header is validated by `CodecFileHandle::open(IndexFile::TermsMeta, ...)`;
/// this second header lives mid-stream in the .tmd body (Java's
/// `Lucene103PostingsReader::init` pattern).
pub(crate) fn check_embedded_postings_header(
    input: &mut IndexInput<'_>,
    segment_id: &[u8; codec_headers::ID_LENGTH],
    segment_suffix: &str,
) -> io::Result<()> {
    check_index_header(
        input,
        TERMS_CODEC,
        VERSION_START,
        VERSION_CURRENT,
        segment_id,
        segment_suffix,
    )?;
    Ok(())
}

/// Per-term metadata stored in .tim blocks.
#[derive(Clone, Copy, Debug)]
pub struct IntBlockTermState {
    // From BlockTermState base class
    pub doc_freq: i32,
    pub total_term_freq: i64,
    /// Ordinal of this term within its block (used for lazy metadata decoding).
    pub term_block_ord: i32,
    // Lucene103-specific fields
    pub doc_start_fp: i64,
    pub pos_start_fp: i64,
    pub pay_start_fp: i64,
    pub last_pos_block_offset: i64,
    pub singleton_doc_id: i32,
}

impl IntBlockTermState {
    pub fn new() -> Self {
        Self {
            doc_freq: 0,
            total_term_freq: 0,
            term_block_ord: 0,
            doc_start_fp: 0,
            pos_start_fp: 0,
            pay_start_fp: 0,
            last_pos_block_offset: -1,
            singleton_doc_id: -1,
        }
    }
}

impl Default for IntBlockTermState {
    fn default() -> Self {
        Self::new()
    }
}
