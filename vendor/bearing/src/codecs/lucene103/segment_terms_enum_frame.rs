// SPDX-License-Identifier: Apache-2.0

//! Per-block frame for the segment terms enumerator.
//!
//! Each frame in the [`super::segment_terms_enum::SegmentTermsEnum`] stack
//! represents one loaded block at a specific depth in the block tree. Frames
//! own their decoded block data (suffix bytes, suffix lengths, stats, metadata)
//! and provide methods to iterate entries and decode term metadata.

use std::cmp::Ordering;
use std::io;

use crate::codecs::lucene103::postings_format::IntBlockTermState;
use crate::codecs::lucene103::segment_terms_enum::read_compressed;
use crate::codecs::lucene103::trie_reader::Node;
use crate::document::IndexOptions;
use crate::encoding::{pfor, zigzag};
use crate::index::terms::SeekStatus;
use crate::store::IndexInput;

/// File pointers tracking where a block lives in the `.tim` file.
#[derive(Debug, Default)]
pub(super) struct BlockPosition {
    /// File pointer where this block was loaded from.
    pub fp: i64,
    /// Original file pointer (before floor block navigation).
    pub fp_orig: i64,
    /// File pointer to the end of this block (start of next block).
    pub fp_end: i64,
}

/// Boolean flags describing the structure of a loaded block.
#[derive(Debug, Default)]
pub(super) struct BlockFlags {
    /// Whether this block has term entries (vs. only sub-blocks).
    pub has_terms: bool,
    /// Original `has_terms` value (before iteration changes it).
    pub has_terms_orig: bool,
    /// Whether this is a floor block.
    pub is_floor: bool,
    /// Whether this is the last sub-block in a floor block.
    pub is_last_in_floor: bool,
    /// Whether all entries are terms (no sub-blocks).
    pub is_leaf_block: bool,
    /// Whether all suffix lengths are equal.
    pub all_equal: bool,
}

/// Decoded block data read from the `.tim` file.
///
/// Contains the five sections of a term block: suffix bytes, suffix lengths,
/// stats, and metadata, along with reader cursors for suffix iteration.
#[derive(Debug, Default)]
pub(super) struct BlockData {
    /// Decompressed suffix bytes for all entries.
    pub suffix_bytes: Vec<u8>,
    /// Suffix reader position (cursor into `suffix_bytes`).
    pub suffix_pos: usize,
    /// Suffix length bytes for all entries.
    pub suffix_length_bytes: Vec<u8>,
    /// Suffix length reader position (cursor into `suffix_length_bytes`).
    pub suffix_length_pos: usize,
    /// Stats bytes for all entries.
    pub stat_bytes: Vec<u8>,
    /// Stats reader position (cursor into `stat_bytes`).
    pub stats_pos: usize,
    /// Metadata bytes for all entries.
    pub meta_bytes: Vec<u8>,
    /// Metadata reader position (cursor into `meta_bytes`).
    pub meta_pos: usize,
    /// Total suffix bytes in block (for stats).
    pub total_suffix_bytes: i64,
}

/// Floor block navigation state from the `.tip` file.
#[derive(Debug, Default)]
pub(super) struct FloorState {
    /// Floor data bytes.
    pub data: Vec<u8>,
    /// Rewind position within `data`.
    pub rewind_pos: usize,
    /// Current read position within `data`.
    pub data_pos: usize,
    /// Label of the next floor block.
    pub next_label: i32,
    /// Number of follow-on floor blocks remaining.
    pub num_follow_blocks: i32,
}

impl FloorState {
    /// Reads floor data from an `IndexInput` at the given file pointer.
    ///
    /// Parses through the floor data structure to determine its size,
    /// then reads all bytes into an owned buffer.
    pub fn load_from_input(
        &mut self,
        index_in: &mut IndexInput<'_>,
        floor_data_fp: i64,
    ) -> io::Result<()> {
        let start = floor_data_fp as usize;
        index_in.seek(start)?;

        // Parse through to find total byte length
        let num_follow = index_in.read_vint()?;
        index_in.read_byte()?; // first label
        for i in 0..num_follow {
            index_in.read_vlong()?; // code
            if i < num_follow - 1 {
                index_in.read_byte()?; // next label
            }
        }
        let total_len = index_in.position() - start;

        // Re-read all bytes into buffer
        self.data = vec![0u8; total_len];
        index_in.seek(start)?;
        index_in.read_bytes(&mut self.data)?;

        // Parse initial fields
        let mut reader = IndexInput::new("floor", &self.data);
        self.rewind_pos = 0;
        self.num_follow_blocks = reader.read_vint()?;
        self.next_label = reader.read_byte()? as i32 & 0xff;
        self.data_pos = reader.position();
        Ok(())
    }

    /// Parses floor data from a byte slice.
    #[cfg(test)]
    pub fn set(&mut self, bytes: &[u8]) -> io::Result<()> {
        self.data = bytes.to_vec();
        let mut reader = IndexInput::new("floor", &self.data);
        self.rewind_pos = 0;
        self.num_follow_blocks = reader.read_vint()?;
        self.next_label = reader.read_byte()? as i32 & 0xff;
        self.data_pos = reader.position();
        Ok(())
    }

    /// Rewinds floor state to its initial position.
    pub fn rewind(&mut self) -> io::Result<()> {
        let mut reader = IndexInput::new("floor", &self.data);
        reader.skip_bytes(self.rewind_pos)?;
        self.num_follow_blocks = reader.read_vint()?;
        debug_assert!(self.num_follow_blocks > 0);
        self.next_label = reader.read_byte()? as i32 & 0xff;
        self.data_pos = reader.position();
        Ok(())
    }
}

/// A single frame in the [`SegmentTermsEnum`](super::segment_terms_enum::SegmentTermsEnum)
/// frame stack.
///
/// Each frame holds the decoded contents of one term block and tracks
/// iteration state within that block.
#[derive(Debug)]
pub(super) struct SegmentTermsEnumFrame {
    /// Index of this frame in the stack.
    pub ord: i32,
    /// Block file pointers in `.tim`.
    pub pos: BlockPosition,
    /// Structural flags for the loaded block.
    pub flags: BlockFlags,
    /// Decoded block section data.
    pub data: BlockData,
    /// Floor block navigation state.
    pub floor: FloorState,

    /// Length of the prefix shared by all terms in this block.
    pub prefix_length: usize,
    /// Number of entries (term or sub-block) in this block.
    pub ent_count: usize,
    /// Which entry we will next read, or -1 if block isn't loaded yet.
    pub next_ent: i32,

    /// Remaining count in a singleton stats run.
    pub stats_singleton_run_length: i32,
    /// How many terms have had their metadata decoded so far.
    pub meta_data_upto: i32,
    /// The current term's block term state.
    pub state: IntBlockTermState,
    /// File pointer of the last sub-block seen during iteration.
    pub last_sub_fp: i64,
    /// Sub-block FP delta code from the suffix lengths stream.
    pub sub_code: i64,

    /// The trie node at this frame's depth.
    pub node: Option<Node>,
    /// Start byte position of the current entry's suffix in `suffix_bytes`.
    pub start_byte_pos: usize,
    /// Length of the current entry's suffix.
    pub suffix_length: usize,
}

impl SegmentTermsEnumFrame {
    /// Creates a new unloaded frame at the given stack position.
    pub fn new(ord: i32) -> Self {
        Self {
            ord,
            pos: BlockPosition::default(),
            flags: BlockFlags::default(),
            data: BlockData::default(),
            floor: FloorState::default(),
            prefix_length: 0,
            ent_count: 0,
            next_ent: -1,
            stats_singleton_run_length: 0,
            meta_data_upto: 0,
            state: IntBlockTermState::new(),
            last_sub_fp: -1,
            sub_code: 0,
            node: None,
            start_byte_pos: 0,
            suffix_length: 0,
        }
    }

    /// Returns the current term block ordinal.
    pub fn get_term_block_ord(&self) -> i32 {
        if self.flags.is_leaf_block {
            self.next_ent
        } else {
            self.state.term_block_ord
        }
    }

    /// Loads the next floor block.
    pub fn load_next_floor_block(&mut self, terms_in: &mut IndexInput<'_>) -> io::Result<()> {
        debug_assert!(self.node.is_none() || self.flags.is_floor);
        self.pos.fp = self.pos.fp_end;
        self.next_ent = -1;
        self.load_block(terms_in)
    }

    /// Loads a term block from the `.tim` file at the current `fp`.
    ///
    /// Reads block header, suffix data, suffix lengths, stats, and metadata
    /// into owned buffers. Does not decode stats or metadata — that happens
    /// lazily via `decode_meta_data`.
    pub fn load_block(&mut self, terms_in: &mut IndexInput<'_>) -> io::Result<()> {
        if self.next_ent != -1 {
            // Already loaded
            return Ok(());
        }

        terms_in.seek(self.pos.fp as usize)?;

        // Section 1: Block header
        let code = terms_in.read_vint()?;
        self.ent_count = (code as u32 >> 1) as usize;
        debug_assert!(self.ent_count > 0);
        self.flags.is_last_in_floor = (code & 1) != 0;

        debug_assert!(
            self.node.is_none() || self.flags.is_last_in_floor || self.flags.is_floor,
            "fp={} node={:?} is_floor={} is_last_in_floor={}",
            self.pos.fp,
            self.node,
            self.flags.is_floor,
            self.flags.is_last_in_floor,
        );

        // Section 2: Suffix data
        let start_suffix_fp = terms_in.position();
        let code_l = terms_in.read_vlong()?;
        self.flags.is_leaf_block = (code_l & 0x04) != 0;
        let num_suffix_bytes = (code_l as u64 >> 3) as usize;
        let compression_code = (code_l & 0x03) as u32;

        self.data.suffix_bytes = read_compressed(terms_in, num_suffix_bytes, compression_code)?;
        self.data.suffix_pos = 0;

        // Section 3: Suffix lengths
        let num_suffix_length_bytes_token = terms_in.read_vint()?;
        self.flags.all_equal = (num_suffix_length_bytes_token & 1) != 0;
        let num_suffix_length_bytes = (num_suffix_length_bytes_token as u32 >> 1) as usize;

        if self.flags.all_equal {
            let common = terms_in.read_byte()?;
            self.data.suffix_length_bytes = vec![common; num_suffix_length_bytes];
        } else {
            self.data.suffix_length_bytes = vec![0u8; num_suffix_length_bytes];
            terms_in.read_bytes(&mut self.data.suffix_length_bytes)?;
        }
        self.data.suffix_length_pos = 0;
        self.data.total_suffix_bytes = terms_in.position() as i64 - start_suffix_fp as i64;

        // Section 4: Stats
        let num_stat_bytes = terms_in.read_vint()? as usize;
        self.data.stat_bytes = vec![0u8; num_stat_bytes];
        terms_in.read_bytes(&mut self.data.stat_bytes)?;
        self.data.stats_pos = 0;
        self.stats_singleton_run_length = 0;
        self.meta_data_upto = 0;

        self.state.term_block_ord = 0;
        self.next_ent = 0;
        self.last_sub_fp = -1;

        // Section 5: Metadata
        let num_meta_bytes = terms_in.read_vint()? as usize;
        self.data.meta_bytes = vec![0u8; num_meta_bytes];
        terms_in.read_bytes(&mut self.data.meta_bytes)?;

        self.data.meta_pos = 0;

        // fp_end = position after all block data
        self.pos.fp_end = terms_in.position() as i64;

        Ok(())
    }

    /// Decodes the next entry in the block. Returns `true` if the entry is a
    /// sub-block, `false` if it is a term.
    ///
    /// For leaf blocks, delegates to [`next_leaf`](Self::next_leaf). For
    /// non-leaf blocks, delegates to [`next_non_leaf`](Self::next_non_leaf).
    pub fn next(
        &mut self,
        term: &mut Vec<u8>,
        term_exists: &mut bool,
        terms_in: &mut IndexInput<'_>,
    ) -> io::Result<bool> {
        if self.flags.is_leaf_block {
            self.next_leaf(term, term_exists)?;
            Ok(false)
        } else {
            self.next_non_leaf(term, term_exists, terms_in)
        }
    }

    /// Advances to the next entry in a leaf block.
    ///
    /// Reads the suffix length and suffix bytes, copies them into the shared
    /// term buffer, and sets `term_exists = true`.
    pub fn next_leaf(&mut self, term: &mut Vec<u8>, term_exists: &mut bool) -> io::Result<()> {
        debug_assert!(self.next_ent != -1 && self.next_ent < self.ent_count as i32);
        self.next_ent += 1;

        let mut suffix_lengths_reader = IndexInput::new(
            "suffix_lengths",
            &self.data.suffix_length_bytes[self.data.suffix_length_pos..],
        );
        self.suffix_length = suffix_lengths_reader.read_vint()? as usize;
        self.data.suffix_length_pos += suffix_lengths_reader.position();

        self.start_byte_pos = self.data.suffix_pos;
        term.resize(self.prefix_length + self.suffix_length, 0);
        term[self.prefix_length..self.prefix_length + self.suffix_length].copy_from_slice(
            &self.data.suffix_bytes
                [self.data.suffix_pos..self.data.suffix_pos + self.suffix_length],
        );
        self.data.suffix_pos += self.suffix_length;

        *term_exists = true;
        Ok(())
    }

    /// Advances to the next entry in a non-leaf block.
    ///
    /// Returns `true` if the entry is a sub-block, `false` if it is a term.
    /// When a sub-block is encountered, `last_sub_fp` is set to the sub-block's
    /// absolute file pointer. When all entries are exhausted, loads the next
    /// floor block and continues.
    pub fn next_non_leaf(
        &mut self,
        term: &mut Vec<u8>,
        term_exists: &mut bool,
        terms_in: &mut IndexInput<'_>,
    ) -> io::Result<bool> {
        loop {
            if self.next_ent == self.ent_count as i32 {
                debug_assert!(
                    self.node.is_none() || (self.flags.is_floor && !self.flags.is_last_in_floor),
                    "is_floor={} is_last_in_floor={}",
                    self.flags.is_floor,
                    self.flags.is_last_in_floor,
                );
                self.load_next_floor_block(terms_in)?;
                if self.flags.is_leaf_block {
                    self.next_leaf(term, term_exists)?;
                    return Ok(false);
                } else {
                    continue;
                }
            }

            debug_assert!(self.next_ent != -1 && self.next_ent < self.ent_count as i32);
            self.next_ent += 1;

            let mut suffix_lengths_reader = IndexInput::new(
                "suffix_lengths",
                &self.data.suffix_length_bytes[self.data.suffix_length_pos..],
            );
            let code = suffix_lengths_reader.read_vint()?;
            self.suffix_length = (code as u32 >> 1) as usize;

            self.start_byte_pos = self.data.suffix_pos;
            term.resize(self.prefix_length + self.suffix_length, 0);
            term[self.prefix_length..self.prefix_length + self.suffix_length].copy_from_slice(
                &self.data.suffix_bytes
                    [self.data.suffix_pos..self.data.suffix_pos + self.suffix_length],
            );
            self.data.suffix_pos += self.suffix_length;

            if (code & 1) == 0 {
                // A normal term
                *term_exists = true;
                self.sub_code = 0;
                self.state.term_block_ord += 1;
                self.data.suffix_length_pos += suffix_lengths_reader.position();
                return Ok(false);
            } else {
                // A sub-block; make sub-FP absolute
                *term_exists = false;
                self.sub_code = suffix_lengths_reader.read_vlong()?;
                self.last_sub_fp = self.pos.fp - self.sub_code;
                self.data.suffix_length_pos += suffix_lengths_reader.position();
                return Ok(true);
            }
        }
    }

    /// Scans floor data to find the correct sub-block for the target term.
    ///
    /// If the target's byte at `prefix_length` is past the current floor
    /// block's label, advances through floor entries until the correct block
    /// is found. Forces a block reload if the file pointer changes.
    pub fn scan_to_floor_frame(&mut self, target: &[u8]) {
        if !self.flags.is_floor || target.len() <= self.prefix_length {
            return;
        }

        let target_label = target[self.prefix_length] as i32 & 0xFF;

        if target_label < self.floor.next_label {
            return;
        }

        debug_assert!(self.floor.num_follow_blocks != 0);

        let mut new_fp;
        let mut reader = IndexInput::new("floor_nav", &self.floor.data[self.floor.data_pos..]);
        loop {
            let code = reader.read_vlong().unwrap();
            new_fp = self.pos.fp_orig + (code >> 1);
            self.flags.has_terms = (code & 1) != 0;

            self.flags.is_last_in_floor = self.floor.num_follow_blocks == 1;
            self.floor.num_follow_blocks -= 1;

            if self.flags.is_last_in_floor {
                self.floor.next_label = 256;
                break;
            } else {
                self.floor.next_label = reader.read_byte().unwrap() as i32 & 0xff;
                if target_label < self.floor.next_label {
                    break;
                }
            }
        }
        self.floor.data_pos += reader.position();
        if new_fp != self.pos.fp {
            self.next_ent = -1;
            self.pos.fp = new_fp;
        }
    }

    /// Scans entries to find the sub-block with the given file pointer.
    ///
    /// Used by `next()` when popping back to a parent frame — advances through
    /// entries until the sub-block matching `sub_fp` is found.
    pub fn scan_to_sub_block(&mut self, sub_fp: i64) {
        debug_assert!(!self.flags.is_leaf_block);
        if self.last_sub_fp == sub_fp {
            return;
        }
        debug_assert!(sub_fp < self.pos.fp);
        let target_sub_code = self.pos.fp - sub_fp;

        let mut suffix_lengths_reader = IndexInput::new(
            "suffix_lengths",
            &self.data.suffix_length_bytes[self.data.suffix_length_pos..],
        );
        let mut suffix_skip = 0usize;

        loop {
            debug_assert!(self.next_ent < self.ent_count as i32);
            self.next_ent += 1;
            let code = suffix_lengths_reader.read_vint().unwrap();
            suffix_skip += (code as u32 >> 1) as usize;
            if (code & 1) != 0 {
                let sub_code = suffix_lengths_reader.read_vlong().unwrap();
                if target_sub_code == sub_code {
                    self.last_sub_fp = sub_fp;
                    self.data.suffix_length_pos += suffix_lengths_reader.position();
                    self.data.suffix_pos += suffix_skip;
                    return;
                }
            } else {
                self.state.term_block_ord += 1;
            }
        }
    }

    /// Lazily decodes stats and metadata up to the current term position.
    ///
    /// Only decodes entries that haven't been decoded yet (`meta_data_upto`
    /// through `get_term_block_ord()`). This makes `next()` fast for
    /// term-only iteration — metadata is only decoded when `doc_freq()`,
    /// `total_term_freq()`, or `term_state()` is called.
    pub fn decode_meta_data(&mut self, index_options: IndexOptions) -> io::Result<()> {
        let limit = self.get_term_block_ord();
        let mut absolute = self.meta_data_upto == 0;
        debug_assert!(limit > 0);

        let mut stats_reader =
            IndexInput::new("stats", &self.data.stat_bytes[self.data.stats_pos..]);
        let mut meta_reader = IndexInput::new("meta", &self.data.meta_bytes[self.data.meta_pos..]);

        while self.meta_data_upto < limit {
            // Stats decoding (with singleton RLE)
            if self.stats_singleton_run_length > 0 {
                self.state.doc_freq = 1;
                self.state.total_term_freq = 1;
                self.stats_singleton_run_length -= 1;
            } else {
                let token = stats_reader.read_vint()?;
                if (token & 1) == 1 {
                    self.state.doc_freq = 1;
                    self.state.total_term_freq = 1;
                    self.stats_singleton_run_length = token >> 1;
                } else {
                    self.state.doc_freq = token >> 1;
                    if index_options == IndexOptions::Docs {
                        self.state.total_term_freq = self.state.doc_freq as i64;
                    } else {
                        self.state.total_term_freq =
                            self.state.doc_freq as i64 + stats_reader.read_vlong()?;
                    }
                }
            }

            // Metadata decoding — state accumulates across iterations.
            // On the first term (absolute=true), file pointers are absolute.
            // On subsequent terms, they are delta-encoded relative to the
            // previous state values already in self.state.
            Self::decode_term_meta(&mut meta_reader, &mut self.state, absolute, index_options)?;

            self.meta_data_upto += 1;
            absolute = false;
        }

        self.data.stats_pos += stats_reader.position();
        self.data.meta_pos += meta_reader.position();
        self.state.term_block_ord = self.meta_data_upto;

        Ok(())
    }

    /// Decode one term's postings metadata from the metadata bytes.
    ///
    /// When `absolute` is true, file pointers are read as absolute values
    /// (state fields are zeroed first). When false, values are delta-encoded
    /// relative to the current state.
    fn decode_term_meta(
        reader: &mut IndexInput<'_>,
        state: &mut IntBlockTermState,
        absolute: bool,
        index_options: IndexOptions,
    ) -> io::Result<()> {
        if absolute {
            state.doc_start_fp = 0;
            state.pos_start_fp = 0;
            state.pay_start_fp = 0;
            state.singleton_doc_id = -1;
        }

        let code = reader.read_vlong()?;
        if (code & 1) != 0 {
            let encoded = code >> 1;
            let delta = zigzag::decode_i64(encoded);
            state.singleton_doc_id = (state.singleton_doc_id as i64 + delta) as i32;
            // doc_start_fp unchanged (same as previous)
        } else {
            let fp_delta = code >> 1;
            state.doc_start_fp += fp_delta;
            if state.doc_freq == 1 {
                state.singleton_doc_id = reader.read_vint()?;
            } else {
                state.singleton_doc_id = -1;
            }
        }

        if index_options.has_positions() {
            state.pos_start_fp += reader.read_vlong()?;
            if index_options.has_offsets() {
                state.pay_start_fp += reader.read_vlong()?;
            }
            if state.total_term_freq > pfor::BLOCK_SIZE as i64 {
                state.last_pos_block_offset = reader.read_vlong()?;
            } else {
                state.last_pos_block_offset = -1;
            }
        }

        Ok(())
    }

    /// Copies the current suffix into the shared term buffer.
    fn fill_term(&self, term: &mut Vec<u8>) {
        let term_length = self.prefix_length + self.suffix_length;
        term.resize(term_length, 0);
        term[self.prefix_length..term_length].copy_from_slice(
            &self.data.suffix_bytes[self.start_byte_pos..self.start_byte_pos + self.suffix_length],
        );
    }

    /// Scans this block for a target term.
    ///
    /// If `exact_only` is true, only exact matches return [`SeekStatus::Found`].
    /// If false, positions to the first term >= target (for `seek_ceil`).
    pub fn scan_to_term(
        &mut self,
        target: &[u8],
        exact_only: bool,
        term: &mut Vec<u8>,
        term_exists: &mut bool,
    ) -> io::Result<SeekStatus> {
        if self.flags.is_leaf_block {
            if self.flags.all_equal {
                self.binary_search_term_leaf(target, exact_only, term, term_exists)
            } else {
                self.scan_to_term_leaf(target, exact_only, term, term_exists)
            }
        } else {
            self.scan_to_term_non_leaf(target, exact_only, term, term_exists)
        }
    }

    /// Scans a leaf block for the target term using linear scan.
    fn scan_to_term_leaf(
        &mut self,
        target: &[u8],
        exact_only: bool,
        term: &mut Vec<u8>,
        term_exists: &mut bool,
    ) -> io::Result<SeekStatus> {
        debug_assert!(self.next_ent != -1);

        *term_exists = true;
        self.sub_code = 0;

        if self.next_ent == self.ent_count as i32 {
            if exact_only {
                self.fill_term(term);
            }
            return Ok(SeekStatus::End);
        }

        loop {
            self.next_ent += 1;

            let mut suffix_lengths_reader = IndexInput::new(
                "suffix_lengths",
                &self.data.suffix_length_bytes[self.data.suffix_length_pos..],
            );
            self.suffix_length = suffix_lengths_reader.read_vint()? as usize;
            self.data.suffix_length_pos += suffix_lengths_reader.position();

            self.start_byte_pos = self.data.suffix_pos;
            self.data.suffix_pos += self.suffix_length;

            let cmp = self.data.suffix_bytes
                [self.start_byte_pos..self.start_byte_pos + self.suffix_length]
                .cmp(&target[self.prefix_length..]);

            if cmp.is_lt() {
                // Current entry is still before the target; keep scanning
            } else if cmp.is_gt() {
                self.fill_term(term);
                return Ok(SeekStatus::NotFound);
            } else {
                self.fill_term(term);
                return Ok(SeekStatus::Found);
            }

            if self.next_ent >= self.ent_count as i32 {
                break;
            }
        }

        if exact_only {
            self.fill_term(term);
        }

        Ok(SeekStatus::End)
    }

    /// Binary searches a leaf block where all suffixes have equal length.
    fn binary_search_term_leaf(
        &mut self,
        target: &[u8],
        exact_only: bool,
        term: &mut Vec<u8>,
        term_exists: &mut bool,
    ) -> io::Result<SeekStatus> {
        debug_assert!(self.next_ent != -1);

        *term_exists = true;
        self.sub_code = 0;

        if self.next_ent == self.ent_count as i32 {
            if exact_only {
                self.fill_term(term);
            }
            return Ok(SeekStatus::End);
        }

        let mut suffix_lengths_reader = IndexInput::new(
            "suffix_lengths",
            &self.data.suffix_length_bytes[self.data.suffix_length_pos..],
        );
        self.suffix_length = suffix_lengths_reader.read_vint()? as usize;

        let mut start = self.next_ent;
        let mut end = self.ent_count as i32 - 1;
        let mut cmp = 0i32;

        while start <= end {
            let mid = ((start + end) as u32 >> 1) as i32;
            self.next_ent = mid + 1;
            self.start_byte_pos = mid as usize * self.suffix_length;

            let suffix = &self.data.suffix_bytes
                [self.start_byte_pos..self.start_byte_pos + self.suffix_length];
            let target_suffix = &target[self.prefix_length..];

            cmp = match suffix.cmp(target_suffix) {
                Ordering::Less => -1,
                Ordering::Equal => 0,
                Ordering::Greater => 1,
            };

            if cmp < 0 {
                start = mid + 1;
            } else if cmp > 0 {
                end = mid - 1;
            } else {
                self.data.suffix_pos = self.start_byte_pos + self.suffix_length;
                self.fill_term(term);
                return Ok(SeekStatus::Found);
            }
        }

        let seek_status;
        if end < self.ent_count as i32 - 1 {
            seek_status = SeekStatus::NotFound;
            if cmp < 0 {
                self.start_byte_pos += self.suffix_length;
                self.next_ent += 1;
            }
            self.data.suffix_pos = self.start_byte_pos + self.suffix_length;
            self.fill_term(term);
        } else {
            seek_status = SeekStatus::End;
            self.data.suffix_pos = self.start_byte_pos + self.suffix_length;
            if exact_only {
                self.fill_term(term);
            }
        }

        Ok(seek_status)
    }

    /// Scans a non-leaf block for the target term.
    fn scan_to_term_non_leaf(
        &mut self,
        target: &[u8],
        exact_only: bool,
        term: &mut Vec<u8>,
        term_exists: &mut bool,
    ) -> io::Result<SeekStatus> {
        debug_assert!(self.next_ent != -1);

        if self.next_ent == self.ent_count as i32 {
            if exact_only {
                self.fill_term(term);
                *term_exists = self.sub_code == 0;
            }
            return Ok(SeekStatus::End);
        }

        while self.next_ent < self.ent_count as i32 {
            self.next_ent += 1;

            let mut suffix_lengths_reader = IndexInput::new(
                "suffix_lengths",
                &self.data.suffix_length_bytes[self.data.suffix_length_pos..],
            );
            let code = suffix_lengths_reader.read_vint()?;
            self.suffix_length = (code as u32 >> 1) as usize;

            self.start_byte_pos = self.data.suffix_pos;
            self.data.suffix_pos += self.suffix_length;
            *term_exists = (code & 1) == 0;
            if *term_exists {
                self.state.term_block_ord += 1;
                self.sub_code = 0;
            } else {
                self.sub_code = suffix_lengths_reader.read_vlong()?;
                self.last_sub_fp = self.pos.fp - self.sub_code;
            }
            self.data.suffix_length_pos += suffix_lengths_reader.position();

            let cmp = self.data.suffix_bytes
                [self.start_byte_pos..self.start_byte_pos + self.suffix_length]
                .cmp(&target[self.prefix_length..]);

            if cmp.is_lt() {
                // Current entry is still before the target; keep scanning
            } else if cmp.is_gt() {
                self.fill_term(term);

                if !exact_only && !*term_exists {
                    // For seek_ceil: we are on a sub-block and caller wants
                    // the next term — return NotFound and let the caller
                    // push into sub-frames. The frame's last_sub_fp is set.
                    // The caller (SegmentTermsEnum) handles the push.
                }

                return Ok(SeekStatus::NotFound);
            } else {
                debug_assert!(*term_exists);
                self.fill_term(term);
                return Ok(SeekStatus::Found);
            }
        }

        if exact_only {
            self.fill_term(term);
        }

        Ok(SeekStatus::End)
    }

    /// Rewinds this frame to its original state, forcing a reload on next access.
    pub fn rewind(&mut self) -> io::Result<()> {
        self.pos.fp = self.pos.fp_orig;
        self.next_ent = -1;
        self.flags.has_terms = self.flags.has_terms_orig;
        if self.flags.is_floor {
            self.floor.rewind()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::io;

    use super::*;
    use crate::codecs::competitive_impact::BufferedNormsLookup;
    use crate::codecs::lucene103::blocktree_reader::BlockTreeTermsReader;
    use crate::codecs::lucene103::blocktree_writer::{BlockTreeTermsWriter, BufferedFieldTerms};
    use crate::document::{DocValuesType, IndexOptions, TermOffset};
    use crate::index::pipeline::terms_hash::{FreqProxTermsWriterPerField, TermsHash};
    use crate::index::{FieldInfo, FieldInfos, PointDimensionConfig};
    use crate::store::memory::MemoryDirectory;
    use crate::store::{Directory, SharedDirectory};
    use crate::util::byte_block_pool::ByteBlockPool;
    use assertables::*;

    fn make_field_info(name: &str, number: u32, index_options: IndexOptions) -> FieldInfo {
        FieldInfo::new(
            name.to_string(),
            number,
            false,
            false,
            index_options,
            DocValuesType::None,
            PointDimensionConfig::default(),
        )
    }

    struct TestTerms {
        writer: FreqProxTermsWriterPerField,
        term_pool: ByteBlockPool,
        terms_hash: TermsHash,
    }

    impl TestTerms {
        fn new(field_name: &str, index_options: IndexOptions) -> Self {
            let term_pool = ByteBlockPool::new(32 * 1024);
            Self {
                writer: FreqProxTermsWriterPerField::new(field_name.to_string(), index_options),
                term_pool,
                terms_hash: TermsHash::new(),
            }
        }

        fn add(&mut self, term: &str, doc_id: i32) {
            self.writer.current_position = 0;
            self.writer.current_offset = TermOffset::default();
            self.writer
                .add(
                    &mut self.term_pool,
                    &mut self.terms_hash,
                    term.as_bytes(),
                    doc_id,
                )
                .unwrap();
        }

        fn finalize(&mut self) {
            self.writer.flush_pending_docs(&mut self.terms_hash);
            self.writer.sort_terms(&self.term_pool);
        }
    }

    fn write_terms(
        terms: &[(&str, &[i32])],
        index_options: IndexOptions,
    ) -> io::Result<(SharedDirectory, FieldInfos, [u8; 16])> {
        let field_infos = FieldInfos::new(vec![make_field_info("f", 0, index_options)]);
        let segment_id = [0u8; 16];
        let shared_dir = MemoryDirectory::create();

        {
            let mut writer =
                BlockTreeTermsWriter::new(&shared_dir, "_0", "", &segment_id, index_options)?;

            let mut tt = TestTerms::new("f", index_options);
            // Add in doc-major order
            let max_doc = terms
                .iter()
                .flat_map(|(_, docs)| docs.iter())
                .copied()
                .max()
                .unwrap_or(-1);
            for doc_id in 0..=max_doc {
                for (term, doc_ids) in terms {
                    if doc_ids.contains(&doc_id) {
                        tt.add(term, doc_id);
                    }
                }
            }
            tt.finalize();

            let field_terms =
                BufferedFieldTerms::new(&tt.writer, &tt.term_pool, &tt.terms_hash, "f", 0);
            let norms = BufferedNormsLookup::no_norms();
            writer.write_field(&field_terms, &norms)?;
            writer.finish()?;
        }

        Ok((shared_dir, field_infos, segment_id))
    }

    fn open_reader(
        dir: &dyn Directory,
        field_infos: &FieldInfos,
        segment_id: &[u8; 16],
    ) -> BlockTreeTermsReader {
        BlockTreeTermsReader::open(dir, "_0", "", segment_id, field_infos).unwrap()
    }

    // --- FloorState tests ---

    // Floor data: VInt(num_follow_blocks) then byte(next_label).
    // VInt 3 = [0x03], label 'a' = [0x61].
    fn sample_floor_data() -> Vec<u8> {
        vec![0x03, 0x61]
    }

    #[test]
    fn test_floor_state_set() {
        let mut floor = FloorState::default();
        floor.set(&sample_floor_data()).unwrap();

        assert_eq!(floor.num_follow_blocks, 3);
        assert_eq!(floor.next_label, 0x61);
        assert_eq!(floor.rewind_pos, 0);
        assert_gt!(floor.data_pos, 0);
    }

    #[test]
    fn test_floor_state_rewind() {
        let mut floor = FloorState::default();
        floor.set(&sample_floor_data()).unwrap();

        // Mutate state to simulate iteration
        floor.num_follow_blocks = 0;
        floor.next_label = 0xFF;

        floor.rewind().unwrap();
        assert_eq!(floor.num_follow_blocks, 3);
        assert_eq!(floor.next_label, 0x61);
    }

    #[test]
    fn test_frame_new_defaults() {
        let frame = SegmentTermsEnumFrame::new(5);

        assert_eq!(frame.ord, 5);
        assert_eq!(frame.next_ent, -1);
        assert_eq!(frame.ent_count, 0);
        assert_eq!(frame.pos.fp, 0);
        assert_eq!(frame.pos.fp_orig, 0);
        assert_eq!(frame.pos.fp_end, 0);
        assert_eq!(frame.last_sub_fp, -1);
        assert_none!(frame.node);
    }

    #[test]
    fn test_get_term_block_ord_leaf() {
        let mut frame = SegmentTermsEnumFrame::new(0);
        frame.flags.is_leaf_block = true;
        frame.next_ent = 7;
        frame.state.term_block_ord = 99;

        assert_eq!(frame.get_term_block_ord(), 7);
    }

    #[test]
    fn test_get_term_block_ord_non_leaf() {
        let mut frame = SegmentTermsEnumFrame::new(0);
        frame.flags.is_leaf_block = false;
        frame.next_ent = 7;
        frame.state.term_block_ord = 3;

        assert_eq!(frame.get_term_block_ord(), 3);
    }

    #[test]
    fn test_frame_rewind() {
        let mut frame = SegmentTermsEnumFrame::new(0);
        frame.pos.fp_orig = 42;
        frame.pos.fp = 100;
        frame.next_ent = 5;
        frame.flags.has_terms_orig = true;
        frame.flags.has_terms = false;
        frame.flags.is_floor = false;

        frame.rewind().unwrap();

        assert_eq!(frame.pos.fp, 42);
        assert_eq!(frame.next_ent, -1);
        assert!(frame.flags.has_terms);
    }

    #[test]
    fn test_frame_rewind_with_floor() {
        let mut frame = SegmentTermsEnumFrame::new(0);
        frame.pos.fp_orig = 42;
        frame.pos.fp = 100;
        frame.flags.is_floor = true;
        frame.flags.has_terms_orig = true;
        frame.floor.set(&sample_floor_data()).unwrap();

        // Mutate floor state
        frame.floor.num_follow_blocks = 0;
        frame.floor.next_label = 0xFF;

        frame.rewind().unwrap();

        assert_eq!(frame.pos.fp, 42);
        assert_eq!(frame.floor.num_follow_blocks, 3);
        assert_eq!(frame.floor.next_label, 0x61);
    }

    // --- load_block tests ---

    #[test]
    fn test_load_block_small() {
        let terms = vec![("alpha", &[0][..]), ("beta", &[1]), ("gamma", &[2])];
        let (dir, fi, id) = write_terms(&terms, IndexOptions::Docs).unwrap();

        let reader = open_reader(&dir, &fi, &id);
        let fr = reader.field_reader(0).unwrap();

        let terms_bytes = reader.terms_bytes();
        let mut terms_in = IndexInput::unnamed(terms_bytes);
        let trie = fr.new_trie_reader().unwrap();
        let trie_result = trie.seek_to_block(b"alpha").unwrap().unwrap();

        let mut frame = SegmentTermsEnumFrame::new(0);
        frame.pos.fp = trie_result.output_fp;
        frame.load_block(&mut terms_in).unwrap();

        assert_eq!(frame.ent_count, 3);
        assert!(frame.flags.is_leaf_block);
        assert_eq!(frame.next_ent, 0);
        assert!(frame.pos.fp_end > frame.pos.fp);
    }

    #[test]
    fn test_load_block_many_terms() {
        let mut terms_data: Vec<(String, Vec<i32>)> = Vec::new();
        for i in 0..100 {
            terms_data.push((format!("term_{i:04}"), vec![i]));
        }
        let terms: Vec<(&str, &[i32])> = terms_data
            .iter()
            .map(|(t, d)| (t.as_str(), d.as_slice()))
            .collect();
        let (dir, fi, id) = write_terms(&terms, IndexOptions::Docs).unwrap();

        let reader = open_reader(&dir, &fi, &id);
        let fr = reader.field_reader(0).unwrap();

        let terms_bytes = reader.terms_bytes();
        let mut terms_in = IndexInput::unnamed(terms_bytes);
        let trie = fr.new_trie_reader().unwrap();
        let trie_result = trie.seek_to_block(b"term_0000").unwrap().unwrap();

        let mut frame = SegmentTermsEnumFrame::new(0);
        frame.pos.fp = trie_result.output_fp;
        frame.load_block(&mut terms_in).unwrap();

        assert!(frame.ent_count > 0);
        assert_eq!(frame.next_ent, 0);
        assert!(frame.pos.fp_end > frame.pos.fp);
    }

    #[test]
    fn test_load_block_idempotent() {
        let terms = vec![("hello", &[0][..]), ("world", &[1])];
        let (dir, fi, id) = write_terms(&terms, IndexOptions::Docs).unwrap();

        let reader = open_reader(&dir, &fi, &id);
        let fr = reader.field_reader(0).unwrap();

        let terms_bytes = reader.terms_bytes();
        let mut terms_in = IndexInput::unnamed(terms_bytes);
        let trie = fr.new_trie_reader().unwrap();
        let trie_result = trie.seek_to_block(b"hello").unwrap().unwrap();

        let mut frame = SegmentTermsEnumFrame::new(0);
        frame.pos.fp = trie_result.output_fp;
        frame.load_block(&mut terms_in).unwrap();

        let ent_count = frame.ent_count;
        let fp_end = frame.pos.fp_end;

        // Loading again should be a no-op (next_ent != -1)
        frame.load_block(&mut terms_in).unwrap();
        assert_eq!(frame.ent_count, ent_count);
        assert_eq!(frame.pos.fp_end, fp_end);
    }

    // --- next_leaf / next tests ---

    #[test]
    fn test_next_leaf_iterates_all_terms() {
        let terms = vec![("alpha", &[0][..]), ("beta", &[1]), ("gamma", &[2])];
        let (dir, fi, id) = write_terms(&terms, IndexOptions::Docs).unwrap();

        let reader = open_reader(&dir, &fi, &id);
        let fr = reader.field_reader(0).unwrap();

        let terms_bytes = reader.terms_bytes();
        let mut terms_in = IndexInput::unnamed(terms_bytes);
        let trie = fr.new_trie_reader().unwrap();
        let trie_result = trie.seek_to_block(b"alpha").unwrap().unwrap();

        let mut frame = SegmentTermsEnumFrame::new(0);
        frame.pos.fp = trie_result.output_fp;
        frame.load_block(&mut terms_in).unwrap();
        assert!(frame.flags.is_leaf_block);

        let mut term = Vec::new();
        let mut term_exists = false;

        frame.next_leaf(&mut term, &mut term_exists).unwrap();
        assert_eq!(term, b"alpha");
        assert!(term_exists);

        frame.next_leaf(&mut term, &mut term_exists).unwrap();
        assert_eq!(term, b"beta");

        frame.next_leaf(&mut term, &mut term_exists).unwrap();
        assert_eq!(term, b"gamma");
    }

    #[test]
    fn test_next_leaf_lexicographic_order() {
        let terms = vec![
            ("aaa", &[0][..]),
            ("aab", &[1]),
            ("bbb", &[2]),
            ("zzz", &[3]),
        ];
        let (dir, fi, id) = write_terms(&terms, IndexOptions::Docs).unwrap();

        let reader = open_reader(&dir, &fi, &id);
        let fr = reader.field_reader(0).unwrap();

        let terms_bytes = reader.terms_bytes();
        let mut terms_in = IndexInput::unnamed(terms_bytes);
        let trie = fr.new_trie_reader().unwrap();
        let trie_result = trie.seek_to_block(b"aaa").unwrap().unwrap();

        let mut frame = SegmentTermsEnumFrame::new(0);
        frame.pos.fp = trie_result.output_fp;
        frame.load_block(&mut terms_in).unwrap();

        let mut term = Vec::new();
        let mut term_exists = false;
        let mut collected = Vec::new();

        for _ in 0..frame.ent_count {
            frame.next_leaf(&mut term, &mut term_exists).unwrap();
            collected.push(term.clone());
        }

        // Verify lexicographic order
        for i in 1..collected.len() {
            assert_lt!(collected[i - 1], collected[i]);
        }
    }

    #[test]
    fn test_next_dispatches_to_leaf() {
        let terms = vec![("hello", &[0][..]), ("world", &[1])];
        let (dir, fi, id) = write_terms(&terms, IndexOptions::Docs).unwrap();

        let reader = open_reader(&dir, &fi, &id);
        let fr = reader.field_reader(0).unwrap();

        let terms_bytes = reader.terms_bytes();
        let mut terms_in = IndexInput::unnamed(terms_bytes);
        let trie = fr.new_trie_reader().unwrap();
        let trie_result = trie.seek_to_block(b"hello").unwrap().unwrap();

        let mut frame = SegmentTermsEnumFrame::new(0);
        frame.pos.fp = trie_result.output_fp;
        frame.load_block(&mut terms_in).unwrap();
        assert!(frame.flags.is_leaf_block);

        let mut term = Vec::new();
        let mut term_exists = false;

        // next() on a leaf block should return false (not a sub-block)
        let is_sub_block = frame
            .next(&mut term, &mut term_exists, &mut terms_in)
            .unwrap();
        assert!(!is_sub_block);
        assert_eq!(term, b"hello");
        assert!(term_exists);
    }

    // --- decode_meta_data tests ---

    #[test]
    fn test_decode_meta_data_doc_freq() {
        let terms = vec![
            ("alpha", &[0, 1][..]),
            ("beta", &[2]),
            ("gamma", &[3, 4, 5]),
        ];
        let (dir, fi, id) = write_terms(&terms, IndexOptions::Docs).unwrap();

        let reader = open_reader(&dir, &fi, &id);
        let fr = reader.field_reader(0).unwrap();

        let terms_bytes = reader.terms_bytes();
        let mut terms_in = IndexInput::unnamed(terms_bytes);
        let trie = fr.new_trie_reader().unwrap();
        let trie_result = trie.seek_to_block(b"alpha").unwrap().unwrap();

        let mut frame = SegmentTermsEnumFrame::new(0);
        frame.pos.fp = trie_result.output_fp;
        frame.load_block(&mut terms_in).unwrap();

        let mut term = Vec::new();
        let mut term_exists = false;

        // Advance to first term
        frame.next_leaf(&mut term, &mut term_exists).unwrap();
        assert_eq!(term, b"alpha");
        frame.decode_meta_data(IndexOptions::Docs).unwrap();
        assert_eq!(frame.state.doc_freq, 2);

        // Advance to second term
        frame.next_leaf(&mut term, &mut term_exists).unwrap();
        assert_eq!(term, b"beta");
        frame.decode_meta_data(IndexOptions::Docs).unwrap();
        assert_eq!(frame.state.doc_freq, 1);

        // Advance to third term
        frame.next_leaf(&mut term, &mut term_exists).unwrap();
        assert_eq!(term, b"gamma");
        frame.decode_meta_data(IndexOptions::Docs).unwrap();
        assert_eq!(frame.state.doc_freq, 3);
    }

    #[test]
    fn test_decode_meta_data_total_term_freq() {
        let terms = vec![("hello", &[0, 1][..]), ("world", &[0])];
        let (dir, fi, id) = write_terms(&terms, IndexOptions::DocsAndFreqs).unwrap();

        let reader = open_reader(&dir, &fi, &id);
        let fr = reader.field_reader(0).unwrap();

        let terms_bytes = reader.terms_bytes();
        let mut terms_in = IndexInput::unnamed(terms_bytes);
        let trie = fr.new_trie_reader().unwrap();
        let trie_result = trie.seek_to_block(b"hello").unwrap().unwrap();

        let mut frame = SegmentTermsEnumFrame::new(0);
        frame.pos.fp = trie_result.output_fp;
        frame.load_block(&mut terms_in).unwrap();

        let mut term = Vec::new();
        let mut term_exists = false;

        frame.next_leaf(&mut term, &mut term_exists).unwrap();
        assert_eq!(term, b"hello");
        frame.decode_meta_data(IndexOptions::DocsAndFreqs).unwrap();
        assert_eq!(frame.state.doc_freq, 2);
        assert_ge!(frame.state.total_term_freq, 2);
    }

    // --- scan_to_floor_frame tests ---

    #[test]
    fn test_scan_to_floor_frame_skip_when_not_floor() {
        let mut frame = SegmentTermsEnumFrame::new(0);
        frame.flags.is_floor = false;
        frame.prefix_length = 0;

        // Should return immediately without touching anything
        let fp_before = frame.pos.fp;
        frame.scan_to_floor_frame(b"anything");
        assert_eq!(frame.pos.fp, fp_before);
    }

    #[test]
    fn test_scan_to_floor_frame_skip_when_target_short() {
        let mut frame = SegmentTermsEnumFrame::new(0);
        frame.flags.is_floor = true;
        frame.prefix_length = 5;

        // Target shorter than prefix — should skip
        let fp_before = frame.pos.fp;
        frame.scan_to_floor_frame(b"abc");
        assert_eq!(frame.pos.fp, fp_before);
    }

    #[test]
    fn test_scan_to_floor_frame_skip_when_already_correct() {
        let mut frame = SegmentTermsEnumFrame::new(0);
        frame.flags.is_floor = true;
        frame.prefix_length = 0;
        frame.floor.next_label = 0x7A; // 'z'

        // Target label 'a' < next_floor_label 'z' — already on correct block
        let fp_before = frame.pos.fp;
        frame.scan_to_floor_frame(b"abc");
        assert_eq!(frame.pos.fp, fp_before);
    }

    // --- scan_to_sub_block tests ---

    #[test]
    fn test_scan_to_sub_block_already_positioned() {
        let mut frame = SegmentTermsEnumFrame::new(0);
        frame.flags.is_leaf_block = false;
        frame.last_sub_fp = 42;

        // Already positioned at the target — should return immediately
        frame.scan_to_sub_block(42);
        assert_eq!(frame.last_sub_fp, 42);
    }
}
