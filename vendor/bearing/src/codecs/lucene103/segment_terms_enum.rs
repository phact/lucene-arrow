// SPDX-License-Identifier: Apache-2.0

//! Term block parser and stateful iterator for the Lucene 103 block tree
//! terms dictionary.
//!
//! [`SegmentTermsEnum`] implements [`TermsEnum`] by navigating the trie in the
//! `.tip` file and scanning term blocks in the `.tim` file.
//!
//! Internal helpers parse `.tim` term blocks found by
//! [`super::trie_reader::TrieReader`] and scan within them to find exact terms
//! and decode their metadata ([`IntBlockTermState`]).

use std::cmp::Ordering;
use std::io;

use crate::codecs::lucene103::postings_format::IntBlockTermState;
use crate::codecs::lucene103::segment_terms_enum_frame::SegmentTermsEnumFrame;
use crate::codecs::lucene103::trie_reader::{Node, TrieReader};
use crate::document::IndexOptions;
use crate::encoding::{lowercase_ascii, lz4, pfor, zigzag};
use crate::index::terms::{SeekStatus, TermsEnum};
use crate::store::IndexInput;

pub(crate) const COMPRESSION_NONE: u32 = 0;
const COMPRESSION_LOWERCASE_ASCII: u32 = 1;
const COMPRESSION_LZ4: u32 = 2;

/// Stateful iterator over terms in a single field of a single segment.
///
/// Wraps trie navigation + block scanning into a [`TermsEnum`] implementation.
/// Created by `FieldReader::iterator()` (via the [`Terms`](crate::index::terms::Terms) trait).
pub struct SegmentTermsEnum<'a> {
    /// Borrowed view of the `.tim` terms dictionary.
    terms_in: IndexInput<'a>,
    /// Borrowed view of the `.tip` field region (for floor data reads).
    index_in: IndexInput<'a>,
    /// Trie navigator for this field.
    trie: TrieReader<'a>,
    /// Index options for this field.
    index_options: IndexOptions,
    /// The shared term buffer, written into by frame iteration methods.
    term: Vec<u8>,
    /// Whether the current position is on a real term (vs a sub-block entry).
    term_exists: bool,
    /// Frame stack — one frame per tree depth.
    stack: Vec<SegmentTermsEnumFrame>,
    /// Static frame used for seek-by-state positioning.
    static_frame: SegmentTermsEnumFrame,
    /// Ordinal of the current frame in the stack (-1 = static_frame).
    current_frame_ord: i32,
    /// Trie nodes parallel to the frame stack.
    nodes: Vec<Node>,
    /// How much of the current term was validated against the trie index.
    valid_index_prefix: usize,
    /// The `currentFrame.ord` value saved before `next()` processes.
    target_before_current_length: i32,
    /// Whether `terms_in` has been initialized for frame-based access.
    initialized: bool,
    /// Whether iteration has reached the end.
    eof: bool,
}

impl<'a> SegmentTermsEnum<'a> {
    /// Creates a new `SegmentTermsEnum` for a field.
    pub(crate) fn new(
        terms_in: IndexInput<'a>,
        index_in: IndexInput<'a>,
        trie: TrieReader<'a>,
        index_options: IndexOptions,
    ) -> Self {
        let root_node = trie.root().clone();
        Self {
            terms_in,
            index_in,
            trie,
            index_options,
            term: Vec::new(),
            term_exists: false,
            stack: Vec::new(),
            static_frame: SegmentTermsEnumFrame::new(-1),
            current_frame_ord: -1,
            nodes: vec![root_node],
            valid_index_prefix: 0,
            target_before_current_length: 0,
            initialized: false,
            eof: false,
        }
    }

    /// Returns (or grows) the frame at the given stack ordinal.
    fn get_frame(&mut self, ord: usize) -> &mut SegmentTermsEnumFrame {
        while ord >= self.stack.len() {
            self.stack
                .push(SegmentTermsEnumFrame::new(self.stack.len() as i32));
        }
        debug_assert!(self.stack[ord].ord == ord as i32);
        &mut self.stack[ord]
    }

    /// Returns (or grows) the trie node at the given ordinal.
    fn get_node(&mut self, ord: usize) -> &mut Node {
        while ord >= self.nodes.len() {
            self.nodes.push(Node::new());
        }
        &mut self.nodes[ord]
    }

    /// Pushes a frame for a trie node with output. Sets up floor data if needed.
    fn push_frame_for_node(&mut self, node: &Node, length: usize) -> io::Result<()> {
        let new_ord = (self.current_frame_ord + 1) as usize;
        self.get_frame(new_ord);
        let f = &mut self.stack[new_ord];
        f.flags.has_terms = node.has_terms();
        f.flags.has_terms_orig = f.flags.has_terms;
        f.flags.is_floor = node.is_floor();
        if f.flags.is_floor {
            f.floor
                .load_from_input(&mut self.index_in, node.floor_data_fp())?;
        }
        self.push_frame_fp(Some(node), node.output_fp(), length)?;
        Ok(())
    }

    /// Pushes a frame at the given file pointer. Reuses the frame if already
    /// loaded at the same position.
    fn push_frame_fp(&mut self, node: Option<&Node>, fp: i64, length: usize) -> io::Result<()> {
        let new_ord = (self.current_frame_ord + 1) as usize;
        self.get_frame(new_ord);
        let f = &mut self.stack[new_ord];
        f.node = node.cloned();
        if f.pos.fp_orig == fp && f.next_ent != -1 {
            if f.ord > self.target_before_current_length {
                f.rewind()?;
            }
            debug_assert!(length == f.prefix_length);
        } else {
            f.next_ent = -1;
            f.prefix_length = length;
            f.state.term_block_ord = 0;
            f.pos.fp_orig = fp;
            f.pos.fp = fp;
            f.last_sub_fp = -1;
        }
        self.current_frame_ord = new_ord as i32;
        Ok(())
    }

    /// Returns a mutable reference to the current frame.
    #[expect(dead_code)]
    fn current_frame(&mut self) -> &mut SegmentTermsEnumFrame {
        if self.current_frame_ord == -1 {
            &mut self.static_frame
        } else {
            &mut self.stack[self.current_frame_ord as usize]
        }
    }

    /// Frame-based seek used by `seek_ceil` and as catch-up when `next()`
    /// follows a flat `seek_exact`.
    ///
    /// Reuses seek state when possible: compares the target against the
    /// current term to find their common prefix, then walks the trie index
    /// only for the remaining bytes.
    fn seek_exact_frame_based(&mut self, target: &[u8]) -> io::Result<bool> {
        self.eof = false;
        self.initialized = true;

        let mut target_upto;

        self.target_before_current_length = if self.current_frame_ord == -1 {
            -1
        } else {
            self.stack[self.current_frame_ord as usize].ord
        };

        if self.current_frame_ord != -1 {
            // We are already seek'd; find the common prefix of new seek term
            // vs current term and re-use the corresponding seek state.
            let node = self.nodes[0].clone();
            debug_assert!(node.has_output());
            target_upto = 0;

            let mut last_frame_ord = 0i32;
            debug_assert!(self.valid_index_prefix <= self.term.len());

            let target_limit = target.len().min(self.valid_index_prefix);

            let mut cmp = 0i32;

            // Compare up to valid seek frames
            while target_upto < target_limit {
                cmp = (self.term[target_upto] as i32) - (target[target_upto] as i32);
                if cmp != 0 {
                    break;
                }
                let next_node = &self.nodes[1 + target_upto];
                debug_assert_eq!(next_node.label(), target[target_upto]);

                if next_node.has_output() {
                    last_frame_ord = self.stack[1 + last_frame_ord as usize].ord;
                }
                target_upto += 1;
            }

            if cmp == 0 {
                // Compare the rest of the term
                cmp = match self.term[target_upto..].cmp(&target[target_upto..]) {
                    Ordering::Less => -1,
                    Ordering::Equal => 0,
                    Ordering::Greater => 1,
                };
            }

            if cmp < 0 {
                // Target is after current term
                self.current_frame_ord = last_frame_ord;
            } else if cmp > 0 {
                // Target is before current term; rewind
                self.target_before_current_length = last_frame_ord;
                self.current_frame_ord = last_frame_ord;
                self.stack[last_frame_ord as usize].rewind()?;
            } else {
                // Target is exactly the same as current term
                debug_assert_eq!(self.term.len(), target.len());
                if self.term_exists {
                    return Ok(true);
                }
            }
        } else {
            self.target_before_current_length = -1;

            let root = self.nodes[0].clone();
            debug_assert!(root.has_output());

            self.current_frame_ord = -1;
            target_upto = 0;
            self.push_frame_for_node(&root, 0)?;
        }

        // Walk the trie index from target_upto
        while target_upto < target.len() {
            let target_label = target[target_upto];

            let node_idx = 1 + target_upto;
            self.get_node(node_idx);

            let parent = self.nodes[target_upto].clone();
            let found = self
                .trie
                .lookup_child(target_label, &parent, &mut self.nodes[node_idx])?;

            if !found {
                // Index is exhausted
                let ord = self.current_frame_ord as usize;
                self.valid_index_prefix = self.stack[ord].prefix_length;

                let target_clone = target.to_vec();
                self.stack[ord].scan_to_floor_frame(&target_clone);

                if !self.stack[ord].flags.has_terms {
                    self.term_exists = false;
                    if self.term.len() <= target_upto {
                        self.term.resize(target_upto + 1, 0);
                    }
                    self.term[target_upto] = target_label;
                    self.term.truncate(1 + target_upto);
                    return Ok(false);
                }

                self.stack[ord].load_block(&mut self.terms_in)?;

                let result = self.stack[ord].scan_to_term(
                    target,
                    true,
                    &mut self.term,
                    &mut self.term_exists,
                )?;
                return Ok(result == SeekStatus::Found);
            } else {
                // Follow this node
                if self.term.len() <= target_upto {
                    self.term.resize(target_upto + 1, 0);
                }
                self.term[target_upto] = target_label;
                target_upto += 1;

                if self.nodes[node_idx].has_output() {
                    let node = self.nodes[node_idx].clone();
                    self.push_frame_for_node(&node, target_upto)?;
                }
            }
        }

        // Target term is entirely contained in the index
        let ord = self.current_frame_ord as usize;
        self.valid_index_prefix = self.stack[ord].prefix_length;

        let target_clone = target.to_vec();
        self.stack[ord].scan_to_floor_frame(&target_clone);

        if !self.stack[ord].flags.has_terms {
            self.term_exists = false;
            self.term.truncate(target_upto);
            return Ok(false);
        }

        self.stack[ord].load_block(&mut self.terms_in)?;

        let result =
            self.stack[ord].scan_to_term(target, true, &mut self.term, &mut self.term_exists)?;
        Ok(result == SeekStatus::Found)
    }
}

impl TermsEnum for SegmentTermsEnum<'_> {
    /// Fast point lookup using the flat (stateless) path.
    ///
    /// Navigates the trie from scratch, parses one block, and discards all
    /// state. Use `seek_exact_frame_based` when frame-stack consistency is
    /// needed (e.g. before `next()`).
    fn seek_exact(&mut self, target: &[u8]) -> io::Result<bool> {
        self.eof = false;
        self.initialized = true;

        let trie_result = match self.trie.seek_to_block(target)? {
            Some(r) => r,
            None => {
                self.current_frame_ord = -1;
                self.term_exists = false;
                return Ok(false);
            }
        };

        let state = seek_exact_in_block(
            &self.terms_in,
            &trie_result,
            target,
            self.index_options,
            &self.index_in,
        )?;

        match state {
            Some(s) => {
                self.term.clear();
                self.term.extend_from_slice(target);
                self.static_frame.state = s;
                self.static_frame.meta_data_upto = self.static_frame.get_term_block_ord();
                self.current_frame_ord = -1;
                self.term_exists = true;
                Ok(true)
            }
            None => {
                self.current_frame_ord = -1;
                self.term_exists = false;
                Ok(false)
            }
        }
    }

    fn seek_ceil(&mut self, target: &[u8]) -> io::Result<SeekStatus> {
        self.eof = false;
        self.initialized = true;

        let mut target_upto;

        self.target_before_current_length = if self.current_frame_ord == -1 {
            -1
        } else {
            self.stack[self.current_frame_ord as usize].ord
        };

        if self.current_frame_ord != -1 {
            // Reuse seek state from current position.
            let node = self.nodes[0].clone();
            debug_assert!(node.has_output());
            target_upto = 0;

            let mut last_frame_ord = 0i32;
            debug_assert!(self.valid_index_prefix <= self.term.len());

            let target_limit = target.len().min(self.valid_index_prefix);

            let mut cmp = 0i32;

            while target_upto < target_limit {
                cmp = (self.term[target_upto] as i32) - (target[target_upto] as i32);
                if cmp != 0 {
                    break;
                }
                let next_node = &self.nodes[1 + target_upto];
                debug_assert_eq!(next_node.label(), target[target_upto]);

                if next_node.has_output() {
                    last_frame_ord = self.stack[1 + last_frame_ord as usize].ord;
                }
                target_upto += 1;
            }

            if cmp == 0 {
                cmp = match self.term[target_upto..].cmp(&target[target_upto..]) {
                    Ordering::Less => -1,
                    Ordering::Equal => 0,
                    Ordering::Greater => 1,
                };
            }

            if cmp < 0 {
                self.current_frame_ord = last_frame_ord;
            } else if cmp > 0 {
                self.target_before_current_length = 0;
                self.current_frame_ord = last_frame_ord;
                self.stack[last_frame_ord as usize].rewind()?;
            } else {
                debug_assert_eq!(self.term.len(), target.len());
                if self.term_exists {
                    return Ok(SeekStatus::Found);
                }
            }
        } else {
            self.target_before_current_length = -1;

            let root = self.nodes[0].clone();
            debug_assert!(root.has_output());

            self.current_frame_ord = -1;
            target_upto = 0;
            self.push_frame_for_node(&root, 0)?;
        }

        // Walk the trie index
        while target_upto < target.len() {
            let target_label = target[target_upto];

            let node_idx = 1 + target_upto;
            self.get_node(node_idx);

            let parent = self.nodes[target_upto].clone();
            let found = self
                .trie
                .lookup_child(target_label, &parent, &mut self.nodes[node_idx])?;

            if !found {
                let ord = self.current_frame_ord as usize;
                self.valid_index_prefix = self.stack[ord].prefix_length;

                let target_clone = target.to_vec();
                self.stack[ord].scan_to_floor_frame(&target_clone);

                self.stack[ord].load_block(&mut self.terms_in)?;

                let result = self.stack[ord].scan_to_term(
                    target,
                    false,
                    &mut self.term,
                    &mut self.term_exists,
                )?;
                if result == SeekStatus::End {
                    self.term.clear();
                    self.term.extend_from_slice(target);
                    self.term_exists = false;
                    if self.next()?.is_some() {
                        return Ok(SeekStatus::NotFound);
                    } else {
                        return Ok(SeekStatus::End);
                    }
                }
                if result == SeekStatus::NotFound && !self.term_exists {
                    // On a sub-block entry past the target; push into sub-frames
                    // to find the first term.
                    let ord = self.current_frame_ord as usize;
                    let last_sub_fp = self.stack[ord].last_sub_fp;
                    let prefix_len = self.stack[ord].prefix_length + self.stack[ord].suffix_length;
                    self.push_frame_fp(None, last_sub_fp, prefix_len)?;
                    self.stack[self.current_frame_ord as usize].load_block(&mut self.terms_in)?;
                    while self.stack[self.current_frame_ord as usize].next(
                        &mut self.term,
                        &mut self.term_exists,
                        &mut self.terms_in,
                    )? {
                        let sub_fp = self.stack[self.current_frame_ord as usize].last_sub_fp;
                        let term_len = self.term.len();
                        self.push_frame_fp(None, sub_fp, term_len)?;
                        self.stack[self.current_frame_ord as usize]
                            .load_block(&mut self.terms_in)?;
                    }
                }
                return Ok(result);
            } else {
                if self.term.len() <= target_upto {
                    self.term.resize(target_upto + 1, 0);
                }
                self.term[target_upto] = target_label;
                target_upto += 1;

                if self.nodes[node_idx].has_output() {
                    let node = self.nodes[node_idx].clone();
                    self.push_frame_for_node(&node, target_upto)?;
                }
            }
        }

        // Target entirely in index
        let ord = self.current_frame_ord as usize;
        self.valid_index_prefix = self.stack[ord].prefix_length;

        let target_clone = target.to_vec();
        self.stack[ord].scan_to_floor_frame(&target_clone);

        self.stack[ord].load_block(&mut self.terms_in)?;

        let result =
            self.stack[ord].scan_to_term(target, false, &mut self.term, &mut self.term_exists)?;

        if result == SeekStatus::End {
            self.term.clear();
            self.term.extend_from_slice(target);
            self.term_exists = false;
            if self.next()?.is_some() {
                return Ok(SeekStatus::NotFound);
            } else {
                return Ok(SeekStatus::End);
            }
        }
        if result == SeekStatus::NotFound && !self.term_exists {
            let ord = self.current_frame_ord as usize;
            let last_sub_fp = self.stack[ord].last_sub_fp;
            let prefix_len = self.stack[ord].prefix_length + self.stack[ord].suffix_length;
            self.push_frame_fp(None, last_sub_fp, prefix_len)?;
            self.stack[self.current_frame_ord as usize].load_block(&mut self.terms_in)?;
            while self.stack[self.current_frame_ord as usize].next(
                &mut self.term,
                &mut self.term_exists,
                &mut self.terms_in,
            )? {
                let sub_fp = self.stack[self.current_frame_ord as usize].last_sub_fp;
                let term_len = self.term.len();
                self.push_frame_fp(None, sub_fp, term_len)?;
                self.stack[self.current_frame_ord as usize].load_block(&mut self.terms_in)?;
            }
        }
        Ok(result)
    }

    fn seek_exact_with_state(&mut self, term: &[u8], state: IntBlockTermState) {
        self.term.clear();
        self.term.extend_from_slice(term);
        self.current_frame_ord = -1;
        self.static_frame.state = state;
        self.static_frame.meta_data_upto = self.static_frame.get_term_block_ord();
        self.term_exists = true;
    }

    fn term(&self) -> &[u8] {
        &self.term
    }

    fn doc_freq(&mut self) -> io::Result<i32> {
        if !self.term_exists {
            return Err(io::Error::other("TermsEnum not positioned"));
        }
        if self.current_frame_ord == -1 {
            Ok(self.static_frame.state.doc_freq)
        } else {
            let f = &mut self.stack[self.current_frame_ord as usize];
            f.decode_meta_data(self.index_options)?;
            Ok(f.state.doc_freq)
        }
    }

    fn total_term_freq(&mut self) -> io::Result<i64> {
        if !self.term_exists {
            return Err(io::Error::other("TermsEnum not positioned"));
        }
        if self.current_frame_ord == -1 {
            Ok(self.static_frame.state.total_term_freq)
        } else {
            let f = &mut self.stack[self.current_frame_ord as usize];
            f.decode_meta_data(self.index_options)?;
            Ok(f.state.total_term_freq)
        }
    }

    fn term_state(&mut self) -> io::Result<IntBlockTermState> {
        if !self.term_exists {
            return Err(io::Error::other("TermsEnum not positioned"));
        }
        if self.current_frame_ord == -1 {
            Ok(self.static_frame.state)
        } else {
            let f = &mut self.stack[self.current_frame_ord as usize];
            f.decode_meta_data(self.index_options)?;
            Ok(f.state)
        }
    }

    fn next(&mut self) -> io::Result<Option<&[u8]>> {
        if self.eof {
            return Ok(None);
        }

        if !self.initialized {
            // First call — push root frame
            let root = self.nodes[0].clone();
            self.push_frame_for_node(&root, 0)?;
            self.stack[self.current_frame_ord as usize].load_block(&mut self.terms_in)?;
            self.initialized = true;
        }

        self.target_before_current_length = self.current_frame_ord;

        if self.current_frame_ord == -1 {
            // Positioned via flat seek_exact or seek_exact_with_state —
            // re-seek with frame-based path to set up frame stack for iteration.
            let result = self.seek_exact_frame_based(&self.term.clone())?;
            debug_assert!(result);
        }

        // Pop exhausted frames
        loop {
            let ord = self.current_frame_ord as usize;
            let f = &self.stack[ord];
            if f.next_ent < f.ent_count as i32 {
                break;
            }
            if !f.flags.is_last_in_floor {
                self.stack[ord].load_next_floor_block(&mut self.terms_in)?;
                break;
            } else {
                if self.current_frame_ord == 0 {
                    self.term.clear();
                    self.valid_index_prefix = 0;
                    self.stack[0].rewind()?;
                    self.term_exists = false;
                    self.eof = true;
                    return Ok(None);
                }
                let last_fp = self.stack[ord].pos.fp_orig;
                self.current_frame_ord -= 1;

                let parent_ord = self.current_frame_ord as usize;
                if self.stack[parent_ord].next_ent == -1
                    || self.stack[parent_ord].last_sub_fp != last_fp
                {
                    let term_clone = self.term.clone();
                    self.stack[parent_ord].scan_to_floor_frame(&term_clone);
                    self.stack[parent_ord].load_block(&mut self.terms_in)?;
                    self.stack[parent_ord].scan_to_sub_block(last_fp);
                }

                self.valid_index_prefix = self
                    .valid_index_prefix
                    .min(self.stack[parent_ord].prefix_length);
            }
        }

        // Advance within frame, pushing into sub-blocks as needed
        loop {
            let ord = self.current_frame_ord as usize;
            let is_sub_block = {
                let f = &mut self.stack[ord];
                f.next(&mut self.term, &mut self.term_exists, &mut self.terms_in)?
            };
            if is_sub_block {
                let last_sub_fp = self.stack[ord].last_sub_fp;
                let term_len = self.term.len();
                self.push_frame_fp(None, last_sub_fp, term_len)?;
                self.stack[self.current_frame_ord as usize].load_block(&mut self.terms_in)?;
            } else {
                return Ok(Some(&self.term));
            }
        }
    }
}

/// Seeks to an exact term in the `.tim` file and returns its metadata.
fn seek_exact_in_block(
    terms_in: &IndexInput<'_>,
    trie_result: &super::trie_reader::TrieSeekResult,
    target: &[u8],
    index_options: IndexOptions,
    index_in: &IndexInput<'_>,
) -> io::Result<Option<IntBlockTermState>> {
    let prefix_length = trie_result.depth;
    let mut block_fp = trie_result.output_fp;

    // Handle floor blocks: find the right sub-block for the target's next byte
    if trie_result.floor_data_fp >= 0 && target.len() > prefix_length {
        let target_label = target[prefix_length];
        block_fp = scan_to_floor_block(
            index_in,
            trie_result.floor_data_fp,
            trie_result.output_fp,
            target_label,
        )?;
    }

    let mut input = terms_in.view("seek_exact", 0, terms_in.length())?;
    input.seek(block_fp as usize)?;

    let result = scan_block(&mut input, target, prefix_length, index_options)?;
    match result {
        ScanResult::Found(state) => Ok(Some(state)),
        ScanResult::NotFound => Ok(None),
    }
}

/// Scans floor data from `.tip` to find the block FP for the given target label.
fn scan_to_floor_block(
    index_in: &IndexInput<'_>,
    floor_data_fp: i64,
    base_fp: i64,
    target_label: u8,
) -> io::Result<i64> {
    let mut input = index_in.view("floor_data", 0, index_in.length())?;
    input.seek(floor_data_fp as usize)?;

    let num_follow_blocks = input.read_vint()?;
    let mut result_fp = base_fp;

    for i in 0..num_follow_blocks {
        let floor_lead_byte = input.read_byte()?;
        let code = input.read_vlong()?;
        let fp_delta = code >> 1;
        let fp = base_fp + fp_delta;

        if target_label < floor_lead_byte {
            break;
        }

        result_fp = fp;

        if i == num_follow_blocks - 1 {
            break;
        }
    }

    Ok(result_fp)
}

enum ScanResult {
    Found(IntBlockTermState),
    NotFound,
}

/// Loads and scans a single term block from the current position.
fn scan_block(
    input: &mut IndexInput<'_>,
    target: &[u8],
    prefix_length: usize,
    index_options: IndexOptions,
) -> io::Result<ScanResult> {
    let code = input.read_vint()?;
    let entry_count = (code >> 1) as usize;
    let _is_last_in_floor = (code & 1) != 0;

    let suffix_token = input.read_vlong()?;
    let is_leaf_block = (suffix_token & 0x04) != 0;
    let num_suffix_bytes = (suffix_token >> 3) as usize;
    let compression_code = (suffix_token & 0x03) as u32;

    let suffix_bytes = read_compressed(input, num_suffix_bytes, compression_code)?;

    let suffix_lengths_token = input.read_vint()?;
    let all_equal = (suffix_lengths_token & 1) != 0;
    let num_suffix_length_bytes = (suffix_lengths_token >> 1) as usize;

    let suffix_length_bytes = if all_equal {
        let common = input.read_byte()?;
        vec![common; num_suffix_length_bytes]
    } else {
        let mut buf = vec![0u8; num_suffix_length_bytes];
        input.read_bytes(&mut buf)?;
        buf
    };

    let num_stats_bytes = input.read_vint()? as usize;
    let mut stats_bytes = vec![0u8; num_stats_bytes];
    input.read_bytes(&mut stats_bytes)?;

    let num_meta_bytes = input.read_vint()? as usize;
    let mut meta_bytes = vec![0u8; num_meta_bytes];
    input.read_bytes(&mut meta_bytes)?;

    let target_suffix = &target[prefix_length..];

    let mut suffix_reader = IndexInput::new("suffixes", &suffix_bytes);
    let mut suffix_lengths_reader = IndexInput::new("suffix_lengths", &suffix_length_bytes);

    let mut term_ord = 0usize;

    for _entry_idx in 0..entry_count {
        let (suffix_len, is_sub_block) = if is_leaf_block {
            let len = suffix_lengths_reader.read_vint()? as usize;
            (len, false)
        } else {
            let code = suffix_lengths_reader.read_vint()?;
            let len = (code >> 1) as usize;
            let is_sub = (code & 1) != 0;
            if is_sub {
                suffix_lengths_reader.read_vlong()?;
            }
            (len, is_sub)
        };

        let suffix_start = suffix_reader.position();
        suffix_reader.skip_bytes(suffix_len)?;

        if is_sub_block {
            continue;
        }

        let suffix = &suffix_bytes[suffix_start..suffix_start + suffix_len];
        let cmp = suffix.cmp(target_suffix);

        match cmp {
            Ordering::Equal => {
                let state = decode_term_state(&stats_bytes, &meta_bytes, term_ord, index_options)?;
                return Ok(ScanResult::Found(state));
            }
            Ordering::Greater => {
                return Ok(ScanResult::NotFound);
            }
            Ordering::Less => {
                term_ord += 1;
            }
        }
    }

    Ok(ScanResult::NotFound)
}

/// Decode stats and metadata for the Nth term in a block.
fn decode_term_state(
    stats_bytes: &[u8],
    meta_bytes: &[u8],
    target_ord: usize,
    index_options: IndexOptions,
) -> io::Result<IntBlockTermState> {
    let mut stats_reader = IndexInput::new("stats", stats_bytes);
    let mut meta_reader = IndexInput::new("meta", meta_bytes);
    let has_freqs = index_options.has_freqs();

    let mut state = IntBlockTermState::new();
    let mut last_state = IntBlockTermState::new();
    let mut singleton_run = 0i32;

    for ord in 0..=target_ord {
        if singleton_run > 0 {
            state.doc_freq = 1;
            state.total_term_freq = 1;
            singleton_run -= 1;
        } else {
            let token = stats_reader.read_vint()?;
            if (token & 1) == 1 {
                state.doc_freq = 1;
                state.total_term_freq = 1;
                singleton_run = token >> 1;
            } else {
                state.doc_freq = token >> 1;
                if !has_freqs {
                    state.total_term_freq = state.doc_freq as i64;
                } else {
                    state.total_term_freq = state.doc_freq as i64 + stats_reader.read_vlong()?;
                }
            }
        }

        let empty_state = IntBlockTermState::new();
        let ref_state = if ord == 0 { &empty_state } else { &last_state };
        decode_term_meta(&mut meta_reader, &mut state, ref_state, index_options)?;

        if ord < target_ord {
            last_state = state;
        }
    }

    Ok(state)
}

/// Decode one term's postings metadata from the metadata bytes.
fn decode_term_meta(
    reader: &mut IndexInput<'_>,
    state: &mut IntBlockTermState,
    last_state: &IntBlockTermState,
    index_options: IndexOptions,
) -> io::Result<()> {
    let code = reader.read_vlong()?;
    if (code & 1) != 0 {
        let encoded = code >> 1;
        let delta = zigzag::decode_i64(encoded);
        state.singleton_doc_id = (last_state.singleton_doc_id as i64 + delta) as i32;
        state.doc_start_fp = last_state.doc_start_fp;
    } else {
        let fp_delta = code >> 1;
        state.doc_start_fp = last_state.doc_start_fp + fp_delta;
        if state.doc_freq == 1 {
            state.singleton_doc_id = reader.read_vint()?;
        } else {
            state.singleton_doc_id = -1;
        }
    }

    if index_options.has_positions() {
        state.pos_start_fp = last_state.pos_start_fp + reader.read_vlong()?;
        if index_options.has_offsets() {
            state.pay_start_fp = last_state.pay_start_fp + reader.read_vlong()?;
        }
        if state.total_term_freq > pfor::BLOCK_SIZE as i64 {
            state.last_pos_block_offset = reader.read_vlong()?;
        } else {
            state.last_pos_block_offset = -1;
        }
    }

    Ok(())
}

/// Read and decompress suffix bytes.
pub(crate) fn read_compressed(
    input: &mut IndexInput<'_>,
    uncompressed_len: usize,
    compression_code: u32,
) -> io::Result<Vec<u8>> {
    match compression_code {
        COMPRESSION_NONE => {
            let mut buf = vec![0u8; uncompressed_len];
            input.read_bytes(&mut buf)?;
            Ok(buf)
        }
        COMPRESSION_LZ4 => Ok(lz4::decompress_from_reader(
            input.cursor_mut(),
            uncompressed_len,
        )?),
        COMPRESSION_LOWERCASE_ASCII => Ok(lowercase_ascii::decompress_from_cursor(
            input.cursor_mut(),
            uncompressed_len,
        )?),
        _ => Err(io::Error::other(format!(
            "unknown compression code: {compression_code}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codecs::competitive_impact::BufferedNormsLookup;
    use crate::codecs::lucene103::blocktree_reader::BlockTreeTermsReader;
    use crate::codecs::lucene103::blocktree_writer::{BlockTreeTermsWriter, BufferedFieldTerms};
    use crate::document::{DocValuesType, TermOffset};
    use crate::index::pipeline::terms_hash::{FreqProxTermsWriterPerField, TermsHash};
    use crate::index::terms::Terms;
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

        fn add(&mut self, term: &str, doc_id: i32, position: i32) {
            self.writer.current_position = position;
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

    /// Add terms in doc-major order from term-major test data.
    fn add_terms_doc_major(tt: &mut TestTerms, terms: &[(&str, &[i32])]) {
        let max_doc = terms
            .iter()
            .flat_map(|(_, docs)| docs.iter())
            .copied()
            .max()
            .unwrap_or(-1);
        for doc_id in 0..=max_doc {
            for (term, doc_ids) in terms {
                if doc_ids.contains(&doc_id) {
                    tt.add(term, doc_id, 0);
                }
            }
        }
    }

    /// Write terms and return (directory, field_infos, segment_id).
    fn write_terms(
        terms: Vec<(&str, &[i32])>,
        index_options: IndexOptions,
    ) -> io::Result<(SharedDirectory, FieldInfos, [u8; 16])> {
        let field_infos = FieldInfos::new(vec![make_field_info("f", 0, index_options)]);
        let segment_name = "_0";
        let segment_suffix = "";
        let segment_id = [0u8; 16];

        let shared_dir = MemoryDirectory::create();

        {
            let mut writer = BlockTreeTermsWriter::new(
                &shared_dir,
                segment_name,
                segment_suffix,
                &segment_id,
                index_options,
            )?;

            let mut tt = TestTerms::new("f", index_options);
            add_terms_doc_major(&mut tt, &terms);
            tt.finalize();

            let field_terms =
                BufferedFieldTerms::new(&tt.writer, &tt.term_pool, &tt.terms_hash, "f", 0);
            let norms = BufferedNormsLookup::no_norms();
            writer.write_field(&field_terms, &norms)?;

            writer.finish()?;
        }

        Ok((shared_dir, field_infos, segment_id))
    }

    /// Open the blocktree reader and seek for a term via TermsEnum.
    fn seek_term(
        dir: &dyn Directory,
        field_infos: &FieldInfos,
        segment_id: &[u8; 16],
        term: &[u8],
    ) -> io::Result<Option<IntBlockTermState>> {
        let reader = BlockTreeTermsReader::open(dir, "_0", "", segment_id, field_infos)?;
        let fr = reader.field_reader(0).unwrap();
        let mut terms_enum = fr.iterator()?;
        if terms_enum.seek_exact(term)? {
            Ok(Some(terms_enum.term_state()?))
        } else {
            Ok(None)
        }
    }

    #[test]
    fn test_seek_exact_singleton_term() {
        let terms = vec![("hello", &[5][..]), ("world", &[10])];
        let (dir, fi, id) = write_terms(terms, IndexOptions::Docs).unwrap();

        let state = seek_term(&dir, &fi, &id, b"hello").unwrap().unwrap();
        assert_eq!(state.doc_freq, 1);
        assert_eq!(state.singleton_doc_id, 5);

        let state = seek_term(&dir, &fi, &id, b"world").unwrap().unwrap();
        assert_eq!(state.doc_freq, 1);
        assert_eq!(state.singleton_doc_id, 10);
    }

    #[test]
    fn test_seek_exact_multi_doc_term_small() {
        let terms = vec![("hello", &[5, 6][..]), ("world", &[10, 11])];
        let (dir, fi, id) = write_terms(terms, IndexOptions::Docs).unwrap();

        let state = seek_term(&dir, &fi, &id, b"hello").unwrap().unwrap();
        assert_eq!(state.doc_freq, 2);

        let state = seek_term(&dir, &fi, &id, b"world").unwrap().unwrap();
        assert_eq!(state.doc_freq, 2);
    }

    #[test]
    fn test_seek_exact_multi_doc_term() {
        let terms = vec![
            ("alpha", &[0, 1, 2][..]),
            ("beta", &[1, 3]),
            ("gamma", &[0, 2, 4]),
        ];
        let (dir, fi, id) = write_terms(terms, IndexOptions::Docs).unwrap();

        let state = seek_term(&dir, &fi, &id, b"alpha").unwrap().unwrap();
        assert_eq!(state.doc_freq, 3);
        assert_eq!(state.singleton_doc_id, -1);

        let state = seek_term(&dir, &fi, &id, b"beta").unwrap().unwrap();
        assert_eq!(state.doc_freq, 2);

        let state = seek_term(&dir, &fi, &id, b"gamma").unwrap().unwrap();
        assert_eq!(state.doc_freq, 3);
    }

    #[test]
    fn test_seek_exact_nonexistent_term() {
        let terms = vec![("alpha", &[0][..]), ("gamma", &[1])];
        let (dir, fi, id) = write_terms(terms, IndexOptions::Docs).unwrap();

        let result = seek_term(&dir, &fi, &id, b"beta").unwrap();
        assert_none!(&result);

        let result = seek_term(&dir, &fi, &id, b"zzz").unwrap();
        assert_none!(&result);
    }

    #[test]
    fn test_seek_exact_with_freqs() {
        let terms = vec![("hello", &[0, 1, 2][..]), ("world", &[0])];
        let (dir, fi, id) = write_terms(terms, IndexOptions::DocsAndFreqs).unwrap();

        let state = seek_term(&dir, &fi, &id, b"hello").unwrap().unwrap();
        assert_eq!(state.doc_freq, 3);
        assert_ge!(state.total_term_freq, 3);
    }

    #[test]
    fn test_seek_exact_many_terms() {
        // Generate enough terms to force multi-block trie navigation
        let mut terms_data: Vec<(String, Vec<i32>)> = Vec::new();
        for i in 0..100 {
            terms_data.push((format!("term_{i:04}"), vec![i]));
        }
        let terms: Vec<(&str, &[i32])> = terms_data
            .iter()
            .map(|(t, d)| (t.as_str(), d.as_slice()))
            .collect();

        let (dir, fi, id) = write_terms(terms, IndexOptions::Docs).unwrap();

        // Seek each term
        for i in 0..100 {
            let term = format!("term_{i:04}");
            let state = seek_term(&dir, &fi, &id, term.as_bytes()).unwrap();
            assert_some!(&state);
            assert_eq!(state.unwrap().doc_freq, 1);
        }

        // Nonexistent term
        let result = seek_term(&dir, &fi, &id, b"term_9999").unwrap();
        assert_none!(&result);
    }

    #[test]
    fn test_seek_exact_singleton_rle_run() {
        // Multiple consecutive singletons exercise the RLE decoding path.
        let term_list = [
            "aardvark", "badger", "cat", "dog", "elephant", "fox", "giraffe", "hippo", "iguana",
            "jaguar",
        ];
        let doc_ids: Vec<Vec<i32>> = (0..10).map(|i| vec![i]).collect();
        let terms: Vec<(&str, &[i32])> = term_list
            .iter()
            .zip(doc_ids.iter())
            .map(|(&t, d)| (t, d.as_slice()))
            .collect();

        let (dir, fi, id) = write_terms(terms, IndexOptions::Docs).unwrap();

        // First term in the block
        let state = seek_term(&dir, &fi, &id, b"aardvark").unwrap().unwrap();
        assert_eq!(state.doc_freq, 1);
        assert_eq!(state.singleton_doc_id, 0);

        // Middle term (exercises skipping through singleton RLE)
        let state = seek_term(&dir, &fi, &id, b"fox").unwrap().unwrap();
        assert_eq!(state.doc_freq, 1);
        assert_eq!(state.singleton_doc_id, 5);

        // Last term in the block
        let state = seek_term(&dir, &fi, &id, b"jaguar").unwrap().unwrap();
        assert_eq!(state.doc_freq, 1);
        assert_eq!(state.singleton_doc_id, 9);
    }

    #[test]
    fn test_seek_exact_mixed_singleton_and_multi_doc() {
        // Mix of singleton (df=1) and multi-doc (df>1) terms in the same block.
        let terms = vec![
            ("alpha", &[0][..]),      // singleton
            ("beta", &[0, 1, 2][..]), // multi-doc
            ("gamma", &[3][..]),      // singleton
            ("delta", &[4, 5][..]),   // multi-doc
        ];
        let (dir, fi, id) = write_terms(terms, IndexOptions::Docs).unwrap();

        let state = seek_term(&dir, &fi, &id, b"alpha").unwrap().unwrap();
        assert_eq!(state.doc_freq, 1);
        assert_eq!(state.singleton_doc_id, 0);

        let state = seek_term(&dir, &fi, &id, b"beta").unwrap().unwrap();
        assert_eq!(state.doc_freq, 3);
        assert_eq!(state.singleton_doc_id, -1);

        let state = seek_term(&dir, &fi, &id, b"delta").unwrap().unwrap();
        assert_eq!(state.doc_freq, 2);
        assert_eq!(state.singleton_doc_id, -1);

        let state = seek_term(&dir, &fi, &id, b"gamma").unwrap().unwrap();
        assert_eq!(state.doc_freq, 1);
        assert_eq!(state.singleton_doc_id, 3);
    }

    #[test]
    fn test_seek_exact_variable_length_suffixes() {
        // Terms with different suffix lengths exercise the allEqual=false path
        // in suffix length decoding.
        let terms = vec![
            ("a", &[0][..]),
            ("bb", &[1]),
            ("ccc", &[2]),
            ("dddddddddd", &[3]),
        ];
        let (dir, fi, id) = write_terms(terms, IndexOptions::Docs).unwrap();

        for (term, doc) in [("a", 0), ("bb", 1), ("ccc", 2), ("dddddddddd", 3)] {
            let state = seek_term(&dir, &fi, &id, term.as_bytes()).unwrap().unwrap();
            assert_eq!(state.doc_freq, 1);
            assert_eq!(state.singleton_doc_id, doc);
        }
    }

    #[test]
    fn test_seek_exact_with_positions() {
        let terms = vec![("hello", &[0, 1][..]), ("world", &[0])];
        let (dir, fi, id) = write_terms(terms, IndexOptions::DocsAndFreqsAndPositions).unwrap();

        let state = seek_term(&dir, &fi, &id, b"hello").unwrap().unwrap();
        assert_eq!(state.doc_freq, 2);
        assert_ge!(state.pos_start_fp, 0);

        let state = seek_term(&dir, &fi, &id, b"world").unwrap().unwrap();
        assert_eq!(state.doc_freq, 1);
    }

    #[test]
    fn test_seek_exact_floor_blocks_different_targets() {
        // 100 terms split across floor blocks. Verify terms in different floor
        // sub-blocks are all found.
        let mut terms_data: Vec<(String, Vec<i32>)> = Vec::new();
        for i in 0..100 {
            terms_data.push((format!("term_{i:04}"), vec![i]));
        }
        let terms: Vec<(&str, &[i32])> = terms_data
            .iter()
            .map(|(t, d)| (t.as_str(), d.as_slice()))
            .collect();
        let (dir, fi, id) = write_terms(terms, IndexOptions::Docs).unwrap();

        // First term
        let state = seek_term(&dir, &fi, &id, b"term_0000").unwrap().unwrap();
        assert_eq!(state.doc_freq, 1);

        // Last term
        let state = seek_term(&dir, &fi, &id, b"term_0099").unwrap().unwrap();
        assert_eq!(state.doc_freq, 1);

        // Term in the middle (likely different floor block than first/last)
        let state = seek_term(&dir, &fi, &id, b"term_0050").unwrap().unwrap();
        assert_eq!(state.doc_freq, 1);

        // Before first
        let result = seek_term(&dir, &fi, &id, b"term_").unwrap();
        assert_none!(&result);

        // After last
        let result = seek_term(&dir, &fi, &id, b"term_0100").unwrap();
        assert_none!(&result);
    }

    #[test]
    fn test_seek_exact_compressed_suffixes() {
        // Long lowercase terms with a shared prefix > 2 chars can trigger LZ4
        // or lowercase ASCII compression. Generate terms that are likely to compress.
        let mut terms_data: Vec<(String, Vec<i32>)> = Vec::new();
        for i in 0..40 {
            // Long suffixes with repeating patterns to encourage compression
            let term = format!("longprefix_abcdefghij_{i:04}_suffix");
            terms_data.push((term, vec![i]));
        }
        let terms: Vec<(&str, &[i32])> = terms_data
            .iter()
            .map(|(t, d)| (t.as_str(), d.as_slice()))
            .collect();

        let (dir, fi, id) = write_terms(terms, IndexOptions::Docs).unwrap();

        // Verify several terms are findable (compression is transparent to reader)
        for i in [0, 10, 20, 39] {
            let term = format!("longprefix_abcdefghij_{i:04}_suffix");
            let state = seek_term(&dir, &fi, &id, term.as_bytes()).unwrap();
            assert_some!(&state);
            assert_eq!(state.unwrap().doc_freq, 1);
        }
    }

    #[test]
    fn test_seek_exact_singleton_singleton_delta() {
        // Two consecutive singletons at the same doc_start_fp exercise the
        // singleton-singleton delta path in metadata decoding.
        // When both terms are singletons (df=1) and their doc_start_fp is the
        // same (no .doc data written), the metadata uses zigzag-encoded docID delta.
        let terms = vec![("aaa", &[10][..]), ("bbb", &[20]), ("ccc", &[30])];
        let (dir, fi, id) = write_terms(terms, IndexOptions::Docs).unwrap();

        let state = seek_term(&dir, &fi, &id, b"aaa").unwrap().unwrap();
        assert_eq!(state.singleton_doc_id, 10);

        let state = seek_term(&dir, &fi, &id, b"bbb").unwrap().unwrap();
        assert_eq!(state.singleton_doc_id, 20);

        let state = seek_term(&dir, &fi, &id, b"ccc").unwrap().unwrap();
        assert_eq!(state.singleton_doc_id, 30);
    }

    // --- SegmentTermsEnum (struct API) tests ---

    /// Open a BlockTreeTermsReader for field 0.
    fn open_reader(
        dir: &dyn Directory,
        field_infos: &FieldInfos,
        segment_id: &[u8; 16],
    ) -> BlockTreeTermsReader {
        BlockTreeTermsReader::open(dir, "_0", "", segment_id, field_infos).unwrap()
    }

    #[test]
    fn test_terms_enum_seek_exact() {
        let terms = vec![("alpha", &[0, 1][..]), ("beta", &[2]), ("gamma", &[3])];
        let (dir, fi, id) = write_terms(terms, IndexOptions::Docs).unwrap();

        let reader = open_reader(&dir, &fi, &id);
        let fr = reader.field_reader(0).unwrap();
        let mut te = fr.iterator().unwrap();
        assert!(te.seek_exact(b"alpha").unwrap());
        assert_eq!(te.term(), b"alpha");
        assert_eq!(te.doc_freq().unwrap(), 2);

        assert!(te.seek_exact(b"gamma").unwrap());
        assert_eq!(te.term(), b"gamma");
        assert_eq!(te.doc_freq().unwrap(), 1);

        assert!(!te.seek_exact(b"nonexistent").unwrap());
    }

    #[test]
    fn test_terms_enum_term_state_roundtrip() {
        let terms = vec![("hello", &[0, 1, 2][..]), ("world", &[3])];
        let (dir, fi, id) = write_terms(terms, IndexOptions::Docs).unwrap();

        let reader = open_reader(&dir, &fi, &id);
        let fr = reader.field_reader(0).unwrap();
        let mut te = fr.iterator().unwrap();
        assert!(te.seek_exact(b"hello").unwrap());
        let state = te.term_state().unwrap();
        assert_eq!(state.doc_freq, 3);

        // Seek to a different term
        assert!(te.seek_exact(b"world").unwrap());
        assert_eq!(te.doc_freq().unwrap(), 1);

        // Restore via seek_exact_with_state
        te.seek_exact_with_state(b"hello", state);
        assert_eq!(te.term(), b"hello");
        assert_eq!(te.doc_freq().unwrap(), 3);
        assert_eq!(te.term_state().unwrap().doc_freq, 3);
    }

    #[test]
    fn test_terms_enum_total_term_freq() {
        let terms = vec![("hello", &[0, 1][..]), ("world", &[0])];
        let (dir, fi, id) = write_terms(terms, IndexOptions::DocsAndFreqs).unwrap();

        let reader = open_reader(&dir, &fi, &id);
        let fr = reader.field_reader(0).unwrap();
        let mut te = fr.iterator().unwrap();
        assert!(te.seek_exact(b"hello").unwrap());
        assert_ge!(te.total_term_freq().unwrap(), 2);
    }

    #[test]
    fn test_terms_enum_unpositioned_errors() {
        let terms = vec![("hello", &[0][..])];
        let (dir, fi, id) = write_terms(terms, IndexOptions::Docs).unwrap();

        let reader = open_reader(&dir, &fi, &id);
        let fr = reader.field_reader(0).unwrap();
        let mut te = fr.iterator().unwrap();
        // Before any seek, doc_freq and term_state should error
        assert!(te.doc_freq().is_err());
        assert!(te.term_state().is_err());
    }

    #[test]
    fn test_terms_enum_many_terms() {
        let mut terms_data: Vec<(String, Vec<i32>)> = Vec::new();
        for i in 0..100 {
            terms_data.push((format!("term_{i:04}"), vec![i]));
        }
        let terms: Vec<(&str, &[i32])> = terms_data
            .iter()
            .map(|(t, d)| (t.as_str(), d.as_slice()))
            .collect();
        let (dir, fi, id) = write_terms(terms, IndexOptions::Docs).unwrap();

        let reader = open_reader(&dir, &fi, &id);
        let fr = reader.field_reader(0).unwrap();
        let mut te = fr.iterator().unwrap();
        for i in 0..100 {
            let term = format!("term_{i:04}");
            assert!(te.seek_exact(term.as_bytes()).unwrap());
            assert_eq!(te.doc_freq().unwrap(), 1);
            assert_eq!(te.term(), term.as_bytes());
        }
    }

    // --- next() tests ---

    #[test]
    fn test_next_single_block() {
        let terms = vec![("alpha", &[0][..]), ("beta", &[1]), ("gamma", &[2])];
        let (dir, fi, id) = write_terms(terms, IndexOptions::Docs).unwrap();

        let reader = open_reader(&dir, &fi, &id);
        let fr = reader.field_reader(0).unwrap();
        let mut te = fr.iterator().unwrap();

        assert_eq!(te.next().unwrap().unwrap(), b"alpha");
        assert_eq!(te.next().unwrap().unwrap(), b"beta");
        assert_eq!(te.next().unwrap().unwrap(), b"gamma");
        assert!(te.next().unwrap().is_none());
    }

    #[test]
    fn test_next_many_terms() {
        let mut terms_data: Vec<(String, Vec<i32>)> = Vec::new();
        for i in 0..100 {
            terms_data.push((format!("term_{i:04}"), vec![i]));
        }
        let terms: Vec<(&str, &[i32])> = terms_data
            .iter()
            .map(|(t, d)| (t.as_str(), d.as_slice()))
            .collect();
        let (dir, fi, id) = write_terms(terms, IndexOptions::Docs).unwrap();

        let reader = open_reader(&dir, &fi, &id);
        let fr = reader.field_reader(0).unwrap();
        let mut te = fr.iterator().unwrap();

        for i in 0..100 {
            let expected = format!("term_{i:04}");
            let term = te.next().unwrap().unwrap();
            assert_eq!(term, expected.as_bytes());
        }
        assert!(te.next().unwrap().is_none());
    }

    #[test]
    fn test_next_lexicographic_order() {
        let terms = vec![
            ("aardvark", &[0][..]),
            ("banana", &[1]),
            ("cherry", &[2]),
            ("date", &[3]),
        ];
        let (dir, fi, id) = write_terms(terms, IndexOptions::Docs).unwrap();

        let reader = open_reader(&dir, &fi, &id);
        let fr = reader.field_reader(0).unwrap();
        let mut te = fr.iterator().unwrap();

        let mut prev: Option<Vec<u8>> = None;
        while let Some(term) = te.next().unwrap() {
            if let Some(ref p) = prev {
                assert_lt!(*p, term.to_vec());
            }
            prev = Some(term.to_vec());
        }
        assert!(prev.is_some());
    }

    #[test]
    fn test_next_doc_freq_after_next() {
        let terms = vec![
            ("alpha", &[0, 1][..]),
            ("beta", &[2]),
            ("gamma", &[3, 4, 5]),
        ];
        let (dir, fi, id) = write_terms(terms, IndexOptions::Docs).unwrap();

        let reader = open_reader(&dir, &fi, &id);
        let fr = reader.field_reader(0).unwrap();
        let mut te = fr.iterator().unwrap();

        te.next().unwrap();
        assert_eq!(te.doc_freq().unwrap(), 2);

        te.next().unwrap();
        assert_eq!(te.doc_freq().unwrap(), 1);

        te.next().unwrap();
        assert_eq!(te.doc_freq().unwrap(), 3);
    }

    #[test]
    fn test_next_returns_none_after_exhaustion() {
        let terms = vec![("only", &[0][..])];
        let (dir, fi, id) = write_terms(terms, IndexOptions::Docs).unwrap();

        let reader = open_reader(&dir, &fi, &id);
        let fr = reader.field_reader(0).unwrap();
        let mut te = fr.iterator().unwrap();

        assert_eq!(te.next().unwrap().unwrap(), b"only");
        assert!(te.next().unwrap().is_none());
        // Second call after exhaustion should also return None
        assert!(te.next().unwrap().is_none());
    }

    // --- seek_exact + next interleaving tests ---

    #[test]
    fn test_seek_exact_then_next_continues() {
        let terms = vec![
            ("alpha", &[0][..]),
            ("beta", &[1]),
            ("gamma", &[2]),
            ("delta", &[3]),
        ];
        let (dir, fi, id) = write_terms(terms, IndexOptions::Docs).unwrap();

        let reader = open_reader(&dir, &fi, &id);
        let fr = reader.field_reader(0).unwrap();
        let mut te = fr.iterator().unwrap();

        // Seek to "beta", then next() should return "delta" (next in lex order)
        assert!(te.seek_exact(b"beta").unwrap());
        assert_eq!(te.term(), b"beta");

        let next = te.next().unwrap().unwrap();
        assert_eq!(next, b"delta");

        let next = te.next().unwrap().unwrap();
        assert_eq!(next, b"gamma");

        assert!(te.next().unwrap().is_none());
    }

    #[test]
    fn test_interleaved_seek_and_next() {
        let mut terms_data: Vec<(String, Vec<i32>)> = Vec::new();
        for i in 0..50 {
            terms_data.push((format!("term_{i:04}"), vec![i]));
        }
        let terms: Vec<(&str, &[i32])> = terms_data
            .iter()
            .map(|(t, d)| (t.as_str(), d.as_slice()))
            .collect();
        let (dir, fi, id) = write_terms(terms, IndexOptions::Docs).unwrap();

        let reader = open_reader(&dir, &fi, &id);
        let fr = reader.field_reader(0).unwrap();
        let mut te = fr.iterator().unwrap();

        // Seek to term_0010, then next should give term_0011
        assert!(te.seek_exact(b"term_0010").unwrap());
        assert_eq!(te.doc_freq().unwrap(), 1);

        let next = te.next().unwrap().unwrap();
        assert_eq!(next, b"term_0011");

        // Now seek to term_0020
        assert!(te.seek_exact(b"term_0020").unwrap());
        assert_eq!(te.term(), b"term_0020");

        // Next should give term_0021
        let next = te.next().unwrap().unwrap();
        assert_eq!(next, b"term_0021");
        assert_eq!(te.doc_freq().unwrap(), 1);
    }

    #[test]
    fn test_seek_exact_with_state_then_next() {
        let terms = vec![("alpha", &[0][..]), ("beta", &[1]), ("gamma", &[2])];
        let (dir, fi, id) = write_terms(terms, IndexOptions::Docs).unwrap();

        let reader = open_reader(&dir, &fi, &id);
        let fr = reader.field_reader(0).unwrap();
        let mut te = fr.iterator().unwrap();

        // Get state for "alpha"
        assert!(te.seek_exact(b"alpha").unwrap());
        let state = te.term_state().unwrap();

        // Move to "gamma"
        assert!(te.seek_exact(b"gamma").unwrap());

        // Restore to "alpha" via state, then next should give "beta"
        te.seek_exact_with_state(b"alpha", state);
        assert_eq!(te.term(), b"alpha");

        let next = te.next().unwrap().unwrap();
        assert_eq!(next, b"beta");

        let next = te.next().unwrap().unwrap();
        assert_eq!(next, b"gamma");

        assert!(te.next().unwrap().is_none());
    }

    #[test]
    fn test_seek_exact_after_next_exhaustion() {
        let terms = vec![("alpha", &[0][..]), ("beta", &[1])];
        let (dir, fi, id) = write_terms(terms, IndexOptions::Docs).unwrap();

        let reader = open_reader(&dir, &fi, &id);
        let fr = reader.field_reader(0).unwrap();
        let mut te = fr.iterator().unwrap();

        // Exhaust via next()
        assert_eq!(te.next().unwrap().unwrap(), b"alpha");
        assert_eq!(te.next().unwrap().unwrap(), b"beta");
        assert!(te.next().unwrap().is_none());

        // seek_exact should still work after exhaustion
        assert!(te.seek_exact(b"alpha").unwrap());
        assert_eq!(te.term(), b"alpha");
        assert_eq!(te.doc_freq().unwrap(), 1);
    }

    #[test]
    fn test_seek_forward_then_backward() {
        let mut terms_data: Vec<(String, Vec<i32>)> = Vec::new();
        for i in 0..100 {
            terms_data.push((format!("term_{i:04}"), vec![i]));
        }
        let terms: Vec<(&str, &[i32])> = terms_data
            .iter()
            .map(|(t, d)| (t.as_str(), d.as_slice()))
            .collect();
        let (dir, fi, id) = write_terms(terms, IndexOptions::Docs).unwrap();

        let reader = open_reader(&dir, &fi, &id);
        let fr = reader.field_reader(0).unwrap();
        let mut te = fr.iterator().unwrap();

        // Forward seek
        assert!(te.seek_exact(b"term_0050").unwrap());
        assert_eq!(te.doc_freq().unwrap(), 1);

        // Backward seek (exercises rewind path)
        assert!(te.seek_exact(b"term_0010").unwrap());
        assert_eq!(te.doc_freq().unwrap(), 1);

        // Next after backward seek
        let next = te.next().unwrap().unwrap();
        assert_eq!(next, b"term_0011");
    }

    // --- seek_ceil tests ---

    #[test]
    fn test_seek_ceil_exact_match() {
        let terms = vec![("alpha", &[0][..]), ("beta", &[1]), ("gamma", &[2])];
        let (dir, fi, id) = write_terms(terms, IndexOptions::Docs).unwrap();

        let reader = open_reader(&dir, &fi, &id);
        let fr = reader.field_reader(0).unwrap();
        let mut te = fr.iterator().unwrap();

        assert_eq!(te.seek_ceil(b"beta").unwrap(), SeekStatus::Found);
        assert_eq!(te.term(), b"beta");
        assert_eq!(te.doc_freq().unwrap(), 1);
    }

    #[test]
    fn test_seek_ceil_not_found() {
        let terms = vec![("alpha", &[0][..]), ("gamma", &[1])];
        let (dir, fi, id) = write_terms(terms, IndexOptions::Docs).unwrap();

        let reader = open_reader(&dir, &fi, &id);
        let fr = reader.field_reader(0).unwrap();
        let mut te = fr.iterator().unwrap();

        // "beta" doesn't exist, ceiling is "gamma"
        assert_eq!(te.seek_ceil(b"beta").unwrap(), SeekStatus::NotFound);
        assert_eq!(te.term(), b"gamma");
        assert_eq!(te.doc_freq().unwrap(), 1);
    }

    #[test]
    fn test_seek_ceil_past_all_terms() {
        let terms = vec![("alpha", &[0][..]), ("beta", &[1])];
        let (dir, fi, id) = write_terms(terms, IndexOptions::Docs).unwrap();

        let reader = open_reader(&dir, &fi, &id);
        let fr = reader.field_reader(0).unwrap();
        let mut te = fr.iterator().unwrap();

        assert_eq!(te.seek_ceil(b"zzz").unwrap(), SeekStatus::End);
    }

    #[test]
    fn test_seek_ceil_before_first_term() {
        let terms = vec![("beta", &[0][..]), ("gamma", &[1])];
        let (dir, fi, id) = write_terms(terms, IndexOptions::Docs).unwrap();

        let reader = open_reader(&dir, &fi, &id);
        let fr = reader.field_reader(0).unwrap();
        let mut te = fr.iterator().unwrap();

        // "alpha" is before all terms, ceiling is "beta"
        assert_eq!(te.seek_ceil(b"alpha").unwrap(), SeekStatus::NotFound);
        assert_eq!(te.term(), b"beta");
    }

    #[test]
    fn test_seek_ceil_then_next() {
        let terms = vec![
            ("alpha", &[0][..]),
            ("beta", &[1]),
            ("delta", &[2]),
            ("gamma", &[3]),
        ];
        let (dir, fi, id) = write_terms(terms, IndexOptions::Docs).unwrap();

        let reader = open_reader(&dir, &fi, &id);
        let fr = reader.field_reader(0).unwrap();
        let mut te = fr.iterator().unwrap();

        // seek_ceil to "charlie" -> "delta"
        assert_eq!(te.seek_ceil(b"charlie").unwrap(), SeekStatus::NotFound);
        assert_eq!(te.term(), b"delta");

        // next() should continue to "gamma"
        let next = te.next().unwrap().unwrap();
        assert_eq!(next, b"gamma");

        assert!(te.next().unwrap().is_none());
    }

    #[test]
    fn test_seek_ceil_many_terms() {
        let mut terms_data: Vec<(String, Vec<i32>)> = Vec::new();
        for i in 0..100 {
            terms_data.push((format!("term_{i:04}"), vec![i]));
        }
        let terms: Vec<(&str, &[i32])> = terms_data
            .iter()
            .map(|(t, d)| (t.as_str(), d.as_slice()))
            .collect();
        let (dir, fi, id) = write_terms(terms, IndexOptions::Docs).unwrap();

        let reader = open_reader(&dir, &fi, &id);
        let fr = reader.field_reader(0).unwrap();
        let mut te = fr.iterator().unwrap();

        // Exact match in multi-block index
        assert_eq!(te.seek_ceil(b"term_0050").unwrap(), SeekStatus::Found);
        assert_eq!(te.term(), b"term_0050");

        // Between two terms
        assert_eq!(te.seek_ceil(b"term_0050x").unwrap(), SeekStatus::NotFound);
        assert_eq!(te.term(), b"term_0051");

        // Past all terms
        assert_eq!(te.seek_ceil(b"term_9999").unwrap(), SeekStatus::End);
    }
}
