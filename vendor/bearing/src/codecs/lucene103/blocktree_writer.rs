// SPDX-License-Identifier: Apache-2.0
//! Block tree terms dictionary writer that organizes terms into a trie of blocks.

use std::collections::HashSet;
use std::io;
use std::io::Write;
use std::str;

use log::debug;

use crate::codecs::competitive_impact::NormsLookup;
use crate::codecs::fields_producer::{FieldTerms, PostingsEnumProducer};
use crate::codecs::lucene103::postings_format::{
    self, BLOCKTREE_VERSION_CURRENT, DEFAULT_MAX_BLOCK_SIZE, DEFAULT_MIN_BLOCK_SIZE,
    IntBlockTermState, TERMS_CODEC, TERMS_CODEC_NAME, TERMS_EXTENSION, TERMS_INDEX_CODEC_NAME,
    TERMS_INDEX_EXTENSION, TERMS_META_CODEC_NAME, TERMS_META_EXTENSION, VERSION_CURRENT,
};
use crate::codecs::{codec_footers, codec_headers};
use crate::document::IndexOptions;
use crate::encoding::write_encoding::WriteEncoding;
use crate::encoding::{lowercase_ascii, lz4};
use crate::index::index_file_names::segment_file_name;
use crate::index::pipeline::terms_hash::{
    BufferedPostingsEnum, DecodedPostings, FreqProxTermsWriterPerField, TermsHash,
};
use crate::store::{DataOutput, Directory, IndexOutput, VecOutput};
use crate::util::byte_block_pool::ByteBlockPool;

use super::postings_writer::PostingsWriter;

// ============================================================================
// BufferedFieldTerms — FieldTerms impl over in-memory pipeline data
// ============================================================================

/// [`FieldTerms`] implementation that wraps the indexing pipeline's in-memory
/// state. Borrows term bytes from `ByteBlockPool` (zero-copy) and decodes
/// postings on demand from the byte pool streams.
pub(crate) struct BufferedFieldTerms<'a> {
    per_field: &'a FreqProxTermsWriterPerField,
    term_byte_pool: &'a ByteBlockPool,
    terms_hash: &'a TermsHash,
    sorted_ids: &'a [i32],
    num_terms: usize,
    field_name: String,
    field_number: u32,
}

impl<'a> BufferedFieldTerms<'a> {
    /// Creates a new `BufferedFieldTerms` for one field.
    pub fn new(
        per_field: &'a FreqProxTermsWriterPerField,
        term_byte_pool: &'a ByteBlockPool,
        terms_hash: &'a TermsHash,
        field_name: &str,
        field_number: u32,
    ) -> Self {
        let num_terms = per_field.num_terms();
        let sorted_ids = per_field.sorted_term_ids();

        Self {
            per_field,
            term_byte_pool,
            terms_hash,
            sorted_ids,
            num_terms,
            field_name: field_name.to_string(),
            field_number,
        }
    }
}

impl FieldTerms for BufferedFieldTerms<'_> {
    fn term_count(&self) -> usize {
        self.num_terms
    }

    fn term_bytes(&self, index: usize) -> &[u8] {
        let term_id = self.sorted_ids[index] as usize;
        self.per_field.term_bytes(self.term_byte_pool, term_id)
    }

    fn postings(&self, index: usize) -> io::Result<Box<dyn PostingsEnumProducer + '_>> {
        let term_id = self.sorted_ids[index] as usize;
        let mut decoded = DecodedPostings::new();
        self.per_field
            .decode_term_into(self.terms_hash, term_id, &mut decoded)?;
        Ok(Box::new(BufferedPostingsEnum::new(
            decoded,
            self.per_field.has_freq,
        )))
    }

    fn index_options(&self) -> IndexOptions {
        self.per_field.index_options
    }

    fn has_payloads(&self) -> bool {
        self.per_field.saw_payloads
    }

    fn field_number(&self) -> u32 {
        self.field_number
    }

    fn field_name(&self) -> &str {
        &self.field_name
    }
}

// ============================================================================
// Block writing helper structs
// ============================================================================

/// Describes a single block to write to the .tim file.
/// Groups the 6 block-descriptor parameters that are always passed together.
struct BlockSpec {
    start: usize,
    end: usize,
    is_floor: bool,
    floor_lead_label: i32,
    has_terms: bool,
    has_sub_blocks: bool,
}

/// Groups field-level context needed when writing blocks to output.
pub(crate) struct FieldWriteContext {
    pub field_name: String,
    pub field_number: u32,
    pub index_options: IndexOptions,
}

// ============================================================================
// BlockTreeTermsWriter
// ============================================================================

/// Writes .tim, .tip, .tmd files plus delegates to PostingsWriter for .doc/.pos/.psm.
pub struct BlockTreeTermsWriter {
    terms_out: Box<dyn IndexOutput>,
    index_out: Box<dyn IndexOutput>,
    meta_out: Box<dyn IndexOutput>,
    postings_writer: PostingsWriter,
    min_items_in_block: usize,
    max_items_in_block: usize,
    field_metas: Vec<Vec<u8>>,
}

impl BlockTreeTermsWriter {
    /// Creates a new BlockTreeTermsWriter that streams to outputs from the directory.
    pub fn new(
        directory: &dyn Directory,
        segment: &str,
        suffix: &str,
        id: &[u8; 16],
        max_index_options: IndexOptions,
    ) -> io::Result<Self> {
        // Create outputs from directory
        let tim_name = segment_file_name(segment, suffix, TERMS_EXTENSION);
        let tip_name = segment_file_name(segment, suffix, TERMS_INDEX_EXTENSION);
        let tmd_name = segment_file_name(segment, suffix, TERMS_META_EXTENSION);

        let mut terms_out = directory.create_output(&tim_name)?;
        let mut index_out = directory.create_output(&tip_name)?;
        let mut meta_out = directory.create_output(&tmd_name)?;

        codec_headers::write_index_header(
            &mut *terms_out,
            TERMS_CODEC_NAME,
            BLOCKTREE_VERSION_CURRENT,
            id,
            suffix,
        )?;
        codec_headers::write_index_header(
            &mut *index_out,
            TERMS_INDEX_CODEC_NAME,
            BLOCKTREE_VERSION_CURRENT,
            id,
            suffix,
        )?;
        codec_headers::write_index_header(
            &mut *meta_out,
            TERMS_META_CODEC_NAME,
            BLOCKTREE_VERSION_CURRENT,
            id,
            suffix,
        )?;

        // Create PostingsWriter (which creates .doc, .pos, .psm)
        let postings_writer =
            PostingsWriter::new(directory, segment, suffix, id, max_index_options)?;

        // Write postings header into .tmd
        // In Java: postingsWriter.init(metaOut, state) writes TERMS_CODEC header + BLOCK_SIZE
        // to the metaOut (.tmd), not termsOut (.tim).
        codec_headers::write_index_header(
            &mut *meta_out,
            TERMS_CODEC,
            VERSION_CURRENT,
            id,
            suffix,
        )?;
        meta_out.write_vint(postings_format::BLOCK_SIZE as i32)?;

        Ok(Self {
            terms_out,
            index_out,
            meta_out,
            postings_writer,
            min_items_in_block: DEFAULT_MIN_BLOCK_SIZE,
            max_items_in_block: DEFAULT_MAX_BLOCK_SIZE,
            field_metas: Vec::new(),
        })
    }

    /// Write all terms for one indexed field via the [`FieldTerms`] abstraction.
    pub fn write_field(
        &mut self,
        field_terms: &dyn FieldTerms,
        norms: &dyn NormsLookup,
    ) -> io::Result<()> {
        let num_terms = field_terms.term_count();
        if num_terms == 0 {
            return Ok(());
        }

        // Derive FieldWriteContext from FieldTerms for internal use
        let field_ctx = FieldWriteContext {
            field_name: field_terms.field_name().to_string(),
            field_number: field_terms.field_number(),
            index_options: field_terms.index_options(),
        };

        debug!(
            "blocktree_writer: write_field name={} num_terms={}",
            field_ctx.field_name, num_terms
        );

        let mut tw =
            BlockTreeFieldWriter::new(&field_ctx, self.min_items_in_block, self.max_items_in_block);

        let mut docs_seen = HashSet::new();

        for i in 0..num_terms {
            let term_bytes = field_terms.term_bytes(i);
            let mut pe = field_terms.postings(i)?;

            let state = self.postings_writer.write_term(
                &mut *pe,
                field_ctx.index_options,
                norms,
                &mut docs_seen,
            )?;

            tw.add_term(
                term_bytes,
                state,
                &mut *self.terms_out,
                &self.postings_writer,
                &field_ctx,
            )?;
        }

        tw.doc_count = docs_seen.len() as i32;

        tw.finish(
            &mut *self.terms_out,
            &mut *self.index_out,
            &self.postings_writer,
            &field_ctx,
            &mut self.field_metas,
        )?;

        Ok(())
    }

    /// Finalize: write footers, drop outputs (auto-persists to directory).
    /// Returns the file names written.
    pub fn finish(mut self) -> io::Result<Vec<String>> {
        let mut all_names = Vec::new();

        // Write .tmd field count + field metas
        self.meta_out.write_vint(self.field_metas.len() as i32)?;
        for field_meta in &self.field_metas {
            self.meta_out.write_all(field_meta)?;
        }

        // Write .tip footer
        codec_footers::write_footer(&mut *self.index_out)?;
        // Write .tip end pointer to .tmd
        self.meta_out
            .write_le_long(self.index_out.file_pointer() as i64)?;

        // Write .tim footer
        codec_footers::write_footer(&mut *self.terms_out)?;
        // Write .tim end pointer to .tmd
        self.meta_out
            .write_le_long(self.terms_out.file_pointer() as i64)?;

        // Write .tmd footer
        codec_footers::write_footer(&mut *self.meta_out)?;

        // Collect file names
        all_names.push(self.terms_out.name().to_string());
        all_names.push(self.index_out.name().to_string());
        all_names.push(self.meta_out.name().to_string());

        // Get postings file names
        let postings_names = self.postings_writer.finish()?;
        all_names.extend(postings_names);

        // Dropping self auto-persists all outputs to directory
        Ok(all_names)
    }
}

// ============================================================================
// PendingEntry / TermsWriter
// ============================================================================

enum PendingEntry<'a> {
    Term(PendingTerm<'a>),
    Block(PendingBlock),
}

impl PendingEntry<'_> {
    fn is_term(&self) -> bool {
        matches!(self, PendingEntry::Term(_))
    }
}

struct PendingTerm<'a> {
    term_bytes: &'a [u8],
    state: IntBlockTermState,
}

struct PendingBlock {
    prefix: Vec<u8>,
    fp: u64,
    has_terms: bool,
    is_floor: bool,
    floor_lead_byte: i32,
    index: Option<TrieBuilder>,
    sub_indices: Option<Vec<TrieBuilder>>,
}

/// Accumulates terms for a single field and builds blocks.
struct BlockTreeFieldWriter<'a> {
    write_freqs: bool,
    min_items_in_block: usize,
    max_items_in_block: usize,
    pending: Vec<PendingEntry<'a>>,
    last_term: Vec<u8>,
    prefix_starts: Vec<usize>,
    num_terms: u64,
    sum_total_term_freq: i64,
    sum_doc_freq: i64,
    doc_count: i32,
    first_term_bytes: Vec<u8>,
    last_term_bytes: Vec<u8>,
    lz4_ht: lz4::HighCompressionHashTable,
}

impl<'a> BlockTreeFieldWriter<'a> {
    fn new(
        field_ctx: &FieldWriteContext,
        min_items_in_block: usize,
        max_items_in_block: usize,
    ) -> Self {
        let write_freqs = field_ctx.index_options.has_freqs();
        Self {
            write_freqs,
            min_items_in_block,
            max_items_in_block,
            pending: Vec::new(),
            last_term: Vec::new(),
            prefix_starts: Vec::new(),
            num_terms: 0,
            sum_total_term_freq: 0,
            sum_doc_freq: 0,
            doc_count: 0,
            lz4_ht: lz4::HighCompressionHashTable::new(),
            first_term_bytes: Vec::new(),
            last_term_bytes: Vec::new(),
        }
    }

    fn add_term(
        &mut self,
        term: &'a [u8],
        state: IntBlockTermState,
        terms_out: &mut dyn IndexOutput,
        postings_writer: &PostingsWriter,
        field_ctx: &FieldWriteContext,
    ) -> io::Result<()> {
        self.sum_doc_freq += state.doc_freq as i64;
        if self.write_freqs {
            self.sum_total_term_freq += state.total_term_freq;
        }
        self.num_terms += 1;

        if self.num_terms == 1 {
            self.first_term_bytes.clear();
            self.first_term_bytes.extend_from_slice(term);
        }
        self.last_term_bytes.clear();
        self.last_term_bytes.extend_from_slice(term);

        self.push_term(term, terms_out, postings_writer, field_ctx)?;

        self.pending.push(PendingEntry::Term(PendingTerm {
            term_bytes: term,
            state,
        }));

        Ok(())
    }

    /// Push a new term, writing blocks as needed when prefix groups get large enough.
    fn push_term(
        &mut self,
        text: &[u8],
        terms_out: &mut dyn IndexOutput,
        postings_writer: &PostingsWriter,
        field_ctx: &FieldWriteContext,
    ) -> io::Result<()> {
        let prefix_length = mismatch(&self.last_term, text);

        // Close abandoned suffixes
        for i in (prefix_length..self.last_term.len()).rev() {
            if self.pending.len() < self.prefix_starts[i] {
                continue;
            }
            let prefix_top_size = self.pending.len() - self.prefix_starts[i];
            if prefix_top_size >= self.min_items_in_block {
                self.write_blocks(
                    i + 1,
                    prefix_top_size,
                    terms_out,
                    postings_writer,
                    field_ctx,
                )?;
                self.prefix_starts[i] = self.prefix_starts[i].saturating_sub(prefix_top_size - 1);
            }
        }

        // Grow prefix_starts if needed
        if self.prefix_starts.len() < text.len() {
            self.prefix_starts.resize(text.len(), 0);
        }

        // Init new tail
        for i in prefix_length..text.len() {
            self.prefix_starts[i] = self.pending.len();
        }

        self.last_term.clear();
        self.last_term.extend_from_slice(text);

        Ok(())
    }

    /// Write the top `count` entries from pending as blocks.
    /// This writes block data directly to `terms_out`, matching Java's BlockTreeTermsWriter.writeBlocks.
    fn write_blocks(
        &mut self,
        prefix_length: usize,
        count: usize,
        terms_out: &mut dyn IndexOutput,
        postings_writer: &PostingsWriter,
        field_ctx: &FieldWriteContext,
    ) -> io::Result<()> {
        let start = self.pending.len() - count;
        let end = self.pending.len();

        let mut last_suffix_lead_label: i32 = -1;
        let mut has_terms = false;
        let mut has_sub_blocks = false;
        let mut next_block_start = start;
        let mut next_floor_lead_label: i32 = -1;

        let mut block_specs: Vec<BlockSpec> = Vec::new();

        for i in start..end {
            let suffix_lead_label = match &self.pending[i] {
                PendingEntry::Term(t) => {
                    if t.term_bytes.len() == prefix_length {
                        -1
                    } else {
                        t.term_bytes[prefix_length] as i32
                    }
                }
                PendingEntry::Block(b) => {
                    if b.prefix.len() > prefix_length {
                        b.prefix[prefix_length] as i32
                    } else {
                        -1
                    }
                }
            };

            if suffix_lead_label != last_suffix_lead_label {
                let items_in_block = i - next_block_start;
                if items_in_block >= self.min_items_in_block
                    && end - next_block_start > self.max_items_in_block
                {
                    let is_floor = items_in_block < count;
                    block_specs.push(BlockSpec {
                        start: next_block_start,
                        end: i,
                        is_floor,
                        floor_lead_label: next_floor_lead_label,
                        has_terms,
                        has_sub_blocks,
                    });
                    has_terms = false;
                    has_sub_blocks = false;
                    next_floor_lead_label = suffix_lead_label;
                    next_block_start = i;
                }
                last_suffix_lead_label = suffix_lead_label;
            }

            if self.pending[i].is_term() {
                has_terms = true;
            } else {
                has_sub_blocks = true;
            }
        }

        if next_block_start < end {
            let items_in_block = end - next_block_start;
            let is_floor = items_in_block < count;
            block_specs.push(BlockSpec {
                start: next_block_start,
                end,
                is_floor,
                floor_lead_label: next_floor_lead_label,
                has_terms,
                has_sub_blocks,
            });
        }

        // Write each block to terms_out
        let mut new_blocks: Vec<PendingBlock> = Vec::new();

        for spec in &block_specs {
            let block = write_block_to_output(
                &BlockWriteInput {
                    pending: &self.pending,
                    last_term: &self.last_term,
                    prefix_length,
                },
                spec,
                terms_out,
                postings_writer,
                field_ctx,
                &mut self.lz4_ht,
            )?;
            new_blocks.push(block);
        }

        // Compile index for all blocks
        if !new_blocks.is_empty() {
            let is_floor = new_blocks[0].is_floor;

            let mut floor_data_bytes: Option<Vec<u8>> = None;
            if is_floor && new_blocks.len() > 1 {
                let mut fd = Vec::new();
                {
                    let mut buf = VecOutput(&mut fd);
                    buf.write_vint((new_blocks.len() - 1) as i32)?;
                    let base_fp = new_blocks[0].fp;
                    for block in new_blocks.iter().skip(1) {
                        buf.write_byte(block.floor_lead_byte as u8)?;
                        let delta = block.fp - base_fp;
                        let encoded = (delta << 1) | if block.has_terms { 1 } else { 0 };
                        buf.write_vlong(encoded as i64)?;
                    }
                }
                floor_data_bytes = Some(fd);
            }

            let output = TrieOutput {
                fp: new_blocks[0].fp as i64,
                has_terms: new_blocks[0].has_terms,
                floor_data: floor_data_bytes,
            };

            let prefix_bytes = new_blocks[0].prefix.clone();
            let mut trie_builder = TrieBuilder::from_bytes(&prefix_bytes, output);

            for block in new_blocks.iter_mut() {
                if let Some(sub_indices) = block.sub_indices.take() {
                    for sub_index in sub_indices {
                        trie_builder.append(sub_index);
                    }
                }
            }

            new_blocks[0].index = Some(trie_builder);
        }

        // Replace pending entries
        let pending_len = self.pending.len();
        self.pending.truncate(pending_len - count);

        if !new_blocks.is_empty() {
            let first_block = new_blocks.remove(0);
            self.pending.push(PendingEntry::Block(first_block));
        }

        Ok(())
    }

    /// Finalize: write remaining blocks, then field metadata.
    fn finish(
        &mut self,
        terms_out: &mut dyn IndexOutput,
        index_out: &mut dyn IndexOutput,
        postings_writer: &PostingsWriter,
        field_ctx: &FieldWriteContext,
        field_metas: &mut Vec<Vec<u8>>,
    ) -> io::Result<()> {
        if self.num_terms == 0 {
            return Ok(());
        }

        // Add empty term to force closing of all final blocks
        self.push_term(&[], terms_out, postings_writer, field_ctx)?;
        self.push_term(&[], terms_out, postings_writer, field_ctx)?;

        // Write all remaining pending entries as root block
        let pending_count = self.pending.len();
        self.write_blocks(0, pending_count, terms_out, postings_writer, field_ctx)?;

        // The last pending entry should be a single root block
        assert!(
            self.pending.len() == 1 && !self.pending[0].is_term(),
            "expected single root block, got {} entries",
            self.pending.len()
        );

        let root = match self.pending.remove(0) {
            PendingEntry::Block(b) => b,
            _ => unreachable!(),
        };

        // Compute doc_count from docs_seen
        let doc_count = self.doc_count;

        // Write field metadata to a buffer
        let mut meta_buf = Vec::new();
        {
            let mut meta = VecOutput(&mut meta_buf);
            meta.write_vint(field_ctx.field_number as i32)?;
            meta.write_vlong(self.num_terms as i64)?;
            if self.write_freqs {
                meta.write_vlong(self.sum_total_term_freq)?;
            }
            meta.write_vlong(self.sum_doc_freq)?;
            meta.write_vint(doc_count)?;

            // Min term
            if !self.first_term_bytes.is_empty() {
                write_bytes_ref(&mut meta, &self.first_term_bytes)?;
            } else {
                meta.write_vint(0)?;
            }

            // Max term
            if !self.last_term_bytes.is_empty() {
                write_bytes_ref(&mut meta, &self.last_term_bytes)?;
            } else {
                meta.write_vint(0)?;
            }

            // Save trie index
            if let Some(trie) = root.index {
                trie.save(&mut meta, index_out)?;
            }
        }

        field_metas.push(meta_buf);

        debug!(
            "blocktree_writer: finish field={} num_terms={} sum_doc_freq={} doc_count={}",
            field_ctx.field_name, self.num_terms, self.sum_doc_freq, doc_count
        );

        Ok(())
    }
}

/// Groups the accumulated terms state passed into block writing.
struct BlockWriteInput<'a, 'b> {
    pending: &'a [PendingEntry<'b>],
    last_term: &'a [u8],
    prefix_length: usize,
}

/// Write a single block to .tim output.
fn write_block_to_output(
    input: &BlockWriteInput<'_, '_>,
    spec: &BlockSpec,
    mut terms_out: &mut dyn IndexOutput,
    postings_writer: &PostingsWriter,
    field_ctx: &FieldWriteContext,
    lz4_ht: &mut lz4::HighCompressionHashTable,
) -> io::Result<PendingBlock> {
    let pending = input.pending;
    let last_term = input.last_term;
    let prefix_length = input.prefix_length;
    let start_fp = terms_out.file_pointer();
    let has_floor_lead_label = spec.is_floor && spec.floor_lead_label != -1;

    let mut prefix = Vec::with_capacity(prefix_length + if has_floor_lead_label { 1 } else { 0 });
    prefix.extend_from_slice(&last_term[..prefix_length.min(last_term.len())]);
    // Pad if last_term is shorter than prefix_length
    while prefix.len() < prefix_length {
        prefix.push(0);
    }

    let num_entries = spec.end - spec.start;
    let is_leaf_block = !spec.has_sub_blocks;

    // Block header: (numEntries << 1) | isLastBlock
    let code = (num_entries << 1) | if spec.end == pending.len() { 1 } else { 0 };
    terms_out.write_vint(code as i32)?;

    // Buffers for suffix data, suffix lengths, stats, and metadata
    let mut suffix_bytes = Vec::new();
    let mut suffix_lengths_bytes = Vec::new();
    let mut stats_bytes = Vec::new();
    let mut meta_bytes = Vec::new();

    let mut suffix_lengths_writer = VecOutput(&mut suffix_lengths_bytes);
    let mut stats_writer = StatsWriter::new(&mut stats_bytes, field_ctx.index_options.has_freqs());
    let mut last_state = IntBlockTermState::new();
    let mut absolute = true;

    let mut sub_indices: Vec<TrieBuilder> = Vec::new();

    if is_leaf_block {
        // Block contains only terms
        for entry in &pending[spec.start..spec.end] {
            if let PendingEntry::Term(term) = entry {
                let suffix_len = term.term_bytes.len() - prefix_length;
                suffix_lengths_writer.write_vint(suffix_len as i32)?;
                suffix_bytes.extend_from_slice(&term.term_bytes[prefix_length..]);

                stats_writer.add(term.state.doc_freq, term.state.total_term_freq)?;

                let empty_state = IntBlockTermState::new();
                let ref_state = if absolute { &empty_state } else { &last_state };
                postings_writer.encode_term(
                    &mut meta_bytes,
                    &term.state,
                    ref_state,
                    field_ctx.index_options.has_positions(),
                    field_ctx.index_options.has_offsets(),
                )?;
                last_state = term.state;
                absolute = false;
            }
        }
        stats_writer.finish()?;
    } else {
        // Block has terms and/or sub-blocks
        for entry in &pending[spec.start..spec.end] {
            match entry {
                PendingEntry::Term(term) => {
                    let suffix_len = term.term_bytes.len() - prefix_length;
                    // Steal 1 bit: even = term
                    suffix_lengths_writer.write_vint((suffix_len << 1) as i32)?;
                    suffix_bytes.extend_from_slice(&term.term_bytes[prefix_length..]);

                    stats_writer.add(term.state.doc_freq, term.state.total_term_freq)?;

                    let empty_state = IntBlockTermState::new();
                    let ref_state = if absolute { &empty_state } else { &last_state };
                    postings_writer.encode_term(
                        &mut meta_bytes,
                        &term.state,
                        ref_state,
                        field_ctx.index_options.has_positions(),
                        field_ctx.index_options.has_offsets(),
                    )?;
                    last_state = term.state;
                    absolute = false;
                }
                PendingEntry::Block(block) => {
                    let suffix_len = block.prefix.len() - prefix_length;
                    // Steal 1 bit: odd = sub-block
                    suffix_lengths_writer.write_vint(((suffix_len << 1) | 1) as i32)?;
                    suffix_bytes.extend_from_slice(&block.prefix[prefix_length..]);

                    // Write sub-block fp delta
                    suffix_lengths_writer.write_vlong((start_fp - block.fp) as i64)?;

                    if let Some(idx) = &block.index {
                        sub_indices.push(idx.clone());
                    }
                }
            }
        }
        stats_writer.finish()?;
    }

    // Write suffix data, optionally compressed with LZ4 or lowercase ASCII.
    let suffix_len = suffix_bytes.len();
    let mut compression_code: i64 = 0; // 0=none, 1=lowercase_ascii, 2=lz4
    let mut compressed_bytes: Option<Vec<u8>> = None;

    // Only try compression when there are enough suffix bytes per entry
    // and the prefix is long enough (>2) to avoid hurting fuzzy query perf.
    if suffix_len > 2 * num_entries && prefix_length > 2 {
        // Try LZ4 first if average suffix > 6 bytes
        if suffix_len > 6 * num_entries {
            let lz4_compressed = lz4::compress_high(&suffix_bytes, lz4_ht);
            // Use LZ4 only if it saves > 25%
            if lz4_compressed.len() < suffix_len - (suffix_len >> 2) {
                compression_code = 2;
                compressed_bytes = Some(lz4_compressed);
            }
        }
        // Fall back to lowercase ASCII if LZ4 didn't help
        if compression_code == 0
            && let Some(ascii_compressed) = lowercase_ascii::compress(&suffix_bytes, suffix_len)
        {
            compression_code = 1;
            compressed_bytes = Some(ascii_compressed);
        }
    }

    let mut token = (suffix_len as i64) << 3;
    if is_leaf_block {
        token |= 0x04;
    }
    token |= compression_code;
    terms_out.write_vlong(token)?;
    if let Some(ref data) = compressed_bytes {
        terms_out.write_all(data)?;
    } else {
        terms_out.write_all(&suffix_bytes)?;
    }

    // Write suffix lengths
    let num_suffix_bytes = suffix_lengths_bytes.len();
    if num_suffix_bytes > 1 && all_equal(&suffix_lengths_bytes[1..], suffix_lengths_bytes[0]) {
        terms_out.write_vint(((num_suffix_bytes << 1) | 1) as i32)?;
        terms_out.write_byte(suffix_lengths_bytes[0])?;
    } else {
        terms_out.write_vint((num_suffix_bytes << 1) as i32)?;
        terms_out.write_all(&suffix_lengths_bytes)?;
    }

    // Write stats
    terms_out.write_vint(stats_bytes.len() as i32)?;
    terms_out.write_all(&stats_bytes)?;

    // Write metadata
    terms_out.write_vint(meta_bytes.len() as i32)?;
    terms_out.write_all(&meta_bytes)?;

    if has_floor_lead_label {
        prefix.push(spec.floor_lead_label as u8);
    }

    Ok(PendingBlock {
        prefix,
        fp: start_fp,
        has_terms: spec.has_terms,
        is_floor: spec.is_floor,
        floor_lead_byte: spec.floor_lead_label,
        index: None,
        sub_indices: if sub_indices.is_empty() {
            None
        } else {
            Some(sub_indices)
        },
    })
}

fn all_equal(bytes: &[u8], value: u8) -> bool {
    bytes.iter().all(|&b| b == value)
}

fn write_bytes_ref(out: &mut VecOutput, bytes: &[u8]) -> io::Result<()> {
    out.write_vint(bytes.len() as i32)?;
    out.write_all(bytes)?;
    Ok(())
}

fn mismatch(a: &[u8], b: &[u8]) -> usize {
    let len = a.len().min(b.len());
    for i in 0..len {
        if a[i] != b[i] {
            return i;
        }
    }
    len
}

// ============================================================================
// StatsWriter — run-length encodes singleton (df=1, ttf=1) terms
// ============================================================================

struct StatsWriter<'a> {
    out: &'a mut Vec<u8>,
    has_freqs: bool,
    singleton_count: i32,
}

impl<'a> StatsWriter<'a> {
    fn new(out: &'a mut Vec<u8>, has_freqs: bool) -> Self {
        Self {
            out,
            has_freqs,
            singleton_count: 0,
        }
    }

    fn add(&mut self, df: i32, ttf: i64) -> io::Result<()> {
        if df == 1 && (!self.has_freqs || ttf == 1) {
            self.singleton_count += 1;
        } else {
            self.flush_singletons()?;
            let mut buf = VecOutput(self.out);
            buf.write_vint(df << 1)?;
            if self.has_freqs {
                buf.write_vlong(ttf - df as i64)?;
            }
        }
        Ok(())
    }

    fn finish(&mut self) -> io::Result<()> {
        self.flush_singletons()
    }

    fn flush_singletons(&mut self) -> io::Result<()> {
        if self.singleton_count > 0 {
            let mut buf = VecOutput(self.out);
            buf.write_vint(((self.singleton_count - 1) << 1) | 1)?;
            self.singleton_count = 0;
        }
        Ok(())
    }
}

// ============================================================================
// TrieBuilder — prefix trie for .tip index
// ============================================================================

// Node type signatures
const SIGN_NO_CHILDREN: u8 = 0x00;
const SIGN_SINGLE_CHILD_WITH_OUTPUT: u8 = 0x01;
const SIGN_SINGLE_CHILD_WITHOUT_OUTPUT: u8 = 0x02;
const SIGN_MULTI_CHILDREN: u8 = 0x03;

const LEAF_NODE_HAS_TERMS: u8 = 1 << 5;
const LEAF_NODE_HAS_FLOOR: u8 = 1 << 6;
const NON_LEAF_NODE_HAS_TERMS: u64 = 1 << 1;
const NON_LEAF_NODE_HAS_FLOOR: u64 = 1 << 0;

#[derive(Clone, Debug)]
pub struct TrieOutput {
    pub fp: i64,
    pub has_terms: bool,
    pub floor_data: Option<Vec<u8>>,
}

#[derive(Clone, Debug)]
struct TrieNode {
    label: u8,
    output: Option<TrieOutput>,
    children: Vec<TrieNode>,
}

#[derive(Clone, Debug)]
pub struct TrieBuilder {
    root: TrieNode,
    min_key: Vec<u8>,
    max_key: Vec<u8>,
}

impl TrieBuilder {
    pub fn from_bytes(key: &[u8], output: TrieOutput) -> Self {
        let mut root = TrieNode {
            label: 0,
            output: if key.is_empty() {
                Some(output.clone())
            } else {
                None
            },
            children: Vec::new(),
        };

        if !key.is_empty() {
            // Build chain of nodes for each byte
            let mut nodes: Vec<TrieNode> = Vec::new();

            for (i, &b) in key.iter().enumerate() {
                let is_last = i == key.len() - 1;
                nodes.push(TrieNode {
                    label: b,
                    output: if is_last { Some(output.clone()) } else { None },
                    children: Vec::new(),
                });
            }

            // Build tree bottom-up
            while nodes.len() > 1 {
                let child = nodes.pop().unwrap();
                nodes.last_mut().unwrap().children.push(child);
            }

            if let Some(node) = nodes.pop() {
                root.children.push(node);
            }
        }

        Self {
            root,
            min_key: key.to_vec(),
            max_key: key.to_vec(),
        }
    }

    /// Append another trie (whose keys are all greater than this one's max key).
    pub fn append(&mut self, other: TrieBuilder) {
        let mismatch_pos = mismatch(&self.max_key, &other.min_key);
        merge_nodes(&mut self.root, &other.root, 0, mismatch_pos);
        self.max_key = other.max_key;
    }

    /// Save the trie to .tmd (meta) and .tip (index) outputs.
    pub fn save(
        &self,
        mut meta: &mut dyn DataOutput,
        index: &mut dyn IndexOutput,
    ) -> io::Result<()> {
        meta.write_vlong(index.file_pointer() as i64)?; // indexStartFP
        let root_fp = self.save_nodes(index)?;
        meta.write_vlong(root_fp)?; // rootCode (offset of root node in .tip)
        index.write_le_long(0)?; // 8 bytes for over-reading
        meta.write_vlong(index.file_pointer() as i64)?; // indexEndFP
        Ok(())
    }

    /// Save trie nodes to the index output in post-order.
    /// Returns the root node's FP (relative to the start of this save).
    fn save_nodes(&self, index: &mut dyn IndexOutput) -> io::Result<i64> {
        let start_fp = index.file_pointer();
        let root_fp = save_node_recursive(&self.root, index, start_fp)?;
        Ok(root_fp)
    }
}

/// Recursively save nodes in post-order (children before parent).
/// Returns the fp (relative to start_fp) where this node was written.
fn save_node_recursive(
    node: &TrieNode,
    index: &mut dyn IndexOutput,
    start_fp: u64,
) -> io::Result<i64> {
    // First save all children
    let mut child_fps: Vec<i64> = Vec::new();
    for child in &node.children {
        let fp = save_node_recursive(child, index, start_fp)?;
        child_fps.push(fp);
    }

    let node_fp = (index.file_pointer() - start_fp) as i64;

    let children_num = node.children.len();

    if children_num == 0 {
        // Leaf node
        if let Some(ref output) = node.output {
            let output_fp_bytes = bytes_required_vlong(output.fp);
            let header: u8 = SIGN_NO_CHILDREN
                | ((output_fp_bytes as u8 - 1) << 2)
                | (if output.has_terms {
                    LEAF_NODE_HAS_TERMS
                } else {
                    0
                })
                | (if output.floor_data.is_some() {
                    LEAF_NODE_HAS_FLOOR
                } else {
                    0
                });
            index.write_byte(header)?;
            write_long_n_bytes(output.fp, output_fp_bytes, index)?;
            if let Some(ref floor_data) = output.floor_data {
                index.write_all(floor_data.as_slice())?;
            }
        }
    } else if children_num == 1 {
        // Single child node
        let child_delta_fp = node_fp - child_fps[0];
        let child_fp_bytes = bytes_required_vlong(child_delta_fp);
        let encoded_output_fp_bytes: u32 = match node.output {
            Some(ref output) => bytes_required_vlong(encode_fp(output)),
            None => 0,
        };

        let sign: u32 = if node.output.is_some() {
            SIGN_SINGLE_CHILD_WITH_OUTPUT as u32
        } else {
            SIGN_SINGLE_CHILD_WITHOUT_OUTPUT as u32
        };

        // Java unconditionally includes encoded_output_fp_bytes in the header.
        // When output is absent (encoded_output_fp_bytes=0), the (0-1)<<5 overflow
        // sets the high bits to 0xE0. The reader ignores these bits for
        // SIGN_SINGLE_CHILD_WITHOUT_OUTPUT, but we must match the bytes.
        let header = (sign
            | ((child_fp_bytes - 1) << 2)
            | (encoded_output_fp_bytes.wrapping_sub(1) << 5)) as u8;
        index.write_byte(header)?;
        index.write_byte(node.children[0].label)?;
        write_long_n_bytes(child_delta_fp, child_fp_bytes, index)?;

        if let Some(ref output) = node.output {
            let encoded_fp = encode_fp(output);
            write_long_n_bytes(encoded_fp, encoded_output_fp_bytes, index)?;
            if let Some(ref floor_data) = output.floor_data {
                index.write_all(floor_data.as_slice())?;
            }
        }
    } else {
        // Multi-children node
        let min_label = node.children[0].label;
        let max_label = node.children.last().unwrap().label;

        let strategy = choose_child_save_strategy(min_label, max_label, children_num);
        let strategy_bytes = strategy.need_bytes(min_label, max_label, children_num);

        let max_child_delta_fp = node_fp - child_fps[0];
        let children_fp_bytes = bytes_required_vlong(max_child_delta_fp);
        let encoded_output_fp_bytes = if let Some(ref output) = node.output {
            bytes_required_vlong(encode_fp(output))
        } else {
            1
        };

        let header: u32 = (SIGN_MULTI_CHILDREN as u32)
            | ((children_fp_bytes - 1) << 2)
            | ((if node.output.is_some() { 1u32 } else { 0 }) << 5)
            | ((encoded_output_fp_bytes - 1) << 6)
            | (strategy.code() << 9)
            | ((strategy_bytes - 1) << 11)
            | ((min_label as u32) << 16);

        write_long_n_bytes(header as i64, 3, index)?;

        if let Some(ref output) = node.output {
            let encoded_fp = encode_fp(output);
            write_long_n_bytes(encoded_fp, encoded_output_fp_bytes, index)?;
            if output.floor_data.is_some() {
                index.write_byte((children_num - 1) as u8)?;
            }
        }

        // Write strategy data
        strategy.save(node, strategy_bytes, index)?;

        // Write children fps (deltas from node_fp)
        for &child_fp in &child_fps {
            write_long_n_bytes(node_fp - child_fp, children_fp_bytes, index)?;
        }

        // Write floor data if present
        if let Some(ref output) = node.output
            && let Some(ref floor_data) = output.floor_data
        {
            index.write_all(floor_data.as_slice())?;
        }
    }

    Ok(node_fp)
}

fn encode_fp(output: &TrieOutput) -> i64 {
    let mut encoded = output.fp << 2;
    if output.has_terms {
        encoded |= NON_LEAF_NODE_HAS_TERMS as i64;
    }
    if output.floor_data.is_some() {
        encoded |= NON_LEAF_NODE_HAS_FLOOR as i64;
    }
    encoded
}

fn bytes_required_vlong(v: i64) -> u32 {
    let v = v as u64 | 1; // ensure at least 1 byte
    (8 - (v.leading_zeros() >> 3)).max(1)
}

fn write_long_n_bytes(v: i64, n: u32, out: &mut dyn DataOutput) -> io::Result<()> {
    out.write_all(&(v as u64).to_le_bytes()[..n as usize])
}

/// Merge nodes from `other` into `this` at the given depth.
fn merge_nodes(this: &mut TrieNode, other: &TrieNode, depth: usize, mismatch_pos: usize) {
    if depth < mismatch_pos {
        // Find matching child and recurse
        if let Some(other_child) = other.children.first() {
            if let Some(this_child) = this
                .children
                .iter_mut()
                .find(|c| c.label == other_child.label)
            {
                merge_nodes(this_child, other_child, depth + 1, mismatch_pos);
                // Also add any additional children from other
                for other_c in other.children.iter().skip(1) {
                    this.children.push(other_c.clone());
                }
            } else {
                // No matching child — just add all other's children
                for c in &other.children {
                    this.children.push(c.clone());
                }
            }
        }
    } else {
        // Past mismatch point — add all children from other
        for c in &other.children {
            if !this.children.iter().any(|tc| tc.label == c.label) {
                this.children.push(c.clone());
            }
        }
    }

    // Keep children sorted by label
    this.children.sort_by_key(|c| c.label);
}

// ============================================================================
// ChildSaveStrategy
// ============================================================================

enum ChildSaveStrategy {
    Bits,
    Array,
    ReverseArray,
}

impl ChildSaveStrategy {
    fn code(&self) -> u32 {
        match self {
            ChildSaveStrategy::ReverseArray => 0,
            ChildSaveStrategy::Array => 1,
            ChildSaveStrategy::Bits => 2,
        }
    }

    fn need_bytes(&self, min_label: u8, max_label: u8, label_cnt: usize) -> u32 {
        let byte_distance = (max_label as u32) - (min_label as u32) + 1;
        match self {
            ChildSaveStrategy::Bits => byte_distance.div_ceil(8),
            ChildSaveStrategy::Array => (label_cnt - 1) as u32,
            ChildSaveStrategy::ReverseArray => byte_distance - label_cnt as u32 + 1,
        }
    }

    fn save(
        &self,
        node: &TrieNode,
        _strategy_bytes: u32,
        out: &mut dyn DataOutput,
    ) -> io::Result<()> {
        match self {
            ChildSaveStrategy::Bits => {
                let min_label = node.children[0].label;
                let mut presence_bits: u8 = 1; // First child always present
                let mut presence_index = 0usize;
                let mut previous_label = min_label;

                for child in node.children.iter().skip(1) {
                    presence_index += (child.label - previous_label) as usize;
                    while presence_index >= 8 {
                        out.write_byte(presence_bits)?;
                        presence_bits = 0;
                        presence_index -= 8;
                    }
                    presence_bits |= 1 << presence_index;
                    previous_label = child.label;
                }
                out.write_byte(presence_bits)?;
            }
            ChildSaveStrategy::Array => {
                // Write all labels except the minimum (which is in the header)
                for child in node.children.iter().skip(1) {
                    out.write_byte(child.label)?;
                }
            }
            ChildSaveStrategy::ReverseArray => {
                // Write max label, then absent labels
                let max_label = node.children.last().unwrap().label;
                out.write_byte(max_label)?;
                let min_label = node.children[0].label;
                let mut last_label = min_label;
                for child in node.children.iter().skip(1) {
                    let mut next = last_label + 1;
                    while next < child.label {
                        out.write_byte(next)?;
                        next += 1;
                    }
                    last_label = child.label;
                }
            }
        }
        Ok(())
    }
}

fn choose_child_save_strategy(min_label: u8, max_label: u8, label_cnt: usize) -> ChildSaveStrategy {
    let strategies = [
        ChildSaveStrategy::Bits,
        ChildSaveStrategy::Array,
        ChildSaveStrategy::ReverseArray,
    ];

    let mut best = &strategies[0];
    let mut best_bytes = u32::MAX;

    for strategy in &strategies {
        let cost = strategy.need_bytes(min_label, max_label, label_cnt);
        if cost < best_bytes {
            best_bytes = cost;
            best = strategy;
        }
    }

    // Return a new instance matching the best
    match best {
        ChildSaveStrategy::Bits => ChildSaveStrategy::Bits,
        ChildSaveStrategy::Array => ChildSaveStrategy::Array,
        ChildSaveStrategy::ReverseArray => ChildSaveStrategy::ReverseArray,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codecs::competitive_impact::BufferedNormsLookup;
    use crate::document::{IndexOptions, TermOffset};
    use crate::store::memory::MemoryIndexOutput;
    use crate::store::{MemoryDirectory, SharedDirectory};
    use crate::util::byte_block_pool::ByteBlockPool;

    fn test_directory() -> SharedDirectory {
        MemoryDirectory::create()
    }

    /// Test helper: builds terms and postings using FreqProxTermsWriterPerField.
    ///
    /// Unlike the old PerFieldPostings-based helper, this requires terms to be
    /// added in document order (all tokens for doc 0 before doc 1, etc.) because
    /// FreqProxTermsWriterPerField uses a shared last_doc_id assertion.
    struct TestTerms {
        writer: FreqProxTermsWriterPerField,
        term_pool: ByteBlockPool,
        terms_hash: TermsHash,
    }

    impl TestTerms {
        fn new(has_positions: bool) -> Self {
            let opts = if has_positions {
                IndexOptions::DocsAndFreqsAndPositions
            } else {
                IndexOptions::DocsAndFreqs
            };
            let term_pool = ByteBlockPool::new(32 * 1024);
            Self {
                writer: FreqProxTermsWriterPerField::new("test".to_string(), opts),
                term_pool,
                terms_hash: TermsHash::new(),
            }
        }

        /// Add a single term occurrence at the given doc/position.
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

        fn field_terms(&self, name: &str) -> BufferedFieldTerms<'_> {
            BufferedFieldTerms::new(&self.writer, &self.term_pool, &self.terms_hash, name, 0)
        }
    }

    #[test]
    fn test_stats_writer_singletons() {
        let mut buf = Vec::new();
        let mut sw = StatsWriter::new(&mut buf, true);
        sw.add(1, 1).unwrap(); // singleton
        sw.add(1, 1).unwrap(); // singleton
        sw.add(1, 1).unwrap(); // singleton
        sw.finish().unwrap();

        // Should be run-length encoded: ((3-1) << 1) | 1 = 5
        assert_eq!(buf, vec![5]);
    }

    #[test]
    fn test_stats_writer_mixed() {
        let mut buf = Vec::new();
        let mut sw = StatsWriter::new(&mut buf, true);
        sw.add(1, 1).unwrap(); // singleton
        sw.add(3, 5).unwrap(); // non-singleton: df=3, ttf=5
        sw.finish().unwrap();

        // First: singleton count=1 → ((1-1)<<1)|1 = 1
        // Then: df=3 → 3<<1 = 6, ttf-df = 5-3 = 2
        assert_eq!(buf, vec![1, 6, 2]);
    }

    #[test]
    fn test_trie_builder_single_key() {
        let output = TrieOutput {
            fp: 42,
            has_terms: true,
            floor_data: None,
        };
        let trie = TrieBuilder::from_bytes(b"abc", output);

        assert_none!(trie.root.output);
        assert_len_eq_x!(&trie.root.children, 1);
        assert_eq!(trie.root.children[0].label, b'a');
    }

    #[test]
    fn test_trie_builder_empty_key() {
        let output = TrieOutput {
            fp: 10,
            has_terms: true,
            floor_data: None,
        };
        let trie = TrieBuilder::from_bytes(b"", output);

        assert_some!(trie.root.output);
        assert_is_empty!(trie.root.children);
    }

    #[test]
    fn test_trie_builder_append() {
        let out1 = TrieOutput {
            fp: 10,
            has_terms: true,
            floor_data: None,
        };
        let out2 = TrieOutput {
            fp: 20,
            has_terms: true,
            floor_data: None,
        };

        let mut trie1 = TrieBuilder::from_bytes(b"abc", out1);
        let trie2 = TrieBuilder::from_bytes(b"abd", out2);

        trie1.append(trie2);

        // Root should have one child 'a'
        assert_len_eq_x!(&trie1.root.children, 1);
        let a_node = &trie1.root.children[0];
        assert_eq!(a_node.label, b'a');
        // 'a' should have one child 'b'
        assert_len_eq_x!(&a_node.children, 1);
        let b_node = &a_node.children[0];
        assert_eq!(b_node.label, b'b');
        // 'b' should have two children 'c' and 'd'
        assert_len_eq_x!(&b_node.children, 2);
        assert_eq!(b_node.children[0].label, b'c');
        assert_eq!(b_node.children[1].label, b'd');
    }

    #[test]
    fn test_trie_save_simple() {
        let output = TrieOutput {
            fp: 42,
            has_terms: true,
            floor_data: None,
        };
        let trie = TrieBuilder::from_bytes(b"a", output);

        let mut meta = Vec::new();
        let mut index = MemoryIndexOutput::new("test.tip".to_string());

        trie.save(&mut VecOutput(&mut meta), &mut index).unwrap();

        // Index should have some bytes
        assert_gt!(index.file_pointer(), 0);
        // Meta should have indexStartFP, rootFP, indexEndFP
        assert_not_empty!(meta);
    }

    #[test]
    fn test_trie_save_root_fp_nonzero_with_children() {
        // Regression test: root_fp must not be 0 when the trie has child nodes.
        // Previously root_fp was always 0 (placeholder), causing Java to read
        // the first child instead of the root.
        let output_a = TrieOutput {
            fp: 10,
            has_terms: true,
            floor_data: None,
        };
        let output_b = TrieOutput {
            fp: 20,
            has_terms: true,
            floor_data: None,
        };
        let mut trie = TrieBuilder::from_bytes(b"a", output_a);
        trie.append(TrieBuilder::from_bytes(b"b", output_b));

        let mut meta = Vec::new();
        let mut index = MemoryIndexOutput::new("test.tip".to_string());

        trie.save(&mut VecOutput(&mut meta), &mut index).unwrap();

        // Parse meta: [indexStartFP (vlong), rootFP (vlong), indexEndFP (vlong)]
        let mut pos = 0;
        let (index_start_fp, n) = read_vlong_at(&meta, pos);
        pos += n;
        let (root_fp, n) = read_vlong_at(&meta, pos);
        pos += n;
        let (index_end_fp, _) = read_vlong_at(&meta, pos);

        assert_eq!(
            index_start_fp, 0,
            "indexStartFP should be 0 (start of .tip)"
        );
        assert_gt!(
            root_fp,
            0,
            "rootFP must be > 0 for a trie with children (got {root_fp})"
        );
        assert_gt!(
            index_end_fp,
            root_fp,
            "indexEndFP ({index_end_fp}) should be > rootFP ({root_fp})"
        );
    }

    /// Read a VLong from a byte slice at a given position. Returns (value, bytes_consumed).
    fn read_vlong_at(data: &[u8], pos: usize) -> (i64, usize) {
        let mut result: i64 = 0;
        let mut shift = 0;
        let mut i = pos;
        loop {
            let b = data[i] as i64;
            i += 1;
            result |= (b & 0x7F) << shift;
            if (b & 0x80) == 0 {
                break;
            }
            shift += 7;
        }
        (result, i - pos)
    }

    #[test]
    fn test_bytes_required_vlong() {
        assert_eq!(bytes_required_vlong(0), 1);
        assert_eq!(bytes_required_vlong(1), 1);
        assert_eq!(bytes_required_vlong(255), 1);
        assert_eq!(bytes_required_vlong(256), 2);
        assert_eq!(bytes_required_vlong(0xFFFF), 2);
        assert_eq!(bytes_required_vlong(0x10000), 3);
    }

    #[test]
    fn test_write_field_simple() {
        let mut tt = TestTerms::new(false);
        // Doc 0: apple x2, banana x1
        tt.add("apple", 0, 0);
        tt.add("apple", 0, 0);
        tt.add("banana", 0, 0);
        // Doc 1: banana x3
        tt.add("banana", 1, 0);
        tt.add("banana", 1, 0);
        tt.add("banana", 1, 0);
        // Doc 2: cherry x1
        tt.add("cherry", 2, 0);
        tt.finalize();

        let id = [0u8; 16];
        let dir = test_directory();
        let mut btw = BlockTreeTermsWriter::new(&dir, "_0", "", &id, IndexOptions::Docs).unwrap();
        btw.write_field(&tt.field_terms("test"), &BufferedNormsLookup::no_norms())
            .unwrap();
        let names = btw.finish().unwrap();

        assert_ge!(
            names.len(),
            4,
            "expected at least 4 files, got {}",
            names.len()
        );

        for ext in &[".tim", ".tip", ".tmd", ".doc", ".psm"] {
            assert!(
                names.iter().any(|n| n.ends_with(ext)),
                "missing {ext}: {names:?}"
            );
        }

        for name in &names {
            let data = dir.read_file(name).unwrap();
            assert_not_empty!(data, "file {name} is empty");
        }
    }

    #[test]
    fn test_write_field_with_positions() {
        let mut tt = TestTerms::new(true);
        // Doc 0: hello@0, world@1, hello@5
        tt.add("hello", 0, 0);
        tt.add("world", 0, 1);
        tt.add("hello", 0, 5);
        // Doc 1: hello@3
        tt.add("hello", 1, 3);
        tt.finalize();

        let id = [0u8; 16];
        let dir = test_directory();
        let mut btw =
            BlockTreeTermsWriter::new(&dir, "_0", "", &id, IndexOptions::DocsAndFreqsAndPositions)
                .unwrap();
        btw.write_field(
            &tt.field_terms("contents"),
            &BufferedNormsLookup::no_norms(),
        )
        .unwrap();
        let names = btw.finish().unwrap();

        assert!(
            names.iter().any(|n| n.ends_with(".pos")),
            "missing .pos: {names:?}"
        );
    }

    #[test]
    fn test_push_term_many_terms_no_overflow() {
        let mut tt = TestTerms::new(true);
        // All in doc 0 — positions just need to be distinct per term
        for i in 0..30 {
            tt.add(&format!("aaa_{i:02}"), 0, i);
        }
        for i in 0..30 {
            tt.add(&format!("bbb_{i:02}"), 0, 30 + i);
        }
        for i in 0..30 {
            tt.add(&format!("ccc_{i:02}"), 0, 60 + i);
        }
        tt.finalize();

        let id = [0u8; 16];
        let dir = test_directory();
        let mut btw =
            BlockTreeTermsWriter::new(&dir, "_0", "", &id, IndexOptions::DocsAndFreqsAndPositions)
                .unwrap();
        btw.write_field(
            &tt.field_terms("contents"),
            &BufferedNormsLookup::no_norms(),
        )
        .unwrap();
        let names = btw.finish().unwrap();

        assert!(
            names.iter().any(|n| n.ends_with(".tim")),
            "missing .tim: {names:?}"
        );
    }

    #[test]
    fn test_doc_count_computed_correctly() {
        let mut tt = TestTerms::new(false);
        // Doc 0: alpha
        tt.add("alpha", 0, 0);
        // Doc 1: alpha, beta
        tt.add("alpha", 1, 0);
        tt.add("beta", 1, 0);
        // Doc 2: beta
        tt.add("beta", 2, 0);
        tt.finalize();

        let id = [0u8; 16];
        let dir = test_directory();
        let mut btw = BlockTreeTermsWriter::new(&dir, "_0", "", &id, IndexOptions::Docs).unwrap();
        btw.write_field(&tt.field_terms("test"), &BufferedNormsLookup::no_norms())
            .unwrap();
        let names = btw.finish().unwrap();

        let tmd_name = names.iter().find(|n| n.ends_with(".tmd")).unwrap();
        let tmd_bytes = dir.read_file(tmd_name).unwrap();

        let meta_hdr_len = codec_headers::index_header_length(TERMS_META_CODEC_NAME, "");
        let terms_hdr_len = codec_headers::index_header_length(TERMS_CODEC, "");
        let mut pos = meta_hdr_len + terms_hdr_len;
        let (_, n) = read_vint(&tmd_bytes[pos..]);
        pos += n;
        let (num_fields, n) = read_vint(&tmd_bytes[pos..]);
        assert_eq!(num_fields, 1);
        pos += n;
        let (field_num, n) = read_vint(&tmd_bytes[pos..]);
        assert_eq!(field_num, 0);
        pos += n;
        let (num_terms, n) = read_vlong(&tmd_bytes[pos..]);
        assert_eq!(num_terms, 2);
        pos += n;
        let (sttf, n) = read_vlong(&tmd_bytes[pos..]);
        assert_eq!(sttf, 4);
        pos += n;
        let (sdf, n) = read_vlong(&tmd_bytes[pos..]);
        assert_eq!(sdf, 4);
        pos += n;
        let (doc_count, _) = read_vint(&tmd_bytes[pos..]);
        assert_eq!(doc_count, 3, "doc_count should be 3 (unique docs: 0, 1, 2)");
    }

    #[test]
    fn test_min_max_terms_written_to_tmd() {
        let mut tt = TestTerms::new(false);
        // Doc 0: apple x2, banana x1
        tt.add("apple", 0, 0);
        tt.add("apple", 0, 0);
        tt.add("banana", 0, 0);
        // Doc 1: banana x3
        tt.add("banana", 1, 0);
        tt.add("banana", 1, 0);
        tt.add("banana", 1, 0);
        // Doc 2: cherry x1
        tt.add("cherry", 2, 0);
        tt.finalize();

        let id = [0u8; 16];
        let dir = test_directory();
        let mut btw = BlockTreeTermsWriter::new(&dir, "_0", "", &id, IndexOptions::Docs).unwrap();
        btw.write_field(&tt.field_terms("test"), &BufferedNormsLookup::no_norms())
            .unwrap();
        let names = btw.finish().unwrap();

        let tmd_name = names.iter().find(|n| n.ends_with(".tmd")).unwrap();
        let tmd_bytes = dir.read_file(tmd_name).unwrap();

        let meta_hdr_len = codec_headers::index_header_length(TERMS_META_CODEC_NAME, "");
        let terms_hdr_len = codec_headers::index_header_length(TERMS_CODEC, "");
        let mut pos = meta_hdr_len + terms_hdr_len;
        let (_, n) = read_vint(&tmd_bytes[pos..]);
        pos += n;
        let (num_fields, n) = read_vint(&tmd_bytes[pos..]);
        assert_eq!(num_fields, 1);
        pos += n;
        let (_, n) = read_vint(&tmd_bytes[pos..]);
        pos += n;
        let (_, n) = read_vlong(&tmd_bytes[pos..]);
        pos += n;
        let (_, n) = read_vlong(&tmd_bytes[pos..]);
        pos += n;
        let (_, n) = read_vlong(&tmd_bytes[pos..]);
        pos += n;
        let (_, n) = read_vint(&tmd_bytes[pos..]);
        pos += n;
        let (min_len, n) = read_vint(&tmd_bytes[pos..]);
        pos += n;
        let min_term = &tmd_bytes[pos..pos + min_len as usize];
        pos += min_len as usize;
        let (max_len, n) = read_vint(&tmd_bytes[pos..]);
        pos += n;
        let max_term = &tmd_bytes[pos..pos + max_len as usize];

        assert_eq!(str::from_utf8(min_term).unwrap(), "apple");
        assert_eq!(str::from_utf8(max_term).unwrap(), "cherry");
    }

    fn read_vint(bytes: &[u8]) -> (i32, usize) {
        let mut result = 0i32;
        let mut shift = 0;
        let mut pos = 0;
        loop {
            let b = bytes[pos] as i32;
            pos += 1;
            result |= (b & 0x7F) << shift;
            if b & 0x80 == 0 {
                break;
            }
            shift += 7;
        }
        (result, pos)
    }

    fn read_vlong(bytes: &[u8]) -> (i64, usize) {
        let mut result = 0i64;
        let mut shift = 0;
        let mut pos = 0;
        loop {
            let b = bytes[pos] as i64;
            pos += 1;
            result |= (b & 0x7F) << shift;
            if b & 0x80 == 0 {
                break;
            }
            shift += 7;
        }
        (result, pos)
    }

    /// Regression test: write_blocks must not corrupt PendingBlock prefix data.
    #[test]
    fn test_pending_block_prefix_preserved_after_write_blocks() {
        let mut tt = TestTerms::new(true);
        let mut p = 0i32;

        // All in doc 0 — different terms at different positions
        for group in b'a'..=b'z' {
            for i in 0..30 {
                tt.add(&format!("a{}{i:02}", group as char), 0, p);
                p += 1;
            }
        }
        for i in 0..30 {
            tt.add(&format!("b_{i:02}"), 0, p + i);
        }
        tt.finalize();

        let id = [0u8; 16];
        let dir = test_directory();
        let mut btw =
            BlockTreeTermsWriter::new(&dir, "_0", "", &id, IndexOptions::DocsAndFreqsAndPositions)
                .unwrap();
        btw.write_field(
            &tt.field_terms("contents"),
            &BufferedNormsLookup::no_norms(),
        )
        .unwrap();
        let names = btw.finish().unwrap();

        assert!(
            names.iter().any(|n| n.ends_with(".tim")),
            "missing .tim: {names:?}"
        );
    }

    /// Regression test: Java's TrieBuilder unconditionally includes
    /// `(encodedOutputFpBytes - 1) << 5` in the single-child node header, even
    /// when the node has no output (encodedOutputFpBytes=0). The integer overflow
    /// `(0 - 1) << 5 = 0xE0` leaks into the high bits. The reader ignores these
    /// bits for SIGN_SINGLE_CHILD_WITHOUT_OUTPUT, but we must write them to be
    /// byte-identical with Java.
    #[test]
    fn test_trie_single_child_without_output_header_matches_java() {
        // Key "ab" creates: root(no output) -> 'a'(no output) -> 'b'(output)
        // Both root and 'a' are single-child-without-output nodes.
        let output = TrieOutput {
            fp: 42,
            has_terms: true,
            floor_data: None,
        };
        let trie = TrieBuilder::from_bytes(b"ab", output);

        let mut meta = Vec::new();
        let mut index = MemoryIndexOutput::new("test.tip".to_string());
        trie.save(&mut VecOutput(&mut meta), &mut index).unwrap();

        let bytes = index.bytes();

        // Post-order serialization:
        //   offset 0: leaf 'b' — header=0x20 (SIGN_NO_CHILDREN | LEAF_NODE_HAS_TERMS), fp=42
        //   offset 2: node 'a' — SIGN_SINGLE_CHILD_WITHOUT_OUTPUT header, label='b', deltaFP
        //   offset 5: root    — SIGN_SINGLE_CHILD_WITHOUT_OUTPUT header, label='a', deltaFP

        // Verify the single-child-without-output headers have 0xE0 high bits set
        let a_header = bytes[2];
        let root_header = bytes[5];

        assert_eq!(
            a_header & 0x03,
            SIGN_SINGLE_CHILD_WITHOUT_OUTPUT,
            "node 'a' should have SIGN_SINGLE_CHILD_WITHOUT_OUTPUT"
        );
        assert_eq!(
            a_header & 0xE0,
            0xE0,
            "node 'a' header high bits should be 0xE0 to match Java (got {:#04x})",
            a_header
        );

        assert_eq!(
            root_header & 0x03,
            SIGN_SINGLE_CHILD_WITHOUT_OUTPUT,
            "root should have SIGN_SINGLE_CHILD_WITHOUT_OUTPUT"
        );
        assert_eq!(
            root_header & 0xE0,
            0xE0,
            "root header high bits should be 0xE0 to match Java (got {:#04x})",
            root_header
        );
    }

    #[test]
    fn test_child_save_strategy_bits() {
        let strategy = ChildSaveStrategy::Bits;
        // Labels 'a' (97) to 'f' (102) = range 6, needs 1 byte
        assert_eq!(strategy.need_bytes(97, 102, 6), 1);
        // Labels 'a' (97) to 'z' (122) = range 26, needs 4 bytes
        assert_eq!(strategy.need_bytes(97, 122, 26), 4);
    }
}
