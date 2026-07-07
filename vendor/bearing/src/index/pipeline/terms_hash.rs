// SPDX-License-Identifier: Apache-2.0
//! Shared term hashing infrastructure for per-field term accumulation.
//!
//! [`TermsHashPerFieldTrait`] matches Java's abstract `TermsHashPerField`
//! class. It provides default implementations of `add()`,
//! `init_stream_slices()`, and `position_stream_slice()` that handle term
//! interning, stream allocation, and cursor positioning. Implementors supply
//! `new_term()`/`add_term()` callbacks for encoding postings data.
//!
//! [`TermsHashPerField`] is the base struct holding shared state (term hash,
//! stream cursors, write methods). It is embedded in each implementor via
//! the `base()`/`base_mut()` accessors.
//!
//! Implementors:
//! - [`FreqProxTermsWriterPerField`] — doc/freq/position/offset encoding

use std::fmt;
use std::io;
use std::mem;

use crate::codecs::fields_producer::{NO_MORE_DOCS, PostingsEnumProducer};
use crate::document::{IndexOptions, TermOffset};
use crate::util::byte_block_pool::{ByteBlockPool, ByteSlicePool, FIRST_LEVEL_SIZE};
use crate::util::bytes_ref_hash::BytesRefHash;

// ---------------------------------------------------------------------------
// ParallelPostingsArray / FreqProxPostingsArray — per-term posting metadata
// ---------------------------------------------------------------------------

/// Bytes per posting in the base array (3 ints = 12 bytes).
pub(crate) const BYTES_PER_POSTING: usize = 3 * mem::size_of::<i32>();

/// Computes a grow size matching Java's `ArrayUtil.oversize`.
pub(crate) fn oversize(min_size: usize, bytes_per_posting: usize) -> usize {
    let extra = min_size >> 3;
    let new_size = min_size + extra;
    let remainder = new_size % bytes_per_posting;
    if remainder != 0 {
        new_size + bytes_per_posting - remainder
    } else {
        new_size
    }
}

/// Base struct-of-arrays for per-term posting metadata.
///
/// Each array is indexed by term ID. The arrays grow together via [`grow`](Self::grow).
#[derive(Debug)]
pub(crate) struct ParallelPostingsArray {
    /// Maps term ID to the term's text start in the `BytesRefHash` pool.
    pub text_starts: Vec<i32>,
    /// Maps term ID to the current stream address offset.
    pub address_offset: Vec<i32>,
    /// Maps term ID to the stream start offset in the byte pool.
    pub byte_starts: Vec<i32>,
}

impl ParallelPostingsArray {
    /// Creates a new array with the given initial capacity.
    pub(crate) fn new(size: usize) -> Self {
        Self {
            text_starts: vec![0; size],
            address_offset: vec![0; size],
            byte_starts: vec![0; size],
        }
    }

    /// Returns the current capacity.
    pub(crate) fn size(&self) -> usize {
        self.text_starts.len()
    }

    /// Returns bytes per posting (for grow size calculation).
    pub(crate) fn bytes_per_posting(&self) -> usize {
        BYTES_PER_POSTING
    }

    /// Grows the arrays to accommodate at least one more entry.
    /// Returns a new array with data copied from `self`.
    #[expect(dead_code)]
    pub(crate) fn grow(&self) -> Self {
        let new_size = oversize(self.size() + 1, self.bytes_per_posting());
        let mut new_array = Self::new(new_size);
        self.copy_to(&mut new_array, self.size());
        new_array
    }

    /// Copies `num_to_copy` elements from `self` into `to_array`.
    pub(crate) fn copy_to(&self, to_array: &mut ParallelPostingsArray, num_to_copy: usize) {
        to_array.text_starts[..num_to_copy].copy_from_slice(&self.text_starts[..num_to_copy]);
        to_array.address_offset[..num_to_copy].copy_from_slice(&self.address_offset[..num_to_copy]);
        to_array.byte_starts[..num_to_copy].copy_from_slice(&self.byte_starts[..num_to_copy]);
    }
}

/// Extended postings array with frequency and proximity fields.
///
/// Adds per-term tracking for document IDs, frequencies, positions,
/// and offsets on top of the base [`ParallelPostingsArray`].
#[derive(Debug)]
pub(crate) struct FreqProxPostingsArray {
    /// Base arrays (text starts, address offsets, byte starts).
    pub base: ParallelPostingsArray,
    /// Term frequency in the current document (only if `has_freq`).
    pub term_freqs: Option<Vec<i32>>,
    /// Last doc ID where each term occurred.
    pub last_doc_ids: Vec<i32>,
    /// Encoded doc code for the prior document.
    pub last_doc_codes: Vec<i32>,
    /// Last position where each term occurred (only if `has_prox`).
    pub last_positions: Option<Vec<i32>>,
    /// Last end offset where each term occurred (only if `has_offsets`).
    pub last_offsets: Option<Vec<i32>>,
}

impl FreqProxPostingsArray {
    /// Creates a new array with the given capacity and feature flags.
    pub(crate) fn new(
        size: usize,
        write_freqs: bool,
        write_prox: bool,
        write_offsets: bool,
    ) -> Self {
        let term_freqs = if write_freqs {
            Some(vec![0; size])
        } else {
            None
        };
        let last_positions = if write_prox {
            Some(vec![0; size])
        } else {
            assert!(!write_offsets);
            None
        };
        let last_offsets = if write_offsets {
            Some(vec![0; size])
        } else {
            None
        };
        Self {
            base: ParallelPostingsArray::new(size),
            term_freqs,
            last_doc_ids: vec![0; size],
            last_doc_codes: vec![0; size],
            last_positions,
            last_offsets,
        }
    }

    /// Returns the current capacity.
    pub(crate) fn size(&self) -> usize {
        self.base.size()
    }

    /// Returns bytes per posting (base + extended fields).
    pub(crate) fn bytes_per_posting(&self) -> usize {
        let mut bytes = self.base.bytes_per_posting();
        // lastDocIDs + lastDocCodes always present
        bytes += 2 * mem::size_of::<i32>();
        if self.term_freqs.is_some() {
            bytes += mem::size_of::<i32>();
        }
        if self.last_positions.is_some() {
            bytes += mem::size_of::<i32>();
        }
        if self.last_offsets.is_some() {
            bytes += mem::size_of::<i32>();
        }
        bytes
    }

    /// Grows the arrays to accommodate at least one more entry.
    pub(crate) fn grow(&self) -> Self {
        let new_size = oversize(self.size() + 1, self.bytes_per_posting());
        let mut new_array = Self::new(
            new_size,
            self.term_freqs.is_some(),
            self.last_positions.is_some(),
            self.last_offsets.is_some(),
        );
        self.copy_to(&mut new_array, self.size());
        new_array
    }

    /// Copies `num_to_copy` elements from `self` into `to_array`.
    pub(crate) fn copy_to(&self, to_array: &mut FreqProxPostingsArray, num_to_copy: usize) {
        self.base.copy_to(&mut to_array.base, num_to_copy);

        to_array.last_doc_ids[..num_to_copy].copy_from_slice(&self.last_doc_ids[..num_to_copy]);
        to_array.last_doc_codes[..num_to_copy].copy_from_slice(&self.last_doc_codes[..num_to_copy]);
        if let (Some(from), Some(to)) = (&self.last_positions, &mut to_array.last_positions) {
            to[..num_to_copy].copy_from_slice(&from[..num_to_copy]);
        }
        if let (Some(from), Some(to)) = (&self.last_offsets, &mut to_array.last_offsets) {
            to[..num_to_copy].copy_from_slice(&from[..num_to_copy]);
        }
        if let (Some(from), Some(to)) = (&self.term_freqs, &mut to_array.term_freqs) {
            to[..num_to_copy].copy_from_slice(&from[..num_to_copy]);
        }
    }
}

// ---------------------------------------------------------------------------
// TermsHash / TermsHashPerField
// ---------------------------------------------------------------------------

/// Initial capacity for the BytesRefHash.
const HASH_INIT_SIZE: usize = 4;

/// Shared pool storage for all TermsHashPerField instances in a segment.
///
/// Owns the int pool (stream address table) and `ByteBlockPool` that all
/// per-field writers share for posting stream data. In Java, this is
/// `TermsHash` which `FreqProxTermsWriter` extends.
pub(crate) struct TermsHash {
    /// Flat table of byte-pool write offsets, indexed by stream address.
    pub(crate) int_pool: Vec<i32>,
    pub(crate) byte_pool: ByteBlockPool,
}

impl TermsHash {
    /// Creates a new `TermsHash` with initialized pools.
    pub(crate) fn new() -> Self {
        Self {
            int_pool: Vec::with_capacity(8192),
            byte_pool: ByteBlockPool::new(32 * 1024),
        }
    }

    /// Resets both pools for reuse. Releases backing storage if capacity
    /// exceeds 64 KB to avoid pinning memory from worst-case documents.
    pub(crate) fn reset(&mut self) {
        const BYTE_POOL_SHRINK_THRESHOLD: usize = 64 * 1024;
        const INT_POOL_SHRINK_THRESHOLD: usize = BYTE_POOL_SHRINK_THRESHOLD / mem::size_of::<i32>();

        if self.byte_pool.data.capacity() > BYTE_POOL_SHRINK_THRESHOLD {
            self.byte_pool.data = Vec::with_capacity(BYTE_POOL_SHRINK_THRESHOLD);
        } else {
            self.byte_pool.reset();
        }
        if self.int_pool.capacity() > INT_POOL_SHRINK_THRESHOLD {
            self.int_pool = Vec::with_capacity(INT_POOL_SHRINK_THRESHOLD);
        } else {
            self.int_pool.clear();
        }
    }
}

impl Default for TermsHash {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for TermsHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TermsHash")
            .field("int_pool_len", &self.int_pool.len())
            .field("byte_pool_len", &self.byte_pool.data.len())
            .finish()
    }
}

impl mem_dbg::MemSize for TermsHash {
    fn mem_size_rec(
        &self,
        flags: mem_dbg::SizeFlags,
        refs: &mut mem_dbg::HashMap<usize, usize>,
    ) -> usize {
        self.int_pool.len() * mem::size_of::<i32>() + self.byte_pool.mem_size_rec(flags, refs)
    }
}

/// Per-field term processing for the inverted index.
///
/// Deduplicates terms, allocates byte stream slices, and provides write methods.
/// Stream addresses are stored in `int_pool` entries. Each term gets
/// `stream_count` consecutive int slots holding the current write address for
/// each stream.
///
/// Pools are owned by [`TermsHash`] and passed to methods that need them.
pub(crate) struct TermsHashPerField {
    /// Direct offset into the flat `int_pool` for the current term's stream addresses.
    pub(crate) stream_address_offset: usize,
    pub(crate) stream_count: usize,
    field_name: String,
    pub(crate) bytes_hash: BytesRefHash,
    sorted_term_ids: Option<Vec<i32>>,
    pub(crate) last_doc_id: i32, // assertion-only
}

impl TermsHashPerField {
    /// Creates a new `TermsHashPerField`.
    ///
    /// `stream_count` is the number of byte streams per term: 1 for doc(+freq),
    /// 2 when positions/offsets are also indexed.
    pub(crate) fn new(
        stream_count: usize,
        field_name: String,
        index_options: IndexOptions,
    ) -> Self {
        assert!(index_options != IndexOptions::None);

        let bytes_hash = BytesRefHash::new(HASH_INIT_SIZE);

        Self {
            stream_address_offset: 0,
            stream_count,
            field_name,
            bytes_hash,
            sorted_term_ids: None,
            last_doc_id: -1,
        }
    }

    /// Clears the term hash and resets state for reuse.
    pub(crate) fn reset(&mut self) {
        self.bytes_hash.clear();
        self.sorted_term_ids = None;
    }

    /// Collapses the hash table and sorts term IDs lexicographically.
    ///
    /// Must not be called twice without a [`reset`](Self::reset) in between.
    pub(crate) fn sort_terms(&mut self, byte_pool: &ByteBlockPool) {
        assert!(self.sorted_term_ids.is_none());
        self.sorted_term_ids = Some(self.bytes_hash.sort(byte_pool));
    }

    /// Returns the sorted term IDs. [`sort_terms`](Self::sort_terms) must be
    /// called first.
    pub(crate) fn sorted_term_ids(&self) -> &[i32] {
        self.sorted_term_ids
            .as_ref()
            .expect("sort_terms not called")
    }

    /// Returns the number of unique terms.
    pub(crate) fn num_terms(&self) -> usize {
        self.bytes_hash.size()
    }

    /// Returns the field name.
    pub(crate) fn field_name(&self) -> &str {
        &self.field_name
    }

    /// Returns the bytes for a given term ID (from the shared byte pool).
    pub(crate) fn term_bytes<'a>(&self, byte_pool: &'a ByteBlockPool, term_id: usize) -> &'a [u8] {
        self.bytes_hash.get(byte_pool, term_id)
    }

    /// Write a single byte to the given stream for the current term.
    pub(crate) fn write_byte(&mut self, terms_hash: &mut TermsHash, stream: usize, b: u8) {
        let stream_address = self.stream_address_offset + stream;
        let upto = terms_hash.int_pool[stream_address] as usize;
        if terms_hash.byte_pool.data[upto] != 0 {
            // End of slice; allocate a new one
            let new_offset = ByteSlicePool::alloc_slice(&mut terms_hash.byte_pool, upto);
            terms_hash.int_pool[stream_address] = new_offset as i32;
            terms_hash.byte_pool.data[new_offset] = b;
            terms_hash.int_pool[stream_address] += 1;
        } else {
            terms_hash.byte_pool.data[upto] = b;
            terms_hash.int_pool[stream_address] += 1;
        }
    }

    /// Write multiple bytes to the given stream for the current term.
    #[expect(dead_code)]
    pub(crate) fn write_bytes(&mut self, terms_hash: &mut TermsHash, stream: usize, data: &[u8]) {
        let end = data.len();
        let stream_address = self.stream_address_offset + stream;
        let mut upto = terms_hash.int_pool[stream_address] as usize;
        let mut offset = 0;

        // Write into current slice while there's room
        while terms_hash.byte_pool.data[upto] == 0 && offset < end {
            terms_hash.byte_pool.data[upto] = data[offset];
            upto += 1;
            offset += 1;
            terms_hash.int_pool[stream_address] += 1;
        }

        // If we still have data, grow slices as needed
        while offset < end {
            let (new_slice_offset, slice_length) =
                ByteSlicePool::alloc_known_size_slice(&mut terms_hash.byte_pool, upto);
            let write_length = (slice_length - 1).min(end - offset);
            terms_hash.byte_pool.data[new_slice_offset..new_slice_offset + write_length]
                .copy_from_slice(&data[offset..offset + write_length]);
            upto = new_slice_offset + write_length;
            offset += write_length;
            terms_hash.int_pool[stream_address] = upto as i32;
        }
    }

    /// Write a variable-length encoded integer to the given stream.
    pub(crate) fn write_v_int(&mut self, terms_hash: &mut TermsHash, stream: usize, mut i: i32) {
        assert!(stream < self.stream_count);
        while (i & !0x7F) != 0 {
            self.write_byte(terms_hash, stream, ((i & 0x7F) | 0x80) as u8);
            i = ((i as u32) >> 7) as i32;
        }
        self.write_byte(terms_hash, stream, i as u8);
    }
}

impl fmt::Debug for TermsHashPerField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TermsHashPerField")
            .field("field_name", &self.field_name)
            .field("stream_count", &self.stream_count)
            .field("num_terms", &self.bytes_hash.size())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// TermsHashPerFieldTrait
// ---------------------------------------------------------------------------

/// Abstract per-field term processing trait.
///
/// Matches Java's abstract `TermsHashPerField` class. Implementors provide
/// access to their base struct and postings array, plus `new_term`/`add_term`
/// callbacks. Default methods implement the concrete `add()`,
/// `init_stream_slices()`, and `position_stream_slice()` logic from Java.
pub(crate) trait TermsHashPerFieldTrait {
    /// Access the base `TermsHashPerField` (shared fields).
    fn base(&self) -> &TermsHashPerField;

    /// Mutable access to the base `TermsHashPerField`.
    fn base_mut(&mut self) -> &mut TermsHashPerField;

    /// Access the `ParallelPostingsArray` base of the concrete postings array.
    fn postings_array_base(&self) -> &ParallelPostingsArray;

    /// Mutable access to the `ParallelPostingsArray` base.
    fn postings_array_base_mut(&mut self) -> &mut ParallelPostingsArray;

    /// Ensure the postings array can hold `term_id`. Grows if needed.
    fn ensure_postings_capacity(&mut self, term_id: usize);

    /// Called when a term is seen for the first time.
    fn new_term(&mut self, terms_hash: &mut TermsHash, term_id: usize, doc_id: i32);

    /// Called when a previously seen term is seen again.
    fn add_term(&mut self, terms_hash: &mut TermsHash, term_id: usize, doc_id: i32);

    /// Primary entry point: intern term bytes, allocate/position streams,
    /// dispatch to `new_term`/`add_term`.
    ///
    /// `term_byte_pool` is the shared pool for term text (from the accumulator).
    /// `terms_hash` holds the per-consumer stream pools for postings data.
    ///
    /// Returns the (positive) term ID.
    fn add(
        &mut self,
        term_byte_pool: &mut ByteBlockPool,
        terms_hash: &mut TermsHash,
        term_bytes: &[u8],
        doc_id: i32,
    ) -> usize {
        {
            let base = self.base_mut();
            debug_assert!(doc_id >= base.last_doc_id);
            base.last_doc_id = doc_id;
        }

        let term_id = self.base_mut().bytes_hash.add(term_byte_pool, term_bytes);

        if term_id >= 0 {
            let tid = term_id as usize;
            self.init_stream_slices(terms_hash, tid, doc_id);
            tid
        } else {
            let tid = ((-term_id) - 1) as usize;
            self.position_stream_slice(terms_hash, tid, doc_id);
            tid
        }
    }

    /// Secondary entry point for pre-interned terms (term vectors).
    fn add_by_text_start(&mut self, terms_hash: &mut TermsHash, text_start: i32, doc_id: i32) {
        let term_id = self.base_mut().bytes_hash.add_by_pool_offset(text_start);

        if term_id >= 0 {
            let tid = term_id as usize;
            self.init_stream_slices(terms_hash, tid, doc_id);
        } else {
            let tid = ((-term_id) - 1) as usize;
            self.position_stream_slice(terms_hash, tid, doc_id);
        }
    }

    /// Allocate stream slices for a new term.
    fn init_stream_slices(&mut self, terms_hash: &mut TermsHash, term_id: usize, doc_id: i32) {
        // Ensure postings array has capacity for this term
        self.ensure_postings_capacity(term_id);

        // Set text_starts
        let byte_start = self.base().bytes_hash.byte_start(term_id);
        self.postings_array_base_mut().text_starts[term_id] = byte_start;

        let stream_count = self.base().stream_count;

        let stream_address_offset = terms_hash.int_pool.len();
        terms_hash
            .int_pool
            .resize(stream_address_offset + stream_count, 0);

        self.base_mut().stream_address_offset = stream_address_offset;

        self.postings_array_base_mut().address_offset[term_id] = stream_address_offset as i32;

        for i in 0..stream_count {
            let upto = ByteSlicePool::new_slice(&mut terms_hash.byte_pool, FIRST_LEVEL_SIZE);
            terms_hash.int_pool[stream_address_offset + i] = upto as i32;
        }

        let byte_starts = terms_hash.int_pool[stream_address_offset];
        self.postings_array_base_mut().byte_starts[term_id] = byte_starts;

        self.new_term(terms_hash, term_id, doc_id);
    }

    /// Position stream cursors for an existing term, then dispatch to `add_term`.
    fn position_stream_slice(&mut self, terms_hash: &mut TermsHash, term_id: usize, doc_id: i32) {
        let int_start = self.postings_array_base().address_offset[term_id] as usize;
        self.base_mut().stream_address_offset = int_start;

        self.add_term(terms_hash, term_id, doc_id);
    }
}

// ---------------------------------------------------------------------------
// FreqProxTermsWriterPerField
// ---------------------------------------------------------------------------

/// Concrete per-field posting writer for frequency and proximity data.
///
/// Extends `TermsHashPerField` via composition. Implements the `newTerm` and
/// `addTerm` logic that encodes doc IDs, frequencies, positions, and offsets
/// into the byte pool streams.
/// One document's entry in a decoded term's posting list.
///
/// Positions are stored in a separate flat buffer; `pos_start` and `pos_count`
/// index into that buffer.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DecodedDoc {
    pub doc_id: i32,
    pub freq: i32,
    pub pos_start: u32,
}

/// Reusable buffers for decoded term postings.
///
/// Avoids per-term allocation by reusing flat buffers across calls to
/// `decode_term_into`. Each [`DecodedDoc`] entry indexes into the flat
/// `positions` buffer.
pub(crate) struct DecodedPostings {
    /// Per-document metadata.
    pub docs: Vec<DecodedDoc>,
    /// Flat positions buffer; docs reference ranges via `pos_start`/`pos_count`.
    pub positions: Vec<i32>,
    /// Flat offsets buffer, parallel to `positions`. Empty when offsets not indexed.
    pub offsets: Vec<TermOffset>,
}

/// Buffer capacity threshold in bytes. Buffers larger than this are replaced
/// to allow the OS to reclaim the pages.
const DECODED_SHRINK_THRESHOLD: usize = 4096;

impl DecodedPostings {
    /// Creates new empty buffers.
    pub fn new() -> Self {
        Self {
            docs: Vec::new(),
            positions: Vec::new(),
            offsets: Vec::new(),
        }
    }

    /// Clears for reuse. Shrinks buffers that have grown beyond the threshold.
    pub fn clear(&mut self) {
        self.docs.clear();
        if self.docs.capacity() * mem::size_of::<DecodedDoc>() > DECODED_SHRINK_THRESHOLD {
            self.docs = Vec::new();
        }
        self.positions.clear();
        if self.positions.capacity() * mem::size_of::<i32>() > DECODED_SHRINK_THRESHOLD {
            self.positions = Vec::new();
        }
        self.offsets.clear();
        if self.offsets.capacity() * mem::size_of::<TermOffset>() > DECODED_SHRINK_THRESHOLD {
            self.offsets = Vec::new();
        }
    }
}

/// [`PostingsEnumProducer`] that owns a [`DecodedPostings`] and iterates
/// over its contents.
pub(crate) struct BufferedPostingsEnum {
    decoded: DecodedPostings,
    doc_freq: i32,
    total_term_freq: i64,
    doc_idx: usize,
    pos_idx: usize,
}

impl BufferedPostingsEnum {
    /// Creates a new enum from owned decoded postings data.
    pub fn new(decoded: DecodedPostings, has_freqs: bool) -> Self {
        let doc_freq = decoded.docs.len() as i32;
        let total_term_freq = if has_freqs {
            decoded.docs.iter().map(|d| d.freq as i64).sum()
        } else {
            -1
        };
        Self {
            decoded,
            doc_freq,
            total_term_freq,
            doc_idx: 0,
            pos_idx: 0,
        }
    }

    fn current_doc(&self) -> &DecodedDoc {
        &self.decoded.docs[self.doc_idx - 1]
    }
}

impl PostingsEnumProducer for BufferedPostingsEnum {
    fn doc_freq(&self) -> i32 {
        self.doc_freq
    }

    fn total_term_freq(&self) -> i64 {
        self.total_term_freq
    }

    fn next_doc(&mut self) -> io::Result<i32> {
        if self.doc_idx >= self.decoded.docs.len() {
            return Ok(NO_MORE_DOCS);
        }
        let doc_id = self.decoded.docs[self.doc_idx].doc_id;
        self.pos_idx = 0;
        self.doc_idx += 1;
        Ok(doc_id)
    }

    fn freq(&self) -> i32 {
        self.current_doc().freq
    }

    fn next_position(&mut self) -> io::Result<i32> {
        let doc = self.current_doc();
        let abs_idx = doc.pos_start as usize + self.pos_idx;
        let pos = self.decoded.positions[abs_idx];
        self.pos_idx += 1;
        Ok(pos)
    }

    fn offset(&self) -> Option<TermOffset> {
        if self.decoded.offsets.is_empty() {
            return None;
        }
        let doc = self.current_doc();
        let abs_idx = doc.pos_start as usize + self.pos_idx - 1;
        Some(self.decoded.offsets[abs_idx])
    }

    fn payload(&self) -> Option<&[u8]> {
        None
    }
}

///
/// Stream 0: doc codes and frequencies
/// Stream 1: position codes, offsets, and payloads (when has_prox)
pub(crate) struct FreqProxTermsWriterPerField {
    /// Base term hash functionality (BytesRefHash, stream cursors).
    pub base: TermsHashPerField,
    /// Per-term posting metadata arrays.
    pub postings_array: FreqProxPostingsArray,
    /// What information is indexed for this field.
    pub index_options: IndexOptions,
    /// Whether this field indexes term frequencies.
    pub has_freq: bool,
    /// Whether this field indexes positions.
    pub has_prox: bool,
    /// Whether this field indexes offsets.
    pub has_offsets: bool,
    /// Whether any token had a payload in the current segment.
    pub saw_payloads: bool,
    /// Tracks max term frequency across all terms for the current document.
    pub max_term_frequency: i32,
    /// Tracks unique term count for the current document.
    pub unique_term_count: i32,
    // Current token state — set before calling trait add(), read by new_term/add_term.
    pub(crate) current_position: i32,
    pub(crate) current_offset: TermOffset,
}

impl FreqProxTermsWriterPerField {
    /// Creates a new `FreqProxTermsWriterPerField`.
    pub(crate) fn new(field_name: String, index_options: IndexOptions) -> Self {
        let has_freq = index_options >= IndexOptions::DocsAndFreqs;
        let has_prox = index_options >= IndexOptions::DocsAndFreqsAndPositions;
        let has_offsets = index_options >= IndexOptions::DocsAndFreqsAndPositionsAndOffsets;

        let stream_count = if has_prox { 2 } else { 1 };

        let base = TermsHashPerField::new(stream_count, field_name, index_options);

        let postings_array = FreqProxPostingsArray::new(2, has_freq, has_prox, has_offsets);

        Self {
            base,
            postings_array,
            index_options,
            has_freq,
            has_prox,
            has_offsets,
            saw_payloads: false,
            max_term_frequency: 0,
            unique_term_count: 0,
            current_position: 0,
            current_offset: TermOffset::default(),
        }
    }

    /// Add a term occurrence for the given document.
    ///
    /// `term_byte_pool` is the shared pool for term text (from the accumulator).
    /// `terms_hash` holds the per-consumer stream pools.
    ///
    /// The caller must set `current_position`, `current_start_offset`, and
    /// `current_offset_length` before calling this method.
    pub(crate) fn add(
        &mut self,
        term_byte_pool: &mut ByteBlockPool,
        terms_hash: &mut TermsHash,
        term_bytes: &[u8],
        doc_id: i32,
    ) -> io::Result<usize> {
        let tid = TermsHashPerFieldTrait::add(self, term_byte_pool, terms_hash, term_bytes, doc_id);
        Ok(tid)
    }

    /// Convenience: sets position/offset state and calls `add()`.
    #[cfg(test)]
    pub(crate) fn add_at(
        &mut self,
        term_byte_pool: &mut ByteBlockPool,
        terms_hash: &mut TermsHash,
        term_bytes: &[u8],
        doc_id: i32,
        pos: TermPosition,
    ) -> io::Result<usize> {
        self.current_position = pos.position;
        self.current_offset = pos.offset;
        self.add(term_byte_pool, terms_hash, term_bytes, doc_id)
    }

    /// Resets this per-field state.
    #[expect(dead_code)]
    pub(crate) fn reset(&mut self) {
        self.base.reset();
    }

    /// Finish adding all instances of this field to the current document.
    #[expect(dead_code)]
    pub(crate) fn finish(&self) {
        // No-op — TV chaining is handled at the consumer level.
    }

    /// Returns the number of unique terms.
    pub(crate) fn num_terms(&self) -> usize {
        self.base.num_terms()
    }

    /// Sort terms lexicographically.
    pub(crate) fn sort_terms(&mut self, byte_pool: &ByteBlockPool) {
        self.base.sort_terms(byte_pool);
    }

    /// Returns sorted term IDs.
    pub(crate) fn sorted_term_ids(&self) -> &[i32] {
        self.base.sorted_term_ids()
    }

    /// Returns the bytes for a given term ID.
    pub(crate) fn term_bytes<'a>(&self, byte_pool: &'a ByteBlockPool, term_id: usize) -> &'a [u8] {
        self.base.term_bytes(byte_pool, term_id)
    }

    /// Flushes the pending (last) document for every term into stream 0.
    ///
    /// Must be called before reading back postings data. Writes the
    /// remaining `last_doc_codes` and `term_freqs` for each term.
    pub(crate) fn flush_pending_docs(&mut self, terms_hash: &mut TermsHash) {
        let num_terms = self.base.num_terms();
        for term_id in 0..num_terms {
            // Position the stream cursor for this term
            let int_start = self.postings_array.base.address_offset[term_id] as usize;
            self.base.stream_address_offset = int_start;

            if !self.has_freq {
                // DOCS only: write last doc code
                let code = self.postings_array.last_doc_codes[term_id];
                self.base.write_v_int(terms_hash, 0, code);
            } else {
                let freq = self.postings_array.term_freqs.as_ref().unwrap()[term_id];
                let code = self.postings_array.last_doc_codes[term_id];
                if freq == 1 {
                    self.base.write_v_int(terms_hash, 0, code | 1);
                } else {
                    self.base.write_v_int(terms_hash, 0, code);
                    self.base.write_v_int(terms_hash, 0, freq);
                }
            }
        }
    }

    /// Decodes one term's postings from the shared pools into reusable buffers.
    ///
    /// Reads the Lucene-style `(doc_delta << 1) | freq_is_1` encoding
    /// from stream 0 and `position << 1` from stream 1.
    pub(crate) fn decode_term_into(
        &self,
        terms_hash: &TermsHash,
        term_id: usize,
        buf: &mut DecodedPostings,
    ) -> io::Result<()> {
        use crate::util::byte_block_pool::ByteSliceReader;

        buf.clear();

        let (start, end) = self.get_stream_range(&terms_hash.int_pool, term_id, 0);
        let mut reader = ByteSliceReader::new(&terms_hash.byte_pool, start, end);

        let mut pos_reader = if self.has_prox {
            let (ps, pe) = self.get_stream_range(&terms_hash.int_pool, term_id, 1);
            Some(ByteSliceReader::new(&terms_hash.byte_pool, ps, pe))
        } else {
            None
        };

        let mut last_doc_id = 0;

        while !reader.eof() {
            let code = reader.read_vint()?;
            let (doc_delta, freq);

            if !self.has_freq {
                // DOCS only: code is plain doc delta
                doc_delta = code;
                freq = 1;
            } else {
                // doc_delta << 1 | freq_is_1
                doc_delta = code >> 1;
                if (code & 1) != 0 {
                    freq = 1;
                } else {
                    freq = reader.read_vint()?;
                }
            }

            let doc_id = last_doc_id + doc_delta;
            last_doc_id = doc_id;

            let pos_start = buf.positions.len() as u32;

            if let Some(ref mut pr) = pos_reader {
                let mut last_pos = 0;
                let mut last_start_offset = 0u32;
                for _ in 0..freq {
                    let prox_code = pr.read_vint()?;
                    // proxCode = positionDelta << 1 (no payload support)
                    let pos_delta = prox_code >> 1;
                    let pos = last_pos + pos_delta;
                    buf.positions.push(pos);
                    last_pos = pos;

                    if self.has_offsets {
                        let start_offset_delta = pr.read_vint()? as u32;
                        let length = pr.read_vint()? as u16;
                        let start = last_start_offset + start_offset_delta;
                        buf.offsets.push(TermOffset { start, length });
                        last_start_offset = start;
                    }
                }
            }

            buf.docs.push(DecodedDoc {
                doc_id,
                freq,
                pos_start,
            });
        }

        Ok(())
    }

    /// Returns the stream range for constructing a reader.
    pub(crate) fn get_stream_range(
        &self,
        int_pool: &[i32],
        term_id: usize,
        stream: usize,
    ) -> (usize, usize) {
        assert!(stream < self.base.stream_count);
        let address_offset = self.postings_array.base.address_offset[term_id] as usize;
        let end = int_pool[address_offset + stream] as usize;
        let start =
            self.postings_array.base.byte_starts[term_id] as usize + stream * FIRST_LEVEL_SIZE;
        (start, end)
    }
}

impl fmt::Debug for FreqProxTermsWriterPerField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FreqProxTermsWriterPerField")
            .field("field_name", &self.base.field_name)
            .field("num_terms", &self.base.bytes_hash.size())
            .field("has_freq", &self.has_freq)
            .field("has_prox", &self.has_prox)
            .field("has_offsets", &self.has_offsets)
            .finish()
    }
}

impl mem_dbg::MemSize for FreqProxTermsWriterPerField {
    fn mem_size_rec(
        &self,
        flags: mem_dbg::SizeFlags,
        refs: &mut mem_dbg::HashMap<usize, usize>,
    ) -> usize {
        mem::size_of::<Self>()
            + self.base.bytes_hash.mem_size_rec(flags, refs)
            + self.postings_array.size() * self.postings_array.bytes_per_posting()
    }
}

impl TermsHashPerFieldTrait for FreqProxTermsWriterPerField {
    fn base(&self) -> &TermsHashPerField {
        &self.base
    }

    fn base_mut(&mut self) -> &mut TermsHashPerField {
        &mut self.base
    }

    fn postings_array_base(&self) -> &ParallelPostingsArray {
        &self.postings_array.base
    }

    fn postings_array_base_mut(&mut self) -> &mut ParallelPostingsArray {
        &mut self.postings_array.base
    }

    fn ensure_postings_capacity(&mut self, term_id: usize) {
        while term_id >= self.postings_array.size() {
            let grown = self.postings_array.grow();
            self.postings_array = grown;
        }
    }

    fn new_term(&mut self, terms_hash: &mut TermsHash, term_id: usize, doc_id: i32) {
        let position = self.current_position;
        let start_offset = self.current_offset.start as i32;
        let offset_length = self.current_offset.length;

        let postings = &mut self.postings_array;

        postings.last_doc_ids[term_id] = doc_id;
        if !self.has_freq {
            postings.last_doc_codes[term_id] = doc_id;
            self.max_term_frequency = self.max_term_frequency.max(1);
        } else {
            postings.last_doc_codes[term_id] = doc_id << 1;
            postings.term_freqs.as_mut().unwrap()[term_id] = 1;
            if self.has_prox {
                self.base.write_v_int(terms_hash, 1, position << 1);
                postings.last_positions.as_mut().unwrap()[term_id] = position;
                if self.has_offsets {
                    postings.last_offsets.as_mut().unwrap()[term_id] = 0;
                    self.base.write_v_int(terms_hash, 1, start_offset);
                    self.base.write_v_int(terms_hash, 1, offset_length as i32);
                    postings.last_offsets.as_mut().unwrap()[term_id] = start_offset;
                }
            }
            self.max_term_frequency = self
                .max_term_frequency
                .max(postings.term_freqs.as_ref().unwrap()[term_id]);
        }
        self.unique_term_count += 1;
    }

    fn add_term(&mut self, terms_hash: &mut TermsHash, term_id: usize, doc_id: i32) {
        let position = self.current_position;
        let start_offset = self.current_offset.start as i32;
        let offset_length = self.current_offset.length;

        let postings = &mut self.postings_array;

        if !self.has_freq {
            if doc_id != postings.last_doc_ids[term_id] {
                assert!(doc_id > postings.last_doc_ids[term_id]);
                self.base
                    .write_v_int(terms_hash, 0, postings.last_doc_codes[term_id]);
                postings.last_doc_codes[term_id] = doc_id - postings.last_doc_ids[term_id];
                postings.last_doc_ids[term_id] = doc_id;
                self.unique_term_count += 1;
            }
        } else if doc_id != postings.last_doc_ids[term_id] {
            assert!(doc_id > postings.last_doc_ids[term_id]);
            if postings.term_freqs.as_ref().unwrap()[term_id] == 1 {
                self.base
                    .write_v_int(terms_hash, 0, postings.last_doc_codes[term_id] | 1);
            } else {
                self.base
                    .write_v_int(terms_hash, 0, postings.last_doc_codes[term_id]);
                self.base.write_v_int(
                    terms_hash,
                    0,
                    postings.term_freqs.as_ref().unwrap()[term_id],
                );
            }

            postings.term_freqs.as_mut().unwrap()[term_id] = 1;
            self.max_term_frequency = self.max_term_frequency.max(1);
            postings.last_doc_codes[term_id] = (doc_id - postings.last_doc_ids[term_id]) << 1;
            postings.last_doc_ids[term_id] = doc_id;
            if self.has_prox {
                self.base.write_v_int(terms_hash, 1, position << 1);
                postings.last_positions.as_mut().unwrap()[term_id] = position;
                if self.has_offsets {
                    postings.last_offsets.as_mut().unwrap()[term_id] = 0;
                    self.base.write_v_int(terms_hash, 1, start_offset);
                    self.base.write_v_int(terms_hash, 1, offset_length as i32);
                    postings.last_offsets.as_mut().unwrap()[term_id] = start_offset;
                }
            }
            self.unique_term_count += 1;
        } else {
            // Same document
            let freq = postings.term_freqs.as_mut().unwrap();
            freq[term_id] = freq[term_id]
                .checked_add(1)
                .expect("term frequency overflow");
            self.max_term_frequency = self.max_term_frequency.max(freq[term_id]);
            if self.has_prox {
                let last_pos = postings.last_positions.as_ref().unwrap()[term_id];
                self.base
                    .write_v_int(terms_hash, 1, (position - last_pos) << 1);
                postings.last_positions.as_mut().unwrap()[term_id] = position;
                if self.has_offsets {
                    let last_offset = postings.last_offsets.as_ref().unwrap()[term_id];
                    self.base
                        .write_v_int(terms_hash, 1, start_offset - last_offset);
                    self.base.write_v_int(terms_hash, 1, offset_length as i32);
                    postings.last_offsets.as_mut().unwrap()[term_id] = start_offset;
                }
            }
        }
    }
}

/// Position and character offset for a term occurrence.
#[cfg(test)]
pub(crate) struct TermPosition {
    position: i32,
    offset: TermOffset,
}

#[cfg(test)]
impl TermPosition {
    fn new(position: i32, start: u32, length: u16) -> Self {
        Self {
            position,
            offset: TermOffset { start, length },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::util::byte_block_pool::ByteSliceReader;
    use assertables::*;

    /// Helper to read a VInt from a byte slice reader.
    fn read_vint(reader: &mut ByteSliceReader<'_>) -> i32 {
        reader.read_vint().unwrap()
    }

    fn new_term_pool() -> ByteBlockPool {
        ByteBlockPool::new(32 * 1024)
    }

    #[test]
    fn test_single_term_single_doc() {
        let mut term_pool = new_term_pool();
        let mut th = TermsHash::new();
        let mut field =
            FreqProxTermsWriterPerField::new("body".to_string(), IndexOptions::DocsAndFreqs);

        field
            .add_at(
                &mut term_pool,
                &mut th,
                b"hello",
                0,
                TermPosition::new(0, 0, 5u16),
            )
            .unwrap();

        assert_eq!(field.num_terms(), 1);
        assert_eq!(field.term_bytes(&term_pool, 0), b"hello");
    }

    #[test]
    fn test_duplicate_term_same_doc() {
        let mut term_pool = new_term_pool();
        let mut th = TermsHash::new();
        let mut field = FreqProxTermsWriterPerField::new(
            "body".to_string(),
            IndexOptions::DocsAndFreqsAndPositions,
        );

        field
            .add_at(
                &mut term_pool,
                &mut th,
                b"hello",
                0,
                TermPosition::new(0, 0, 5u16),
            )
            .unwrap();
        field
            .add_at(
                &mut term_pool,
                &mut th,
                b"hello",
                0,
                TermPosition::new(1, 6, 5),
            )
            .unwrap();

        assert_eq!(field.num_terms(), 1);
        // Freq should be 2
        assert_eq!(field.postings_array.term_freqs.as_ref().unwrap()[0], 2);
    }

    #[test]
    fn test_multiple_terms() {
        let mut term_pool = new_term_pool();
        let mut th = TermsHash::new();
        let mut field =
            FreqProxTermsWriterPerField::new("body".to_string(), IndexOptions::DocsAndFreqs);

        field
            .add_at(
                &mut term_pool,
                &mut th,
                b"hello",
                0,
                TermPosition::new(0, 0, 5u16),
            )
            .unwrap();
        field
            .add_at(
                &mut term_pool,
                &mut th,
                b"world",
                0,
                TermPosition::new(1, 6, 5),
            )
            .unwrap();
        field
            .add_at(
                &mut term_pool,
                &mut th,
                b"hello",
                0,
                TermPosition::new(2, 12, 5),
            )
            .unwrap();

        assert_eq!(field.num_terms(), 2);
        // "hello" freq should be 2
        let hello_id = field.base.bytes_hash.find(&term_pool, b"hello");
        assert_ge!(hello_id, 0);
        assert_eq!(
            field.postings_array.term_freqs.as_ref().unwrap()[hello_id as usize],
            2
        );
    }

    #[test]
    fn test_term_across_documents() {
        let mut term_pool = new_term_pool();
        let mut th = TermsHash::new();
        let mut field = FreqProxTermsWriterPerField::new(
            "body".to_string(),
            IndexOptions::DocsAndFreqsAndPositions,
        );

        // Doc 0: "hello world"
        field
            .add_at(
                &mut term_pool,
                &mut th,
                b"hello",
                0,
                TermPosition::new(0, 0, 5u16),
            )
            .unwrap();
        field
            .add_at(
                &mut term_pool,
                &mut th,
                b"world",
                0,
                TermPosition::new(1, 6, 5),
            )
            .unwrap();

        // Doc 1: "hello"
        field
            .add_at(
                &mut term_pool,
                &mut th,
                b"hello",
                1,
                TermPosition::new(0, 0, 5),
            )
            .unwrap();

        assert_eq!(field.num_terms(), 2);

        // Read back the doc/freq stream for "hello" (stream 0)
        let hello_id = field.base.bytes_hash.find(&term_pool, b"hello") as usize;
        let (start, end) = field.get_stream_range(&th.int_pool, hello_id, 0);
        let mut reader = ByteSliceReader::new(&th.byte_pool, start, end);

        // First doc: code = 0 << 1 = 0, freq = 1 → encoded as (0 | 1) = 1
        let code = read_vint(&mut reader);
        assert_eq!(code, 1); // doc_delta=0 << 1 | 1 (freq=1 packed)

        // Second doc hasn't been flushed yet (it's the current pending doc)
        assert!(reader.eof());
    }

    #[test]
    fn test_sort_terms_lexicographic() {
        let mut term_pool = new_term_pool();
        let mut th = TermsHash::new();
        let mut field =
            FreqProxTermsWriterPerField::new("body".to_string(), IndexOptions::DocsAndFreqs);

        field
            .add_at(
                &mut term_pool,
                &mut th,
                b"cherry",
                0,
                TermPosition::new(0, 0, 6u16),
            )
            .unwrap();
        field
            .add_at(
                &mut term_pool,
                &mut th,
                b"apple",
                0,
                TermPosition::new(1, 7, 5),
            )
            .unwrap();
        field
            .add_at(
                &mut term_pool,
                &mut th,
                b"banana",
                0,
                TermPosition::new(2, 13, 6),
            )
            .unwrap();

        field.sort_terms(&term_pool);
        let sorted = field.sorted_term_ids();
        assert_len_eq_x!(sorted, 3);

        assert_eq!(field.term_bytes(&term_pool, sorted[0] as usize), b"apple");
        assert_eq!(field.term_bytes(&term_pool, sorted[1] as usize), b"banana");
        assert_eq!(field.term_bytes(&term_pool, sorted[2] as usize), b"cherry");
    }

    #[test]
    fn test_docs_only_no_freq() {
        let mut term_pool = new_term_pool();
        let mut th = TermsHash::new();
        let mut field = FreqProxTermsWriterPerField::new("tags".to_string(), IndexOptions::Docs);

        field
            .add_at(
                &mut term_pool,
                &mut th,
                b"tag1",
                0,
                TermPosition::new(0, 0, 4u16),
            )
            .unwrap();
        field
            .add_at(
                &mut term_pool,
                &mut th,
                b"tag1",
                1,
                TermPosition::new(0, 0, 4u16),
            )
            .unwrap();
        field
            .add_at(
                &mut term_pool,
                &mut th,
                b"tag1",
                2,
                TermPosition::new(0, 0, 4u16),
            )
            .unwrap();

        assert_eq!(field.num_terms(), 1);
        assert!(!field.has_freq);

        // Read stream 0 — should have doc codes
        let tid = field.base.bytes_hash.find(&term_pool, b"tag1") as usize;
        let (start, end) = field.get_stream_range(&th.int_pool, tid, 0);
        let mut reader = ByteSliceReader::new(&th.byte_pool, start, end);

        // Doc 0 encoded, doc 1 delta encoded (1-0=1)
        let code0 = read_vint(&mut reader);
        assert_eq!(code0, 0); // doc 0
        let code1 = read_vint(&mut reader);
        assert_eq!(code1, 1); // delta: 1-0=1
        // Doc 2 is still pending (not flushed)
    }

    #[test]
    fn test_positions_stream() {
        let mut term_pool = new_term_pool();
        let mut th = TermsHash::new();
        let mut field = FreqProxTermsWriterPerField::new(
            "body".to_string(),
            IndexOptions::DocsAndFreqsAndPositions,
        );

        // "hello" appears at positions 0 and 3 in doc 0
        field
            .add_at(
                &mut term_pool,
                &mut th,
                b"hello",
                0,
                TermPosition::new(0, 0, 5u16),
            )
            .unwrap();
        field
            .add_at(
                &mut term_pool,
                &mut th,
                b"other",
                0,
                TermPosition::new(1, 6, 5),
            )
            .unwrap();
        field
            .add_at(
                &mut term_pool,
                &mut th,
                b"stuff",
                0,
                TermPosition::new(2, 12, 5),
            )
            .unwrap();
        field
            .add_at(
                &mut term_pool,
                &mut th,
                b"hello",
                0,
                TermPosition::new(3, 18, 5),
            )
            .unwrap();

        let hello_id = field.base.bytes_hash.find(&term_pool, b"hello") as usize;

        // Read position stream (stream 1)
        let (start, end) = field.get_stream_range(&th.int_pool, hello_id, 1);
        let mut reader = ByteSliceReader::new(&th.byte_pool, start, end);

        // Position 0: proxCode = 0 << 1 = 0 (no payload)
        let pos0 = read_vint(&mut reader);
        assert_eq!(pos0, 0); // position 0 << 1

        // Position 3: proxCode = (3 - 0) << 1 = 6 (delta from last position)
        let pos1 = read_vint(&mut reader);
        assert_eq!(pos1, 6); // delta 3 << 1
    }

    #[test]
    fn test_multi_doc_freq_encoding() {
        let mut term_pool = new_term_pool();
        let mut th = TermsHash::new();
        let mut field =
            FreqProxTermsWriterPerField::new("body".to_string(), IndexOptions::DocsAndFreqs);

        // Doc 0: "hello" x3
        field
            .add_at(
                &mut term_pool,
                &mut th,
                b"hello",
                0,
                TermPosition::new(0, 0, 5u16),
            )
            .unwrap();
        field
            .add_at(
                &mut term_pool,
                &mut th,
                b"hello",
                0,
                TermPosition::new(1, 6, 5),
            )
            .unwrap();
        field
            .add_at(
                &mut term_pool,
                &mut th,
                b"hello",
                0,
                TermPosition::new(2, 12, 5),
            )
            .unwrap();

        // Doc 1: "hello" x1
        field
            .add_at(
                &mut term_pool,
                &mut th,
                b"hello",
                1,
                TermPosition::new(0, 0, 5),
            )
            .unwrap();

        // Doc 2: "hello" x2
        field
            .add_at(
                &mut term_pool,
                &mut th,
                b"hello",
                2,
                TermPosition::new(0, 0, 5),
            )
            .unwrap();
        field
            .add_at(
                &mut term_pool,
                &mut th,
                b"hello",
                2,
                TermPosition::new(1, 6, 5),
            )
            .unwrap();

        let hello_id = field.base.bytes_hash.find(&term_pool, b"hello") as usize;
        let (start, end) = field.get_stream_range(&th.int_pool, hello_id, 0);
        let mut reader = ByteSliceReader::new(&th.byte_pool, start, end);

        // Doc 0: delta=0, freq=3 → docCode = 0<<1 = 0, then freq=3
        let doc0_code = read_vint(&mut reader);
        assert_eq!(doc0_code, 0); // 0 << 1
        let doc0_freq = read_vint(&mut reader);
        assert_eq!(doc0_freq, 3);

        // Doc 1: delta=1, freq=1 → docCode = 1<<1|1 = 3 (freq packed)
        let doc1_code = read_vint(&mut reader);
        assert_eq!(doc1_code, 3); // (1 << 1) | 1

        // Doc 2 is still pending (current doc, not yet flushed)
        assert!(reader.eof());
    }

    #[test]
    fn test_max_term_frequency_tracking() {
        let mut term_pool = new_term_pool();
        let mut th = TermsHash::new();
        let mut field =
            FreqProxTermsWriterPerField::new("body".to_string(), IndexOptions::DocsAndFreqs);

        field
            .add_at(
                &mut term_pool,
                &mut th,
                b"a",
                0,
                TermPosition::new(0, 0, 1u16),
            )
            .unwrap();
        field
            .add_at(&mut term_pool, &mut th, b"b", 0, TermPosition::new(1, 2, 1))
            .unwrap();
        field
            .add_at(&mut term_pool, &mut th, b"a", 0, TermPosition::new(2, 4, 1))
            .unwrap();
        field
            .add_at(&mut term_pool, &mut th, b"a", 0, TermPosition::new(3, 6, 1))
            .unwrap();

        assert_eq!(field.max_term_frequency, 3); // "a" appeared 3 times
    }

    #[test]
    fn test_unique_term_count_tracking() {
        let mut term_pool = new_term_pool();
        let mut th = TermsHash::new();
        let mut field =
            FreqProxTermsWriterPerField::new("body".to_string(), IndexOptions::DocsAndFreqs);

        field
            .add_at(
                &mut term_pool,
                &mut th,
                b"a",
                0,
                TermPosition::new(0, 0, 1u16),
            )
            .unwrap();
        field
            .add_at(&mut term_pool, &mut th, b"b", 0, TermPosition::new(1, 2, 1))
            .unwrap();
        field
            .add_at(&mut term_pool, &mut th, b"a", 0, TermPosition::new(2, 4, 1))
            .unwrap();

        assert_eq!(field.unique_term_count, 2);
    }

    #[test]
    fn test_oversize_aligns_to_bytes_per_posting() {
        // When remainder != 0, oversize rounds up
        let result = oversize(3, 12);
        assert_eq!(result % 12, 0);
        assert_ge!(result, 3);
    }

    #[test]
    fn test_terms_hash_reset() {
        let mut term_pool = new_term_pool();
        let mut th = TermsHash::new();
        let mut field =
            FreqProxTermsWriterPerField::new("body".to_string(), IndexOptions::DocsAndFreqs);
        field
            .add_at(
                &mut term_pool,
                &mut th,
                b"hello",
                0,
                TermPosition::new(0, 0, 5u16),
            )
            .unwrap();

        th.reset();
        // After reset, pools are cleared — new terms can be added
        let mut field2 =
            FreqProxTermsWriterPerField::new("body".to_string(), IndexOptions::DocsAndFreqs);
        field2
            .add_at(
                &mut term_pool,
                &mut th,
                b"world",
                0,
                TermPosition::new(0, 0, 5),
            )
            .unwrap();
        assert_eq!(field2.num_terms(), 1);
    }

    #[test]
    fn test_terms_hash_default() {
        let th = TermsHash::default();
        assert_eq!(th.int_pool.len(), 0);
    }

    #[test]
    fn test_terms_hash_debug() {
        let th = TermsHash::new();
        let debug = format!("{th:?}");
        assert_contains!(debug, "TermsHash");
        assert_contains!(debug, "int_pool_len");
    }

    #[test]
    fn test_terms_hash_per_field_debug() {
        let thpf = TermsHashPerField::new(1, "body".to_string(), IndexOptions::DocsAndFreqs);
        let debug = format!("{thpf:?}");
        assert_contains!(debug, "TermsHashPerField");
        assert_contains!(debug, "body");
    }

    #[test]
    fn test_freq_prox_debug() {
        let field =
            FreqProxTermsWriterPerField::new("body".to_string(), IndexOptions::DocsAndFreqs);
        let debug = format!("{field:?}");
        assert_contains!(debug, "FreqProxTermsWriterPerField");
        assert_contains!(debug, "body");
        assert_contains!(debug, "has_freq");
    }

    #[test]
    fn test_freq_prox_mem_size() {
        use mem_dbg::{MemSize, SizeFlags};
        let field =
            FreqProxTermsWriterPerField::new("body".to_string(), IndexOptions::DocsAndFreqs);
        let size = field.mem_size(SizeFlags::CAPACITY);
        assert_gt!(size, 0);
    }

    #[test]
    fn test_positions_and_offsets() {
        let mut term_pool = new_term_pool();
        let mut th = TermsHash::new();
        let mut field = FreqProxTermsWriterPerField::new(
            "body".to_string(),
            IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
        );

        // "hello" at position 0 with offsets [0, 5)
        field
            .add_at(
                &mut term_pool,
                &mut th,
                b"hello",
                0,
                TermPosition::new(0, 0, 5u16),
            )
            .unwrap();
        // "hello" at position 3 with offsets [18, 23)
        field
            .add_at(
                &mut term_pool,
                &mut th,
                b"hello",
                0,
                TermPosition::new(3, 18, 5),
            )
            .unwrap();

        assert_eq!(field.num_terms(), 1);
        assert!(field.has_offsets);

        let hello_id = field.base.bytes_hash.find(&term_pool, b"hello") as usize;

        // Read prox stream (stream 1) — positions and offsets interleaved
        let (start, end) = field.get_stream_range(&th.int_pool, hello_id, 1);
        let mut reader = ByteSliceReader::new(&th.byte_pool, start, end);

        // Position 0: proxCode = 0 << 1 = 0
        let pos0 = read_vint(&mut reader);
        assert_eq!(pos0, 0);
        // Offset for position 0: start_offset = 0, length = 5
        let off0_start = read_vint(&mut reader);
        assert_eq!(off0_start, 0);
        let off0_len = read_vint(&mut reader);
        assert_eq!(off0_len, 5);

        // Position 3: delta = (3 - 0) << 1 = 6
        let pos1 = read_vint(&mut reader);
        assert_eq!(pos1, 6);
        // Offset for position 3: start_offset delta = 18 - 0 = 18, length = 5
        let off1_start_delta = read_vint(&mut reader);
        assert_eq!(off1_start_delta, 18);
        let off1_len = read_vint(&mut reader);
        assert_eq!(off1_len, 5);
    }

    #[test]
    fn test_offsets_across_documents() {
        let mut term_pool = new_term_pool();
        let mut th = TermsHash::new();
        let mut field = FreqProxTermsWriterPerField::new(
            "body".to_string(),
            IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
        );

        // Doc 0: "hello" at pos 0, offsets [0, 5)
        field
            .add_at(
                &mut term_pool,
                &mut th,
                b"hello",
                0,
                TermPosition::new(0, 0, 5u16),
            )
            .unwrap();
        // Doc 1: "hello" at pos 0, offsets [0, 5) — offsets reset per new doc
        field
            .add_at(
                &mut term_pool,
                &mut th,
                b"hello",
                1,
                TermPosition::new(0, 0, 5),
            )
            .unwrap();

        let hello_id = field.base.bytes_hash.find(&term_pool, b"hello") as usize;

        // Read prox stream — doc 0 prox data should be there
        let (start, end) = field.get_stream_range(&th.int_pool, hello_id, 1);
        let mut reader = ByteSliceReader::new(&th.byte_pool, start, end);

        // Doc 0, position 0
        let pos0 = read_vint(&mut reader);
        assert_eq!(pos0, 0);
        let off_start = read_vint(&mut reader);
        assert_eq!(off_start, 0);
        let off_len = read_vint(&mut reader);
        assert_eq!(off_len, 5);

        // Doc 1, position 0 — offset reset to 0 for new doc
        let pos1 = read_vint(&mut reader);
        assert_eq!(pos1, 0);
        let off1_start = read_vint(&mut reader);
        assert_eq!(off1_start, 0);
        let off1_len = read_vint(&mut reader);
        assert_eq!(off1_len, 5);
    }

    #[test]
    fn test_buffered_postings_enum_returns_offsets() {
        let mut decoded = DecodedPostings::new();
        decoded.docs.push(DecodedDoc {
            doc_id: 0,
            freq: 2,
            pos_start: 0,
        });
        decoded.positions.extend_from_slice(&[0, 5]);
        decoded.offsets.extend_from_slice(&[
            TermOffset {
                start: 0,
                length: 5,
            },
            TermOffset {
                start: 6,
                length: 5,
            },
        ]);

        let mut pe = BufferedPostingsEnum::new(decoded, true);
        assert_eq!(pe.next_doc().unwrap(), 0);
        assert_eq!(pe.freq(), 2);

        assert_eq!(pe.next_position().unwrap(), 0);
        assert_eq!(
            pe.offset(),
            Some(TermOffset {
                start: 0,
                length: 5
            })
        );

        assert_eq!(pe.next_position().unwrap(), 5);
        assert_eq!(
            pe.offset(),
            Some(TermOffset {
                start: 6,
                length: 5
            })
        );
    }

    #[test]
    fn test_buffered_postings_enum_no_offsets() {
        let mut decoded = DecodedPostings::new();
        decoded.docs.push(DecodedDoc {
            doc_id: 0,
            freq: 1,
            pos_start: 0,
        });
        decoded.positions.push(0);
        // No offsets added

        let mut pe = BufferedPostingsEnum::new(decoded, true);
        assert_eq!(pe.next_doc().unwrap(), 0);
        assert_eq!(pe.next_position().unwrap(), 0);
        assert_none!(pe.offset());
    }
}
