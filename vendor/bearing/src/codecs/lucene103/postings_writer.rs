// SPDX-License-Identifier: Apache-2.0
//! Postings list writer that encodes doc IDs, frequencies, positions, and offsets.

use std::collections::HashSet;
use std::io;
use std::mem;

use log::debug;

use crate::codecs::competitive_impact::{CompetitiveImpactAccumulator, Impact, NormsLookup};
use crate::codecs::fields_producer::{NO_MORE_DOCS, PostingsEnumProducer};
use crate::codecs::lucene103::postings_format::{
    self, DOC_CODEC, DOC_EXTENSION, IntBlockTermState, META_CODEC, META_EXTENSION, PAY_CODEC,
    PAY_EXTENSION, POS_CODEC, POS_EXTENSION, VERSION_CURRENT,
};
use crate::codecs::{codec_footers, codec_headers};
use crate::document::{IndexOptions, TermOffset};
use crate::encoding::pfor::{self, BLOCK_SIZE};
use crate::encoding::write_encoding::WriteEncoding;
use crate::encoding::zigzag;
use crate::index::index_file_names::segment_file_name;
use crate::store::{DataOutput, Directory, IndexOutput, VecOutput};

/// Buffers position deltas (and optionally offset deltas) and PFOR-encodes
/// them in blocks of 128.
struct PositionEncoder {
    pos_delta_buffer: [i32; BLOCK_SIZE],
    offset_buffer: [TermOffset; BLOCK_SIZE],
    pos_buffer_upto: usize,
    last_start_offset: i32,
    write_offsets: bool,
}

impl PositionEncoder {
    fn new(write_offsets: bool) -> Self {
        Self {
            pos_delta_buffer: [0i32; BLOCK_SIZE],
            offset_buffer: [TermOffset::default(); BLOCK_SIZE],
            pos_buffer_upto: 0,
            last_start_offset: 0,
            write_offsets,
        }
    }

    fn pos_buffer_upto(&self) -> usize {
        self.pos_buffer_upto
    }

    /// Reset per-document state. Must be called before each document's positions.
    fn reset_doc(&mut self) {
        self.last_start_offset = 0;
    }

    /// Buffer a position delta (and optional offset), flushing a PFOR block when full.
    fn add_position(
        &mut self,
        pos_delta: i32,
        offset: Option<TermOffset>,
        pos_out: &mut dyn DataOutput,
        pay_out: Option<&mut dyn DataOutput>,
    ) -> io::Result<()> {
        self.pos_delta_buffer[self.pos_buffer_upto] = pos_delta;
        if self.write_offsets {
            let offset = offset.expect("offset required when write_offsets is true");
            let start_delta = offset.start as i32 - self.last_start_offset;
            self.offset_buffer[self.pos_buffer_upto] = TermOffset {
                start: start_delta as u32,
                length: offset.length,
            };
            self.last_start_offset = offset.start as i32;
        }
        self.pos_buffer_upto += 1;
        if self.pos_buffer_upto == BLOCK_SIZE {
            let mut longs = [0i64; BLOCK_SIZE];
            for (i, &val) in self.pos_delta_buffer.iter().enumerate().take(BLOCK_SIZE) {
                longs[i] = val as i64;
            }
            pfor::pfor_encode(&mut longs, pos_out)?;
            if self.write_offsets {
                let pay_out = pay_out.expect("pay_out required when write_offsets is true");
                let mut longs = [0i64; BLOCK_SIZE];
                for (i, offset) in self.offset_buffer.iter().enumerate().take(BLOCK_SIZE) {
                    longs[i] = offset.start as i64;
                }
                pfor::pfor_encode(&mut longs, pay_out)?;
                let mut longs = [0i64; BLOCK_SIZE];
                for (i, offset) in self.offset_buffer.iter().enumerate().take(BLOCK_SIZE) {
                    longs[i] = offset.length as i64;
                }
                pfor::pfor_encode(&mut longs, pay_out)?;
            }
            self.pos_buffer_upto = 0;
        }
        Ok(())
    }

    /// Write VInt tail for remaining buffered positions/offsets and compute
    /// last_pos_block_offset.
    fn finish(
        &self,
        mut pos_out: &mut dyn IndexOutput,
        total_term_freq: i64,
        pos_start_fp: i64,
    ) -> io::Result<i64> {
        let last_pos_block_offset = if total_term_freq > BLOCK_SIZE as i64 {
            pos_out.file_pointer() as i64 - pos_start_fp
        } else {
            -1
        };
        // VInt encode remaining positions and offsets to .pos file
        let mut last_offset_length = u16::MAX; // force first length to be written
        for i in 0..self.pos_buffer_upto {
            let pos_delta = self.pos_delta_buffer[i];
            // No payload support: write plain position delta
            pos_out.write_vint(pos_delta)?;

            if self.write_offsets {
                let offset = &self.offset_buffer[i];
                let delta = offset.start as i32;
                let length = offset.length;
                if length == last_offset_length {
                    pos_out.write_vint(delta << 1)?;
                } else {
                    pos_out.write_vint(delta << 1 | 1)?;
                    pos_out.write_vint(length as i32)?;
                    last_offset_length = length;
                }
            }
        }
        Ok(last_pos_block_offset)
    }
}

/// Groups buffers and per-term state for block encoding's flush_doc_block.
/// Reduces the 14-parameter free function to a 5-parameter method.
struct BlockFlushState {
    level0_buf: Vec<u8>,
    level1_buf: Vec<u8>,
    scratch_buf: Vec<u8>,
    /// Reusable bitset buffer for doc ID encoding (max BLOCK_SIZE/2 = 64 longs).
    bitset_buf: [u64; BLOCK_SIZE / 2],
    level0_last_doc_id: i32,
    level0_last_pos_fp: i64,
    level0_last_pay_fp: i64,
    level1_last_doc_id: i32,
    level1_last_pos_fp: i64,
    level1_last_pay_fp: i64,
    write_freqs: bool,
    write_positions: bool,
    write_offsets: bool,
    max_num_impacts_at_level0: i32,
    max_impact_num_bytes_at_level0: i32,
    max_num_impacts_at_level1: i32,
    max_impact_num_bytes_at_level1: i32,
    /// Accumulates competitive (freq, norm) pairs for the current level 0 block.
    level0_accumulator: CompetitiveImpactAccumulator,
    /// Accumulates competitive impacts across level 0 blocks for level 1 skip data.
    level1_accumulator: CompetitiveImpactAccumulator,
}

impl BlockFlushState {
    fn new(pos_start_fp: i64, pay_start_fp: i64, index_options: IndexOptions) -> Self {
        Self {
            level0_buf: Vec::new(),
            level1_buf: Vec::new(),
            scratch_buf: Vec::new(),
            bitset_buf: [0u64; BLOCK_SIZE / 2],
            level0_last_doc_id: -1,
            level0_last_pos_fp: pos_start_fp,
            level0_last_pay_fp: pay_start_fp,
            level1_last_doc_id: -1,
            level1_last_pos_fp: pos_start_fp,
            level1_last_pay_fp: pay_start_fp,
            write_freqs: index_options.has_freqs(),
            write_positions: index_options.has_positions(),
            write_offsets: index_options.has_offsets(),
            max_num_impacts_at_level0: 0,
            max_impact_num_bytes_at_level0: 0,
            max_num_impacts_at_level1: 0,
            max_impact_num_bytes_at_level1: 0,
            level0_accumulator: CompetitiveImpactAccumulator::new(),
            level1_accumulator: CompetitiveImpactAccumulator::new(),
        }
    }

    /// Flush one full doc block (128 docs) with skip data.
    fn flush_doc_block(
        &mut self,
        doc_delta_buffer: &[i32; BLOCK_SIZE],
        freq_buffer: &mut [i32; BLOCK_SIZE],
        last_doc_id: i32,
        pos_buffer_upto: usize,
        pos_fp: i64,
        pay_fp: i64,
    ) -> io::Result<()> {
        self.level0_buf.clear();
        self.scratch_buf.clear();

        // 1. Write impact data (only when we have freqs)
        if self.write_freqs {
            let impacts = self.level0_accumulator.get_competitive_freq_norm_pairs();

            let impact_start = self.scratch_buf.len();
            let impact_bytes_len = write_impacts(&impacts, &mut self.scratch_buf)?;

            if impacts.len() as i32 > self.max_num_impacts_at_level0 {
                self.max_num_impacts_at_level0 = impacts.len() as i32;
            }
            if impact_bytes_len as i32 > self.max_impact_num_bytes_at_level0 {
                self.max_impact_num_bytes_at_level0 = impact_bytes_len as i32;
            }

            {
                let mut enc = VecOutput(&mut self.level0_buf);
                enc.write_vlong(impact_bytes_len as i64)?;
            }
            let impact_end = impact_start + impact_bytes_len;
            self.level0_buf
                .extend_from_slice(&self.scratch_buf[impact_start..impact_end]);
            self.scratch_buf.clear();

            if self.write_positions {
                let mut enc = VecOutput(&mut self.level0_buf);
                enc.write_vlong(pos_fp - self.level0_last_pos_fp)?;
                enc.write_byte(pos_buffer_upto as u8)?;
                self.level0_last_pos_fp = pos_fp;

                if self.write_offsets {
                    let mut enc = VecOutput(&mut self.level0_buf);
                    enc.write_vlong(pay_fp - self.level0_last_pay_fp)?;
                    enc.write_vint(0)?; // payloadByteUpto = 0 (no payloads)
                    self.level0_last_pay_fp = pay_fp;
                }
            }
        }

        let num_skip_bytes_before_doc = self.level0_buf.len();

        // 2. Doc encoding: decide between consecutive, FOR, or bitset
        let doc_range = (last_doc_id - self.level0_last_doc_id) as usize;
        let bpv = pfor::for_delta_bits_required(doc_delta_buffer);

        if doc_range == BLOCK_SIZE {
            self.level0_buf.push(0u8);
        } else {
            let num_bitset_longs = bits2words(doc_range);
            let num_bits_next_bpv = (bpv + 1).min(32) as usize * BLOCK_SIZE;

            if num_bits_next_bpv <= doc_range {
                self.level0_buf.push(bpv as u8);
                pfor::for_delta_encode(bpv, doc_delta_buffer, &mut self.level0_buf)?;
            } else {
                assert!(num_bitset_longs <= BLOCK_SIZE / 2);
                self.level0_buf.push((-(num_bitset_longs as i8)) as u8);
                // Reuse stack-resident bitset buffer instead of heap-allocating
                self.bitset_buf[..num_bitset_longs].fill(0);
                let mut s: i32 = -1;
                for &delta in doc_delta_buffer.iter() {
                    s += delta;
                    let word = s as usize / 64;
                    let bit = s as usize % 64;
                    self.bitset_buf[word] |= 1u64 << bit;
                }
                let mut enc = VecOutput(&mut self.level0_buf);
                for &word in &self.bitset_buf[..num_bitset_longs] {
                    enc.write_le_long(word as i64)?;
                }
            }
        }

        // 3. Freq encoding
        if self.write_freqs {
            let mut longs = [0i64; BLOCK_SIZE];
            for i in 0..BLOCK_SIZE {
                longs[i] = freq_buffer[i] as i64;
            }
            pfor::pfor_encode(&mut longs, &mut self.level0_buf)?;
        }

        // 4. Write skip header to scratch_buf
        let doc_delta = last_doc_id - self.level0_last_doc_id;
        write_vint15(&mut VecOutput(&mut self.scratch_buf), doc_delta)?;
        write_vlong15(
            &mut VecOutput(&mut self.scratch_buf),
            self.level0_buf.len() as i64,
        )?;

        let num_skip_bytes = num_skip_bytes_before_doc + self.scratch_buf.len();

        // 5. Append to level1_buf: VLong(numSkipBytes) + scratch + level0
        {
            let mut enc = VecOutput(&mut self.level1_buf);
            enc.write_vlong(num_skip_bytes as i64)?;
        }
        self.level1_buf.extend_from_slice(&self.scratch_buf);
        self.level1_buf.extend_from_slice(&self.level0_buf);

        // Update tracking state for next block
        self.level0_last_doc_id = last_doc_id;
        if self.write_freqs {
            self.level1_accumulator.add_all(&self.level0_accumulator);
        }
        self.level0_accumulator.clear();

        Ok(())
    }

    /// Write level1 skip metadata and flush accumulated level0 blocks to `doc_out`.
    ///
    /// Called every `LEVEL1_NUM_DOCS` (4096) documents. Writes impact data and
    /// position file pointers for the level1 block, then copies the accumulated
    /// level0 blocks from `level1_buf`.
    fn write_level1_skip_data(
        &mut self,
        mut doc_out: &mut dyn DataOutput,
        last_doc_id: i32,
        pos_buffer_upto: usize,
        pos_fp: i64,
        pay_fp: i64,
    ) -> io::Result<()> {
        doc_out.write_vint(last_doc_id - self.level1_last_doc_id)?;

        if self.write_freqs {
            let impacts = self.level1_accumulator.get_competitive_freq_norm_pairs();
            if impacts.len() as i32 > self.max_num_impacts_at_level1 {
                self.max_num_impacts_at_level1 = impacts.len() as i32;
            }
            self.scratch_buf.clear();
            let num_impact_bytes = write_impacts(&impacts, &mut self.scratch_buf)?;
            if num_impact_bytes as i32 > self.max_impact_num_bytes_at_level1 {
                self.max_impact_num_bytes_at_level1 = num_impact_bytes as i32;
            }
            if self.write_positions {
                let mut enc = VecOutput(&mut self.scratch_buf);
                enc.write_vlong(pos_fp - self.level1_last_pos_fp)?;
                enc.write_byte(pos_buffer_upto as u8)?;
                self.level1_last_pos_fp = pos_fp;
                if self.write_offsets {
                    let mut enc = VecOutput(&mut self.scratch_buf);
                    enc.write_vlong(pay_fp - self.level1_last_pay_fp)?;
                    enc.write_vint(0)?; // payloadByteUpto = 0 (no payloads)
                    self.level1_last_pay_fp = pay_fp;
                }
            }
            let level1_len =
                2 * mem::size_of::<i16>() + self.scratch_buf.len() + self.level1_buf.len();
            doc_out.write_vlong(level1_len as i64)?;
            debug_assert!(num_impact_bytes <= i16::MAX as usize);
            debug_assert!(self.scratch_buf.len() + mem::size_of::<i16>() <= i16::MAX as usize);
            doc_out.write_le_short((self.scratch_buf.len() + mem::size_of::<i16>()) as i16)?;
            doc_out.write_le_short(num_impact_bytes as i16)?;
            doc_out.write_all(&self.scratch_buf)?;
            self.scratch_buf.clear();
        } else {
            doc_out.write_vlong(self.level1_buf.len() as i64)?;
        }
        doc_out.write_all(&self.level1_buf)?;
        self.level1_buf.clear();

        self.level1_last_doc_id = last_doc_id;
        self.level1_accumulator.clear();

        Ok(())
    }
}

/// Groups per-term context for the block encoding path.
struct TermWriteContext {
    doc_freq: i32,
    total_term_freq: i64,
    doc_start_fp: i64,
    pos_start_fp: i64,
    index_options: IndexOptions,
}

/// Writes postings (.doc, .pos, .pay, .psm) files.
pub struct PostingsWriter {
    doc_out: Box<dyn IndexOutput>,
    pos_out: Option<Box<dyn IndexOutput>>,
    pay_out: Option<Box<dyn IndexOutput>>,
    meta_out: Box<dyn IndexOutput>,
    max_num_impacts_at_level0: i32,
    max_impact_num_bytes_at_level0: i32,
    max_num_impacts_at_level1: i32,
    max_impact_num_bytes_at_level1: i32,
}

impl PostingsWriter {
    /// Creates a new PostingsWriter that streams to outputs from the directory.
    pub fn new(
        directory: &dyn Directory,
        segment: &str,
        suffix: &str,
        id: &[u8; 16],
        max_index_options: IndexOptions,
    ) -> io::Result<Self> {
        let doc_name = segment_file_name(segment, suffix, DOC_EXTENSION);
        let meta_name = segment_file_name(segment, suffix, META_EXTENSION);

        let mut doc_out = directory.create_output(&doc_name)?;
        let mut meta_out = directory.create_output(&meta_name)?;
        let mut pos_out = if max_index_options.has_positions() {
            let pos_name = segment_file_name(segment, suffix, POS_EXTENSION);
            Some(directory.create_output(&pos_name)?)
        } else {
            None
        };

        let mut pay_out = if max_index_options.has_offsets() {
            let pay_name = segment_file_name(segment, suffix, PAY_EXTENSION);
            Some(directory.create_output(&pay_name)?)
        } else {
            None
        };

        codec_headers::write_index_header(&mut *doc_out, DOC_CODEC, VERSION_CURRENT, id, suffix)?;
        codec_headers::write_index_header(&mut *meta_out, META_CODEC, VERSION_CURRENT, id, suffix)?;

        if let Some(ref mut pos_out) = pos_out {
            codec_headers::write_index_header(
                &mut **pos_out,
                POS_CODEC,
                VERSION_CURRENT,
                id,
                suffix,
            )?;
        }
        if let Some(ref mut pay_out) = pay_out {
            codec_headers::write_index_header(
                &mut **pay_out,
                PAY_CODEC,
                VERSION_CURRENT,
                id,
                suffix,
            )?;
        }

        Ok(Self {
            doc_out,
            pos_out,
            pay_out,
            meta_out,
            max_num_impacts_at_level0: 0,
            max_impact_num_bytes_at_level0: 0,
            max_num_impacts_at_level1: 0,
            max_impact_num_bytes_at_level1: 0,
        })
    }

    /// Write postings for one term. Returns metadata for the term dictionary.
    ///
    /// Pulls postings data from the producer via `next_doc()` / `freq()` /
    /// `next_position()`, matching Java's `PushPostingsWriterBase.writeTerm()`.
    ///
    /// - Singleton (docFreq==1): docID pulsed into IntBlockTermState, nothing to .doc
    /// - Block encoding (docFreq >= 128): FOR-encoded blocks with skip data
    /// - VInt tail (docFreq < 128):
    ///   - With freqs: GroupVInt doc deltas with freq bits, then freqs for non-1
    ///   - Without freqs (DOCS only): GroupVInt doc deltas
    /// - Positions: PFOR-encoded blocks of 128 + VInt tail for remainder
    pub(crate) fn write_term(
        &mut self,
        postings: &mut dyn PostingsEnumProducer,
        index_options: IndexOptions,
        norms: &dyn NormsLookup,
        docs_seen: &mut HashSet<i32>,
    ) -> io::Result<IntBlockTermState> {
        let doc_freq = postings.doc_freq();
        let total_term_freq = postings.total_term_freq();
        let doc_start_fp = self.doc_out.file_pointer() as i64;
        let pos_start_fp = self
            .pos_out
            .as_ref()
            .map(|p| p.file_pointer() as i64)
            .unwrap_or(0);
        let pay_start_fp = self
            .pay_out
            .as_ref()
            .map(|p| p.file_pointer() as i64)
            .unwrap_or(0);

        if doc_freq == 1 {
            // Singleton: pulse docID into term dictionary
            let doc_id = postings.next_doc()?;
            docs_seen.insert(doc_id);
            debug!("postings_writer: singleton term, doc_id={}", doc_id);

            // Write positions for singleton
            let mut last_pos_block_offset = -1i64;
            if index_options.has_positions()
                && let Some(ref mut pos_out) = self.pos_out
            {
                let freq = postings.freq();
                let mut pe = PositionEncoder::new(index_options.has_offsets());
                pe.reset_doc();
                let mut last_pos = 0i32;
                for _ in 0..freq {
                    let pos = postings.next_position()?;
                    let offset = postings.offset();
                    pe.add_position(
                        pos - last_pos,
                        offset,
                        &mut **pos_out,
                        self.pay_out
                            .as_deref_mut()
                            .map(|p| p as &mut dyn DataOutput),
                    )?;
                    last_pos = pos;
                }
                last_pos_block_offset = pe.finish(&mut **pos_out, total_term_freq, pos_start_fp)?;
            }

            Ok(IntBlockTermState {
                doc_freq,
                total_term_freq,
                term_block_ord: 0,
                doc_start_fp,
                pos_start_fp,
                pay_start_fp,
                last_pos_block_offset,
                singleton_doc_id: doc_id,
            })
        } else if doc_freq >= BLOCK_SIZE as i32 {
            // Block encoding path for high-frequency terms
            debug!("postings_writer: block encoding, doc_freq={}", doc_freq);
            self.write_term_blocks(
                postings,
                &TermWriteContext {
                    doc_freq,
                    total_term_freq,
                    doc_start_fp,
                    pos_start_fp,
                    index_options,
                },
                norms,
                docs_seen,
            )
        } else {
            // VInt block (docFreq < BLOCK_SIZE)
            let mut doc_deltas: Vec<i32> = Vec::with_capacity(doc_freq as usize);
            let mut freqs: Vec<i32> = Vec::with_capacity(doc_freq as usize);
            let mut last_doc_id = -1i32;

            // First pass: collect doc deltas and freqs
            // We need the full arrays for GroupVInt encoding before writing positions,
            // so we buffer doc/freq data, then iterate positions in a second pass
            // via a separate PostingsEnumProducer (the caller provides a fresh one
            // if needed). For now, we collect positions inline.
            //
            // To write positions we need to pull them during the doc iteration,
            // so we accumulate positions into a temporary buffer.
            let mut all_positions: Vec<Vec<i32>> = Vec::new();
            let mut all_offsets: Vec<Vec<Option<TermOffset>>> = Vec::new();

            loop {
                let doc_id = postings.next_doc()?;
                if doc_id == NO_MORE_DOCS {
                    break;
                }
                docs_seen.insert(doc_id);
                let delta = doc_id - last_doc_id;
                doc_deltas.push(delta);
                let freq = postings.freq();
                freqs.push(freq);
                last_doc_id = doc_id;

                if index_options.has_positions() {
                    let mut positions = Vec::with_capacity(freq as usize);
                    let mut offsets = Vec::with_capacity(freq as usize);
                    for _ in 0..freq {
                        positions.push(postings.next_position()?);
                        offsets.push(postings.offset());
                    }
                    all_positions.push(positions);
                    all_offsets.push(offsets);
                }
            }

            write_vint_block(
                &mut *self.doc_out,
                &mut doc_deltas,
                &freqs,
                doc_freq as usize,
                index_options.has_freqs(),
            )?;

            debug!(
                "postings_writer: VInt block, doc_freq={}, bytes_written={}",
                doc_freq,
                self.doc_out.file_pointer() as i64 - doc_start_fp
            );

            // Write positions
            let mut last_pos_block_offset = -1i64;
            if index_options.has_positions()
                && let Some(ref mut pos_out) = self.pos_out
            {
                let mut pe = PositionEncoder::new(index_options.has_offsets());
                for (positions, offsets) in all_positions.iter().zip(all_offsets.iter()) {
                    pe.reset_doc();
                    let mut last_pos = 0i32;
                    for (i, &pos) in positions.iter().enumerate() {
                        pe.add_position(
                            pos - last_pos,
                            offsets[i],
                            &mut **pos_out,
                            self.pay_out
                                .as_deref_mut()
                                .map(|p| p as &mut dyn DataOutput),
                        )?;
                        last_pos = pos;
                    }
                }
                last_pos_block_offset = pe.finish(&mut **pos_out, total_term_freq, pos_start_fp)?;
            }

            Ok(IntBlockTermState {
                doc_freq,
                total_term_freq,
                term_block_ord: 0,
                doc_start_fp,
                pos_start_fp,
                pay_start_fp,
                last_pos_block_offset,
                singleton_doc_id: -1,
            })
        }
    }

    /// Write a term with docFreq >= BLOCK_SIZE using block encoding with skip data.
    fn write_term_blocks(
        &mut self,
        postings: &mut dyn PostingsEnumProducer,
        ctx: &TermWriteContext,
        norms: &dyn NormsLookup,
        docs_seen: &mut HashSet<i32>,
    ) -> io::Result<IntBlockTermState> {
        let doc_freq = ctx.doc_freq;
        let total_term_freq = ctx.total_term_freq;
        let doc_start_fp = ctx.doc_start_fp;
        let pos_start_fp = ctx.pos_start_fp;
        let index_options = ctx.index_options;
        let mut doc_delta_buffer = [0i32; BLOCK_SIZE];
        let mut freq_buffer = [0i32; BLOCK_SIZE];
        let mut doc_buffer_upto = 0usize;

        let pay_start_fp = self
            .pay_out
            .as_ref()
            .map(|p| p.file_pointer() as i64)
            .unwrap_or(0);
        let mut pe = PositionEncoder::new(index_options.has_offsets());
        let mut bfs = BlockFlushState::new(pos_start_fp, pay_start_fp, index_options);

        let mut last_doc_id = -1i32;
        let mut doc_count = 0usize;

        loop {
            let doc_id = postings.next_doc()?;
            if doc_id == NO_MORE_DOCS {
                break;
            }
            docs_seen.insert(doc_id);

            // Buffer doc delta and freq
            let doc_delta = doc_id - last_doc_id;
            doc_delta_buffer[doc_buffer_upto] = doc_delta;
            let freq = postings.freq();
            if index_options.has_freqs() {
                freq_buffer[doc_buffer_upto] = freq;
                bfs.level0_accumulator.add(freq, norms.get(doc_id));
            }
            last_doc_id = doc_id;

            // Buffer positions
            if index_options.has_positions()
                && let Some(ref mut pos_out) = self.pos_out
            {
                pe.reset_doc();
                let mut last_pos = 0i32;
                for _ in 0..freq {
                    let pos = postings.next_position()?;
                    let offset = postings.offset();
                    pe.add_position(
                        pos - last_pos,
                        offset,
                        &mut **pos_out,
                        self.pay_out
                            .as_deref_mut()
                            .map(|p| p as &mut dyn DataOutput),
                    )?;
                    last_pos = pos;
                }
            }

            doc_buffer_upto += 1;
            doc_count += 1;

            // Flush a full doc block
            if doc_buffer_upto == BLOCK_SIZE {
                let pos_fp = self
                    .pos_out
                    .as_ref()
                    .map(|p| p.file_pointer() as i64)
                    .unwrap_or(0);
                let pay_fp = self
                    .pay_out
                    .as_ref()
                    .map(|p| p.file_pointer() as i64)
                    .unwrap_or(0);
                bfs.flush_doc_block(
                    &doc_delta_buffer,
                    &mut freq_buffer,
                    doc_id,
                    pe.pos_buffer_upto(),
                    pos_fp,
                    pay_fp,
                )?;
                doc_buffer_upto = 0;

                if doc_count & postings_format::LEVEL1_MASK == 0 {
                    bfs.write_level1_skip_data(
                        &mut *self.doc_out,
                        doc_id,
                        pe.pos_buffer_upto(),
                        pos_fp,
                        pay_fp,
                    )?;
                }
            }
        }

        // Flush remaining VInt tail (< BLOCK_SIZE docs)
        if doc_buffer_upto > 0 {
            let mut tail_deltas = doc_delta_buffer[..doc_buffer_upto].to_vec();
            write_vint_block_to_buf(
                &mut bfs.level1_buf,
                &mut tail_deltas,
                &freq_buffer[..doc_buffer_upto],
                doc_buffer_upto,
                index_options.has_freqs(),
            )?;
        }

        // Flush remaining level1_buf to doc_out
        self.doc_out.write_all(&bfs.level1_buf)?;

        // Write remaining position VInt tail
        let mut last_pos_block_offset = -1i64;
        if index_options.has_positions()
            && let Some(ref mut pos_out) = self.pos_out
        {
            last_pos_block_offset = pe.finish(&mut **pos_out, total_term_freq, pos_start_fp)?;
        }

        // Copy impact metadata back from BlockFlushState
        self.max_num_impacts_at_level0 = self
            .max_num_impacts_at_level0
            .max(bfs.max_num_impacts_at_level0);
        self.max_impact_num_bytes_at_level0 = self
            .max_impact_num_bytes_at_level0
            .max(bfs.max_impact_num_bytes_at_level0);
        self.max_num_impacts_at_level1 = self
            .max_num_impacts_at_level1
            .max(bfs.max_num_impacts_at_level1);
        self.max_impact_num_bytes_at_level1 = self
            .max_impact_num_bytes_at_level1
            .max(bfs.max_impact_num_bytes_at_level1);

        Ok(IntBlockTermState {
            doc_freq,
            total_term_freq,
            term_block_ord: 0,
            doc_start_fp,
            pos_start_fp,
            pay_start_fp,
            last_pos_block_offset,
            singleton_doc_id: -1,
        })
    }

    /// Encode a term's metadata for the block tree terms dictionary.
    /// Called by BlockTreeTermsWriter for each term in a block.
    ///
    pub fn encode_term(
        &self,
        out: &mut Vec<u8>,
        state: &IntBlockTermState,
        last_state: &IntBlockTermState,
        write_positions: bool,
        write_offsets: bool,
    ) -> io::Result<()> {
        let mut buf = VecOutput(out);

        if last_state.singleton_doc_id != -1
            && state.singleton_doc_id != -1
            && state.doc_start_fp == last_state.doc_start_fp
        {
            // Both singletons at same file position: encode docID delta
            let delta = state.singleton_doc_id as i64 - last_state.singleton_doc_id as i64;
            let encoded = zigzag::encode_i64(delta);
            buf.write_vlong((encoded << 1) | 0x01)?;
        } else {
            // Normal: encode file pointer delta
            buf.write_vlong((state.doc_start_fp - last_state.doc_start_fp) << 1)?;
            if state.singleton_doc_id != -1 {
                buf.write_vint(state.singleton_doc_id)?;
            }
        }

        if write_positions {
            buf.write_vlong(state.pos_start_fp - last_state.pos_start_fp)?;
            if write_offsets {
                buf.write_vlong(state.pay_start_fp - last_state.pay_start_fp)?;
            }
        }

        if write_positions && state.last_pos_block_offset != -1 {
            buf.write_vlong(state.last_pos_block_offset)?;
        }

        Ok(())
    }

    /// Finalize: write footers, drop outputs (auto-persists to directory).
    /// Returns the file names written.
    pub fn finish(mut self) -> io::Result<Vec<String>> {
        let mut names = Vec::new();

        // Write .doc footer
        codec_footers::write_footer(&mut *self.doc_out)?;

        // Write .pos footer if present
        if let Some(ref mut pos_out) = self.pos_out {
            codec_footers::write_footer(&mut **pos_out)?;
        }

        // Write .pay footer if present
        if let Some(ref mut pay_out) = self.pay_out {
            codec_footers::write_footer(&mut **pay_out)?;
        }

        // Write .psm metadata: impact stats + file lengths + footer
        self.meta_out.write_le_int(self.max_num_impacts_at_level0)?;
        self.meta_out
            .write_le_int(self.max_impact_num_bytes_at_level0)?;
        self.meta_out.write_le_int(self.max_num_impacts_at_level1)?;
        self.meta_out
            .write_le_int(self.max_impact_num_bytes_at_level1)?;
        self.meta_out
            .write_le_long(self.doc_out.file_pointer() as i64)?;
        if let Some(ref pos_out) = self.pos_out {
            self.meta_out.write_le_long(pos_out.file_pointer() as i64)?;
            if let Some(ref pay_out) = self.pay_out {
                self.meta_out.write_le_long(pay_out.file_pointer() as i64)?;
            }
        }
        codec_footers::write_footer(&mut *self.meta_out)?;

        names.push(self.doc_out.name().to_string());
        if let Some(ref pos_out) = self.pos_out {
            names.push(pos_out.name().to_string());
        }
        if let Some(ref pay_out) = self.pay_out {
            names.push(pay_out.name().to_string());
        }
        names.push(self.meta_out.name().to_string());

        // Dropping self auto-persists all outputs to directory
        Ok(names)
    }
}

/// Write freq buffer with variable-length encoding and doc buffer with group-varint encoding.
fn write_vint_block(
    out: &mut dyn DataOutput,
    doc_deltas: &mut [i32],
    freqs: &[i32],
    num: usize,
    write_freqs: bool,
) -> io::Result<()> {
    write_vint_block_impl(out, doc_deltas, freqs, num, write_freqs)
}

/// Special vints encoded on 2 bytes if they require 15 bits or less.
fn write_vlong15(mut out: &mut dyn DataOutput, v: i64) -> io::Result<()> {
    debug_assert!(v >= 0);
    if (v & !0x7FFF_i64) == 0 {
        out.write_le_short(v as i16)?;
    } else {
        out.write_le_short((0x8000 | (v & 0x7FFF)) as i16)?;
        out.write_vlong(v >> 15)?;
    }
    Ok(())
}

fn write_vint15(out: &mut dyn DataOutput, v: i32) -> io::Result<()> {
    debug_assert!(v >= 0);
    write_vlong15(out, v as i64)
}

/// Delta-encodes competitive impacts into the buffer.
///
/// Each impact is encoded relative to the previous one:
/// - `freqDelta = freq - prevFreq - 1`
/// - `normDelta = norm - prevNorm - 1`
/// - If normDelta == 0: write `freqDelta << 1` as VInt
/// - Otherwise: write `(freqDelta << 1) | 1` as VInt, then normDelta as ZLong
///
/// Returns number of bytes written.
fn write_impacts(impacts: &[Impact], buf: &mut Vec<u8>) -> io::Result<usize> {
    let start = buf.len();
    let mut enc = VecOutput(buf);
    let mut prev = Impact { freq: 0, norm: 0 };
    for &impact in impacts {
        let freq_delta = impact.freq - prev.freq - 1;
        let norm_delta = impact.norm - prev.norm - 1;
        if norm_delta == 0 {
            enc.write_vint(freq_delta << 1)?;
        } else {
            enc.write_vint((freq_delta << 1) | 1)?;
            enc.write_zlong(norm_delta)?;
        }
        prev = impact;
    }
    let bytes_written = enc.0.len() - start;
    Ok(bytes_written)
}

/// Returns the number of longs needed to store the given number of bits.
fn bits2words(num_bits: usize) -> usize {
    num_bits.div_ceil(64)
}

/// Write VInt block to a `Vec<u8>` buffer (for the tail in block encoding).
fn write_vint_block_to_buf(
    buf: &mut Vec<u8>,
    doc_deltas: &mut [i32],
    freqs: &[i32],
    num: usize,
    write_freqs: bool,
) -> io::Result<()> {
    let mut enc = VecOutput(buf);
    write_vint_block_impl(&mut enc, doc_deltas, freqs, num, write_freqs)
}

/// Shared VInt block implementation that works with any DataOutput.
fn write_vint_block_impl(
    mut out: &mut dyn DataOutput,
    doc_deltas: &mut [i32],
    freqs: &[i32],
    num: usize,
    write_freqs: bool,
) -> io::Result<()> {
    if write_freqs {
        for i in 0..num {
            doc_deltas[i] = (doc_deltas[i] << 1) | if freqs[i] == 1 { 1 } else { 0 };
        }
    }
    out.write_group_vints(doc_deltas, num)?;
    if write_freqs {
        for &freq in &freqs[..num] {
            if freq != 1 {
                out.write_vint(freq)?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codecs::competitive_impact::BufferedNormsLookup;
    use crate::index::pipeline::terms_hash::{BufferedPostingsEnum, DecodedDoc, DecodedPostings};
    use crate::store::{MemoryDirectory, SharedDirectory};

    fn test_directory() -> SharedDirectory {
        MemoryDirectory::create()
    }

    /// Build a `DecodedPostings` from doc_id/freq pairs with no positions.
    fn build_postings(docs: &[(i32, i32)]) -> DecodedPostings {
        let mut decoded = DecodedPostings::new();
        for &(doc_id, freq) in docs {
            decoded.docs.push(DecodedDoc {
                doc_id,
                freq,
                pos_start: 0,
            });
        }
        decoded
    }

    /// Build a `DecodedPostings` from doc_id/freq/positions triples.
    fn build_postings_with_positions(docs: &[(i32, i32, &[i32])]) -> DecodedPostings {
        let mut decoded = DecodedPostings::new();
        for &(doc_id, freq, positions) in docs {
            let pos_start = decoded.positions.len() as u32;
            decoded.positions.extend_from_slice(positions);
            decoded.docs.push(DecodedDoc {
                doc_id,
                freq,
                pos_start,
            });
        }
        decoded
    }

    /// Build a `DecodedPostings` from doc_id/freq/positions/offsets.
    fn build_postings_with_offsets(docs: &[(i32, i32, &[i32], &[TermOffset])]) -> DecodedPostings {
        let mut decoded = DecodedPostings::new();
        for &(doc_id, freq, positions, offsets) in docs {
            let pos_start = decoded.positions.len() as u32;
            decoded.positions.extend_from_slice(positions);
            decoded.offsets.extend_from_slice(offsets);
            decoded.docs.push(DecodedDoc {
                doc_id,
                freq,
                pos_start,
            });
        }
        decoded
    }

    #[test]
    fn test_singleton_term() {
        let dir = test_directory();
        let mut pw = PostingsWriter::new(&dir, "_0", "", &[0u8; 16], IndexOptions::Docs).unwrap();
        let decoded = build_postings(&[(5, 1)]);
        let mut pe = BufferedPostingsEnum::new(decoded, true);
        let state = pw
            .write_term(
                &mut pe,
                IndexOptions::DocsAndFreqs,
                &BufferedNormsLookup::no_norms(),
                &mut HashSet::new(),
            )
            .unwrap();

        assert_eq!(state.doc_freq, 1);
        assert_eq!(state.singleton_doc_id, 5);
        // Nothing written to .doc beyond the header
        let header_size = codec_headers::index_header_length(DOC_CODEC, "");
        assert_eq!(pw.doc_out.file_pointer() as usize, header_size);
    }

    #[test]
    fn test_multi_doc_term_with_freqs() {
        let dir = test_directory();
        let mut pw = PostingsWriter::new(&dir, "_0", "", &[0u8; 16], IndexOptions::Docs).unwrap();
        let decoded = build_postings(&[(0, 1), (5, 3), (10, 1)]);
        let header_size = pw.doc_out.file_pointer();
        let mut pe = BufferedPostingsEnum::new(decoded, true);
        let state = pw
            .write_term(
                &mut pe,
                IndexOptions::DocsAndFreqs,
                &BufferedNormsLookup::no_norms(),
                &mut HashSet::new(),
            )
            .unwrap();

        assert_eq!(state.doc_freq, 3);
        assert_eq!(state.singleton_doc_id, -1);
        assert_eq!(state.total_term_freq, 5); // 1 + 3 + 1
        assert_eq!(state.doc_start_fp, header_size as i64);
        // Something was written to .doc
        assert_gt!(pw.doc_out.file_pointer(), header_size);
    }

    #[test]
    fn test_multi_doc_term_docs_only() {
        let dir = test_directory();
        let mut pw = PostingsWriter::new(&dir, "_0", "", &[0u8; 16], IndexOptions::Docs).unwrap();
        let decoded = build_postings(&[(7, 1), (11, 1)]);
        let header_size = pw.doc_out.file_pointer();
        let mut pe = BufferedPostingsEnum::new(decoded, false);
        let state = pw
            .write_term(
                &mut pe,
                IndexOptions::Docs,
                &BufferedNormsLookup::no_norms(),
                &mut HashSet::new(),
            )
            .unwrap();

        assert_eq!(state.doc_freq, 2);
        assert_eq!(state.singleton_doc_id, -1);
        assert_eq!(state.total_term_freq, -1); // DOCS only
        assert_gt!(pw.doc_out.file_pointer(), header_size);
    }

    #[test]
    fn test_term_with_positions_vint_tail() {
        // totalTermFreq < BLOCK_SIZE uses VInt tail only
        let dir = test_directory();
        let mut pw = PostingsWriter::new(
            &dir,
            "_0",
            "",
            &[0u8; 16],
            IndexOptions::DocsAndFreqsAndPositions,
        )
        .unwrap();
        let decoded = build_postings_with_positions(&[(0, 3, &[0, 5, 10]), (2, 1, &[3])]);
        let mut pe = BufferedPostingsEnum::new(decoded, true);
        let state = pw
            .write_term(
                &mut pe,
                IndexOptions::DocsAndFreqsAndPositions,
                &BufferedNormsLookup::no_norms(),
                &mut HashSet::new(),
            )
            .unwrap();

        assert_eq!(state.doc_freq, 2);
        assert_eq!(state.total_term_freq, 4);
        assert_ge!(state.pos_start_fp, 0);
        // No PFOR blocks: last_pos_block_offset must be -1
        assert_eq!(state.last_pos_block_offset, -1);

        // Finish writing to flush output to directory
        let names = pw.finish().unwrap();
        let pos_name = names.iter().find(|n| n.ends_with(".pos")).unwrap();
        let pos_data = dir.read_file(pos_name).unwrap();
        let pos_header_size = codec_headers::index_header_length(POS_CODEC, "");
        let footer_size = codec_footers::FOOTER_LENGTH;
        let pos_bytes = &pos_data[pos_header_size..pos_data.len() - footer_size];
        // 4 positions with deltas: [0, 5, 10] -> deltas [0, 5, 5], [3] -> delta [3]
        // VInt encoding: 0, 5, 5, 3 = 4 bytes (all single-byte VInts)
        assert_len_eq_x!(&pos_bytes, 4);
        assert_eq!(pos_bytes[0], 0); // pos delta 0
        assert_eq!(pos_bytes[1], 5); // pos delta 5
        assert_eq!(pos_bytes[2], 5); // pos delta 5
        assert_eq!(pos_bytes[3], 3); // pos delta 3 (new doc, last_pos resets)
    }

    #[test]
    fn test_term_with_positions_pfor_one_block() {
        // Ported from Lucene103PostingsWriter: exactly BLOCK_SIZE positions = one PFOR block, no tail
        let dir = test_directory();
        let mut pw = PostingsWriter::new(
            &dir,
            "_0",
            "",
            &[0u8; 16],
            IndexOptions::DocsAndFreqsAndPositions,
        )
        .unwrap();

        // Create a single doc with exactly 128 positions (0, 1, 2, ..., 127)
        let positions: Vec<i32> = (0..postings_format::BLOCK_SIZE as i32).collect();
        let decoded =
            build_postings_with_positions(&[(0, postings_format::BLOCK_SIZE as i32, &positions)]);
        let mut pe = BufferedPostingsEnum::new(decoded, true);

        let state = pw
            .write_term(
                &mut pe,
                IndexOptions::DocsAndFreqsAndPositions,
                &BufferedNormsLookup::no_norms(),
                &mut HashSet::new(),
            )
            .unwrap();

        assert_eq!(state.doc_freq, 1);
        assert_eq!(state.total_term_freq, postings_format::BLOCK_SIZE as i64);
        assert_eq!(state.singleton_doc_id, 0);
        // Exactly BLOCK_SIZE: last_pos_block_offset = -1
        // (Java: totalTermFreq > BLOCK_SIZE is the threshold, not >=)
        assert_eq!(state.last_pos_block_offset, -1);

        // Verify PFOR block was written (pos data starts with a PFOR token byte)
        let pos_out = pw.pos_out.as_ref().unwrap();
        let pos_header_size = codec_headers::index_header_length(POS_CODEC, "");
        let pos_data_len = pos_out.file_pointer() as usize - pos_header_size;
        // Must have written something (PFOR block, not empty)
        assert_gt!(pos_data_len, 0);
    }

    #[test]
    fn test_term_with_positions_pfor_plus_tail() {
        // Ported from Lucene103PostingsWriter: totalTermFreq > BLOCK_SIZE = PFOR blocks + VInt tail
        let dir = test_directory();
        let mut pw = PostingsWriter::new(
            &dir,
            "_0",
            "",
            &[0u8; 16],
            IndexOptions::DocsAndFreqsAndPositions,
        )
        .unwrap();

        // 130 positions across 2 docs: first doc has 128 positions, second has 2
        let positions0: Vec<i32> = (0..128).collect();
        let positions1: Vec<i32> = vec![0, 5];
        let decoded = build_postings_with_positions(&[(0, 128, &positions0), (1, 2, &positions1)]);
        let mut pe = BufferedPostingsEnum::new(decoded, true);

        let state = pw
            .write_term(
                &mut pe,
                IndexOptions::DocsAndFreqsAndPositions,
                &BufferedNormsLookup::no_norms(),
                &mut HashSet::new(),
            )
            .unwrap();

        assert_eq!(state.doc_freq, 2);
        assert_eq!(state.total_term_freq, 130);
        // totalTermFreq > BLOCK_SIZE: last_pos_block_offset must be set (not -1)
        assert_ne!(state.last_pos_block_offset, -1);
        assert_gt!(state.last_pos_block_offset, 0);

        // Verify the .pos file has data (PFOR block + VInt tail)
        let pos_out = pw.pos_out.as_ref().unwrap();
        let pos_header_size = codec_headers::index_header_length(POS_CODEC, "");
        let pos_data_len = pos_out.file_pointer() as usize - pos_header_size;
        assert_gt!(
            pos_data_len,
            0,
            "PFOR block + VInt tail should produce output"
        );

        // last_pos_block_offset points to the start of the VInt tail
        // It must be less than the total pos data length (tail comes after blocks)
        assert_lt!(
            state.last_pos_block_offset as usize,
            pos_data_len,
            "last_pos_block_offset ({}) should be < total pos data ({})",
            state.last_pos_block_offset,
            pos_data_len
        );
    }

    #[test]
    fn test_term_with_positions_pfor_multiple_blocks() {
        // Ported from Lucene103PostingsWriter: multiple PFOR blocks + tail
        let dir = test_directory();
        let mut pw = PostingsWriter::new(
            &dir,
            "_0",
            "",
            &[0u8; 16],
            IndexOptions::DocsAndFreqsAndPositions,
        )
        .unwrap();

        // 300 positions: 2 full PFOR blocks (256) + 44 VInt tail
        let positions: Vec<i32> = (0..300).collect();
        let decoded = build_postings_with_positions(&[(0, 300, &positions)]);
        let mut pe = BufferedPostingsEnum::new(decoded, true);

        let state = pw
            .write_term(
                &mut pe,
                IndexOptions::DocsAndFreqsAndPositions,
                &BufferedNormsLookup::no_norms(),
                &mut HashSet::new(),
            )
            .unwrap();

        assert_eq!(state.doc_freq, 1);
        assert_eq!(state.total_term_freq, 300);
        assert_ne!(state.last_pos_block_offset, -1);

        let pos_out = pw.pos_out.as_ref().unwrap();
        let pos_header_size = codec_headers::index_header_length(POS_CODEC, "");
        let pos_data_len = pos_out.file_pointer() as usize - pos_header_size;

        // Must have written data (2 PFOR blocks + 44 VInt tail)
        assert_gt!(
            pos_data_len,
            44,
            "2 PFOR blocks + tail should be larger than tail alone"
        );

        // last_pos_block_offset should point past the 2 PFOR blocks, before the VInt tail
        assert_lt!(
            state.last_pos_block_offset as usize,
            pos_data_len,
            "last_pos_block_offset ({}) should be < total pos data ({})",
            state.last_pos_block_offset,
            pos_data_len
        );
    }

    #[test]
    fn test_term_with_positions_exactly_two_blocks() {
        // Ported from Lucene103PostingsWriter: exactly 2*BLOCK_SIZE positions = 2 PFOR blocks, no tail
        let dir = test_directory();
        let mut pw = PostingsWriter::new(
            &dir,
            "_0",
            "",
            &[0u8; 16],
            IndexOptions::DocsAndFreqsAndPositions,
        )
        .unwrap();

        let positions: Vec<i32> = (0..256).collect();
        let decoded = build_postings_with_positions(&[(0, 256, &positions)]);
        let mut pe = BufferedPostingsEnum::new(decoded, true);

        let state = pw
            .write_term(
                &mut pe,
                IndexOptions::DocsAndFreqsAndPositions,
                &BufferedNormsLookup::no_norms(),
                &mut HashSet::new(),
            )
            .unwrap();

        assert_eq!(state.total_term_freq, 256);
        assert_ne!(state.last_pos_block_offset, -1);

        let pos_out = pw.pos_out.as_ref().unwrap();
        let pos_header_size = codec_headers::index_header_length(POS_CODEC, "");
        let pos_data_len = pos_out.file_pointer() as usize - pos_header_size;

        // 2 PFOR blocks, no VInt tail
        assert_gt!(pos_data_len, 0);

        // last_pos_block_offset should equal total length (tail is empty, offset is at end)
        assert_eq!(
            state.last_pos_block_offset as usize, pos_data_len,
            "With no VInt tail, last_pos_block_offset should equal total pos data length"
        );
    }

    #[test]
    fn test_term_with_positions_multi_doc_block_boundary() {
        // Verify position deltas reset per document when spanning a PFOR block boundary
        let dir = test_directory();
        let mut pw = PostingsWriter::new(
            &dir,
            "_0",
            "",
            &[0u8; 16],
            IndexOptions::DocsAndFreqsAndPositions,
        )
        .unwrap();

        // Doc 0: 100 positions (0..100), Doc 1: 50 positions (0..50)
        // Total = 150 > BLOCK_SIZE, so we get 1 PFOR block + 22 VInt tail
        // Crucially, last_pos resets at doc boundary: doc1's first delta is 0, not (0 - 99)
        let positions0: Vec<i32> = (0..100).collect();
        let positions1: Vec<i32> = (0..50).collect();
        let decoded = build_postings_with_positions(&[(0, 100, &positions0), (1, 50, &positions1)]);
        let mut pe = BufferedPostingsEnum::new(decoded, true);

        let state = pw
            .write_term(
                &mut pe,
                IndexOptions::DocsAndFreqsAndPositions,
                &BufferedNormsLookup::no_norms(),
                &mut HashSet::new(),
            )
            .unwrap();

        assert_eq!(state.total_term_freq, 150);
        assert_ne!(state.last_pos_block_offset, -1);

        // Verify positions were written (no panic from negative deltas)
        let pos_out = pw.pos_out.as_ref().unwrap();
        let pos_header_size = codec_headers::index_header_length(POS_CODEC, "");
        assert_gt!(pos_out.file_pointer() as usize, pos_header_size);
    }

    #[test]
    fn test_term_with_positions_large_deltas() {
        // Ported from TestPForUtil: positions with large gaps need more bits per value
        let dir = test_directory();
        let mut pw = PostingsWriter::new(
            &dir,
            "_0",
            "",
            &[0u8; 16],
            IndexOptions::DocsAndFreqsAndPositions,
        )
        .unwrap();

        // 128 positions with gaps of 1000 each — requires ~10 bits per value
        let positions: Vec<i32> = (0..128).map(|i| i * 1000).collect();
        let decoded = build_postings_with_positions(&[(0, 128, &positions)]);
        let mut pe = BufferedPostingsEnum::new(decoded, true);

        let state = pw
            .write_term(
                &mut pe,
                IndexOptions::DocsAndFreqsAndPositions,
                &BufferedNormsLookup::no_norms(),
                &mut HashSet::new(),
            )
            .unwrap();

        assert_eq!(state.total_term_freq, 128);
        // Exactly BLOCK_SIZE: last_pos_block_offset = -1
        assert_eq!(state.last_pos_block_offset, -1);

        let pos_out = pw.pos_out.as_ref().unwrap();
        let pos_header_size = codec_headers::index_header_length(POS_CODEC, "");
        let pos_data_len = pos_out.file_pointer() as usize - pos_header_size;

        // Deltas are all 1000 (except first which is 0). bits_required(1000) = 10.
        // PFOR block: token(1) + 10*128/8 = 1 + 160 = 161 bytes
        assert!(
            pos_data_len > 16 && pos_data_len < 200,
            "PFOR block with 10bpv should be ~161 bytes, got {}",
            pos_data_len
        );
    }

    #[test]
    fn test_term_with_positions_pfor_compresses_real_data() {
        // Verify PFOR is more compact than VInt for realistic position data
        let dir = test_directory();
        let mut pw = PostingsWriter::new(
            &dir,
            "_0",
            "",
            &[0u8; 16],
            IndexOptions::DocsAndFreqsAndPositions,
        )
        .unwrap();

        // 256 positions with varied gaps (simulating real text positions)
        // Mix of common deltas (1-5) with occasional larger gaps
        let mut positions = Vec::with_capacity(256);
        let mut pos = 0i32;
        for i in 0..256 {
            positions.push(pos);
            // Vary the gap: mostly small (1-3), occasionally larger
            pos += match i % 10 {
                0 => 15,
                1 => 8,
                _ => 1 + (i % 3),
            };
        }
        let decoded = build_postings_with_positions(&[(0, 256, &positions)]);
        let mut pe = BufferedPostingsEnum::new(decoded, true);

        let state = pw
            .write_term(
                &mut pe,
                IndexOptions::DocsAndFreqsAndPositions,
                &BufferedNormsLookup::no_norms(),
                &mut HashSet::new(),
            )
            .unwrap();

        assert_eq!(state.total_term_freq, 256);
        assert_ne!(state.last_pos_block_offset, -1);

        let pos_out = pw.pos_out.as_ref().unwrap();
        let pos_header_size = codec_headers::index_header_length(POS_CODEC, "");
        let pos_data_len = pos_out.file_pointer() as usize - pos_header_size;

        // With varied deltas (max ~15, ~4 bpv), PFOR should be much smaller than VInt.
        // VInt for 256 values averaging ~2 bytes each = ~512 bytes.
        // PFOR for 2 blocks at ~4bpv each = 2*(1 + 64) = ~130 bytes.
        assert_lt!(
            pos_data_len,
            256,
            "PFOR should compress 256 varied position deltas below 256 bytes, got {}",
            pos_data_len
        );
    }

    #[test]
    fn test_encode_term_singleton() {
        let dir = test_directory();
        let pw = PostingsWriter::new(&dir, "_0", "", &[0u8; 16], IndexOptions::Docs).unwrap();

        let empty = IntBlockTermState::new();
        let state = IntBlockTermState {
            doc_freq: 1,
            total_term_freq: 1,
            term_block_ord: 0,
            doc_start_fp: 0,
            pos_start_fp: 0,
            pay_start_fp: 0,
            last_pos_block_offset: -1,
            singleton_doc_id: 5,
        };

        let mut buf = Vec::new();
        pw.encode_term(&mut buf, &state, &empty, false, false)
            .unwrap();

        // First VLong: (doc_start_fp_delta << 1) = (0 << 1) = 0
        assert_eq!(buf[0], 0);
        // Then VInt(singletonDocID=5)
        assert_eq!(buf[1], 5);
    }

    #[test]
    fn test_finish_produces_files() {
        let dir = test_directory();
        let pw = PostingsWriter::new(
            &dir,
            "_0",
            "",
            &[0u8; 16],
            IndexOptions::DocsAndFreqsAndPositions,
        )
        .unwrap();
        let names = pw.finish().unwrap();

        // Should produce .doc, .pos, .psm files
        assert_len_eq_x!(&names, 3);
        assert_ends_with!(names[0], ".doc");
        assert_ends_with!(names[1], ".pos");
        assert_ends_with!(names[2], ".psm");

        // Each file should have at least header + footer
        for name in &names {
            let data = dir.read_file(name).unwrap();
            assert_ge!(
                data.len(),
                codec_footers::FOOTER_LENGTH,
                "file {name} too small: {} bytes",
                data.len()
            );
        }
    }

    // --- Tests for vint15/vlong15 helpers ---

    #[test]
    fn test_vint15_small_value() {
        // Value <= 0x7FFF fits in 2 bytes (a short)
        let mut buf = Vec::new();
        write_vint15(&mut VecOutput(&mut buf), 0).unwrap();
        assert_len_eq_x!(&buf, 2); // always 2 bytes (a short)

        let mut buf = Vec::new();
        write_vint15(&mut VecOutput(&mut buf), 0x7FFF).unwrap();
        assert_len_eq_x!(&buf, 2);
    }

    #[test]
    fn test_vint15_large_value() {
        // Value > 0x7FFF needs 2 bytes (short with high bit) + VLong
        let mut buf = Vec::new();
        write_vint15(&mut VecOutput(&mut buf), 0x8000).unwrap();
        assert_gt!(buf.len(), 2); // short + at least 1 vlong byte
        // High bit of first short should be set
        // Low byte first, high byte second
        assert_ne!(buf[1] & 0x80, 0, "high bit of short should be set");
    }

    #[test]
    fn test_vlong15_roundtrip() {
        // Verify various values encode without panic
        for v in [0i64, 1, 127, 128, 0x7FFF, 0x8000, 0xFFFF, 100000] {
            let mut buf = Vec::new();
            write_vlong15(&mut VecOutput(&mut buf), v).unwrap();
            assert_not_empty!(buf);
        }
    }

    // --- Tests for write_impacts ---

    #[test]
    fn test_write_impacts_single_norm_one() {
        // Single impact with freq=1, norm=1
        // freqDelta = 1 - 0 - 1 = 0, normDelta = 1 - 0 - 1 = 0
        // Encoding: VInt(0 << 1) = VInt(0)
        let impacts = [Impact { freq: 1, norm: 1 }];
        let mut buf = Vec::new();
        let len = write_impacts(&impacts, &mut buf).unwrap();
        assert_eq!(len, 1);
        assert_eq!(buf[0], 0);

        // Single impact with freq=5, norm=1
        // freqDelta = 5 - 0 - 1 = 4, normDelta = 1 - 0 - 1 = 0
        // Encoding: VInt(4 << 1) = VInt(8)
        let impacts = [Impact { freq: 5, norm: 1 }];
        let mut buf = Vec::new();
        let len = write_impacts(&impacts, &mut buf).unwrap();
        assert_eq!(len, 1);
        assert_eq!(buf[0], 8);

        // Larger freq
        // freqDelta = 100 - 0 - 1 = 99, normDelta = 0
        // VInt(99 << 1) = VInt(198) = 2 bytes
        let impacts = [Impact { freq: 100, norm: 1 }];
        let mut buf = Vec::new();
        let len = write_impacts(&impacts, &mut buf).unwrap();
        assert_eq!(len, 2);
    }

    #[test]
    fn test_write_impacts_with_norm() {
        // Single impact with freq=3, norm=5
        // freqDelta = 3 - 0 - 1 = 2, normDelta = 5 - 0 - 1 = 4
        // Encoding: VInt((2 << 1) | 1) = VInt(5), ZLong(4) = VLong(zigzag(4)) = VLong(8)
        let impacts = [Impact { freq: 3, norm: 5 }];
        let mut buf = Vec::new();
        let len = write_impacts(&impacts, &mut buf).unwrap();
        assert_ge!(len, 2); // At least VInt + ZLong
        assert_eq!(buf[0], 5); // (2 << 1) | 1

        // Two impacts: delta-encoded
        let impacts = [Impact { freq: 3, norm: 5 }, Impact { freq: 10, norm: 13 }];
        let mut buf = Vec::new();
        let len = write_impacts(&impacts, &mut buf).unwrap();
        assert_gt!(len, 2); // Two impacts encoded
    }

    // --- Tests for block encoding (docFreq >= 128) ---

    #[test]
    fn test_block_encoding_128_docs() {
        // Exactly 128 docs: 1 full block, no VInt tail
        let dir = test_directory();
        let mut pw = PostingsWriter::new(&dir, "_0", "", &[0u8; 16], IndexOptions::Docs).unwrap();
        let decoded = build_postings(&(0..128).map(|i| (i, 1)).collect::<Vec<_>>());
        let header_size = pw.doc_out.file_pointer();
        let mut pe = BufferedPostingsEnum::new(decoded, true);
        let state = pw
            .write_term(
                &mut pe,
                IndexOptions::DocsAndFreqs,
                &BufferedNormsLookup::no_norms(),
                &mut HashSet::new(),
            )
            .unwrap();

        assert_eq!(state.doc_freq, 128);
        assert_eq!(state.singleton_doc_id, -1);
        assert_gt!(pw.doc_out.file_pointer(), header_size);
        // Should have written block data (skip header + doc encoding + freq encoding)
        // Consecutive docs with freq=1 are very compact (~10 bytes per block)
        let written = pw.doc_out.file_pointer() - header_size;
        assert_ge!(
            written,
            8,
            "block encoding should produce output, got {written}"
        );
    }

    #[test]
    fn test_block_encoding_130_docs() {
        // 130 docs: 1 full block + 2 VInt tail docs
        let dir = test_directory();
        let mut pw = PostingsWriter::new(&dir, "_0", "", &[0u8; 16], IndexOptions::Docs).unwrap();
        let decoded = build_postings(&(0..130).map(|i| (i, 1)).collect::<Vec<_>>());
        let header_size = pw.doc_out.file_pointer();
        let mut pe = BufferedPostingsEnum::new(decoded, true);
        let state = pw
            .write_term(
                &mut pe,
                IndexOptions::DocsAndFreqs,
                &BufferedNormsLookup::no_norms(),
                &mut HashSet::new(),
            )
            .unwrap();

        assert_eq!(state.doc_freq, 130);
        assert_eq!(state.singleton_doc_id, -1);
        let written = pw.doc_out.file_pointer() - header_size;
        assert_gt!(
            written,
            10,
            "block + tail should produce output, got {written}"
        );
    }

    #[test]
    fn test_block_encoding_256_docs() {
        // 256 docs: 2 full blocks, no tail
        let dir = test_directory();
        let mut pw = PostingsWriter::new(&dir, "_0", "", &[0u8; 16], IndexOptions::Docs).unwrap();
        let decoded = build_postings(&(0..256).map(|i| (i, 1)).collect::<Vec<_>>());
        let header_size = pw.doc_out.file_pointer();
        let mut pe = BufferedPostingsEnum::new(decoded, true);
        let state = pw
            .write_term(
                &mut pe,
                IndexOptions::DocsAndFreqs,
                &BufferedNormsLookup::no_norms(),
                &mut HashSet::new(),
            )
            .unwrap();

        assert_eq!(state.doc_freq, 256);
        let written = pw.doc_out.file_pointer() - header_size;
        // 2 blocks should produce more output than 1
        assert_ge!(written, 16);
    }

    #[test]
    fn test_block_encoding_with_positions() {
        // 128+ docs with positions: verify pos skip data is present
        let dir = test_directory();
        let mut pw = PostingsWriter::new(
            &dir,
            "_0",
            "",
            &[0u8; 16],
            IndexOptions::DocsAndFreqsAndPositions,
        )
        .unwrap();
        // Each doc has 2 positions [0, 1]
        let docs_with_pos: Vec<(i32, i32, &[i32])> =
            (0..128).map(|i| (i, 2, [0i32, 1].as_slice())).collect();
        let decoded = build_postings_with_positions(&docs_with_pos);
        let mut pe = BufferedPostingsEnum::new(decoded, true);

        let state = pw
            .write_term(
                &mut pe,
                IndexOptions::DocsAndFreqsAndPositions,
                &BufferedNormsLookup::no_norms(),
                &mut HashSet::new(),
            )
            .unwrap();

        assert_eq!(state.doc_freq, 128);
        assert_eq!(state.total_term_freq, 256); // 128 * 2
        // totalTermFreq > 128, so last_pos_block_offset should be set
        assert_ne!(state.last_pos_block_offset, -1);

        // Verify pos data was written
        let pos_out = pw.pos_out.as_ref().unwrap();
        let pos_header_size = codec_headers::index_header_length(POS_CODEC, "");
        assert_gt!(pos_out.file_pointer() as usize, pos_header_size);
    }

    #[test]
    fn test_block_encoding_consecutive_docs() {
        // Docs 0..127 with deltas all = 1 → doc_range == 128 == BLOCK_SIZE
        // This should use the byte(0) consecutive encoding
        let dir = test_directory();
        let mut pw = PostingsWriter::new(&dir, "_0", "", &[0u8; 16], IndexOptions::Docs).unwrap();
        // doc IDs: 0, 1, 2, ..., 127 — deltas from -1 are 1, 1, 1, ... = range 128
        let decoded = build_postings(&(0..128).map(|i| (i, 1)).collect::<Vec<_>>());
        let mut pe = BufferedPostingsEnum::new(decoded, true);
        let state = pw
            .write_term(
                &mut pe,
                IndexOptions::DocsAndFreqs,
                &BufferedNormsLookup::no_norms(),
                &mut HashSet::new(),
            )
            .unwrap();

        assert_eq!(state.doc_freq, 128);
        // The block encoding should succeed (we can't easily inspect the byte,
        // but the fact that it doesn't panic or error is the key assertion)
    }

    #[test]
    fn test_block_encoding_sparse_docs() {
        // Sparse doc IDs with large deltas → uses FOR encoding
        let dir = test_directory();
        let mut pw = PostingsWriter::new(&dir, "_0", "", &[0u8; 16], IndexOptions::Docs).unwrap();
        // doc IDs: 0, 100, 200, ..., 12700 — delta = 100 each (except first which is 1 from -1)
        let decoded = build_postings(&(0..128).map(|i| (i * 100, 1)).collect::<Vec<_>>());
        let mut pe = BufferedPostingsEnum::new(decoded, true);
        let state = pw
            .write_term(
                &mut pe,
                IndexOptions::DocsAndFreqs,
                &BufferedNormsLookup::no_norms(),
                &mut HashSet::new(),
            )
            .unwrap();

        assert_eq!(state.doc_freq, 128);
        assert_eq!(state.singleton_doc_id, -1);
    }

    #[test]
    fn test_block_encoding_docs_only() {
        // DOCS-only index option (no freqs) with 128 docs
        let dir = test_directory();
        let mut pw = PostingsWriter::new(&dir, "_0", "", &[0u8; 16], IndexOptions::Docs).unwrap();
        let decoded = build_postings(&(0..128).map(|i| (i, 1)).collect::<Vec<_>>());
        let mut pe = BufferedPostingsEnum::new(decoded, false);
        let state = pw
            .write_term(
                &mut pe,
                IndexOptions::Docs,
                &BufferedNormsLookup::no_norms(),
                &mut HashSet::new(),
            )
            .unwrap();

        assert_eq!(state.doc_freq, 128);
        assert_eq!(state.total_term_freq, -1);
        assert_eq!(state.singleton_doc_id, -1);
    }

    #[test]
    fn test_block_encoding_varied_freqs() {
        // 128 docs with varied frequencies
        let dir = test_directory();
        let mut pw = PostingsWriter::new(&dir, "_0", "", &[0u8; 16], IndexOptions::Docs).unwrap();
        let decoded = build_postings(&(0..128).map(|i| (i, (i % 10) + 1)).collect::<Vec<_>>());
        let mut pe = BufferedPostingsEnum::new(decoded, true);
        let state = pw
            .write_term(
                &mut pe,
                IndexOptions::DocsAndFreqs,
                &BufferedNormsLookup::no_norms(),
                &mut HashSet::new(),
            )
            .unwrap();

        assert_eq!(state.doc_freq, 128);
        let expected_ttf: i64 = (0..128).map(|i: i64| (i % 10) + 1).sum();
        assert_eq!(state.total_term_freq, expected_ttf);
    }

    #[test]
    fn test_psm_level1_impact_metadata_zero_below_threshold() {
        // Java only updates level 1 impact metadata inside writeLevel1SkipData(),
        // which triggers every 4096 docs. For terms below that threshold, the
        // level 1 fields must be 0. This test writes 128 docs with varied freqs
        // (enough to produce non-zero level 0 impacts) and verifies level 1 stays 0.
        let dir = test_directory();
        let mut pw = PostingsWriter::new(&dir, "_0", "", &[0u8; 16], IndexOptions::Docs).unwrap();
        let decoded = build_postings(&(0..128).map(|i| (i, (i % 10) + 1)).collect::<Vec<_>>());
        let mut pe = BufferedPostingsEnum::new(decoded, true);
        pw.write_term(
            &mut pe,
            IndexOptions::DocsAndFreqs,
            &BufferedNormsLookup::no_norms(),
            &mut HashSet::new(),
        )
        .unwrap();

        let names = pw.finish().unwrap();
        let psm_name = names.iter().find(|n| n.ends_with(".psm")).unwrap();
        let psm_data = dir.read_file(psm_name).unwrap();

        // Parse the 4 LE i32 metadata fields after the codec header
        let header_size = codec_headers::index_header_length(META_CODEC, "");
        let meta = &psm_data[header_size..];
        let max_num_impacts_l0 = i32::from_le_bytes(meta[0..4].try_into().unwrap());
        let max_impact_bytes_l0 = i32::from_le_bytes(meta[4..8].try_into().unwrap());
        let max_num_impacts_l1 = i32::from_le_bytes(meta[8..12].try_into().unwrap());
        let max_impact_bytes_l1 = i32::from_le_bytes(meta[12..16].try_into().unwrap());

        // Level 0 should be non-zero (128 docs with freqs triggers impact computation)
        assert_gt!(
            max_num_impacts_l0,
            0,
            "level 0 impact count should be > 0, got {max_num_impacts_l0}"
        );
        assert_gt!(
            max_impact_bytes_l0,
            0,
            "level 0 impact bytes should be > 0, got {max_impact_bytes_l0}"
        );

        // Level 1 must be 0 (no level 1 skip triggered for < 4096 docs)
        assert_eq!(
            max_num_impacts_l1, 0,
            "level 1 impact count should be 0, got {max_num_impacts_l1}"
        );
        assert_eq!(
            max_impact_bytes_l1, 0,
            "level 1 impact bytes should be 0, got {max_impact_bytes_l1}"
        );
    }

    // --- Tests for level1 skip data (docFreq > 4096) ---

    #[test]
    fn test_level1_skip_data_with_freqs() {
        // 5000 docs crosses one level1 boundary (4096).
        // Verifies write_level1_skip_data fires and produces valid output.
        let dir = test_directory();
        let mut pw = PostingsWriter::new(&dir, "_0", "", &[0u8; 16], IndexOptions::Docs).unwrap();
        let decoded = build_postings(&(0..5000).map(|i| (i, (i % 10) + 1)).collect::<Vec<_>>());
        let header_size = pw.doc_out.file_pointer();
        let mut pe = BufferedPostingsEnum::new(decoded, true);
        let state = pw
            .write_term(
                &mut pe,
                IndexOptions::DocsAndFreqs,
                &BufferedNormsLookup::no_norms(),
                &mut HashSet::new(),
            )
            .unwrap();

        assert_eq!(state.doc_freq, 5000);
        assert_eq!(state.singleton_doc_id, -1);
        let expected_ttf: i64 = (0..5000).map(|i: i64| (i % 10) + 1).sum();
        assert_eq!(state.total_term_freq, expected_ttf);
        // Data was written to .doc
        assert_gt!(pw.doc_out.file_pointer(), header_size);
    }

    #[test]
    fn test_level1_skip_data_docs_only() {
        // 5000 docs with DOCS-only (no freqs) — exercises the !write_freqs branch
        // in write_level1_skip_data.
        let dir = test_directory();
        let mut pw = PostingsWriter::new(&dir, "_0", "", &[0u8; 16], IndexOptions::Docs).unwrap();
        let decoded = build_postings(&(0..5000).map(|i| (i, 1)).collect::<Vec<_>>());
        let header_size = pw.doc_out.file_pointer();
        let mut pe = BufferedPostingsEnum::new(decoded, false);
        let state = pw
            .write_term(
                &mut pe,
                IndexOptions::Docs,
                &BufferedNormsLookup::no_norms(),
                &mut HashSet::new(),
            )
            .unwrap();

        assert_eq!(state.doc_freq, 5000);
        assert_eq!(state.total_term_freq, -1);
        assert_eq!(state.singleton_doc_id, -1);
        assert_gt!(pw.doc_out.file_pointer(), header_size);
    }

    #[test]
    fn test_level1_skip_data_with_positions() {
        // 5000 docs with positions — exercises the write_positions branch
        // in write_level1_skip_data (pos FP delta + pos_buffer_upto).
        let dir = test_directory();
        let mut pw = PostingsWriter::new(
            &dir,
            "_0",
            "",
            &[0u8; 16],
            IndexOptions::DocsAndFreqsAndPositions,
        )
        .unwrap();
        let docs_with_pos: Vec<(i32, i32, &[i32])> =
            (0..5000).map(|i| (i, 2, [0i32, 1].as_slice())).collect();
        let decoded = build_postings_with_positions(&docs_with_pos);
        let mut pe = BufferedPostingsEnum::new(decoded, true);
        let state = pw
            .write_term(
                &mut pe,
                IndexOptions::DocsAndFreqsAndPositions,
                &BufferedNormsLookup::no_norms(),
                &mut HashSet::new(),
            )
            .unwrap();

        assert_eq!(state.doc_freq, 5000);
        assert_eq!(state.total_term_freq, 10000);
        assert_ne!(state.last_pos_block_offset, -1);
    }

    #[test]
    fn test_level1_skip_data_exactly_4096() {
        // Exactly 4096 docs — triggers level1 skip data once, with no remainder.
        let dir = test_directory();
        let mut pw = PostingsWriter::new(&dir, "_0", "", &[0u8; 16], IndexOptions::Docs).unwrap();
        let decoded = build_postings(
            &(0..postings_format::LEVEL1_NUM_DOCS as i32)
                .map(|i| (i, (i % 5) + 1))
                .collect::<Vec<_>>(),
        );
        let mut pe = BufferedPostingsEnum::new(decoded, true);
        let state = pw
            .write_term(
                &mut pe,
                IndexOptions::DocsAndFreqs,
                &BufferedNormsLookup::no_norms(),
                &mut HashSet::new(),
            )
            .unwrap();

        assert_eq!(state.doc_freq, postings_format::LEVEL1_NUM_DOCS as i32);
        assert_eq!(state.singleton_doc_id, -1);
    }

    #[test]
    fn test_psm_level1_impact_metadata_nonzero_above_threshold() {
        // With 5000 docs and varied freqs, level1 skip data fires and
        // level1 impact metadata in .psm must be non-zero.
        let dir = test_directory();
        let mut pw = PostingsWriter::new(&dir, "_0", "", &[0u8; 16], IndexOptions::Docs).unwrap();
        let decoded = build_postings(&(0..5000).map(|i| (i, (i % 10) + 1)).collect::<Vec<_>>());
        let mut pe = BufferedPostingsEnum::new(decoded, true);
        pw.write_term(
            &mut pe,
            IndexOptions::DocsAndFreqs,
            &BufferedNormsLookup::no_norms(),
            &mut HashSet::new(),
        )
        .unwrap();

        let names = pw.finish().unwrap();
        let psm_name = names.iter().find(|n| n.ends_with(".psm")).unwrap();
        let psm_data = dir.read_file(psm_name).unwrap();

        let header_size = codec_headers::index_header_length(META_CODEC, "");
        let meta = &psm_data[header_size..];
        let max_num_impacts_l1 = i32::from_le_bytes(meta[8..12].try_into().unwrap());
        let max_impact_bytes_l1 = i32::from_le_bytes(meta[12..16].try_into().unwrap());

        assert_gt!(
            max_num_impacts_l1,
            0,
            "level 1 impact count should be > 0 for 5000-doc term"
        );
        assert_gt!(
            max_impact_bytes_l1,
            0,
            "level 1 impact bytes should be > 0 for 5000-doc term"
        );
    }

    // --- Offset encoding tests ---

    #[test]
    fn test_singleton_with_offsets() {
        let dir = test_directory();
        let mut pw = PostingsWriter::new(
            &dir,
            "_0",
            "",
            &[0u8; 16],
            IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
        )
        .unwrap();
        let offsets = [TermOffset {
            start: 0,
            length: 5,
        }];
        let decoded = build_postings_with_offsets(&[(0, 1, &[0], &offsets)]);
        let mut pe = BufferedPostingsEnum::new(decoded, true);
        let state = pw
            .write_term(
                &mut pe,
                IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
                &BufferedNormsLookup::no_norms(),
                &mut HashSet::new(),
            )
            .unwrap();

        assert_eq!(state.doc_freq, 1);
        assert_eq!(state.singleton_doc_id, 0);
        assert_ge!(state.pay_start_fp, 0);
    }

    #[test]
    fn test_vint_tail_with_offsets() {
        let dir = test_directory();
        let mut pw = PostingsWriter::new(
            &dir,
            "_0",
            "",
            &[0u8; 16],
            IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
        )
        .unwrap();
        let offsets0 = [
            TermOffset {
                start: 0,
                length: 5,
            },
            TermOffset {
                start: 6,
                length: 5,
            },
            TermOffset {
                start: 12,
                length: 5,
            },
        ];
        let offsets1 = [TermOffset {
            start: 0,
            length: 3,
        }];
        let decoded =
            build_postings_with_offsets(&[(0, 3, &[0, 5, 10], &offsets0), (2, 1, &[3], &offsets1)]);
        let mut pe = BufferedPostingsEnum::new(decoded, true);
        let state = pw
            .write_term(
                &mut pe,
                IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
                &BufferedNormsLookup::no_norms(),
                &mut HashSet::new(),
            )
            .unwrap();

        assert_eq!(state.doc_freq, 2);
        assert_eq!(state.total_term_freq, 4);
        // VInt tail: last_pos_block_offset = -1 (< BLOCK_SIZE positions)
        assert_eq!(state.last_pos_block_offset, -1);

        // Verify .pos file has offset data (more bytes than positions alone)
        let names = pw.finish().unwrap();
        let pos_name = names.iter().find(|n| n.ends_with(".pos")).unwrap();
        let pos_data = dir.read_file(pos_name).unwrap();
        let pos_header_size = codec_headers::index_header_length(POS_CODEC, "");
        let footer_size = codec_footers::FOOTER_LENGTH;
        let pos_bytes = &pos_data[pos_header_size..pos_data.len() - footer_size];
        // 4 positions + offset VInts: must be > 4 bytes (positions alone would be 4)
        assert_gt!(
            pos_bytes.len(),
            4,
            "offset VInt data should increase .pos size beyond position-only"
        );
    }

    #[test]
    fn test_block_encoding_with_offsets() {
        // 128 docs each with 2 positions and offsets — exercises PFOR offset blocks in .pay
        let dir = test_directory();
        let mut pw = PostingsWriter::new(
            &dir,
            "_0",
            "",
            &[0u8; 16],
            IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
        )
        .unwrap();
        let positions = [0i32, 1];
        let offsets = [
            TermOffset {
                start: 0,
                length: 5,
            },
            TermOffset {
                start: 6,
                length: 5,
            },
        ];
        let docs: Vec<(i32, i32, &[i32], &[TermOffset])> = (0..128)
            .map(|i| (i, 2, positions.as_slice(), offsets.as_slice()))
            .collect();
        let decoded = build_postings_with_offsets(&docs);
        let mut pe = BufferedPostingsEnum::new(decoded, true);
        let state = pw
            .write_term(
                &mut pe,
                IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
                &BufferedNormsLookup::no_norms(),
                &mut HashSet::new(),
            )
            .unwrap();

        assert_eq!(state.doc_freq, 128);
        assert_eq!(state.total_term_freq, 256);
        assert_ne!(state.last_pos_block_offset, -1);

        let names = pw.finish().unwrap();
        // Must have .pay file
        let pay_name = names.iter().find(|n| n.ends_with(".pay"));
        assert_some!(&pay_name);
        let pay_data = dir.read_file(pay_name.unwrap()).unwrap();
        let pay_header_size = codec_headers::index_header_length(PAY_CODEC, "");
        let footer_size = codec_footers::FOOTER_LENGTH;
        let pay_bytes = &pay_data[pay_header_size..pay_data.len() - footer_size];
        // PFOR offset blocks should have produced data
        assert_gt!(
            pay_bytes.len(),
            0,
            "PFOR offset blocks should produce .pay data"
        );
    }

    #[test]
    fn test_finish_produces_pay_file() {
        let dir = test_directory();
        let pw = PostingsWriter::new(
            &dir,
            "_0",
            "",
            &[0u8; 16],
            IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
        )
        .unwrap();
        let names = pw.finish().unwrap();

        // Should produce .doc, .pos, .pay, .psm files
        assert_len_eq_x!(&names, 4);
        assert_ends_with!(names[0], ".doc");
        assert_ends_with!(names[1], ".pos");
        assert_ends_with!(names[2], ".pay");
        assert_ends_with!(names[3], ".psm");
    }
}
