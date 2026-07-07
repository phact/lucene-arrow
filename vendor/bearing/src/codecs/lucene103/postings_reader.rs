// SPDX-License-Identifier: Apache-2.0

//! Postings reader for the Lucene103 postings format.
//!
//! Reads `.psm` (metadata), `.doc` (document IDs), and `.pos` (positions) files
//! written by [`super::postings_writer::PostingsWriter`]. Only metadata and file
//! headers are read during construction; posting list data is read lazily via
//! [`BlockPostingsEnum`].

use std::io;

use log::debug;

use crate::codecs::codec_file_handle::{CodecFileHandle, IndexFile};
use crate::codecs::codec_headers;
use crate::codecs::competitive_impact::Impact;
use crate::codecs::lucene103::postings_format::{self, LEVEL1_NUM_DOCS};
use crate::encoding::group_vint;
use crate::encoding::pfor;
use crate::encoding::pfor::BLOCK_SIZE;
use crate::index::FieldInfos;
use crate::search::doc_id_set_iterator::{DocIdSetIterator, NO_MORE_DOCS};
use crate::search::scorer::{Impacts, ImpactsSource};
use crate::store::{Directory, FileBacking, IndexInput};

/// Dummy impacts: maximum possible term frequency and lowest possible unsigned norm.
/// Used on tail blocks that don't record impacts.
const DUMMY_IMPACTS: [Impact; 1] = [Impact {
    freq: i32::MAX,
    norm: 1,
}];

/// Impacts when there is no frequency: max frequency is 1.
const IMPACTS_NO_FREQ: [Impact; 1] = [Impact { freq: 1, norm: 1 }];

/// Impact statistics read from the `.psm` metadata file.
#[derive(Debug, Clone, Copy)]
pub struct ImpactStats {
    max_num_impacts_at_level0: i32,
    max_impact_num_bytes_at_level0: i32,
    max_num_impacts_at_level1: i32,
    max_impact_num_bytes_at_level1: i32,
}

/// Per-field index feature flags passed to [`BlockPostingsEnum`].
#[derive(Debug, Clone, Copy)]
pub struct IndexFeatures {
    /// Whether the field indexes term frequencies.
    pub has_freq: bool,
    /// Whether the field indexes positions.
    pub has_pos: bool,
    /// Whether the field indexes offsets or payloads.
    pub has_offsets_or_payloads: bool,
}

/// Reads postings metadata for a segment and provides access to doc ID iteration.
///
/// Opens `.psm`, `.doc`, and `.pos` files during construction. Only the `.psm`
/// metadata (impact statistics) is read eagerly; `.doc` and `.pos` file handles
/// are retained for lazy posting list reads via [`BlockPostingsEnum`].
pub struct PostingsReader {
    /// Owned bytes of the `.doc` file. `IndexInput<'_>` views are constructed on
    /// demand for each [`BlockPostingsEnum`].
    doc: FileBacking,
    /// Impact statistics from the `.psm` metadata file.
    impact_stats: ImpactStats,
}

impl PostingsReader {
    /// Opens postings files (`.psm`, `.doc`, `.pos`) for the given segment.
    pub fn open(
        directory: &dyn Directory,
        segment_name: &str,
        segment_suffix: &str,
        segment_id: &[u8; codec_headers::ID_LENGTH],
        field_infos: &FieldInfos,
    ) -> io::Result<Self> {
        let psm = CodecFileHandle::open(
            directory,
            IndexFile::PostingsMeta,
            segment_name,
            segment_id,
            segment_suffix,
        )?;
        let mut meta_in = psm.body();

        // Read impact statistics
        let max_num_impacts_at_level0 = meta_in.read_le_int()?;
        let max_impact_num_bytes_at_level0 = meta_in.read_le_int()?;
        let max_num_impacts_at_level1 = meta_in.read_le_int()?;
        let max_impact_num_bytes_at_level1 = meta_in.read_le_int()?;

        // Read expected file lengths
        let _expected_doc_file_length = meta_in.read_le_long()?;
        if field_infos.has_prox() {
            let _expected_pos_file_length = meta_in.read_le_long()?;
            if field_infos.has_payloads() || field_infos.has_offsets() {
                let _expected_pay_file_length = meta_in.read_le_long()?;
            }
        }

        let doc = CodecFileHandle::open(
            directory,
            IndexFile::PostingsData,
            segment_name,
            segment_id,
            segment_suffix,
        )?;

        // Validate .pos header if positions exist. The handle is dropped at
        // the end of this block — positions aren't read yet.
        if field_infos.has_prox() {
            let _pos = CodecFileHandle::open(
                directory,
                IndexFile::PostingsPositions,
                segment_name,
                segment_id,
                segment_suffix,
            )?;
        }

        let impact_stats = ImpactStats {
            max_num_impacts_at_level0,
            max_impact_num_bytes_at_level0,
            max_num_impacts_at_level1,
            max_impact_num_bytes_at_level1,
        };

        debug!(
            "postings_reader: opened for segment {segment_name}, \
             impacts=[{max_num_impacts_at_level0}, {max_impact_num_bytes_at_level0}, \
             {max_num_impacts_at_level1}, {max_impact_num_bytes_at_level1}]"
        );

        Ok(Self {
            doc: doc.into_backing(),
            impact_stats,
        })
    }

    /// Returns the impact statistics read from segment metadata.
    pub fn impact_stats(&self) -> &ImpactStats {
        &self.impact_stats
    }

    /// Creates a [`BlockPostingsEnum`] for the given term state, matching Java's `postings()`.
    ///
    /// The `flags` parameter is currently unused (reserved for FREQS/POSITIONS/etc).
    pub fn postings(
        &self,
        term_state: &postings_format::IntBlockTermState,
        index_features: IndexFeatures,
        needs_freq: bool,
    ) -> io::Result<BlockPostingsEnum<'_>> {
        BlockPostingsEnum::new(
            IndexInput::unnamed(self.doc.as_bytes()),
            term_state,
            index_features,
            needs_freq,
            false,
            &self.impact_stats,
        )
    }

    /// Creates a [`BlockPostingsEnum`] with impacts support, matching Java's `impacts()`.
    pub fn impacts(
        &self,
        term_state: &postings_format::IntBlockTermState,
        index_features: IndexFeatures,
        needs_freq: bool,
    ) -> io::Result<BlockPostingsEnum<'_>> {
        BlockPostingsEnum::new(
            IndexInput::unnamed(self.doc.as_bytes()),
            term_state,
            index_features,
            needs_freq,
            true,
            &self.impact_stats,
        )
    }
}

// ---------------------------------------------------------------------------
// DeltaEncoding
// ---------------------------------------------------------------------------

/// How doc ID deltas are encoded in a full block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DeltaEncoding {
    /// Deltas stored as packed integers (Frame Of Reference).
    Packed,
    /// Deltas stored using unary coding (offset + bit set).
    Unary,
}

// ---------------------------------------------------------------------------
// MutableImpactList
// ---------------------------------------------------------------------------

/// A reusable list of impacts with a logical length.
struct MutableImpactList {
    impacts: Vec<Impact>,
    length: usize,
}

impl MutableImpactList {
    fn new(capacity: i32) -> Self {
        let cap = capacity as usize;
        let impacts = vec![
            Impact {
                freq: i32::MAX,
                norm: 1
            };
            cap
        ];
        Self { impacts, length: 0 }
    }

    fn as_slice(&self) -> &[Impact] {
        &self.impacts[..self.length]
    }
}

// ---------------------------------------------------------------------------
// BlockPostingsEnum
// ---------------------------------------------------------------------------

/// Iterator over doc IDs (and optionally frequencies/impacts) for a single term.
///
/// Handles three encoding paths:
/// - **Singleton** (docFreq=1): doc ID pulsed in term metadata, no I/O
/// - **VInt tail** (remaining docs < 128): group-varint encoded deltas
/// - **Full blocks** (128 docs): FOR-delta or UNARY encoded with skip headers
///
/// Supports skip-level navigation via level 0 and level 1 skip data for efficient
/// `advance()` and `advance_shallow()` operations.
pub struct BlockPostingsEnum<'a> {
    // Current state
    encoding: DeltaEncoding,
    doc: i32,

    // PACKED encoding buffers
    doc_buffer: [i32; BLOCK_SIZE + 1],

    // UNARY encoding buffers
    // bit set for UNARY blocks: BLOCK_SIZE * 32 bits = 4096 bits = 64 u64 words max
    doc_bit_set: [u64; 64],
    doc_bit_set_base: i32,
    // Cumulative word pop counts — aliases doc_buffer conceptually, but stored separately
    // to avoid borrow issues. In Java, docCumulativeWordPopCounts = docBuffer.
    // We use doc_buffer directly for this purpose (same trick as Java).

    // Level 0 skip data
    level0_last_doc_id: i32,
    level0_doc_end_fp: i64,

    // Level 1 skip data
    level1_last_doc_id: i32,
    level1_doc_end_fp: i64,
    level1_doc_count_upto: i32,

    // Term metadata
    doc_freq: i32,
    total_term_freq: i64,
    singleton_doc_id: i32,

    // Block iteration state
    doc_count_left: i32,
    prev_doc_id: i32,
    doc_buffer_size: usize,
    doc_buffer_upto: usize,

    // Doc input
    doc_in: IndexInput<'a>,

    // Freq buffer
    freq_buffer: [i32; BLOCK_SIZE],
    freq_fp: i64,

    // Index flags
    index_has_freq: bool,
    index_has_pos: bool,
    index_has_offsets_or_payloads: bool,
    needs_freq: bool,
    needs_impacts: bool,
    needs_docs_and_freqs_only: bool,

    // Serialized impacts for level 0
    level0_serialized_impacts: Option<Vec<u8>>,
    level0_serialized_impacts_length: usize,
    level0_impacts: Option<MutableImpactList>,

    // Serialized impacts for level 1
    level1_serialized_impacts: Option<Vec<u8>>,
    level1_serialized_impacts_length: usize,
    level1_impacts: Option<MutableImpactList>,

    // true if we shallow-advanced to a new block that we have not decoded yet
    needs_refilling: bool,
}

impl<'a> BlockPostingsEnum<'a> {
    /// Creates a new `BlockPostingsEnum` and resets it for the given term state.
    fn new(
        doc_in: IndexInput<'a>,
        term_state: &postings_format::IntBlockTermState,
        index_features: IndexFeatures,
        needs_freq: bool,
        needs_impacts: bool,
        impact_stats: &ImpactStats,
    ) -> io::Result<Self> {
        let needs_docs_and_freqs_only = !needs_impacts;

        let mut freq_buffer = [0i32; BLOCK_SIZE];
        if !needs_freq {
            freq_buffer.fill(1);
        }

        let (level0_serialized_impacts, level0_impacts) = if needs_freq && needs_impacts {
            (
                Some(vec![
                    0u8;
                    impact_stats.max_impact_num_bytes_at_level0 as usize
                ]),
                Some(MutableImpactList::new(
                    impact_stats.max_num_impacts_at_level0,
                )),
            )
        } else {
            (None, None)
        };

        let (level1_serialized_impacts, level1_impacts) = if needs_freq && needs_impacts {
            (
                Some(vec![
                    0u8;
                    impact_stats.max_impact_num_bytes_at_level1 as usize
                ]),
                Some(MutableImpactList::new(
                    impact_stats.max_num_impacts_at_level1,
                )),
            )
        } else {
            (None, None)
        };

        let mut this = Self {
            encoding: DeltaEncoding::Packed,
            doc: -1,
            doc_buffer: [0i32; BLOCK_SIZE + 1],
            doc_bit_set: [0u64; 64],
            doc_bit_set_base: 0,
            level0_last_doc_id: -1,
            level0_doc_end_fp: 0,
            level1_last_doc_id: -1,
            level1_doc_end_fp: 0,
            level1_doc_count_upto: 0,
            doc_freq: 0,
            total_term_freq: 0,
            singleton_doc_id: -1,
            doc_count_left: 0,
            prev_doc_id: -1,
            doc_buffer_size: BLOCK_SIZE,
            doc_buffer_upto: BLOCK_SIZE,
            doc_in,
            freq_buffer,
            freq_fp: -1,
            index_has_freq: index_features.has_freq,
            index_has_pos: index_features.has_pos,
            index_has_offsets_or_payloads: index_features.has_offsets_or_payloads,
            needs_freq,
            needs_impacts,
            needs_docs_and_freqs_only,
            level0_serialized_impacts,
            level0_serialized_impacts_length: 0,
            level0_impacts,
            level1_serialized_impacts,
            level1_serialized_impacts_length: 0,
            level1_impacts,
            needs_refilling: false,
        };

        this.reset(term_state)?;
        Ok(this)
    }

    /// Resets this enum to iterate over a new term, matching Java's `reset()`.
    fn reset(&mut self, term_state: &postings_format::IntBlockTermState) -> io::Result<()> {
        self.doc_freq = term_state.doc_freq;
        self.singleton_doc_id = term_state.singleton_doc_id;
        self.total_term_freq = if self.index_has_freq {
            term_state.total_term_freq
        } else {
            term_state.doc_freq as i64
        };

        self.doc = -1;
        self.prev_doc_id = -1;
        self.doc_count_left = self.doc_freq;
        self.freq_fp = -1;
        self.level0_last_doc_id = -1;
        if self.doc_freq < LEVEL1_NUM_DOCS as i32 {
            self.level1_last_doc_id = NO_MORE_DOCS;
            if self.doc_freq > 1 {
                self.doc_in.seek(term_state.doc_start_fp as usize)?;
            }
        } else {
            self.level1_last_doc_id = -1;
            self.level1_doc_end_fp = term_state.doc_start_fp;
        }
        self.level1_doc_count_upto = 0;
        self.doc_buffer_size = BLOCK_SIZE;
        self.doc_buffer_upto = BLOCK_SIZE;
        self.needs_refilling = false;

        Ok(())
    }

    /// Returns the current document's frequency.
    pub fn freq(&mut self) -> io::Result<i32> {
        if self.freq_fp != -1 {
            self.doc_in.seek(self.freq_fp as usize)?;
            let mut longs = [0i64; BLOCK_SIZE];
            pfor::pfor_decode(self.doc_in.cursor_mut(), &mut longs)?;
            for (i, &val) in longs.iter().enumerate() {
                self.freq_buffer[i] = val as i32;
            }
            self.freq_fp = -1;
        }
        Ok(self.freq_buffer[self.doc_buffer_upto - 1])
    }

    fn refill_full_block(&mut self) -> io::Result<()> {
        let input = &mut self.doc_in;
        let bits_per_value = input.read_byte()? as i8;
        if bits_per_value > 0 {
            // Block is encoded as 128 packed integers (FOR delta)
            let mut arr = [0i32; BLOCK_SIZE];
            pfor::for_delta_decode(
                bits_per_value as u32,
                input.cursor_mut(),
                self.prev_doc_id,
                &mut arr,
            )?;
            self.doc_buffer[..BLOCK_SIZE].copy_from_slice(&arr);
            self.encoding = DeltaEncoding::Packed;
        } else {
            // Block is encoded as a bit set
            debug_assert!(self.level0_last_doc_id != NO_MORE_DOCS);
            self.doc_bit_set_base = self.prev_doc_id + 1;
            let num_longs;
            if bits_per_value == 0 {
                // 0 means all 128 docs are consecutive
                num_longs = BLOCK_SIZE / 64; // 2
                // Set all BLOCK_SIZE bits
                self.doc_bit_set[0] = u64::MAX;
                self.doc_bit_set[1] = u64::MAX;
                for w in &mut self.doc_bit_set[2..] {
                    *w = 0;
                }
            } else {
                num_longs = (-bits_per_value) as usize;
                for i in 0..num_longs {
                    self.doc_bit_set[i] = input.read_le_long()? as u64;
                }
                for w in &mut self.doc_bit_set[num_longs..] {
                    *w = 0;
                }
            }
            if self.needs_freq {
                // Compute cumulative word pop counts using doc_buffer
                // Note: we know that BLOCK_SIZE bits are set, so the last index = BLOCK_SIZE
                for i in 0..num_longs - 1 {
                    self.doc_buffer[i] = self.doc_bit_set[i].count_ones() as i32;
                }
                for i in 1..num_longs - 1 {
                    self.doc_buffer[i] += self.doc_buffer[i - 1];
                }
                self.doc_buffer[num_longs - 1] = BLOCK_SIZE as i32;
                debug_assert_eq!(
                    self.doc_buffer[num_longs - 2]
                        + self.doc_bit_set[num_longs - 1].count_ones() as i32,
                    BLOCK_SIZE as i32
                );
            }
            self.encoding = DeltaEncoding::Unary;
        }
        if self.index_has_freq {
            if self.needs_freq {
                self.freq_fp = self.doc_in.position() as i64;
            }
            skip_pfor(&mut self.doc_in)?;
        }
        self.doc_count_left -= BLOCK_SIZE as i32;
        self.prev_doc_id = self.doc_buffer[BLOCK_SIZE - 1];
        self.doc_buffer_upto = 0;
        Ok(())
    }

    fn refill_remainder(&mut self) -> io::Result<()> {
        debug_assert!(self.doc_count_left >= 0 && self.doc_count_left < BLOCK_SIZE as i32);
        if self.doc_freq == 1 {
            self.doc_buffer[0] = self.singleton_doc_id;
            self.freq_buffer[0] = self.total_term_freq as i32;
            self.doc_buffer[1] = NO_MORE_DOCS;
            debug_assert!(self.freq_fp == -1);
            self.doc_count_left = 0;
            self.doc_buffer_size = 1;
        } else {
            // Read vInts
            let num = self.doc_count_left as usize;
            read_vint_block(
                &mut self.doc_in,
                &mut self.doc_buffer,
                &mut self.freq_buffer,
                num,
                self.index_has_freq,
                self.needs_freq,
            )?;
            prefix_sum(&mut self.doc_buffer, num, self.prev_doc_id as i64);
            self.doc_buffer[num] = NO_MORE_DOCS;
            self.freq_fp = -1;
            self.doc_buffer_size = num;
            self.doc_count_left = 0;
        }
        self.prev_doc_id = self.doc_buffer[BLOCK_SIZE - 1];
        self.doc_buffer_upto = 0;
        self.encoding = DeltaEncoding::Packed;
        debug_assert!(self.doc_buffer[self.doc_buffer_size] == NO_MORE_DOCS);
        Ok(())
    }

    fn refill_docs(&mut self) -> io::Result<()> {
        debug_assert!(self.doc_count_left >= 0);

        if self.doc_count_left >= BLOCK_SIZE as i32 {
            self.refill_full_block()?;
        } else {
            self.refill_remainder()?;
        }
        Ok(())
    }

    fn skip_level1_to(&mut self, target: i32) -> io::Result<()> {
        loop {
            self.prev_doc_id = self.level1_last_doc_id;
            self.level0_last_doc_id = self.level1_last_doc_id;
            self.doc_in.seek(self.level1_doc_end_fp as usize)?;
            self.doc_count_left = self.doc_freq - self.level1_doc_count_upto;
            self.level1_doc_count_upto += LEVEL1_NUM_DOCS as i32;

            if self.doc_count_left < LEVEL1_NUM_DOCS as i32 {
                self.level1_last_doc_id = NO_MORE_DOCS;
                break;
            }

            self.level1_last_doc_id += self.doc_in.read_vint()?;
            let delta = self.doc_in.read_vlong()?;
            self.level1_doc_end_fp = delta + self.doc_in.position() as i64;

            if self.index_has_freq {
                let skip1_end_fp =
                    self.doc_in.read_le_short()? as i64 + self.doc_in.position() as i64;
                let num_impact_bytes = self.doc_in.read_le_short()? as usize;
                if self.needs_impacts && self.level1_last_doc_id >= target {
                    if let Some(ref mut buf) = self.level1_serialized_impacts {
                        self.doc_in.read_bytes(&mut buf[..num_impact_bytes])?;
                        self.level1_serialized_impacts_length = num_impact_bytes;
                    }
                } else {
                    self.doc_in.skip_bytes(num_impact_bytes)?;
                }
                // Skip pos/pay data (positions not supported yet)
                // Seek to skip1EndFP to skip any remaining pos/pay skip data
                self.doc_in.seek(skip1_end_fp as usize)?;
            }

            if self.level1_last_doc_id >= target {
                break;
            }
        }
        Ok(())
    }

    fn do_move_to_next_level0_block(&mut self) -> io::Result<()> {
        debug_assert!(self.doc == self.level0_last_doc_id);
        // Skip pos/pay data — positions not supported yet

        if self.doc_count_left >= BLOCK_SIZE as i32 {
            self.doc_in.read_vlong()?; // level0NumBytes
            let doc_delta = read_vint15(&mut self.doc_in)?;
            self.level0_last_doc_id += doc_delta;
            let block_length = read_vlong15(&mut self.doc_in)?;
            self.level0_doc_end_fp = self.doc_in.position() as i64 + block_length;
            if self.index_has_freq {
                let num_impact_bytes = self.doc_in.read_vint()? as usize;
                if self.needs_impacts {
                    if let Some(ref mut buf) = self.level0_serialized_impacts {
                        self.doc_in.read_bytes(&mut buf[..num_impact_bytes])?;
                        self.level0_serialized_impacts_length = num_impact_bytes;
                    }
                } else {
                    self.doc_in.skip_bytes(num_impact_bytes)?;
                }

                if self.index_has_pos {
                    self.doc_in.read_vlong()?; // level0PosEndFP delta
                    self.doc_in.read_byte()?; // level0BlockPosUpto
                    if self.index_has_offsets_or_payloads {
                        self.doc_in.read_vlong()?; // level0PayEndFP delta
                        self.doc_in.read_vint()?; // level0BlockPayUpto
                    }
                }
            }
            self.refill_full_block()?;
        } else {
            self.level0_last_doc_id = NO_MORE_DOCS;
            self.refill_remainder()?;
        }
        Ok(())
    }

    fn move_to_next_level0_block(&mut self) -> io::Result<()> {
        if self.doc == self.level1_last_doc_id {
            // advance level 1 skip data
            self.skip_level1_to(self.doc + 1)?;
        }

        // Now advance level 0 skip data
        self.prev_doc_id = self.level0_last_doc_id;

        if self.needs_docs_and_freqs_only && self.doc_count_left >= BLOCK_SIZE as i32 {
            // Optimize the common path for exhaustive evaluation
            let level0_num_bytes = self.doc_in.read_vlong()?;
            let level0_end = self.doc_in.position() + level0_num_bytes as usize;
            let doc_delta = read_vint15(&mut self.doc_in)?;
            self.level0_last_doc_id += doc_delta;
            self.doc_in.seek(level0_end)?;
            self.refill_full_block()?;
        } else {
            self.do_move_to_next_level0_block()?;
        }
        Ok(())
    }

    fn skip_level0_to(&mut self, target: i32) -> io::Result<()> {
        loop {
            self.prev_doc_id = self.level0_last_doc_id;

            if self.doc_count_left >= BLOCK_SIZE as i32 {
                let num_skip_bytes = self.doc_in.read_vlong()?;
                let skip0_end = self.doc_in.position() + num_skip_bytes as usize;
                let doc_delta = read_vint15(&mut self.doc_in)?;
                self.level0_last_doc_id += doc_delta;
                let found = target <= self.level0_last_doc_id;
                let block_length = read_vlong15(&mut self.doc_in)?;
                self.level0_doc_end_fp = self.doc_in.position() as i64 + block_length;

                if self.index_has_freq {
                    if !found {
                        self.doc_in.seek(skip0_end)?;
                    } else {
                        let num_impact_bytes = self.doc_in.read_vint()? as usize;
                        if self.needs_impacts && found {
                            if let Some(ref mut buf) = self.level0_serialized_impacts {
                                self.doc_in.read_bytes(&mut buf[..num_impact_bytes])?;
                                self.level0_serialized_impacts_length = num_impact_bytes;
                            }
                        } else {
                            self.doc_in.skip_bytes(num_impact_bytes)?;
                        }
                        // Skip pos data — positions not supported yet
                        self.doc_in.seek(skip0_end)?;
                    }
                }

                if found {
                    break;
                }

                self.doc_in.seek(self.level0_doc_end_fp as usize)?;
                self.doc_count_left -= BLOCK_SIZE as i32;
            } else {
                self.level0_last_doc_id = NO_MORE_DOCS;
                break;
            }
        }
        // Skip pos data — positions not supported yet
        Ok(())
    }

    fn do_advance_shallow(&mut self, target: i32) -> io::Result<()> {
        if target > self.level1_last_doc_id {
            // advance skip data on level 1
            self.skip_level1_to(target)?;
        } else if self.needs_refilling {
            self.doc_in.seek(self.level0_doc_end_fp as usize)?;
            self.doc_count_left -= BLOCK_SIZE as i32;
        }

        self.skip_level0_to(target)?;
        Ok(())
    }
}

impl DocIdSetIterator for BlockPostingsEnum<'_> {
    fn doc_id(&self) -> i32 {
        self.doc
    }

    fn next_doc(&mut self) -> io::Result<i32> {
        if self.doc == self.level0_last_doc_id || self.needs_refilling {
            if self.needs_refilling {
                self.refill_docs()?;
                self.needs_refilling = false;
            } else {
                self.move_to_next_level0_block()?;
            }
        }

        match self.encoding {
            DeltaEncoding::Packed => {
                self.doc = self.doc_buffer[self.doc_buffer_upto];
            }
            DeltaEncoding::Unary => {
                let next = next_set_bit(
                    &self.doc_bit_set,
                    (self.doc - self.doc_bit_set_base + 1) as usize,
                );
                debug_assert!(next != NO_MORE_DOCS as usize);
                self.doc = self.doc_bit_set_base + next as i32;
            }
        }

        self.doc_buffer_upto += 1;
        Ok(self.doc)
    }

    fn advance(&mut self, target: i32) -> io::Result<i32> {
        if target > self.level0_last_doc_id || self.needs_refilling {
            if target > self.level0_last_doc_id {
                self.do_advance_shallow(target)?;
            }
            self.refill_docs()?;
            self.needs_refilling = false;
        }

        match self.encoding {
            DeltaEncoding::Packed => {
                let next = find_next_geq(
                    &self.doc_buffer,
                    target,
                    self.doc_buffer_upto,
                    self.doc_buffer_size,
                );
                self.doc = self.doc_buffer[next];
                self.doc_buffer_upto = next + 1;
            }
            DeltaEncoding::Unary => {
                let next =
                    next_set_bit(&self.doc_bit_set, (target - self.doc_bit_set_base) as usize);
                debug_assert!(next != NO_MORE_DOCS as usize);
                self.doc = self.doc_bit_set_base + next as i32;
                if self.needs_freq {
                    let word_index = next >> 6;
                    // Take the cumulative pop count for the given word, and subtract bits on
                    // the left of the current doc.
                    self.doc_buffer_upto = 1 + self.doc_buffer[word_index] as usize
                        - (self.doc_bit_set[word_index] >> (next & 0x3F)).count_ones() as usize;
                } else {
                    // When only docs needed and block is UNARY encoded, we do not need to
                    // maintain doc_buffer_upto to record the iteration position in the block.
                    self.doc_buffer_upto = 1;
                }
            }
        }

        Ok(self.doc)
    }

    fn cost(&self) -> i64 {
        self.doc_freq as i64
    }
}

impl ImpactsSource for BlockPostingsEnum<'_> {
    fn advance_shallow(&mut self, target: i32) -> io::Result<()> {
        if target > self.level0_last_doc_id {
            // advance level 0 skip data
            self.do_advance_shallow(target)?;
            self.needs_refilling = true;
        }
        Ok(())
    }

    fn get_impacts(&mut self) -> io::Result<&dyn Impacts> {
        // Decode impacts lazily — use a decode helper to avoid borrow conflicts
        if self.index_has_freq {
            decode_impacts_if_needed(
                &self.level0_serialized_impacts,
                self.level0_serialized_impacts_length,
                &mut self.level0_impacts,
                self.level0_last_doc_id != NO_MORE_DOCS,
            );
            decode_impacts_if_needed(
                &self.level1_serialized_impacts,
                self.level1_serialized_impacts_length,
                &mut self.level1_impacts,
                true,
            );
        }
        Ok(self as &dyn Impacts)
    }
}

// Implement `Impacts` directly on `BlockPostingsEnum` so `get_impacts` can return `&self`.
impl Impacts for BlockPostingsEnum<'_> {
    fn num_levels(&self) -> usize {
        if !self.index_has_freq || self.level1_last_doc_id == NO_MORE_DOCS {
            1
        } else {
            2
        }
    }

    fn get_doc_id_up_to(&self, level: usize) -> i32 {
        if !self.index_has_freq {
            return NO_MORE_DOCS;
        }
        if level == 0 {
            return self.level0_last_doc_id;
        }
        if level == 1 {
            return self.level1_last_doc_id;
        }
        NO_MORE_DOCS
    }

    fn get_impacts(&self, level: usize) -> &[Impact] {
        if self.index_has_freq {
            if level == 0
                && self.level0_last_doc_id != NO_MORE_DOCS
                && let Some(ref list) = self.level0_impacts
            {
                return list.as_slice();
            }
            if level == 1
                && let Some(ref list) = self.level1_impacts
            {
                return list.as_slice();
            }
            return &DUMMY_IMPACTS;
        }
        &IMPACTS_NO_FREQ
    }
}

// ---------------------------------------------------------------------------
// Module-level helper functions
// ---------------------------------------------------------------------------

/// Reads a VInt15 value: a short followed optionally by a VInt for the high bits.
fn read_vint15(input: &mut IndexInput<'_>) -> io::Result<i32> {
    let s = input.read_le_short()?;
    if s >= 0 {
        Ok(s as i32)
    } else {
        Ok((s as i32 & 0x7FFF) | (input.read_vint()? << 15))
    }
}

/// Reads a VLong15 value: a short followed optionally by a VLong for the high bits.
fn read_vlong15(input: &mut IndexInput<'_>) -> io::Result<i64> {
    let s = input.read_le_short()?;
    if s >= 0 {
        Ok(s as i64)
    } else {
        Ok((s as i64 & 0x7FFF) | (input.read_vlong()? << 15))
    }
}

/// Conditionally decodes serialized impacts into a reusable impact list.
fn decode_impacts_if_needed(
    serialized: &Option<Vec<u8>>,
    serialized_length: usize,
    impacts: &mut Option<MutableImpactList>,
    guard: bool,
) {
    if guard
        && serialized_length > 0
        && let (Some(buf), Some(list)) = (serialized, impacts)
    {
        read_impacts(&buf[..serialized_length], list);
    }
}

/// Decodes serialized impacts into a reusable impact list.
fn read_impacts(serialized: &[u8], reuse: &mut MutableImpactList) {
    let mut freq = 0i32;
    let mut norm = 0i64;
    let mut length = 0;
    let mut pos = 0;
    let data = serialized;

    while pos < data.len() {
        // Read vint from byte slice
        let (freq_delta, new_pos) = read_vint_from_bytes(data, pos);
        pos = new_pos;
        if (freq_delta & 0x01) != 0 {
            freq += 1 + (freq_delta >> 1);
            // Read zlong
            let (zlong_val, new_pos2) = read_zlong_from_bytes(data, pos);
            pos = new_pos2;
            norm += 1 + zlong_val;
        } else {
            freq += 1 + (freq_delta >> 1);
            norm += 1;
        }
        reuse.impacts[length] = Impact { freq, norm };
        length += 1;
    }
    reuse.length = length;
}

/// Reads a vint from a byte slice, returns (value, new_position).
fn read_vint_from_bytes(data: &[u8], mut pos: usize) -> (i32, usize) {
    let mut result = 0i32;
    let mut shift = 0;
    loop {
        let b = data[pos] as i32;
        pos += 1;
        result |= (b & 0x7F) << shift;
        if b & 0x80 == 0 {
            break;
        }
        shift += 7;
    }
    (result, pos)
}

/// Reads a zigzag-encoded vlong from a byte slice, returns (value, new_position).
fn read_zlong_from_bytes(data: &[u8], pos: usize) -> (i64, usize) {
    let (vlong, new_pos) = read_vlong_from_bytes(data, pos);
    // zigzag decode
    let decoded = (vlong as u64 >> 1) as i64 ^ -(vlong & 1);
    (decoded, new_pos)
}

/// Reads a vlong from a byte slice, returns (value, new_position).
fn read_vlong_from_bytes(data: &[u8], mut pos: usize) -> (i64, usize) {
    let mut result = 0i64;
    let mut shift = 0;
    loop {
        let b = data[pos] as i64;
        pos += 1;
        result |= (b & 0x7F) << shift;
        if b & 0x80 == 0 {
            break;
        }
        shift += 7;
    }
    (result, pos)
}

/// Prefix-sums a buffer: `buffer[0] += base; buffer[i] += buffer[i-1]`.
fn prefix_sum(buffer: &mut [i32], count: usize, base: i64) {
    buffer[0] += base as i32;
    for i in 1..count {
        buffer[i] += buffer[i - 1];
    }
}

/// Finds the first index in `buffer[from..to]` where `buffer[index] >= target`.
fn find_next_geq(buffer: &[i32], target: i32, from: usize, to: usize) -> usize {
    for (i, &val) in buffer.iter().enumerate().take(to).skip(from) {
        if val >= target {
            return i;
        }
    }
    to
}

/// Finds the next set bit at or after `from_index` in a u64 bitset.
/// Returns the bit index, or `NO_MORE_DOCS as usize` if none found.
fn next_set_bit(bits: &[u64; 64], from_index: usize) -> usize {
    let word_index = from_index >> 6;
    if word_index >= 64 {
        return NO_MORE_DOCS as usize;
    }
    let word = bits[word_index] >> (from_index & 63);
    if word != 0 {
        return from_index + word.trailing_zeros() as usize;
    }
    for (i, &w) in bits.iter().enumerate().take(64).skip(word_index + 1) {
        if w != 0 {
            return (i << 6) + w.trailing_zeros() as usize;
        }
    }
    NO_MORE_DOCS as usize
}

/// Skips a PFOR-encoded block of 128 values without decoding.
fn skip_pfor(input: &mut IndexInput<'_>) -> io::Result<()> {
    let token = input.read_byte()? as u32;
    let bpv = token & 0x1F;
    let num_exceptions = token >> 5;
    if bpv == 0 {
        input.read_vlong()?; // constant value
        input.skip_bytes((num_exceptions * 2) as usize)?;
    } else {
        let for_bytes = (bpv << 4) as usize; // bpv * 16
        input.skip_bytes(for_bytes + (num_exceptions * 2) as usize)?;
    }
    Ok(())
}

/// Reads a VInt block: group-varint doc deltas with optional freq separation.
///
/// Matches Java's `PostingsUtil.readVIntBlock`.
fn read_vint_block(
    input: &mut IndexInput<'_>,
    doc_buffer: &mut [i32],
    freq_buffer: &mut [i32],
    num: usize,
    index_has_freq: bool,
    decode_freq: bool,
) -> io::Result<()> {
    group_vint::read_group_vints(input.cursor_mut(), doc_buffer, num)?;
    if index_has_freq && decode_freq {
        for i in 0..num {
            freq_buffer[i] = doc_buffer[i] & 0x01;
            doc_buffer[i] = ((doc_buffer[i] as u32) >> 1) as i32;
            if freq_buffer[i] == 0 {
                freq_buffer[i] = input.read_vint()?;
            }
        }
    } else if index_has_freq {
        for val in &mut doc_buffer[..num] {
            *val = ((*val as u32) >> 1) as i32;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::sync::Arc;

    use crate::codecs::competitive_impact::BufferedNormsLookup;
    use crate::codecs::lucene94::field_infos_format;
    use crate::codecs::lucene99::segment_info_format;
    use crate::codecs::lucene103::postings_writer::PostingsWriter;
    use crate::document::{DocValuesType, DocumentBuilder, IndexOptions};
    use crate::index::config::IndexWriterConfig;
    use crate::index::field::text;
    use crate::index::pipeline::terms_hash::{BufferedPostingsEnum, DecodedDoc, DecodedPostings};
    use crate::index::segment_infos;
    use crate::index::writer::IndexWriter;
    use crate::index::{FieldInfo, FieldInfos, PointDimensionConfig};
    use crate::store::{MemoryDirectory, SharedDirectory};
    use assertables::*;

    const NO_FEATURES: IndexFeatures = IndexFeatures {
        has_freq: false,
        has_pos: false,
        has_offsets_or_payloads: false,
    };

    const FREQ_FEATURES: IndexFeatures = IndexFeatures {
        has_freq: true,
        has_pos: false,
        has_offsets_or_payloads: false,
    };

    /// Write postings for a single DOCS-only term and return the term state + directory.
    fn write_single_term(
        doc_ids: &[i32],
    ) -> io::Result<(postings_format::IntBlockTermState, SharedDirectory)> {
        write_single_term_with_options(doc_ids, IndexOptions::Docs)
    }

    fn write_single_term_with_options(
        doc_ids: &[i32],
        options: IndexOptions,
    ) -> io::Result<(postings_format::IntBlockTermState, SharedDirectory)> {
        let segment_name = "_0";
        let segment_suffix = "";
        let segment_id = [0u8; 16];
        let shared_dir = MemoryDirectory::create();

        let mut decoded = DecodedPostings::new();
        for &id in doc_ids {
            decoded.docs.push(DecodedDoc {
                doc_id: id,
                freq: 1,
                pos_start: 0,
            });
        }
        let norms = BufferedNormsLookup::no_norms();

        let term_state = {
            let mut writer = PostingsWriter::new(
                &shared_dir,
                segment_name,
                segment_suffix,
                &segment_id,
                options,
            )?;
            let mut pe = BufferedPostingsEnum::new(decoded, options.has_freqs());
            let state = writer.write_term(&mut pe, options, &norms, &mut HashSet::new())?;
            writer.finish()?;
            state
        };

        Ok((term_state, shared_dir))
    }

    /// Open PostingsReader from a directory written by `write_single_term`.
    fn open_reader(dir: &dyn Directory) -> io::Result<PostingsReader> {
        let field_infos = FieldInfos::new(vec![make_field_info("f", 0)]);
        PostingsReader::open(dir, "_0", "", &[0u8; 16], &field_infos)
    }

    fn make_field_info(name: &str, number: u32) -> FieldInfo {
        FieldInfo::new(
            name.to_string(),
            number,
            false,
            false,
            IndexOptions::Docs,
            DocValuesType::None,
            PointDimensionConfig::default(),
        )
    }

    /// Collect all doc IDs using `DocIdSetIterator::next_doc`.
    fn collect_docs(iter: &mut BlockPostingsEnum) -> io::Result<Vec<i32>> {
        let mut docs = Vec::new();
        loop {
            let doc = iter.next_doc()?;
            if doc == NO_MORE_DOCS {
                break;
            }
            docs.push(doc);
        }
        Ok(docs)
    }

    #[test]
    fn test_open_postings_reader() {
        let config = IndexWriterConfig::default().num_threads(1);
        let directory: SharedDirectory = MemoryDirectory::create();
        let writer = IndexWriter::new(config, Arc::clone(&directory));

        writer
            .add_document(
                DocumentBuilder::new()
                    .add_field(text("content").value("hello world"))
                    .build(),
            )
            .unwrap();

        writer.commit().unwrap();
        let dir = &*directory;

        let files = dir.list_all().unwrap();
        let segments_file = files.iter().find(|f| f.starts_with("segments_")).unwrap();
        let infos = segment_infos::read(dir, segments_file).unwrap();
        let seg = &infos.segments[0];

        let si = segment_info_format::read(dir, &seg.name, &seg.id).unwrap();
        let field_infos = field_infos_format::read(dir, &si, "").unwrap();

        let suffix = field_infos
            .iter()
            .find_map(|fi| {
                let format = fi.get_attribute("PerFieldPostingsFormat.format")?;
                let sfx = fi.get_attribute("PerFieldPostingsFormat.suffix")?;
                Some(format!("{format}_{sfx}"))
            })
            .unwrap();

        let reader = PostingsReader::open(dir, &seg.name, &suffix, &seg.id, &field_infos).unwrap();

        let stats = reader.impact_stats();
        assert_ge!(stats.max_num_impacts_at_level0, 0);
        assert_ge!(stats.max_impact_num_bytes_at_level0, 0);
        assert_ge!(stats.max_num_impacts_at_level1, 0);
        assert_ge!(stats.max_impact_num_bytes_at_level1, 0);
    }

    #[test]
    fn test_singleton() {
        let (state, dir) = write_single_term(&[42]).unwrap();
        assert_eq!(state.doc_freq, 1);
        assert_eq!(state.singleton_doc_id, 42);

        let reader = open_reader(&dir).unwrap();
        let mut iter = reader.postings(&state, NO_FEATURES, false).unwrap();
        let docs = collect_docs(&mut iter).unwrap();
        assert_eq!(docs, vec![42]);
    }

    #[test]
    fn test_vint_tail() {
        let doc_ids: Vec<i32> = (0..10).collect();
        let (state, dir) = write_single_term(&doc_ids).unwrap();
        assert_eq!(state.doc_freq, 10);
        assert_eq!(state.singleton_doc_id, -1);

        let reader = open_reader(&dir).unwrap();
        let mut iter = reader.postings(&state, NO_FEATURES, false).unwrap();
        let docs = collect_docs(&mut iter).unwrap();
        assert_eq!(docs, doc_ids);
    }

    #[test]
    fn test_vint_tail_sparse() {
        let doc_ids = vec![0, 5, 10, 50, 100, 500, 1000];
        let (state, dir) = write_single_term(&doc_ids).unwrap();
        assert_eq!(state.doc_freq, 7);

        let reader = open_reader(&dir).unwrap();
        let mut iter = reader.postings(&state, NO_FEATURES, false).unwrap();
        let docs = collect_docs(&mut iter).unwrap();
        assert_eq!(docs, doc_ids);
    }

    #[test]
    fn test_full_block() {
        // 128 docs — exactly one full block, no tail
        let doc_ids: Vec<i32> = (0..128).collect();
        let (state, dir) = write_single_term(&doc_ids).unwrap();
        assert_eq!(state.doc_freq, 128);

        let reader = open_reader(&dir).unwrap();
        let mut iter = reader.postings(&state, NO_FEATURES, false).unwrap();
        let docs = collect_docs(&mut iter).unwrap();
        assert_eq!(docs, doc_ids);
    }

    #[test]
    fn test_block_plus_tail() {
        // 200 docs — one full block (128) + VInt tail (72)
        let doc_ids: Vec<i32> = (0..200).collect();
        let (state, dir) = write_single_term(&doc_ids).unwrap();
        assert_eq!(state.doc_freq, 200);

        let reader = open_reader(&dir).unwrap();
        let mut iter = reader.postings(&state, NO_FEATURES, false).unwrap();
        let docs = collect_docs(&mut iter).unwrap();
        assert_eq!(docs, doc_ids);
    }

    #[test]
    fn test_multiple_blocks() {
        // 300 docs — two full blocks (256) + VInt tail (44)
        let doc_ids: Vec<i32> = (0..300).collect();
        let (state, dir) = write_single_term(&doc_ids).unwrap();
        assert_eq!(state.doc_freq, 300);

        let reader = open_reader(&dir).unwrap();
        let mut iter = reader.postings(&state, NO_FEATURES, false).unwrap();
        let docs = collect_docs(&mut iter).unwrap();
        assert_eq!(docs, doc_ids);
    }

    #[test]
    fn test_exhausted_returns_no_more_docs() {
        let (state, dir) = write_single_term(&[7]).unwrap();
        let reader = open_reader(&dir).unwrap();
        let mut iter = reader.postings(&state, NO_FEATURES, false).unwrap();

        assert_eq!(iter.next_doc().unwrap(), 7);
        assert_eq!(iter.next_doc().unwrap(), NO_MORE_DOCS);
        assert_eq!(iter.doc_id(), NO_MORE_DOCS);
    }

    #[test]
    fn test_advance_singleton() {
        let (state, dir) = write_single_term(&[42]).unwrap();
        let reader = open_reader(&dir).unwrap();
        let mut iter = reader.postings(&state, NO_FEATURES, false).unwrap();

        assert_eq!(iter.advance(42).unwrap(), 42);
        assert_eq!(iter.next_doc().unwrap(), NO_MORE_DOCS);
    }

    #[test]
    fn test_advance_past_end() {
        let (state, dir) = write_single_term(&[42]).unwrap();
        let reader = open_reader(&dir).unwrap();
        let mut iter = reader.postings(&state, NO_FEATURES, false).unwrap();

        assert_eq!(iter.advance(100).unwrap(), NO_MORE_DOCS);
    }

    #[test]
    fn test_advance_vint_tail() {
        let doc_ids: Vec<i32> = (0..10).collect();
        let (state, dir) = write_single_term(&doc_ids).unwrap();
        let reader = open_reader(&dir).unwrap();
        let mut iter = reader.postings(&state, NO_FEATURES, false).unwrap();

        assert_eq!(iter.advance(5).unwrap(), 5);
        assert_eq!(iter.next_doc().unwrap(), 6);
        assert_eq!(iter.advance(9).unwrap(), 9);
        assert_eq!(iter.next_doc().unwrap(), NO_MORE_DOCS);
    }

    #[test]
    fn test_advance_full_block() {
        let doc_ids: Vec<i32> = (0..128).collect();
        let (state, dir) = write_single_term(&doc_ids).unwrap();
        let reader = open_reader(&dir).unwrap();
        let mut iter = reader.postings(&state, NO_FEATURES, false).unwrap();

        assert_eq!(iter.advance(64).unwrap(), 64);
        assert_eq!(iter.advance(127).unwrap(), 127);
        assert_eq!(iter.next_doc().unwrap(), NO_MORE_DOCS);
    }

    #[test]
    fn test_advance_across_blocks() {
        let doc_ids: Vec<i32> = (0..300).collect();
        let (state, dir) = write_single_term(&doc_ids).unwrap();
        let reader = open_reader(&dir).unwrap();
        let mut iter = reader.postings(&state, NO_FEATURES, false).unwrap();

        // Advance into second block
        assert_eq!(iter.advance(200).unwrap(), 200);
        assert_eq!(iter.next_doc().unwrap(), 201);
        assert_eq!(iter.advance(299).unwrap(), 299);
        assert_eq!(iter.next_doc().unwrap(), NO_MORE_DOCS);
    }

    #[test]
    fn test_advance_sparse() {
        let doc_ids = vec![0, 100, 200, 300, 400, 500, 600, 700, 800, 900];
        let (state, dir) = write_single_term(&doc_ids).unwrap();
        let reader = open_reader(&dir).unwrap();
        let mut iter = reader.postings(&state, NO_FEATURES, false).unwrap();

        assert_eq!(iter.advance(250).unwrap(), 300);
        assert_eq!(iter.advance(600).unwrap(), 600);
        assert_eq!(iter.advance(901).unwrap(), NO_MORE_DOCS);
    }

    #[test]
    fn test_cost() {
        let doc_ids: Vec<i32> = (0..50).collect();
        let (state, dir) = write_single_term(&doc_ids).unwrap();
        let reader = open_reader(&dir).unwrap();
        let iter = reader.postings(&state, NO_FEATURES, false).unwrap();
        assert_eq!(iter.cost(), 50);
    }

    #[test]
    fn test_next_set_bit_basic() {
        let mut bits = [0u64; 64];
        bits[0] = 0b1010; // bits 1 and 3 set
        assert_eq!(next_set_bit(&bits, 0), 1);
        assert_eq!(next_set_bit(&bits, 1), 1);
        assert_eq!(next_set_bit(&bits, 2), 3);
        assert_eq!(next_set_bit(&bits, 4), NO_MORE_DOCS as usize);
    }

    #[test]
    fn test_next_set_bit_second_word() {
        let mut bits = [0u64; 64];
        bits[1] = 1; // bit 64 set
        assert_eq!(next_set_bit(&bits, 0), 64);
        assert_eq!(next_set_bit(&bits, 64), 64);
        assert_eq!(next_set_bit(&bits, 65), NO_MORE_DOCS as usize);
    }

    #[test]
    fn test_find_next_geq() {
        let buffer = [1, 3, 5, 7, 9, 11, 13, 15];
        assert_eq!(find_next_geq(&buffer, 5, 0, 8), 2);
        assert_eq!(find_next_geq(&buffer, 6, 0, 8), 3);
        assert_eq!(find_next_geq(&buffer, 1, 0, 8), 0);
        assert_eq!(find_next_geq(&buffer, 16, 0, 8), 8);
    }

    #[test]
    fn test_read_vint15() {
        // Values < 0x8000 are stored as a single LE short
        let data = [0x42u8, 0x00]; // 0x0042 = 66
        let mut input = IndexInput::unnamed(&data);
        let val = read_vint15(&mut input).unwrap();
        assert_eq!(val, 66);
    }

    #[test]
    fn test_prefix_sum() {
        let mut buf = [1, 2, 3, 4, 0, 0, 0, 0];
        prefix_sum(&mut buf, 4, 10);
        assert_eq!(&buf[..4], &[11, 13, 16, 20]);
    }

    #[test]
    fn test_advance_many_blocks() {
        // 2000 docs to exercise multiple level 0 blocks (under LEVEL1_NUM_DOCS limit)
        let doc_ids: Vec<i32> = (0..2000).collect();
        let (state, dir) = write_single_term(&doc_ids).unwrap();
        let reader = open_reader(&dir).unwrap();
        let mut iter = reader.postings(&state, NO_FEATURES, false).unwrap();

        assert_eq!(iter.advance(1500).unwrap(), 1500);
        assert_eq!(iter.next_doc().unwrap(), 1501);
        assert_eq!(iter.advance(1999).unwrap(), 1999);
        assert_eq!(iter.next_doc().unwrap(), NO_MORE_DOCS);
    }

    #[test]
    fn test_impacts_postings_reader() {
        // Verify impacts() returns a BlockPostingsEnum that works
        let doc_ids: Vec<i32> = (0..300).collect();
        let (state, dir) = write_single_term(&doc_ids).unwrap();
        let reader = open_reader(&dir).unwrap();
        let mut iter = reader.impacts(&state, NO_FEATURES, false).unwrap();
        let docs = collect_docs(&mut iter).unwrap();
        assert_eq!(docs, doc_ids);
    }

    #[test]
    fn test_next_doc_sequential_all() {
        // Verify sequential next_doc covers every doc
        let doc_ids: Vec<i32> = (0..500).collect();
        let (state, dir) = write_single_term(&doc_ids).unwrap();
        let reader = open_reader(&dir).unwrap();
        let mut iter = reader.postings(&state, NO_FEATURES, false).unwrap();
        let docs = collect_docs(&mut iter).unwrap();
        assert_eq!(docs, doc_ids);
    }

    #[test]
    fn test_advance_unary_with_freq() {
        // Consecutive doc IDs trigger UNARY block encoding. Advancing with needs_freq=true
        // into a position where the bit index >= 64 exercises the shift in the UNARY advance
        // path. Java's `long >>> next` implicitly masks to `& 63`; Rust must do this
        // explicitly to avoid overflow.
        let doc_ids: Vec<i32> = (0..200).collect();
        let (state, dir) =
            write_single_term_with_options(&doc_ids, IndexOptions::DocsAndFreqs).unwrap();
        let reader = open_reader(&dir).unwrap();
        let mut iter = reader.postings(&state, FREQ_FEATURES, true).unwrap();

        // Advance to doc 100, which is bit index 100 in the UNARY block (word_index=1,
        // bit offset 36). This is fine.
        assert_eq!(iter.advance(100).unwrap(), 100);

        // Advance to doc 128 (bit index 128 = word 2, bit 0). The shift `>> 128` would
        // overflow without masking to `& 63`.
        assert_eq!(iter.advance(128).unwrap(), 128);

        // Verify continued iteration works
        assert_eq!(iter.next_doc().unwrap(), 129);
    }

    #[test]
    fn test_freq_returns_one_for_docs_only() {
        let doc_ids: Vec<i32> = (0..5).collect();
        let (state, dir) =
            write_single_term_with_options(&doc_ids, IndexOptions::DocsAndFreqs).unwrap();
        let reader = open_reader(&dir).unwrap();
        let mut iter = reader.postings(&state, FREQ_FEATURES, false).unwrap();

        assert_eq!(iter.next_doc().unwrap(), 0);
        // Each doc has freq=1 since write_single_term uses freq=1
        assert_eq!(iter.freq().unwrap(), 1);
        assert_eq!(iter.next_doc().unwrap(), 1);
        assert_eq!(iter.freq().unwrap(), 1);
    }

    #[test]
    fn test_freq_full_block() {
        // 128+ docs triggers full block encoding — freq() must PFor-decode
        let doc_ids: Vec<i32> = (0..200).collect();
        let (state, dir) =
            write_single_term_with_options(&doc_ids, IndexOptions::DocsAndFreqs).unwrap();
        let reader = open_reader(&dir).unwrap();
        let mut iter = reader.postings(&state, FREQ_FEATURES, false).unwrap();

        // Read into the packed block
        for expected in 0..128 {
            assert_eq!(iter.next_doc().unwrap(), expected);
            assert_eq!(iter.freq().unwrap(), 1);
        }
        // Continue into vint tail
        assert_eq!(iter.next_doc().unwrap(), 128);
        assert_eq!(iter.freq().unwrap(), 1);
    }

    #[test]
    fn test_int_block_term_state_default() {
        let state = postings_format::IntBlockTermState::default();
        assert_eq!(state.doc_freq, 0);
        assert_eq!(state.singleton_doc_id, -1);
        assert_eq!(state.last_pos_block_offset, -1);
    }
}
