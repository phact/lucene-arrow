// SPDX-License-Identifier: Apache-2.0

//! In-order full walk of `.tim` term blocks — terms come out in sorted
//! order with their stats (docFreq/totalTermFreq) and postings metadata
//! (docStartFP / singleton doc id) decoded from the per-block streams.

use lucene_arrow_core::cursor::Cursor;
use lucene_arrow_core::{Error, Result};

/// Per-term postings pointers (subset of Lucene's IntBlockTermState we
/// need for a docs+freqs COO relation; positions fields are consumed but
/// not surfaced yet).
#[derive(Debug, Clone, Copy, Default)]
pub struct TermMeta {
    pub doc_start_fp: u64,
    /// Docid when `doc_freq == 1` (postings inlined into metadata); -1
    /// otherwise.
    pub singleton_doc_id: i64,
}

/// Field traits the walker needs from `.fnm`.
#[derive(Debug, Clone, Copy)]
pub struct FieldTraits {
    pub has_freqs: bool,
    pub has_positions: bool,
    pub has_offsets: bool,
}

struct Block {
    ent_count: usize,
    is_last_in_floor: bool,
    is_leaf: bool,
    suffixes: Vec<u8>,
    suffix_lengths: Vec<u8>,
    stats: Vec<u8>,
    meta: Vec<u8>,
    /// Absolute .tim offset where this block starts (sub-block deltas are
    /// relative to it).
    fp: u64,
    /// Absolute .tim offset just past this block (next floor block).
    fp_end: u64,
}

fn read_block(tim: &[u8], fp: u64) -> Result<Block> {
    let mut c = Cursor::at(tim, fp as usize);
    let code = c.vint()? as u32;
    let ent_count = (code >> 1) as usize;
    let is_last_in_floor = code & 1 != 0;

    let token = c.vlong()? as u64;
    let is_leaf = token & 0x04 != 0;
    let num_suffix_bytes = (token >> 3) as usize;
    let compression = (token & 0x03) as u32;
    let suffixes = match compression {
        0 => c.take(num_suffix_bytes)?.to_vec(),
        1 => {
            let mut cur = std::io::Cursor::new(&tim[c.pos()..]);
            let out =
                bearing::encoding::lowercase_ascii::decompress_from_cursor(&mut cur, num_suffix_bytes)
                    .map_err(|e| Error::Codec(e.to_string()))?;
            c.skip(cur.position() as usize)?;
            out
        }
        2 => {
            let mut cur = std::io::Cursor::new(&tim[c.pos()..]);
            let out = bearing::encoding::lz4::decompress_from_reader(&mut cur, num_suffix_bytes)
                .map_err(|e| Error::Codec(e.to_string()))?;
            c.skip(cur.position() as usize)?;
            out
        }
        other => return Err(Error::invalid(format!("suffix compression {other}"))),
    };

    let t = c.vint()? as u32;
    let all_equal = t & 1 != 0;
    let n_len_bytes = (t >> 1) as usize;
    let suffix_lengths = if all_equal {
        vec![c.u8()?; n_len_bytes]
    } else {
        c.take(n_len_bytes)?.to_vec()
    };

    let n_stats = c.vint()? as usize;
    let stats = c.take(n_stats)?.to_vec();
    let n_meta = c.vint()? as usize;
    let meta = c.take(n_meta)?.to_vec();

    Ok(Block {
        ent_count,
        is_last_in_floor,
        is_leaf,
        suffixes,
        suffix_lengths,
        stats,
        meta,
        fp,
        fp_end: c.pos() as u64,
    })
}

struct MetaChain {
    doc_start_fp: u64,
    pos_start_fp: u64,
    pay_start_fp: u64,
    singleton_doc_id: i64,
    absolute: bool,
}

/// Decode one term's stats + metadata. Streams are cursors over the
/// block's stat/meta sections; the delta chain resets per block.
#[allow(clippy::too_many_arguments)]
fn decode_term(
    stats: &mut Cursor,
    singleton_run: &mut u32,
    meta: &mut Cursor,
    chain: &mut MetaChain,
    traits: FieldTraits,
) -> Result<(u32, i64, TermMeta)> {
    let (doc_freq, total_term_freq) = if *singleton_run > 0 {
        *singleton_run -= 1;
        (1u32, 1i64)
    } else {
        let token = stats.vint()? as u32;
        if token & 1 == 1 {
            *singleton_run = token >> 1;
            (1, 1)
        } else {
            let df = token >> 1;
            let ttf =
                if traits.has_freqs { df as i64 + stats.vlong()? } else { df as i64 };
            (df, ttf)
        }
    };

    if chain.absolute {
        chain.doc_start_fp = 0;
        chain.pos_start_fp = 0;
        chain.pay_start_fp = 0;
        chain.singleton_doc_id = -1;
        chain.absolute = false;
    }
    let code = meta.vlong()? as u64;
    if code & 1 != 0 {
        // Singleton docid delta (zigzag); docStartFP unchanged.
        let enc = code >> 1;
        let delta = ((enc >> 1) as i64) ^ -((enc & 1) as i64);
        chain.singleton_doc_id += delta;
    } else {
        chain.doc_start_fp += code >> 1;
        chain.singleton_doc_id =
            if doc_freq == 1 { meta.vint()? as i64 } else { -1 };
    }
    if traits.has_positions {
        chain.pos_start_fp += meta.vlong()? as u64;
        if traits.has_offsets {
            chain.pay_start_fp += meta.vlong()? as u64;
        }
        if total_term_freq > super::POSTINGS_BLOCK_SIZE as i64 {
            let _last_pos_block_offset = meta.vlong()?;
        }
    }

    Ok((
        doc_freq,
        total_term_freq,
        TermMeta {
            doc_start_fp: chain.doc_start_fp,
            singleton_doc_id: if doc_freq == 1 { chain.singleton_doc_id } else { -1 },
        },
    ))
}

/// Walk one block chain (a block plus its floor continuations), recursing
/// into sub-blocks in entry order. `prefix` is the term prefix for this
/// chain. Calls `emit(term, doc_freq, total_term_freq, meta)` per term.
fn walk_chain(
    tim: &[u8],
    fp: u64,
    prefix: &mut Vec<u8>,
    traits: FieldTraits,
    emit: &mut impl FnMut(&[u8], u32, i64, TermMeta) -> Result<()>,
) -> Result<()> {
    let mut fp = fp;
    loop {
        let block = read_block(tim, fp)?;
        let mut suffix_pos = 0usize;
        let mut lengths = Cursor::at(&block.suffix_lengths, 0);
        let mut stats = Cursor::at(&block.stats, 0);
        let mut meta = Cursor::at(&block.meta, 0);
        let mut singleton_run = 0u32;
        let mut chain = MetaChain {
            doc_start_fp: 0,
            pos_start_fp: 0,
            pay_start_fp: 0,
            singleton_doc_id: -1,
            absolute: true,
        };
        let base_len = prefix.len();

        for _ in 0..block.ent_count {
            let (suffix_len, is_sub) = if block.is_leaf {
                (lengths.vint()? as u32 as usize, false)
            } else {
                let code = lengths.vint()? as u32;
                ((code >> 1) as usize, code & 1 != 0)
            };
            let suffix = &block.suffixes[suffix_pos..suffix_pos + suffix_len];
            suffix_pos += suffix_len;
            prefix.truncate(base_len);
            prefix.extend_from_slice(suffix);

            if is_sub {
                let delta = lengths.vlong()? as u64;
                let child_fp = block
                    .fp
                    .checked_sub(delta)
                    .ok_or_else(|| Error::invalid("sub-block fp underflow"))?;
                let mut child_prefix = prefix.clone();
                walk_chain(tim, child_fp, &mut child_prefix, traits, emit)?;
            } else {
                let (df, ttf, tm) =
                    decode_term(&mut stats, &mut singleton_run, &mut meta, &mut chain, traits)?;
                emit(prefix, df, ttf, tm)?;
            }
        }
        prefix.truncate(base_len);

        if block.is_last_in_floor {
            return Ok(());
        }
        fp = block.fp_end; // next floor block is physically adjacent
    }
}

/// Enumerate every term of a field in sorted order.
pub fn walk_terms(
    tim: &[u8],
    root_block_fp: u64,
    traits: FieldTraits,
    mut emit: impl FnMut(&[u8], u32, i64, TermMeta) -> Result<()>,
) -> Result<()> {
    let mut prefix = Vec::new();
    walk_chain(tim, root_block_fp, &mut prefix, traits, &mut emit)
}
