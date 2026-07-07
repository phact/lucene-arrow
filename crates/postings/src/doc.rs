// SPDX-License-Identifier: Apache-2.0

//! `.doc` postings scan — full-enumeration decode of one term's doc ids
//! (+ freqs) from the Lucene103 postings file. Read sequence traced from
//! Bearing `postings_reader.rs` (reset / move_to_next_level0_block /
//! refill_full_block / refill_remainder).
//!
//! Level-1 skip entries (written every 4096 docs when `docFreq >= 4096`)
//! are consumed inline during the sequential scan.

use crate::walk::{FieldTraits, TermMeta};
use lucene_arrow_core::cursor::Cursor;
use lucene_arrow_core::{Error, Result};

pub const BLOCK_SIZE: usize = 128;
pub const LEVEL1_NUM_DOCS: u32 = 4096;

fn vint15(c: &mut Cursor) -> Result<i32> {
    let s = c.le_i16()?;
    if s >= 0 { Ok(s as i32) } else { Ok((s as i32 & 0x7FFF) | (c.vint()? << 15)) }
}

/// Emit `(doc, freq)` for every posting of one term, in doc order.
/// `freq` is 1 for fields without frequencies.
pub fn scan_postings(
    doc_file: &[u8],
    doc_freq: u32,
    total_term_freq: i64,
    tm: TermMeta,
    traits: FieldTraits,
    mut emit: impl FnMut(u32, u32) -> Result<()>,
) -> Result<()> {
    if doc_freq == 1 {
        let freq = if traits.has_freqs { total_term_freq as u32 } else { 1 };
        return emit(tm.singleton_doc_id as u32, freq);
    }
    let mut c = Cursor::at(doc_file, tm.doc_start_fp as usize);
    let mut remaining = doc_freq as usize;
    let mut prev: i32 = -1;
    let mut consumed = 0usize;

    while remaining >= BLOCK_SIZE {
        // Level-1 entry every 4096 docs, present only while ≥ 4096 docs
        // remain past the boundary. Sequential scan: consume its header
        // and skip the impact/positions metadata span.
        if consumed.is_multiple_of(LEVEL1_NUM_DOCS as usize) && remaining >= LEVEL1_NUM_DOCS as usize {
            let _doc_delta = c.vint()?;
            let _level1_len = c.vlong()?;
            if traits.has_freqs {
                let span = c.le_i16()? as u16 as usize;
                c.seek(c.pos() + span)?;
            }
        }

        // Level-0 skip entry (docs+freqs fast path: take the block's last
        // doc id, then skip the rest of the metadata).
        let num_bytes = c.vlong()? as usize;
        let end = c.pos() + num_bytes;
        let _level0_last = vint15(&mut c)?;
        c.seek(end)?;

        // Packed doc block.
        let bpv = c.u8()? as i8;
        let mut docs = [0i32; BLOCK_SIZE];
        if bpv > 0 {
            let (decoded, used) =
                crate::pfor::for_delta_decode(bpv as u32, &doc_file[c.pos()..], prev)
                    .ok_or_else(|| Error::invalid(format!("bad FOR block bpv={bpv}")))?;
            docs = decoded;
            c.skip(used)?;
        } else if bpv == 0 {
            // CONSECUTIVE: prev+1 ..= prev+128.
            for (i, d) in docs.iter_mut().enumerate() {
                *d = prev + 1 + i as i32;
            }
        } else {
            // BITSET over base prev+1.
            let num_longs = (-bpv) as usize;
            let base = prev + 1;
            let mut n = 0usize;
            for w in 0..num_longs {
                let mut word = u64::from_le_bytes(
                    c.take(8)?.try_into().map_err(|_| Error::invalid("bitset word"))?,
                );
                while word != 0 {
                    let bit = word.trailing_zeros();
                    docs[n] = base + (w as u32 * 64 + bit) as i32;
                    n += 1;
                    word &= word - 1;
                }
            }
            if n != BLOCK_SIZE {
                return Err(Error::invalid(format!("bitset block decoded {n} docs")));
            }
        }

        // Frequency block (PFor) follows when the field has freqs.
        let mut freqs = [0i64; BLOCK_SIZE];
        if traits.has_freqs {
            let mut cur = std::io::Cursor::new(&doc_file[c.pos()..]);
            bearing::encoding::pfor::pfor_decode(&mut cur, &mut freqs)
                .map_err(|e| Error::Codec(e.to_string()))?;
            c.skip(cur.position() as usize)?;
        } else {
            freqs.fill(1);
        }

        for i in 0..BLOCK_SIZE {
            emit(docs[i] as u32, freqs[i] as u32)?;
        }
        prev = docs[BLOCK_SIZE - 1];
        remaining -= BLOCK_SIZE;
        consumed += BLOCK_SIZE;
    }

    if remaining > 0 {
        // VInt tail: group-varint doc codes; low bit flags freq==1 when
        // the field has freqs.
        let mut codes = vec![0i32; remaining];
        let mut cur = std::io::Cursor::new(&doc_file[c.pos()..]);
        bearing::encoding::group_vint::read_group_vints(&mut cur, &mut codes, remaining)
            .map_err(|e| Error::Codec(e.to_string()))?;
        c.skip(cur.position() as usize)?;
        for code in codes {
            let (delta, freq) = if traits.has_freqs {
                let delta = (code as u32 >> 1) as i32;
                let freq = if code & 1 == 1 { 1 } else { c.vint()? as u32 };
                (delta, freq)
            } else {
                (code, 1)
            };
            prev += delta;
            emit(prev as u32, freq)?;
        }
    }
    Ok(())
}
