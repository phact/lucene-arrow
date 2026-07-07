// SPDX-License-Identifier: Apache-2.0

//! SORTED/SORTED_SET terms dictionaries (SPEC §7.3): bulk materialization.
//!
//! On disk (Lucene90, traced from `Lucene90DocValuesConsumer.addTermsDict`
//! / Bearing's port): terms are byte-sorted, in blocks of 64. Each block
//! stores its first term raw (`vint len + bytes`); the remaining ≤63 terms
//! are prefix-coded against their predecessor into a suffix stream that is
//! LZ4-compressed with the block's first term as dictionary. Block start
//! offsets are a DirectMonotonic sequence. The reverse index (binary
//! search support) is skipped — analytics reads materialize the whole
//! dictionary once per segment, in order.

use crate::monotonic;
use lucene_arrow_core::cursor::Cursor;
use lucene_arrow_core::{Error, Result};

pub const TERMS_DICT_BLOCK_SHIFT: u32 = 6;
pub const TERMS_DICT_BLOCK_SIZE: u64 = 1 << TERMS_DICT_BLOCK_SHIFT;

/// Everything the `.dvm` entry records about one terms dictionary.
#[derive(Debug, Clone)]
pub struct TermsDictPlan {
    pub num_terms: u64,
    /// DirectMonotonic meta for the block-address sequence
    /// (`ceil(num_terms / 64)` values, relative to `terms_offset`).
    pub address_meta: Vec<monotonic::MetaBlock>,
    pub address_block_shift: u32,
    pub max_term_length: u32,
    pub max_block_length: u32,
    /// Term-block bytes in `.dvd`.
    pub terms_offset: u64,
    pub terms_len: u64,
    /// Packed DirectMonotonic address payload in `.dvd`.
    pub addresses_offset: u64,
    pub addresses_len: u64,
}

impl TermsDictPlan {
    pub fn num_blocks(&self) -> u64 {
        self.num_terms.div_ceil(TERMS_DICT_BLOCK_SIZE)
    }
}

/// Materialized dictionary: concatenated term bytes + offsets (Arrow
/// binary/utf8 layout, ready to wrap).
#[derive(Debug)]
pub struct TermsDict {
    pub bytes: Vec<u8>,
    /// `num_terms + 1` offsets into `bytes`.
    pub offsets: Vec<i32>,
}

impl TermsDict {
    pub fn term(&self, ord: usize) -> &[u8] {
        &self.bytes[self.offsets[ord] as usize..self.offsets[ord + 1] as usize]
    }
    pub fn len(&self) -> usize {
        self.offsets.len() - 1
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Decode every term of the dictionary from the full `.dvd` bytes.
pub fn materialize(plan: &TermsDictPlan, dvd: &[u8]) -> Result<TermsDict> {
    let num_blocks = plan.num_blocks();
    let terms_end = plan.terms_offset + plan.terms_len;
    if terms_end as usize > dvd.len() || (plan.addresses_offset + plan.addresses_len) as usize > dvd.len() {
        return Err(Error::corrupt("terms dict region beyond data file"));
    }
    let addr_region = &dvd[plan.addresses_offset as usize..];
    let starts = monotonic::decode(&plan.address_meta, addr_region, num_blocks, plan.address_block_shift)?;

    let mut dict = TermsDict {
        bytes: Vec::with_capacity((plan.num_terms as usize) * 8),
        offsets: Vec::with_capacity(plan.num_terms as usize + 1),
    };
    dict.offsets.push(0);

    let mut suffix_buf: Vec<u8> = Vec::with_capacity(plan.max_block_length as usize);
    let mut remaining = plan.num_terms;

    for (bi, &rel_start) in starts.iter().enumerate() {
        if rel_start < 0 {
            return Err(Error::corrupt("negative terms block offset"));
        }
        let block_start = plan.terms_offset + rel_start as u64;
        let block_end = if bi + 1 < starts.len() {
            plan.terms_offset + starts[bi + 1] as u64
        } else {
            terms_end
        };
        if block_end < block_start || block_end > terms_end {
            return Err(Error::corrupt("terms block bounds out of order"));
        }
        let mut c = Cursor::at(dvd, block_start as usize);

        // First term, stored raw.
        let first_len = c.vint()?;
        if first_len < 0 || first_len as u32 > plan.max_term_length {
            return Err(Error::corrupt(format!("bad first-term length {first_len}")));
        }
        let first = c.take(first_len as usize)?;
        let mut prev_start = dict.bytes.len();
        dict.bytes.extend_from_slice(first);
        dict.offsets.push(dict.bytes.len() as i32);
        let terms_in_block = remaining.min(TERMS_DICT_BLOCK_SIZE);
        remaining -= terms_in_block;

        if terms_in_block > 1 {
            // LZ4 block with the first term as dictionary.
            let uncompressed_len = c.vint()?;
            if uncompressed_len < 0 {
                return Err(Error::corrupt("negative suffix stream length"));
            }
            let compressed = &dvd[c.pos()..block_end as usize];
            suffix_buf = lz4_flex::block::decompress_with_dict(compressed, uncompressed_len as usize, first)
                .map_err(|e| Error::corrupt(format!("terms LZ4: {e}")))?;
            if suffix_buf.len() != uncompressed_len as usize {
                return Err(Error::corrupt("terms LZ4 length mismatch"));
            }

            let mut s = Cursor::new(&suffix_buf);
            for _ in 1..terms_in_block {
                let token = s.u8()?;
                let mut prefix = (token & 0xF) as usize;
                let mut suffix = ((token >> 4) as usize) + 1;
                if prefix == 15 {
                    prefix += s.vint()? as usize;
                }
                if suffix == 16 {
                    suffix += s.vint()? as usize;
                }
                let prev_len = dict.bytes.len() - prev_start;
                if prefix > prev_len {
                    return Err(Error::corrupt("term prefix longer than previous term"));
                }
                let new_start = dict.bytes.len();
                // previous[..prefix] + suffix bytes
                dict.bytes.extend_from_within(prev_start..prev_start + prefix);
                let sfx = s.take(suffix)?;
                dict.bytes.extend_from_slice(sfx);
                dict.offsets.push(dict.bytes.len() as i32);
                prev_start = new_start;
            }
        }
    }
    let _ = suffix_buf;

    if dict.len() as u64 != plan.num_terms {
        return Err(Error::corrupt(format!(
            "materialized {} terms, expected {}",
            dict.len(),
            plan.num_terms
        )));
    }
    Ok(dict)
}
