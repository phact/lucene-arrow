// SPDX-License-Identifier: Apache-2.0

//! Lucene90NormsFormat writer (`.nvm` meta + `.nvd` data) — required for
//! BM25: the norm is the SmallFloat-encoded field length per document
//! (SPEC §10; P9). Layout traced from Bearing `codecs/lucene90/norms.rs`
//! (pub(crate) there, hence this port). v1 writes the DENSE (`ALL`) and
//! constant patterns — every scored doc has the text field; sparse DISI
//! coverage can reuse `lucene_arrow_docvalues::disi` when needed.

use lucene_arrow_core::Result;
use lucene_arrow_core::cursor::{SEGMENT_ID_LENGTH, write_footer, write_index_header};

pub const DATA_CODEC: &str = "Lucene90NormsData";
pub const META_CODEC: &str = "Lucene90NormsMetadata";
pub const VERSION: i32 = 0;

fn num_bytes_per_value(min: i64, max: i64) -> u8 {
    if min >= max {
        0
    } else if (-128..=127).contains(&min) && max <= 127 {
        1
    } else if (-32768..=32767).contains(&min) && max <= 32767 {
        2
    } else if min >= i32::MIN as i64 && max <= i32::MAX as i64 {
        4
    } else {
        8
    }
}

/// Builds one `.nvm` + `.nvd` pair, dense fields only (norm per doc for
/// all `max_doc` docs, in doc order).
pub struct NormsFilesBuilder {
    nvm: Vec<u8>,
    nvd: Vec<u8>,
}

impl NormsFilesBuilder {
    pub fn new(segment_id: &[u8; SEGMENT_ID_LENGTH], suffix: &str) -> Self {
        let mut nvm = Vec::new();
        let mut nvd = Vec::new();
        write_index_header(&mut nvm, META_CODEC, VERSION, segment_id, suffix);
        write_index_header(&mut nvd, DATA_CODEC, VERSION, segment_id, suffix);
        NormsFilesBuilder { nvm, nvd }
    }

    /// Dense field: `norms[d]` is doc d's norm (SmallFloat-encoded length).
    pub fn add_dense_field(&mut self, field_number: i32, norms: &[i64]) -> Result<()> {
        let min = norms.iter().copied().min().unwrap_or(0);
        let max = norms.iter().copied().max().unwrap_or(0);
        let m = &mut self.nvm;
        m.extend_from_slice(&field_number.to_le_bytes());
        m.extend_from_slice(&(-1i64).to_le_bytes()); // docsWithFieldOffset: ALL
        m.extend_from_slice(&0i64.to_le_bytes()); // docsWithFieldLength
        m.extend_from_slice(&(-1i16).to_le_bytes()); // jumpTableEntryCount
        m.push(0xFF); // denseRankPower
        m.extend_from_slice(&(norms.len() as i32).to_le_bytes()); // numDocsWithField
        let width = num_bytes_per_value(min, max);
        m.push(width);
        if width == 0 {
            m.extend_from_slice(&min.to_le_bytes()); // constant in offset slot
        } else {
            m.extend_from_slice(&(self.nvd.len() as i64).to_le_bytes()); // normsOffset
            for &v in norms {
                match width {
                    1 => self.nvd.push(v as u8),
                    2 => self.nvd.extend_from_slice(&(v as i16).to_le_bytes()),
                    4 => self.nvd.extend_from_slice(&(v as i32).to_le_bytes()),
                    _ => self.nvd.extend_from_slice(&v.to_le_bytes()),
                }
            }
        }
        Ok(())
    }

    pub fn finish(mut self) -> (Vec<u8>, Vec<u8>) {
        self.nvm.extend_from_slice(&(-1i32).to_le_bytes()); // EOF marker
        write_footer(&mut self.nvm);
        write_footer(&mut self.nvd);
        (self.nvm, self.nvd)
    }
}
