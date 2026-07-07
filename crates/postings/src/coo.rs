// SPDX-License-Identifier: Apache-2.0

//! Segment-scoped postings relation (SPEC §7.8): the whole terms dict of
//! one field as a CSR matrix — `term_offsets[t]..term_offsets[t+1]` spans
//! term `t`'s rows in `docs`/`freqs`; `terms` holds the sorted term bytes.
//! COO's `term_ord` column is implied by the spans (materialize on demand).

use crate::doc::scan_postings;
use crate::walk::{FieldTraits, walk_terms};
use lucene_arrow_core::Result;

#[derive(Debug, Default)]
pub struct CsrPostings {
    /// Flattened term bytes + offsets (len = num_terms + 1).
    pub term_bytes: Vec<u8>,
    pub term_bytes_offsets: Vec<u64>,
    /// Row spans per term ordinal (len = num_terms + 1).
    pub term_offsets: Vec<u64>,
    pub docs: Vec<u32>,
    pub freqs: Vec<u32>,
}

impl CsrPostings {
    pub fn num_terms(&self) -> usize {
        self.term_offsets.len().saturating_sub(1)
    }
    pub fn num_rows(&self) -> usize {
        self.docs.len()
    }
    pub fn term(&self, ord: usize) -> &[u8] {
        &self.term_bytes
            [self.term_bytes_offsets[ord] as usize..self.term_bytes_offsets[ord + 1] as usize]
    }
    /// Materialize the COO `term_ord` column.
    pub fn term_ords(&self) -> Vec<u32> {
        let mut out = Vec::with_capacity(self.num_rows());
        for t in 0..self.num_terms() {
            let n = (self.term_offsets[t + 1] - self.term_offsets[t]) as usize;
            out.extend(std::iter::repeat_n(t as u32, n));
        }
        out
    }
}

/// Read one field's full postings into CSR form.
pub fn read_csr(
    tim: &[u8],
    doc_file: &[u8],
    root_block_fp: u64,
    traits: FieldTraits,
) -> Result<CsrPostings> {
    let mut csr = CsrPostings::default();
    csr.term_bytes_offsets.push(0);
    csr.term_offsets.push(0);
    walk_terms(tim, root_block_fp, traits, |term, df, ttf, tm| {
        csr.term_bytes.extend_from_slice(term);
        csr.term_bytes_offsets.push(csr.term_bytes.len() as u64);
        scan_postings(doc_file, df, ttf, tm, traits, |doc, freq| {
            csr.docs.push(doc);
            csr.freqs.push(freq);
            Ok(())
        })?;
        csr.term_offsets.push(csr.docs.len() as u64);
        Ok(())
    })?;
    Ok(csr)
}
