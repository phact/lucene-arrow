// SPDX-License-Identifier: Apache-2.0

//! Whole-file assembly for `.vemf`/`.vec` (headers, sentinel, footers) —
//! same role as `lucene_arrow_docvalues::file` (SPEC §10.5 note applies).

use crate::consts;
use crate::write::encode_flat_field;
use crate::{Similarity, VectorEncoding};
use lucene_arrow_core::Result;
use lucene_arrow_core::cursor::{SEGMENT_ID_LENGTH, write_footer, write_index_header};

pub struct VectorsFileBuilder {
    vemf: Vec<u8>,
    vec: Vec<u8>,
}

impl VectorsFileBuilder {
    pub fn new(segment_id: &[u8; SEGMENT_ID_LENGTH], suffix: &str) -> Self {
        let mut vemf = Vec::new();
        let mut vec = Vec::new();
        write_index_header(&mut vemf, consts::META_CODEC, consts::VERSION, segment_id, suffix);
        write_index_header(&mut vec, consts::DATA_CODEC, consts::VERSION, segment_id, suffix);
        VectorsFileBuilder { vemf, vec }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_field(
        &mut self,
        field_number: i32,
        encoding: VectorEncoding,
        similarity: Similarity,
        dim: u32,
        docs: &[u32],
        vectors: &[u8],
        max_doc: u32,
    ) -> Result<()> {
        let encoded = encode_flat_field(
            field_number,
            encoding,
            similarity,
            dim,
            docs,
            vectors,
            max_doc,
            self.vec.len() as u64,
        )?;
        self.vemf.extend_from_slice(&encoded.meta);
        self.vec.extend_from_slice(&encoded.data);
        Ok(())
    }

    /// Close both files. Returns `(vemf, vec)`.
    pub fn finish(mut self) -> (Vec<u8>, Vec<u8>) {
        self.vemf.extend_from_slice(&(-1i32).to_le_bytes());
        write_footer(&mut self.vemf);
        write_footer(&mut self.vec);
        (self.vemf, self.vec)
    }
}
