// SPDX-License-Identifier: Apache-2.0

//! Write one field's inverted index as real Lucene103 postings files
//! (`.tim/.tip/.tmd/.doc/.psm`) by feeding our aggregated
//! [`InvertedField`] into Bearing's public `BlockTreeTermsWriter` —
//! byte-correct terms dict, trie, skip data and competitive impacts come
//! from Bearing (transitively Java-identical), the aggregation from us
//! (CPU reference now, GPU sort/aggregate behind the same seam).

use std::io;
use std::path::Path;

use bearing::codecs::fields_producer::{FieldTerms, PostingsEnumProducer};
use bearing::codecs::competitive_impact::BufferedNormsLookup;
use bearing::codecs::lucene103::blocktree_writer::BlockTreeTermsWriter;
use bearing::document::IndexOptions;
use bearing::store::fs::FSDirectory;

use crate::build::InvertedField;
use lucene_arrow_core::{Error, Result};

struct Field<'a> {
    inv: &'a InvertedField,
    name: &'a str,
    number: u32,
}

struct Producer<'a> {
    docs: &'a [u32],
    freqs: &'a [u32],
    at: i32, // index of current doc, -1 before first
}

impl PostingsEnumProducer for Producer<'_> {
    fn doc_freq(&self) -> i32 {
        self.docs.len() as i32
    }
    fn total_term_freq(&self) -> i64 {
        self.freqs.iter().map(|&f| f as i64).sum()
    }
    fn next_doc(&mut self) -> io::Result<i32> {
        self.at += 1;
        if (self.at as usize) < self.docs.len() {
            Ok(self.docs[self.at as usize] as i32)
        } else {
            Ok(i32::MAX) // NO_MORE_DOCS
        }
    }
    fn freq(&self) -> i32 {
        self.freqs[self.at as usize] as i32
    }
    fn next_position(&mut self) -> io::Result<i32> {
        Err(io::Error::other("positions not indexed (DOCS_AND_FREQS)"))
    }
    fn offset(&self) -> Option<bearing::document::TermOffset> {
        None
    }
    fn payload(&self) -> Option<&[u8]> {
        None
    }
}

impl FieldTerms for Field<'_> {
    fn term_count(&self) -> usize {
        self.inv.num_terms()
    }
    fn term_bytes(&self, index: usize) -> &[u8] {
        self.inv.term(index)
    }
    fn postings(&self, index: usize) -> io::Result<Box<dyn PostingsEnumProducer + '_>> {
        let (docs, freqs) = self.inv.postings(index);
        Ok(Box::new(Producer { docs, freqs, at: -1 }))
    }
    fn index_options(&self) -> IndexOptions {
        IndexOptions::DocsAndFreqs
    }
    fn has_payloads(&self) -> bool {
        false
    }
    fn field_number(&self) -> u32 {
        self.number
    }
    fn field_name(&self) -> &str {
        self.name
    }
}

/// Write `.tim/.tip/.tmd/.doc/.psm` (suffix `Lucene103_0`) for one text
/// field into `dir`. Returns the created file names.
pub fn write_postings_files(
    dir: &Path,
    segment_name: &str,
    segment_id: &[u8; 16],
    field_name: &str,
    field_number: u32,
    inv: &InvertedField,
) -> Result<Vec<String>> {
    let fsdir = FSDirectory::open(dir).map_err(|e| Error::Codec(e.to_string()))?;
    let mut writer = BlockTreeTermsWriter::new(
        &fsdir,
        segment_name,
        "Lucene103_0",
        segment_id,
        IndexOptions::DocsAndFreqs,
    )
    .map_err(|e| Error::Codec(e.to_string()))?;

    let docs_dense: Vec<i32> = (0..inv.norms.len() as i32).collect();
    let norms = BufferedNormsLookup::new(&inv.norms, &docs_dense);
    let field = Field { inv, name: field_name, number: field_number };
    writer.write_field(&field, &norms).map_err(|e| Error::Codec(e.to_string()))?;
    writer.finish().map_err(|e| Error::Codec(e.to_string()))
}
