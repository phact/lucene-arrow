// SPDX-License-Identifier: Apache-2.0

//! Segment-level reader that wires all codec readers for a single segment.
//!
//! [`SegmentReader`] is the read-side counterpart to the index writer: it opens
//! all the files for one segment and provides access to the
//! codec readers that decode stored fields, norms, doc values, term vectors,
//! points, terms, and postings.
//!
//! # Example
//!
//! ```no_run
//! use bearing::index::segment_reader::SegmentReader;
//! use bearing::store::FSDirectory;
//! use std::path::Path;
//!
//! let dir = FSDirectory::open(Path::new("/path/to/index")).unwrap();
//! let reader = SegmentReader::open(&dir, "_0", &[0u8; 16]).unwrap();
//! println!("Segment has {} documents", reader.max_doc());
//! ```

use std::fmt;
use std::io;

use log::debug;

use crate::codecs::codec_headers;
use crate::codecs::lucene90::compound_reader::CompoundDirectory;
use crate::codecs::lucene90::doc_values_producer::DocValuesReader;
use crate::codecs::lucene90::norms_producer::NormsReader;
use crate::codecs::lucene90::points_reader::PointsReader;
use crate::codecs::lucene90::stored_fields_reader::StoredFieldsReader;
use crate::codecs::lucene90::term_vectors_reader::TermVectorsReader;
use crate::codecs::lucene94::field_infos_format;
use crate::codecs::lucene99::segment_info_format;
use crate::codecs::lucene103::blocktree_reader::{BlockTreeTermsReader, FieldReader};
use crate::codecs::lucene103::postings_reader::PostingsReader;
use crate::index::doc_values_iterators::NumericDocValues;
use crate::index::{FieldInfos, SegmentInfo};
use crate::store::Directory;

/// Reads all data for a single segment of a Lucene index.
///
/// Owns the codec readers for stored fields, norms, doc values, term vectors,
/// points, terms, and postings. Each reader is created only if the segment
/// contains the corresponding data (determined by [`FieldInfos`] flags).
///
/// Handles both compound (`.cfs`/`.cfe`) and non-compound segments
/// transparently — callers do not need to know the storage format.
pub struct SegmentReader {
    segment_name: String,
    field_infos: FieldInfos,
    max_doc: i32,
    stored_fields_reader: Option<StoredFieldsReader>,
    norms_reader: Option<NormsReader>,
    doc_values_reader: Option<DocValuesReader>,
    term_vectors_reader: Option<TermVectorsReader>,
    points_reader: Option<PointsReader>,
    terms_reader: Option<BlockTreeTermsReader>,
    postings_reader: Option<PostingsReader>,
}

impl fmt::Debug for SegmentReader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SegmentReader")
            .field("segment_name", &self.segment_name)
            .field("max_doc", &self.max_doc)
            .finish()
    }
}

impl SegmentReader {
    /// Opens all codec readers for a single segment.
    ///
    /// Reads segment info to determine the file format (compound vs non-compound),
    /// then opens each codec reader conditionally based on which data the segment
    /// contains. File handles are kept open for lazy data access.
    ///
    /// # Errors
    ///
    /// Returns an error if any segment file is missing, corrupt, or has a
    /// version mismatch.
    pub fn open(
        directory: &dyn Directory,
        segment_name: &str,
        segment_id: &[u8; codec_headers::ID_LENGTH],
    ) -> io::Result<Self> {
        let si = segment_info_format::read(directory, segment_name, segment_id)?;

        let reader = if si.is_compound_file {
            let compound_dir = CompoundDirectory::open(directory, segment_name, segment_id)?;
            Self::open_from_directory(&compound_dir, &si)?
        } else {
            Self::open_from_directory(directory, &si)?
        };

        debug!(
            "segment_reader: opened segment {segment_name}, max_doc={}, fields={}",
            reader.max_doc,
            reader.field_infos.len()
        );

        Ok(reader)
    }

    /// Opens all codec readers from a specific directory (compound or raw).
    fn open_from_directory(dir: &dyn Directory, si: &SegmentInfo) -> io::Result<Self> {
        let field_infos = field_infos_format::read(dir, si, "")?;
        let segment_name = &si.name;
        let segment_id = &si.id;
        let max_doc = si.max_doc;

        let stored_fields_reader =
            Some(StoredFieldsReader::open(dir, segment_name, "", segment_id)?);

        let norms_reader = if field_infos.has_norms() {
            Some(NormsReader::open(
                dir,
                segment_name,
                "",
                segment_id,
                &field_infos,
                max_doc,
            )?)
        } else {
            None
        };

        let doc_values_reader = if field_infos.has_doc_values() {
            let suffix =
                derive_suffix(&field_infos, "PerFieldDocValuesFormat").ok_or_else(|| {
                    io::Error::other("segment has doc values but no PerFieldDocValuesFormat suffix")
                })?;
            Some(DocValuesReader::open(
                dir,
                segment_name,
                &suffix,
                segment_id,
                &field_infos,
            )?)
        } else {
            None
        };

        let term_vectors_reader = if field_infos.has_vectors() {
            Some(TermVectorsReader::open(dir, segment_name, "", segment_id)?)
        } else {
            None
        };

        let points_reader = if field_infos.has_point_values() {
            Some(PointsReader::open(
                dir,
                segment_name,
                "",
                segment_id,
                &field_infos,
            )?)
        } else {
            None
        };

        let (terms_reader, postings_reader) = if field_infos.has_postings() {
            let suffix =
                derive_suffix(&field_infos, "PerFieldPostingsFormat").ok_or_else(|| {
                    io::Error::other("segment has postings but no PerFieldPostingsFormat suffix")
                })?;
            let terms =
                BlockTreeTermsReader::open(dir, segment_name, &suffix, segment_id, &field_infos)?;
            let postings =
                PostingsReader::open(dir, segment_name, &suffix, segment_id, &field_infos)?;
            (Some(terms), Some(postings))
        } else {
            (None, None)
        };

        Ok(Self {
            segment_name: segment_name.to_string(),
            field_infos,
            max_doc,
            stored_fields_reader,
            norms_reader,
            doc_values_reader,
            term_vectors_reader,
            points_reader,
            terms_reader,
            postings_reader,
        })
    }

    /// Returns the segment name (e.g., `"_0"`).
    pub fn segment_name(&self) -> &str {
        &self.segment_name
    }

    /// Returns the field metadata for this segment.
    pub fn field_infos(&self) -> &FieldInfos {
        &self.field_infos
    }

    /// Returns the total number of documents in this segment (including deleted).
    pub fn max_doc(&self) -> i32 {
        self.max_doc
    }

    /// Returns a mutable reference to the stored fields reader.
    ///
    /// Matches Java's `CodecReader.getFieldsReader()`.
    pub fn get_fields_reader(&mut self) -> Option<&mut StoredFieldsReader> {
        self.stored_fields_reader.as_mut()
    }

    /// Returns a reference to the norms reader, or `None` if no fields have norms.
    pub fn norms_reader(&self) -> Option<&NormsReader> {
        self.norms_reader.as_ref()
    }

    /// Returns a lazy [`NumericDocValues`] for the given field's norms, or `None`
    /// if no norms exist for this field.
    ///
    /// Matches Java's `LeafReader.getNormValues(String field)`.
    pub fn get_norm_values(
        &self,
        field: &str,
    ) -> io::Result<Option<Box<dyn NumericDocValues + '_>>> {
        let field_info = match self.field_infos.field_info_by_name(field) {
            Some(fi) => fi,
            None => return Ok(None),
        };
        match &self.norms_reader {
            Some(nr) => nr.get_norms(field_info),
            None => Ok(None),
        }
    }

    /// Returns the doc values reader, or `None` if no fields have doc values.
    pub fn doc_values_reader(&self) -> Option<&DocValuesReader> {
        self.doc_values_reader.as_ref()
    }

    /// Returns the term vectors reader, or `None` if no fields have term vectors.
    pub fn term_vectors_reader(&self) -> Option<&TermVectorsReader> {
        self.term_vectors_reader.as_ref()
    }

    /// Returns the points/BKD reader, or `None` if no fields have point values.
    pub fn points_reader(&self) -> Option<&PointsReader> {
        self.points_reader.as_ref()
    }

    /// Returns the postings reader, or `None` if no fields are indexed.
    pub fn postings_reader(&self) -> Option<&PostingsReader> {
        self.postings_reader.as_ref()
    }

    /// Returns a borrowed view of the field's terms dictionary, or `None`
    /// if the field does not exist or has no indexed terms.
    ///
    /// Matches Java's `LeafReader.terms(String)`. The returned
    /// [`FieldReader`] implements
    /// [`Terms`](crate::index::terms::Terms).
    pub fn terms(&self, field: &str) -> Option<FieldReader<'_>> {
        self.terms_reader.as_ref()?.terms(field, &self.field_infos)
    }
}

/// Derives a per-field codec suffix (e.g., `"Lucene103_0"`) from field attributes.
fn derive_suffix(field_infos: &FieldInfos, prefix: &str) -> Option<String> {
    let format_key = format!("{prefix}.format");
    let suffix_key = format!("{prefix}.suffix");
    field_infos.iter().find_map(|fi| {
        let format = fi.get_attribute(&format_key)?;
        let suffix = fi.get_attribute(&suffix_key)?;
        Some(format!("{format}_{suffix}"))
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::codecs::lucene103::postings_reader::{BlockPostingsEnum, IndexFeatures};
    use crate::document::{DocumentBuilder, IndexOptions};
    use crate::index::config::IndexWriterConfig;
    use crate::index::field::{string, text};
    use crate::index::segment_infos;
    use crate::index::terms::Terms;
    use crate::index::writer::IndexWriter;
    use crate::search::doc_id_set_iterator::{DocIdSetIterator, NO_MORE_DOCS};
    use crate::store::{MemoryDirectory, SharedDirectory};

    fn write_test_index(compound: bool) -> (SharedDirectory, String, [u8; 16]) {
        let config = IndexWriterConfig::default()
            .num_threads(1)
            .use_compound_file(compound);
        let directory: SharedDirectory = MemoryDirectory::create();
        let writer = IndexWriter::new(config, Arc::clone(&directory));

        let doc = DocumentBuilder::new()
            .add_field(text("content").value("hello world"))
            .add_field(string("path").stored().value("/test.txt"))
            .build();
        writer.add_document(doc).unwrap();

        let doc2 = DocumentBuilder::new()
            .add_field(text("content").value("goodbye world"))
            .add_field(string("path").stored().value("/other.txt"))
            .build();
        writer.add_document(doc2).unwrap();

        writer.commit().unwrap();

        // Find segment info
        let files = directory.list_all().unwrap();
        let segments_file = files.iter().find(|f| f.starts_with("segments_")).unwrap();
        let infos = segment_infos::read(&*directory, segments_file).unwrap();
        let seg = &infos.segments[0];
        let name = seg.name.clone();
        let id = seg.id;

        (directory, name, id)
    }

    #[test]
    fn test_open_non_compound() {
        let (dir, name, id) = write_test_index(false);
        let reader = SegmentReader::open(&*dir, &name, &id).unwrap();

        assert_eq!(reader.max_doc(), 2);
        assert_eq!(reader.segment_name(), &name);
        assert_not_empty!(reader.field_infos());
        assert!(reader.terms("content").is_some());
        assert!(reader.postings_reader().is_some());
    }

    #[test]
    fn test_open_compound() {
        let (dir, name, id) = write_test_index(true);
        let reader = SegmentReader::open(&*dir, &name, &id).unwrap();

        assert_eq!(reader.max_doc(), 2);
        assert!(reader.terms("content").is_some());
        assert!(reader.postings_reader().is_some());
    }

    #[test]
    fn test_stored_fields_access() {
        let (dir, name, id) = write_test_index(false);
        let mut reader = SegmentReader::open(&*dir, &name, &id).unwrap();

        let sfr = reader.get_fields_reader().unwrap();
        let fields = sfr.document(0).unwrap();
        assert!(!fields.is_empty());
    }

    #[test]
    fn test_norms_access() {
        let (dir, name, id) = write_test_index(false);
        let reader = SegmentReader::open(&*dir, &name, &id).unwrap();

        // "content" is a TextField with norms — access via get_norm_values
        let mut norms = reader.get_norm_values("content").unwrap().unwrap();
        assert!(norms.advance_exact(0).unwrap());
        assert_ne!(norms.long_value().unwrap(), 0);
    }

    #[test]
    fn test_field_metadata() {
        let (dir, name, id) = write_test_index(false);
        let reader = SegmentReader::open(&*dir, &name, &id).unwrap();

        let fi = reader.field_infos();
        assert_some!(fi.field_info_by_name("content"));
        assert_some!(fi.field_info_by_name("path"));
        assert!(fi.has_postings());
    }

    /// Test helper: seek a term and create a postings iterator, following the
    /// same path production code uses (Terms → TermsEnum → PostingsReader).
    fn seek_postings<'a>(
        reader: &'a SegmentReader,
        field: &str,
        term: &[u8],
    ) -> io::Result<Option<BlockPostingsEnum<'a>>> {
        let field_info = match reader.field_infos().field_info_by_name(field) {
            Some(fi) => fi,
            None => return Ok(None),
        };
        let terms = match reader.terms(field) {
            Some(t) => t,
            None => return Ok(None),
        };
        let postings_reader = match reader.postings_reader() {
            Some(pr) => pr,
            None => return Ok(None),
        };
        let mut terms_enum = terms.iterator()?;
        if !terms_enum.seek_exact(term)? {
            return Ok(None);
        }
        let state = terms_enum.term_state()?;
        let index_has_offsets =
            field_info.index_options() >= IndexOptions::DocsAndFreqsAndPositionsAndOffsets;
        let index_features = IndexFeatures {
            has_freq: field_info.index_options().has_freqs(),
            has_pos: field_info.index_options().has_positions(),
            has_offsets_or_payloads: index_has_offsets || field_info.has_payloads(),
        };
        let iter = postings_reader.postings(&state, index_features, false)?;
        Ok(Some(iter))
    }

    fn collect_docs(iter: &mut BlockPostingsEnum<'_>) -> Vec<i32> {
        let mut docs = Vec::new();
        loop {
            let doc = iter.next_doc().unwrap();
            if doc == NO_MORE_DOCS {
                break;
            }
            docs.push(doc);
        }
        docs
    }

    #[test]
    fn test_postings_term_found() {
        let (dir, name, id) = write_test_index(false);
        let reader = SegmentReader::open(&*dir, &name, &id).unwrap();

        // "world" appears in both docs (doc 0: "hello world", doc 1: "goodbye world")
        let mut iter = seek_postings(&reader, "content", b"world")
            .unwrap()
            .unwrap();
        let docs = collect_docs(&mut iter);
        assert_eq!(docs, vec![0, 1]);
    }

    #[test]
    fn test_postings_term_in_one_doc() {
        let (dir, name, id) = write_test_index(false);
        let reader = SegmentReader::open(&*dir, &name, &id).unwrap();

        // "hello" appears only in doc 0
        let mut iter = seek_postings(&reader, "content", b"hello")
            .unwrap()
            .unwrap();
        let docs = collect_docs(&mut iter);
        assert_eq!(docs, vec![0]);

        // "goodbye" appears only in doc 1
        let mut iter = seek_postings(&reader, "content", b"goodbye")
            .unwrap()
            .unwrap();
        let docs = collect_docs(&mut iter);
        assert_eq!(docs, vec![1]);
    }

    #[test]
    fn test_postings_nonexistent_term() {
        let (dir, name, id) = write_test_index(false);
        let reader = SegmentReader::open(&*dir, &name, &id).unwrap();

        let result = seek_postings(&reader, "content", b"nonexistent").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_postings_nonexistent_field() {
        let (dir, name, id) = write_test_index(false);
        let reader = SegmentReader::open(&*dir, &name, &id).unwrap();

        let result = seek_postings(&reader, "no_such_field", b"hello").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_postings_string_field() {
        let (dir, name, id) = write_test_index(false);
        let reader = SegmentReader::open(&*dir, &name, &id).unwrap();

        // StringField "path" stores exact values as single tokens
        let mut iter = seek_postings(&reader, "path", b"/test.txt")
            .unwrap()
            .unwrap();
        let docs = collect_docs(&mut iter);
        assert_eq!(docs, vec![0]);

        let mut iter = seek_postings(&reader, "path", b"/other.txt")
            .unwrap()
            .unwrap();
        let docs = collect_docs(&mut iter);
        assert_eq!(docs, vec![1]);
    }

    #[test]
    fn test_postings_compound_segment() {
        let (dir, name, id) = write_test_index(true);
        let reader = SegmentReader::open(&*dir, &name, &id).unwrap();

        let mut iter = seek_postings(&reader, "content", b"world")
            .unwrap()
            .unwrap();
        let docs = collect_docs(&mut iter);
        assert_eq!(docs, vec![0, 1]);
    }

    #[test]
    fn test_postings_many_docs() {
        // Write 200 docs to exercise full block + VInt tail
        let config = IndexWriterConfig::default().num_threads(1);
        let directory: SharedDirectory = MemoryDirectory::create();
        let writer = IndexWriter::new(config, Arc::clone(&directory));

        for i in 0..200 {
            let mut builder = DocumentBuilder::new().add_field(text("content").value("common"));
            if i % 2 == 0 {
                builder = builder.add_field(text("content").value("even"));
            }
            writer.add_document(builder.build()).unwrap();
        }

        writer.commit().unwrap();

        let dir = &*directory;
        let files = dir.list_all().unwrap();
        let segments_file = files.iter().find(|f| f.starts_with("segments_")).unwrap();
        let infos = segment_infos::read(dir, segments_file).unwrap();
        let seg = &infos.segments[0];

        let reader = SegmentReader::open(dir, &seg.name, &seg.id).unwrap();

        // "common" should be in all 200 docs
        let mut iter = seek_postings(&reader, "content", b"common")
            .unwrap()
            .unwrap();
        let docs = collect_docs(&mut iter);
        assert_len_eq_x!(&docs, 200);
        assert_eq!(docs[0], 0);
        assert_eq!(docs[199], 199);

        // "even" should be in 100 docs (0, 2, 4, ..., 198)
        let mut iter = seek_postings(&reader, "content", b"even").unwrap().unwrap();
        let docs = collect_docs(&mut iter);
        assert_len_eq_x!(&docs, 100);
        assert_eq!(docs[0], 0);
        assert_eq!(docs[1], 2);
        assert_eq!(docs[99], 198);
    }

    #[test]
    fn test_missing_stored_fields_file_is_error() {
        // Write a valid index, then copy to memory dir without .fdt
        let (directory, name, id) = write_test_index(false);
        let mem_dir = MemoryDirectory::create();

        // Copy all files except .fdt (stored fields data)
        let dir = &*directory;
        for filename in dir.list_all().unwrap() {
            if !filename.ends_with(".fdt") {
                let data = dir.read_file(&filename).unwrap();
                mem_dir.write_file(&filename, &data).unwrap();
            }
        }

        let result = SegmentReader::open(&mem_dir, &name, &id);
        assert!(
            result.is_err(),
            "expected error when stored fields file is missing"
        );
    }

    #[test]
    fn test_segment_without_norms_has_no_norms_reader() {
        // KeywordField has omit_norms=true, no TextField => no norms
        use crate::index::field::keyword;

        let config = IndexWriterConfig::default().num_threads(1);
        let directory: SharedDirectory = MemoryDirectory::create();
        let writer = IndexWriter::new(config, Arc::clone(&directory));

        let doc = DocumentBuilder::new()
            .add_field(keyword("id").value("abc"))
            .build();
        writer.add_document(doc).unwrap();

        writer.commit().unwrap();

        let dir = &*directory;
        let files = dir.list_all().unwrap();
        let segments_file = files.iter().find(|f| f.starts_with("segments_")).unwrap();
        let infos = segment_infos::read(dir, segments_file).unwrap();
        let seg = &infos.segments[0];

        let reader = SegmentReader::open(dir, &seg.name, &seg.id).unwrap();
        assert!(
            reader.norms_reader().is_none(),
            "segment without norms should have no norms reader"
        );
    }

    #[test]
    fn test_terms_by_name() {
        let (dir, name, id) = write_test_index(false);
        let reader = SegmentReader::open(&*dir, &name, &id).unwrap();

        let terms = reader.terms("content");
        assert!(terms.is_some());
        let terms = terms.unwrap();
        assert_gt!(terms.size(), 0);
        assert_gt!(terms.get_doc_count(), 0);

        assert!(reader.terms("nonexistent").is_none());
    }
}
