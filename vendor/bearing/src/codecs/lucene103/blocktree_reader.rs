// SPDX-License-Identifier: Apache-2.0

//! Block tree terms reader for the Lucene 103 codec.
//!
//! Reads `.tim` (terms dictionary), `.tip` (terms index / trie), and `.tmd`
//! (terms metadata) files written by [`super::blocktree_writer`].
//!
//! Follows the Java `Lucene103BlockTreeTermsReader` design: metadata is read
//! eagerly from `.tmd` during construction, while `.tim` and `.tip` bytes are
//! retained via [`FileBacking`] for lazy term enumeration at query time.

use std::collections::HashMap;
use std::fmt;
use std::io;

use crate::codecs::codec_file_handle::{CodecFileHandle, IndexFile};
use crate::codecs::codec_headers;
use crate::codecs::lucene103::postings_format;
use crate::document::IndexOptions;
use crate::index::FieldInfos;
use crate::index::terms::{Terms, TermsEnum};
use crate::store::{Directory, FileBacking, IndexInput};

/// Per-field metadata read from the `.tmd` terms metadata file.
///
/// Holds all lightweight data needed to access a field's terms dictionary.
/// Bytes are owned by the parent [`BlockTreeTermsReader`]; [`FieldReader`]
/// views are constructed on demand via [`BlockTreeTermsReader::terms`].
#[derive(Debug)]
pub struct FieldMeta {
    /// Field number from [`FieldInfos`].
    pub field_number: u32,
    /// Total number of unique terms in this field.
    pub num_terms: i64,
    /// Sum of `totalTermFreq` across all terms in this field.
    pub sum_total_term_freq: i64,
    /// Sum of `docFreq` across all terms in this field.
    pub sum_doc_freq: i64,
    /// Number of documents that have at least one term in this field.
    pub doc_count: i32,
    /// Lexicographically smallest term in this field.
    pub min_term: Box<[u8]>,
    /// Lexicographically largest term in this field.
    pub max_term: Box<[u8]>,
    /// Index options for this field (controls freqs, positions, offsets).
    pub index_options: IndexOptions,
    /// Whether this field stores payloads.
    pub has_payloads: bool,
    /// Trie index start file pointer in `.tip`.
    pub index_start: i64,
    /// Trie root node file pointer in `.tip`.
    pub root_fp: i64,
    /// Trie index end file pointer in `.tip`.
    pub index_end: i64,
}

/// Borrowed view of a single field's terms dictionary.
///
/// Constructed on demand by [`BlockTreeTermsReader::terms`]; carries
/// references to the parent's `.tim` / `.tip` bytes and the field's
/// [`FieldMeta`].
#[derive(Debug)]
pub struct FieldReader<'a> {
    meta: &'a FieldMeta,
    /// Whole `.tim` bytes.
    terms_bytes: &'a [u8],
    /// Whole `.tip` bytes. The field's trie region is
    /// `&index_bytes[index_start..index_end]`.
    index_bytes: &'a [u8],
}

impl<'a> FieldReader<'a> {
    /// Returns the underlying metadata for this field.
    pub fn meta(&self) -> &'a FieldMeta {
        self.meta
    }

    /// Creates a [`super::trie_reader::TrieReader`] for this field.
    ///
    /// Constructs an `IndexInput` over the field's `.tip` region and loads
    /// the root node (a handful of bytes). `root_fp` is already stored
    /// slice-relative by the writer (see `TrieBuilder::save_nodes`).
    pub fn new_trie_reader(&self) -> io::Result<super::trie_reader::TrieReader<'a>> {
        let slice = &self.index_bytes[self.meta.index_start as usize..self.meta.index_end as usize];
        let access = IndexInput::new("trie index", slice);
        super::trie_reader::TrieReader::new(access, self.meta.root_fp)
    }

    /// Returns an [`IndexInput`] over the field's `.tip` region (for floor
    /// data reads in [`super::segment_terms_enum::SegmentTermsEnum`]).
    pub(crate) fn index_input(&self) -> IndexInput<'a> {
        let slice = &self.index_bytes[self.meta.index_start as usize..self.meta.index_end as usize];
        IndexInput::new("index input", slice)
    }

    /// Returns an [`IndexInput`] over the whole `.tim` terms dictionary.
    pub(crate) fn terms_input(&self) -> IndexInput<'a> {
        IndexInput::new("terms_in", self.terms_bytes)
    }
}

impl Terms for FieldReader<'_> {
    fn iterator(&self) -> io::Result<Box<dyn TermsEnum + '_>> {
        let terms_in = self.terms_input();
        let index_in = self.index_input();
        let trie = self.new_trie_reader()?;
        Ok(Box::new(super::segment_terms_enum::SegmentTermsEnum::new(
            terms_in,
            index_in,
            trie,
            self.meta.index_options,
        )))
    }

    fn size(&self) -> i64 {
        self.meta.num_terms
    }

    fn get_sum_total_term_freq(&self) -> i64 {
        self.meta.sum_total_term_freq
    }

    fn get_sum_doc_freq(&self) -> i64 {
        self.meta.sum_doc_freq
    }

    fn get_doc_count(&self) -> i32 {
        self.meta.doc_count
    }

    fn has_freqs(&self) -> bool {
        self.meta.index_options >= IndexOptions::DocsAndFreqs
    }

    fn has_offsets(&self) -> bool {
        self.meta.index_options >= IndexOptions::DocsAndFreqsAndPositionsAndOffsets
    }

    fn has_positions(&self) -> bool {
        self.meta.index_options >= IndexOptions::DocsAndFreqsAndPositions
    }

    fn has_payloads(&self) -> bool {
        self.meta.has_payloads
    }

    fn get_min(&self) -> Option<&[u8]> {
        Some(&self.meta.min_term)
    }

    fn get_max(&self) -> Option<&[u8]> {
        Some(&self.meta.max_term)
    }
}

/// Block tree terms reader that owns the `.tim` and `.tip` bytes and per-field
/// metadata.
///
/// Mirrors Java's `Lucene103BlockTreeTermsReader`. Construction reads all
/// per-field metadata from `.tmd` (then drops it); `.tim` and `.tip`
/// [`FileBacking`]s are retained for the lifetime of this reader.
pub struct BlockTreeTermsReader {
    /// Owned `.tim` terms dictionary bytes.
    terms_file: FileBacking,
    /// Owned `.tip` terms index bytes.
    index_file: FileBacking,
    /// Per-field metadata, keyed by field number.
    fields: HashMap<u32, FieldMeta>,
}

impl BlockTreeTermsReader {
    /// Opens the block tree terms reader for a segment.
    ///
    /// Reads `.tmd` metadata eagerly (verifying its full CRC), validates
    /// headers on `.tim` and `.tip`, and performs an O(1) footer sanity check
    /// on both. The `.tmd` bytes are dropped after construction.
    pub fn open(
        directory: &dyn Directory,
        segment_name: &str,
        segment_suffix: &str,
        segment_id: &[u8; codec_headers::ID_LENGTH],
        field_infos: &FieldInfos,
    ) -> io::Result<Self> {
        let terms_handle = CodecFileHandle::open(
            directory,
            IndexFile::TermsData,
            segment_name,
            segment_id,
            segment_suffix,
        )?;
        let index_handle = CodecFileHandle::open(
            directory,
            IndexFile::TermsIndex,
            segment_name,
            segment_id,
            segment_suffix,
        )?;
        let meta_handle = CodecFileHandle::open(
            directory,
            IndexFile::TermsMeta,
            segment_name,
            segment_id,
            segment_suffix,
        )?;

        // Cross-file version check: .tmd must agree with .tip on the
        // blocktree version.
        if meta_handle.version() != index_handle.version() {
            return Err(io::Error::other(format!(
                "blocktree version mismatch: meta={}, index={}",
                meta_handle.version(),
                index_handle.version()
            )));
        }

        let mut meta_in = meta_handle.body();

        // Embedded postings header: a second `check_index_header` on the same
        // .tmd body. The handle abstraction validates the outer terms-meta
        // header; this inner one stays explicit but its constants live in
        // postings_format where they belong.
        postings_format::check_embedded_postings_header(&mut meta_in, segment_id, segment_suffix)?;
        let block_size = meta_in.read_vint()?;
        if block_size != postings_format::BLOCK_SIZE as i32 {
            return Err(io::Error::other(format!(
                "invalid block size: {block_size}, expected {}",
                postings_format::BLOCK_SIZE
            )));
        }

        // Read per-field metadata
        let num_fields = meta_in.read_vint()?;
        if num_fields < 0 {
            return Err(io::Error::other(format!("invalid numFields: {num_fields}")));
        }

        let mut fields = HashMap::with_capacity(num_fields as usize);
        for _ in 0..num_fields {
            let field_meta = read_field_metadata(&mut meta_in, field_infos)?;
            let field_number = field_meta.field_number;
            if fields.insert(field_number, field_meta).is_some() {
                return Err(io::Error::other(format!(
                    "duplicate field number: {field_number}"
                )));
            }
        }

        // Read .tip and .tim end pointers (for future integrity checks)
        let _index_length = meta_in.read_le_long()?;
        let _terms_length = meta_in.read_le_long()?;

        Ok(Self {
            terms_file: terms_handle.into_backing(),
            index_file: index_handle.into_backing(),
            fields,
        })
    }

    /// Returns the [`FieldReader`] for the given field number, if it exists.
    pub fn field_reader(&self, field_number: u32) -> Option<FieldReader<'_>> {
        let meta = self.fields.get(&field_number)?;
        Some(FieldReader {
            meta,
            terms_bytes: self.terms_file.as_bytes(),
            index_bytes: self.index_file.as_bytes(),
        })
    }

    /// Returns the [`FieldReader`] for the given field name, if it exists.
    ///
    /// Matches Java's `FieldsProducer.terms(String)`.
    pub fn terms(&self, field_name: &str, field_infos: &FieldInfos) -> Option<FieldReader<'_>> {
        let fi = field_infos.field_info_by_name(field_name)?;
        self.field_reader(fi.number())
    }

    /// Returns the number of indexed fields.
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    /// Returns `true` if there are no indexed fields.
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    /// Returns the whole `.tim` terms dictionary bytes. Primarily useful for
    /// low-level frame tests that need to construct their own
    /// `IndexInput` over the dictionary.
    pub fn terms_bytes(&self) -> &[u8] {
        self.terms_file.as_bytes()
    }
}

impl fmt::Debug for BlockTreeTermsReader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BlockTreeTermsReader")
            .field("terms_len", &self.terms_file.len())
            .field("index_len", &self.index_file.len())
            .field("field_count", &self.fields.len())
            .finish()
    }
}

/// Reads a single field's metadata from the `.tmd` stream.
fn read_field_metadata(
    input: &mut IndexInput<'_>,
    field_infos: &FieldInfos,
) -> io::Result<FieldMeta> {
    let field_number = input.read_vint()?;
    if field_number < 0 {
        return Err(io::Error::other(format!(
            "invalid field number: {field_number}"
        )));
    }
    let field_number = field_number as u32;

    let num_terms = input.read_vlong()?;
    if num_terms <= 0 {
        return Err(io::Error::other(format!(
            "invalid numTerms for field {field_number}: {num_terms}"
        )));
    }

    let field_info = field_infos
        .field_info_by_number(field_number)
        .ok_or_else(|| io::Error::other(format!("invalid field number: {field_number}")))?;

    // When frequencies are omitted (DOCS only), sumDocFreq == sumTotalTermFreq
    // and only one value is written.
    let sum_total_term_freq = input.read_vlong()?;
    let sum_doc_freq = if field_info.index_options() == IndexOptions::Docs {
        sum_total_term_freq
    } else {
        input.read_vlong()?
    };

    let doc_count = input.read_vint()?;

    let min_term = read_bytes_ref(input)?;
    let max_term = read_bytes_ref(input)?;

    // Validate metadata consistency (matches Java reader checks)
    if doc_count < 0 {
        return Err(io::Error::other(format!("invalid docCount: {doc_count}")));
    }
    if sum_doc_freq < doc_count as i64 {
        return Err(io::Error::other(format!(
            "invalid sumDocFreq: {sum_doc_freq} docCount: {doc_count}"
        )));
    }
    if sum_total_term_freq < sum_doc_freq {
        return Err(io::Error::other(format!(
            "invalid sumTotalTermFreq: {sum_total_term_freq} sumDocFreq: {sum_doc_freq}"
        )));
    }

    // Trie index pointers
    let index_start = input.read_vlong()?;
    let root_fp = input.read_vlong()?;
    let index_end = input.read_vlong()?;

    Ok(FieldMeta {
        field_number,
        num_terms,
        sum_total_term_freq,
        sum_doc_freq,
        doc_count,
        min_term,
        max_term,
        index_options: field_info.index_options(),
        has_payloads: field_info.has_payloads(),
        index_start,
        root_fp,
        index_end,
    })
}

/// Reads a length-prefixed byte array (VInt length + raw bytes).
fn read_bytes_ref(input: &mut IndexInput<'_>) -> io::Result<Box<[u8]>> {
    let len = input.read_vint()?;
    if len < 0 {
        return Err(io::Error::other(format!("invalid bytes length: {len}")));
    }
    let mut buf = vec![0u8; len as usize];
    input.read_bytes(&mut buf)?;
    Ok(buf.into_boxed_slice())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codecs::competitive_impact::BufferedNormsLookup;
    use crate::codecs::lucene103::blocktree_writer::{BlockTreeTermsWriter, BufferedFieldTerms};
    use crate::document::{DocValuesType, IndexOptions, TermOffset};
    use crate::index::pipeline::terms_hash::{FreqProxTermsWriterPerField, TermsHash};
    use crate::index::{FieldInfo, FieldInfos, PointDimensionConfig};
    use crate::store::memory::MemoryDirectory;
    use crate::util::byte_block_pool::ByteBlockPool;

    fn make_field_info(name: &str, number: u32, index_options: IndexOptions) -> FieldInfo {
        FieldInfo::new(
            name.to_string(),
            number,
            false, // store_term_vector
            false, // omit_norms
            index_options,
            DocValuesType::None,
            PointDimensionConfig::default(),
        )
    }

    struct TestTerms {
        writer: FreqProxTermsWriterPerField,
        term_pool: ByteBlockPool,
        terms_hash: TermsHash,
    }

    impl TestTerms {
        fn new(field_name: &str, index_options: IndexOptions) -> Self {
            let opts = index_options;
            let term_pool = ByteBlockPool::new(32 * 1024);
            Self {
                writer: FreqProxTermsWriterPerField::new(field_name.to_string(), opts),
                term_pool,
                terms_hash: TermsHash::new(),
            }
        }

        fn add(&mut self, term: &str, doc_id: i32, position: i32) {
            self.writer.current_position = position;
            self.writer.current_offset = TermOffset::default();
            self.writer
                .add(
                    &mut self.term_pool,
                    &mut self.terms_hash,
                    term.as_bytes(),
                    doc_id,
                )
                .unwrap();
        }

        fn finalize(&mut self) {
            self.writer.flush_pending_docs(&mut self.terms_hash);
            self.writer.sort_terms(&self.term_pool);
        }
    }

    /// Add terms in doc-major order from term-major test data.
    fn add_terms_doc_major(tt: &mut TestTerms, terms: &[(&str, &[i32])]) {
        let max_doc = terms
            .iter()
            .flat_map(|(_, docs)| docs.iter())
            .copied()
            .max()
            .unwrap_or(-1);
        for doc_id in 0..=max_doc {
            for (term, doc_ids) in terms {
                if doc_ids.contains(&doc_id) {
                    tt.add(term, doc_id, 0);
                }
            }
        }
    }

    type FieldData<'a> = Vec<(&'a str, IndexOptions, Vec<(&'a str, &'a [i32])>)>;

    fn write_and_read(
        field_infos_vec: Vec<FieldInfo>,
        fields_data: FieldData<'_>,
    ) -> io::Result<BlockTreeTermsReader> {
        let field_infos = FieldInfos::new(field_infos_vec);
        let segment_name = "_0";
        let segment_suffix = "";
        let segment_id = [0u8; 16];

        let max_index_options = fields_data
            .iter()
            .map(|(_, opts, _)| *opts)
            .max()
            .unwrap_or(IndexOptions::Docs);

        let shared_dir = MemoryDirectory::create();

        {
            let mut writer = BlockTreeTermsWriter::new(
                &shared_dir,
                segment_name,
                segment_suffix,
                &segment_id,
                max_index_options,
            )?;

            for (field_number, (field_name, index_options, terms)) in fields_data.iter().enumerate()
            {
                let mut tt = TestTerms::new(field_name, *index_options);
                add_terms_doc_major(&mut tt, terms);
                tt.finalize();

                let field_terms = BufferedFieldTerms::new(
                    &tt.writer,
                    &tt.term_pool,
                    &tt.terms_hash,
                    field_name,
                    field_number as u32,
                );
                let norms = BufferedNormsLookup::no_norms();
                writer.write_field(&field_terms, &norms)?;
            }

            writer.finish()?;
        }

        BlockTreeTermsReader::open(
            &shared_dir,
            segment_name,
            segment_suffix,
            &segment_id,
            &field_infos,
        )
    }

    // Ported from org.apache.lucene.codecs.lucene103.blocktree.TestBlockTreeTermsReader
    #[test]
    fn test_roundtrip_single_field() -> io::Result<()> {
        let fi = make_field_info("title", 0, IndexOptions::Docs);
        let terms = vec![
            ("alpha", &[0, 1, 2][..]),
            ("beta", &[1, 3]),
            ("gamma", &[0, 2, 4]),
        ];

        let reader = write_and_read(vec![fi], vec![("title", IndexOptions::Docs, terms)])?;

        assert_eq!(reader.len(), 1);

        let fr = reader.field_reader(0).unwrap();
        let meta = fr.meta();
        assert_eq!(meta.field_number, 0);
        assert_eq!(meta.num_terms, 3);
        // For DOCS-only, sumTotalTermFreq == sumDocFreq
        assert_eq!(meta.sum_total_term_freq, meta.sum_doc_freq);
        assert_eq!(&*meta.min_term, b"alpha");
        assert_eq!(&*meta.max_term, b"gamma");
        assert_ge!(meta.doc_count, 1);

        Ok(())
    }

    #[test]
    fn test_roundtrip_with_freqs() -> io::Result<()> {
        let fi = make_field_info("body", 0, IndexOptions::DocsAndFreqs);
        let terms = vec![("hello", &[0, 1][..]), ("world", &[0])];

        let reader = write_and_read(vec![fi], vec![("body", IndexOptions::DocsAndFreqs, terms)])?;

        let fr = reader.field_reader(0).unwrap();
        let meta = fr.meta();
        assert_eq!(meta.num_terms, 2);
        // With freqs, sumTotalTermFreq >= sumDocFreq
        assert_ge!(meta.sum_total_term_freq, meta.sum_doc_freq);
        assert_ge!(meta.sum_doc_freq, meta.doc_count as i64);
        assert_eq!(&*meta.min_term, b"hello");
        assert_eq!(&*meta.max_term, b"world");

        Ok(())
    }

    #[test]
    fn test_roundtrip_multiple_fields() -> io::Result<()> {
        let fi0 = make_field_info("path", 0, IndexOptions::Docs);
        let fi1 = make_field_info("contents", 1, IndexOptions::DocsAndFreqsAndPositions);

        let reader = write_and_read(
            vec![fi0, fi1],
            vec![
                (
                    "path",
                    IndexOptions::Docs,
                    vec![("doc1.txt", &[0][..]), ("doc2.txt", &[1])],
                ),
                (
                    "contents",
                    IndexOptions::DocsAndFreqsAndPositions,
                    vec![("hello", &[0, 1][..]), ("world", &[0]), ("rust", &[1])],
                ),
            ],
        )?;

        assert_eq!(reader.len(), 2);

        let path_fr = reader.field_reader(0).unwrap();
        assert_eq!(path_fr.meta().num_terms, 2);

        let contents_fr = reader.field_reader(1).unwrap();
        assert_eq!(contents_fr.meta().num_terms, 3);
        assert_ge!(
            contents_fr.meta().sum_total_term_freq,
            contents_fr.meta().sum_doc_freq
        );

        // Non-existent field
        assert_none!(reader.field_reader(99));

        Ok(())
    }

    // --- Validation tests for read_field_metadata ---

    use crate::encoding::write_encoding::WriteEncoding;
    use crate::store::{DataOutput, VecOutput};

    /// Builds a valid field metadata byte stream for field 0 (DOCS-only).
    fn valid_field_metadata_bytes() -> Vec<u8> {
        let mut buf = Vec::new();
        let mut out = VecOutput(&mut buf);
        out.write_vint(0).unwrap(); // field_number
        out.write_vlong(5).unwrap(); // num_terms
        out.write_vlong(10).unwrap(); // sumTotalTermFreq (== sumDocFreq for DOCS)
        out.write_vint(3).unwrap(); // doc_count
        out.write_vint(1).unwrap(); // min_term length
        out.write_byte(b'a').unwrap(); // min_term
        out.write_vint(1).unwrap(); // max_term length
        out.write_byte(b'z').unwrap(); // max_term
        out.write_vlong(0).unwrap(); // index_start
        out.write_vlong(0).unwrap(); // root_fp
        out.write_vlong(0).unwrap(); // index_end
        buf
    }

    fn docs_field_infos() -> FieldInfos {
        FieldInfos::new(vec![make_field_info("test", 0, IndexOptions::Docs)])
    }

    #[test]
    fn test_valid_field_metadata_parses() {
        let data = valid_field_metadata_bytes();
        let fi = docs_field_infos();
        let mut input = IndexInput::unnamed(&data);
        let meta = read_field_metadata(&mut input, &fi).unwrap();
        assert_eq!(meta.field_number, 0);
        assert_eq!(meta.num_terms, 5);
        assert_eq!(meta.doc_count, 3);
        assert_eq!(&*meta.min_term, b"a");
        assert_eq!(&*meta.max_term, b"z");
    }

    #[test]
    fn test_invalid_num_terms_zero() {
        let mut buf = Vec::new();
        let mut out = VecOutput(&mut buf);
        out.write_vint(0).unwrap(); // field_number
        out.write_vlong(0).unwrap(); // num_terms = 0 (invalid)

        let fi = docs_field_infos();
        let mut input = IndexInput::unnamed(&buf);
        let err = read_field_metadata(&mut input, &fi).unwrap_err();
        assert_contains!(err.to_string(), "invalid numTerms");
    }

    #[test]
    fn test_invalid_field_number_not_in_infos() {
        let mut buf = Vec::new();
        let mut out = VecOutput(&mut buf);
        out.write_vint(99).unwrap(); // field_number not in FieldInfos
        out.write_vlong(1).unwrap(); // num_terms

        let fi = docs_field_infos();
        let mut input = IndexInput::unnamed(&buf);
        let err = read_field_metadata(&mut input, &fi).unwrap_err();
        assert_contains!(err.to_string(), "invalid field number");
    }

    #[test]
    fn test_invalid_sum_doc_freq_less_than_doc_count() {
        let mut buf = Vec::new();
        let mut out = VecOutput(&mut buf);
        out.write_vint(0).unwrap(); // field_number
        out.write_vlong(5).unwrap(); // num_terms
        out.write_vlong(2).unwrap(); // sumTotalTermFreq (== sumDocFreq for DOCS)
        out.write_vint(10).unwrap(); // doc_count > sumDocFreq (invalid)
        out.write_vint(0).unwrap(); // min_term (empty)
        out.write_vint(0).unwrap(); // max_term (empty)

        let fi = docs_field_infos();
        let mut input = IndexInput::unnamed(&buf);
        let err = read_field_metadata(&mut input, &fi).unwrap_err();
        assert_contains!(err.to_string(), "invalid sumDocFreq");
    }

    #[test]
    fn test_invalid_sum_total_term_freq_less_than_sum_doc_freq() {
        // Need a non-DOCS field to have separate sumTotalTermFreq and sumDocFreq
        let fi = FieldInfos::new(vec![make_field_info("body", 0, IndexOptions::DocsAndFreqs)]);

        let mut buf = Vec::new();
        let mut out = VecOutput(&mut buf);
        out.write_vint(0).unwrap(); // field_number
        out.write_vlong(5).unwrap(); // num_terms
        out.write_vlong(3).unwrap(); // sumTotalTermFreq
        out.write_vlong(5).unwrap(); // sumDocFreq > sumTotalTermFreq (invalid)
        out.write_vint(2).unwrap(); // doc_count
        out.write_vint(0).unwrap(); // min_term (empty)
        out.write_vint(0).unwrap(); // max_term (empty)

        let mut input = IndexInput::unnamed(&buf);
        let err = read_field_metadata(&mut input, &fi).unwrap_err();
        assert_contains!(err.to_string(), "invalid sumTotalTermFreq");
    }

    #[test]
    fn test_open_missing_file_returns_error() {
        let dir = MemoryDirectory::create();
        let fi = docs_field_infos();
        let result = BlockTreeTermsReader::open(&dir, "_0", "", &[0u8; 16], &fi);
        assert!(result.is_err());
    }

    // --- Terms trait tests ---

    #[test]
    fn test_terms_trait_docs_only() -> io::Result<()> {
        let fi = make_field_info("title", 0, IndexOptions::Docs);
        let terms = vec![
            ("alpha", &[0, 1, 2][..]),
            ("beta", &[1, 3]),
            ("gamma", &[0, 2, 4]),
        ];

        let reader = write_and_read(vec![fi], vec![("title", IndexOptions::Docs, terms)])?;
        let fr = reader.field_reader(0).unwrap();

        assert_eq!(fr.size(), 3);
        assert_eq!(fr.get_doc_count(), 5);
        assert_eq!(fr.get_sum_total_term_freq(), fr.get_sum_doc_freq());
        assert_eq!(fr.get_min(), Some(b"alpha".as_slice()));
        assert_eq!(fr.get_max(), Some(b"gamma".as_slice()));
        assert!(!fr.has_freqs());
        assert!(!fr.has_positions());
        assert!(!fr.has_offsets());
        assert!(!fr.has_payloads());

        Ok(())
    }

    #[test]
    fn test_terms_trait_with_freqs_and_positions() -> io::Result<()> {
        let fi = make_field_info("body", 0, IndexOptions::DocsAndFreqsAndPositions);
        let terms = vec![("hello", &[0, 1][..]), ("world", &[0])];

        let reader = write_and_read(
            vec![fi],
            vec![("body", IndexOptions::DocsAndFreqsAndPositions, terms)],
        )?;
        let fr = reader.field_reader(0).unwrap();

        assert_eq!(fr.size(), 2);
        assert!(fr.has_freqs());
        assert!(fr.has_positions());
        assert!(!fr.has_offsets());
        assert_ge!(fr.get_sum_total_term_freq(), fr.get_sum_doc_freq());

        Ok(())
    }
}
