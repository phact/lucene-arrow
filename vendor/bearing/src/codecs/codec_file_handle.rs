// SPDX-License-Identifier: Apache-2.0

//! Unified open + validate API for codec-framed files.
//!
//! Encapsulates the per-file ritual of "open the file, validate the footer the
//! right way, validate the header, return an [`IndexInput`] positioned past
//! the header" behind a single typed call. The [`IndexFile`] enum centralises
//! every per-file-kind constant (extension, codec name, version range, footer
//! mechanism) in one `match` arm so codec readers never name them again.
//!
//! See `docs/backlog/codec_file_handle.md` for the full design rationale.

use std::io;

use crate::codecs::codec_footers::{
    FOOTER_LENGTH, retrieve_checksum, retrieve_checksum_with_length, verify_checksum,
};
use crate::codecs::codec_headers;
use crate::codecs::codec_headers::check_index_header;
use crate::index::index_file_names;
use crate::store::{Directory, FileBacking, IndexInput};

/// How the trailing codec footer should be validated at open time.
#[derive(Clone, Copy)]
pub(crate) enum FooterValidation {
    /// Eager full-file CRC. Used by metadata files (`.fdm`, `.nvm`, `.dvm`,
    /// `.tvm`, `.kdm`, `.psm`, `.tmd`, `.fnm`, `.si`, `.cfe`).
    VerifyFullCrc,
    /// O(1) footer sanity check (magic + algorithm + stored CRC). Used by
    /// large data/index files where full CRC at open time would be wasteful.
    /// Pair with [`CodecFileHandle::verify_length`] when a sibling meta file
    /// reveals the expected length.
    RetrieveStructure,
}

/// Per-file-kind metadata. Materialised by [`IndexFile::metadata`].
pub(crate) struct FileMetadata {
    pub extension: &'static str,
    pub codec_name: &'static str,
    pub version_min: i32,
    pub version_max: i32,
    pub footer: FooterValidation,
}

/// One variant per (codec, role) across every codec module that produces a
/// codec-framed segment file. `segments_N` is intentionally absent — it
/// discovers its segment id from the stream rather than validating against a
/// caller-supplied value, so it doesn't fit the per-segment contract.
#[derive(Clone, Copy)]
pub(crate) enum IndexFile {
    NormsMeta,
    NormsData,
    DocValuesMeta,
    DocValuesData,
    StoredFieldsMeta,
    StoredFieldsData,
    StoredFieldsIndex,
    TermVectorsMeta,
    TermVectorsData,
    TermVectorsIndex,
    PointsMeta,
    PointsData,
    PointsIndex,
    PostingsMeta,
    PostingsData,
    PostingsPositions,
    // PostingsPayloads,  // .pay — uncomment when postings_reader actually opens it
    TermsMeta,
    TermsData,
    TermsIndex,
    FieldInfos,
    SegmentInfo,
    CompoundEntries,
    CompoundData,
}

impl IndexFile {
    pub(crate) fn metadata(self) -> FileMetadata {
        use FooterValidation::*;

        match self {
            // ---------- norms (lucene90) ----------
            IndexFile::NormsMeta => FileMetadata {
                extension: "nvm",
                codec_name: "Lucene90NormsMetadata",
                version_min: 0,
                version_max: 0,
                footer: VerifyFullCrc,
            },
            IndexFile::NormsData => FileMetadata {
                extension: "nvd",
                codec_name: "Lucene90NormsData",
                version_min: 0,
                version_max: 0,
                footer: RetrieveStructure,
            },

            // ---------- doc values (lucene90) ----------
            IndexFile::DocValuesMeta => FileMetadata {
                extension: "dvm",
                codec_name: "Lucene90DocValuesMetadata",
                version_min: 0,
                version_max: 0,
                footer: VerifyFullCrc,
            },
            IndexFile::DocValuesData => FileMetadata {
                extension: "dvd",
                codec_name: "Lucene90DocValuesData",
                version_min: 0,
                version_max: 0,
                footer: RetrieveStructure,
            },

            // ---------- stored fields (lucene90) ----------
            IndexFile::StoredFieldsMeta => FileMetadata {
                extension: "fdm",
                codec_name: "Lucene90FieldsIndexMeta",
                version_min: 1,
                version_max: 1,
                footer: VerifyFullCrc,
            },
            IndexFile::StoredFieldsData => FileMetadata {
                extension: "fdt",
                codec_name: "Lucene90StoredFieldsFastData",
                version_min: 1,
                version_max: 1,
                footer: RetrieveStructure,
            },
            IndexFile::StoredFieldsIndex => FileMetadata {
                extension: "fdx",
                codec_name: "Lucene90FieldsIndexIdx",
                version_min: 0,
                version_max: 0,
                footer: RetrieveStructure,
            },

            // ---------- term vectors (lucene90) ----------
            IndexFile::TermVectorsMeta => FileMetadata {
                extension: "tvm",
                codec_name: "Lucene90TermVectorsIndexMeta",
                version_min: 0,
                version_max: 0,
                footer: VerifyFullCrc,
            },
            IndexFile::TermVectorsData => FileMetadata {
                extension: "tvd",
                codec_name: "Lucene90TermVectorsData",
                version_min: 0,
                version_max: 0,
                footer: RetrieveStructure,
            },
            IndexFile::TermVectorsIndex => FileMetadata {
                extension: "tvx",
                codec_name: "Lucene90TermVectorsIndexIdx",
                version_min: 0,
                version_max: 0,
                footer: RetrieveStructure,
            },

            // ---------- points (lucene90) ----------
            IndexFile::PointsMeta => FileMetadata {
                extension: "kdm",
                codec_name: "Lucene90PointsFormatMeta",
                version_min: 1,
                version_max: 1,
                footer: VerifyFullCrc,
            },
            IndexFile::PointsData => FileMetadata {
                extension: "kdd",
                codec_name: "Lucene90PointsFormatData",
                version_min: 1,
                version_max: 1,
                footer: RetrieveStructure,
            },
            IndexFile::PointsIndex => FileMetadata {
                extension: "kdi",
                codec_name: "Lucene90PointsFormatIndex",
                version_min: 1,
                version_max: 1,
                footer: RetrieveStructure,
            },

            // ---------- postings (lucene103) ----------
            IndexFile::PostingsMeta => FileMetadata {
                extension: "psm",
                codec_name: "Lucene103PostingsWriterMeta",
                version_min: 0,
                version_max: 0,
                footer: VerifyFullCrc,
            },
            IndexFile::PostingsData => FileMetadata {
                extension: "doc",
                codec_name: "Lucene103PostingsWriterDoc",
                version_min: 0,
                version_max: 0,
                footer: RetrieveStructure,
            },
            IndexFile::PostingsPositions => FileMetadata {
                extension: "pos",
                codec_name: "Lucene103PostingsWriterPos",
                version_min: 0,
                version_max: 0,
                footer: RetrieveStructure,
            },
            // IndexFile::PostingsPayloads => FileMetadata {
            //     extension: "pay",
            //     codec_name: "Lucene103PostingsWriterPay",
            //     version_min: 0,
            //     version_max: 0,
            //     footer: RetrieveStructure,
            // },

            // ---------- blocktree terms (lucene103) ----------
            IndexFile::TermsMeta => FileMetadata {
                extension: "tmd",
                codec_name: "BlockTreeTermsMeta",
                version_min: 0,
                version_max: 0,
                footer: VerifyFullCrc,
            },
            IndexFile::TermsData => FileMetadata {
                extension: "tim",
                codec_name: "BlockTreeTermsDict",
                version_min: 0,
                version_max: 0,
                footer: RetrieveStructure,
            },
            IndexFile::TermsIndex => FileMetadata {
                extension: "tip",
                codec_name: "BlockTreeTermsIndex",
                version_min: 0,
                version_max: 0,
                footer: RetrieveStructure,
            },

            // ---------- field infos (lucene94) ----------
            IndexFile::FieldInfos => FileMetadata {
                extension: "fnm",
                codec_name: "Lucene94FieldInfos",
                version_min: 2,
                version_max: 2,
                footer: VerifyFullCrc,
            },

            // ---------- segment info (lucene99) ----------
            IndexFile::SegmentInfo => FileMetadata {
                extension: "si",
                codec_name: "Lucene90SegmentInfo",
                version_min: 0,
                version_max: 0,
                footer: VerifyFullCrc,
            },

            // ---------- compound files (lucene90) ----------
            IndexFile::CompoundEntries => FileMetadata {
                extension: "cfe",
                codec_name: "Lucene90CompoundEntries",
                version_min: 0,
                version_max: 0,
                footer: VerifyFullCrc,
            },
            IndexFile::CompoundData => FileMetadata {
                extension: "cfs",
                codec_name: "Lucene90CompoundData",
                version_min: 0,
                version_max: 0,
                footer: RetrieveStructure,
            },
        }
    }
}

/// A validated open codec file. Owns the file's bytes via [`FileBacking`] and
/// caches the validated header length for fast [`body`](Self::body) construction.
pub(crate) struct CodecFileHandle {
    kind: IndexFile,
    backing: FileBacking,
    /// Byte length of the header validated at open time. [`body`](Self::body)
    /// returns an `IndexInput` starting at this offset.
    header_len: usize,
    /// Format version read from the validated header. Exposed via
    /// [`version`](Self::version) so callers can cross-check sibling files
    /// (meta vs data) when the format declares a version range.
    version: i32,
}

impl CodecFileHandle {
    /// Opens `kind` for the given segment, validates its footer per
    /// `kind.metadata().footer`, validates its header, and returns the handle.
    ///
    /// The returned handle owns the file's bytes; the underlying `FileBacking`
    /// stays alive for the handle's lifetime.
    pub(crate) fn open(
        directory: &dyn Directory,
        kind: IndexFile,
        segment_name: &str,
        segment_id: &[u8; codec_headers::ID_LENGTH],
        segment_suffix: &str,
    ) -> io::Result<Self> {
        let meta = kind.metadata();
        let file_name =
            index_file_names::segment_file_name(segment_name, segment_suffix, meta.extension);
        let backing = directory.open_file(&file_name)?;

        // Footer first (full-file CRC for meta, O(1) for data/index).
        match meta.footer {
            FooterValidation::VerifyFullCrc => verify_checksum(backing.as_bytes())?,
            FooterValidation::RetrieveStructure => {
                retrieve_checksum(backing.as_bytes())?;
            }
        }

        // Body to parse the header out of: meta files exclude the footer
        // bytes (parsing must stop before the footer); data/index files
        // include them (no body parsing happens here, just header validation).
        let header_input_bytes: &[u8] = match meta.footer {
            FooterValidation::VerifyFullCrc => {
                &backing.as_bytes()[..backing.as_bytes().len() - FOOTER_LENGTH]
            }
            FooterValidation::RetrieveStructure => backing.as_bytes(),
        };
        let mut header_input = IndexInput::new(&file_name, header_input_bytes);

        // Header validation (currently always Index — Plain reserved).
        let version = check_index_header(
            &mut header_input,
            meta.codec_name,
            meta.version_min,
            meta.version_max,
            segment_id,
            segment_suffix,
        )?;

        let header_len = codec_headers::index_header_length(meta.codec_name, segment_suffix);

        Ok(Self {
            kind,
            backing,
            header_len,
            version,
        })
    }

    /// Returns the format version read from the validated header.
    pub(crate) fn version(&self) -> i32 {
        self.version
    }

    /// Returns an [`IndexInput`] positioned immediately after the validated
    /// header. For meta files, the underlying slice excludes the footer;
    /// for data/index files, the full post-header bytes (including footer).
    pub(crate) fn body(&self) -> IndexInput<'_> {
        let bytes = self.backing.as_bytes();
        let end = match self.kind.metadata().footer {
            FooterValidation::VerifyFullCrc => bytes.len() - FOOTER_LENGTH,
            FooterValidation::RetrieveStructure => bytes.len(),
        };
        IndexInput::new(self.kind.metadata().extension, &bytes[self.header_len..end])
    }

    /// Verifies the file length against an `expected` value learned from a
    /// sibling meta file. Applies [`retrieve_checksum_with_length`] on
    /// `RetrieveStructure` files; no-op on `VerifyFullCrc` files (the full
    /// CRC at open time already implies the length).
    pub(crate) fn verify_length(&self, expected: i64) -> io::Result<()> {
        match self.kind.metadata().footer {
            FooterValidation::VerifyFullCrc => Ok(()),
            FooterValidation::RetrieveStructure => {
                retrieve_checksum_with_length(self.backing.as_bytes(), expected)?;
                Ok(())
            }
        }
    }

    /// Returns the underlying [`FileBacking`] by consuming the handle. Used by
    /// readers that need to retain the backing past handle lifetime (e.g.
    /// `PostingsReader` keeps the `.doc` backing for lazy posting list reads).
    pub(crate) fn into_backing(self) -> FileBacking {
        self.backing
    }
}
