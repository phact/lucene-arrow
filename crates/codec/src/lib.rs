// SPDX-License-Identifier: Apache-2.0

//! `lucene-arrow-codec` — thin wrapper over Bearing's Lucene103 container
//! parsing (SPEC §3.3, §5; milestone P0, SPEC §13).
//!
//! This crate is the only place that talks to Bearing. It opens a Lucene
//! segment directory (as written by Lucene 10.x / OpenSearch 3.x), parses
//! `segments_N` → `.si` → `.fnm` (transparently through `.cfs` compound
//! files), and exposes:
//!
//! - [`SegmentDirectory`] — the opened commit: typed [`SegmentMeta`] /
//!   [`FieldMeta`] inventory plus [`open_input`] for raw per-file bytes
//!   (the docvalues/vectors crates' door to `.dvm`/`.dvd`/`.vec`).
//! - [`framing`] — minimal CodecUtil header/footer validation on `&[u8]`,
//!   for downstream crates that slice files below Bearing's readers.
//!
//! The codec is pinned: anything but `Lucene103` returns
//! [`Error::Unsupported`](lucene_arrow_core::Error::Unsupported) (SPEC §3.2).
//! Bearing failures surface as
//! [`Error::Codec`](lucene_arrow_core::Error::Codec), stringified at this
//! boundary so no Bearing type leaks upward.
//!
//! [`open_input`]: SegmentDirectory::open_input

pub mod fnm;
pub mod liv;
pub mod norms;
pub mod writer;
pub mod framing;
mod segment;

pub use segment::{DocValuesKind, FieldMeta, PINNED_CODEC, SegmentDirectory, SegmentMeta, StoredVal};
