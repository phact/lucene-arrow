// SPDX-License-Identifier: Apache-2.0

//! `lucene-arrow-core` — the spine of lucene-arrow (SPEC §5, §6, §9).
//!
//! This crate holds the pieces every other crate hangs off:
//!
//! - [`plan`] — [`DecodePlan`]/[`EncodePlan`]: serializable descriptions of how
//!   byte ranges become typed columns (and back). No Lucene knowledge leaks
//!   into executors; no executor knowledge leaks into the codec layer.
//! - [`source`] — [`SegmentSource`]/[`ByteRange`]: bandwidth-agnostic IO.
//!   Executors see buffers (host or device pointers), never the transport.
//! - [`extent`] — contiguous byte ranges of one file, the unit of IO and of
//!   batched kernel launches (SPEC §11.2).
//! - [`meta`] — the `lucene.*` metadata namespace and frame version
//!   ([CONTRACT] keys, SPEC §7.1).
//!
//! [`DecodePlan`]: plan::DecodePlan
//! [`EncodePlan`]: plan::EncodePlan
//! [`SegmentSource`]: source::SegmentSource
//! [`ByteRange`]: source::ByteRange

pub mod cursor;
pub mod error;
pub mod extent;
pub mod meta;
pub mod plan;
pub mod source;

pub use error::{Error, Result};
pub use plan::{BlockDecode, DecodePlan, DisiPlan, EncodePlan, FieldId};
pub use source::{BufferTarget, ByteRange, MmapSource, SegmentSource};
