// SPDX-License-Identifier: Apache-2.0

//! Arrow Flight front door (SPEC §8): DoGet/DoPut/DoAction.
//!
//! The server itself lands in P4. What lives here now is the wire-visible
//! request/response vocabulary — field names are [CONTRACT], values are
//! [DEFAULT] (SPEC §8.1, §10.1) — so the contract is nailed down and
//! testable before the transport exists.

pub mod engine;
pub mod hydrate;
pub mod request;
pub mod server;
pub mod write;

pub use request::{DictModes, HydrateRequest, ReadRequest, SegmentManifest, SegmentSelector, WriteJob};
pub use server::LuceneFlightService;
