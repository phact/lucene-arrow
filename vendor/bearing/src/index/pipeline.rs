// SPDX-License-Identifier: Apache-2.0

//! Internal write pipeline modules.
//!
//! These modules implement the segment-building pipeline: document consumers,
//! field processing, flush control, and segment workers. They are not part of
//! the public API.

pub(crate) mod channel;
pub(crate) mod consumer;
pub(crate) mod coordinator;
pub(crate) mod default_worker_factory;
pub(crate) mod doc_values_consumer;
pub(crate) mod field_info_registry;
pub(crate) mod field_infos_consumer;
pub(crate) mod flush_control;
pub(crate) mod id_generator;
pub(crate) mod norms_consumer;
pub(crate) mod points_consumer;
pub(crate) mod postings_consumer;
pub(crate) mod segment_accumulator;
pub(crate) mod segment_context;
pub(crate) mod segment_worker;
pub(crate) mod stored_fields_consumer;
pub(crate) mod term_vectors_consumer;
pub(crate) mod term_vectors_consumer_per_field;
pub(crate) mod terms_hash;
