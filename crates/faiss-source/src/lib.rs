// SPDX-License-Identifier: Apache-2.0

//! OpenSearch Faiss-engine sidecar reader, read-only (SPEC §7.6).
//!
//! TODO(P2): `handle` manifest mode first (no decode — point cuVS at the
//! native Faiss file), then `vectors` extraction mode. This is the only
//! module tracking OpenSearch's custom k-NN codec.
