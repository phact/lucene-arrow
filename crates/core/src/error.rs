// SPDX-License-Identifier: Apache-2.0

//! Shared error type for all lucene-arrow crates.

/// Crate-wide result alias.
pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// The bytes are not what the pinned codec (Lucene103) says they should
    /// be: bad magic, bad checksum, truncated file, out-of-range value.
    #[error("corrupt segment data: {0}")]
    Corrupt(String),

    /// The data is valid Lucene but outside v1 scope (SPEC §2), e.g. a
    /// codec other than Lucene103, HNSW graphs, BKD-only fields.
    #[error("unsupported: {0}")]
    Unsupported(String),

    /// A request or plan is internally inconsistent (caller error).
    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    /// Arrow-side failure (schema mismatch, buffer construction).
    #[error("arrow error: {0}")]
    Arrow(#[from] arrow_schema::ArrowError),

    /// Underlying codec library (Bearing) failure, stringified at the
    /// boundary so core does not depend on bearing.
    #[error("codec error: {0}")]
    Codec(String),
}

impl Error {
    pub fn corrupt(msg: impl Into<String>) -> Self {
        Error::Corrupt(msg.into())
    }
    pub fn unsupported(msg: impl Into<String>) -> Self {
        Error::Unsupported(msg.into())
    }
    pub fn invalid(msg: impl Into<String>) -> Self {
        Error::InvalidArgument(msg.into())
    }
}
