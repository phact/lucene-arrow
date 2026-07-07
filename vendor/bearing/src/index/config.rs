// SPDX-License-Identifier: Apache-2.0

use std::fmt;
use std::sync::Arc;

use crate::analysis::{AnalyzerFactory, StandardAnalyzerFactory};

/// Configuration for an [`IndexWriter`](crate::index::writer::IndexWriter).
///
/// # Defaults
///
/// | Field | Default |
/// |---|---|
/// | `num_threads` | `8` |
/// | `ram_buffer_size_mb` | `64.0` |
/// | `max_buffered_docs` | `-1` (disabled) |
/// | `use_compound_file` | `false` |
/// | `analyzer_factory` | [`StandardAnalyzerFactory`] |
///
/// # Example
///
/// ```
/// use bearing::index::config::IndexWriterConfig;
///
/// let config = IndexWriterConfig::default()
///     .num_threads(4)
///     .ram_buffer_size_mb(64.0)
///     .use_compound_file(true);
/// ```
pub struct IndexWriterConfig {
    num_threads: usize,
    ram_buffer_size_mb: f64,
    max_buffered_docs: i32,
    use_compound_file: bool,
    analyzer_factory: Arc<dyn AnalyzerFactory>,
}

impl Clone for IndexWriterConfig {
    fn clone(&self) -> Self {
        Self {
            num_threads: self.num_threads,
            ram_buffer_size_mb: self.ram_buffer_size_mb,
            max_buffered_docs: self.max_buffered_docs,
            use_compound_file: self.use_compound_file,
            analyzer_factory: Arc::clone(&self.analyzer_factory),
        }
    }
}

impl fmt::Debug for IndexWriterConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IndexWriterConfig")
            .field("num_threads", &self.num_threads)
            .field("ram_buffer_size_mb", &self.ram_buffer_size_mb)
            .field("max_buffered_docs", &self.max_buffered_docs)
            .field("use_compound_file", &self.use_compound_file)
            .field("analyzer_factory", &self.analyzer_factory)
            .finish()
    }
}

impl IndexWriterConfig {
    /// Sets the number of threads in the internal indexing pool.
    pub fn num_threads(mut self, n: usize) -> Self {
        self.num_threads = n;
        self
    }

    /// Sets the RAM budget in MB for buffered documents before flushing.
    pub fn ram_buffer_size_mb(mut self, mb: f64) -> Self {
        self.ram_buffer_size_mb = mb;
        self
    }

    /// Sets the maximum number of documents per segment before flushing.
    /// Use -1 to disable (flush is RAM-driven only).
    pub fn max_buffered_docs(mut self, n: i32) -> Self {
        self.max_buffered_docs = n;
        self
    }

    /// Sets whether to package segment files into compound format (.cfs/.cfe).
    pub fn use_compound_file(mut self, enabled: bool) -> Self {
        self.use_compound_file = enabled;
        self
    }

    /// Sets the analyzer factory used to create per-worker analyzers.
    pub fn analyzer_factory(mut self, factory: Arc<dyn AnalyzerFactory>) -> Self {
        self.analyzer_factory = factory;
        self
    }

    /// Returns the number of threads in the internal indexing pool.
    pub fn get_num_threads(&self) -> usize {
        self.num_threads
    }

    /// Returns the RAM budget in MB for buffered documents before flushing.
    pub fn get_ram_buffer_size_mb(&self) -> f64 {
        self.ram_buffer_size_mb
    }

    /// Returns the maximum number of documents per segment before flushing.
    pub fn get_max_buffered_docs(&self) -> i32 {
        self.max_buffered_docs
    }

    /// Returns whether segment files are packaged into compound format.
    pub fn get_use_compound_file(&self) -> bool {
        self.use_compound_file
    }

    /// Returns the analyzer factory.
    pub fn get_analyzer_factory(&self) -> &Arc<dyn AnalyzerFactory> {
        &self.analyzer_factory
    }
}

impl Default for IndexWriterConfig {
    fn default() -> Self {
        Self {
            num_threads: 8,
            ram_buffer_size_mb: 64.0,
            max_buffered_docs: -1,
            use_compound_file: false,
            analyzer_factory: Arc::new(StandardAnalyzerFactory),
        }
    }
}

#[cfg(test)]
mod tests {
    use assertables::*;

    use super::*;

    #[test]
    fn test_default_values() {
        let config = IndexWriterConfig::default();
        assert_eq!(config.get_num_threads(), 8);
        assert_in_delta!(config.get_ram_buffer_size_mb(), 64.0, f64::EPSILON);
        assert_eq!(config.get_max_buffered_docs(), -1);
        assert!(!config.get_use_compound_file());
    }

    #[test]
    fn test_builder_chain() {
        let config = IndexWriterConfig::default()
            .num_threads(4)
            .ram_buffer_size_mb(64.0)
            .max_buffered_docs(1000)
            .use_compound_file(true);

        assert_eq!(config.get_num_threads(), 4);
        assert_in_delta!(config.get_ram_buffer_size_mb(), 64.0, f64::EPSILON);
        assert_eq!(config.get_max_buffered_docs(), 1000);
        assert!(config.get_use_compound_file());
    }

    #[test]
    fn test_partial_builder() {
        let config = IndexWriterConfig::default().use_compound_file(true);

        assert_eq!(config.get_num_threads(), 8);
        assert!(config.get_use_compound_file());
    }
}
