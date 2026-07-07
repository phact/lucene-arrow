// SPDX-License-Identifier: Apache-2.0

use mem_dbg::{MemSize, SizeFlags};
use std::collections::HashMap;
use std::env;
use std::io;

use crate::analysis::Analyzer;
use crate::codecs::lucene99::segment_info_format;
use crate::codecs::lucene99::segment_info_format::SegmentInfoFieldData;
use crate::document::Document;
use crate::index::field::InvertableValue;
use crate::index::pipeline::consumer::{FieldConsumer, TokenInterest};
use crate::index::pipeline::field_info_registry::FieldInfoRegistry;
use crate::index::pipeline::segment_accumulator::SegmentAccumulator;
use crate::index::pipeline::segment_context::SegmentContext;
use crate::index::segment::{FlushedSegment, SegmentId};

/// Per-thread worker that accumulates documents into a single segment.
///
/// Owns all per-segment state. Processes documents sequentially — no
/// concurrency within a worker. Shared resources (accumulator, codec
/// writers) are passed as `&mut` to each processing step in sequence.
///
/// A worker is disposable — it is consumed by `flush()` and the
/// coordinator creates a fresh one for the next segment.
///
/// # Lifecycle
///
/// ## Construction
/// - Created by the coordinator with a segment name/ID and directory access.
/// - Allocates an empty segment accumulator and a fresh FieldInfoRegistry.
/// - Creates the set of field consumers (postings, stored fields, norms, etc.).
/// - No files are opened yet.
///
/// ## Per-document processing (`add_document`)
///
/// ### 1. Start document
/// - Call `start_document(doc_id)` on each field consumer.
/// - Some field consumers (stored fields) may need the doc ID.
/// - No files opened here unless this is the first document in the segment.
///
/// ### 2. Field iteration
/// - Iterate the document's fields sequentially.
/// - For each field:
///     - Register the field in FieldInfoRegistry (get_or_register).
///     - Call `start_field(field)` on each field consumer. The consumer
///       prepares its per-field state for incoming data.
///     - If the field is tokenized: the consumer's `start_field` return
///       value determines whether it receives tokens. Run the analyzer
///       to produce a token stream from the field's value. For each token,
///       call `add_token(field, token)` on only the consumers that opted in.
///     - Call `finish_field(field)` on each field consumer. The consumer
///       finalizes per-field per-document state (e.g., record final term
///       frequency, compute norm value).
///     - Each field consumer borrows &mut accumulator for the duration
///       of that field, then releases. No overlapping borrows.
///
/// ### 3. Finish document
/// - Call `finish_document(doc_id)` on each field consumer.
/// - Some field consumers write data at this point:
///   - Term vectors: flush per-document TV data to codec, reset TV pools.
///   - Stored fields: finalize the document's stored data.
/// - Update document count.
///
/// ### 4. Check thresholds
/// - After each document, check RAM usage and document count against
///   config thresholds.
/// - If a threshold is exceeded, the worker signals the coordinator
///   that it needs to flush, or the coordinator detects it.
///
/// ## Segment flush (`flush`)
/// 1. Call `flush()` on each field consumer in order to write accumulated
///    data to codec files via the directory. Codec writers borrow
///    &accumulator (immutable) to read accumulated data. Consumer order
///    matters — some consumers read files written by earlier consumers.
/// 2. Write the segment info (`.si`) file containing segment identity,
///    codec version, document count, and the list of files produced by
///    all consumers.
/// 3. Return `FlushedSegment` metadata to the coordinator.
/// - The worker is consumed — all state is dropped.
///
/// ## Reset (via disposal)
/// - There is no in-place reset. Flushing consumes the worker: the
///   accumulator, field consumers, registry, and all accumulated state
///   are dropped.
/// - The index coordinator creates a fresh `SegmentWorker` with a new
///   accumulator, a new `FieldInfoRegistry`, and new field consumer
///   instances.
/// - This avoids the class of bugs where stale state leaks across
///   segments (e.g., pool data from a previous segment corrupting
///   the next one).
///
/// # Ownership summary
///
/// The worker owns:
/// - SegmentAccumulator (shared data pools and cross-consumer metadata)
/// - FieldInfoRegistry (per-segment field metadata)
/// - Field consumers (each manages its own per-field accumulators)
///
/// The worker borrows:
/// - Directory (from coordinator, for creating output files)
/// - Config (thresholds for flush decisions)
pub struct SegmentWorker {
    segment_id: SegmentId,
    registry: FieldInfoRegistry,
    field_consumers: Vec<Box<dyn FieldConsumer>>,
    analyzer: Box<dyn Analyzer>,
    doc_count: i32,
    /// Shared state passed to consumers sequentially.
    accumulator: SegmentAccumulator,
}

impl SegmentWorker {
    /// Creates a new worker for a specific segment.
    pub fn new(
        segment_id: SegmentId,
        field_consumers: Vec<Box<dyn FieldConsumer>>,
        analyzer: Box<dyn Analyzer>,
    ) -> Self {
        Self {
            segment_id,
            registry: FieldInfoRegistry::new(),
            field_consumers,
            analyzer,
            doc_count: 0,
            accumulator: SegmentAccumulator::new(),
        }
    }

    /// Processes a single document through the indexing pipeline.
    pub fn add_document(&mut self, mut doc: Document, context: &SegmentContext) -> io::Result<()> {
        let doc_id = self.doc_count;

        // 1. Start document — notify all field consumers
        for consumer in &mut self.field_consumers {
            consumer.start_document(doc_id)?;
        }

        // 2. Field iteration
        for field in doc.fields() {
            let field_id = self
                .registry
                .get_or_register(field.name(), field.field_type())?;

            // 2a. Start field — every consumer prepares for this field
            //     and declares whether it wants tokens.
            let mut interested = Vec::new();
            for (i, consumer) in self.field_consumers.iter_mut().enumerate() {
                let interest = consumer.start_field(field_id, field, &mut self.accumulator)?;
                if interest == TokenInterest::WantsTokens {
                    interested.push(i);
                }
            }

            // 2b. Tokenized fields: run the analyzer once, stream tokens
            //     to only the field consumers that opted in.
            if field.field_type().is_tokenized() && !interested.is_empty() {
                let provider = match field.field_type().invertable() {
                    Some(InvertableValue::Tokenized(provider, _)) => provider,
                    _ => continue,
                };
                self.analyzer.set_reader(provider.open()?);
                while let Some(token) = self.analyzer.next_token()? {
                    for &i in &interested {
                        self.field_consumers[i].add_token(
                            field_id,
                            field,
                            &token,
                            &mut self.accumulator,
                        )?;
                    }
                }
            }

            // 2c. Finish field — every consumer finalizes per-field state
            for consumer in &mut self.field_consumers {
                consumer.finish_field(field_id, field, &mut self.accumulator)?;
            }
        }

        // 3. Finish document — notify all field consumers
        for consumer in &mut self.field_consumers {
            consumer.finish_document(doc_id, &mut self.accumulator, context)?;
        }

        self.doc_count += 1;
        self.accumulator.increment_doc_count();

        Ok(())
    }

    /// Returns the estimated RAM bytes used by this worker's accumulators.
    pub fn ram_bytes_used(&self) -> usize {
        let consumers: usize = self
            .field_consumers
            .iter()
            .map(|c| c.mem_size(SizeFlags::default()))
            .sum();
        let accumulator = self.accumulator.mem_size(SizeFlags::default());
        consumers + accumulator
    }

    /// Flushes all accumulated data as a segment to the directory.
    /// Consumes the worker — the coordinator creates a new one for
    /// the next segment.
    pub fn flush(mut self, context: &SegmentContext) -> io::Result<FlushedSegment> {
        // 1. Flush all field consumers (stored fields, field infos, etc.)
        let mut file_names = Vec::new();
        for consumer in &mut self.field_consumers {
            file_names.extend(consumer.flush(context, &self.accumulator)?);
        }

        // 2. Write .si file — must come after consumers so the file list is complete
        let mut diagnostics = HashMap::new();
        diagnostics.insert("source".to_string(), "flush".to_string());
        diagnostics.insert("os.name".to_string(), env::consts::OS.to_string());
        diagnostics.insert("os.arch".to_string(), env::consts::ARCH.to_string());

        let mut attributes = HashMap::new();
        attributes.insert(
            "Lucene90StoredFieldsFormat.mode".to_string(),
            "BEST_SPEED".to_string(),
        );

        let si = SegmentInfoFieldData {
            name: context.segment_name.clone(),
            max_doc: self.doc_count,
            is_compound_file: false,
            id: context.segment_id,
            diagnostics,
            attributes,
            has_blocks: false,
        };
        let si_name = segment_info_format::write(&*context.directory, &si, &file_names)?;
        file_names.push(si_name);

        Ok(FlushedSegment {
            segment_id: self.segment_id,
            doc_count: self.doc_count,
            file_names,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::analysis::StandardAnalyzer;
    use crate::analysis::Token;
    use crate::document::DocumentBuilder;
    use crate::index::field::{Field, text};
    use crate::index::pipeline::consumer::FieldConsumer;
    use crate::index::segment::SegmentId;
    use crate::store::MemoryDirectory;

    /// No-op consumer that returns an empty file list.
    struct NoOpConsumer;

    impl mem_dbg::MemSize for NoOpConsumer {
        fn mem_size_rec(
            &self,
            _flags: mem_dbg::SizeFlags,
            _refs: &mut mem_dbg::HashMap<usize, usize>,
        ) -> usize {
            0
        }
    }

    impl FieldConsumer for NoOpConsumer {
        fn start_document(&mut self, _doc_id: i32) -> io::Result<()> {
            Ok(())
        }
        fn start_field(
            &mut self,
            _field_id: u32,
            _field: &Field,
            _acc: &mut SegmentAccumulator,
        ) -> io::Result<TokenInterest> {
            Ok(TokenInterest::NoTokens)
        }
        fn add_token(
            &mut self,
            _field_id: u32,
            _field: &Field,
            _token: &Token<'_>,
            _acc: &mut SegmentAccumulator,
        ) -> io::Result<()> {
            Ok(())
        }
        fn finish_field(
            &mut self,
            _field_id: u32,
            _field: &Field,
            _acc: &mut SegmentAccumulator,
        ) -> io::Result<()> {
            Ok(())
        }
        fn finish_document(
            &mut self,
            _doc_id: i32,
            _acc: &mut SegmentAccumulator,
            _context: &SegmentContext,
        ) -> io::Result<()> {
            Ok(())
        }
        fn flush(
            &mut self,
            _context: &SegmentContext,
            _acc: &SegmentAccumulator,
        ) -> io::Result<Vec<String>> {
            Ok(vec![])
        }
    }

    fn test_context() -> SegmentContext {
        SegmentContext {
            directory: MemoryDirectory::create(),
            segment_name: "_0".to_string(),
            segment_id: [0u8; 16],
        }
    }

    #[test]
    fn flush_writes_si_file() {
        let context = test_context();
        let segment_id = SegmentId {
            name: "_0".to_string(),
            id: [0u8; 16],
        };
        let worker = SegmentWorker::new(
            segment_id,
            vec![Box::new(NoOpConsumer)],
            Box::new(StandardAnalyzer::default()),
        );

        let flushed = worker.flush(&context).unwrap();

        // .si should be in the file list
        assert!(flushed.file_names.contains(&"_0.si".to_string()));

        // Verify the file exists in the directory
        let guard = &*context.directory;
        let data = guard.read_file("_0.si").unwrap();
        // Header magic
        assert_eq!(&data[0..4], &[0x3f, 0xd7, 0x6c, 0x17]);
    }

    #[test]
    fn flush_includes_consumer_files_in_si() {
        /// Consumer that claims it wrote a file.
        struct FakeConsumer;

        impl mem_dbg::MemSize for FakeConsumer {
            fn mem_size_rec(
                &self,
                _flags: mem_dbg::SizeFlags,
                _refs: &mut mem_dbg::HashMap<usize, usize>,
            ) -> usize {
                0
            }
        }

        impl FieldConsumer for FakeConsumer {
            fn start_document(&mut self, _: i32) -> io::Result<()> {
                Ok(())
            }
            fn start_field(
                &mut self,
                _: u32,
                _: &Field,
                _: &mut SegmentAccumulator,
            ) -> io::Result<TokenInterest> {
                Ok(TokenInterest::NoTokens)
            }
            fn add_token(
                &mut self,
                _: u32,
                _: &Field,
                _: &Token<'_>,
                _: &mut SegmentAccumulator,
            ) -> io::Result<()> {
                Ok(())
            }
            fn finish_field(
                &mut self,
                _: u32,
                _: &Field,
                _: &mut SegmentAccumulator,
            ) -> io::Result<()> {
                Ok(())
            }
            fn finish_document(
                &mut self,
                _: i32,
                _: &mut SegmentAccumulator,
                _: &SegmentContext,
            ) -> io::Result<()> {
                Ok(())
            }
            fn flush(
                &mut self,
                _: &SegmentContext,
                _: &SegmentAccumulator,
            ) -> io::Result<Vec<String>> {
                Ok(vec!["_0.fdt".to_string(), "_0.fdx".to_string()])
            }
        }

        let context = test_context();
        let segment_id = SegmentId {
            name: "_0".to_string(),
            id: [0u8; 16],
        };
        let worker = SegmentWorker::new(
            segment_id,
            vec![Box::new(FakeConsumer)],
            Box::new(StandardAnalyzer::default()),
        );

        let flushed = worker.flush(&context).unwrap();

        // Consumer files + .si
        assert_eq!(flushed.file_names.len(), 3);
        assert!(flushed.file_names.contains(&"_0.fdt".to_string()));
        assert!(flushed.file_names.contains(&"_0.fdx".to_string()));
        assert!(flushed.file_names.contains(&"_0.si".to_string()));
    }

    #[test]
    fn add_document_with_tokenized_field() {
        /// Consumer that accepts and counts tokens.
        struct TokenCountingConsumer {
            token_count: usize,
        }

        impl mem_dbg::MemSize for TokenCountingConsumer {
            fn mem_size_rec(
                &self,
                _flags: mem_dbg::SizeFlags,
                _refs: &mut mem_dbg::HashMap<usize, usize>,
            ) -> usize {
                0
            }
        }

        impl FieldConsumer for TokenCountingConsumer {
            fn start_document(&mut self, _: i32) -> io::Result<()> {
                Ok(())
            }
            fn start_field(
                &mut self,
                _: u32,
                _: &Field,
                _: &mut SegmentAccumulator,
            ) -> io::Result<TokenInterest> {
                Ok(TokenInterest::WantsTokens)
            }
            fn add_token(
                &mut self,
                _: u32,
                _: &Field,
                _: &Token<'_>,
                _: &mut SegmentAccumulator,
            ) -> io::Result<()> {
                self.token_count += 1;
                Ok(())
            }
            fn finish_field(
                &mut self,
                _: u32,
                _: &Field,
                _: &mut SegmentAccumulator,
            ) -> io::Result<()> {
                Ok(())
            }
            fn finish_document(
                &mut self,
                _: i32,
                _: &mut SegmentAccumulator,
                _: &SegmentContext,
            ) -> io::Result<()> {
                Ok(())
            }
            fn flush(
                &mut self,
                _: &SegmentContext,
                _: &SegmentAccumulator,
            ) -> io::Result<Vec<String>> {
                Ok(vec![])
            }
        }

        let context = test_context();
        let segment_id = SegmentId {
            name: "_0".to_string(),
            id: [0u8; 16],
        };
        let consumer = TokenCountingConsumer { token_count: 0 };
        let mut worker = SegmentWorker::new(
            segment_id,
            vec![Box::new(consumer)],
            Box::new(StandardAnalyzer::default()),
        );

        let doc = DocumentBuilder::new()
            .add_field(text("body").value("hello world"))
            .build();
        worker.add_document(doc, &context).unwrap();

        // add_document ran through the tokenization path without error
        assert_eq!(worker.doc_count, 1);
    }

    #[test]
    fn ram_bytes_used_includes_consumers() {
        let segment_id = SegmentId {
            name: "_0".to_string(),
            id: [0u8; 16],
        };
        let worker = SegmentWorker::new(
            segment_id,
            vec![Box::new(NoOpConsumer)],
            Box::new(StandardAnalyzer::default()),
        );
        let ram = worker.ram_bytes_used();
        assert_gt!(ram, 0);
    }
}
