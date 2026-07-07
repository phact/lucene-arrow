// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::env;
use std::io;
use std::sync::{Arc, Mutex};
use std::thread;

use log::debug;

use crate::codecs::lucene90::compound;
use crate::codecs::lucene99::segment_info_format;
use crate::codecs::lucene99::segment_info_format::SegmentInfoFieldData;
use crate::document::Document;
use crate::index::config::IndexWriterConfig;
use crate::index::index_file_names::{self, radix36 as radix_fmt};
use crate::index::pipeline::channel::{self, Receiver, Sender};
use crate::index::pipeline::flush_control::FlushControl;
use crate::index::pipeline::id_generator::IdGenerator;
use crate::index::pipeline::segment_context::SegmentContext;
use crate::index::pipeline::segment_worker::SegmentWorker;
use crate::index::segment::{FlushedSegment, SegmentId};
use crate::index::segment_infos::SegmentInfos;
use crate::store::memory::MemoryIndexOutput;
use crate::store::{Directory, SegmentFile, SharedDirectory};

/// Creates [`SegmentWorker`] instances for worker threads.
///
/// Called once per thread at startup, and again after each mid-stream
/// flush to create a replacement worker. Implementations encapsulate
/// knowledge of which consumers and analyzer to use.
///
/// The order of consumers in the worker matters — consumers are
/// flushed in order, and some consumers depend on files written by
/// earlier consumers during flush. Implementations must ensure the
/// consumer list is correctly ordered to satisfy these dependencies.
pub trait WorkerFactory: Send + Sync {
    /// Creates a new `SegmentWorker` and its flush-time `SegmentContext`
    /// for the given segment identity.
    fn create_worker(&self, segment_id: SegmentId) -> (SegmentWorker, SegmentContext);
}

/// Thread-safe delegate for creating [`SegmentWorker`] instances.
///
/// Owns the segment counter and ID generator behind a [`Mutex`], and
/// holds a reference to the [`WorkerFactory`]. Worker threads call
/// [`create_worker`](Self::create_worker) to get their initial worker
/// and to obtain replacements after mid-stream flushes.
///
/// The [`Mutex`] is only taken when a worker is created — once per
/// segment, not per document. Lock hold time is nanoseconds (one
/// counter increment + 16 random bytes).
struct WorkerSource {
    state: Mutex<WorkerSourceState>,
    factory: Arc<dyn WorkerFactory>,
}

struct WorkerSourceState {
    next_segment_num: u64,
    id_generator: Box<dyn IdGenerator>,
}

impl WorkerSource {
    fn new(id_generator: Box<dyn IdGenerator>, factory: Arc<dyn WorkerFactory>) -> Self {
        Self {
            state: Mutex::new(WorkerSourceState {
                next_segment_num: 0,
                id_generator,
            }),
            factory,
        }
    }

    /// Mints a unique [`SegmentId`] and creates a new worker + context.
    fn create_worker(&self) -> (SegmentWorker, SegmentContext) {
        let segment_id = {
            let mut state = self.state.lock().unwrap();
            let name = format!("_{}", radix_fmt(state.next_segment_num));
            state.next_segment_num += 1;
            SegmentId {
                name,
                id: state.id_generator.next_id(),
            }
        };
        self.factory.create_worker(segment_id)
    }
}

/// Worker thread loop: pulls documents from the channel, processes
/// them through the worker, and flushes segments when thresholds are
/// reached. Returns all flushed segments when the channel closes.
fn worker_thread_loop(
    doc_rx: Receiver,
    worker_source: Arc<WorkerSource>,
    flush_control: Arc<FlushControl>,
    worker_id: usize,
) -> io::Result<Vec<FlushedSegment>> {
    let mut segments = Vec::new();
    let (mut worker, mut context) = worker_source.create_worker();

    loop {
        flush_control.wait_if_stalled();
        let Some(doc) = doc_rx.recv() else { break };
        worker.add_document(doc, &context)?;

        flush_control.after_document(worker_id, worker.ram_bytes_used() as u64);

        if flush_control.should_flush(worker_id) {
            segments.push(worker.flush(&context)?);
            flush_control.reset_worker(worker_id);
            let (new_worker, new_context) = worker_source.create_worker();
            worker = new_worker;
            context = new_context;
        }
    }

    // Channel closed — flush remaining buffered data.
    let flushed = worker.flush(&context)?;
    flush_control.reset_worker(worker_id);
    if flushed.doc_count > 0 {
        segments.push(flushed);
    }

    Ok(segments)
}

/// Packages a flushed segment's files into compound format (.cfs/.cfe).
///
/// Reads sub-files from the directory, builds the compound file via the
/// existing codec writer, rewrites the .si with `is_compound_file: true`,
/// deletes the originals, and updates the segment's file list.
fn package_compound_segment(
    segment: &mut FlushedSegment,
    directory: &dyn Directory,
) -> io::Result<()> {
    let seg_name = &segment.segment_id.name;
    let si_name = index_file_names::segment_file_name(seg_name, "", "si");
    let cfs_name = index_file_names::segment_file_name(seg_name, "", "cfs");
    let cfe_name = index_file_names::segment_file_name(seg_name, "", "cfe");

    // Collect sub-files (everything except .si) for compound packaging
    let sub_file_names: Vec<&String> = segment
        .file_names
        .iter()
        .filter(|f| !f.ends_with(".si"))
        .collect();

    // Read sub-files into SegmentFile structs
    let sub_files: Vec<SegmentFile> = sub_file_names
        .iter()
        .map(|name| {
            let data = directory.read_file(name)?;
            Ok(SegmentFile {
                name: (*name).clone(),
                data,
            })
        })
        .collect::<io::Result<Vec<_>>>()?;

    // Build compound file
    let mut cfs_out = MemoryIndexOutput::new(cfs_name.clone());
    let cfe = compound::write_to(seg_name, &segment.segment_id.id, &sub_files, &mut cfs_out)?;
    directory.write_file(&cfs_name, cfs_out.bytes())?;
    directory.write_file(&cfe.name, &cfe.data)?;

    // Delete original sub-files
    for name in &sub_file_names {
        directory.delete_file(name)?;
    }

    // Rewrite .si with is_compound_file = true
    let compound_files = vec![si_name.clone(), cfs_name.clone(), cfe_name.clone()];

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
        name: seg_name.clone(),
        max_doc: segment.doc_count,
        is_compound_file: true,
        id: segment.segment_id.id,
        diagnostics,
        attributes,
        has_blocks: false,
    };

    // Delete old .si before rewriting
    directory.delete_file(&si_name)?;
    segment_info_format::write(directory, &si, &compound_files)?;

    debug!(
        "compound: packaged {} ({} sub-files → .cfs/.cfe)",
        seg_name,
        sub_file_names.len()
    );

    // Update segment's file list
    segment.file_names = compound_files;

    Ok(())
}

/// Manages the worker thread pool and document processing pipeline.
///
/// Owns the bounded channel and worker threads. Documents are sent to
/// workers via the channel. Workers process documents independently,
/// flushing segments when buffer thresholds are reached.
///
/// # Worker lifecycle
///
/// Each thread pulls a worker from the shared `WorkerSource`.
/// The thread loop is:
///
/// ```text
/// worker = worker_source.create_worker()
/// loop {
///     doc = receive from channel       // blocks if empty, exits if closed
///     worker.add_document(doc)?        // error → thread exits with Err
///     if worker.should_flush(config) {
///         flushed = worker.flush()?    // consumes worker
///         collect flushed segment
///         worker = worker_source.create_worker()
///     }
/// }
/// // channel closed — flush remaining buffered data
/// worker.flush()?
/// ```
///
/// Workers are disposable — flushing consumes the worker and drops
/// all accumulated state (pools, consumers, registry). The thread
/// then creates a brand new worker for the next segment. There is
/// no in-place reset — disposal prevents stale state from leaking
/// across segments.
///
/// # Error handling
///
/// If a worker hits an error (I/O failure, codec error), the thread
/// exits with that error. On `shutdown()`, the coordinator joins all
/// threads and returns the first error encountered. The caller
/// receives it from `commit()`. No poison flags or error channels —
/// just `Result` propagation through the thread join.
pub struct IndexCoordinator {
    sender: Sender,
    workers: Vec<thread::JoinHandle<io::Result<Vec<FlushedSegment>>>>,
    /// Whether to package segment files into compound format (.cfs/.cfe).
    use_compound_file: bool,
    /// Shared directory for writing segment infos at commit time.
    directory: SharedDirectory,
    /// Tracks committed segments and writes `segments_N`.
    segment_infos: SegmentInfos,
}

impl IndexCoordinator {
    /// Creates a new coordinator, spawning worker threads.
    pub fn new(
        config: &IndexWriterConfig,
        id_generator: Box<dyn IdGenerator>,
        directory: SharedDirectory,
        worker_factory: Arc<dyn WorkerFactory>,
    ) -> Self {
        let queue_capacity = config.get_num_threads();
        let (sender, receiver) = channel::bounded(queue_capacity);
        let worker_source = Arc::new(WorkerSource::new(id_generator, worker_factory));
        let flush_control = Arc::new(FlushControl::new(
            config.get_num_threads(),
            config.get_ram_buffer_size_mb(),
            config.get_max_buffered_docs(),
        ));

        let mut workers = Vec::with_capacity(config.get_num_threads());

        for worker_id in 0..config.get_num_threads() {
            let rx = receiver.clone();
            let source = Arc::clone(&worker_source);
            let fc = Arc::clone(&flush_control);

            let handle = thread::spawn(move || worker_thread_loop(rx, source, fc, worker_id));
            workers.push(handle);
        }

        Self {
            sender,
            workers,
            use_compound_file: config.get_use_compound_file(),
            directory,
            segment_infos: SegmentInfos::new(),
        }
    }

    /// Sends a document to the worker pool. Blocks if the queue is full.
    pub fn add_document(&self, doc: Document) -> io::Result<()> {
        self.sender.send(doc)
    }

    /// Shuts down the coordinator: closes the channel, waits for all
    /// workers to drain remaining documents and flush their final segments,
    /// then writes the `segments_N` commit point.
    ///
    /// Returns all flushed segments, or the first error from any worker.
    pub fn shutdown(mut self) -> io::Result<Vec<FlushedSegment>> {
        // Dropping the sender closes the channel, signaling workers to
        // exit their recv loop. Each worker flushes its remaining data
        // before the thread exits.
        drop(self.sender);

        let mut all_segments = Vec::new();
        for handle in self.workers {
            match handle.join() {
                Ok(result) => all_segments.extend(result?),
                Err(_) => return Err(io::Error::other("worker thread panicked")),
            }
        }

        // Package segments into compound files if configured
        if self.use_compound_file {
            for segment in &mut all_segments {
                package_compound_segment(segment, &*self.directory)?;
            }
        }

        // Write the commit point
        if !all_segments.is_empty() {
            for segment in &all_segments {
                self.segment_infos.add(segment.clone());
            }
            self.segment_infos.commit(&*self.directory)?;
        }

        Ok(all_segments)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assertables::*;
    use std::collections::HashSet;

    use crate::analysis::StandardAnalyzer;
    use crate::analysis::Token;
    use crate::index::field::Field;
    use crate::index::pipeline::consumer::{FieldConsumer, TokenInterest};
    use crate::index::pipeline::flush_control::FlushControl;
    use crate::index::pipeline::id_generator::RandomIdGenerator;
    use crate::index::pipeline::segment_accumulator::SegmentAccumulator;
    use crate::store::MemoryDirectory;

    /// Creates a disabled FlushControl for tests that don't need flush triggering.
    fn disabled_flush_control() -> Arc<FlushControl> {
        Arc::new(FlushControl::new(1, 0.0, -1))
    }

    /// Deterministic ID generator for reproducible tests.
    struct SequentialIdGenerator(u8);

    impl IdGenerator for SequentialIdGenerator {
        fn next_id(&mut self) -> [u8; 16] {
            let id = [self.0; 16];
            self.0 += 1;
            id
        }
    }

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

    /// Factory that creates workers with a no-op consumer.
    struct NoOpWorkerFactory {
        directory: SharedDirectory,
    }

    impl WorkerFactory for NoOpWorkerFactory {
        fn create_worker(&self, segment_id: SegmentId) -> (SegmentWorker, SegmentContext) {
            let context = SegmentContext {
                directory: Arc::clone(&self.directory),
                segment_name: segment_id.name.clone(),
                segment_id: segment_id.id,
            };
            let worker = SegmentWorker::new(
                segment_id,
                vec![Box::new(NoOpConsumer)],
                Box::new(StandardAnalyzer::default()),
            );
            (worker, context)
        }
    }

    /// Factory that creates workers with real stored fields + field infos
    /// consumers, producing actual files for compound packaging tests.
    struct StoredFieldsWorkerFactory {
        directory: SharedDirectory,
    }

    impl WorkerFactory for StoredFieldsWorkerFactory {
        fn create_worker(&self, segment_id: SegmentId) -> (SegmentWorker, SegmentContext) {
            use crate::index::pipeline::field_infos_consumer::FieldInfosConsumer;
            use crate::index::pipeline::stored_fields_consumer::StoredFieldsConsumer;

            let context = SegmentContext {
                directory: Arc::clone(&self.directory),
                segment_name: segment_id.name.clone(),
                segment_id: segment_id.id,
            };
            let consumers: Vec<Box<dyn FieldConsumer>> = vec![
                Box::new(StoredFieldsConsumer::new()),
                Box::new(FieldInfosConsumer::new()),
            ];
            let worker =
                SegmentWorker::new(segment_id, consumers, Box::new(StandardAnalyzer::default()));
            (worker, context)
        }
    }

    // --- WorkerSource tests ---

    #[test]
    fn worker_source_creates_sequential_segment_names() {
        let dir: SharedDirectory = MemoryDirectory::create();
        let factory = Arc::new(NoOpWorkerFactory {
            directory: Arc::clone(&dir),
        });
        let source = WorkerSource::new(Box::new(RandomIdGenerator), factory);

        let (_, ctx0) = source.create_worker();
        let (_, ctx1) = source.create_worker();
        let (_, ctx2) = source.create_worker();

        assert_eq!(ctx0.segment_name, "_0");
        assert_eq!(ctx1.segment_name, "_1");
        assert_eq!(ctx2.segment_name, "_2");
    }

    #[test]
    fn worker_source_creates_unique_segment_ids() {
        let dir: SharedDirectory = MemoryDirectory::create();
        let factory = Arc::new(NoOpWorkerFactory {
            directory: Arc::clone(&dir),
        });
        let source = WorkerSource::new(Box::new(RandomIdGenerator), factory);

        let (_, ctx0) = source.create_worker();
        let (_, ctx1) = source.create_worker();

        assert_ne!(ctx0.segment_id, ctx1.segment_id);
    }

    // --- worker_thread_loop tests ---

    fn make_doc() -> Document {
        use crate::document::DocumentBuilder;
        use crate::index::field::stored;
        DocumentBuilder::new()
            .add_field(stored("f").string("v"))
            .build()
    }

    #[test]
    fn thread_loop_flushes_on_channel_close() {
        let dir: SharedDirectory = MemoryDirectory::create();
        let factory: Arc<dyn WorkerFactory> = Arc::new(NoOpWorkerFactory {
            directory: Arc::clone(&dir),
        });
        let source = Arc::new(WorkerSource::new(
            Box::new(SequentialIdGenerator(0)),
            factory,
        ));
        let (tx, rx) = channel::bounded(4);

        tx.send(make_doc()).unwrap();
        tx.send(make_doc()).unwrap();
        drop(tx);

        let segments = worker_thread_loop(rx, source, disabled_flush_control(), 0).unwrap();
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].doc_count, 2);
    }

    #[test]
    fn thread_loop_mid_flush_creates_replacement() {
        let dir: SharedDirectory = MemoryDirectory::create();
        let factory: Arc<dyn WorkerFactory> = Arc::new(NoOpWorkerFactory {
            directory: Arc::clone(&dir),
        });
        let source = Arc::new(WorkerSource::new(
            Box::new(SequentialIdGenerator(0)),
            factory,
        ));
        let (tx, rx) = channel::bounded(10);

        for _ in 0..7 {
            tx.send(make_doc()).unwrap();
        }
        drop(tx);

        // max_buffered_docs=3 → segments of 3, 3, 1
        let fc = Arc::new(FlushControl::new(1, 0.0, 3));
        let segments = worker_thread_loop(rx, source, fc, 0).unwrap();
        assert_eq!(segments.len(), 3);
        assert_eq!(segments[0].doc_count, 3);
        assert_eq!(segments[1].doc_count, 3);
        assert_eq!(segments[2].doc_count, 1);

        // Each segment has a unique name
        let names: HashSet<_> = segments.iter().map(|s| &s.segment_id.name).collect();
        assert_eq!(names.len(), 3);
    }

    #[test]
    fn thread_loop_empty_channel_produces_no_segments() {
        let dir: SharedDirectory = MemoryDirectory::create();
        let factory: Arc<dyn WorkerFactory> = Arc::new(NoOpWorkerFactory {
            directory: Arc::clone(&dir),
        });
        let source = Arc::new(WorkerSource::new(
            Box::new(SequentialIdGenerator(0)),
            factory,
        ));
        let (tx, rx) = channel::bounded(4);
        drop(tx);

        let segments = worker_thread_loop(rx, source, disabled_flush_control(), 0).unwrap();
        assert_is_empty!(segments);
    }

    // --- package_compound_segment tests ---

    #[test]
    fn compound_packaging_creates_cfs_cfe() {
        let dir: SharedDirectory = MemoryDirectory::create();
        let factory: Arc<dyn WorkerFactory> = Arc::new(StoredFieldsWorkerFactory {
            directory: Arc::clone(&dir),
        });
        let source = Arc::new(WorkerSource::new(
            Box::new(SequentialIdGenerator(0)),
            factory,
        ));
        let (tx, rx) = channel::bounded(4);

        tx.send(make_doc()).unwrap();
        drop(tx);

        let mut segments = worker_thread_loop(rx, source, disabled_flush_control(), 0).unwrap();
        assert_eq!(segments.len(), 1);

        // Verify sub-files exist before packaging (.fdt, .fdm, .fdx, .fnm, .si)
        let pre_files = segments[0].file_names.clone();
        assert_ge!(pre_files.len(), 5);

        package_compound_segment(&mut segments[0], &*dir).unwrap();

        // After packaging: .si, .cfs, .cfe
        assert_eq!(segments[0].file_names.len(), 3);
        assert_any!(segments[0].file_names.iter(), |f: &String| f
            .ends_with(".si"));
        assert_any!(segments[0].file_names.iter(), |f: &String| f
            .ends_with(".cfs"));
        assert_any!(segments[0].file_names.iter(), |f: &String| f
            .ends_with(".cfe"));

        // Original sub-files (except .si) should be deleted
        let guard = &*dir;
        for f in &pre_files {
            if !f.ends_with(".si") {
                assert!(
                    guard.read_file(f).is_err(),
                    "original file {f} should have been deleted"
                );
            }
        }

        // Compound files should exist
        assert!(guard.read_file("_0.cfs").is_ok());
        assert!(guard.read_file("_0.cfe").is_ok());
    }
}
