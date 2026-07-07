// SPDX-License-Identifier: Apache-2.0

//! Default [`WorkerFactory`] implementation.

use std::fmt;
use std::sync::Arc;

use crate::analysis::AnalyzerFactory;
use crate::index::pipeline::consumer::FieldConsumer;
use crate::index::pipeline::coordinator::WorkerFactory;
use crate::index::pipeline::doc_values_consumer::DocValuesConsumer;
use crate::index::pipeline::field_infos_consumer::FieldInfosConsumer;
use crate::index::pipeline::norms_consumer::NormsConsumer;
use crate::index::pipeline::points_consumer::PointsConsumer;
use crate::index::pipeline::postings_consumer::PostingsConsumer;
use crate::index::pipeline::segment_context::SegmentContext;
use crate::index::pipeline::segment_worker::SegmentWorker;
use crate::index::pipeline::stored_fields_consumer::StoredFieldsConsumer;
use crate::index::pipeline::term_vectors_consumer::TermVectorsConsumer;
use crate::index::segment::SegmentId;
use crate::store::SharedDirectory;

/// Creates workers with the standard set of field consumers.
pub struct DefaultWorkerFactory {
    directory: SharedDirectory,
    analyzer_factory: Arc<dyn AnalyzerFactory>,
}

impl fmt::Debug for DefaultWorkerFactory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DefaultWorkerFactory")
            .finish_non_exhaustive()
    }
}

impl DefaultWorkerFactory {
    /// Creates a new factory backed by the given directory and analyzer factory.
    pub fn new(directory: SharedDirectory, analyzer_factory: Arc<dyn AnalyzerFactory>) -> Self {
        Self {
            directory,
            analyzer_factory,
        }
    }
}

impl WorkerFactory for DefaultWorkerFactory {
    fn create_worker(&self, segment_id: SegmentId) -> (SegmentWorker, SegmentContext) {
        let context = SegmentContext {
            directory: Arc::clone(&self.directory),
            segment_name: segment_id.name.clone(),
            segment_id: segment_id.id,
        };

        // Order matters: norms before postings, postings before term vectors
        // (TV reads term byte pool offsets set by postings), FieldInfosConsumer last.
        let consumers: Vec<Box<dyn FieldConsumer>> = vec![
            Box::new(NormsConsumer::new()),
            Box::new(DocValuesConsumer::new()),
            Box::new(PointsConsumer::new()),
            Box::new(StoredFieldsConsumer::new()),
            Box::new(PostingsConsumer::new()),
            Box::new(TermVectorsConsumer::new()),
            Box::new(FieldInfosConsumer::new()),
        ];

        let worker = SegmentWorker::new(segment_id, consumers, self.analyzer_factory.create());

        (worker, context)
    }
}
