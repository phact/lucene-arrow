# Lucene 10.3.2 Indexing Data Flow Reference

Comprehensive documentation of Lucene's indexing path, tracing every piece of data
from document ingestion through to on-disk file output. Based entirely on the
Lucene 10.3.2 Java source code in `reference/lucene-10.3.2/lucene/core/`.

## Documents

### Architecture

| # | Document | Description |
|---|----------|-------------|
| 00 | [Indexing Pipeline Overview](00_indexing_pipeline_overview.md) | Top-to-bottom flow: IndexWriter to IndexingChain, DWPT lifecycle, flush triggers, object ownership graph |
| 08 | [Layer Interactions](08_layer_interactions.md) | Cross-cutting concerns: layer boundaries, consumer dispatch, shared pool architecture, flush boundary mechanics, FieldInfo accumulation, document/segment lifecycle |
| 09 | [Naming Conventions](09_naming_conventions.md) | Format vs Writer vs Consumer suffixes: factory layer, push/streaming API, pull/iterator API, and index-layer naming |

### Per-Format Data Flow

| # | Document | Files | Description |
|---|----------|-------|-------------|
| 01 | [Stored Fields](01_stored_fields.md) | `.fdt` `.fdx` `.fdm` | Compression, chunking, write triggers, memory lifecycle |
| 02 | [Postings](02_postings.md) | `.doc` `.pos` `.pay` `.tim` `.tip` `.tmd` `.psm` | Term collection, byte pools, block encoding, BlockTree, impacts |
| 03 | [Norms](03_norms.md) | `.nvd` `.nvm` | Norm computation, SmallFloat encoding, codec strategies |
| 04 | [Points / BKD](04_points_bkd.md) | `.kdd` `.kdi` `.kdm` | Point collection, BKD tree construction, partitioning |
| 05 | [Term Vectors](05_term_vectors.md) | `.tvd` `.tvx` `.tvm` | Shared pools, per-doc reset, compression |
| 06 | [Doc Values](06_doc_values.md) | `.dvd` `.dvm` | All 5 DV types, ordinal computation, encoding strategies |
| 07 | [Metadata Files](07_metadata_files.md) | `.si` `.fnm` `.liv` `.cfs` `.cfe` `segments_N` | Segment info, field infos, live docs, compound file, codec headers |

## Complete File Inventory

All files emitted per segment during indexing:

| Extension | Format Version | Consumer | Written During |
|-----------|---------------|----------|----------------|
| `.si` | Lucene99 | SegmentInfoFormat | sealFlushedSegment() |
| `.fnm` | Lucene94 | FieldInfosFormat | flush() - last consumer |
| `.fdt` | Lucene90 | StoredFieldsWriter | streaming during indexing |
| `.fdx` | Lucene90 | StoredFieldsWriter | streaming during indexing |
| `.fdm` | Lucene90 | StoredFieldsWriter | finish() at flush |
| `.nvd` | Lucene90 | NormsConsumer | flush() - first consumer |
| `.nvm` | Lucene90 | NormsConsumer | flush() - first consumer |
| `.doc` | Lucene103 | PostingsWriter | flush() via FieldsConsumer |
| `.pos` | Lucene103 | PostingsWriter | flush() via FieldsConsumer (conditional) |
| `.pay` | Lucene103 | PostingsWriter | flush() via FieldsConsumer (conditional) |
| `.psm` | Lucene103 | PostingsWriter | flush() via FieldsConsumer |
| `.tim` | Lucene103 | BlockTreeTermsWriter | flush() via FieldsConsumer |
| `.tip` | Lucene103 | BlockTreeTermsWriter | flush() via FieldsConsumer |
| `.tmd` | Lucene103 | BlockTreeTermsWriter | flush() via FieldsConsumer |
| `.kdd` | Lucene90 | PointsWriter | flush() |
| `.kdi` | Lucene90 | PointsWriter | flush() |
| `.kdm` | Lucene90 | PointsWriter | flush() |
| `.tvd` | Lucene90 | TermVectorsWriter | streaming during indexing |
| `.tvx` | Lucene90 | TermVectorsWriter | streaming during indexing |
| `.tvm` | Lucene90 | TermVectorsWriter | finish() at flush |
| `.dvd` | Lucene90 | DocValuesConsumer | flush() |
| `.dvm` | Lucene90 | DocValuesConsumer | flush() |
| `.liv` | Lucene90 | LiveDocsFormat | only if deletions exist |
| `.cfs` | Lucene90 | CompoundFormat | sealFlushedSegment() |
| `.cfe` | Lucene90 | CompoundFormat | sealFlushedSegment() |

## Reading Guide

**Start with** `08_layer_interactions.md` to understand the overall architecture and
how layers connect. Then read `00_indexing_pipeline_overview.md` for the detailed
flow from addDocument() through flush. Finally, dive into individual format documents
as needed.

**Key architectural insight**: Stored fields and term vectors stream data to codec
writers during indexing (push model). All other consumers (postings, norms, points,
doc values) buffer in memory and are pulled by codec writers at flush time.
