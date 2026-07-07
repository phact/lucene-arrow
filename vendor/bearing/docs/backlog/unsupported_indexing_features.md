# Unsupported Indexing Features

This document catalogs indexing features present in Java Lucene 10.3.2 that are not yet implemented in this Rust port. It covers **indexing-only** features — querying, searching, and the `IndexReader` API are out of scope here.

The canonical reference is **Apache Lucene 10.3.2** (`reference/lucene-10.3.2/`).

---

## 1. Field Types

**Java feature:** Lucene provides many specialized field types beyond what the Rust port supports:

| Field Type | Status | Purpose |
|---|---|---|
| `NumericDocValuesField` | **Implemented** | Per-document long values for sorting/faceting |
| `BinaryDocValuesField` | **Implemented** | Per-document arbitrary byte arrays |
| `SortedDocValuesField` | **Implemented** | Per-document ordinal-mapped byte arrays |
| `SortedSetDocValuesField` | **Implemented** | Doc-values-only sorted byte arrays (single and multi-valued) |
| `SortedNumericDocValuesField` | **Implemented** | Doc-values-only sorted longs (single and multi-valued) |
| `FeatureField` | **Implemented** | Static feature scores (BM25 boosting) |
| `LatLonPoint` | **Implemented** | Latitude/longitude points for geo queries |
| Range fields | **Implemented** | `IntRange`, `LongRange`, `FloatRange`, `DoubleRange` |
**Rust status:** The Rust port supports 20 field types: `KeywordField`, `LongField`, `TextField`, `StringField`, `IntField`, `FloatField`, `DoubleField`, `StoredField`, `LatLonPoint`, `FeatureField`, 4 range types (`IntRange`, `LongRange`, `FloatRange`, `DoubleRange`), plus 5 doc-values-only types (`NumericDocValuesField`, `BinaryDocValuesField`, `SortedDocValuesField`, `SortedSetDocValuesField`, `SortedNumericDocValuesField`). The codec writes all 5 doc values types (NUMERIC, BINARY, SORTED, SORTED_SET, SORTED_NUMERIC).

Remaining field types (KNN vectors, shape fields) are deferred — see below.

## 2. Multi-Valued Fields

**Java feature:** `SORTED_SET` and `SORTED_NUMERIC` doc values support multiple values per document (e.g., a document with multiple tags or multiple timestamps).

**Rust status:** **Implemented** — the indexing chain merges values for the same doc, the codec writes address tables for multi-valued fields, and the output is byte-identical to Java Lucene.

## 3. Sparse Doc Values / Norms

**Java feature:** Fields do not need to be present in every document. Lucene tracks which documents have a given field via dense/sparse encoding and a `docsWithField` bitset.

**Rust status:** **Implemented** — IndexedDISI bitset writer encodes which documents have values. Both doc values (all 5 types) and norms support sparse fields. Output is byte-identical to Java Lucene for sparse numeric doc values.

## 4. Term Vectors

**Java feature:** Per-document term vectors (`.tvd`/`.tvx`/`.tvm` files) store the terms, frequencies, positions, and offsets for a document's fields, enabling features like "more like this" and hit highlighting.

**Rust status:** **Implemented** — writes `.tvd`/`.tvx`/`.tvm` files via `Lucene90CompressingTermVectorsWriter` format. Supports term vectors with positions, offsets, and structural payload support. Output is byte-identical to Java Lucene and cross-validated with `VerifyIndex`.

## 5. Payloads

**Java feature:** Postings can carry per-position payload bytes (stored in `.pay` files), used for custom scoring and annotation.

**Rust status:** Not implemented. No `.pay` file is written.

**Priority:** Low — payloads are a niche feature used in specialized scoring applications.

## 6. Posting Offsets

**Java feature:** Fields indexed with `IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS` store character start/end offsets per position in the postings (`.pos` file). Used for highlighting to map positions back to character ranges without re-analyzing the text.

**Rust status:** Implemented. `TextFieldBuilder::with_index_options(TokenizedIndexOptions::PositionsAndOffsets)` enables offset indexing. The `PostingsWriter` encodes offsets to the `.pay` file (PFOR blocks) and `.pos` file (VInt tail), with skip data tracking.

## 7. Index-Time Sorting

**Java feature:** `IndexWriterConfig.setIndexSort(Sort)` pre-sorts segments at flush time so that queries can exploit sorted order for early termination.

**Rust status:** Not implemented.

**Priority:** Low — an optimization, not a correctness requirement.

## 8. Analysis

**Java feature:** Rich analysis pipeline including stop word filters, synonym filters, `CharFilter` chains, per-field `AnalyzerWrapper`, and dozens of language-specific analyzers.

**Rust status:** Partial — `StandardAnalyzer` performs Unicode tokenization and lowercasing, but has no stop words and no synonym support. `CharFilter` is not implemented. All fields use the same analyzer.

**Priority:** Medium — stop words are the most impactful gap. Per-field analyzers and synonym support matter for advanced use cases.

## 9. Configuration

**Java feature:** `IndexWriterConfig` exposes many options:

| Option | Purpose |
|---|---|
| `OpenMode` | `CREATE`, `APPEND`, `CREATE_OR_APPEND` |
| Custom `Codec` | Override the codec used for writing |
| `IndexDeletionPolicy` | Control which commits are retained |
| Soft deletes field | Track soft deletes for NRT use cases |
| `Similarity` | Custom scoring model (e.g., BM25 parameters) |
| Reader pooling | Reuse `SegmentReader` instances across NRT opens |
| Segment warming | Pre-warm new segments before making them searchable |

**Rust status:** Partial — `OpenMode::CREATE` is supported. `max_buffered_docs`, `ram_buffer_size_mb`, and thread count are configurable. Other options are not implemented.

**Priority:** Medium — `APPEND` and `CREATE_OR_APPEND` modes are important for incremental indexing.

## 10. Codec Features

**Java feature:** Lucene's `PerFieldPostingsFormat` and `PerFieldDocValuesFormat` allow different fields to use different codec implementations. The `Codec` class aggregates format implementations for all file types (postings, doc values, stored fields, norms, term vectors, live docs, etc.).

**Rust status:** Partial — the Rust port uses fixed Lucene103 codec implementations for all fields. Per-field format overrides are not supported. Live docs format is not implemented.

**Priority:** Low — per-field format overrides are rarely needed in practice.

---

## Deferred

### KNN Vector Fields

**Java feature:** `KnnFloatVectorField` and `KnnByteVectorField` provide HNSW graph indexing for approximate nearest-neighbor search over float or byte vectors. Writes `.vec`, `.vex`, `.vem`, `.vemf` files via the `Lucene99HnswVectorsFormat`.

**Rust status:** Not implemented.

**Why deferred:** High complexity, low priority. Requires a full HNSW graph builder (~650 LOC algorithm), 4-file codec, and vector similarity scoring. Orthogonal to core text indexing.

### Shape Fields

**Java feature:** `LatLonShape` and `XYShape` decompose polygons, lines, and points into triangles via a `Tessellator`, encoding each as a 7-dimension point field indexed by BKD tree.

**Rust status:** Not implemented.

**Why deferred:** High complexity, low priority. Requires porting the ear-clipping `Tessellator` (~1700 LOC) and triangle encoding. Rarely used compared to core text and point field types.

### Blocked on Read Path

The following features fundamentally require a read path (segment reading, `IndexReader`, terms index parsing) that does not exist yet. They cannot be incrementally implemented on the write path alone.

### Document Operations

**Java feature:** `IndexWriter` supports `updateDocument`, `deleteDocuments` (by term, query, or doc ID), `softUpdateDocument`, bulk `addDocuments`, `deleteAll`, and doc values updates (`updateNumericDocValue`, `updateBinaryDocValue`).

**Rust status:** Not implemented. Only `add_document` (single document) is supported. There is no mechanism for updates, deletes, or bulk operations.

**Why blocked:** All operations except bulk `addDocuments` require a read path (parsing `segments_N`, reading the terms index, etc.). Deletes and updates need to resolve existing documents in prior segments; `deleteAll` needs to parse segment metadata. Bulk `addDocuments` is the only item here that is purely write-path and could be implemented independently as an API convenience over `add_document`.

### Merging

**Java feature:** `MergePolicy` (e.g., `TieredMergePolicy`, `LogByteSizeMergePolicy`) selects segments for merging. `MergeScheduler` (e.g., `ConcurrentMergeScheduler`) executes merges in background threads. `forceMerge` and `forceMergeDeletes` allow explicit merge control.

**Rust status:** Not implemented. Each flush creates a new segment; segments are never combined.

**Why blocked:** Merging fundamentally requires reading existing segments to combine them. No part of this feature is possible without a read path implementation.

### Live Docs

**Java feature:** When documents are deleted, Lucene writes a `.liv` file containing a bitset of live (non-deleted) documents and tracks `del_count` in segment metadata.

**Rust status:** Not implemented. Depends on delete support.

**Why blocked:** Required for deletes and updates to work, which themselves require the read path.

### Near-Real-Time (NRT)

**Java feature:** `IndexWriter.getReader()` returns a `DirectoryReader` that sees all changes (including uncommitted ones) for near-real-time search.

**Rust status:** Not implemented. The Rust port has no `IndexReader`.

**Why blocked:** Depends on having a search stack first.

### Two-Phase Commit

**Java feature:** `prepareCommit()` writes the new segments file but defers making it active, enabling coordination with external transactional systems. `rollback()` discards uncommitted changes.

**Rust status:** Not implemented. `commit()` is a single-phase operation. `rollback()` is not supported.

**Why blocked:** Low priority — only needed for transactional coordination with external systems.
