# Read Path Research

Research notes on Lucene's read/query architecture to inform Bearing's future implementation.

---

## Write vs Read: Fundamentally Different Designs

Lucene uses **separate, independent class hierarchies** for writing and reading indexes. They share almost no in-memory data structures.

### Write path (what Bearing has today)

- **Consumer** classes (`FieldsConsumer`, `StoredFieldsWriter`, `DocValuesConsumer`, `NormsConsumer`)
- Builds temporary in-memory buffers (delta arrays, compressed byte buffers, etc.)
- Flushes everything to disk in bulk, then discards the buffers
- Design philosophy: **buffer in RAM, batch-encode, flush once**

### Read path (what Bearing will build)

- **Producer** classes (`FieldsProducer`, `StoredFieldsReader`, `DocValuesProducer`, `NormsProducer`)
- Holds `IndexInput` file handles and metadata (file offsets, field shapes) — not actual data
- Reads lazily on demand via random-access seeks into index files
- Creates stateful iterators (e.g., `BlockPostingsEnum`) per query — not persistent data structures
- Design philosophy: **open file handles, store only metadata/offsets, seek + decompress on demand**

### Shared between read and write

Only metadata types are shared:

- **FieldInfo / FieldInfos** — field metadata (index options, doc values type, norms, etc.)
- **SegmentInfo / SegmentInfos** — segment metadata (doc count, codec version, diagnostics)
- **SegmentReadState / SegmentWriteState** — context objects passed to codec readers/writers (separate types, but symmetric)

The codec implementations themselves never reuse write-side buffers for reading. Each codec pair (Writer/Reader or Consumer/Producer) is completely independent.

---

## Write-Side In-Memory Structures (for contrast)

The write path builds temporary buffers that are flushed and discarded:

**Postings writer** (`Lucene103PostingsWriter`):
- `docDeltaBuffer[BLOCK_SIZE]`, `freqBuffer[BLOCK_SIZE]`, `posDeltaBuffer[BLOCK_SIZE]`
- `payloadLengthBuffer`, `offsetStartDeltaBuffer`, `offsetLengthBuffer`
- `ByteBuffersDataOutput` scratch buffers for encoding
- `FixedBitSet` for dense doc encoding

**Stored fields writer** (`Lucene90CompressingStoredFieldsWriter`):
- `bufferedDocs` — accumulates fields for current chunk
- `numStoredFields[]`, `endOffsets[]` — per-doc metadata
- Compressor instance — chunks are compressed when full, buffers reused

---

## Read-Side In-Memory Structures

The read path stores metadata and file handles, not data:

**Postings reader** (`Lucene103PostingsReader`):
- `IndexInput docIn, posIn, payIn` — file pointers for lazy seeking
- Skip metadata (`maxNumImpactsAtLevel0/1`, `maxImpactNumBytesAtLevel0/1`)
- Creates `BlockPostingsEnum` on demand — a stateful iterator, not a data structure

**Stored fields reader** (`Lucene90CompressingStoredFieldsReader`):
- `IndexInput fieldsStream` — lazy file access
- `FieldsIndex indexReader` — chunk index for random access
- `Decompressor` — per-thread, decompresses individual chunks on demand
- `BlockState` — single reusable state machine
- Small prefetch cache (16 entries)

**Doc values reader** (`Lucene90DocValuesProducer`):
- `IntObjectHashMap<NumericEntry>`, `<BinaryEntry>`, `<SortedEntry>`, etc. — metadata pointers only
- `IndexInput data` — lazy file access
- Values are read from `IndexInput` on demand, not loaded into memory

**Terms reader** (`Lucene103BlockTreeTermsReader`):
- FST (finite state transducer) for term dictionary prefix lookup
- Block metadata for navigating the block tree
- `SegmentTermsEnum` created per query — walks the block tree lazily

---

## Java Package Organization

### Principle: group by format version, not by read/write direction

Readers and writers for the same format **coexist in the same package**. There are no separate read vs write packages.

### Codec packages — mixed by format version

Each versioned codec package contains both reader and writer plus shared utilities:

```
codecs/lucene103/
├── Lucene103PostingsReader.java      # read
├── Lucene103PostingsWriter.java      # write
├── Lucene103PostingsFormat.java      # format definition
├── ForUtil.java, PForUtil.java       # shared encoding utilities
└── blocktree/
    ├── Lucene103BlockTreeTermsReader.java
    ├── Lucene103BlockTreeTermsWriter.java
    ├── FieldReader.java              # read-side helper
    ├── SegmentTermsEnum.java         # read-side iterator
    └── TrieBuilder.java              # shared utility

codecs/lucene90/
├── Lucene90DocValuesProducer.java    # read
├── Lucene90DocValuesConsumer.java    # write
├── Lucene90NormsProducer.java        # read
├── Lucene90NormsConsumer.java        # write
├── Lucene90PointsReader.java         # read
├── Lucene90PointsWriter.java         # write
├── Lucene90CompoundFormat.java       # format definition
└── IndexedDISI.java                  # shared utility
```

### Top-level `codecs/` — abstract base classes with symmetric naming

Format-agnostic abstractions live in `codecs/` directly:

| Write | Read |
|---|---|
| `DocValuesConsumer` | `DocValuesProducer` |
| `NormsConsumer` | `NormsProducer` |
| `StoredFieldsWriter` | `StoredFieldsReader` |
| `PostingsWriterBase` | `PostingsReaderBase` |
| `PointsWriter` | `PointsReader` |
| `TermVectorsWriter` | `TermVectorsReader` |
| `KnnVectorsWriter` | `KnnVectorsReader` |

Shared utilities also live here:
- `CodecUtil` — header/footer/checksum utilities (used by both)
- `BlockTermState` — term state for block-based codecs
- `CompetitiveImpactAccumulator` — impact accumulation
- `codecs/compressing/` — compression modes shared by multiple codecs
- `codecs/perfield/` — per-field format routing

### `index/` package — heavily mixed (201 files)

Both `IndexWriter` and `IndexReader` hierarchies live in the same package alongside shared types.

**Write-side classes:**
- `IndexWriter`, `IndexWriterConfig`
- `DocumentsWriter`, `DocumentsWriterPerThread`, flush control
- `FreqProxTermsWriter`, `StoredFieldsConsumer`
- DocValues writers: `BinaryDocValuesWriter`, `NumericDocValuesWriter`, `SortedDocValuesWriter`, etc.
- Merge infrastructure: `SegmentMerger`, `MergePolicy`, `MergeScheduler`

**Read-side classes:**
- `IndexReader` (abstract), `DirectoryReader`, `StandardDirectoryReader`
- `LeafReader`, `CodecReader`, `SegmentReader`
- `CompositeReader`, `BaseCompositeReader`, `MultiReader`
- Filter readers: `FilterDirectoryReader`, `FilterCodecReader`, `FilterLeafReader`
- Iterators: `Terms`, `TermsEnum`, `PostingsEnum`, `ImpactsEnum`
- DocValues: `BinaryDocValues`, `NumericDocValues`, `SortedDocValues`, `SortedNumericDocValues`, `SortedSetDocValues`
- `StoredFields`, `TermVectors`, `PointValues`
- `SegmentCoreReaders` — opens and caches codec readers per segment

**Shared infrastructure:**
- `SegmentInfo`, `SegmentInfos`, `SegmentCommitInfo`
- `SegmentReadState`, `SegmentWriteState`
- `FieldInfo`, `FieldInfos`
- `Term`, `TermState`
- `IndexableField`, `IndexableFieldType`

### `search/` — purely read-side

The only package that is exclusively read-path:
- `IndexSearcher` — query execution entry point
- `Query`, `Weight`, `Scorer`, `ScorerSupplier`, `BulkScorer`
- `Collector`, `CollectorManager`
- Query types: `BooleanQuery`, `TermQuery`, `PhraseQuery`, etc.
- Scoring: `BooleanScorer`, `ConjunctionScorer`, `DisjunctionScorer`
- Sorting and comparison infrastructure

### `store/` — neutral I/O abstraction

Used equally by readers and writers:
- `Directory` — storage abstraction
- `IndexInput` (read) / `IndexOutput` (write)
- `DataInput` / `DataOutput` — low-level primitives
- Implementations: `ByteBuffersDirectory`, `MMapDirectory`, `FSDirectory`
- Checksum utilities

### `document/` — write-heavy but shared

Primarily field types for indexing (`TextField`, `StringField`, `IntField`, etc.) but also used at read time for `Document` reconstruction.

### `analysis/` — write-only

Tokenization pipeline (`Analyzer`, `Tokenizer`, `TokenFilter`) used only during indexing. Query-time analysis reuses the same classes but the package itself is write-oriented.

### Naming conventions

| Pattern | Direction | Example |
|---|---|---|
| `*Writer` / `*Consumer` | Write | `Lucene103PostingsWriter`, `DocValuesConsumer` |
| `*Reader` / `*Producer` | Read | `Lucene103PostingsReader`, `DocValuesProducer` |
| `*Format` | Both (factory) | `PostingsFormat`, `DocValuesFormat` |
| `*Enum` | Read (iterator) | `TermsEnum`, `PostingsEnum` |
| `Filter*` | Read (wrapper) | `FilterLeafReader`, `FilterCodecReader` |

---

## IndexReader Hierarchy

The read-side reader hierarchy provides the unified public API:

```
IndexReader (abstract)
├── CompositeReader (abstract) — multi-segment view
│   └── DirectoryReader — opens an index directory
│       └── StandardDirectoryReader — default implementation
└── LeafReader (abstract) — single-segment view
    └── CodecReader (abstract) — delegates to codec producers
        └── SegmentReader — reads one segment via codec
```

`DirectoryReader` wraps multiple `SegmentReader` instances (one per segment). Query execution iterates over leaf readers, executing against each segment independently.

**SegmentCoreReaders** is the key internal class that opens all codec readers for a segment:
- `FieldsProducer` (terms + postings)
- `NormsProducer`
- `StoredFieldsReader`
- `DocValuesProducer`
- `PointsReader`

These are opened once per segment and shared across all queries on that segment.

---

## Luke and the Public Reader API

Luke (the Lucene index inspection tool) uses the **exact same public `IndexReader` API** that search code uses. It does not access codec internals.

### API surface Luke relies on

| Feature | API |
|---|---|
| Open index | `DirectoryReader.open(dir)` |
| Field metadata | `FieldInfos` / `FieldInfo` |
| Enumerate all terms | `Terms` → `TermsEnum.next()` |
| Term postings | `TermsEnum.postings()` → `PostingsEnum` |
| Stored fields | `IndexReader.storedFields().document(docid)` |
| Doc values | `NumericDocValues`, `SortedDocValues`, etc. via `LeafReader` |
| Term vectors | `IndexReader.termVectors().get(docid, field)` |
| Segment info | `SegmentInfos.readCommit()` → `SegmentCommitInfo` |
| Search | `IndexSearcher.search(query, collector)` |
| Score explanation | `IndexSearcher.explain(query, docid)` |
| Top terms by freq | `TermsEnum` iteration + `docFreq()` / `totalTermFreq()` |

Luke wraps common access patterns in an `IndexUtils` helper that handles single vs multi-segment logic, but underneath it's all standard `IndexReader` / `LeafReader` APIs.

### CheckIndex goes deeper

Lucene's `CheckIndex` diagnostic tool is the exception — it validates codec-level details like `FieldsProducer` internals for corruption detection. But this is specialized validation, not a pattern that normal consumers need.

### Key insight

**There is one public API surface to get right.** The `IndexReader` / `LeafReader` abstraction with its term, postings, doc values, and stored field iterators serves search, inspection tooling, and any other consumer equally. There is no separate "query API" vs "inspection API."

---

## Implications for Bearing

### Module structure

Follow Lucene's pattern — add readers alongside writers in existing codec modules:

```
src/codecs/lucene103/
├── postings_writer.rs      # existing
├── postings_reader.rs      # new
├── block_tree_writer.rs    # existing
├── block_tree_reader.rs    # new
├── for_util.rs             # existing, shared

src/codecs/lucene90/
├── stored_fields_writer.rs # existing
├── stored_fields_reader.rs # new
├── doc_values_consumer.rs  # existing
├── doc_values_producer.rs  # new
├── norms_consumer.rs       # existing
├── norms_producer.rs       # new
```

### New modules

```
src/search/                 # new — purely read-side
├── index_searcher.rs
├── query.rs
├── term_query.rs
├── boolean_query.rs
├── scorer.rs
├── collector.rs
└── ...

src/index/
├── index_writer.rs         # existing
├── index_reader.rs         # new
├── directory_reader.rs     # new
├── segment_reader.rs       # new
├── terms.rs                # new — Terms, TermsEnum traits
├── postings_enum.rs        # new — PostingsEnum trait
└── ...
```

### Public API design

The `IndexReader` / `LeafReader` facade should be the primary public API. All consumers (search, inspection, analysis) use the same trait surface:

- `Terms` / `TermsEnum` — term dictionary iteration and seeking
- `PostingsEnum` — posting list iteration (docs, freqs, positions)
- `DocValues` types — per-document value access
- `StoredFields` — stored field retrieval by doc ID

### Codec readers as implementation detail

Codec-level readers (`FieldsProducer`, `PostingsReaderBase`, etc.) are internal. Public consumers interact only through the `LeafReader` trait, which delegates to the appropriate codec reader. This keeps the public API stable even if codec internals change.

### Existing shared types

Bearing already has `FieldInfo`, `FieldInfos`, `SegmentInfo`, and the codec format/utility code. These are directly reusable by the read path.
