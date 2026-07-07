# Naming Conventions: Format, Writer, Consumer

Reference: Lucene 10.3.2 Java source — `codecs/` and `index/` packages

---

## Three Layers, Three Suffixes

The naming reflects three distinct layers with different responsibilities and calling patterns.

### `Format` — Factory Layer (`codecs` package)

Classes: `StoredFieldsFormat`, `NormsFormat`, `PostingsFormat`, `DocValuesFormat`, `TermVectorsFormat`, `PointsFormat`, `FieldInfosFormat`, `SegmentInfoFormat`, `LiveDocsFormat`, `CompoundFormat`, `KnnVectorsFormat`

- Each is a method on `Codec` (e.g., `codec.storedFieldsFormat()`, `codec.normsFormat()`)
- **Factory only** — has no state, does no I/O itself
- Returns a Writer/Consumer for writing and a Reader/Producer for reading
- Factory methods: `fieldsWriter()`, `normsConsumer()`, `fieldsReader()`, etc.
- One instance per codec, shared across all segments
- SPI entry point for pluggable format implementations

### `Writer` — Push/Streaming API (`codecs` package)

Classes: `StoredFieldsWriter`, `TermVectorsWriter`, `PointsWriter`

The caller drives a **document-at-a-time protocol**. The writer receives individual values as they arrive.

**Calling pattern:**
```
startDocument()
  writeField(fieldInfo, value)   // called per field
  writeField(fieldInfo, value)
finishDocument()
startDocument()
  ...
finishDocument()
finish(maxDoc)
close()
```

**When used:** Stored fields and term vectors stream data to their Writers during indexing, before flush. Points use a Writer but push data at flush time via `writeField(FieldInfo, PointsReader)`.

**Key characteristic:** Data flows in incrementally. The Writer must handle buffering/chunking/compression internally. The caller cannot re-iterate — data passes through once.

### `Consumer` — Pull/Iterator API (`codecs` package)

Classes: `NormsConsumer`, `DocValuesConsumer`, `FieldsConsumer`

The caller hands over an **entire field's data at once** as an iterable or iterator. The Consumer pulls values from it.

**Calling pattern:**
```
// NormsConsumer
addNormsField(fieldInfo, NumericDocValues)    // one call per field, entire field's data
close()

// DocValuesConsumer
addNumericField(fieldInfo, DocValuesProducer)
addBinaryField(fieldInfo, DocValuesProducer)
addSortedField(fieldInfo, DocValuesProducer)
close()

// FieldsConsumer
write(Fields, NormsProducer)                 // ALL fields and terms in one call
close()
```

**When used:** At flush time, after all documents have been buffered in memory. The index-layer buffers (byte pools, PackedLongValues, etc.) are wrapped in iterators/adapters and handed to the Consumer.

**Key characteristic:** The Consumer is free to iterate over the data multiple times (the javadoc for `NormsConsumer` explicitly states this). This enables multi-pass encoding strategies (e.g., Lucene90NormsConsumer makes up to 3 passes for statistics, bitset, then values).

## Index-Layer Classes (`index` package)

The `index` package contains the **indexing-side counterparts** that sit between `IndexingChain` and the codec layer. Their naming is less systematic — it reflects historical evolution rather than a consistent convention.

| Index-layer class | Codec-layer target | Interaction model |
|---|---|---|
| `StoredFieldsConsumer` | `StoredFieldsWriter` | Thin wrapper; delegates directly during indexing (push) |
| `FreqProxTermsWriter` (extends `TermsHash`) | `FieldsConsumer` | Buffers in byte pools; wraps as `FreqProxFields` at flush (pull) |
| `TermVectorsConsumer` (extends `TermsHash`) | `TermVectorsWriter` | Streams per-document during indexing (push) |
| `NormValuesWriter` | `NormsConsumer` | Buffers in `PackedLongValues`; provides `NumericDocValues` at flush (pull) |
| `NumericDocValuesWriter` | `DocValuesConsumer` | Buffers in `PackedLongValues`; provides iterator at flush (pull) |
| `BinaryDocValuesWriter` | `DocValuesConsumer` | Buffers in `PagedBytes`; provides iterator at flush (pull) |
| `SortedDocValuesWriter` | `DocValuesConsumer` | Buffers in `BytesRefHash`; sorts + provides iterator at flush (pull) |
| `SortedNumericDocValuesWriter` | `DocValuesConsumer` | Buffers in `PackedLongValues`; provides iterator at flush (pull) |
| `SortedSetDocValuesWriter` | `DocValuesConsumer` | Buffers in `BytesRefHash` + counts; sorts + provides iterator at flush (pull) |
| `PointValuesWriter` | `PointsWriter` | Buffers in `PagedBytes`; provides `MutablePointTree` at flush (push at flush) |

### Why the inconsistency

The index-layer naming predates the current codec API. `FreqProxTermsWriter` is called a "Writer" because it writes to byte pools (its own internal storage), not because it writes to disk. `TermVectorsConsumer` is called a "Consumer" because it consumes tokens from the analysis chain. `StoredFieldsConsumer` consumes field values from IndexingChain. The names describe their relationship to the *indexing* pipeline, not to the codec layer.

The codec-layer naming (`Writer` vs `Consumer`) is the architecturally meaningful distinction — it tells you whether data is pushed incrementally or pulled in bulk at flush time.
