# Layer Interactions and Cross-Cutting Concerns

Reference: Lucene 10.3.2 Java source
Focus: Layer boundaries, data ownership, lifecycle, and flush mechanics

---

## Key Architectural Constraint: Streaming vs Buffered Data

Not all index data has the same lifetime in memory. Some data can be written to disk incrementally during indexing; other data must accumulate in memory for the entire segment until flush. This is the most important cross-cutting distinction in the indexing architecture.

**Can stream to disk during indexing (per-document complete):**

| Data | Why it can stream | When it writes | Chunk trigger |
|---|---|---|---|
| Stored fields | Each document's stored values are self-contained — no cross-document dependencies | `finishDocument()` checks chunk threshold | bufferedDocs >= 80KB or 1024 docs (BEST_SPEED) |
| Term vectors | Each document's vectors are self-contained — terms, freqs, positions are all known at end-of-document | `TermVectorsConsumer.finishDocument()` streams to codec writer | 4KB or 128 docs per compressed chunk |

These are the only consumers that hold a codec Writer open *during* indexing. Their in-memory footprint is bounded by one chunk's worth of uncompressed data, regardless of how many documents are indexed.

**Must buffer in memory until segment flush:**

| Data | Why it must wait | What accumulates |
|---|---|---|
| Postings | Block encoding requires 128 doc IDs; term dictionary requires all terms sorted; FST index is segment-wide | ByteBlockPool + IntBlockPool (the dominant RAM consumer) |
| Norms | Encoding strategy (bit width) depends on segment-wide min/max | PackedLongValues.Builder per field |
| Doc values | Encoding depends on seeing all values — ordinal sort, GCD, table encoding | PackedLongValues / BytesRefHash per field |
| Points/BKD | BKD tree partitioning requires all points | PagedBytes + docID arrays per field |

These consumers open their codec Consumer/Writer only at flush time, pass all buffered data through an iterator or adapter, and then release their buffers.

**Written after flush completes:**

| Data | Why it's last |
|---|---|
| Field infos (`.fnm`) | Consumers modify FieldInfo during flush (e.g., postings set impact stats) |
| Segment info (`.si`) | Needs final doc count, file list, diagnostics |
| Compound file (`.cfs`/`.cfe`) | Aggregates all other files — they must exist first |

This three-tier timing (streaming → flush → seal) is a fundamental architectural constraint. The streaming tier frees memory early but requires the codec writer to be held open across many documents. The buffered tier has unbounded memory growth but can use multi-pass encoding. Any restructuring of the indexing path must preserve these timing relationships.

---

## 1. Layer Architecture

```
IndexWriter
  |
  |  doFlush() --> fullFlushLock --> docWriter.flushAllThreads()
  v
DocumentsWriter
  |
  |  updateDocuments() --> flushControl.obtainAndLock() --> dwpt.updateDocuments()
  |  doFlush() --> flushingDWPT.flush()
  v
DocumentsWriterPerThread  (one per thread, one segment)
  |
  |  updateDocuments() --> indexingChain.processDocument()
  |  flush() --> indexingChain.flush(flushState)
  v
IndexingChain  (THE routing layer)
  |
  |  processDocument() dispatches to all consumers
  |  flush() calls each consumer's flush in sequence
  v
Consumers:                              Codec Writers:
  StoredFieldsConsumer    ----------->    StoredFieldsWriter
  FreqProxTermsWriter     --(FreqProxFields)-->  FieldsConsumer
  TermVectorsConsumer     ----------->    TermVectorsWriter
  NormValuesWriter        ----------->    NormsConsumer
  PointValuesWriter       ----------->    PointsWriter
  DocValuesWriter<T>      ----------->    DocValuesConsumer
  VectorValuesConsumer    ----------->    KnnVectorsWriter
```

### Boundary: IndexWriter --> DocumentsWriter

**Calls crossing:**
- `IndexWriter.doFlush()` calls `docWriter.flushAllThreads()` under `fullFlushLock`
- `IndexWriter.updateDocument()/addDocument()` calls `docWriter.updateDocuments()`
- `IndexWriter` never touches DWPT or IndexingChain directly

**Data crossing:**
- Document iterables flow downward
- Sequence numbers (`long seqNo`) flow upward
- `FlushNotifications` callback interface flows downward for event signaling

**Ownership:**
- IndexWriter owns: `DocumentsWriter`, segment commit lifecycle, merge policy, global field number map
- DocumentsWriter owns: `DocumentsWriterPerThreadPool`, `DocumentsWriterFlushControl`, `DocumentsWriterDeleteQueue`, ticket queue

### Boundary: DocumentsWriter --> DWPT

**Calls crossing:**
- `flushControl.obtainAndLock()` acquires a DWPT from the pool
- `dwpt.updateDocuments(docs, delNode, ...)` processes a doc batch
- `flushControl.doAfterDocument(dwpt)` checks RAM and may return a flushingDWPT
- `dwpt.prepareFlush()` freezes the delete queue
- `dwpt.flush(flushNotifications)` creates a `FlushedSegment`

**Data crossing:**
- Iterable of IndexableField flows down
- `FlushedSegment` (containing `SegmentCommitInfo`, `FieldInfos`, frozen deletes, liveDocs, sortMap) flows up

**Ownership:**
- DocumentsWriter owns: the pool, flush control, delete queue, ticket queue
- Each DWPT owns: its `IndexingChain`, `SegmentInfo`, `FieldInfos.Builder`, `BufferedUpdates`, doc count

### Boundary: DWPT --> IndexingChain

**Calls crossing:**
- Constructor: `new IndexingChain(...)` in DWPT constructor
- Per-document: `indexingChain.processDocument(docID, fields)`
- Flush: `indexingChain.flush(flushState)` returns `Sorter.DocMap`
- Abort: `indexingChain.abort()`
- RAM accounting: `indexingChain.ramBytesUsed()` (called by `DWPT.ramBytesUsed()`)

**Data crossing:**
- `SegmentWriteState` (constructed by DWPT from `fieldInfos.finish()`) flows into `flush()`
- `Sorter.DocMap` flows back from `flush()`

**Ownership:**
- DWPT owns: the `SegmentInfo`, `FieldInfos.Builder`, `Codec` reference
- IndexingChain owns: all consumers, the `Counter bytesUsed`, the pool allocators, `PerField[]` hash table

### Boundary: IndexingChain --> Consumers

**Calls crossing (per-document):**
- `termsHash.startDocument()` / `termsHash.finishDocument(docID)`
- `storedFieldsConsumer.startDocument(docID)` / `storedFieldsConsumer.finishDocument()`
- `perField.invert()` --> `termsHashPerField.add(bytes, docID)` for indexed fields
- `storedFieldsConsumer.writeField(fieldInfo, storedValue)` for stored fields
- `docValuesWriter.addValue(docID, value)` for doc values
- `pointValuesWriter.addPackedValue(docID, bytes)` for points
- `vectorValuesConsumer.addField(fieldInfo)` for KNN vectors

**Calls crossing (flush):** See Section 4.

**Ownership:**
- IndexingChain owns all consumer instances
- IndexingChain owns the `PerField[]` hash table which holds per-field references to `TermsHashPerField`, `NormValuesWriter`, `DocValuesWriter`, `PointValuesWriter`, `KnnFieldVectorsWriter`

### Boundary: Consumers --> Codec Writers

**Calls crossing:** See Section 4 for the complete flush protocol.

**Ownership:**
- Consumers own their in-memory data structures (pools, packed values, etc.)
- Codec writers own the file handles and on-disk format
- `SegmentWriteState` bridges the two, owned by DWPT but passed to both

---

## 2. Consumer Hierarchy and Dispatch

### Complete list of consumers created by IndexingChain

Created in `IndexingChain` constructor:

| Consumer | Type | Created By |
|---|---|---|
| `storedFieldsConsumer` | `StoredFieldsConsumer` (or `SortingStoredFieldsConsumer`) | Constructor, directly |
| `termVectorsWriter` | `TermVectorsConsumer` (or `SortingTermVectorsConsumer`) | Constructor, directly |
| `termsHash` | `FreqProxTermsWriter` (which IS-A `TermsHash`) | Constructor, wrapping `termVectorsWriter` as `nextTermsHash` |
| `vectorValuesConsumer` | `VectorValuesConsumer` | Constructor, directly |
| `docValuesBytePool` | `ByteBlockPool` | Constructor (shared pool, not a consumer itself) |

Created lazily per-field in `initializeFieldInfo()`:

| Consumer | Type | Created When |
|---|---|---|
| `perField.norms` | `NormValuesWriter` | First time a field with `!omitsNorms` and `indexOptions != NONE` is seen |
| `perField.docValuesWriter` | `NumericDocValuesWriter` / `BinaryDocValuesWriter` / `SortedDocValuesWriter` / `SortedNumericDocValuesWriter` / `SortedSetDocValuesWriter` | First time a field with `docValuesType != NONE` is seen |
| `perField.pointValuesWriter` | `PointValuesWriter` | First time a field with `pointDimensionCount != 0` is seen |
| `perField.knnFieldVectorsWriter` | `KnnFieldVectorsWriter<?>` | First time a field with `vectorDimension != 0` is seen |
| `perField.termsHashPerField` | `FreqProxTermsWriterPerField` (linked to `TermVectorsConsumerPerField`) | First time a field with `indexOptions != NONE` is seen |

### Per-segment vs per-field vs per-document

| Granularity | Consumers |
|---|---|
| **Per-segment** (one instance for entire DWPT lifetime) | `StoredFieldsConsumer`, `TermVectorsConsumer`, `FreqProxTermsWriter`, `VectorValuesConsumer` |
| **Per-field** (one instance per unique field name, lives for segment lifetime) | `FreqProxTermsWriterPerField`, `TermVectorsConsumerPerField`, `NormValuesWriter`, `DocValuesWriter<T>` variants, `PointValuesWriter`, `KnnFieldVectorsWriter` |
| **Per-document** (reset or accumulated per doc) | `FieldInvertState` (reset per field per doc), `TermVectorsConsumer.perFields[]` (reset per doc via `resetFields()`), `FieldSchema` (reset per doc) |

### How processField() dispatches

`IndexingChain.processField(docID, field, perField)` checks `IndexableFieldType` flags in this order:

```java
// 1. Indexed fields (postings + term vectors)
if (fieldType.indexOptions() != IndexOptions.NONE) {
    pf.invert(docID, field, first);  // --> termsHashPerField.add() --> FreqProxTermsWriterPerField + TermVectorsConsumerPerField
    indexedField = true;
}

// 2. Stored fields
if (fieldType.stored()) {
    storedFieldsConsumer.writeField(pf.fieldInfo, storedValue);
}

// 3. Doc values
if (dvType != DocValuesType.NONE) {
    indexDocValue(docID, pf, dvType, field);  // dispatches to typed writer
}

// 4. Points
if (fieldType.pointDimensionCount() != 0) {
    pf.pointValuesWriter.addPackedValue(docID, field.binaryValue());
}

// 5. Vectors (KNN)
if (fieldType.vectorDimension() != 0) {
    indexVectorValue(docID, pf, fieldType.vectorEncoding(), field);
}
```

### IndexableFieldType flags --> consumer mapping

| Flag | Consumer(s) Activated |
|---|---|
| `indexOptions() != NONE` | `FreqProxTermsWriterPerField.add()`, and if `storeTermVectors()` also `TermVectorsConsumerPerField.add()` |
| `stored()` | `StoredFieldsConsumer.writeField()` |
| `docValuesType() != NONE` | Typed `DocValuesWriter<T>` variant |
| `pointDimensionCount() != 0` | `PointValuesWriter.addPackedValue()` |
| `vectorDimension() != 0` | `KnnFieldVectorsWriter.addValue()` |
| `storeTermVectors()` | `TermVectorsConsumerPerField` (controlled via `start()` returning `doVectors`) |
| `omitNorms() == false` (with index options) | `NormValuesWriter.addValue()` (at `PerField.finish()` time, not during `processField`) |

A single field can activate multiple consumers simultaneously. For example, a `TextField` with term vectors and stored activates: `FreqProxTermsWriterPerField`, `TermVectorsConsumerPerField`, `StoredFieldsConsumer`, and `NormValuesWriter`.

---

## 3. Shared Resources and Pool Architecture

### ByteBlockPool and IntBlockPool

**Creation:** In `IndexingChain` constructor:
```java
byteBlockAllocator = new ByteBlockPool.DirectTrackingAllocator(bytesUsed);
IntBlockPool.Allocator intBlockAllocator = new IntBlockAllocator(bytesUsed);
```

Both allocators track bytes against `IndexingChain.bytesUsed` (a `Counter`), which feeds into `DWPT.ramBytesUsed()` for flush decisions.

**Pool instances and ownership:**

| Pool | Created In | Owned By | Used By |
|---|---|---|---|
| `FreqProxTermsWriter.intPool` | `TermsHash` constructor | `FreqProxTermsWriter` | `FreqProxTermsWriterPerField`, `TermVectorsConsumerPerField` |
| `FreqProxTermsWriter.bytePool` | `TermsHash` constructor | `FreqProxTermsWriter` | `FreqProxTermsWriterPerField`, `TermVectorsConsumerPerField` |
| `FreqProxTermsWriter.termBytePool` | Set to `bytePool` in TermsHash constructor | Same as `bytePool` | `BytesRefHash` for term dedup in both FreqProx and TV |
| `TermVectorsConsumer.intPool` | `TermsHash` constructor | `TermVectorsConsumer` | `TermVectorsConsumerPerField` |
| `TermVectorsConsumer.bytePool` | `TermsHash` constructor | `TermVectorsConsumer` | `TermVectorsConsumerPerField` |
| `docValuesBytePool` | `IndexingChain` constructor | `IndexingChain` | `SortedDocValuesWriter`, `SortedSetDocValuesWriter` |

### The TWO TermsHash instances

`IndexingChain` creates a linked pair:

```java
termVectorsWriter = new TermVectorsConsumer(intBlockAllocator, byteBlockAllocator, ...);
termsHash = new FreqProxTermsWriter(intBlockAllocator, byteBlockAllocator, bytesUsed, termVectorsWriter);
```

The `TermsHash` base class constructor links them:
```java
TermsHash(... TermsHash nextTermsHash) {
    this.nextTermsHash = nextTermsHash;      // FreqProx.nextTermsHash = TermVectorsConsumer
    intPool = new IntBlockPool(intBlockAllocator);
    bytePool = new ByteBlockPool(byteBlockAllocator);
    if (nextTermsHash != null) {
        termBytePool = bytePool;                    // FreqProx is primary
        nextTermsHash.termBytePool = bytePool;      // TV shares FreqProx's termBytePool
    }
}
```

**Key insight: `termBytePool` sharing.** The primary `FreqProxTermsWriter` creates the `bytePool` and sets `termBytePool = bytePool`. It then sets `TermVectorsConsumer.termBytePool` to the SAME pool. This means both `FreqProxTermsWriterPerField` and `TermVectorsConsumerPerField` use the same `ByteBlockPool` for term byte storage (`BytesRefHash`). Term vectors can reference terms by pool offset (`addByPoolOffset`) instead of re-interning the bytes.

However, each `TermsHash` has its OWN `intPool` and `bytePool` for stream data (postings data / TV data). Only `termBytePool` is shared.

**Counter sharing:** `FreqProxTermsWriter` uses `IndexingChain.bytesUsed`. `TermVectorsConsumer` creates its OWN `Counter.newCounter()` (passed to its `TermsHash` super constructor). This means TV pool allocations are NOT tracked by the shared counter. TV memory shows up via `termVectorsWriter.accountable.ramBytesUsed()` in `IndexingChain.ramBytesUsed()`.

### Per-document reset cycle vs per-segment lifecycle

**Per-document resets:**

| What | When | How |
|---|---|---|
| `TermVectorsConsumer.perFields[]` | `startDocument()` and `finishDocument()` | `resetFields()` -- clears array, sets `numVectorFields = 0` |
| `TermVectorsConsumer` pools | `finishDocument()` after writing TV to codec | `super.reset()` -- drops all pool buffers (`intPool.reset(false,false)`, `bytePool.reset(false,false)`) |
| `TermVectorsConsumerPerField` hash | `start()` when `first == true` | `reinitHash()` -- clears `BytesRefHash`, allows re-use of same pool |
| `FieldSchema` | `PerField.reset(docID)` at start of each doc | Resets all schema fields to defaults |
| `FieldInvertState` | `PerField.invert()` when `first == true` | `invertState.reset()` -- clears position, offset, length counters |

**Per-segment lifecycle (accumulate until flush):**

| What | Lifecycle |
|---|---|
| `FreqProxTermsWriter` pools (`intPool`, `bytePool`) | Grow monotonically across all documents; never reset until flush |
| `FreqProxTermsWriterPerField.bytesHash` | Accumulates all unique terms for this field across all docs; cleared at flush |
| `StoredFieldsConsumer.writer` | Streams to disk per-document (lazy init); closed at flush |
| `NormValuesWriter.pending` | `PackedLongValues.Builder` -- accumulates one norm per doc; built at flush |
| `DocValuesWriter` variants | Accumulate values in packed builders; flushed and nulled out |
| `PointValuesWriter` | Accumulates `(docID, packedValue)` pairs in `PagedBytes`; flushed and nulled out |

---

## 4. The Flush Boundary

### Exact sequence in DWPT.flush()

```
DWPT.flush():
  1. segmentInfo.setMaxDoc(numDocsInRAM)
  2. Construct SegmentWriteState from fieldInfos.finish() + pendingUpdates
  3. Apply delete-by-docID (liveDocs)
  4. Call indexingChain.flush(flushState)  --> returns sortMap
  5. Construct SegmentCommitInfo
  6. sealFlushedSegment() --> compound file, .si file, liveDocs file
  7. Return FlushedSegment
```

### Exact sequence in IndexingChain.flush()

```
IndexingChain.flush(state):
  1. maybeSortSegment(state)  --> Sorter.DocMap (null if unsorted)
  2. writeNorms(state, sortMap)
  3. writeDocValues(state, sortMap)
  4. writePoints(state, sortMap)
  5. vectorValuesConsumer.flush(state, sortMap)
  6. storedFieldsConsumer.finish(maxDoc)  --> pad missing docs
     storedFieldsConsumer.flush(state, sortMap)  --> writer.finish() + close
  7. Collect fieldsToFlush map (fields with invertState != null)
  8. Open NormsProducer for read-back
  9. termsHash.flush(fieldsToFlush, state, sortMap, norms)
       --> TermVectorsConsumer.flush() first (via nextTermsHash chain)
       --> Then FreqProxTermsWriter.flush()
  10. Write FieldInfos via codec.fieldInfosFormat().write()
```

**Order matters:** FieldInfos is written LAST because consumers may modify FieldInfo during flush (e.g., `FreqProxTermsWriterPerField.finish()` calls `fieldInfo.setStorePayloads()`). Norms are written first and then read back via `NormsProducer` so they can be passed to the postings writer.

### Per-consumer flush details

#### StoredFieldsConsumer
- **Flush method:** `flush(state, sortMap)`
- **Bridge:** None -- data was already streamed to `StoredFieldsWriter` per-document via `startDocument()`/`writeField()`/`finishDocument()`
- **Push vs Pull:** PUSH -- data is written to codec writer during indexing, not at flush time
- **Flush action:** `writer.finish(maxDoc)` finalizes, then `writer.close()`
- **Note:** `finish(maxDoc)` is called BEFORE `flush()` to pad empty docs for any gaps

#### FreqProxTermsWriter (postings)
- **Flush method:** `flush(fieldsToFlush, state, sortMap, norms)` (overrides `TermsHash.flush()`)
- **Bridge:** `FreqProxFields` -- wraps `List<FreqProxTermsWriterPerField>` as a `Fields` implementation
- **Push vs Pull:** PULL -- codec's `FieldsConsumer.write(Fields, NormsProducer)` iterates over the `FreqProxFields` to pull terms, postings, positions, offsets
- **Flush action:**
  1. Calls `super.flush()` which chains to `TermVectorsConsumer.flush()` first
  2. Gathers all `FreqProxTermsWriterPerField` with `numTerms > 0`
  3. Sorts terms within each field via `perField.sortTerms()`
  4. Creates `FreqProxFields` wrapper
  5. Applies pending term deletes via `applyDeletes(state, fields)`
  6. Opens `FieldsConsumer` via `codec.postingsFormat().fieldsConsumer(state)`
  7. Calls `consumer.write(fields, norms)` -- the codec pulls data through the Fields/Terms/PostingsEnum API
  8. Closes `FieldsConsumer`

#### TermVectorsConsumer
- **Flush method:** `flush(fieldsToFlush, state, sortMap, norms)` (overrides `TermsHash.flush()`)
- **Bridge:** None at segment flush time -- TV data was already written per-document
- **Push vs Pull:** PUSH -- data is written to `TermVectorsWriter` per-document in `finishDocument()`
- **Flush action:** `writer.finish(numDocs)` + `writer.close()` (only if `writer != null`, i.e., if any doc had TVs)
- **Per-document writing:** In `TermVectorsConsumer.finishDocument()`:
  1. Sort fields by name
  2. `initTermVectorsWriter()` (lazy init of codec writer)
  3. `fill(docID)` -- write empty TV docs for any gaps
  4. `writer.startDocument(numVectorFields)`
  5. For each field: `TermVectorsConsumerPerField.finishDocument()` writes terms/positions/offsets
  6. `writer.finishDocument()`
  7. `super.reset()` -- reset TV pools for next document

#### NormValuesWriter
- **Flush method:** `flush(state, sortMap, normsConsumer)` called from `IndexingChain.writeNorms()`
- **Bridge:** Anonymous `NormsProducer` wrapping `BufferedNorms` (a `NumericDocValues` over `PackedLongValues`)
- **Push vs Pull:** PULL -- `NormsConsumer.addNormsField(fieldInfo, NormsProducer)` pulls values via `NormsProducer.getNorms()` iterator
- **Flush action:** Builds `PackedLongValues` from pending builder, wraps in iterator, passes to codec

#### PointValuesWriter
- **Flush method:** `flush(state, sortMap, pointsWriter)` called from `IndexingChain.writePoints()`
- **Bridge:** Anonymous `PointsReader` wrapping `MutablePointTree` over buffered `(docID, packedValue)` arrays
- **Push vs Pull:** PULL -- `PointsWriter.writeField(fieldInfo, PointsReader)` pulls point data via `PointsReader.getValues().getPointTree()`
- **Flush action:** Freezes `PagedBytes`, creates `MutablePointTree`, wraps in `PointsReader`, passes to codec

#### DocValuesWriter<T> variants
- **Flush method:** `flush(state, sortMap, dvConsumer)` called from `IndexingChain.writeDocValues()`
- **Bridge:** Anonymous `DocValuesProducer` wrapping typed iterators (e.g., `BufferedNumericDocValues`)
- **Push vs Pull:** PULL -- `DocValuesConsumer.addNumericField(fieldInfo, DocValuesProducer)` (or addBinaryField, addSortedField, etc.) pulls values via producer
- **Flush action:** Builds packed values, wraps in typed producer, passes to codec

#### VectorValuesConsumer
- **Flush method:** `flush(state, sortMap)`
- **Bridge:** Internal `KnnFieldVectorsWriter` instances
- **Push vs Pull:** PUSH -- delegates to `KnnVectorsWriter.flush()`

### Summary: Push vs Pull

| Consumer | Strategy | When Data Hits Disk |
|---|---|---|
| StoredFieldsConsumer | PUSH (streaming) | During indexing, per-document |
| TermVectorsConsumer | PUSH (streaming) | During indexing, per-document (in finishDocument) |
| FreqProxTermsWriter | PULL (buffered) | At segment flush, codec pulls from FreqProxFields |
| NormValuesWriter | PULL (buffered) | At segment flush, codec pulls from NormsProducer |
| PointValuesWriter | PULL (buffered) | At segment flush, codec pulls from PointsReader |
| DocValuesWriter<T> | PULL (buffered) | At segment flush, codec pulls from DocValuesProducer |
| VectorValuesConsumer | PUSH | At segment flush, delegates to KnnVectorsWriter |

---

## 5. FieldInfo as Cross-Cutting State

### How FieldInfo is built incrementally

1. **First encounter in segment:** `IndexingChain.processDocument()` first pass collects `FieldSchema` per unique field name. If `pf.fieldInfo == null`, `initializeFieldInfo()` is called.

2. **initializeFieldInfo()** creates a `FieldInfo` with all schema attributes and adds it via `fieldInfos.add()` (`FieldInfos.Builder`). The builder checks against `globalFieldNumberMap` for cross-segment consistency.

3. **Subsequent documents:** `FieldSchema.assertSameSchema(pf.fieldInfo)` verifies that the field's configuration hasn't changed. This is a per-document check.

### What each consumer contributes to FieldInfo

| Consumer / Phase | FieldInfo Property Set |
|---|---|
| Schema detection (processDocument 1st pass) | `indexOptions`, `omitNorms`, `storeTermVector`, `docValuesType`, `docValuesSkipIndex`, `pointDimensionCount`, `pointIndexDimensionCount`, `pointNumBytes`, `vectorDimension`, `vectorEncoding`, `vectorSimilarityFunction`, `attributes` |
| `initializeFieldInfo()` | Creates FieldInfo with `storePayloads=false` initially |
| `FreqProxTermsWriterPerField.finish()` | `fieldInfo.setStorePayloads()` if any payload was seen |
| `TermVectorsConsumerPerField.finishDocument()` | `fieldInfo.setStoreTermVectors()` |
| `FieldInfos.Builder.add()` | Assigns field number from `globalFieldNumberMap` |

### When FieldInfos is "frozen"

`FieldInfos` is frozen by calling `fieldInfos.finish()` which returns an immutable `FieldInfos` instance. This happens in `DWPT.flush()`:

```java
final SegmentWriteState flushState = new SegmentWriteState(
    ..., fieldInfos.finish(), ...);
```

This is called BEFORE `indexingChain.flush()`. However, consumers can still modify individual `FieldInfo` objects during flush (e.g., `setStorePayloads()`). The final `FieldInfos` write to disk happens LAST in `IndexingChain.flush()`:

```java
// Last step in IndexingChain.flush():
indexWriterConfig.getCodec().fieldInfosFormat().write(
    state.directory, state.segmentInfo, "", state.fieldInfos, IOContext.DEFAULT);
```

### How FieldInfos flows to codec writers via SegmentWriteState

`SegmentWriteState` is the container:
```java
public class SegmentWriteState {
    public final InfoStream infoStream;
    public final Directory directory;
    public final SegmentInfo segmentInfo;
    public final FieldInfos fieldInfos;     // <-- frozen FieldInfos
    public final BufferedUpdates segUpdates;
    public final String segmentSuffix;
    public final IOContext context;
    public FixedBitSet liveDocs;
    public int delCountOnFlush;
}
```

Every codec writer factory method receives `SegmentWriteState`:
- `codec.postingsFormat().fieldsConsumer(state)`
- `codec.normsFormat().normsConsumer(state)`
- `codec.docValuesFormat().fieldsConsumer(state)`
- `codec.pointsFormat().fieldsWriter(state)`

The codec uses `state.fieldInfos` to discover which fields exist and their properties, `state.directory` to create output files, and `state.segmentInfo` for segment naming.

---

## 6. Document Lifecycle

Trace of a single document from entry to storage:

### 1. Thread acquires DWPT

```
DocumentsWriter.updateDocuments():
  dwpt = flushControl.obtainAndLock()   // Thread gets exclusive DWPT
```

### 2. DWPT.updateDocuments called

```
DWPT.updateDocuments(docs, delNode, flushNotifications, onNewDocOnRAM):
  for each doc in docs:
    reserveOneDoc()                      // Atomic increment of pendingNumDocs
    indexingChain.processDocument(numDocsInRAM++, doc)
    onNewDocOnRAM.run()                  // Increments DocumentsWriter.numDocsInRAM
  finishDocuments(deleteNode, docsInRamBefore)  // Apply deletes
```

### 3. IndexingChain.processDocument called

```
IndexingChain.processDocument(docID, document):
  fieldGen = nextFieldGen++
  termsHash.startDocument()               // FreqProx: no-op. Chains to TV: resetFields()
  startStoredFields(docID)                // StoredFieldsConsumer.startDocument(docID) --> codec writer

  // PASS 1: Schema validation
  for each field in document:
    pf = getOrAddPerField(field.name())   // Lookup or create PerField
    if first time this field in this doc:
      fields[fieldCount++] = pf
      pf.reset(docID)                     // Reset FieldSchema
    docFields[docFieldIdx++] = pf
    updateDocFieldSchema(...)             // Accumulate schema from field type

  for each unique field:
    if pf.fieldInfo == null:
      initializeFieldInfo(pf)             // Create FieldInfo, create per-field consumers
    else:
      pf.schema.assertSameSchema(pf.fieldInfo)  // Verify consistency

  // PASS 2: Index each field
  for each field in document:
    processField(docID, field, pf)        // Dispatches to consumers (see Section 2)
    if field was indexed:
      indexedFieldCount++

  // FINALLY:
  for each indexed field:
    fields[i].finish(docID)               // Compute norms, call termsHashPerField.finish()
  finishStoredFields()                    // StoredFieldsConsumer.finishDocument()
  termsHash.finishDocument(docID)         // FreqProx: no-op. Chains to TV: writes TV to codec
```

### 4. Per-field consumer dispatch order

For an indexed, stored field with term vectors and doc values:

1. **Inversion:** `pf.invert(docID, field, first)` -->
   - `invertState.reset()` (if first)
   - `termsHashPerField.start(field, first)` --> `TermVectorsConsumerPerField.start()` returns `doVectors`
   - Token loop: `termsHashPerField.add(termBytes, docID)` -->
     - `FreqProxTermsWriterPerField.newTerm()/addTerm()` writes postings to byte streams
     - If doVectors: `nextPerField.add(textStart, docID)` --> `TermVectorsConsumerPerField.newTerm()/addTerm()`
2. **Stored:** `storedFieldsConsumer.writeField(fieldInfo, storedValue)` --> directly to codec writer on disk
3. **Doc values:** `docValuesWriter.addValue(docID, value)` --> buffered in `PackedLongValues.Builder`
4. **Points:** `pointValuesWriter.addPackedValue(docID, bytes)` --> buffered in `PagedBytes`

### 5. startDocument / finishDocument hooks

| Consumer | startDocument | finishDocument |
|---|---|---|
| `StoredFieldsConsumer` | `startDocument(docID)` -- calls `writer.startDocument()`, fills gaps for skipped docs | `finishDocument()` -- calls `writer.finishDocument()` |
| `FreqProxTermsWriter` | `startDocument()` -- no-op (chains to TV) | `finishDocument(docID)` -- no-op (chains to TV) |
| `TermVectorsConsumer` | `startDocument()` -- `resetFields()`, `numVectorFields = 0` | `finishDocument(docID)` -- writes all TV data to codec writer, resets pools |
| `NormValuesWriter` | (none) | `PerField.finish()` calls `norms.addValue(docID, normValue)` |

### 6. Where data lives after processing

| Data Type | Location After processDocument | Until |
|---|---|---|
| Postings (doc/freq/prox/offset) | `FreqProxTermsWriter.bytePool` + `intPool` via linked slices | Segment flush |
| Term bytes | `FreqProxTermsWriter.termBytePool` (shared) via `BytesRefHash` | Segment flush |
| Stored fields | Already on disk via `StoredFieldsWriter` | Permanent |
| Term vectors | Already on disk via `TermVectorsWriter` (written in `finishDocument`) | Permanent |
| Norms | `NormValuesWriter.pending` (`PackedLongValues.Builder`) | Segment flush |
| Doc values | Typed `DocValuesWriter.pending` builders | Segment flush |
| Points | `PointValuesWriter.bytes` (`PagedBytes`) + `docIDs[]` | Segment flush |
| KNN vectors | `KnnFieldVectorsWriter` internal buffers | Segment flush |

---

## 7. Segment Lifecycle

### State accumulated in a DWPT over many documents

| State | Location | Growth Pattern |
|---|---|---|
| Postings data | `FreqProxTermsWriter` pools | Grows with total tokens across all docs |
| Term dictionary | `BytesRefHash` per field (in shared `termBytePool`) | Grows with unique terms per field |
| Norm values | `NormValuesWriter.pending` per field | One long per doc per indexed field |
| Doc values | Typed builders per field | One value per doc per DV field |
| Point values | `PagedBytes` + `docIDs[]` per field | One packed value per point per doc |
| KNN vectors | Per-field writer buffers | One vector per doc per vector field |
| Stored fields | Disk (via streaming `StoredFieldsWriter`) | Grows on disk, RAM is codec buffer only |
| Term vectors | Disk (via streaming `TermVectorsWriter`) | Grows on disk, RAM is codec buffer only |
| Delete tracking | `BufferedUpdates`, `deleteDocIDs[]` | Grows with deletes applied to this segment |
| Field metadata | `FieldInfos.Builder`, `PerField[]` hash | Grows with unique field count |

### What triggers flush

RAM accounting chain:
1. `DWPT.ramBytesUsed()` = `deleteDocIDs` memory + `pendingUpdates.ramBytesUsed()` + `indexingChain.ramBytesUsed()`
2. `IndexingChain.ramBytesUsed()` = `bytesUsed.get()` (pools + packed builders) + `storedFieldsConsumer.accountable.ramBytesUsed()` + `termVectorsWriter.accountable.ramBytesUsed()` + `vectorValuesConsumer.getAccountable().ramBytesUsed()`
3. After each document, `DocumentsWriterFlushControl.doAfterDocument(dwpt)` is called:
   - Computes `delta = dwpt.ramBytesUsed() - dwpt.lastCommittedBytesUsed`
   - If delta exceeds `ramBufferGranularity()`, commits bytes and calls `flushPolicy.onChange()`
   - `FlushPolicy` checks if total active bytes exceed `ramBufferSizeMB`
   - If so, `setFlushPending(perThread)` marks the DWPT for flush
   - Hard limit: if a single DWPT exceeds `ramPerThreadHardLimitMB`, it's force-flushed

### Transition from DWPT to flushed segment

1. `DocumentsWriter.doFlush(flushingDWPT)`:
   - `dwpt.prepareFlush()` -- freezes global deletes, applies pending deletes
   - `dwpt.flush(flushNotifications)` -- calls `indexingChain.flush()`, returns `FlushedSegment`
   - `FlushedSegment` contains: `SegmentCommitInfo`, `FieldInfos`, frozen deletes, liveDocs, sortMap
2. `sealFlushedSegment()`:
   - Optionally creates compound file (`.cfs`/`.cfe`)
   - Writes `.si` file via `codec.segmentInfoFormat().write()`
   - Writes liveDocs if any deletions
3. Ticket queue orders flushed segments for publishing

### Cleanup after flush

| What | Cleanup |
|---|---|
| DWPT | Unlocked and removed from pool (never reused after flush) |
| IndexingChain pools | Implicitly dropped -- DWPT is discarded, all pools become garbage |
| Stored fields writer | Closed in `StoredFieldsConsumer.flush()` |
| Term vectors writer | Closed in `TermVectorsConsumer.flush()` |
| Per-field writers | Nulled out during `IndexingChain.flush()` (e.g., `perField.pointValuesWriter = null`) |
| Codec writers | Created and closed within `IndexingChain.flush()` scope (norms, DV, points, postings) |
| File tracking | `directory.getCreatedFiles()` captured in `segmentInfo.setFiles()` |
| RAM accounting | `DocumentsWriterFlushControl` subtracts flushed bytes from `activeBytes` |

**Key point:** A DWPT is never reused after flush. The entire DWPT and its IndexingChain are discarded. A new DWPT will be created by the pool for subsequent indexing on that thread.
