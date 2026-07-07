# Lucene 10.3.2 Indexing Pipeline Overview

Reference: `lucene/core/src/java/org/apache/lucene/index/`

---

## 1. Document Ingestion Flow

### Entry Point: IndexWriter.addDocument()

`IndexWriter.addDocument()` (line 1497) is a thin wrapper:

```
addDocument(doc) -> updateDocument(null, doc) -> updateDocuments(null, List.of(doc))
```

The private `updateDocuments()` (line 1566) calls:

```java
final long seqNo = maybeProcessEvents(docWriter.updateDocuments(docs, delNode));
```

### DocumentsWriter.updateDocuments() (line 409)

This is the central coordination point. The method:

1. **preUpdate()** (line 383): Checks for stalled threads. If any DWPT flushes are queued or threads are stalled, helps flush pending DWPTs via `maybeFlush()` and blocks via `flushControl.waitIfStalled()`.

2. **Obtain a DWPT** (line 415): `flushControl.obtainAndLock()` acquires a locked DWPT from the pool. This is the synchronization boundary -- once a thread has a DWPT, it proceeds without contention.

3. **Index the documents** (line 424-425): Calls `dwpt.updateDocuments(docs, delNode, flushNotifications, numDocsInRAM::incrementAndGet)` on the obtained DWPT without holding any global lock.

4. **Check flush status** (line 431): `flushControl.doAfterDocument(dwpt)` updates RAM accounting and consults the FlushPolicy. May return a DWPT that needs flushing.

5. **Return or recycle the DWPT** (lines 437-441): If the DWPT is not flush-pending, not aborted, and has not had its delete queue advanced, it is returned to the free pool via `perThreadPool.marksAsFreeAndUnlock(dwpt)`. Otherwise, it is simply unlocked (and will be checked out for flush).

6. **postUpdate()** (line 447): Applies any pending deletes, and if a flushing DWPT was returned from step 4, calls `doFlush(flushingDWPT)`.

### DocumentsWriterPerThread.updateDocuments() (line 226)

For each document in the batch:

1. **reserveOneDoc()** (line 272): Atomically increments `pendingNumDocs` and checks against `MAX_DOCS`.

2. **indexingChain.processDocument(numDocsInRAM++, doc)** (line 274): The actual indexing work. `numDocsInRAM` is the DWPT-local doc ID counter.

3. **onNewDocOnRAM.run()** (line 276): Increments the global `numDocsInRAM` counter on DocumentsWriter.

After all docs, calls `finishDocuments(deleteNode, docsInRamBefore)` (line 284) which updates the delete slice.

### IndexingChain.processDocument() (line 561)

This is where individual field processing happens. The method uses a **two-pass approach**:

**Pass 1 -- Schema validation** (lines 578-612):
- Iterates over all fields in the document
- For each field, calls `getOrAddPerField(field.name(), false)` to obtain or create a `PerField` entry
- Tracks unique fields with a generation counter (`fieldGen`). First time a field name is seen in this doc, calls `pf.reset(docID)` and adds it to the `fields[]` array.
- Every field instance goes into `docFields[]` (allows multi-valued fields).
- Calls `updateDocFieldSchema()` to accumulate the field's schema (index options, doc values type, point dims, vector dims).
- After the loop, for new fields calls `initializeFieldInfo(pf)` which creates the `FieldInfo`, `DocValuesWriter`, `PointValuesWriter`, etc. For existing fields, calls `pf.schema.assertSameSchema(pf.fieldInfo)`.

**Pass 2 -- Field indexing** (lines 617-623):
- Iterates again over all fields
- Calls `processField(docID, field, docFields[docFieldIdx])` for each
- Tracks which fields were indexed with postings (for the finish step)

**Finish** (lines 626-641):
- For each field that had postings: `fields[i].finish(docID)` computes norms and calls `termsHashPerField.finish()`.
- `finishStoredFields()` calls `storedFieldsConsumer.finishDocument()`.
- `termsHash.finishDocument(docID)` which chains to `TermVectorsConsumer.finishDocument()`.

### IndexingChain.processField() (line 734)

Handles four types of field data:

1. **Inverted fields** (lines 739-747): If `indexOptions != NONE`, calls `pf.invert(docID, field, first)`. This obtains a TokenStream from the analyzer, then for each token calls `termsHashPerField.add(termBytes, docID)`.

2. **Stored fields** (lines 750-769): If `fieldType.stored()`, calls `storedFieldsConsumer.writeField(pf.fieldInfo, storedValue)` which writes directly to the codec's `StoredFieldsWriter`.

3. **Doc values** (lines 771-774): If `docValuesType != NONE`, calls `indexDocValue(docID, pf, dvType, field)` which dispatches to the appropriate `DocValuesWriter` (Numeric, Binary, Sorted, SortedNumeric, SortedSet).

4. **Points** (lines 775-777): If `pointDimensionCount != 0`, calls `pf.pointValuesWriter.addPackedValue(docID, field.binaryValue())`.

5. **Vectors** (lines 778-780): If `vectorDimension != 0`, calls `indexVectorValue()`.

### PerField.invert() (line 1183)

Dispatches based on `field.invertableType()`:

- **TOKEN_STREAM** (line 1196): `invertTokenStream()` -- gets a TokenStream from the analyzer, iterates tokens, validates positions/offsets, calls `termsHashPerField.add(termBytes, docID)` for each token (line 1300).

- **BINARY** (line 1193): `invertTerm()` -- for keyword/string fields that produce a single binary term. Calls `termsHashPerField.add(binaryValue, docID)` (line 1373).

### TermsHashPerField.add() -- Primary Entry (line 190)

```java
int termID = bytesHash.add(termBytes);
if (termID >= 0) {  // New term
    initStreamSlices(termID, docID);
} else {  // Existing term
    termID = positionStreamSlice(termID, docID);
}
if (doNextCall) {
    nextPerField.add(postingsArray.textStarts[termID], docID);
}
```

- `bytesHash.add()` interns the term text and returns a term ID (positive = new, negative = existing).
- `initStreamSlices()` allocates int and byte pool slices for the new term's postings streams, then calls `newTerm(termID, docID)`.
- `positionStreamSlice()` positions the write cursors to the existing term's stream, then calls `addTerm(termID, docID)`.
- The chain continues via `nextPerField.add()` for term vectors (using `textStarts[termID]` as the key, so term text is shared).

---

## 2. DWPT Lifecycle

### Creation

DWPTs are created on demand by `DocumentsWriterPerThreadPool.newWriter()` (line 85), which calls a factory lambda set up in the `DocumentsWriter` constructor (line 122-134):

```java
new DocumentsWriterPerThread(
    indexCreatedVersionMajor,
    segmentNameSupplier.get(),   // e.g., "_0", "_1", ...
    directoryOrig,
    directory,
    config,
    deleteQueue,
    new FieldInfos.Builder(globalFieldNumberMap),
    pendingNumDocs,
    enableTestPoints);
```

Each DWPT creates its own:
- `SegmentInfo` with a unique segment name and random ID (DWPT constructor, line 167-180)
- `TrackingDirectoryWrapper` to track created files (line 156)
- `IndexingChain` (line 192-198)
- `BufferedUpdates` (line 162)
- `DeleteSlice` from the shared delete queue (line 165)

### Assignment

`DocumentsWriterPerThreadPool.getAndLock()` (line 115):
1. First tries `freeList.lockAndPoll()` to reuse an existing free DWPT
2. If none available, calls `newWriter()` to create a fresh one

The free list is a `LockableConcurrentApproximatePriorityQueue` that prioritizes DWPTs by RAM usage.

### Work Processing

The calling thread holds the DWPT's lock throughout `updateDocuments()`. The DWPT is single-threaded during indexing -- no other thread touches it.

### Return to Pool

After `updateDocuments()` completes (DocumentsWriter line 437-441):
- If not flush-pending, not aborted, and delete queue not advanced: `perThreadPool.marksAsFreeAndUnlock(dwpt)` returns it to the free list with its current RAM usage as priority.
- Otherwise: `dwpt.unlock()` releases the lock. The DWPT will be checked out for flush and never reused.

### Flush

When a DWPT is marked flush-pending:
1. It's checked out of the pool via `checkOutForFlush()` in FlushControl
2. A flush ticket is created via `ticketQueue.addTicket()` (DocumentsWriter line 496-497)
3. `dwpt.prepareFlush()` freezes the global buffer and applies pending deletes (DWPT line 391-403)
4. `dwpt.flush()` writes the segment to disk (DWPT line 406)
5. After flush, `flushControl.doAfterFlush()` subtracts the DWPT's bytes from `flushBytes`

### State Held Per DWPT

| Field | Type | Purpose |
|---|---|---|
| `segmentInfo` | `SegmentInfo` | Segment metadata (name, codec, sort, maxDoc) |
| `directory` | `TrackingDirectoryWrapper` | Tracks files created during flush |
| `indexingChain` | `IndexingChain` | All in-memory indexing state |
| `pendingUpdates` | `BufferedUpdates` | Buffered deletes for this segment |
| `deleteSlice` | `DeleteSlice` | Tracks position in the global delete queue |
| `numDocsInRAM` | `int` | Document count in this DWPT |
| `deleteDocIDs` | `int[]` | Doc IDs to mark deleted (from indexing exceptions) |
| `codec` | `Codec` | The codec used for writing |
| `fieldInfos` | `FieldInfos.Builder` | Per-segment field metadata |
| `lastCommittedBytesUsed` | `long` | Last RAM usage committed to flush control |

---

## 3. IndexingChain Consumer Hierarchy

### Construction (IndexingChain constructor, line 100-143)

```
IndexingChain
  |
  +-- storedFieldsConsumer: StoredFieldsConsumer
  |     Wraps codec's StoredFieldsWriter
  |     Writes directly to disk per document
  |
  +-- termsHash: FreqProxTermsWriter (extends TermsHash)
  |     |
  |     +-- owns: intPool (IntBlockPool)
  |     +-- owns: bytePool (ByteBlockPool)
  |     +-- shares: termBytePool = bytePool (primary TermsHash)
  |     |
  |     +-- nextTermsHash: TermVectorsConsumer (extends TermsHash)
  |           |
  |           +-- owns: intPool (IntBlockPool)  -- separate from FreqProx
  |           +-- owns: bytePool (ByteBlockPool) -- separate from FreqProx
  |           +-- shares: termBytePool = FreqProx's bytePool (set by parent)
  |           +-- owns: writer (TermVectorsWriter) -- lazy-init on first vector doc
  |
  +-- termVectorsWriter: TermVectorsConsumer (same as termsHash.nextTermsHash)
  |
  +-- docValuesBytePool: ByteBlockPool
  |     Shared pool for SORTED and SORTED_SET doc value terms
  |
  +-- vectorValuesConsumer: VectorValuesConsumer
  |     Manages KNN vector writers
  |
  +-- fieldHash: PerField[]  (hash table of per-field state)
```

### Per-Field Consumers (created in initializeFieldInfo, line 652)

Each `PerField` holds:

| Field | Created When | Purpose |
|---|---|---|
| `termsHashPerField` | `setInvertState()` (line 1138) | Postings accumulation (FreqProxTermsWriterPerField) |
| `norms` | `setInvertState()` if `!omitsNorms` (line 1147) | NormValuesWriter for this field |
| `docValuesWriter` | `initializeFieldInfo()` (lines 702-718) | One of: Numeric/Binary/Sorted/SortedNumeric/SortedSet DocValuesWriter |
| `pointValuesWriter` | `initializeFieldInfo()` (line 721) | PointValuesWriter for BKD-indexed fields |
| `knnFieldVectorsWriter` | `initializeFieldInfo()` (line 724) | KNN vector writer via codec |

The `termsHash.addField()` call (line 1142) creates the per-field chain:
```
FreqProxTermsWriterPerField
  +-- nextPerField: TermVectorsConsumerPerField
```

### TermsHash Chain

The TermsHash uses a linked-list chain pattern (TermsHash line 35-56):

```
FreqProxTermsWriter (primary)
  nextTermsHash -> TermVectorsConsumer (secondary)
```

The primary TermsHash owns the `termBytePool` (the ByteBlockPool that stores actual term bytes). The secondary TermsHash shares this same pool:
```java
// TermsHash constructor, line 52-56
if (nextTermsHash != null) {
    termBytePool = bytePool;
    nextTermsHash.termBytePool = bytePool;
}
```

This means term text is stored once (in FreqProx's bytePool) and term vectors reference it by offset.

### Document Lifecycle Through Consumers

Per document, the call sequence through IndexingChain.processDocument():

```
1. termsHash.startDocument()
     -> TermVectorsConsumer.startDocument()  (via nextTermsHash)

2. storedFieldsConsumer.startDocument(docID)
     -> StoredFieldsWriter.startDocument()  [writes to disk]

3. For each field:
     processField(docID, field, pf)
       -> pf.invert()                        [postings + term vectors to memory pools]
       -> storedFieldsConsumer.writeField()   [stored fields to disk]
       -> indexDocValue()                     [doc values to memory]
       -> pointValuesWriter.addPackedValue()  [points to memory]
       -> indexVectorValue()                  [vectors to memory]

4. For each indexed field:
     pf.finish(docID)
       -> similarity.computeNorm()  -> norms.addValue()
       -> termsHashPerField.finish()

5. storedFieldsConsumer.finishDocument()
     -> StoredFieldsWriter.finishDocument()  [writes to disk]

6. termsHash.finishDocument(docID)
     -> TermVectorsConsumer.finishDocument(docID)
          -> initTermVectorsWriter()          [lazy-init writer on first vector doc]
          -> fill(docID)                      [write empty entries for skipped docs]
          -> writer.startDocument(numVectorFields)
          -> for each vector field: perField.finishDocument()
          -> writer.finishDocument()          [writes to disk]
          -> reset() + resetFields()          [clear per-doc TV state]
```

---

## 4. Flush Triggering

### FlushPolicy (FlushByRamOrCountsPolicy, line 34)

After every document, `DocumentsWriterFlushControl.doAfterDocument()` (line 205) is called. It:

1. Calculates the RAM delta since the last committed measurement
2. Skips accounting if delta is below `ramBufferGranularity()` (optimization for small docs, line 209-213)
3. Under lock, commits the delta to `flushBytes` or `activeBytes`
4. Calls `flushPolicy.onChange(this, perThread)` (line 236)

`FlushByRamOrCountsPolicy.onChange()` (line 34):

- **Doc count trigger**: If `perThread.getNumDocsInRAM() >= maxBufferedDocs`, marks the DWPT flush-pending.
- **RAM trigger**: If `activeRam + deletesRam >= limit` (where limit = `RAMBufferSizeMB * 1024 * 1024`), finds the largest non-pending DWPT and marks it flush-pending via `control.setFlushPending()`.
- **Hard per-DWPT limit**: After FlushPolicy runs, if `perThread.ramBytesUsed() > hardMaxBytesPerDWPT` (line 237-240), the DWPT is force-flushed to prevent address space exhaustion.

### Stall Control

`updateStallState()` (FlushControl line 305):

```java
final boolean stall = (activeBytes + flushBytes) > limit && activeBytes < limit && !closed;
```

Where `limit = 2 * RAMBufferSizeMB`. Threads are stalled (blocked in `waitIfStalled()`) when total memory exceeds 2x the RAM buffer but active memory alone is below the limit (meaning flushes are in progress but can't keep up).

### setFlushPending() (FlushControl line 363)

```java
perThread.setFlushPending();
flushBytes += bytes;
activeBytes -= bytes;
numPending++;
```

Moves the DWPT's memory accounting from `activeBytes` to `flushBytes`.

### Flush Sequence (DocumentsWriter.doFlush, line 462)

```
1. ticketQueue.addTicket(() -> FlushTicket(dwpt.prepareFlush(), true))
     - Freezes global deletes
     - Assigns a ticket to maintain flush ordering

2. dwpt.flush(flushNotifications)
     - Runs concurrently, no global lock held
     - Returns FlushedSegment

3. ticketQueue.addSegment(ticket, newSegment)
     - Attaches the flushed segment to its ticket

4. flushControl.doAfterFlush(flushingDWPT)
     - Subtracts bytes from flushBytes
     - Updates stall state

5. Loop: while nextPendingFlush() != null, flush that too

6. flushNotifications.afterSegmentsFlushed()
     - Triggers publishFlushedSegments on IndexWriter
```

### Full Flush (DocumentsWriter.flushAllThreads, line 628)

Triggered by `IndexWriter.commit()` or NRT reader open:

1. Cuts over to a new delete queue (line 641): `flushControl.markForFullFlush()`
2. All DWPTs associated with the old delete queue are marked flush-pending
3. Calls `maybeFlush()` to start processing them
4. `flushControl.waitForFlush()` blocks until all flushing writers complete

---

## 5. Segment Creation

### DWPT.flush() (line 406)

1. **Create SegmentWriteState** (line 411-418):
   ```java
   new SegmentWriteState(
       infoStream, directory, segmentInfo,
       fieldInfos.finish(),     // finalize FieldInfos
       pendingUpdates,
       IOContext.flush(new FlushInfo(numDocsInRAM, lastCommittedBytesUsed)));
   ```

2. **Apply delete-by-docID** (lines 424-432): Creates a `FixedBitSet` liveDocs if any docs were marked deleted during indexing (from exceptions).

3. **indexingChain.flush(flushState)** (line 456): The main flush -- writes all segment files.

4. **Record segment files** (line 468): `segmentInfo.setFiles(directory.getCreatedFiles())`

5. **Create SegmentCommitInfo** (line 470-478)

6. **sealFlushedSegment()** (line 538): Writes compound file (if enabled) and segment info.

### IndexingChain.flush() (line 270)

Writes segment files in this order:

| Step | Method | Files Written | Line |
|---|---|---|---|
| 1 | `writeNorms(state, sortMap)` | `.nvd`, `.nvm` (norms data + metadata) | 277 |
| 2 | `writeDocValues(state, sortMap)` | `.dvd`, `.dvm` (doc values data + metadata) | 291 |
| 3 | `writePoints(state, sortMap)` | `.kdi`, `.kdd`, `.kdm` (BKD tree) | 298 |
| 4 | `vectorValuesConsumer.flush()` | `.vec`, `.vemf`, `.veq`, `.vex` (knn vectors) | 305 |
| 5 | `storedFieldsConsumer.finish() + flush()` | `.fdt`, `.fdm` (stored fields) | 313-314 |
| 6 | `termsHash.flush(...)` | Postings + term vectors (see below) | 321-343 |
| 7 | `fieldInfosFormat().write()` | `.fnm` (field infos) | 356-359 |

**Step 6 detail -- termsHash.flush():**

```
FreqProxTermsWriter.flush()
  |
  +-- super.flush()  ->  TermVectorsConsumer.flush()
  |     Calls writer.finish(numDocs) + close writer
  |     Writes: .tvd, .tvx (term vectors data + index)
  |
  +-- For each field with postings:
  |     perField.sortTerms()   -- sort terms by bytesHash
  |
  +-- applyDeletes(state, fields)  -- resolve term deletes against in-memory postings
  |
  +-- codec.postingsFormat().fieldsConsumer(state)
  |     consumer.write(fields, norms)
  |     Writes: .tim, .tip, .doc, .pos, .pay (terms dict, postings, positions, payloads)
  |
  +-- Close FieldsConsumer
```

### sealFlushedSegment() (DWPT line 591)

After `indexingChain.flush()` returns:

1. **Compound file** (lines 607-618): If `useCompoundFile` is true, creates `.cfs`/`.cfe` from all segment files, then deletes originals.

2. **Segment info** (line 624): `codec.segmentInfoFormat().write()` writes `.si` file. Must happen AFTER compound file creation so `.si` isn't included in CFS and reflects `useCompoundFile=true`.

3. **Live docs** (lines 632-663): If there are deleted docs, writes live docs via `codec.liveDocsFormat().writeLiveDocs()`.

### SegmentWriteState (line 31)

State object passed to all codec writers:

| Field | Type | Purpose |
|---|---|---|
| `infoStream` | `InfoStream` | Debug logging |
| `directory` | `Directory` | Where to write files |
| `segmentInfo` | `SegmentInfo` | Segment metadata (name, maxDoc, codec) |
| `fieldInfos` | `FieldInfos` | Finalized field metadata for the segment |
| `segUpdates` | `BufferedUpdates` | Pending deletes/updates |
| `liveDocs` | `FixedBitSet` | Non-null if any docs deleted |
| `delCountOnFlush` | `int` | Number of deleted docs |
| `softDelCountOnFlush` | `int` | Number of soft-deleted docs |
| `segmentSuffix` | `String` | Suffix for per-field postings format files |
| `context` | `IOContext` | I/O context (flush info) |

---

## 6. Data Structures in Memory

### ByteBlockPool

**File**: `util/ByteBlockPool.java`

- Block size: 32KB (`BYTE_BLOCK_SIZE = 1 << 15 = 32768`)
- Array of `byte[]` buffers, allocated on demand
- Used for: term bytes storage, postings byte streams, doc values terms

Three instances per IndexingChain:
1. **FreqProxTermsWriter.bytePool**: Postings data (freq/prox byte streams) + term bytes (shared as `termBytePool`)
2. **TermVectorsConsumer.bytePool**: Term vector byte streams (NOT term bytes -- those come from FreqProx's pool via `termBytePool`)
3. **IndexingChain.docValuesBytePool**: Shared pool for SORTED and SORTED_SET doc value terms

### IntBlockPool

**File**: `util/IntBlockPool.java`

- Block size: 8192 (`INT_BLOCK_SIZE = 1 << 13`)
- Array of `int[]` buffers
- Used for: stream address pointers (per-term write cursors into ByteBlockPool)

Two instances per IndexingChain:
1. **FreqProxTermsWriter.intPool**: Stream addresses for postings
2. **TermVectorsConsumer.intPool**: Stream addresses for term vectors

### ByteSlicePool

Wraps a `ByteBlockPool` and manages allocation of variable-length slices within it. Slices grow through multiple levels (5 bytes, 14 bytes, 20 bytes, ..., up to full block size). Each slice ends with a forwarding pointer to the next slice.

### BytesRefHash

**File**: `util/BytesRefHash.java`

Per-field term deduplication. One per `TermsHashPerField`. Stores term bytes in the shared `termBytePool` (FreqProx's ByteBlockPool) and maintains a hash table mapping term -> termID.

### ParallelPostingsArray

Per-field array indexed by termID. Subclasses add field-specific data:

- `textStarts[termID]`: Offset into termBytePool where term text starts
- `addressOffset[termID]`: Offset into IntBlockPool for this term's stream addresses
- `byteStarts[termID]`: Offset into ByteBlockPool where this term's first stream starts

FreqProxTermsWriterPerField adds:
- `termFreqs[termID]`: Current doc term frequency
- `lastDocIDs[termID]`: Last doc ID for this term
- `lastDocCodes[termID]`: Encoded doc delta (deferred write)
- `lastPositions[termID]`: Last position for this term

### DocValuesWriters (in-memory)

Each per-field DocValuesWriter buffers all values in memory until flush:

- `NumericDocValuesWriter`: Packed long array + `DocsWithFieldSet`
- `BinaryDocValuesWriter`: `PagedBytes` + lengths array + `DocsWithFieldSet`
- `SortedDocValuesWriter`: `BytesRefHash` (backed by `docValuesBytePool`) + `PackedLongValues` for ord mapping
- `SortedNumericDocValuesWriter`: `PackedLongValues.Builder` for values + `PackedLongValues.Builder` for doc-to-address mapping
- `SortedSetDocValuesWriter`: `BytesRefHash` (backed by `docValuesBytePool`) + `PackedLongValues` for ords + address mapping

### PointValuesWriter

Buffers all point values in memory as a flat `byte[]` array plus `int[] docIDs`. Flushed to BKD writer on segment flush.

### NormValuesWriter

Stores one `long` norm value per document in a packed long array. Flushed to norms codec writer.

### What Writes to Disk During Indexing (Before Flush)

Only two consumers write to disk incrementally during document processing:

1. **StoredFieldsConsumer**: Calls `StoredFieldsWriter.startDocument()` / `writeField()` / `finishDocument()` for every document. Data is buffered internally by the codec writer and compressed in blocks (e.g., Lucene90 uses 16KB chunks with LZ4 or DEFLATE).

2. **TermVectorsConsumer**: Calls `TermVectorsWriter.startDocument()` / `startField()` / `startTerm()` etc. per document that has term vectors. Lazy-initialized on first doc with vectors.

Everything else (postings, doc values, points, norms) is buffered entirely in memory and only written during `flush()`.

### Memory Freed During Flush

After `indexingChain.flush()`:
- All byte/int block pools are dropped (no longer referenced)
- BytesRefHash tables are cleared
- ParallelPostingsArrays are freed
- DocValuesWriters are nulled out per-field
- PointValuesWriters are nulled out per-field
- The entire IndexingChain becomes garbage-collectable when the DWPT is discarded

The DWPT itself is never reused after flush -- it is checked out of the pool and discarded.

---

## 7. Key Object Ownership

### Full Object Graph

```
IndexWriter
  |
  +-- docWriter: DocumentsWriter
  |     |
  |     +-- deleteQueue: DocumentsWriterDeleteQueue  (shared, swapped on full flush)
  |     +-- ticketQueue: DocumentsWriterFlushQueue    (ordered flush results)
  |     +-- perThreadPool: DocumentsWriterPerThreadPool
  |     |     |
  |     |     +-- dwpts: Set<DocumentsWriterPerThread>  (all live DWPTs)
  |     |     +-- freeList: LockableConcurrentApproximatePriorityQueue<DWPT>
  |     |     +-- dwptFactory: Supplier<DWPT>  (creates new DWPTs)
  |     |
  |     +-- flushControl: DocumentsWriterFlushControl
  |           |
  |           +-- perThreadPool (same ref as above)
  |           +-- flushPolicy: FlushByRamOrCountsPolicy
  |           +-- flushQueue: Queue<DWPT>            (DWPTs ready to flush)
  |           +-- blockedFlushes: Queue<DWPT>         (blocked during full flush)
  |           +-- flushingWriters: List<DWPT>         (currently flushing)
  |           +-- stallControl: DocumentsWriterStallControl
  |
  +-- globalFieldNumberMap: FieldNumbers             (global field name -> number mapping)
  +-- segmentInfos: SegmentInfos                     (committed segment list)
  +-- deleter: IndexFileDeleter
  +-- readerPool: ReaderPool
  +-- bufferedUpdatesStream: BufferedUpdatesStream
```

### Per-DWPT Object Graph

```
DocumentsWriterPerThread
  |
  +-- segmentInfo: SegmentInfo                 (owned, one per DWPT)
  +-- directory: TrackingDirectoryWrapper       (wraps shared Directory)
  +-- fieldInfos: FieldInfos.Builder            (owned, accumulates field metadata)
  +-- pendingUpdates: BufferedUpdates            (owned, segment-local deletes)
  +-- deleteSlice: DeleteSlice                  (view into shared deleteQueue)
  +-- deleteQueue: DocumentsWriterDeleteQueue   (shared reference)
  |
  +-- indexingChain: IndexingChain
        |
        +-- bytesUsed: Counter                  (tracks RAM for postings pools)
        +-- byteBlockAllocator: DirectTrackingAllocator  (allocates & tracks byte blocks)
        +-- fieldInfos (same ref as DWPT's)
        |
        +-- storedFieldsConsumer: StoredFieldsConsumer
        |     +-- writer: StoredFieldsWriter    (lazy-init, codec-owned)
        |
        +-- termsHash: FreqProxTermsWriter
        |     +-- intPool: IntBlockPool         (owned)
        |     +-- bytePool: ByteBlockPool       (owned, also used as termBytePool)
        |     +-- bytesUsed: Counter            (shared with IndexingChain)
        |     |
        |     +-- nextTermsHash: TermVectorsConsumer ----+
        |           +-- intPool: IntBlockPool   (owned)  |
        |           +-- bytePool: ByteBlockPool (owned)  |
        |           +-- termBytePool: ByteBlockPool      |  = FreqProx's bytePool
        |           +-- writer: TermVectorsWriter (lazy)  |
        |                                                |
        +-- termVectorsWriter -------(same object)-------+
        |
        +-- docValuesBytePool: ByteBlockPool    (owned, shared by Sorted/SortedSet DV)
        |
        +-- vectorValuesConsumer: VectorValuesConsumer
        |     +-- per-field KnnFieldVectorsWriter instances
        |
        +-- fieldHash: PerField[]               (hash table)
              |
              +-- PerField[0]
              |     +-- fieldInfo: FieldInfo
              |     +-- invertState: FieldInvertState
              |     +-- termsHashPerField: FreqProxTermsWriterPerField
              |     |     +-- nextPerField: TermVectorsConsumerPerField
              |     |     +-- bytesHash: BytesRefHash (uses termsHash.termBytePool)
              |     |     +-- postingsArray: FreqProxPostingsArray
              |     +-- norms: NormValuesWriter
              |     +-- docValuesWriter: NumericDocValuesWriter (etc.)
              |     +-- pointValuesWriter: PointValuesWriter
              |     +-- knnFieldVectorsWriter: KnnFieldVectorsWriter
              |
              +-- PerField[1] ...
```

### Key Shared References

1. **termBytePool**: FreqProxTermsWriter's `bytePool` is shared as `termBytePool` with TermVectorsConsumer. Term bytes are stored once and referenced by offset from both postings and term vectors.

2. **docValuesBytePool**: Shared across all SORTED and SORTED_SET DocValuesWriters within the same DWPT. Each field's `SortedDocValuesWriter` / `SortedSetDocValuesWriter` receives this pool at construction.

3. **bytesUsed Counter**: Shared between IndexingChain and FreqProxTermsWriter. The `DirectTrackingAllocator` increments this counter when allocating byte blocks. Used by FlushControl for RAM accounting.

4. **deleteQueue**: Shared across all DWPTs created in the same "generation". When a full flush occurs, the queue is advanced and new DWPTs get the new queue.

5. **globalFieldNumberMap**: Shared across the entire IndexWriter. Ensures consistent field numbering across segments. Each DWPT's `FieldInfos.Builder` references this for validation and number assignment.

### Ownership Rules

- Each DWPT exclusively owns its IndexingChain and all memory pools within it
- DWPTs are single-writer: only one thread touches a DWPT at a time (enforced by locking)
- Codec writers (StoredFieldsWriter, TermVectorsWriter) are owned per-DWPT, created lazily
- After flush, the DWPT and all its owned objects become garbage -- nothing is recycled
- The DocumentsWriterPerThreadPool tracks all live DWPTs and manages the free list
- FlushControl tracks memory across all DWPTs via `activeBytes` and `flushBytes`
