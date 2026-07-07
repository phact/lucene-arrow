# Term Vectors Indexing Path (Lucene 10.3.2)

## 1. Files Produced

The Lucene90 term vectors format produces three files per segment:

| Extension | File | Purpose |
|-----------|------|---------|
| `.tvd` | Vector data | Compressed chunks containing terms, frequencies, positions, offsets, and payloads |
| `.tvx` | Vector index | Chunk index mapping doc IDs to file pointers in `.tvd`, via `FieldsIndexWriter` |
| `.tvm` | Vector metadata | PackedInts version, chunk size, chunk index metadata, chunk/dirty counts, footer |

These names are defined as constants in `Lucene90CompressingTermVectorsWriter`:
- `VECTORS_EXTENSION = "tvd"`
- `VECTORS_INDEX_EXTENSION = "tvx"`
- `VECTORS_META_EXTENSION = "tvm"`

## 2. Term Vector Collection During Inversion

### Class Hierarchy

```
TermsHash (abstract)
  └── TermVectorsConsumer (concrete)
        └── SortingTermVectorsConsumer (for index sorting)

TermsHashPerField (abstract)
  └── TermVectorsConsumerPerField (concrete)
```

`TermVectorsConsumer` is the secondary `TermsHash` in the chain. The primary is `FreqProxTermsWriter`. In the `TermsHash` constructor, when `nextTermsHash != null`, the primary shares its `termBytePool` with the secondary:

```java
// TermsHash constructor (lines 52-56)
if (nextTermsHash != null) {
    termBytePool = bytePool;
    nextTermsHash.termBytePool = bytePool;
}
```

This means term text bytes are stored once (in the primary's `bytePool`) and shared. The TV consumer references term text by pool offset via `addByPoolOffset()` rather than re-interning.

### Per-Token Flow

1. **Primary chain** (`FreqProxTermsWriter`): `TermsHashPerField.add(BytesRef, docID)` interns the term into `BytesRefHash`, calls `initStreamSlices()` or `positionStreamSlice()`, then if `doNextCall` is true, calls:

2. **Secondary chain** (`TermVectorsConsumerPerField`): `TermsHashPerField.add(int textStart, docID)` -- the secondary entry point. Uses `bytesHash.addByPoolOffset(textStart)` to deduplicate terms within this document using the text already stored in the primary's byte pool.

3. On new term: `TermVectorsConsumerPerField.newTerm(termID, docID)` initializes `freqs[termID]`, resets `lastOffsets` and `lastPositions` to 0, then calls `writeProx()`.

4. On existing term: `TermVectorsConsumerPerField.addTerm(termID, docID)` increments `freqs[termID]` and calls `writeProx()`.

### Stream Count

`TermVectorsConsumerPerField` passes `streamCount=2` to its `TermsHashPerField` super constructor:
- **Stream 0**: Position data (delta-encoded positions with optional payload length + payload bytes)
- **Stream 1**: Offset data (delta-encoded start offsets, offset lengths)

## 3. Shared vs Per-Field Pools

### Pool Architecture

`TermVectorsConsumer` (extending `TermsHash`) owns three shared pools:

| Pool | Type | Purpose |
|------|------|---------|
| `intPool` | `IntBlockPool` | Stores per-term stream address pointers (current write position in `bytePool`) |
| `bytePool` | `ByteBlockPool` | Stores the actual variable-length byte slices for position/offset streams |
| `termBytePool` | `ByteBlockPool` | **Shared from primary TermsHash** -- stores term text bytes |

All `TermVectorsConsumerPerField` instances for a document share the same `intPool` and `bytePool` from their parent `TermVectorsConsumer`. This is established in the constructor:

```java
// TermVectorsConsumerPerField constructor (lines 48-62)
super(2, termsHash.intPool, termsHash.bytePool, termsHash.termBytePool,
      termsHash.bytesUsed, null, fieldInfo.name, fieldInfo.getIndexOptions());
```

Each per-field also creates a `BytesRefBlockPool` wrapping the shared `termBytePool`, used to read back term bytes during flush:

```java
termBytePool = new BytesRefBlockPool(termsHash.termBytePool);
```

### Per-Field State

Each `TermVectorsConsumerPerField` has its own:
- `BytesRefHash bytesHash` -- for deduplicating terms within this field/doc
- `ParallelPostingsArray` (specifically `TermVectorsPostingsArray`) -- per-term metadata
- Boolean flags: `doVectors`, `doVectorPositions`, `doVectorOffsets`, `doVectorPayloads`

### Per-Field BytesRefHash

Each `TermVectorsConsumerPerField`'s `bytesHash` is initialized in the `TermsHashPerField` constructor with the shared `termBytePool`. The hash does NOT copy term bytes; it references them by offset in the shared pool. For the TV case (secondary TermsHash), `addByPoolOffset()` is used to look up terms that are already in the pool.

## 4. Per-Term Data

### TermVectorsPostingsArray

Extends `ParallelPostingsArray` with three additional parallel arrays (defined at line 319):

| Array | Type | Purpose |
|-------|------|---------|
| `textStarts` | `int[]` | (inherited) Offset into `termBytePool` where this term's text begins |
| `addressOffset` | `int[]` | (inherited) Offset into `intPool` where this term's stream addresses are stored |
| `byteStarts` | `int[]` | (inherited) Offset into `bytePool` where this term's first stream slice begins |
| `freqs` | `int[]` | Number of times this term occurred in the current document |
| `lastOffsets` | `int[]` | Last end offset seen (for delta encoding offsets) |
| `lastPositions` | `int[]` | Last position seen (for delta encoding positions) |

### writeProx() Encoding (TermVectorsConsumerPerField, line 235)

**Stream 1 (offsets)** -- written when `doVectorOffsets`:
- VInt: `startOffset - lastOffsets[termID]` (delta from last end offset)
- VInt: `endOffset - startOffset` (offset length)
- Updates `lastOffsets[termID] = endOffset`

**Stream 0 (positions)** -- written when `doVectorPositions`:
- If payload present and non-empty:
  - VInt: `(positionDelta << 1) | 1` (low bit = has payload)
  - VInt: payload length
  - Raw bytes: payload data
- Else:
  - VInt: `positionDelta << 1` (low bit = 0, no payload)
- Updates `lastPositions[termID] = fieldState.position`

## 5. Flush Sequence

### Document-Level: TermVectorsConsumer.finishDocument() (line 117)

Called after all fields in a document are inverted:

1. If `!hasVectors`, return immediately.
2. **Sort fields** by field name (UTF-16 order): `ArrayUtil.introSort(perFields, 0, numVectorFields)`. This uses `TermsHashPerField.compareTo()` which compares `fieldName`.
3. **Initialize writer** lazily via `initTermVectorsWriter()` -- creates the codec `TermVectorsWriter` on first document with vectors.
4. **Fill gaps**: `fill(docID)` writes empty documents (0 fields) for any doc IDs between `lastDocID` and current `docID` that had no vectors.
5. **Write document**: `writer.startDocument(numVectorFields)`.
6. **Write each field**: Iterates sorted `perFields`, calling `perFields[i].finishDocument()`.
7. **Finish document**: `writer.finishDocument()`.
8. Increment `lastDocID`.
9. **Reset pools**: `super.reset()` clears `intPool` and `bytePool` (drops all data, no zero-fill).
10. **Reset field list**: `resetFields()` nulls out `perFields` array and sets `numVectorFields = 0`.

### Per-Field: TermVectorsConsumerPerField.finishDocument() (line 76)

For each field that had term vectors:

1. **Sort terms**: `sortTerms()` sorts the `bytesHash` by term text, returning sorted term IDs.
2. **Start field**: `tv.startField(fieldInfo, numPostings, doVectorPositions, doVectorOffsets, hasPayloads)`.
3. **For each term** (in sorted order):
   a. Read term text from pool: `termBytePool.fillBytesRef(flushTerm, postings.textStarts[termID])`.
   b. `tv.startTerm(flushTerm, freq)`.
   c. If positions or offsets enabled:
      - Initialize `ByteSliceReader` for position stream (stream 0) and/or offset stream (stream 1) via `initReader()`.
      - `tv.addProx(freq, posReader, offReader)` -- reads VInt-encoded data from the byte slices.
   d. `tv.finishTerm()`.
4. `tv.finishField()`.
5. **Reset per-field state**: `reset()` clears the `bytesHash`.
6. Mark `fieldInfo.setStoreTermVectors()`.

### Per-Field Field Registration

During inversion, `TermVectorsConsumerPerField.finish()` (line 69) is called after all tokens for a field instance. If vectors are enabled and terms exist, it calls `termsWriter.addFieldToFlush(this)`, which appends the per-field to the `perFields` array in `TermVectorsConsumer`.

### Segment-Level: TermVectorsConsumer.flush() (line 72)

Called at segment flush time:

1. If `writer != null` (at least one doc had vectors):
   a. `fill(numDocs)` -- write empty entries for any trailing docs without vectors.
   b. `writer.finish(numDocs)` -- finalize the codec writer.
   c. Close writer.

## 6. Compression (Lucene90CompressingTermVectorsWriter)

### Configuration (from Lucene90TermVectorsFormat)

```java
new Lucene90CompressingTermVectorsFormat(
    "Lucene90TermVectorsData", "", CompressionMode.FAST, 1 << 12, 128, 10);
```

- **Format name**: `"Lucene90TermVectorsData"`
- **Compression**: `CompressionMode.FAST` (LZ4)
- **Chunk size**: 4096 bytes (term suffix buffer threshold)
- **Max docs per chunk**: 128
- **Block shift**: 10 (for `FieldsIndexWriter` block size)

### Chunking Strategy

Documents are accumulated in `pendingDocs` (a `Deque<DocData>`). A chunk is flushed when either:
- `termSuffixes.size() >= chunkSize` (4096 bytes of term suffix data accumulated)
- `pendingDocs.size() >= maxDocsPerChunk` (128 documents accumulated)

This is checked in `triggerFlush()` (line 380) after every `finishDocument()`.

### Buffered Data Structures

The writer buffers per-chunk data in memory:

| Buffer | Type | Content |
|--------|------|---------|
| `pendingDocs` | `Deque<DocData>` | Doc metadata: field count, field list with term counts/flags |
| `termSuffixes` | `ByteBuffersDataOutput` | Term suffix bytes (prefix-compressed within each field) |
| `payloadBytes` | `ByteBuffersDataOutput` | Payload bytes per document (appended to `termSuffixes` at `finishDocument()`) |
| `positionsBuf` | `int[]` | Absolute positions for all terms in the chunk |
| `startOffsetsBuf` | `int[]` | Start offsets for all terms |
| `lengthsBuf` | `int[]` | Offset lengths (endOffset - startOffset) |
| `payloadLengthsBuf` | `int[]` | Payload lengths per position |

Per-field metadata (`FieldData`) stores: `freqs[]`, `prefixLengths[]`, `suffixLengths[]`, and index offsets into the shared position/offset/payload buffers.

### startTerm() -- Prefix Compression (line 349)

Terms within a field are prefix-compressed against the previous term:
1. Compute `prefix = StringHelper.bytesDifference(lastTerm, term)`.
2. Record `prefixLength` and `suffixLength = term.length - prefix` in `FieldData`.
3. Write only the suffix bytes to `termSuffixes`.

### addProx() Override (line 735)

The compressing writer overrides `addProx()` to decode the VInt-encoded position/offset streams from `TermVectorsConsumerPerField` and store absolute values in its own buffers:

- **Positions**: Decodes delta-encoded positions from the `ByteSliceReader`, accumulates absolute positions in `positionsBuf[]`. For payloads, reads payload length and bytes.
- **Offsets**: Decodes delta-encoded start offsets and lengths, stores absolute start offsets in `startOffsetsBuf[]` and lengths in `lengthsBuf[]`.

### finishDocument() (line 324)

1. Appends `payloadBytes` to `termSuffixes` (payload bytes follow term bytes in the compressed blob).
2. Resets `payloadBytes`.
3. Increments `numDocs`.
4. Calls `triggerFlush()` and `flush(false)` if threshold reached.

## 7. Data Layout

### .tvm (Vector Metadata)

```
IndexHeader ("Lucene90TermVectorsIndex" + "Meta", VERSION_CURRENT, segmentID, suffix)
VInt: PackedInts.VERSION_CURRENT
VInt: chunkSize
ChunkIndexMetadata (written by FieldsIndexWriter.finish())
VLong: numChunks
VLong: numDirtyChunks
VLong: numDirtyDocs
CodecFooter
```

### .tvx (Vector Index)

```
IndexHeader ("Lucene90TermVectorsIndex", VERSION_CURRENT, segmentID, suffix)
ChunkIndex (written by FieldsIndexWriter -- DirectMonotonic encoding of doc-to-pointer mapping)
CodecFooter
```

### .tvd (Vector Data)

```
IndexHeader ("Lucene90TermVectorsData", VERSION_CURRENT, segmentID, suffix)
<Chunk>*
CodecFooter
```

#### Chunk Layout (flush() method, line 384)

Each chunk contains data for 1 to `maxDocsPerChunk` documents:

```
VInt: docBase (first doc ID in chunk)
VInt: (chunkDocs << 1) | dirtyBit

--- NumFields ---
If chunkDocs == 1: VInt(numFields)
If chunkDocs > 1:  BlockPackedWriter(numFields per doc, block size 64)

--- FieldNums (sorted unique field numbers) ---
Byte: token = (min(numDistinctFields-1, 0x07) << 5) | bitsRequired
If numDistinctFields-1 >= 0x07: VInt(numDistinctFields - 1 - 0x07)
PackedInts(fieldNum for each distinct field)

--- FieldNumOffs (index of each field's number in FieldNums) ---
VLong: size of packed data
DirectWriter(fieldNumIndex per field instance)

--- Flags ---
VInt: 0 if all instances of a field always have the same flags, 1 otherwise
VInt: size of packed flags data
If constant: DirectWriter(flags per distinct field, 3 bits)
If varying:  DirectWriter(flags per field instance, 3 bits)
  Flag bits: POSITIONS=0x01, OFFSETS=0x02, PAYLOADS=0x04

--- NumTerms ---
VInt: bitsRequired
VInt: size of packed data
DirectWriter(numTerms per field instance)

--- TermLengths ---
BlockPackedWriter(prefixLength per term, block size 64)
BlockPackedWriter(suffixLength per term, block size 64)

--- TermFreqs ---
BlockPackedWriter(freq - 1 per term, block size 64)

--- Positions (if any field has positions) ---
BlockPackedWriter(positionDelta per occurrence, block size 64)
  Delta is reset to 0 at the start of each term.

--- StartOffsets (if any field has offsets) ---
Int[numDistinctFields]: Float.floatToRawIntBits(avgCharsPerTerm)
BlockPackedWriter(startOffsetDelta per occurrence, block size 64)
  Delta = startOffset - previousOffset - (int)(avgCharsPerTerm * (position - previousPosition))

--- Lengths (if any field has offsets) ---
BlockPackedWriter(endOffset - startOffset - termLength per occurrence, block size 64)

--- PayloadLengths (if any field has payloads) ---
BlockPackedWriter(payloadLength per occurrence, block size 64)

--- TermAndPayloads ---
LZ4-compressed(termSuffixes + payloadBytes)
```

## 8. Memory Lifecycle

### Allocation

**TermVectorsConsumer pools** -- allocated once in the `TermVectorsConsumer` constructor (via `TermsHash`):
- `intPool = new IntBlockPool(intBlockAllocator)` -- empty initially, blocks allocated on demand
- `bytePool = new ByteBlockPool(byteBlockAllocator)` -- empty initially, blocks allocated on demand
- `termBytePool` is set to the primary `TermsHash`'s `bytePool` (shared, not owned)

**TermVectorsWriter** -- lazily created on the first document that has term vectors, via `initTermVectorsWriter()`. Allocates:
- `positionsBuf[1024]`, `startOffsetsBuf[1024]`, `lengthsBuf[1024]`, `payloadLengthsBuf[1024]`
- `termSuffixes` and `payloadBytes` (`ByteBuffersDataOutput`, resettable)
- `pendingDocs` (ArrayDeque)
- `lastTerm` (BytesRef with initial capacity ~30)

**Per-field** -- `TermVectorsConsumerPerField` allocates:
- `BytesRefHash` with initial size 4 (wrapping shared `termBytePool`)
- `TermVectorsPostingsArray` initially size 2 (via `PostingsBytesStartArray.init()`)

### Per-Document Reset

After each document, `TermVectorsConsumer.finishDocument()` (line 117) performs:

1. **`super.reset()`** (`TermsHash.reset()`, line 71): Resets `intPool` and `bytePool` with `reset(false, false)` -- drops all buffers without zeroing or reuse. This frees all per-term stream data that was accumulated during the document.

2. **`resetFields()`** (line 155): Nulls out the `perFields` array and sets `numVectorFields = 0`.

3. Each `TermVectorsConsumerPerField.finishDocument()` calls `reset()` which calls `bytesHash.clear(false)` -- clears the hash without zeroing the underlying storage.

This per-document reset is critical: unlike `FreqProxTermsWriter` which accumulates across the entire segment, term vectors data is flushed to the codec writer document-by-document, so the pools can be recycled after each document.

### Per-Chunk Reset (Codec Writer)

In `Lucene90CompressingTermVectorsWriter.flush()` (line 430):
- `pendingDocs.clear()` -- drops all `DocData` and `FieldData` objects
- `termSuffixes.reset()` -- resets the term suffix buffer
- `payloadBytes` was already reset in each `finishDocument()`

The `positionsBuf`, `startOffsetsBuf`, `lengthsBuf`, and `payloadLengthsBuf` arrays are NOT reset between chunks -- they are simply overwritten, with each `FieldData` tracking its own start index.

### Segment Flush Cleanup

At `TermVectorsConsumer.flush()` (line 72):
- `writer.finish(numDocs)` finalizes the codec writer (flushes any remaining dirty chunk, writes metadata)
- `IOUtils.close(writer)` closes all output streams

The `TermVectorsConsumer` itself (and its pools) persists for the lifetime of the `IndexingChain` / `DocumentsWriterPerThread`.

### SortingTermVectorsConsumer

When index sorting is enabled, `SortingTermVectorsConsumer` overrides `initTermVectorsWriter()` to write to a temporary directory with an uncompressed format (`NO_COMPRESSION`, 8KB chunks). At `flush()`, it reads back the temporary files in sorted doc order and re-writes them through the real codec writer. The temporary files are deleted after the sorted write completes.
