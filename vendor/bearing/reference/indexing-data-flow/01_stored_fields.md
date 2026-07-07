# Stored Fields Indexing Path — Lucene 10.3.2

## Class Hierarchy

```
StoredFieldsConsumer                          (index package, per-segment consumer)
  └─ StoredFieldsWriter                       (codecs, abstract writer API)
       └─ Lucene90CompressingStoredFieldsWriter  (concrete codec writer)
            ├─ FieldsIndexWriter              (writes .fdx index via DirectMonotonicWriter)
            └─ Compressor                     (LZ4WithPresetDict or DeflateWithPresetDict)

StoredFieldsFormat                            (codecs, abstract format factory)
  └─ Lucene90StoredFieldsFormat               (selects BEST_SPEED or BEST_COMPRESSION)
       └─ Lucene90CompressingStoredFieldsFormat  (parameterized format, creates reader/writer)
```

## 1. Files Produced

| Extension | File | Purpose |
|-----------|------|---------|
| `.fdt` | Fields data | Compressed document data in chunks. Contains chunk headers (docBase, numDocs, numStoredFields[], lengths[]) followed by compressed field data. |
| `.fdx` | Fields index | Two `DirectMonotonicWriter` arrays: (1) first doc ID of each chunk, (2) start file pointer of each chunk in .fdt. Used for binary search at read time. |
| `.fdm` | Fields meta | Metadata for the monotonic arrays in .fdx. Also stores chunkSize, numChunks, numDirtyChunks, numDirtyDocs, and the DirectMonotonicWriter metadata blocks. |

### Temporary Files (deleted during finish)

`FieldsIndexWriter` creates two temp files during indexing:
- `{segment}_{codecName}-doc_ids_{suffix}` — accumulates VInt-encoded doc counts per chunk
- `{segment}_{codecName}file_pointers_{suffix}` — accumulates VLong-encoded delta file pointers per chunk

These are read back, converted into `DirectMonotonicWriter` format, written to .fdx, then deleted.

## 2. Data Input

`StoredFieldsConsumer.writeField(FieldInfo, StoredValue)` dispatches on `StoredValue.getType()`:

| StoredValue Type | Writer Method | Type Code | Value |
|------------------|---------------|-----------|-------|
| `INTEGER` | `writeField(FieldInfo, int)` | `NUMERIC_INT` (0x02) | ZigZag-encoded int via `writeZInt` |
| `LONG` | `writeField(FieldInfo, long)` | `NUMERIC_LONG` (0x04) | Timestamp-compressed long via `writeTLong` |
| `FLOAT` | `writeField(FieldInfo, float)` | `NUMERIC_FLOAT` (0x03) | Variable-length float via `writeZFloat` |
| `DOUBLE` | `writeField(FieldInfo, double)` | `NUMERIC_DOUBLE` (0x05) | Variable-length double via `writeZDouble` |
| `BINARY` | `writeField(FieldInfo, BytesRef)` | `BYTE_ARR` (0x01) | VInt length + raw bytes |
| `DATA_INPUT` | `writeField(FieldInfo, StoredFieldDataInput)` | `BYTE_ARR` (0x01) | VInt length + copied bytes |
| `STRING` | `writeField(FieldInfo, String)` | `STRING` (0x00) | `writeString` (VInt UTF-8 length + UTF-8 bytes) |

Each field is written to the in-memory buffer as:
```
VLong(fieldNumber << TYPE_BITS | typeCode)   // TYPE_BITS = 3, TYPE_MASK = 0x07
<encoded value>
```

The `fieldNumber` is `FieldInfo.number` (the field's ordinal in the segment's `FieldInfos`).

### Gap Documents

`StoredFieldsConsumer.startDocument(docID)` fills gaps for missing doc IDs by emitting empty `startDocument()`/`finishDocument()` pairs (zero stored fields). This ensures every doc ID from 0 to maxDoc-1 has an entry.

## 3. In-Memory Buffering

### Data Structures in `Lucene90CompressingStoredFieldsWriter`

| Field | Type | Purpose |
|-------|------|---------|
| `bufferedDocs` | `ByteBuffersDataOutput` (resettable) | Serialized field data for the current chunk. All fields from all docs in the chunk are appended here. |
| `numStoredFields` | `int[]` (initial size 16, grows via `ArrayUtil.oversize`) | Per-document count of stored fields. Index = position within current chunk. |
| `endOffsets` | `int[]` (initial size 16, grows with numStoredFields) | Per-document end offset within `bufferedDocs`. Used to compute per-doc lengths. |
| `numStoredFieldsInDoc` | `int` | Running count of fields written for the current document. Reset to 0 in `finishDocument()`. |
| `numBufferedDocs` | `int` | Number of documents accumulated in the current chunk. |
| `docBase` | `int` | Doc ID of the first document in the current chunk. |

### Buffer Sizes (BEST_SPEED mode)

- `chunkSize`: 80KB (`10 * 8 * 1024 = 81920`)
- `maxDocsPerChunk`: 1024
- `blockShift`: 10 (for DirectMonotonicWriter, meaning blocks of 2^10 = 1024 entries)

### Buffer Sizes (BEST_COMPRESSION mode)

- `chunkSize`: 480KB (`10 * 48 * 1024 = 491520`)
- `maxDocsPerChunk`: 4096
- `blockShift`: 10

### RAM Accounting

`ramBytesUsed()` returns:
```
bufferedDocs.ramBytesUsed() + numStoredFields.length * 4 + endOffsets.length * 4
```

## 4. Compression

### BEST_SPEED: `LZ4WithPresetDictCompressionMode`

Class: `codecs.lucene90.LZ4WithPresetDictCompressionMode`

The compressor (`LZ4WithPresetDictCompressor`) works as follows:

1. **Dictionary size**: `min(LZ4.MAX_DISTANCE, len / (NUM_SUB_BLOCKS * DICT_SIZE_FACTOR))` where `NUM_SUB_BLOCKS = 10`, `DICT_SIZE_FACTOR = 2`. So dict is roughly `len / 20`.
2. **Block size**: `(len - dictLength + NUM_SUB_BLOCKS - 1) / NUM_SUB_BLOCKS` — targets 10 sub-blocks.
3. **Output format**:
   - `VInt(dictLength)`
   - `VInt(blockLength)`
   - `VInt(compressedDictLength)` — compressed length of dictionary
   - `VInt(compressedBlockLength)` * N — compressed length of each sub-block
   - Raw compressed bytes (dictionary + all sub-blocks concatenated)
4. The dictionary is compressed standalone with `LZ4.compressWithDictionary(bytes, 0, 0, dictLen, ...)`.
5. Each sub-block is compressed with the dictionary as context: `LZ4.compressWithDictionary(bytes, 0, dictLen, blockLen, ...)`.
6. All compressed data is buffered in a `ByteBuffersDataOutput`, then written to the output after all lengths are emitted.

### BEST_COMPRESSION: `DeflateWithPresetDictCompressionMode`

Uses `java.util.zip.Deflater` level 6 with `nowrap=true` (raw DEFLATE, no zlib headers). Targets 10 sub-blocks of 48KB each.

### Sliced Chunks

When `bufferedDocs.size() >= 2 * chunkSize` (the "sliced" flag), the chunk is compressed in multiple pieces of `chunkSize` bytes each, rather than as a single block. This allows partial decompression at read time — a reader only needs to decompress enough sub-blocks to reach the desired document.

## 5. Write Triggers

Flushing from buffer to disk is triggered by `triggerFlush()` in `Lucene90CompressingStoredFieldsWriter`:

```java
private boolean triggerFlush() {
    return bufferedDocs.size() >= chunkSize       // buffer >= 80KB (BEST_SPEED)
        || numBufferedDocs >= maxDocsPerChunk;     // >= 1024 docs (BEST_SPEED)
}
```

This is checked at the end of every `finishDocument()` call.

### Forced Flush

`flush(true)` is called from `finish(int numDocs)` when there are remaining buffered docs that haven't hit the threshold. These are marked as **dirty chunks**:
- `numDirtyChunks` is incremented
- `numDirtyDocs` is incremented by the number of docs in the forced flush

Dirty chunk tracking is used during merge to decide whether to recompress a segment (`tooDirty()` check).

## 6. Flush Sequence

### Per-Document Calls (from `StoredFieldsConsumer`)

```
StoredFieldsConsumer.startDocument(docID)
    → writer.startDocument()                     // no-op in compressing writer
    → (fills gap docs if docID > lastDoc + 1)

StoredFieldsConsumer.writeField(info, value)
    → writer.writeField(info, value)             // appends to bufferedDocs, increments numStoredFieldsInDoc

StoredFieldsConsumer.finishDocument()
    → writer.finishDocument()
        1. Grow numStoredFields[] and endOffsets[] if needed
        2. numStoredFields[numBufferedDocs] = numStoredFieldsInDoc
        3. numStoredFieldsInDoc = 0
        4. endOffsets[numBufferedDocs] = bufferedDocs.size()
        5. ++numBufferedDocs
        6. if triggerFlush() → flush(false)
```

### Chunk Flush (internal `flush(boolean force)`)

```
flush(force):
    1. numChunks++
    2. if force: numDirtyChunks++, numDirtyDocs += numBufferedDocs
    3. indexWriter.writeIndex(numBufferedDocs, fieldsStream.getFilePointer())
    4. Convert endOffsets[] to per-doc lengths (in-place, backwards)
    5. Determine sliced = bufferedDocs.size() >= 2 * chunkSize
    6. writeHeader(docBase, numBufferedDocs, numStoredFields, lengths, sliced, force)
         → fieldsStream.writeVInt(docBase)
         → fieldsStream.writeVInt((numBufferedDocs << 2) | dirtyBit | slicedBit)
         → saveInts(numStoredFields, numBufferedDocs, fieldsStream)
         → saveInts(lengths, numBufferedDocs, fieldsStream)
    7. Compress bufferedDocs to fieldsStream
         → if sliced: compress in chunkSize slices
         → else: compress as single block
    8. Reset: docBase += numBufferedDocs, numBufferedDocs = 0, bufferedDocs.reset()
```

### Segment Finish (`finish(int numDocs)`)

```
finish(numDocs):
    1. if numBufferedDocs > 0: flush(true)      // force-flush remaining
    2. assert docBase == numDocs
    3. indexWriter.finish(numDocs, fieldsStream.getFilePointer(), metaStream)
         → Writes footers to temp files, closes them
         → Creates .fdx file with index header
         → Reads temp doc_ids file → DirectMonotonicWriter (cumulative doc IDs)
         → Reads temp file_pointers file → DirectMonotonicWriter (cumulative file pointers)
         → Deletes temp files
         → Writes metadata pointers to metaStream
    4. metaStream.writeVLong(numChunks)
    5. metaStream.writeVLong(numDirtyChunks)
    6. metaStream.writeVLong(numDirtyDocs)
    7. CodecUtil.writeFooter(metaStream)
    8. CodecUtil.writeFooter(fieldsStream)
```

### Close (from `StoredFieldsConsumer.flush`)

```
StoredFieldsConsumer.flush(state, sortMap):
    1. writer.finish(state.segmentInfo.maxDoc())
    2. IOUtils.close(writer)                    // closes metaStream, fieldsStream, indexWriter, compressor
```

## 7. Data Layout

### .fdm (Meta) File Layout

```
CodecUtil.IndexHeader("Lucene90FieldsIndexMeta", VERSION_CURRENT=1, segmentID, suffix)
VInt(chunkSize)                              // e.g., 81920 for BEST_SPEED

--- FieldsIndexWriter.finish() writes: ---
Int(numDocs)
Int(blockShift)                              // 10
Int(totalChunks + 1)
Long(dataPointer)                            // file pointer in .fdx where doc IDs start
  [DirectMonotonicWriter meta for doc IDs]
Long(dataPointer)                            // file pointer in .fdx where file pointers start
  [DirectMonotonicWriter meta for file pointers]
Long(dataPointer)                            // file pointer at end of .fdx data
Long(maxPointer)                             // file pointer at end of last chunk in .fdt
--- end FieldsIndexWriter ---

VLong(numChunks)
VLong(numDirtyChunks)
VLong(numDirtyDocs)
CodecUtil.Footer
```

### .fdx (Index) File Layout

```
CodecUtil.IndexHeader("Lucene90FieldsIndexIdx", VERSION_CURRENT=0, segmentID, suffix)
[DirectMonotonicWriter data: doc IDs array, totalChunks+1 entries]
[DirectMonotonicWriter data: file pointers array, totalChunks+1 entries]
CodecUtil.Footer
```

The doc IDs array contains cumulative first-doc-of-chunk values: `[0, chunk0_docs, chunk0_docs+chunk1_docs, ...]`.
The file pointers array contains the .fdt file offset where each chunk starts, plus a sentinel entry for `maxPointer`.

### .fdt (Fields Data) File Layout

```
CodecUtil.IndexHeader(formatName, VERSION_CURRENT=1, segmentID, suffix)

--- Repeated per chunk: ---
VInt(docBase)                                // first doc ID in this chunk
VInt((numBufferedDocs << 2) | dirtyBit | slicedBit)
                                              // bit 0: sliced flag
                                              // bit 1: dirty chunk flag
                                              // bits 2+: doc count in chunk

// numStoredFields array (one entry per doc in chunk):
if numBufferedDocs == 1:
    VInt(numStoredFields[0])
else:
    StoredFieldsInts.writeInts(numStoredFields, 0, count, out)
        Byte(bitsPerValue)                   // 0=all-equal, 8, 16, or 32
        if bpv == 0: VInt(value)
        else: packed int array (128-element blocks written as longs, remainder scalar)

// lengths array (per-doc byte lengths within decompressed data):
// Same encoding as numStoredFields via StoredFieldsInts.writeInts

// Compressed field data:
if sliced:
    for each chunkSize slice of bufferedDocs:
        <compressed block>
else:
    <compressed block>
--- end chunk ---

CodecUtil.Footer
```

### Per-Field Encoding Within Decompressed Data

Each field within the decompressed `bufferedDocs` buffer:
```
VLong(fieldNumber << 3 | typeCode)           // TYPE_BITS = 3

// Then based on typeCode:
STRING (0x00):      writeString(value)       // VInt(utf8Length) + UTF-8 bytes
BYTE_ARR (0x01):    VInt(length) + bytes
NUMERIC_INT (0x02): writeZInt(value)         // ZigZag-encoded VInt
NUMERIC_FLOAT (0x03): writeZFloat(value)     // 1-5 bytes, see below
NUMERIC_LONG (0x04): writeTLong(value)       // Timestamp-compressed, 1-10 bytes
NUMERIC_DOUBLE (0x05): writeZDouble(value)   // 1-9 bytes, see below
```

### Numeric Encoding Details

**writeZFloat**: Small integers [-1..125] = 1 byte (`0x80 | (1+intVal)`). Positive floats = 4 bytes (big-endian float bits). Negative floats = 5 bytes (`0xFF` + 4-byte int bits).

**writeZDouble**: Small integers [-1..124] = 1 byte. Values representable as float = 5 bytes (`0xFE` + float bits). Positive doubles = 8 bytes. Negative doubles = 9 bytes (`0xFF` + 8-byte long bits).

**writeTLong**: Timestamp compression. Header byte encodes time granularity (bits 7-6: `00`=raw, `01`=seconds, `10`=hours, `11`=days), continuation bit (bit 5), and 5 low bits of zigzag-encoded value. If continuation bit set, remaining upper bits follow as VLong.

### StoredFieldsInts Encoding

`StoredFieldsInts.writeInts` encodes an array of ints with adaptive bit width:

| Tag Byte | Meaning | Encoding |
|----------|---------|----------|
| 0 | All values equal | `VInt(value)` |
| 8 | 8 bits per value | Blocks of 128 packed as 16 longs, remainder as individual bytes |
| 16 | 16 bits per value | Blocks of 128 packed as 32 longs, remainder as individual shorts |
| 32 | 32 bits per value | Blocks of 128 packed as 64 longs, remainder as individual ints |

## 8. Memory Lifecycle

### Allocation

- **Constructor**: `bufferedDocs` (resettable `ByteBuffersDataOutput`), `numStoredFields[16]`, `endOffsets[16]`, `Compressor` instance (includes LZ4 hash table).
- **FieldsIndexWriter constructor**: Opens two temp `IndexOutput` files for doc IDs and file pointers.
- **LZ4WithPresetDictCompressor**: Internal `ByteBuffersDataOutput` for compressed data, `LZ4.FastCompressionHashTable`, reusable `byte[] buffer`.

### Growth

- `numStoredFields[]` and `endOffsets[]` grow via `ArrayUtil.oversize(numBufferedDocs + 1, 4)` when the chunk's doc count exceeds current array length. They are never shrunk.
- `bufferedDocs` grows internally as documents are added. It is a `ByteBuffersDataOutput` backed by a list of byte buffers.
- The LZ4 compressor's internal `buffer` grows to `dictLength + blockLength` via `ArrayUtil.growNoCopy`.

### Per-Chunk Reset

In `flush()`, after compression:
- `bufferedDocs.reset()` — releases internal byte buffers (resettable instance recycles them).
- `numBufferedDocs = 0` — logical reset of the numStoredFields/endOffsets arrays (no deallocation).
- `docBase` advances by the flushed doc count.

### Final Cleanup

In `finish()`:
- Remaining buffered docs are force-flushed.
- `FieldsIndexWriter.finish()` reads temp files, writes .fdx, deletes temp files.
- `close()` closes and nulls `metaStream`, `fieldsStream`, `indexWriter`, `compressor`.

### Peak Memory Profile

Peak memory occurs just before a chunk flush, when `bufferedDocs` has accumulated up to `chunkSize` bytes:
- **BEST_SPEED**: ~80KB of buffered doc data + compression overhead (LZ4 hash table is 16KB for `FastCompressionHashTable`, buffer ~8KB + dictLength)
- **BEST_COMPRESSION**: ~480KB of buffered doc data + Deflater internal state
- Plus: `numStoredFields[]` and `endOffsets[]` at 4 bytes each per doc in chunk (max 1024 or 4096 docs = 8KB or 32KB)
- Plus: the `ByteBuffersDataOutput` inside `LZ4WithPresetDictCompressor` holds all compressed sub-block data until it's copied to the output

The writer is created lazily in `StoredFieldsConsumer.initStoredFieldsWriter()` — it's only allocated when the first document with stored fields arrives. The `Accountable` interface tracks memory via `ramBytesUsed()`.
