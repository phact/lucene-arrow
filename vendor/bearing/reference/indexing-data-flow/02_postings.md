# Postings: Term Collection, Encoding, and Flush

This document covers the complete postings indexing path in Lucene 10.3.2: how terms
are collected during field inversion, stored in memory, and flushed to codec files.

## 1. Files Produced

The postings subsystem produces up to six files per segment:

| File | Extension | Class | Purpose |
|------|-----------|-------|---------|
| Postings metadata | `.psm` | `Lucene103PostingsWriter` | Impact stats (max impacts/bytes at level 0 and 1), file lengths for `.doc`, `.pos`, `.pay` |
| Doc IDs + frequencies | `.doc` | `Lucene103PostingsWriter` | Block-encoded doc deltas and frequencies, with two-level skip data |
| Positions | `.pos` | `Lucene103PostingsWriter` | PFor-encoded position deltas; vInt tail for final partial block; payloads inline for tail |
| Payloads + offsets | `.pay` | `Lucene103PostingsWriter` | PFor-encoded payload lengths, raw payload bytes, PFor-encoded offset deltas and lengths |
| Terms dictionary | `.tim` | `Lucene103BlockTreeTermsWriter` | Term suffix blocks (25-48 entries each), with per-term stats and postings metadata |
| Terms index | `.tip` | `Lucene103BlockTreeTermsWriter` | Trie index mapping term prefixes to `.tim` block file pointers |
| Terms metadata | `.tmd` | `Lucene103BlockTreeTermsWriter` | Per-field stats (numTerms, sumDocFreq, docCount, minTerm, maxTerm) and trie root pointers |

Conditional creation:
- `.pos` is created only if any field has `IndexOptions >= DOCS_AND_FREQS_AND_POSITIONS`
- `.pay` is created only if any field has payloads OR offsets
- `.psm`, `.doc` are always created when postings exist

## 2. Term Collection During Field Inversion

### 2.1 Pool Architecture

`TermsHash` owns three shared pools that all per-field instances draw from:

```
TermsHash (FreqProxTermsWriter)
  ├── intPool:  IntBlockPool     (blocks of 8192 ints, shift=13)
  ├── bytePool: ByteBlockPool   (blocks of 32768 bytes, shift=15)
  └── termBytePool: ByteBlockPool  (shared reference — same as bytePool for primary)
```

**Key detail**: When `TermsHash` has a `nextTermsHash` (term vectors), the primary's
`bytePool` is shared as both `termBytePool` references. The secondary `TermsHash`
(term vectors) gets its own `intPool` and `bytePool`, but uses the primary's pool for
term bytes via `termBytePool`. This is set in the `TermsHash` constructor (line 54-56):

```java
if (nextTermsHash != null) {
    termBytePool = bytePool;
    nextTermsHash.termBytePool = bytePool;
}
```

### 2.2 BytesRefHash: Term Deduplication

Each `TermsHashPerField` owns a `BytesRefHash` initialized with `HASH_INIT_SIZE = 4`.
The `BytesRefHash` stores term bytes in the shared `termBytePool` and maintains a hash
table for deduplication.

When `TermsHashPerField.add(BytesRef termBytes, int docID)` is called:

1. `bytesHash.add(termBytes)` interns the term text into `termBytePool`, returning:
   - `termID >= 0`: New term. First occurrence since last flush.
   - `termID < 0`: Existing term. Actual ID is `(-termID) - 1`.

2. For **new terms** (`termID >= 0`): calls `initStreamSlices(termID, docID)`
3. For **existing terms** (`termID < 0`): calls `positionStreamSlice(termID, docID)`

### 2.3 Stream Slice Initialization (New Term)

`TermsHashPerField.initStreamSlices(int termID, int docID)` (line 150):

1. **Reserve int slots**: If the current `intPool` buffer doesn't have room for
   `streamCount` ints, advance to the next buffer. Then reserve `streamCount` int
   slots by advancing `intPool.intUpto`. These slots track the current write position
   for each stream.

2. **Reserve byte slices**: If the current `bytePool` buffer doesn't have room for
   `2 * streamCount * FIRST_LEVEL_SIZE` bytes, advance to next buffer.

3. **Store address offset**: `postingsArray.addressOffset[termID]` stores the global
   int pool offset for this term's stream address slots.

4. **Allocate initial slices**: For each stream (0..streamCount-1), call
   `slicePool.newSlice(FIRST_LEVEL_SIZE)` to allocate a 5-byte initial slice in the
   `bytePool`. Store the global byte offset in the corresponding int slot.

5. **Record byte start**: `postingsArray.byteStarts[termID]` records the start of
   stream 0's first slice.

6. Call `newTerm(termID, docID)` (implemented by `FreqProxTermsWriterPerField`).

### 2.4 Stream Positioning (Existing Term)

`TermsHashPerField.positionStreamSlice(int termID, int docID)` (line 209):

1. Look up `postingsArray.addressOffset[termID]` to find this term's int pool slots.
2. Set `termStreamAddressBuffer` and `streamAddressOffset` to point at the correct
   int buffer and offset.
3. Call `addTerm(termID, docID)` (implemented by `FreqProxTermsWriterPerField`).

### 2.5 ByteSlicePool: Linked Slice Allocation

`ByteSlicePool` manages variable-length slices within the `ByteBlockPool`. Slices grow
through 10 levels:

| Level | Size (bytes) |
|-------|-------------|
| 0 | 5 |
| 1 | 14 |
| 2 | 20 |
| 3 | 30 |
| 4 | 40 |
| 5 | 40 |
| 6 | 80 |
| 7 | 80 |
| 8 | 120 |
| 9 | 200 |

Level 9 is the maximum; `NEXT_LEVEL_ARRAY[9] == 9`, so all subsequent allocations
remain at 200 bytes.

**Slice structure**: Each slice is zero-filled. The last byte stores `16 | level` as
a non-zero sentinel. When writing encounters a non-zero byte at the current position,
it knows the slice is full and calls `allocSlice`.

**Forwarding address**: When a new slice is allocated (`allocSlice`, line 97), the last
4 bytes of the old slice are overwritten with a little-endian int pointing to the new
slice's global offset. The 3 data bytes that were there are copied forward to the
start of the new slice.

### 2.6 Writing to Byte Streams

`TermsHashPerField.writeByte(int stream, byte b)` (line 218):

1. Look up the current write address for this stream from `termStreamAddressBuffer`.
2. Resolve to the actual byte array and offset within the `ByteBlockPool`.
3. If the byte at that position is non-zero (sentinel), allocate a new slice.
4. Write the byte and increment the address.

`writeVInt(int stream, int i)` (line 260) encodes a variable-length integer using
standard VInt encoding (7 bits per byte, high bit = continuation).

## 3. Per-Term Data in Memory

### 3.1 Stream Layout

`FreqProxTermsWriterPerField` uses 1 or 2 streams per term:

- **Stream 0** (always): Doc IDs and frequencies
- **Stream 1** (if `hasProx`): Positions, offsets, and payloads

The `streamCount` is set in the constructor (line 50-51):
- `streamCount = 2` if `indexOptions >= DOCS_AND_FREQS_AND_POSITIONS`
- `streamCount = 1` otherwise

### 3.2 Stream 0: Doc/Freq Encoding

**On `newTerm` (first occurrence)**: Only the parallel arrays are updated. No bytes are
written to stream 0 yet. The doc code and freq are held in `postingsArray.lastDocCodes`
and `postingsArray.termFreqs`.

- Without freq: `lastDocCodes[termID] = docID`
- With freq: `lastDocCodes[termID] = docID << 1`

**On `addTerm` (same term, new doc)**: The *previous* doc's data is flushed to stream 0:

- Without freq: `writeVInt(0, lastDocCodes[termID])` then update codes with delta
- With freq, freq==1: `writeVInt(0, lastDocCodes[termID] | 1)` (freq packed into low bit)
- With freq, freq>1: `writeVInt(0, lastDocCodes[termID])` then `writeVInt(0, termFreqs[termID])`

The new doc's code is computed as `(docID - lastDocIDs[termID]) << 1`.

**On `addTerm` (same term, same doc)**: Only `termFreqs[termID]` is incremented. No
bytes written to stream 0.

**Critical**: The *last* doc's data for each term is never written to the byte stream.
It remains in the parallel arrays (`lastDocIDs`, `lastDocCodes`, `termFreqs`) and is
consumed directly during the flush readback by `FreqProxDocsEnum`.

### 3.3 Stream 1: Position/Offset/Payload Encoding

Written by `writeProx(int termID, int proxCode)` (line 84):

- Position delta is `proxCode << 1`, with low bit indicating payload presence
- If payload present: `writeVInt(1, (proxCode << 1) | 1)`, then `writeVInt(1, payload.length)`,
  then `writeBytes(1, payload.bytes, ...)`
- If no payload: `writeVInt(1, proxCode << 1)`

Offsets written by `writeOffsets(int termID, int offsetAccum)` (line 103):
- `writeVInt(1, startOffset - lastOffsets[termID])` (delta from last start offset)
- `writeVInt(1, endOffset - startOffset)` (length)

For `newTerm`: proxCode is `fieldState.position` (absolute position).
For `addTerm` same doc: proxCode is `fieldState.position - lastPositions[termID]` (delta).
For `addTerm` new doc: proxCode is `fieldState.position` (absolute), and `lastOffsets` is reset to 0.

## 4. The Parallel Postings Array

### 4.1 Base Arrays (ParallelPostingsArray)

Every `TermsHashPerField` has a `ParallelPostingsArray` indexed by termID. The base class
provides three `int[]` arrays:

| Array | Purpose |
|-------|---------|
| `textStarts[termID]` | Offset into `termBytePool` where this term's text bytes begin |
| `addressOffset[termID]` | Global offset into `intPool` where this term's stream address slots begin |
| `byteStarts[termID]` | Global offset into `bytePool` where stream 0's first slice begins |

`BYTES_PER_POSTING = 3 * Integer.BYTES = 12` bytes for the base.

### 4.2 FreqProxPostingsArray Extensions

`FreqProxPostingsArray` extends with additional arrays depending on `IndexOptions`:

| Array | Condition | Purpose |
|-------|-----------|---------|
| `lastDocIDs[termID]` | always | Last docID where this term was seen |
| `lastDocCodes[termID]` | always | Encoded doc code pending write (delta << 1 or raw docID) |
| `termFreqs[termID]` | `hasFreq` | Current document's term frequency |
| `lastPositions[termID]` | `hasProx` | Last position where this term occurred (for delta computation) |
| `lastOffsets[termID]` | `hasOffsets` | Last start offset (for delta computation) |

### 4.3 Growth Strategy

- Initial size: 2 (hardcoded in `PostingsBytesStartArray.init()`)
- Growth: `ArrayUtil.oversize(size + 1, bytesPerPosting())` -- geometric growth
- The `BytesStartArray` callback pattern means growth is triggered by `BytesRefHash`
  when it needs more term slots, which in turn calls `postingsArray.grow()`
- Memory accounting is tracked via the `Counter bytesUsed`

## 5. Flush Sequence

### 5.1 Triggering Flush

`FreqProxTermsWriter.flush()` (line 82):

1. Calls `super.flush()` which propagates to `nextTermsHash` (term vectors).
2. Collects all `FreqProxTermsWriterPerField` instances that have `numTerms > 0`.
3. Calls `perField.sortTerms()` on each — this sorts the `BytesRefHash` in
   lexicographic order and stores sorted term IDs.
4. Sorts the field list by field name (`CollectionUtil.introSort`).
5. Creates `FreqProxFields` adapter wrapping the sorted per-field list.
6. Applies pending deletes (if any).
7. Wraps in sorting adapter if index sort is active.
8. Opens `FieldsConsumer` (the `Lucene103BlockTreeTermsWriter`) and calls `consumer.write(fields, norms)`.

### 5.2 FreqProxFields Adapter

`FreqProxFields` implements the `Fields` interface over in-RAM buffered postings. The
class hierarchy is:

```
FreqProxFields (Fields)
  └── FreqProxTerms (Terms)
       └── FreqProxTermsEnum (BaseTermsEnum)
            ├── FreqProxDocsEnum (PostingsEnum) — for docs/freqs only
            └── FreqProxPostingsEnum (PostingsEnum) — for positions/offsets/payloads
```

**FreqProxTermsEnum**: Iterates terms in sorted order using `sortedTermIDs`. Uses
`BytesRefBlockPool` to read back term bytes from the pool. Provides binary search via
`seekCeil`.

**FreqProxDocsEnum** (line 301): Reads stream 0 back via `ByteSliceReader`:

1. `reset(termID)` calls `terms.initReader(reader, termID, 0)` to position the reader
   at stream 0's start.
2. `nextDoc()` reads vInts from the slice reader, decoding doc deltas and frequencies:
   - Without freq: `docID += reader.readVInt()`
   - With freq: `code = reader.readVInt()`, `docID += code >>> 1`, freq from low bit or next vInt
3. When `reader.eof()` is true, the **last document** is served from `postingsArray.lastDocIDs[termID]`
   and `postingsArray.termFreqs[termID]` directly (since it was never written to the stream).

**FreqProxPostingsEnum** (line 407): Uses two `ByteSliceReader` instances — one for
stream 0 (doc/freq) and one for stream 1 (pos/offset/payload). Same doc/freq logic as
above, plus position decoding in `nextPosition()`.

### 5.3 ByteSliceReader

`ByteSliceReader` (extends `DataInput`) reads data back from byte slices:

- `init(pool, startIndex, endIndex)`: Sets up to read from `startIndex` to `endIndex`.
  The `startIndex` is from `postingsArray.byteStarts[termID] + stream * FIRST_LEVEL_SIZE`.
  The `endIndex` is from the int pool's stream address slot.
- `eof()`: Returns true when `upto + bufferOffset == endIndex`.
- On reaching the end of a slice (`upto == limit`), `nextSlice()` reads the
  4-byte little-endian forwarding address from `buffer[limit]`, advances to the
  next level size, and jumps to the new buffer position.

### 5.4 TermsHashPerField.initReader

`initReader(ByteSliceReader reader, int termID, int stream)` (line 94):

- `startIndex` = `postingsArray.byteStarts[termID] + stream * FIRST_LEVEL_SIZE` —
  each stream's initial slice is at a fixed offset from the term's byte start.
- `endIndex` = `streamAddressBuffer[offsetInAddressBuffer + stream]` — the current
  write position, read from the int pool.

## 6. Block Encoding (Lucene103PostingsWriter)

### 6.1 Block Size and Skip Levels

- `BLOCK_SIZE = 128` documents per block
- `LEVEL1_FACTOR = 32` blocks per level-1 skip group
- `LEVEL1_NUM_DOCS = 4096` documents per level-1 skip entry
- `LEVEL1_MASK = 4095`

### 6.2 Document Block Encoding

`Lucene103PostingsWriter.flushDocBlock(boolean finishTerm)` (line 391):

**Full block (128 docs)**:

1. **Level-0 skip data** (written to `level0Output`):
   - Competitive impacts (from `level0FreqNormAccumulator`):
     `writeVLong(impactBytesLength)` then serialized impacts
   - Position file pointer delta: `writeVLong(posOut.getFilePointer() - level0LastPosFP)`
   - Position buffer position: `writeByte(posBufferUpto)`
   - If payloads/offsets: pay file pointer delta and `payloadByteUpto`

2. **Doc delta encoding** — one of three strategies:
   - **All-same delta** (`docRange == BLOCK_SIZE`): Write `bitsPerValue = 0`
   - **FOR delta encoding**: When `numBitsNextBitsPerValue <= docRange`, write
     `bitsPerValue` byte, then `ForDeltaUtil.encodeDeltas(bitsPerValue, docDeltaBuffer, out)`
   - **Bit set encoding**: When bit set is more compact, write `-numBitSetLongs` as a
     negative byte, then the long[] bit set. Used for dense doc ID ranges.

3. **Frequency encoding**: `pforUtil.encode(freqBuffer, level0Output)` — PFor encoding
   of 128 frequencies.

4. **Level-0 header** (written to `level1Output`):
   - `writeVInt15(docID - level0LastDocID)` — doc ID advance
   - `writeVLong15(level0Output.size())` — block data size
   - `writeVLong(numSkipBytes)` — total skip + block size

**Partial block (< 128 docs, only at term end)**:
- `PostingsUtil.writeVIntBlock()` — simple vInt encoding of doc deltas and frequencies.
  No skip data, no block encoding.

### 6.3 Level-1 Skip Data

Every 32 blocks (4,096 docs), `writeLevel1SkipData()` (line 494) writes to `docOut`:

1. `writeVInt(docID - level1LastDocID)` — doc ID advance for the group
2. If freqs: competitive impacts accumulated across all 32 blocks, plus position/payload
   file pointer state
3. `writeVLong(level1Len)` — total length of skip data + block data for this group
4. Two shorts: `(scratchOutput.size() + Short.BYTES)` and `numImpactBytes`
5. Buffered level-1 skip content
6. All 32 blocks' data from `level1Output`

### 6.4 PFor Encoding (PForUtil)

`PForUtil.encode(int[] ints, DataOutput out)` encodes 128 integers:

1. Find the top `MAX_EXCEPTIONS + 1 = 8` values using a `LongHeap`.
2. Compute `patchedBitsRequired` — the bits per value after patching (min: `maxBitsRequired - 8`).
3. Count exceptions (values exceeding `(1 << patchedBitsRequired) - 1`), max 7.
4. Write header byte: `(numExceptions << 5) | patchedBitsRequired`
5. Encode the 128 values at `patchedBitsRequired` bits per value using `ForUtil`.
6. Write exception patches as `(index, high_byte)` pairs — the index locates the
   value in the block, the byte stores the bits above `patchedBitsRequired`.

Special case: If all 128 values are equal, writes token `0` followed by the single value
as a vInt.

### 6.5 ForDeltaUtil

`ForDeltaUtil` encodes doc ID deltas specifically. It exploits the fact that deltas are
always positive, encoding them directly with Frame-of-Reference. The `encodeDeltas`
method writes cumulative sums that allow direct doc ID reconstruction during decoding.

### 6.6 Position Block Encoding

`Lucene103PostingsWriter.addPosition()` (line 303):

- Accumulates position deltas, payload lengths, offset deltas, and offset lengths into
  buffers of size `BLOCK_SIZE`.
- When `posBufferUpto == BLOCK_SIZE`:
  - `pforUtil.encode(posDeltaBuffer, posOut)` — PFor-encode 128 position deltas
  - If payloads: `pforUtil.encode(payloadLengthBuffer, payOut)`, then
    `payOut.writeVInt(payloadByteUpto)` + raw payload bytes
  - If offsets: `pforUtil.encode(offsetStartDeltaBuffer, payOut)` +
    `pforUtil.encode(offsetLengthBuffer, payOut)`

**Tail positions** (partial block at term end, in `finishTerm`):
- Written as vInts directly to `posOut` (not PFor-encoded)
- Payloads are interleaved with position data: `(posDelta << 1) | hasPayloadChanged`
- Offsets: `(delta << 1) | hasLengthChanged`

### 6.7 Singleton Doc Pulsing

When `docFreq == 1`, the single doc ID is "pulsed" into the terms dictionary metadata
instead of writing to `.doc`. In `finishTerm` (line 567):
- `singletonDocID = docDeltaBuffer[0] - 1`
- No `flushDocBlock` call
- The doc ID is encoded in `encodeTerm` via the `IntBlockTermState`

### 6.8 Term State Encoding (encodeTerm)

`encodeTerm(DataOutput out, FieldInfo fieldInfo, BlockTermState state, boolean absolute)` (line 654):

For consecutive singleton terms with the same `docStartFP`:
- `writeVLong((zigZagEncode(docIdDelta) << 1) | 0x01)` — exploits monotonic IDs

Otherwise:
- `writeVLong((state.docStartFP - lastState.docStartFP) << 1)` — doc file pointer delta
- If singleton: `writeVInt(singletonDocID)`
- If positions: `writeVLong(posStartFP delta)`
- If payloads/offsets: `writeVLong(payStartFP delta)`
- If `lastPosBlockOffset != -1`: `writeVLong(lastPosBlockOffset)`

### 6.9 Postings Metadata File (.psm)

Written in `Lucene103PostingsWriter.close()` (line 695), after the codec footer for `.doc`/`.pos`/`.pay`:

- `writeInt(maxNumImpactsAtLevel0)`
- `writeInt(maxImpactNumBytesAtLevel0)`
- `writeInt(maxNumImpactsAtLevel1)`
- `writeInt(maxImpactNumBytesAtLevel1)`
- `writeLong(docOut.getFilePointer())` — total `.doc` file length
- If positions: `writeLong(posOut.getFilePointer())` — total `.pos` file length
- If payloads/offsets: `writeLong(payOut.getFilePointer())` — total `.pay` file length

## 7. BlockTree Terms Dictionary

### 7.1 Architecture

`Lucene103BlockTreeTermsWriter` extends `FieldsConsumer`. It receives sorted terms from
`FreqProxFields` and organizes them into blocks for the `.tim` file, with a trie index
in `.tip` and per-field metadata in `.tmd`.

### 7.2 Block Formation

The `TermsWriter` inner class processes terms for a single field:

**Pending stack**: As terms arrive in sorted order, each becomes a `PendingTerm` or
`PendingBlock` on the `pending` list. `prefixStarts[i]` tracks where terms sharing a
prefix of length `i` begin in the pending list.

**pushTerm(BytesRef text)** (line 1017): For each term:
1. Compute the common prefix length with the previous term.
2. Walk backward from the old term's length to the common prefix length. At each level,
   if the number of pending entries sharing that prefix >= `minItemsInBlock` (25), and
   total remaining > `maxItemsInBlock` (48), call `writeBlocks`.
3. Update `prefixStarts` for the new term's suffix bytes.

**writeBlocks(int prefixLength, int count)** (line 573): Segments `count` entries from
the top of the pending stack into blocks:
- Iterates entries, grouping by suffix lead byte.
- When a group reaches `minItemsInBlock` and more entries remain than `maxItemsInBlock`,
  creates a **floor block** — a sub-block with a recorded floor lead byte.
- Each block is written via `writeBlock()`, producing a `PendingBlock`.
- The first block's trie index is compiled, aggregating sub-block indices.
- The written entries are replaced on the pending stack with a single `PendingBlock`.

**finish()** (line 1061): Pushes two empty terms to force all remaining entries through
`writeBlocks(0, pending.size())`, producing the root block.

### 7.3 Block Encoding (.tim)

Each block in `writeBlock()` (line 713) writes:

1. **Header**: `writeVInt(numEntries << 1 | isLast)` — entry count with last-block flag

2. **Suffix bytes**: Concatenated suffixes (term bytes after the common prefix), optionally
   compressed:
   - `NO_COMPRESSION`: Raw bytes
   - `LZ4`: When average suffix > 6 bytes and LZ4 saves > 25%
   - `LOWERCASE_ASCII`: Tried when LZ4 doesn't apply
   - Token: `(suffixLength << 3) | (isLeaf ? 0x04 : 0) | compressionCode`

3. **Suffix lengths**: VInt-encoded lengths per entry. For leaf blocks, raw suffix length.
   For inner blocks, `(suffix << 1)` for terms, `(suffix << 1) | 1` for sub-blocks.
   If all lengths are equal, run-length encoded as `(numBytes << 1) | 1` + single byte.

4. **Stats**: Per-term `docFreq` and `totalTermFreq`, run-length encoded for singletons
   (`df==1, ttf==1`). Written by `StatsWriter`.

5. **Metadata**: Per-term postings metadata from `PostingsWriterBase.encodeTerm()`.
   Delta-encoded file pointers. First entry is absolute, rest are relative.

For inner blocks (containing sub-block references), sub-block entries additionally encode
`writeVLong(startFP - block.fp)` — a backward file pointer delta to the child block.

### 7.4 Trie Index (.tip)

Each field has a trie (built by `TrieBuilder`) that maps term prefixes to `.tim` block
file pointers. The trie output contains:
- `fp`: File pointer into `.tim`
- `hasTerms`: Whether this block contains actual terms (not just sub-block pointers)
- `floorData`: For floor blocks, encodes `(numFloorBlocks - 1)`, then for each floor:
  `floorLeadByte` + `writeVLong((subFP - fp) << 1 | hasTerms)`

The trie is saved via `root.index.save(metaOut, indexOut)` — the metadata goes to
`.tmd` and the trie data to `.tip`.

### 7.5 Field Metadata (.tmd)

Written in `TermsWriter.finish()` and `Lucene103BlockTreeTermsWriter.close()`:

Per field:
- `writeVInt(fieldInfo.number)`
- `writeVLong(numTerms)`
- `writeVLong(sumTotalTermFreq)` (if field has freqs)
- `writeVLong(sumDocFreq)`
- `writeVInt(docsSeen.cardinality())`
- `writeBytesRef(firstPendingTerm)` — min term
- `writeBytesRef(lastPendingTerm)` — max term
- Trie index root metadata

File footer:
- `writeVInt(numFields)`
- Each field's metadata blob
- `writeLong(indexOut.getFilePointer())` — `.tip` length
- `writeLong(termsOut.getFilePointer())` — `.tim` length

## 8. Memory Lifecycle

### 8.1 Allocation Phase (During Indexing)

Memory grows throughout document indexing within a segment:

- **ByteBlockPool**: Allocates 32 KB blocks on demand. All term byte streams and term
  text share these blocks.
- **IntBlockPool**: Allocates 8192-int blocks. Stores per-term stream address pointers.
- **ParallelPostingsArray**: Starts at size 2, grows geometrically. Each entry costs
  `BYTES_PER_POSTING` (12 base) plus field-specific arrays (8-20 additional bytes).
- **BytesRefHash**: Maintains a hash table over terms. Grows independently.
- **ByteSlicePool**: Slices within `ByteBlockPool` grow from 5 to 200 bytes per
  allocation level. High-frequency terms accumulate many slices.

Memory accounting goes through `Counter bytesUsed` to the `DocumentsWriter` flush policy.

### 8.2 Peak Memory

Peak memory occurs at flush time, because:
1. All byte pools are still fully allocated (no compaction during indexing).
2. `sortTerms()` allocates a sorted int[] array for each field.
3. The codec writer (`Lucene103BlockTreeTermsWriter`) allocates its own buffers.
4. `FreqProxFields` adapter creates `ByteSliceReader` instances per term iteration.

The byte pools are **not** freed until `TermsHash.reset()` is called after flush completes.

### 8.3 Deallocation Phase

`TermsHash.reset()` (line 71):
- `intPool.reset(false, false)` — drops all int blocks without zeroing
- `bytePool.reset(false, false)` — drops all byte blocks without zeroing

Each `TermsHashPerField.reset()`:
- `bytesHash.clear(false)` — clears the hash table and nulls the postings array via
  `PostingsBytesStartArray.clear()`, which decrements `bytesUsed`
- `sortedTermIDs = null`

## 9. Competitive Impact Data

### 9.1 CompetitiveImpactAccumulator

`CompetitiveImpactAccumulator` tracks `(freq, norm)` pairs that could produce
competitive scores. It maintains:

- `maxFreqs[256]`: An array mapping byte-range norms (the common case, -128..127 mapped
  to unsigned 0..255) to the maximum frequency seen for that norm value.
- `otherFreqNormPairs`: A `TreeSet<Impact>` for norm values outside byte range (rare
  with default similarity).

### 9.2 Accumulation During Encoding

In `Lucene103PostingsWriter.startDoc()` (line 259):
```java
level0FreqNormAccumulator.add(termDocFreq, norm);
```

The norm is read from `NormsProducer` via `norms.advanceExact(docID)`.

### 9.3 Impact Pruning

`getCompetitiveFreqNormPairs()` (line 103) produces a pruned, sorted list:
- Iterates norms in unsigned order (0..255).
- Only keeps entries where `maxFreq > maxFreqForLowerNorms` — meaning this norm value
  has a higher max frequency than all lower norm values.
- This produces a Pareto-optimal frontier: each entry has either a higher freq or a
  higher norm than any other entry.

### 9.4 Impact Serialization

`writeImpacts(Collection<Impact> impacts, DataOutput out)` (line 537):

Delta-encoded pairs:
- `freqDelta = impact.freq - previous.freq - 1`
- `normDelta = impact.norm - previous.norm - 1`
- If `normDelta == 0`: `writeVInt(freqDelta << 1)` — common case, norm increments by 1
- Otherwise: `writeVInt((freqDelta << 1) | 1)` then `writeZLong(normDelta)`

### 9.5 Two-Level Impact Structure

- **Level 0** (`level0FreqNormAccumulator`): Accumulated per block of 128 docs. Written
  as skip data before each full block. Cleared after each block.
- **Level 1** (`level1CompetitiveFreqNormAccumulator`): Accumulated across 32 blocks
  (4,096 docs) by merging each level-0 accumulator via `addAll()`. Written as level-1
  skip data. Cleared after each level-1 group.

At search time, these impacts allow the scorer to skip blocks/groups whose maximum
possible score cannot beat the current competitive threshold.
