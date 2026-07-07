# Doc Values Indexing Path

## 1. Files Produced

| File | Extension | Codec Name | Purpose |
|---|---|---|---|
| Doc Values Data | `.dvd` | `Lucene90DocValuesData` | Encoded per-document values, IndexedDISI bitsets, terms dictionaries, addresses |
| Doc Values Metadata | `.dvm` | `Lucene90DocValuesMetadata` | Per-field metadata: offsets into `.dvd`, compression parameters, value statistics |

Both files are opened in `Lucene90DocValuesConsumer` constructor and written with `CodecUtil.writeIndexHeader`. On `close()`, meta writes an EOF marker (`int -1`) followed by `CodecUtil.writeFooter`; data writes only `CodecUtil.writeFooter`.

**Format version**: `Lucene90DocValuesFormat.VERSION_CURRENT = 0`.

## 2. Doc Values Types

Defined in `DocValuesType` enum, encoded as single bytes in `.dvm`:

| Type | Byte | Data Stored | Cardinality |
|---|---|---|---|
| `NUMERIC` | `0` | One `long` per document | Single-valued |
| `BINARY` | `1` | One `byte[]` per document | Single-valued |
| `SORTED` | `2` | One `byte[]` per document, deduplicated via ordinals | Single-valued |
| `SORTED_NUMERIC` | `4` | Multiple `long` values per document | Multi-valued |
| `SORTED_SET` | `3` | Multiple `byte[]` values per document, deduplicated via ordinals | Multi-valued |

## 3. In-Memory Buffering

Each doc values type has a dedicated `DocValuesWriter<T>` subclass that buffers values in RAM during indexing.

### NumericDocValuesWriter

| Field | Type | Purpose |
|---|---|---|
| `pending` | `PackedLongValues.Builder` (delta-packed, COMPACT) | Stream of one `long` value per document |
| `docsWithField` | `DocsWithFieldSet` | Tracks which doc IDs have a value |
| `finalValues` | `PackedLongValues` | Built from `pending` at flush time |

**addValue(docID, value)**: Appends value to `pending`, adds docID to `docsWithField`. Enforces single-valued (docID must be strictly increasing).

### BinaryDocValuesWriter

| Field | Type | Purpose |
|---|---|---|
| `bytes` | `PagedBytes` (4 KB blocks, `BLOCK_BITS=12`) | Raw byte content of all values, concatenated |
| `bytesOut` | `DataOutput` | Output handle to `bytes` |
| `lengths` | `PackedLongValues.Builder` (delta-packed, COMPACT) | Length of each value |
| `docsWithField` | `DocsWithFieldSet` | Tracks which doc IDs have a value |
| `maxLength` | `int` | Maximum value length seen |
| `finalLengths` | `PackedLongValues` | Built from `lengths` at flush time |

**addValue(docID, value)**: Writes `value.bytes` to `bytesOut`, appends `value.length` to `lengths`. Maximum value length is `ArrayUtil.MAX_ARRAY_LENGTH`.

### SortedDocValuesWriter

| Field | Type | Purpose |
|---|---|---|
| `hash` | `BytesRefHash` (backed by `ByteBlockPool`) | Deduplicates byte values, assigns term IDs |
| `pending` | `PackedLongValues.Builder` (delta-packed, COMPACT) | Stream of term IDs (one per document) |
| `docsWithField` | `DocsWithFieldSet` | Tracks which doc IDs have a value |
| `finalOrds` | `PackedLongValues` | Built from `pending` at flush |
| `finalSortedValues` | `int[]` | Sorted-ord-to-termID mapping from `hash.sort()` |
| `finalOrdMap` | `int[]` | termID-to-sorted-ord mapping (inverse of `finalSortedValues`) |

**addValue(docID, value)**: Calls `hash.add(value)` to get or assign a term ID. Appends term ID to `pending`. On first insertion of a new unique value, reserves `2 * Integer.BYTES` for future rehash + ordMap overhead. Maximum value length is `BYTE_BLOCK_SIZE - 2`.

### SortedNumericDocValuesWriter

| Field | Type | Purpose |
|---|---|---|
| `pending` | `PackedLongValues.Builder` (delta-packed, COMPACT) | Stream of all values across all documents |
| `pendingCounts` | `PackedLongValues.Builder` (delta-packed, COMPACT) | Value count per document (lazily created) |
| `docsWithField` | `DocsWithFieldSet` | Tracks which doc IDs have a value |
| `currentValues` | `long[]` (initial size 8) | Temporary buffer for current document's values |
| `currentUpto` | `int` | Number of values accumulated for current document |

**addValue(docID, value)**: If `docID != currentDoc`, calls `finishCurrentDoc()` which sorts `currentValues[0..currentUpto)` and flushes them to `pending`. The `pendingCounts` builder is lazily initialized only when a document has count != 1 (optimization for the common single-valued case). When all documents have exactly one value, `pendingCounts` stays null and flush delegates to `NumericDocValuesWriter.getDocValuesProducer` via `DocValues.singleton()`.

### SortedSetDocValuesWriter

| Field | Type | Purpose |
|---|---|---|
| `hash` | `BytesRefHash` (backed by `ByteBlockPool`) | Deduplicates byte values, assigns term IDs |
| `pending` | `PackedLongValues.Builder` (packed, COMPACT) | Stream of all deduplicated term IDs |
| `pendingCounts` | `PackedLongValues.Builder` (delta-packed, COMPACT) | Unique term count per document (lazily created) |
| `docsWithField` | `DocsWithFieldSet` | Tracks which doc IDs have a value |
| `currentValues` | `int[]` (initial size 8) | Temporary buffer for current document's term IDs |
| `currentUpto` | `int` | Number of term IDs accumulated for current document |
| `maxCount` | `int` | Maximum unique term count for any single document |
| `finalOrds` | `PackedLongValues` | Built from `pending` at flush |
| `finalOrdCounts` | `PackedLongValues` | Built from `pendingCounts` at flush |
| `finalSortedValues` | `int[]` | Sorted-ord-to-termID mapping |
| `finalOrdMap` | `int[]` | termID-to-sorted-ord mapping |

**addValue(docID, value)**: Gets term ID from `hash.add(value)`. Appends to `currentValues`. On `finishCurrentDoc()`, sorts `currentValues`, deduplicates (skips consecutive equal term IDs), appends unique term IDs to `pending`, and records count. Like `SortedNumericDocValuesWriter`, `pendingCounts` is lazily created only when count != 1. When all documents have exactly one value, flush delegates to `SortedDocValuesWriter.getDocValuesProducer` via `DocValues.singleton()`.

## 4. DocsWithFieldSet

`DocsWithFieldSet` (`index/DocsWithFieldSet.java`) tracks which documents have a value for a given field. It uses an adaptive encoding strategy:

**Dense mode** (default): When documents are added in consecutive order starting from 0 (i.e., `docID == cardinality` on every `add()`), no bit set is allocated. The `set` field stays null and only `cardinality` is tracked.

**Sparse mode**: When a gap is detected (`docID != cardinality`), a `FixedBitSet` is allocated and initialized with bits `[0, cardinality)` set, then the new docID is set. Subsequent adds go through the bit set path, growing it as needed via `FixedBitSet.ensureCapacity`.

**Iterator**:
- Dense: `DocIdSetIterator.all(cardinality)` -- iterates 0..cardinality-1
- Sparse: `BitSetIterator(set, cardinality)` -- iterates set bits

**Memory**: Dense mode costs zero beyond the base object. Sparse mode costs `FixedBitSet.ramBytesUsed()`.

## 5. Flush Sequence

Flush is triggered by `DocValuesWriter.flush(SegmentWriteState, Sorter.DocMap, DocValuesConsumer)`. Each writer:

1. Builds final packed values from their builders (if not already done)
2. Computes sort mappings if `sortMap != null` (index sorting)
3. Wraps data in a `DocValuesProducer` (anonymous `EmptyDocValuesProducer` subclass)
4. Calls the appropriate `DocValuesConsumer.addXxxField(fieldInfo, producer)`

### NumericDocValuesWriter.flush()

1. `finalValues = pending.build()`
2. Creates `EmptyDocValuesProducer` whose `getNumeric()` returns `BufferedNumericDocValues(finalValues, docsWithField.iterator())`
3. Calls `dvConsumer.addNumericField(fieldInfo, producer)`

### BinaryDocValuesWriter.flush()

1. `bytes.freeze(false)` -- freezes the PagedBytes
2. `finalLengths = lengths.build()`
3. Creates `EmptyDocValuesProducer` whose `getBinary()` returns `BufferedBinaryDocValues(finalLengths, maxLength, bytes.getDataInput(), docsWithField.iterator())`
4. Calls `dvConsumer.addBinaryField(fieldInfo, producer)`

### SortedDocValuesWriter.flush()

1. `finish()`: builds `finalOrds = pending.build()`, computes `finalSortedValues = hash.sort()`, builds `finalOrdMap` (termID-to-sorted-ord inverse mapping)
2. Creates `EmptyDocValuesProducer` whose `getSorted()` returns `BufferedSortedDocValues(hash, finalOrds, finalSortedValues, finalOrdMap, docsWithField.iterator())`
3. Calls `dvConsumer.addSortedField(fieldInfo, producer)`

### SortedNumericDocValuesWriter.flush()

1. `finishCurrentDoc()` to flush the last document's values
2. `values = pending.build()`, `valueCounts = pendingCounts.build()` (or null)
3. **Single-valued optimization**: If `valueCounts == null` (all docs had exactly 1 value), delegates to `NumericDocValuesWriter.getDocValuesProducer()` and wraps result via `DocValues.singleton()`
4. Otherwise creates `EmptyDocValuesProducer` whose `getSortedNumeric()` returns `BufferedSortedNumericDocValues`
5. Calls `dvConsumer.addSortedNumericField(fieldInfo, producer)`

### SortedSetDocValuesWriter.flush()

1. `finish()`: builds `finalOrds`, `finalOrdCounts`, `finalSortedValues`, `finalOrdMap`
2. **Single-valued optimization**: If `finalOrdCounts == null`, delegates to `SortedDocValuesWriter.getDocValuesProducer()` and wraps via `DocValues.singleton()`
3. Otherwise creates `EmptyDocValuesProducer` whose `getSortedSet()` returns `BufferedSortedSetDocValues`
4. Calls `dvConsumer.addSortedSetField(fieldInfo, producer)`

## 6. Codec Encoding (Lucene90DocValuesConsumer)

The consumer iterates over the values provided by the producer (pull model -- it can iterate multiple times).

### 6.1 addNumericField

`Lucene90DocValuesConsumer.addNumericField()` (line 136):

1. Writes field number and type byte (`NUMERIC = 0`) to meta
2. Writes skip index if configured (`writeSkipIndex`)
3. Wraps the numeric values as `SortedNumericDocValues` via `DocValues.singleton()`
4. Calls `writeValues(field, producer, ords=false)`

### 6.2 writeValues -- Numeric Encoding Core

`writeValues()` (line 353) is the shared numeric encoding method used by NUMERIC, SORTED (for ordinals), SORTED_NUMERIC, and SORTED_SET (for ordinals). It performs two passes:

**First pass** -- statistics gathering:
- Computes `min`, `max`, `gcd`, `numDocsWithValue`, `numValues`
- Collects up to 256 unique values (for table compression)
- Tracks per-block min/max (`NUMERIC_BLOCK_SIZE = 16384`) to evaluate block-based compression

**Compression strategy selection** (written to meta):

| Strategy | Condition | tablesize in meta | numBitsPerValue |
|---|---|---|---|
| **Const** | `min >= max` (all values identical or no values) | `-1` | `0` |
| **Table** | <= 256 unique values AND table encoding uses fewer bits than delta | `uniqueValues.length` (followed by sorted table entries as longs) | bits to represent table index |
| **Block** | Per-block encoding saves >= 10% over single-block | `-2 - NUMERIC_BLOCK_SHIFT` | `0xFF` |
| **Delta/GCD** | Default | `-1` | bits for `(max - min) / gcd` |

**Metadata written** (in order):
1. **DocsWithField**: offset/length of IndexedDISI in `.dvd`, or `-2`/`-1` sentinel for none/all
2. `jumpTableEntryCount` (short), `denseRankPower` (byte)
3. `numValues` (long)
4. `tablesize` (int) -- encoding strategy indicator
5. `numBitsPerValue` (byte)
6. `min` (long)
7. `gcd` (long)
8. `startOffset` (long) -- pointer into `.dvd`
9. `valuesLength` (long)
10. `jumpTableOffset` (long) -- for block encoding, or `-1`

**Data written to `.dvd`**:

- **Single block** (`writeValuesSingleBlock`): Uses `DirectWriter` with computed bitsPerValue. Each value stored as `(v - min) / gcd` (or table index if table-compressed).
- **Multiple blocks** (`writeValuesMultipleBlocks`): Values written in blocks of `NUMERIC_BLOCK_SIZE` (16384). Each block: 1 byte bitsPerValue, long min, int encodedSize, DirectWriter-encoded deltas. A jump table of block offsets is appended at the end. If block min == max, writes bitsPerValue=0 + min only.

### 6.3 addBinaryField

`Lucene90DocValuesConsumer.addBinaryField()` (line 588):

**First pass** -- writes data and computes statistics:
1. Writes field number and type byte (`BINARY = 1`) to meta
2. Records `dataOffset` in meta
3. Iterates all values, writes raw bytes to `.dvd`, tracks `minLength`, `maxLength`, `numDocsWithField`
4. Writes `dataLength` to meta

**DocsWithField** -- same three-way encoding as numeric (none/all/IndexedDISI)

**Per-field metadata**:
- `numDocsWithField` (int)
- `minLength` (int), `maxLength` (int)

**Variable-length addressing** (only when `maxLength > minLength`):
- Writes cumulative byte addresses using `DirectMonotonicWriter` with `DIRECT_MONOTONIC_BLOCK_SHIFT = 16`
- `numDocsWithField + 1` entries (starting with 0, ending with total byte length)
- Meta records: address start offset, block shift, address data length

**Fixed-length**: When `minLength == maxLength`, no address table is needed; values are accessed by `docOrd * length`.

### 6.4 addSortedField

`Lucene90DocValuesConsumer.addSortedField()` (line 656):

1. Writes field number and type byte (`SORTED = 2`) to meta
2. Calls `doAddSortedField()` which:
   - Wraps ordinals as `SortedNumericDocValues` and calls `writeValues(field, producer, ords=true)` to encode ordinals using the numeric strategies above
   - Calls `addTermsDict()` to write the deduplicated term dictionary

### 6.5 addTermsDict -- Terms Dictionary Encoding

`addTermsDict()` (line 714) writes deduplicated, sorted byte values with LZ4 prefix compression:

**Block structure** (block size = `TERMS_DICT_BLOCK_LZ4_SIZE = 64`):
- First term of each block: written uncompressed as VInt length + raw bytes to `.dvd`
- Subsequent terms: prefix-compressed relative to previous term. Stored in a buffer with the first term as LZ4 dictionary:
  - 1 byte: `min(prefixLen, 15) | (min(suffixLen-1, 15) << 4)`
  - Optional VInt: `prefixLen - 15` (if prefixLen >= 15)
  - Optional VInt: `suffixLen - 16` (if suffixLen >= 16)
  - Raw suffix bytes
- At block boundary: buffer is LZ4-compressed using first term as dictionary, written as VInt uncompressedLength + compressed data

**Address table**: `DirectMonotonicWriter` maps block index to file offset in `.dvd`.

**Metadata written**:
- `size` (VLong) -- total number of unique terms
- `DIRECT_MONOTONIC_BLOCK_SHIFT` (int)
- `maxLength` (int), `maxBlockLength` (int)
- Term data start/length, address data start/length

**Reverse index** (`writeTermsIndex`, line 818):
- Samples every `TERMS_DICT_REVERSE_INDEX_SIZE` (1024) terms
- Writes sort key bytes (minimum distinguishing prefix from previous sample)
- `DirectMonotonicWriter` for offsets into the sort key data
- Enables reverse lookup: given a `BytesRef`, find its ordinal

### 6.6 addSortedNumericField

`Lucene90DocValuesConsumer.addSortedNumericField()` (line 868):

1. Writes field number and type byte (`SORTED_NUMERIC = 4`) to meta
2. Calls `doAddSortedNumericField()` which:
   - Writes skip index if configured
   - Calls `writeValues()` for the numeric values (same encoding as NUMERIC)
   - Writes `numDocsWithField` (int) to meta
   - If multi-valued (`numValues > numDocsWithField`): writes per-document value count addresses using `DirectMonotonicWriter`

### 6.7 addSortedSetField

`Lucene90DocValuesConsumer.addSortedSetField()` (line 928):

1. Writes field number and type byte (`SORTED_SET = 3`) to meta

2. **Single-valued optimization**: Calls `isSingleValued()` to check if all docs have 0 or 1 value. If true, wraps as `SortedDocValues` via `SortedSetSelector.wrap()` and delegates to `doAddSortedField()` (writes type byte `0` for single-valued).

3. **Multi-valued path**:
   - Wraps ordinals as `SortedNumericDocValues` and calls `doAddSortedNumericField(ords=true)` which writes type byte `1` for multi-valued, encodes ordinals + per-doc counts
   - Calls `addTermsDict()` for the term dictionary

## 7. Data Layout

### .dvm (Metadata) File Layout

```
[CodecHeader: "Lucene90DocValuesMetadata", version=0, segmentID, suffix]

For each field (repeated):
  fieldNumber           : int
  typeByte              : byte (0=NUMERIC, 1=BINARY, 2=SORTED, 3=SORTED_SET, 4=SORTED_NUMERIC)

  [Type-specific metadata -- see below]

EOF marker                : int (-1)
[CodecFooter]
```

### .dvd (Data) File Layout

```
[CodecHeader: "Lucene90DocValuesData", version=0, segmentID, suffix]

For each field (interleaved, offsets recorded in .dvm):
  [Skip index data]           -- if field has skip index
  [IndexedDISI bitset]        -- if sparse (not all/none docs have values)
  [Numeric/ordinal values]    -- DirectWriter encoded blocks
  [Binary data]               -- raw concatenated bytes (BINARY type)
  [Address table]             -- DirectMonotonic (variable-length binary or multi-valued counts)
  [Terms dictionary]          -- LZ4-compressed blocks (SORTED, SORTED_SET)
  [Terms address table]       -- DirectMonotonic block offsets
  [Reverse terms index]       -- sort key samples + DirectMonotonic offsets

[CodecFooter]
```

### Per-Type Metadata Detail

**NUMERIC metadata**:
```
[skipIndex: offset(long), length(long), globalMax(long), globalMin(long), docCount(int), maxDocId(int)]  -- if skip index
docsWithFieldOffset     : long (-2=none, -1=all, or offset into .dvd)
docsWithFieldLength     : long
jumpTableEntryCount     : short
denseRankPower          : byte
numValues               : long
tablesize               : int (-1=delta/const, >0=table entries follow, <-1=block)
[table entries]         : long[] (if tablesize > 0)
numBitsPerValue         : byte
min                     : long
gcd                     : long
valueOffset             : long
valuesLength            : long
jumpTableOffset         : long (-1 if not block-encoded)
```

**BINARY metadata**:
```
dataOffset              : long
dataLength              : long
docsWithFieldOffset     : long
docsWithFieldLength     : long
jumpTableEntryCount     : short
denseRankPower          : byte
numDocsWithField        : int
minLength               : int
maxLength               : int
[if variable-length:]
  addressOffset         : long
  blockShift            : VInt
  [DirectMonotonic meta]
  addressLength         : long
```

**SORTED metadata**:
```
[skipIndex metadata]    -- if skip index
[NUMERIC metadata for ordinals (ords=true)]
termsCount              : VLong
blockShift              : int (DIRECT_MONOTONIC_BLOCK_SHIFT)
[DirectMonotonic meta for term addresses]
maxTermLength           : int
maxBlockLength          : int
termDataOffset          : long
termDataLength          : long
termAddressOffset       : long
termAddressLength       : long
reverseIndexShift       : int
[DirectMonotonic meta for reverse index addresses]
reverseDataOffset       : long
reverseDataLength       : long
reverseAddressOffset    : long
reverseAddressLength    : long
```

**SORTED_SET metadata**:
```
[If single-valued (byte 0):]
  -- same as SORTED metadata

[If multi-valued (byte 1):]
  [skipIndex metadata]
  [NUMERIC metadata for ordinals (ords=true)]
  numDocsWithField      : int
  [if numValues > numDocsWithField:]
    addressOffset       : long
    blockShift          : VInt
    [DirectMonotonic meta for per-doc value counts]
    addressLength       : long
  [Terms dictionary metadata -- same as SORTED]
```

**SORTED_NUMERIC metadata**:
```
[skipIndex metadata]    -- if skip index
[NUMERIC metadata for values]
numDocsWithField        : int
[if numValues > numDocsWithField:]
  addressOffset         : long
  blockShift            : VInt
  [DirectMonotonic meta for per-doc value counts]
  addressLength         : long
```

## 8. Memory Lifecycle

### Allocation

Each `DocValuesWriter` is created by `DefaultIndexingChain` when a field with doc values is first encountered for a segment. The writer's constructor:

1. Allocates its `PackedLongValues.Builder` (delta-packed or packed, COMPACT overhead)
2. Creates a `DocsWithFieldSet` (initially zero-cost if dense)
3. For SORTED/SORTED_SET: creates a `BytesRefHash` backed by a `ByteBlockPool`
4. For BINARY: creates a `PagedBytes` with 4 KB blocks
5. Reports initial `bytesUsed` to the `Counter iwBytesUsed` (shared with IndexWriter's RAM accounting)

Every `addValue()` call updates `bytesUsed` via `updateBytesUsed()`, which computes the delta and reports it to `iwBytesUsed`. This allows IndexWriter to trigger flush when RAM usage exceeds `maxBufferedDocs` or `ramBufferSizeMB`.

### Flush-Time Finalization

At flush, each writer builds immutable structures from its builders:

- **NumericDocValuesWriter**: `finalValues = pending.build()` -- freezes the PackedLongValues
- **BinaryDocValuesWriter**: `bytes.freeze(false)` + `finalLengths = lengths.build()`
- **SortedDocValuesWriter**: `finish()` calls `hash.sort()` to get sorted ordinals, builds `finalOrds = pending.build()`, computes `finalOrdMap[]` (termID-to-sorted-ord)
- **SortedNumericDocValuesWriter**: `finishCurrentDoc()` flushes last doc's values, then `pending.build()` and `pendingCounts.build()`
- **SortedSetDocValuesWriter**: `finish()` calls `hash.sort()`, builds `finalOrds`, `finalOrdCounts`, `finalSortedValues`, `finalOrdMap`

### Ordinal Computation (SORTED and SORTED_SET)

Both sorted types use `BytesRefHash` for deduplication during indexing. At flush:

1. `hash.sort()` returns `int[] finalSortedValues` -- an array where `finalSortedValues[sortedOrd] = termID`
2. The inverse mapping `finalOrdMap[termID] = sortedOrd` is computed in a loop
3. During iteration, raw term IDs from `pending` are remapped through `finalOrdMap` to produce sorted ordinals
4. For `SortedSetDocValuesWriter`, the `BufferedSortedSetDocValues.nextDoc()` remaps all term IDs for a document through `ordMap` and then sorts them, since the original insertion order may not match sorted-ordinal order

### Deallocation

After `flush()` completes and the `DocValuesConsumer` finishes writing, the `DocValuesWriter` instances are no longer referenced by the `DefaultIndexingChain`. The `PackedLongValues`, `BytesRefHash`, `PagedBytes`, and all arrays become eligible for garbage collection. The `iwBytesUsed` counter is not explicitly decremented -- the entire per-segment accounting is discarded when the segment flush completes.

The `Lucene90DocValuesConsumer` holds `IndexOutput` handles to `.dvd` and `.dvm` which are closed in `close()`, after which the consumer itself is discarded.
