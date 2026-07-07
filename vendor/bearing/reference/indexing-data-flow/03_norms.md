# Norms Indexing Path (Lucene 10.3.2)

## 1. Files Produced

| Extension | Codec Name | Purpose |
|---|---|---|
| `.nvd` | `Lucene90NormsData` | Per-document norm values and docs-with-field bitsets |
| `.nvm` | `Lucene90NormsMetadata` | Per-field metadata: offsets, encoding type, doc counts |

Both files are produced by `Lucene90NormsFormat` / `Lucene90NormsConsumer`. The format version is `VERSION_START = 0` (`VERSION_CURRENT = VERSION_START`).

## 2. Norm Computation

### Source: `Similarity.computeNorm(FieldInvertState)`

The default `Similarity.computeNorm()` (lines 153-163) computes a term count, then encodes it as a single byte via `SmallFloat.intToByte4()`:

```java
public long computeNorm(FieldInvertState state) {
    final int numTerms;
    if (state.getIndexOptions() == IndexOptions.DOCS) {
        numTerms = state.getUniqueTermCount();
    } else if (discountOverlaps) {
        numTerms = state.getLength() - state.getNumOverlap();
    } else {
        numTerms = state.getLength();
    }
    return SmallFloat.intToByte4(numTerms);
}
```

Three modes for computing term count:
- **DOCS-only index options**: uses `uniqueTermCount` (number of distinct terms)
- **discountOverlaps=true** (the default): uses `length - numOverlap` (total terms minus zero-position-increment tokens like synonyms)
- **discountOverlaps=false**: uses `length` (total terms including overlaps)

`BM25Similarity` does **not** override `computeNorm` -- it uses this default.

### `FieldInvertState` Fields

Set during token processing in `IndexingChain.PerField`:

| Field | Type | Meaning |
|---|---|---|
| `length` | `int` | Total number of tokens indexed for the field in this document |
| `numOverlap` | `int` | Number of tokens with `positionIncrement == 0` |
| `uniqueTermCount` | `int` | Number of distinct terms |
| `position` | `int` | Last processed term position |
| `maxTermFrequency` | `int` | Highest frequency of any single term |
| `offset` | `int` | End offset of last processed term |

These are reset per-document via `FieldInvertState.reset()`, which sets `length=0`, `numOverlap=0`, `uniqueTermCount=0`, etc.

### `SmallFloat.intToByte4` Encoding

`SmallFloat.intToByte4(int)` encodes a positive integer into a single unsigned byte (stored as `long`). It uses a float-like representation with 4 significant bits:

1. Compute `NUM_FREE_VALUES = 255 - longToInt4(Integer.MAX_VALUE)`. Values below this threshold are stored exactly.
2. For values `i < NUM_FREE_VALUES`: return `i` directly (lossless).
3. For values `i >= NUM_FREE_VALUES`: return `NUM_FREE_VALUES + longToInt4(i - NUM_FREE_VALUES)`, which uses a float-like encoding with 3 mantissa bits and a shift-based exponent.

The `longToInt4` subroutine:
- For values with fewer than 4 significant bits (subnormal): stored directly.
- For values with 4+ bits (normal): keep 3 explicit mantissa bits (the 4th is implicit), encode the shift in higher bits.

Result: a `long` in range [0, 255] where:
- 0 means "no indexed tokens" (special case handled in `PerField.finish()`)
- 1 is the highest-scoring norm (shortest field)
- Higher unsigned values mean longer fields (lower scores)

### Special Case: Zero-Length Fields

In `IndexingChain.PerField.finish()` (lines 1159-1175):
- If `invertState.length == 0`: the field was present but had no indexed tokens, so `normValue = 0`.
- Otherwise: `normValue = similarity.computeNorm(invertState)`. If the similarity returns 0, an `IllegalStateException` is thrown (0 is not a legal norm for non-empty fields).

## 3. In-Memory Storage

### `NormValuesWriter`

Created once per field in `IndexingChain.PerField.setInvertState()` (line 1147), if `fieldInfo.omitsNorms() == false`.

Data structures:
- **`pending`**: `PackedLongValues.Builder` (delta-packed, `PackedInts.COMPACT`) -- stores one `long` norm value per document that has norms.
- **`docsWithField`**: `DocsWithFieldSet` -- tracks which doc IDs have a norm value.
- **`lastDocID`**: `int` -- enforces that each document contributes at most one norm value per field (doc IDs must be strictly increasing).

### Per-Document Addition

`NormValuesWriter.addValue(int docID, long value)`:
1. Validates `docID > lastDocID` (no duplicates).
2. Appends `value` to `pending`.
3. Adds `docID` to `docsWithField`.
4. Updates memory accounting via `iwBytesUsed` counter.

### Memory Accounting

The writer tracks `bytesUsed = pending.ramBytesUsed() + docsWithField.ramBytesUsed()` and updates the shared `iwBytesUsed` counter (an `AtomicLong`-backed `Counter`) on every `addValue` call. This feeds into `IndexWriter`'s RAM buffer tracking for flush decisions.

## 4. Flush Sequence

### Entry Point: `IndexingChain.flush(SegmentWriteState)`

The flush method (line 270) calls `writeNorms(state, sortMap)` as the **first** data write operation (before doc values, points, vectors, stored fields, and postings).

### `IndexingChain.writeNorms(SegmentWriteState, Sorter.DocMap)` (lines 475-505)

1. Check `state.fieldInfos.hasNorms()`. If no fields have norms, skip entirely.
2. Create a `NormsConsumer` via `normsFormat.normsConsumer(state)` -- this opens the `.nvd` and `.nvm` files and writes their codec headers.
3. Iterate over all `FieldInfo` in `state.fieldInfos`:
   - Skip fields where `fi.omitsNorms() == true` or `fi.getIndexOptions() == IndexOptions.NONE`.
   - Call `perField.norms.finish(maxDoc)` (currently a no-op).
   - Call `perField.norms.flush(state, sortMap, normsConsumer)`.
4. Close the `NormsConsumer` (writes EOF marker and footers).

### `NormValuesWriter.flush()` (lines 70-105)

1. Build the `PackedLongValues` from the builder: `values = pending.build()`.
2. If `sortMap != null` (index sorting), sort the doc values accordingly.
3. Call `normsConsumer.addNormsField(fieldInfo, normsProducer)`, passing an anonymous `NormsProducer` that wraps a `BufferedNorms` iterator over the in-memory values.

The `NormsProducer` passed to the codec can be iterated **multiple times** (each call to `getNorms()` creates a fresh `BufferedNorms` iterator). The codec exploits this: it iterates once to find min/max, then again to write the bitset, then again to write the values.

### Post-Flush Norms Read-Back

After writing norms, `IndexingChain.flush()` (lines 333-342) immediately opens a `NormsProducer` (reader) from the just-written files. This reader is passed to the postings flush (`termsHash.flush()`) so that postings can read back norm values for computing impacts (skip data). This is the "read-back" pattern -- norms are written first, then read back for use during postings encoding.

## 5. Codec Encoding

### `Lucene90NormsConsumer.addNormsField(FieldInfo, NormsProducer)` (lines 91-138)

#### Pass 1: Statistics Gathering

Iterate all values to compute:
- `numDocsWithValue`: count of documents that have a norm
- `min`: minimum norm value across all documents
- `max`: maximum norm value across all documents

#### Metadata Write: Docs-With-Field

Three cases for the docs-with-field bitset:

| Condition | `docsWithFieldOffset` | Written to `.nvd` |
|---|---|---|
| `numDocsWithValue == 0` | `-2` | Nothing |
| `numDocsWithValue == maxDoc` | `-1` | Nothing (all docs have norms) |
| Otherwise | File pointer into `.nvd` | `IndexedDISI.writeBitSet()` -- compressed bitset |

For the sparse case, metadata also records `jumpTableEntryCount` and `denseRankPower` from the IndexedDISI write.

#### Value Encoding: `numBytesPerValue`

The `numBytesPerValue(min, max)` method (lines 140-152) selects the minimum byte width:

| Condition | Bytes | Encoding |
|---|---|---|
| `min >= max` (all values identical) | 0 | Constant: store `min` in metadata only |
| Range fits in `[-128, 127]` | 1 | One byte per value |
| Range fits in `[-32768, 32767]` | 2 | Two bytes (short) per value |
| Range fits in `[-2^31, 2^31-1]` | 4 | Four bytes (int) per value |
| Otherwise | 8 | Eight bytes (long) per value |

**Note**: This is pure min/max range encoding, not delta or table encoding. The Lucene90 norms format does not subtract `min` from each value -- it stores raw values at the chosen width.

#### Pass 2 (sparse case): Write IndexedDISI Bitset

If not all docs have norms, re-iterate the values to write the `IndexedDISI` bitset to `.nvd`.

#### Pass 3: Write Values

If `numBytesPerValue > 0`, re-iterate to write each value at the determined byte width to `.nvd` via `writeValues()`.

## 6. Data Layout

### `.nvm` (Metadata) File

```
[IndexHeader: "Lucene90NormsMetadata", version=0, segmentID, segmentSuffix]

For each field with norms:
  Int32:  fieldNumber
  Int64:  docsWithFieldOffset     (-2=none, -1=all, else pointer into .nvd)
  Int64:  docsWithFieldLength     (byte length of IndexedDISI data)
  Int16:  jumpTableEntryCount     (-1 if no DISI written)
  Byte:   denseRankPower          (-1 if no DISI written)
  Int32:  numDocsWithField
  Byte:   numBytesPerValue        (0, 1, 2, 4, or 8)
  Int64:  normsOffset/constValue  (if numBytesPerValue==0: the constant norm value;
                                   otherwise: pointer into .nvd for norm data)

Int32:  -1                        (EOF marker)
[CodecFooter]
```

Per-field metadata size: 4 + 8 + 8 + 2 + 1 + 4 + 1 + 8 = **36 bytes**.

### `.nvd` (Data) File

```
[IndexHeader: "Lucene90NormsData", version=0, segmentID, segmentSuffix]

For each field (interleaved, order matches metadata):
  [Optional: IndexedDISI bitset]  (only if sparse, pointed to by docsWithFieldOffset)
  [Optional: norm values]         (only if numBytesPerValue > 0, pointed to by normsOffset)
    numDocsWithField values, each numBytesPerValue bytes wide

[CodecFooter]
```

When `numBytesPerValue == 0` (constant), nothing is written to `.nvd` for that field's values -- the constant is stored in `.nvm`.

When `numDocsWithValue == maxDoc`, no IndexedDISI bitset is written (all docs implicitly have norms).

## 7. Memory Lifecycle

### Allocation

- **`NormValuesWriter`** created in `IndexingChain.PerField.setInvertState()` -- once per field, when the field is first seen in the segment. This happens lazily (not at segment start).
- The `PackedLongValues.Builder` and `DocsWithFieldSet` grow incrementally as documents are indexed.
- Memory usage is tracked via the `iwBytesUsed` counter, which participates in `IndexWriter`'s RAM buffer flush decisions.

### During Indexing

- Each `addValue` call appends one long to the delta-packed builder and one doc ID to the `DocsWithFieldSet`.
- `PackedLongValues.Builder` (delta-packed, COMPACT) stores values in blocks, using the minimum number of bits per value within each block.

### At Flush

1. `NormValuesWriter.flush()` calls `pending.build()`, which finalizes the `PackedLongValues` (compacts the builder into an immutable, read-only packed structure).
2. The `NormsConsumer` iterates the values (up to 3 passes) and writes them to disk.
3. After `writeNorms` completes, the `NormsConsumer` is closed.
4. The `NormValuesWriter` objects and their `PackedLongValues`/`DocsWithFieldSet` are not explicitly freed -- they become eligible for GC when the `IndexingChain` (and its `PerField` entries) are discarded after the segment flush completes.

### Post-Flush Read-Back

After norms are flushed to disk, `IndexingChain.flush()` opens a `Lucene90NormsProducer` (reader) to read back norms from the just-written files. This reader is used during postings flush for computing score impacts, then closed when the postings flush completes.
