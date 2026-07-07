# Points (BKD Tree) Indexing Path

Lucene 10.3.2 indexes multi-dimensional point values using a Block KD-tree (BKD tree). This document covers the complete write path from point collection through BKD tree construction to the final on-disk format.

## 1. Files Produced

The Lucene90 points format writes three files per segment (`Lucene90PointsFormat`):

| Extension | Codec Name | Purpose |
|---|---|---|
| `.kdd` | `Lucene90PointsFormatData` | **Leaf block data** -- packed point values and doc IDs for each leaf node |
| `.kdi` | `Lucene90PointsFormatIndex` | **Inner node index** -- the packed BKD tree structure for navigating to leaves |
| `.kdm` | `Lucene90PointsFormatMeta` | **Per-field metadata** -- BKD config, min/max values, pointers into .kdi and .kdd |

All three files share codec headers/footers. The `.kdm` file is read at segment open time; `.kdi` and `.kdd` are read lazily during queries.

At the end of `.kdm`, after all field entries, a sentinel `int` value of `-1` is written, followed by the file lengths of `.kdi` and `.kdd` (as `long` values), then the codec footer (`Lucene90PointsWriter.finish()`).

## 2. Point Collection During Indexing

### PointValuesWriter -- the in-memory buffer

Each field that indexes points gets a `PointValuesWriter` instance (in `index/PointValuesWriter.java`). It is created with a `Counter` for tracking IndexWriter memory usage and a `FieldInfo` describing the field.

**Constructor** (`PointValuesWriter(Counter, FieldInfo)`):
- Creates a `PagedBytes(12)` (4096-byte pages) for storing packed point values
- Allocates an `int[] docIDs` starting at size 16
- Computes `packedBytesLength = pointDimensionCount * pointNumBytes`

**Adding points** (`addPackedValue(int docID, BytesRef value)`):
1. Validates the value is non-null and has exactly `packedBytesLength` bytes
2. Grows `docIDs` array if needed (via `ArrayUtil.grow`)
3. Writes the raw packed value bytes into the `PagedBytes` output stream
4. Records `docIDs[numPoints] = docID`
5. Tracks distinct doc count (`numDocs`) by detecting doc ID transitions
6. Increments `numPoints`
7. Reports memory delta to the IndexWriter's `Counter` for both the docIDs array growth and PagedBytes growth

Key detail: a single document can contribute multiple points to the same field (e.g., multi-valued fields). The `docIDs` array has one entry per point, not per document.

## 3. In-Memory Storage

### During collection (PointValuesWriter)

| Data | Storage | Layout |
|---|---|---|
| Packed point values | `PagedBytes` (paged byte array, 4KB pages) | Contiguous stream of fixed-length `packedBytesLength` byte arrays |
| Document IDs | `int[] docIDs` | One int per point, indexed by point ordinal |
| Point count | `int numPoints` | Total points buffered |
| Doc count | `int numDocs` | Distinct documents with points |

The `PagedBytes` stores all point values in insertion order. Value `i` is at offset `i * packedBytesLength`. The `PagedBytes.Reader` (created at flush time via `bytes.freeze(false)`) provides random access via `fillSlice()` and `getByte()`.

### During BKD construction (BKDWriter)

BKDWriter uses two different storage strategies depending on the expected point count:

| Strategy | Class | Condition | Storage |
|---|---|---|---|
| **Heap** | `HeapPointWriter` | `totalPointCount <= maxPointsSortInHeap` | Single `byte[]` array of size `bytesPerDoc * count` |
| **Offline** | `OfflinePointWriter` | `totalPointCount > maxPointsSortInHeap` | Temp file on disk via `Directory.createTempOutput` |

The threshold `maxPointsSortInHeap` is computed as:
```
maxPointsSortInHeap = (int)((maxMBSortInHeap * 1024 * 1024) / bytesPerDoc)
```
where `maxMBSortInHeap` defaults to `16.0` MB and `bytesPerDoc = packedBytesLength + 4` (point bytes + doc ID).

**HeapPointWriter** stores points in a single flat `byte[]` block. Each point occupies `bytesPerDoc` bytes: the packed value followed by the doc ID in big-endian format. Random access to point `i` is at offset `i * bytesPerDoc`.

**OfflinePointWriter** writes points to a temporary file. Each point is written as: packed value bytes followed by `Integer.reverseBytes(docID)` (big-endian for lexicographic sorting). The file gets a codec footer on close.

## 4. Flush to BKD

### PointValuesWriter.flush()

The `flush(SegmentWriteState, Sorter.DocMap, PointsWriter)` method bridges collection to codec writing:

1. **Freezes the PagedBytes** into a `PagedBytes.Reader` for random access
2. **Creates a MutablePointTree** anonymous implementation backed by:
   - An `int[] ords` array (initially `[0, 1, 2, ..., numPoints-1]`) for indirection
   - An `int[] temp` array (lazily allocated) for the save/restore protocol
   - `getValue(i)` reads from `bytesReader` at offset `packedBytesLength * ords[i]`
   - `getDocID(i)` returns `docIDs[ords[i]]`
   - `swap(i, j)` swaps entries in the `ords` array (not the underlying data)
   - `getByteAt(i, k)` reads a single byte at position `packedBytesLength * ords[i] + k`
3. If a sort map exists (index sorting), wraps the tree in `MutableSortingPointValues` which remaps doc IDs through `docMap.oldToNew()`
4. **Creates a PointsReader** wrapper that returns the tree from `getValues(fieldName).getPointTree()`
5. Calls `writer.writeField(fieldInfo, reader)` on the codec's `PointsWriter`

### Lucene90PointsWriter.writeField()

`writeField(FieldInfo, PointsReader)` in `Lucene90PointsWriter`:

1. Gets the `PointTree` from the reader
2. Creates a `BKDConfig` with the field's dimension count, index dimension count, bytes per dimension, and `maxPointsInLeafNode`
3. Creates a `BKDWriter` with the segment's `maxDoc`, the `Directory`, segment name, config, `maxMBSortInHeap`, point count, and BKD version
4. **If the tree is a MutablePointTree** (which it always is from flush): calls `writer.writeField(metaOut, indexOut, dataOut, fieldName, mutablePointTree)` -- this is the fast path that avoids copying points into BKDWriter's own buffer
5. **Otherwise** (merge path): visits all points via `visitDocValues`, calling `writer.add(packedValue, docID)` for each, then calls `writer.finish(metaOut, indexOut, dataOut)`
6. If the finalizer is non-null, writes `fieldInfo.number` as an `int` to metaOut, then runs the finalizer (which writes the BKD index)

## 5. BKD Tree Construction

### Two code paths based on dimension count

BKDWriter dispatches on `config.numDims()`:

| Dimensions | Method | Strategy |
|---|---|---|
| 1D | `writeField1Dim()` | Sort all points, write leaves sequentially via `OneDimensionBKDWriter` |
| N-D (2+) | `writeFieldNDims()` | Recursive partitioning in-place on the `MutablePointTree` |

There is also a third path for the `add()`/`finish()` flow (used during merges with N-D points), which uses `BKDRadixSelector` for offline partitioning.

### 1D Path: writeField1Dim()

1. **Sort** all points using `MutablePointTreeReaderUtils.sort()` on the single dimension
2. Create an `OneDimensionBKDWriter`
3. **Visit** all sorted points via `visitDocValues`, adding each to `OneDimensionBKDWriter.add()`
4. `OneDimensionBKDWriter` accumulates points into fixed-size leaf buffers:
   - `byte[] leafValues` of size `maxPointsInLeafNode * packedBytesLength`
   - `int[] leafDocs` of size `maxPointsInLeafNode`
5. When `leafCount == maxPointsInLeafNode`, writes a leaf block via `writeLeafBlock()`
6. Tracks `leafBlockFPs` (file pointers) and `leafBlockStartValues` (first value of each block, for building the index)
7. After all points are added, `finish()` builds a `BKDTreeLeafNodes` from the accumulated FPs and start values, and returns a finalizer that calls `writeIndex()`

### N-D Path: writeFieldNDims()

1. Compute `numLeaves = ceil(pointCount / maxPointsInLeafNode)`
2. Allocate `splitPackedValues[numSplits * bytesPerDim]`, `splitDimensionValues[numSplits]`, `leafBlockFPs[numLeaves]`
3. Compute min/max packed values across all points
4. Call the recursive `build()` method (the MutablePointTree variant)

### Recursive build() for MutablePointTree

`build(leavesOffset, numLeaves, reader, from, to, out, minPackedValue, maxPackedValue, parentSplits, splitPackedValues, splitDimensionValues, leafBlockFPs, spareDocIds)`:

**Leaf node** (numLeaves == 1):
1. Compute common prefix lengths across all dimensions by comparing all points
2. Find the dimension with the fewest unique bytes at the common prefix boundary (`sortedDim`) -- this is the "compression dimension"
3. Sort the points by `sortedDim` using `MutablePointTreeReaderUtils.sortByDim()`
4. Compute leaf cardinality
5. Record `leafBlockFPs[leavesOffset] = out.getFilePointer()`
6. Write the leaf block: doc IDs, common prefixes, packed values

**Inner node** (numLeaves > 1):
1. **Choose split dimension** via `split(minPackedValue, maxPackedValue, parentSplits)`:
   - First, check for underrepresented dimensions (split count < maxSplits/2) that have non-equal min/max -- return that dimension to ensure balanced indexing
   - Otherwise, pick the dimension with the largest value range (max - min)
2. Compute `numLeftLeafNodes` using `getNumLeftLeafNodes()` which creates a partially balanced tree:
   - Find the last full level of the binary tree
   - Distribute overflow leaves to the left side first
3. Compute `mid = from + numLeftLeafNodes * maxPointsInLeafNode`
4. **Partition** using `MutablePointTreeReaderUtils.partition()` -- rearranges points so that all points in `[from, mid)` would go to the left subtree
5. Record the split value (the value at position `mid` in the split dimension)
6. Recurse on left `[from, mid)` and right `[mid, to)` subtrees

### Recursive build() for offline path (add/finish)

Used when points were added via `add()` and need offline partitioning:

1. Uses `BKDRadixSelector` for partitioning -- a radix-based selection algorithm that works with both heap and disk-based point storage
2. At leaf nodes, if the data is still in an `OfflinePointWriter`, it is pulled back into a `HeapPointWriter` via `switchToHeap()`
3. Uses `radixSelector.select()` to partition points and produce left/right `PathSlice` objects
4. `BKDRadixSelector` uses histogram-based counting over byte values for offline data, and MSB radix sort for heap data

### BKDRadixSelector

`BKDRadixSelector` performs nth-element selection (partitioning) along a chosen dimension:
- Uses a 256-bucket histogram over the byte at the current radix position
- For offline data, streams through the file building the histogram, then splits into left/right temp files
- For heap data, delegates to `IntroSelector` (intro-select with fallback)
- Tie-breaking: when dimension bytes are equal, compares data-only dimensions, then doc IDs
- Selection is done on `bytesPerDim + (numDims - numIndexDims) * bytesPerDim + Integer.BYTES` bytes total

## 6. Data Layout

### .kdd -- Leaf Block Data

Each leaf block contains (written by `writeLeafBlockDocs`, `writeCommonPrefixes`, `writeLeafBlockPackedValues`):

```
LeafBlock:
  VInt:   count                       // number of points in this leaf
  bytes:  docIDs                      // encoded by DocIdsWriter (see below)
  for each dimension d:
    VInt:  commonPrefixLength[d]      // prefix shared by all values in dim d
    bytes: commonPrefix[d]            // the shared prefix bytes (commonPrefixLength[d] bytes)
  byte:   leafEncoding                // -1 = all equal, -2 = low cardinality, 0..N = sorted dim (high cardinality)
  bytes:  packedValues                // encoding depends on leafEncoding
```

**leafEncoding** determines how the suffix bytes are stored:

- **-1 (all equal)**: All points in the block are identical. No additional data.
- **-2 (low cardinality)**: Used when run-length encoding is more expensive than per-unique-value encoding. For N-D, actual min/max bounds per index dimension are written first. Then for each unique value: `VInt(cardinality)` followed by the suffix bytes for all dimensions.
- **sortedDim (0..N) (high cardinality)**: The block is sorted by `sortedDim`. For N-D, actual bounds per index dimension are written first. Then run-length encoded: `byte(prefixByte), byte(runLength)`, followed by the remaining suffix bytes for each point in the run.

**DocIdsWriter encoding** (`DocIdsWriter.writeDocIds`):

| Flag byte | Meaning |
|---|---|
| `-2` (CONTINUOUS_IDS) | Doc IDs are sequential: writes `min` as int |
| `-1` (BITSET_IDS) | Sparse: writes as a `FixedBitSet` |
| `16` (DELTA_BPV_16) | Delta-coded with 16 bits per value |
| `21` (BPV_21) | Packed at 21 bits per value |
| `24` (BPV_24) | Packed at 24 bits per value |
| `32` (BPV_32) | Raw 32-bit integers |

### .kdi -- Packed Index

The BKD tree index is written as a compact byte array by `packIndex()`/`recursePackIndex()`. It encodes a semi-balanced binary tree using prefix-coded split values:

**For each inner node** (recursively, pre-order):
```
InnerNode:
  VLong: deltaFP              // delta from parent's left-most leaf FP (omitted for left children)
  VInt:  code                  // encodes (firstDiffByteDelta, prefix, splitDim) as:
                               //   (firstDiffByteDelta * (1 + bytesPerDim) + prefix) * numIndexDims + splitDim
  bytes: suffixBytes           // remaining split value bytes after prefix+1 (suffix-1 bytes)
  VInt:  leftNumBytes          // byte size of left subtree (omitted when left child is a leaf)
  [left subtree]
  [right subtree]
```

**For leaf nodes**:
- Left leaves at the leftmost position of any subtree: 0 bytes (FP is implicit)
- Right leaves: `VLong(deltaFP)` only

The split values are prefix-coded against the previous split value in the same dimension, using a "negative delta" scheme for left children vs right children.

### .kdm -- Per-field Metadata

For each field with points (written by `writeIndex()`):

```
FieldEntry:
  int:     fieldNumber                // from Lucene90PointsWriter
  Header:  BKD codec header           // "BKD", version 10
  VInt:    numDims                     // total dimensions (data + index)
  VInt:    numIndexDims                // dimensions used for tree splitting
  VInt:    maxPointsInLeafNode         // typically 512
  VInt:    bytesPerDim                 // bytes per dimension value
  VInt:    numLeaves                   // number of leaf nodes
  bytes:   minPackedValue              // global min across all index dimensions (numIndexDims * bytesPerDim)
  bytes:   maxPackedValue              // global max across all index dimensions
  VLong:   pointCount                  // total number of indexed points
  VInt:    docCount                    // number of distinct documents with points
  VInt:    packedIndexLength           // byte length of the packed index in .kdi
  long:    dataStartFP                 // file pointer to first leaf block in .kdd
  long:    indexStartFP                // file pointer to packed index in .kdi
```

After all fields:
```
int:  -1                              // sentinel
long: indexFileLength                  // .kdi file length (for checksum verification)
long: dataFileLength                   // .kdd file length (for checksum verification)
Footer: codec footer
```

## 7. Memory Lifecycle

### Phase 1: Point Collection (PointValuesWriter)

Memory grows linearly with point count:
- `PagedBytes`: ~`numPoints * packedBytesLength` bytes in 4KB pages
- `int[] docIDs`: ~`numPoints * 4` bytes (with amortized growth)
- All tracked by the IndexWriter's `Counter` for RAM accounting and flush triggers

### Phase 2: Flush -- MutablePointTree creation

At flush time, `PointValuesWriter.flush()` creates:
- `int[] ords` of size `numPoints` (for indirection)
- `int[] temp` of size `numPoints` (lazily allocated during sorting)
- The `PagedBytes.Reader` reuses the existing pages (no copy)

Peak memory at this point: ~`numPoints * (packedBytesLength + 4 + 4 + 4)` bytes (values + docIDs + ords + temp).

### Phase 3: BKD Tree Construction

**MutablePointTree path** (flush): No additional bulk storage. The tree is built by rearranging `ords` in-place. Leaf blocks are written directly from the MutablePointTree. Temporary allocations:
- `byte[] splitPackedValues` = `numSplits * bytesPerDim`
- `byte[] splitDimensionValues` = `numSplits`
- `long[] leafBlockFPs` = `numLeaves * 8`
- `int[] spareDocIds` = `maxPointsInLeafNode * 4`
- `FixedBitSet docsSeen` = `maxDoc / 8` bits

**add()/finish() path** (offline): If `totalPointCount > maxPointsSortInHeap`, points are spilled to a temp file. During recursive partitioning:
- `BKDRadixSelector` may create additional temp files for left/right partitions
- At leaf nodes, data is pulled back into heap via `switchToHeap()` (up to `maxPointsInLeafNode` points)
- Temp files are deleted as partitions are consumed

### Phase 4: Index Writing

After all leaf blocks are written to `.kdd`, the `writeIndex()` finalizer:
1. Calls `packIndex()` which recursively builds the index into `List<byte[]>` blocks
2. Compacts blocks into a single `byte[] packedIndex`
3. Writes metadata to `.kdm` and the packed index to `.kdi`

Memory for the packed index is proportional to `numLeaves`, not `numPoints`.

### Summary of peak memory

| Path | Peak memory | When |
|---|---|---|
| Flush (MutablePointTree) | ~`numPoints * (packedBytesLength + 12)` | During sort (ords + temp arrays) |
| add/finish (heap) | ~`numPoints * bytesPerDoc` | HeapPointWriter holds all points |
| add/finish (offline) | ~`maxMBSortInHeap` (default 16MB) | Points on disk, only leaf-sized chunks in memory |

## 8. Multi-dimensional Points

### Data dimensions vs index dimensions

BKDConfig distinguishes:
- **numDims**: Total number of dimensions stored (up to 16)
- **numIndexDims**: Dimensions used for tree splitting (up to 8, must be <= numDims)

Extra data-only dimensions (`numDims - numIndexDims`) are stored in leaf blocks but not used for tree partitioning. This supports cases like LatLonPoint (2 index dims) or range fields (e.g., IntRange with 2 data dims, 1 index dim per range endpoint pair).

### How multiple dimensions affect tree construction

1. **Split dimension selection** (`BKDWriter.split()`):
   - Checks for underrepresented dimensions first (any dimension split fewer than `maxSplits/2` times)
   - Otherwise picks the dimension with the largest value range
   - This ensures all index dimensions get reasonable coverage in the tree

2. **Bounds recomputation**: For `numIndexDims > 2`, the algorithm periodically recomputes exact bounds (every `SPLITS_BEFORE_EXACT_BOUNDS = 4` splits) because inherited bounds from ancestors become loose.

3. **Leaf compression dimension** (`sortedDim`): Within each leaf block, the algorithm picks the dimension with the fewest unique byte values at the common prefix boundary. This minimizes entropy and improves compression regardless of which dimension was used for tree splitting.

4. **Packed value layout**: All dimensions are concatenated into a single `byte[]` of length `numDims * bytesPerDim`. Index dimensions come first, followed by data-only dimensions. Within each leaf block, common prefixes are computed and stored per-dimension.

5. **Actual bounds in N-D leaves**: When `numIndexDims != 1`, leaf blocks in high-cardinality and low-cardinality encoding write per-index-dimension actual min/max suffix bytes. This tightens the bounds beyond what the tree structure implies, enabling more effective pruning during intersection queries.

### BKDConfig constants

| Constant | Value |
|---|---|
| `DEFAULT_MAX_POINTS_IN_LEAF_NODE` | 512 |
| `MAX_DIMS` | 16 |
| `MAX_INDEX_DIMS` | 8 |
| `DEFAULT_MAX_MB_SORT_IN_HEAP` | 16.0 MB |

### Common configurations (pre-cached in BKDConfig)

| numDims | numIndexDims | bytesPerDim | Use case |
|---|---|---|---|
| 1 | 1 | 4 | IntField, FloatField |
| 1 | 1 | 8 | LongField, DoubleField |
| 2 | 2 | 4 | LatLonPoint |
| 1 | 1 | 16 | InetAddressPoint |
| 7 | 4 | 4 | Lucene shapes |

## Class Relationships

```
IndexWriter
  --> PointValuesWriter          (per field, buffers points during indexing)
        --> PagedBytes           (packed values)
        --> int[] docIDs         (one per point)

PointValuesWriter.flush()
  --> MutablePointTree           (anonymous impl with ords[] indirection)
  --> Lucene90PointsWriter.writeField()
        --> BKDWriter
              --> HeapPointWriter   OR  OfflinePointWriter  (only for add/finish path)
              --> BKDRadixSelector  (only for offline partitioning)
              --> OneDimensionBKDWriter  (1D fast path)
              --> DocIdsWriter      (leaf doc ID encoding)
              --> writeIndex()      --> packIndex() --> recursePackIndex()
```

## Key Java Source Files

| File | Path | Purpose |
|---|---|---|
| `PointValuesWriter` | `index/PointValuesWriter.java` | Buffers points per field during indexing |
| `PointsWriter` | `codecs/PointsWriter.java` | Abstract codec write API |
| `PointsFormat` | `codecs/PointsFormat.java` | Abstract format factory |
| `MutablePointTree` | `codecs/MutablePointTree.java` | Reorderable point sequence for flush optimization |
| `Lucene90PointsWriter` | `codecs/lucene90/Lucene90PointsWriter.java` | Concrete writer, creates BKDWriter |
| `Lucene90PointsFormat` | `codecs/lucene90/Lucene90PointsFormat.java` | Format factory, version mapping |
| `Lucene90PointsReader` | `codecs/lucene90/Lucene90PointsReader.java` | Reader (reads .kdm at open, lazy .kdi/.kdd) |
| `BKDWriter` | `util/bkd/BKDWriter.java` | Core BKD tree builder |
| `BKDConfig` | `util/bkd/BKDConfig.java` | Dimension counts, bytes per dim, leaf size |
| `BKDRadixSelector` | `util/bkd/BKDRadixSelector.java` | Offline radix-based partitioning |
| `HeapPointWriter` | `util/bkd/HeapPointWriter.java` | In-memory point storage (flat byte array) |
| `OfflinePointWriter` | `util/bkd/OfflinePointWriter.java` | Disk-based point storage (temp files) |
| `DocIdsWriter` | `util/bkd/DocIdsWriter.java` | Leaf doc ID encoding strategies |
| `MutablePointTreeReaderUtils` | `util/bkd/MutablePointTreeReaderUtils.java` | Sort/partition utilities for MutablePointTree |
