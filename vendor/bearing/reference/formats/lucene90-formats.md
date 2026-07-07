# Lucene 9.0 Formats

Byte-level format documentation for all Lucene 9.0 codec components.

**Source of truth**: Java writer implementation code, not Javadoc. See `encoding-primitives.md` for shared encoding details.

**Endianness rule**: All data writes use `DataOutput.writeInt/writeLong/writeShort` which are **little-endian**. Only `CodecUtil` header/footer fields are big-endian. This applies throughout this document unless explicitly noted otherwise.

---

## Table of Contents

1. [Stored Fields (.fdt, .fdx, .fdm)](#stored-fields)
2. [Doc Values (.dvd, .dvm)](#doc-values)
3. [Norms (.nvd, .nvm)](#norms)
4. [Points/BKD (.kdd, .kdi, .kdm)](#pointsbkd)
5. [Compound (.cfs, .cfe)](#compound)
6. [IndexedDISI](#indexeddisi)
7. [DirectMonotonicWriter](#directmonotonicwriter)
8. [DirectWriter](#directwriter)

---

## Stored Fields

Files: `.fdt` (field data), `.fdx` (field index), `.fdm` (field metadata)

### Constants
```
Codec names:
  .fdt: (compression-specific, e.g. "Lucene90CompressingStoredFields")
  .fdx: "Lucene90FieldsIndex"
  .fdm: "Lucene90FieldsIndexMeta"

VERSION_CURRENT = 1

Field type constants (stored in lower 3 bits of infoAndBits):
  STRING       = 0x00
  BYTE_ARR     = 0x01
  NUMERIC_INT  = 0x02
  NUMERIC_FLOAT  = 0x03
  NUMERIC_LONG   = 0x04
  NUMERIC_DOUBLE = 0x05
  TYPE_BITS = 3
  TYPE_MASK = 0x07

Timestamp compression units:
  SECOND = 1000
  HOUR   = 3600000
  DAY    = 86400000
  SECOND_ENCODING = 0x40
  HOUR_ENCODING   = 0x80
  DAY_ENCODING    = 0xC0
```

Verified: `CompressingStoredFieldsWriter.java`, `StoredFieldsInts.java`

### .fdm (Metadata File)

```
IndexHeader ("Lucene90FieldsIndexMeta" + segmentSuffix)
VInt:   chunkSize (minimum byte size of a chunk)

--- FieldsIndexWriter.finish() metadata ---
Int:    numDocs (LE)
Int:    blockShift (LE)
Int:    totalChunks + 1 (LE, entry count for DirectMonotonic arrays)
Long:   pointer to doc IDs data in .fdx (LE)
        DirectMonotonicWriter metadata for doc IDs (see DirectMonotonicWriter section)
Long:   pointer after docs index in .fdx (LE)
        DirectMonotonicWriter metadata for file pointers
Long:   pointer after file pointers in .fdx (LE)
Long:   maxPointer — end of .fdt data (LE)

VLong:  numChunks (total chunks written)
VLong:  numDirtyChunks (incomplete/forced-flush chunks)
VLong:  numDirtyDocs (total docs in dirty chunks)

CodecFooter
```

Verified: `FieldsIndexWriter.finish()`, `CompressingStoredFieldsWriter.close()`

### .fdx (Index File)

```
IndexHeader ("Lucene90FieldsIndex" + segmentSuffix)

DirectMonotonic data for doc IDs:
  Maps chunk index → starting docID
  Entry count: totalChunks + 1 (last entry = numDocs)

DirectMonotonic data for file pointers:
  Maps chunk index → byte offset in .fdt
  Entry count: totalChunks + 1 (last entry = maxPointer)

CodecFooter
```

Verified: `FieldsIndexWriter.java`

### .fdt (Fields Data File)

```
IndexHeader (compression-specific codec name)

Chunk[0..numChunks-1]:
  VInt:   docBase (starting document ID)
  VInt:   code
          bits 0: sliced (1 = bufferedDocs >= 2 × chunkSize)
          bit 1:  dirtyChunk (1 = forced flush)
          bits 2+: numBufferedDocs (code >> 2)

  numStoredFields[numBufferedDocs]:
    If numBufferedDocs == 1: single VInt
    Else: StoredFieldsInts encoded (see below)

  lengths[numBufferedDocs]:
    Document byte lengths, same StoredFieldsInts encoding

  Compressed document data:
    LZ4 or DEFLATE compressed stream of concatenated raw documents
    If sliced: split into chunkSize sub-blocks, each independently compressed

CodecFooter
```

#### StoredFieldsInts Encoding

For arrays of integers (numStoredFields, lengths):
```
Byte:  format indicator
  0x00 = all values equal → VInt(value)
  0x08 = max ≤ 0xFF     → 8-bit packed values
  0x10 = max ≤ 0xFFFF   → 16-bit packed values
  0x20 = max > 0xFFFF   → 32-bit packed values

Packed values: written in 128-value blocks, interleaved into longs
```

Verified: `CompressingStoredFieldsWriter.writeHeader()`, `StoredFieldsInts.java`

#### Raw Document Layout (before compression)

Each document contains its stored fields in order:
```
For each stored field:
  VLong:  infoAndBits
          lower TYPE_BITS (3): field type (0-5)
          upper bits: field number (>> TYPE_BITS)

  Field value (depends on type):
    STRING:       VInt(byte_length) + UTF-8 bytes
    BYTE_ARR:     VInt(byte_length) + raw bytes
    NUMERIC_INT:  ZInt (zig-zag encoded VInt)
    NUMERIC_FLOAT:  zFloat (variable-length, 1-5 bytes)
    NUMERIC_LONG:   tLong (timestamp-optimized, 1-9 bytes)
    NUMERIC_DOUBLE: zDouble (variable-length, 1-9 bytes)
```

#### zFloat Encoding
```
If value == -0.0f or not representable as int in [-1, 125]:
  If negative zero or high bit set:
    Byte: 0xFF + 4 bytes (int bits, unclear endianness — trace writeZFloat)
  Else:
    Byte: low byte of intBits + 3 more bytes
Else (small integer representable):
  Byte: value + 1 + 0x80  (single byte, range 0x80-0xFE for [-1..125])
```

#### tLong Encoding (timestamp-optimized)
```
Header byte:
  bits 0-4: lower 5 bits of zig-zag encoded value
  bit 5:    continuation flag (1 = more bytes follow)
  bits 6-7: time unit
    0x00 = raw (no division)
    0x40 = seconds (÷ 1000)
    0x80 = hours (÷ 3600000)
    0xC0 = days (÷ 86400000)

If continuation bit set:
  VLong: upper bits of zig-zag encoded value

Reconstruction:
  zigZagDecode(lower_5 | (upper << 5)) × time_unit
```

#### zDouble Encoding
```
0xFF:       negative double → + 8 bytes (long bits)
0xFE:       lossless float representation → + 4 bytes
0x80-0xFD:  small integer [-1..124] → single byte (value + 1 + 0x80)
0x00-0x7F:  positive double → low byte of longBits + 7 more bytes
```

Verified: `CompressingStoredFieldsWriter.writeField()`, `CompressingStoredFieldsReader.java`

---

## Doc Values

Files: `.dvd` (data), `.dvm` (metadata)

### Constants
```
DATA_CODEC = "Lucene90DocValuesData"
DATA_EXTENSION = "dvd"
META_CODEC = "Lucene90DocValuesMetadata"
META_EXTENSION = "dvm"
VERSION_CURRENT = 0

Type markers (byte):
  NUMERIC        = 0
  BINARY         = 1
  SORTED         = 2
  SORTED_SET     = 3
  SORTED_NUMERIC = 4

DIRECT_MONOTONIC_BLOCK_SHIFT = 16  (2^16 = 65536 values per block)
NUMERIC_BLOCK_SHIFT = 14           (2^14 = 16384 values per block)
NUMERIC_BLOCK_SIZE = 16384
TERMS_DICT_BLOCK_LZ4_SHIFT = 6    (64 terms per block)
TERMS_DICT_BLOCK_LZ4_SIZE = 64
TERMS_DICT_REVERSE_INDEX_SHIFT = 10 (1024 terms per reverse index block)
SKIP_INDEX_LEVEL_SHIFT = 3
SKIP_INDEX_MAX_LEVEL = 4
```

Verified: `Lucene90DocValuesFormat.java`

### .dvm / .dvd General Structure

```
.dvm:
  IndexHeader ("Lucene90DocValuesMetadata")
  [Per-field metadata blocks...]
  Int: -1 (sentinel, LE)
  CodecFooter

.dvd:
  IndexHeader ("Lucene90DocValuesData")
  [Per-field data blocks...]
  CodecFooter
```

### NUMERIC

#### Metadata (.dvm)
```
Int:    fieldNumber (LE)
Byte:   type = NUMERIC (0)

[If skip index enabled]:
  Long:   skipIndexOffset in .dvd (LE)
  Long:   skipIndexLength in .dvd (LE)
  Long:   globalMaxValue (LE)
  Long:   globalMinValue (LE)
  Int:    globalDocCount (LE)
  Int:    maxDocId (LE)

[Document presence — IndexedDISI metadata]:
  Long:   docsWithFieldOffset (LE)
          -2 = no docs, -1 = all docs, else = offset in .dvd
  Long:   docsWithFieldLength (LE)
  Short:  jumpTableEntryCount (LE), -1 if N/A
  Byte:   denseRankPower (9 default, -1 if disabled)

Long:   numValues (LE)

[Encoding strategy]:
  Int:    tableSize (LE)
          -1 = no table (single block or constant)
          positive = table compression (≤256 unique values)
          negative with NUMERIC_BLOCK_SHIFT = multi-block mode

  [If tableSize > 0 (table compression)]:
    Long[tableSize]: unique values (LE)

  Byte:   numBitsPerValue
          0 = all values identical
          0xFF = multi-block mode
          else = bits for (value - min) / gcd

  Long:   min (LE)
  Long:   gcd (LE)
  Long:   valueOffset in .dvd (LE)
  Long:   valuesLength in .dvd (LE)
  Long:   jumpTableOffset (LE), -1 for single block
```

#### Data (.dvd)
```
[If skip index enabled]:
  Skip index data (hierarchical min/max per interval)

[Document presence — IndexedDISI data (if sparse)]:
  See IndexedDISI section

[Values]:
  Single block mode:
    DirectWriter packed values: (value - min) / gcd

  Multi-block mode (NUMERIC_BLOCK_SIZE = 16384 values per block):
    For each block:
      [If all same]:
        Byte: 0
        Long: value (LE)
      [If varied]:
        Byte: bitsPerValue
        Long: blockMin (LE)
        Int:  bufferSize (LE)
        DirectWriter: packed deltas

    Jump table at end:
      Long[numBlocks]: offset from values start (LE)
      Long: offset of jump table itself (LE)
```

Verified: `Lucene90DocValuesConsumer.addNumericField()`, `writeValues()`

### BINARY

#### Metadata (.dvm)
```
Int:    fieldNumber (LE)
Byte:   type = BINARY (1)
Long:   dataOffset in .dvd (LE)
Long:   dataLength in .dvd (LE)

[Document presence — same as NUMERIC]

Int:    numDocsWithField (LE)
Int:    minLength (LE)
Int:    maxLength (LE)

[If maxLength > minLength (variable-length)]:
  Long:   addressesOffset in .dvd (LE)
  Int:    DIRECT_MONOTONIC_BLOCK_SHIFT (LE, value 16)
  DirectMonotonicWriter metadata
  Long:   addressesLength in .dvd (LE)
```

#### Data (.dvd)
```
Raw bytes: all binary values concatenated in document order
[IndexedDISI data if sparse]
[DirectMonotonic address data if variable-length]
```

Verified: `Lucene90DocValuesConsumer.addBinaryField()`

### SORTED

#### Metadata (.dvm)
```
Int:    fieldNumber (LE)
Byte:   type = SORTED (2)

[Ordinals as NUMERIC metadata (values 0 to numTerms-1)]

VLong:  numTerms
Int:    DIRECT_MONOTONIC_BLOCK_SHIFT (LE, value 16)
DirectMonotonicWriter metadata for term addresses
Int:    maxTermLength (LE)
Int:    maxBlockLength (LE)
Long:   termsOffset in .dvd (LE)
Long:   termsLength in .dvd (LE)
Long:   addressesOffset in .dvd (LE)
Long:   addressesLength in .dvd (LE)

[Reverse index metadata]:
  Int:    TERMS_DICT_REVERSE_INDEX_SHIFT (LE, value 10)
  DirectMonotonicWriter metadata for reverse addresses
  Long:   reverseIndexOffset in .dvd (LE)
  Long:   reverseIndexLength in .dvd (LE)
  Long:   reverseAddressesOffset in .dvd (LE)
  Long:   reverseAddressesLength in .dvd (LE)
```

#### Data (.dvd) — Terms Dictionary
```
For each block of 64 terms:
  First term:
    VInt: term length
    Bytes: term data

  Subsequent terms (prefix-coded):
    Byte: packed(min(prefixLen,15), min(suffixLen-1,15)) in low/high nibbles
    [If prefixLen >= 15]: VInt(prefixLen - 15)
    [If suffixLen >= 16]: VInt(suffixLen - 16)
    Bytes: suffix

  Block compressed with LZ4 (dictionary = first term bytes)

[Reverse index data]:
  Sort keys at 1024-term intervals for binary search
```

Verified: `Lucene90DocValuesConsumer.addSortedField()`, `addTermsDict()`

### SORTED_SET

```
Int:    fieldNumber (LE)
Byte:   type = SORTED_SET (3)
Byte:   multiValued flag (0 = single-valued, 1 = multi-valued)

If single-valued (0):
  Encoded identically to SORTED

If multi-valued (1):
  Ordinals as SORTED_NUMERIC + terms dictionary
  Address array via DirectMonotonic (cumulative ordinal counts)
```

### SORTED_NUMERIC

```
Int:    fieldNumber (LE)
Byte:   type = SORTED_NUMERIC (4)
Int:    numDocsWithField (LE)

If single-valued (numValues == numDocsWithField):
  Encoded identically to NUMERIC

If multi-valued:
  NUMERIC values + address array:
    Long:   addressesOffset in .dvd (LE)
    Int:    DIRECT_MONOTONIC_BLOCK_SHIFT (LE, value 16)
    DirectMonotonicWriter metadata
    Long:   addressesLength in .dvd (LE)
```

Verified: `Lucene90DocValuesConsumer.addSortedNumericField()`, `addSortedSetField()`

---

## Norms

Files: `.nvd` (data), `.nvm` (metadata)

### Constants
```
DATA_CODEC = "Lucene90NormsData"
DATA_EXTENSION = "nvd"
METADATA_CODEC = "Lucene90NormsMetadata"
METADATA_EXTENSION = "nvm"
VERSION_CURRENT = 0
```

Verified: `Lucene90NormsFormat.java`

### .nvm (Norms Metadata)

```
IndexHeader ("Lucene90NormsMetadata")

For each field with norms:
  Int:    fieldNumber (LE)
  Long:   docsWithFieldOffset (LE)
          -2 = no docs have field
          -1 = all docs have field
          else = offset of IndexedDISI data in .nvd
  Long:   docsWithFieldLength (LE)
  Short:  jumpTableEntryCount (LE), -1 if N/A
  Byte:   denseRankPower (9 default, -1 if disabled)
  Int:    numDocsWithValue (LE)
  Byte:   bytesPerNorm (0, 1, 2, 4, or 8)
  Long:   normsOffset (LE)
          If bytesPerNorm == 0: this IS the singleton norm value
          Else: pointer to norm values in .nvd

Int:    -1 (sentinel, marks end of metadata, LE)
CodecFooter
```

### .nvd (Norms Data)

```
IndexHeader ("Lucene90NormsData")

For each field (same order as .nvm):
  [If sparse (docsWithFieldOffset >= 0)]:
    IndexedDISI bitset data

  [If bytesPerNorm > 0]:
    For each doc with a norm value:
      bytesPerNorm bytes (LE): norm value
      (1 byte for byte, 2 for short, 4 for int, 8 for long)

CodecFooter
```

### bytesPerNorm Selection
```
If min == max:              0 (constant, stored in normsOffset field)
If min >= -128, max <= 127: 1
If min >= -32768, max <= 32767: 2
If min >= -2^31, max <= 2^31-1: 4
Else:                       8
```

Verified: `Lucene90NormsConsumer.addNormsField()`

---

## Points/BKD

Files: `.kdd` (leaf data), `.kdi` (inner node index), `.kdm` (metadata)

### Constants
```
.kdd codec: "Lucene90PointsFormatData"
.kdi codec: "Lucene90PointsFormatIndex"
.kdm codec: "Lucene90PointsFormatMeta"
VERSION_CURRENT = 1 (maps to BKDWriter VERSION_VECTORIZE_BPV24_AND_INTRODUCE_BPV21 = 10)
DEFAULT_MAX_POINTS_IN_LEAF_NODE = 512
MAX_DIMS = 16
MAX_INDEX_DIMS = 8
```

Verified: `Lucene90PointsFormat.java`, `BKDWriter.java`, `BKDConfig.java`

### .kdm (Metadata)

```
IndexHeader ("Lucene90PointsFormatMeta")

For each field with points:
  VInt:   numDims (total dimensions)
  VInt:   numIndexDims (indexed dimensions, used in inner nodes)
  VInt:   countPerLeaf (max points per leaf, typically 512)
  VInt:   bytesPerDim
  VInt:   numLeaves
  Bytes:  minPackedValue (numIndexDims × bytesPerDim bytes)
  Bytes:  maxPackedValue (numIndexDims × bytesPerDim bytes)
  VLong:  pointCount (total points)
  VInt:   docsSeen cardinality (unique documents)
  VInt:   packedIndex length (size of inner node index bytes)
  Long:   dataStartFP in .kdd (LE)
  Long:   indexStartFP in .kdi (LE)

Int:    -1 (sentinel field number, LE)
Long:   end of .kdi file pointer (LE)
Long:   end of .kdd file pointer (LE)
CodecFooter
```

Verified: `Lucene90PointsWriter.writeField()`, `BKDWriter.finish()`

### .kdd (Leaf Data)

Each leaf block:
```
VInt:   leafCount (number of points)

[DocIDs — encoded by docIdsWriter]

[Common prefixes per dimension]:
  For each dim in 0..numDims:
    VInt:   commonPrefixLength
    Bytes:  commonPrefixBytes (commonPrefixLength bytes)

[Point values — 3 strategies]:
  Byte:   marker
    -1 (0xFF): all values identical (no more data)
    -2 (0xFE): low cardinality encoding
    0-N: sortedDim (high cardinality, run-length encoding)

  Low cardinality (-2):
    [If numIndexDims > 1: min/max bounds per index dim]
    For each unique value:
      VInt: cardinality (count of points with this value)
      Bytes: value suffix (packedBytesLength - prefix)

  High cardinality (sortedDim):
    [If numIndexDims > 1: min/max bounds per index dim]
    Runs of points sharing a prefix byte:
      Byte: prefixByte
      Byte: runLen (1-255)
      Bytes: suffixes for runLen points
```

Verified: `BKDWriter.writeLeafBlockPackedValues()`

### .kdi (Inner Node Index)

Recursive packed binary tree:
```
At each internal node:
  VLong:  delta (file pointer delta, if not leftmost leaf)
  VInt:   code = (firstDiffByteDelta × (1 + bytesPerDim) + prefix) × numIndexDims + splitDim
  Bytes:  splitValue suffix (bytesPerDim - prefix - 1 bytes, if suffix > 1)
  [Left subtree...]
  VInt:   leftNumBytes (if numLeftLeaves > 1)
  [Right subtree...]
```

Verified: `BKDWriter.writeIndex()`

---

## Compound

Files: `.cfs` (compound data), `.cfe` (compound entries)

### Constants
```
DATA_CODEC = "Lucene90CompoundData"
ENTRY_CODEC = "Lucene90CompoundEntries"
DATA_EXTENSION = "cfs"
ENTRIES_EXTENSION = "cfe"
VERSION_CURRENT = 0
ALIGNMENT = 8 (Long.BYTES)
```

Verified: `Lucene90CompoundFormat.java`

### .cfs (Compound File Storage)

```
IndexHeader ("Lucene90CompoundData")

For each file in segment (sorted by size ascending):
  [Padding to 8-byte alignment]
  File content:
    Index header (verified + copied via CodecUtil.verifyAndCopyIndexHeader)
    Data bytes (all content between header and footer)
    Footer (16 bytes):
      BE Int: FOOTER_MAGIC (0xc02893e8)
      BE Int: 0 (algorithm ID)
      BE Long: original file's CRC32 checksum

CodecFooter (for the .cfs file itself)
```

### .cfe (Compound File Entries)

```
IndexHeader ("Lucene90CompoundEntries")
VInt:   fileCount

For each file:
  String: fileName (segment-stripped name)
  Long:   dataOffset in .cfs (LE)
  Long:   dataLength in .cfs (LE)

CodecFooter
```

**Key detail**: Files are sorted by size (ascending) before writing. Each file's start in `.cfs` is aligned to 8-byte boundary for mmap compatibility.

Verified: `Lucene90CompoundFormat.write()`, `writeCompoundFile()`

---

## IndexedDISI

Encodes which documents have values, in 65536-document blocks. Used by doc values and norms.

### Constants
```
BLOCK_SIZE = 65536 (docs per block)
DENSE_BLOCK_LONGS = 1024 (65536 / 64)
MAX_ARRAY_LENGTH = 4095 (threshold: SPARSE if ≤ 4095, DENSE if > 4095)
DEFAULT_DENSE_RANK_POWER = 9 (rank entry every 512 docIDs)
```

### Block Encoding

For each 65536-doc block with at least one set bit:
```
Short:  blockID (LE) — doc >> 16
Short:  cardinality - 1 (LE)

If cardinality == 65536 (ALL):
  No data bytes

If cardinality > 4095 (DENSE):
  [If denseRankPower != -1]:
    Rank table:
      Entry count: DENSE_BLOCK_LONGS >> (denseRankPower - 7)
      Each entry: 2 bytes (big-endian within entry: MSB first, LSB second)
      For denseRankPower=9: 256 entries = 512 bytes
  Longs[1024]: bitmap (LE, 8192 bytes total)

If cardinality <= 4095 (SPARSE):
  Short[cardinality]: lower 16 bits of each doc ID (LE)
```

### Jump Table (appended after all blocks)
```
For each block:
  Int:    index — cumulative set bits before this block (LE)
  Int:    offset — byte offset from IndexedDISI start (LE)
Short:  blockCount (LE), -1 if ≤ 1 block
```

Verified: `IndexedDISI.writeBitSet()` lines 96-268

---

## DirectMonotonicWriter

Encodes monotonically increasing long sequences in blocks. Metadata goes to one output, data to another.

### Constants
```
Block size: 2^blockShift values per block
blockShift range: [2, 22]
Typical blockShift: 16 (65536 values per block)
```

### Per-Block Metadata (written to meta output)
```
Long:   min (LE)
Int:    Float.floatToIntBits(avgInc) (LE)
Long:   offset — data file pointer relative to base (LE)
Byte:   bitsRequired (0-64)
```
Total: **21 bytes per block**

### Per-Block Data (written to data output, if bitsRequired > 0)
```
DirectWriter: packed deltas (value - expected - min)
  where expected = (long)(avgInc × index)
```

### Computation
```
avgInc = (lastValue - firstValue) / max(1, blockSize - 1)
For each value at index i:
  delta = value - (long)(avgInc × i)
min = minimum delta in block
encoded[i] = delta - min
bitsRequired = unsignedBitsRequired(max(encoded))
```

If all deltas equal (bitsRequired == 0), no data section written.

Verified: `DirectMonotonicWriter.flush()` lines 73-112

---

## DirectWriter

Packs unsigned integers into a fixed number of bits per value. All data written **little-endian**.

### Supported Bits Per Value
```
1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64
```

### Encoding Strategies

**Byte-aligned** (bitsPerValue ∈ {8, 16, 24, 32, 40, 48, 56, 64}):
```
Each value written in bitsPerValue/8 bytes, LE
  8:  1 byte
  16: 2 bytes (LE short)
  24: 3 bytes (low 3 bytes of LE int)
  32: 4 bytes (LE int)
  40-64: 8 bytes (LE long, upper bytes zero)
```

**Sub-byte** (bitsPerValue ∈ {1, 2, 4}):
```
valuesPerLong = 64 / bitsPerValue
Pack valuesPerLong values into one long, LSB-first:
  value[0] in bits [0..bpv-1]
  value[1] in bits [bpv..2×bpv-1]
  ...
Write long as 8 bytes LE
```

**Partial-byte** (bitsPerValue ∈ {12, 20, 28}):
```
Pack 2 values into one wider integer:
  merged = value1 | (value2 << bitsPerValue)
  12: write merged as 3 bytes (LE, from 4-byte int)
  20: write merged as 5 bytes (LE, from 8-byte long)
  28: write merged as 7 bytes (LE, from 8-byte long)
```

### Padding (finish)
```
After all values, write padding bytes for alignment:
  bpv > 32: (64 - bpv) / 8 padding bytes
  bpv > 16: (32 - bpv) / 8 padding bytes
  bpv > 8:  (16 - bpv) / 8 padding bytes
  bpv ≤ 8:  no padding
```

Verified: `DirectWriter.flush()`, `DirectWriter.finish()`

---

## Common Pitfalls

1. **IndexedDISI uses LE**: All `writeShort`, `writeInt`, `writeLong` calls in IndexedDISI use `DataOutput` methods which are LE. The rank table entries are an exception — they're written as raw bytes with MSB first (big-endian within the 2-byte entry).

2. **DirectMonotonicWriter avgInc is a float stored as int bits**: The average increment is converted to float, then stored as `Float.floatToIntBits()` via `writeInt()` (LE). This is an IEEE 754 float reinterpreted as an integer.

3. **Compound file alignment**: `.cfs` pads to 8-byte boundaries between files. The original file's footer checksum is preserved (not recomputed).

4. **Doc values sentinel**: The `.dvm` file ends with `Int: -1` before the codec footer to mark end of field metadata.

5. **Norms sentinel**: Same as doc values — `.nvm` ends with `Int: -1` (LE) sentinel.

6. **Stored fields dirty chunks**: The `code` VInt in each chunk header encodes both flags (sliced, dirty) and doc count. Dirty chunks are written during forced flushes.

7. **DENSE rank table endianness**: Unlike all other IndexedDISI data, rank entries are written as raw byte pairs with MSB first — this is a subtle endianness difference within the format.

8. **Reader and writer field ordering must match exactly**: The Java reader consumes metadata fields in a strict sequence — any extra, missing, or reordered field shifts the stream and corrupts all subsequent reads. Always trace the Reader's `read*()` calls to verify your write order, not just the Writer.

9. **Segment attributes are part of the format contract**: Some codec readers expect specific entries in the segment info `attributes` map (e.g., `"Lucene90StoredFieldsFormat.mode"`). These aren't optional metadata — missing attributes cause `IllegalStateException` at open time.

10. **LZ4: boundary conditions around LAST_LITERALS**: The LZ4 spec requires the final N bytes of input to be emitted as literals, not consumed by a match. Off-by-one errors in match search limits or match extension caps produce streams that decompress incorrectly or misalign.

---

## Java Source Files

| File | Purpose |
|---|---|
| `codecs/lucene90/compressing/CompressingStoredFieldsWriter.java` | .fdt writer |
| `codecs/lucene90/compressing/FieldsIndexWriter.java` | .fdx/.fdm writer |
| `codecs/lucene90/Lucene90DocValuesConsumer.java` | .dvd/.dvm writer |
| `codecs/lucene90/Lucene90DocValuesFormat.java` | Doc values constants |
| `codecs/lucene90/Lucene90NormsConsumer.java` | .nvd/.nvm writer |
| `codecs/lucene90/Lucene90NormsFormat.java` | Norms constants |
| `codecs/lucene90/Lucene90PointsWriter.java` | .kdd/.kdi/.kdm writer |
| `util/bkd/BKDWriter.java` | BKD tree construction |
| `codecs/lucene90/Lucene90CompoundFormat.java` | .cfs/.cfe writer |
| `codecs/lucene90/IndexedDISI.java` | Document presence encoding |
| `util/packed/DirectMonotonicWriter.java` | Monotonic sequence encoding |
| `util/packed/DirectWriter.java` | Packed integer encoding |
