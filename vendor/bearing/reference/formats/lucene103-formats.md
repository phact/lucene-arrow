# Lucene 10.3 Formats — Postings & Terms Dictionary

Byte-level format documentation for the Lucene 10.3 postings format and BlockTree terms dictionary.

**Source of truth**: Java writer implementation code. See `encoding-primitives.md` for shared encoding details.

---

## Table of Contents

1. [Constants](#constants)
2. [Postings (.doc, .pos, .pay, .psm)](#postings)
3. [BlockTree Terms Dictionary (.tim, .tip, .tmd)](#blocktree-terms-dictionary)
4. [ForUtil / PForUtil / ForDeltaUtil](#forutil--pforutil--fordeltautil)
5. [VInt15 / VLong15 Encoding](#vint15--vlong15-encoding)

---

## Constants

### Postings Format
```
BLOCK_SIZE = 128 (ForUtil.BLOCK_SIZE)
BLOCK_MASK = 127
LEVEL1_FACTOR = 32
LEVEL1_NUM_DOCS = 4096 (32 × 128)
LEVEL1_MASK = 4095

Codec names:
  TERMS_CODEC = "Lucene103PostingsWriterTerms"
  META_CODEC  = "Lucene103PostingsWriterMeta"
  DOC_CODEC   = "Lucene103PostingsWriterDoc"
  POS_CODEC   = "Lucene103PostingsWriterPos"
  PAY_CODEC   = "Lucene103PostingsWriterPay"

Extensions:
  .doc  (DOC_EXTENSION)
  .pos  (POS_EXTENSION)
  .pay  (PAY_EXTENSION)
  .psm  (META_EXTENSION)

VERSION_START = 0
VERSION_CURRENT = 0
```

### BlockTree Terms Dictionary
```
TERMS_CODEC_NAME       = "BlockTreeTermsDict"
TERMS_INDEX_CODEC_NAME = "BlockTreeTermsIndex"
TERMS_META_CODEC_NAME  = "BlockTreeTermsMeta"

Extensions:
  .tim  (TERMS_EXTENSION)
  .tip  (TERMS_INDEX_EXTENSION)
  .tmd  (TERMS_META_EXTENSION)

DEFAULT_MIN_BLOCK_SIZE = 25
DEFAULT_MAX_BLOCK_SIZE = 48

VERSION_START = 0
VERSION_CURRENT = 0
```

Verified: `Lucene103PostingsFormat.java`, `Lucene103BlockTreeTermsWriter.java`

---

## Postings

### .doc File — Document IDs + Frequencies + Skip Data

The `.doc` file interleaves skip data with packed blocks of doc IDs and frequencies.

#### Per-Term Structure
```
For each term:
  [Level 1 skip data (every 4096 docs)]
    [Level 0 skip data (every 128 docs)]
      [Packed block (128 docs)]
    ...repeated up to 32 times per L1 block...
  ...repeated for all L1 blocks...
  [VInt tail block (remaining < 128 docs)]
```

#### Level 0 Skip Data (before each packed block)

Written to a scratch buffer, then prefixed with its length:
```
VLong:   level0NumBytes (total bytes of this skip entry)

--- Skip metadata (level0NumBytes bytes) ---
VInt15:  docDelta (docID - level0LastDocID)
VLong15: level0Size (total level0Output size: impacts + doc encoding + freq encoding)

[If frequencies indexed]:
  VLong:   impactBytesLength
  [Impact pairs]:
    For each competitive impact:
      VInt:  (freqDelta << 1) | (normDelta != 0 ? 1 : 0)
      [If normDelta != 0]: ZLong(normDelta)

  [If positions indexed]:
    VLong: posFPDelta (posOut.filePointer - level0LastPosFP)
    Byte:  posBufferUpto (position offset within last block)

  [If payloads/offsets]:
    VLong: payFPDelta (payOut.filePointer - level0LastPayFP)
    VInt:  payByteUpto (byte offset within payload block)
```

Verified: `Lucene103PostingsWriter.java` lines 391-473

#### Level 1 Skip Data (every 4096 docs = 32 packed blocks)

Written around groups of 32 Level 0 entries:
```
VInt:    docDelta (docID - level1LastDocID)

[Impact pairs — same format as Level 0]

[If positions indexed]:
  VLong: posFPDelta
  Byte:  posBufferUpto

[If payloads/offsets]:
  VLong: payFPDelta
  VInt:  payByteUpto

VLong:   level1Len (2 × Short.BYTES + scratchOutput.size + level1Output.size)
Short:   skipMetaLen (LE, includes impact/pos/pay metadata + Short.BYTES)
Short:   impactBytesLen (LE, byte count of impact data only)

[32 × Level 0 skip entries + packed blocks]
```

Verified: lines 494-535

#### Packed Doc Delta Block (128 docs)

Three encoding strategies selected per block:
```
Byte:   encoding flag

If flag == 0: CONSECUTIVE
  All 128 doc IDs are consecutive (docRange == 128)
  No further data — reconstruct from skip data

If flag > 0: FOR (Frame-of-Reference)
  flag = bitsPerValue (1-32)
  ForDeltaUtil encoded block:
    128 doc deltas packed at bitsPerValue bits each
    Total bytes: bitsPerValue × 128 / 8

If flag < 0: BITSET
  flag = -numBitSetLongs (negative, range -1 to -64)
  Long[|flag|]: bitset (LE), bit i set if doc ID i is present
```

Verified: `flushDocBlock()` lines 391-458

#### Packed Frequency Block (128 frequencies)

Written immediately after doc deltas (if frequencies indexed):
```
PForUtil encoded block (see PForUtil section)
```

Verified: line 461

#### VInt Tail Block (remaining < 128 docs)

Written via `PostingsUtil.writeVIntBlock()`:
```
[If frequencies indexed]:
  For each doc:
    VInt: (docDelta << 1) | (freq == 1 ? 1 : 0)
    [If freq != 1]: VInt(freq)

[If no frequencies]:
  For each doc:
    VInt: docDelta
```

Verified: lines 394-397

### .pos File — Positions

#### Packed Position Block (128 positions)
```
PForUtil: 128 position deltas

[If payloads — written to .pay]:
  PForUtil: 128 payload lengths
  VInt:     sum of payload byte lengths
  Bytes:    concatenated payload data

[If offsets — written to .pay]:
  PForUtil: 128 offset start deltas
  PForUtil: 128 offset lengths
```

Verified: lines 344-355

#### VInt Tail Positions (remaining < 128)
```
For each position:
  [If payloads]:
    VInt: (posDelta << 1) | (payloadLength changed ? 1 : 0)
    [If changed]: VInt(payloadLength)
    Bytes: payload data (payloadLength bytes)
  [Else]:
    VInt: posDelta

  [If offsets]:
    VInt: (offsetDelta << 1) | (offsetLength changed ? 1 : 0)
    [If changed]: VInt(offsetLength)
```

Verified: lines 595-630

### .pay File — Payloads & Offsets

Contains packed payload lengths, payload data, offset start deltas, and offset lengths — written from the `.pos` writer (see packed position block above).

### .psm File — Postings Metadata

Written at close time:
```
IndexHeader ("Lucene103PostingsWriterMeta")
Int:    maxNumImpactsAtLevel0 (LE)
Int:    maxImpactNumBytesAtLevel0 (LE)
Int:    maxNumImpactsAtLevel1 (LE)
Int:    maxImpactNumBytesAtLevel1 (LE)
Long:   docFileSize (LE)
[If positions]: Long: posFileSize (LE)
[If payloads/offsets]: Long: payFileSize (LE)
CodecFooter
```

Verified: lines 708-720

### encodeTerm() — Term Metadata (written to .tim)

Called by the terms dictionary writer for each term:
```
[DocStartFP or SingletonDocID]:
  If both current and previous term are singletons:
    VLong: (zigZagEncode(docIDDelta) << 1) | 0x01
  Else:
    VLong: (docStartFPDelta << 1) | 0x00
    [If current term is singleton]:
      VInt: singletonDocID

[If positions indexed]:
  VLong: posStartFPDelta

[If payloads/offsets]:
  VLong: payStartFPDelta

[If positions indexed and lastPosBlockOffset != -1]:
  VLong: lastPosBlockOffset
```

The low bit of the first VLong distinguishes singleton pairs (1) from file pointer deltas (0).

Verified: `encodeTerm()` lines 654-692

---

## BlockTree Terms Dictionary

### .tmd File — Terms Metadata

```
IndexHeader ("BlockTreeTermsMeta")

For each field:
  VInt:    fieldNumber
  VLong:   numTerms
  VInt:    rootCodeLength
  Bytes:   rootCode (rootCodeLength bytes)
  [If field has DOCS_AND_FREQS or higher]:
    VLong: sumTotalTermFreq
  VLong:   sumDocFreq
  VInt:    docCount
  VInt:    minTermLength
  Bytes:   minTerm (minTermLength bytes)
  VInt:    maxTermLength
  Bytes:   maxTerm (maxTermLength bytes)
  VLong:   indexStartFP (start of FST in .tip)
  VLong:   trieRootNodeFP (FST root node offset)
  VLong:   indexEndFP (end of FST in .tip)

Long:    termIndexLength — size of .tip (LE)
Long:    termDictLength — size of .tim (LE)
CodecFooter
```

Verified: `Lucene103BlockTreeTermsWriter.close()`

### .tim File — Terms Dictionary Blocks

```
IndexHeader ("BlockTreeTermsDict")

PostingsHeader (written by PostingsWriterBase.init())

[Term blocks for all fields...]

CodecFooter
```

#### Block Structure

Each block contains 25-48 entries (terms and/or sub-block pointers). If a node has more than 48 entries, it's split into "floor blocks" with leading-byte markers.

```
Block:
  VInt:   code = (numEntries << 1) | (isLastBlock ? 1 : 0)

  --- Suffix Data ---
  VLong:  token = (suffixBytesLength << 3) | (isLeafBlock ? 0x04 : 0) | compressionCode
          compressionCode: 0x00 = none, 0x01 = LOWERCASE_ASCII, 0x02 = LZ4
  Bytes:  suffixData (suffixBytesLength bytes, possibly compressed)

  --- Suffix Lengths ---
  [If all suffixes same length]:
    VInt:  (numSuffixBytesTotal << 1) | 1
    Byte:  commonLength
  [Else]:
    VInt:  numSuffixBytesTotal << 1
    Bytes: per-entry suffix length values (numSuffixBytesTotal bytes)

  --- Term Stats ---
  VInt:   numStatsBytes
  Bytes:  stats data (numStatsBytes bytes)
    Singleton RLE: VInt((count - 1) << 1 | 1) — for runs of df=1, ttf=1 terms
    Other:         VInt(docFreq << 1), [VLong(totalTermFreq - docFreq)]

  --- Term Metadata ---
  VInt:   numMetadataBytes
  Bytes:  metadata (numMetadataBytes bytes, from encodeTerm() calls)
```

Verified: `writeBlock()` in `Lucene103BlockTreeTermsWriter.java`

#### Suffix Encoding Within Blocks

**Leaf blocks** (only terms):
```
Per entry: VInt(suffixLength)
```

**Inner blocks** (terms + sub-block references):
```
Per term:      VInt(suffixLength << 1 | 0)
Per sub-block: VInt(suffixLength << 1 | 1), VLong(blockFPDelta)
```

#### Suffix Compression Selection
```
If suffixBytesLength <= 2 × numEntries: NO_COMPRESSION (too sparse)
Elif prefixLength <= 2:                 NO_COMPRESSION (fuzzy query optimization)
Else:
  Try LZ4 — use if >= 25% savings
  Else try LOWERCASE_ASCII — use if successful
  Else NO_COMPRESSION
```

#### Floor Blocks

When a trie node has more than `maxItemsInBlock` (48) entries, it's split into floor blocks sharing a common prefix. The first block contains entries with no additional distinguishing byte. Subsequent floor blocks are identified by a leading byte.

Floor data (stored in FST output):
```
VInt:   numFloorBlocks - 1
For each floor block:
  Byte:  leadingByte
  VLong: (blockFPDelta << 1) | hasTerms
```

### .tip File — Terms Index (FST/Trie)

```
IndexHeader ("BlockTreeTermsIndex")

[Per-field FST data, written in post-order depth-first traversal]

CodecFooter
```

#### FST Node Format

Nodes are saved in post-order (children before parents). Each node encodes pointers to `.tim` blocks.

**Leaf node (no children):**
```
[Bytes: floorData (optional)]
[Bytes: outputFP (1-7 bytes, variable-length encoded file pointer)]
Byte:   header
        bits 0-1: SIGN = 0x00 (SIGN_NO_CHILDREN)
        bits 2-4: outputFpBytes - 1
        bit 5:    hasTerms
        bit 6:    hasFloor
```

**Single-child node:**
```
[Bytes: floorData (optional)]
Byte:   childLabel
[Bytes: childDeltaFP (variable-length)]
[Bytes: outputFP (optional, if SIGN indicates output present)]
Byte:   header
        bits 0-1: SIGN = 0x01 (with output) or 0x02 (without output)
        bits 2-4: childFpBytes - 1
        bits 5-7: outputFpBytes - 1 (if with output)
        bit 5:    hasFloor (repurposed in some SIGN modes)
```

**Multi-child node:**
```
[Bytes: floorData (optional)]
For each child:
  Byte:  childLabel
  [Bytes: childDeltaFP]
  [Bytes: childOutput (optional)]
[Bytes: outputFP (parent output, optional)]
Byte:   header
        bits 0-1: SIGN = 0x03 (SIGN_MULTI_CHILDREN)
        remaining: childFpBytes, outputFpBytes, flags
```

**FST Constants:**
```
SIGN_NO_CHILDREN = 0x00
SIGN_SINGLE_CHILD_WITH_OUTPUT = 0x01
SIGN_SINGLE_CHILD_WITHOUT_OUTPUT = 0x02
SIGN_MULTI_CHILDREN = 0x03

LEAF_NODE_HAS_TERMS = 1 << 5
LEAF_NODE_HAS_FLOOR = 1 << 6
NON_LEAF_NODE_HAS_TERMS = 1 << 1
NON_LEAF_NODE_HAS_FLOOR = 1 << 0
```

Verified: `TrieBuilder.saveNodes()` in `Lucene103BlockTreeTermsWriter.java`

---

## ForUtil / PForUtil / ForDeltaUtil

### ForUtil — Frame-of-Reference for 128 integers

Packs 128 unsigned integers at a fixed `bitsPerValue` (1-32).

```
Output size: bitsPerValue × 128 / 8 bytes

Encoding by primitive size:
  bitsPerValue <= 8:  collapse to 8-bit primitives (4 ints → 1 int)
  bitsPerValue <= 16: collapse to 16-bit primitives (2 ints → 1 int)
  bitsPerValue > 16:  use 32-bit scalars (no collapsing)

Bit-packing within each primitive:
  Values packed LSB-first into the primitive width
  Written as packed int arrays via DataOutput
```

| bitsPerValue | Output Bytes |
|---|---|
| 1 | 16 |
| 2 | 32 |
| 4 | 64 |
| 8 | 128 |
| 16 | 256 |
| 24 | 384 |
| 32 | 512 |

Verified: `ForUtil.java` encode(), numBytes()

### PForUtil — Patched Frame-of-Reference

Handles values with occasional outliers by encoding most values at a reduced bit width and patching exceptions.

```
Byte:   token = (numExceptions << 5) | patchedBitsRequired
        numExceptions: 0-7 (top 3 bits)
        patchedBitsRequired: 0-31 (bottom 5 bits)

[If patchedBitsRequired == 0 (all values identical)]:
  VInt: constantValue

[Else]:
  ForUtil block: 128 values at patchedBitsRequired bits each
  [For each exception (numExceptions times)]:
    Byte: index (position in block, 0-127)
    Byte: highBits (upper bits that didn't fit in main encoding)
```

Maximum 7 exceptions per block. The main block uses a reduced bit width; exception values have their upper bits stored separately.

Verified: `PForUtil.java` encode()

### ForDeltaUtil — Delta-Encoded FOR

Optimized for monotonically increasing sequences (doc ID deltas ≥ 1). Uses the same bit-packing as ForUtil but with delta-specific collapsing:

```
Encoding by bitsPerValue:
  <= 3:  collapse to 8-bit primitives
  <= 10: collapse to 16-bit primitives
  > 10:  use 32-bit scalars

Decoding: prefix-sum reconstruction from base value + deltas
```

Verified: `ForDeltaUtil.java` encodeDeltas()

---

## VInt15 / VLong15 Encoding

A 2-byte-optimized variable-length encoding used in skip data. Designed for values that typically fit in 15 bits, avoiding the branch-heavy standard VInt for this common case.

```
If value fits in 15 bits (value & ~0x7FFF == 0):
  Short: value (LE, high bit clear — indicates no continuation)

Else:
  Short: 0x8000 | (value & 0x7FFF) (LE, high bit set — indicates continuation)
  VLong: value >> 15 (remaining upper bits)
```

- First 2 bytes always present (LE short)
- High bit of the short is the continuation flag
- Low 15 bits carry the first 15 bits of the value
- If continuation, a standard VLong follows with the remaining bits

Verified: `Lucene103PostingsWriter.writeVLong15()` lines 381-388

---

## Common Pitfalls

1. **Skip data is interleaved**: Level 0 skip data appears *before* each packed block in the `.doc` file, not in a separate skip list. Level 1 wraps groups of 32 Level 0 entries.

2. **Consecutive doc encoding**: When all 128 doc IDs in a block are consecutive, the encoding flag is 0 and no doc data is written — the reader reconstructs from skip data. Easy to miss.

3. **Bitset encoding for dense blocks**: When flag < 0, `|flag|` gives the number of longs in the bitset (not 128). Sparse-within-block uses fewer longs.

4. **Singleton optimization**: Terms with docFreq=1 don't write to `.doc` at all. The doc ID is embedded in the term metadata via encodeTerm(). The low bit of the first VLong distinguishes this case.

5. **VInt15 is not VInt**: Skip data uses the 2-byte-first VInt15/VLong15 encoding, not standard VInt. The first read is always a LE short.

6. **Suffix compression in terms dictionary**: Block suffix data may be LZ4 or LOWERCASE_ASCII compressed. The compression code is in the low 2 bits of the suffix token VLong.

7. **Floor blocks**: Large trie nodes (>48 entries) are split into floor blocks. The FST output contains floor data with leading bytes for dispatch. Missing floor block handling will corrupt the terms dictionary.

8. **Stats singleton RLE**: Consecutive terms with df=1 and ttf=1 are run-length encoded. The low bit of the stats VInt distinguishes singleton runs (1) from normal stats (0).

9. **Headers go to the file the reader expects**: Each codec header must be written to the output that the corresponding reader calls `init()` on. For example, the postings writer's TERMS_CODEC index header goes in `.tmd` (metadata), not `.tim` (data). Verify by checking what `Reader.init()` reads from.

10. **File pointers and counts must be real values**: Placeholders (0, -1, etc.) for file pointers or counts that the reader validates will cause seek errors or assertion failures. Common examples: trie root FP, per-field doc count, BKD data start FP. Java's reader validates invariants like `sumDocFreq >= docCount`.

11. **Java signed arithmetic vs Rust unsigned**: Java's `int` allows negative intermediate values in index arithmetic (e.g., blocktree `prefixStarts`). Rust's `usize` panics on underflow. Use bounds checks or `saturating_sub` when porting arithmetic that may go temporarily negative.

---

## Java Source Files

| File | Purpose |
|---|---|
| `codecs/lucene103/Lucene103PostingsFormat.java` | Constants, block size, extensions |
| `codecs/lucene103/Lucene103PostingsWriter.java` | .doc/.pos/.pay/.psm writer, skip data, encodeTerm() |
| `codecs/lucene103/blocktree/Lucene103BlockTreeTermsWriter.java` | .tim/.tip/.tmd writer, block structure, FST |
| `codecs/lucene103/ForUtil.java` | FOR bit-packing for 128-int blocks |
| `codecs/lucene103/PForUtil.java` | Patched FOR with exception handling |
| `codecs/lucene103/ForDeltaUtil.java` | Delta-optimized FOR encoding |
