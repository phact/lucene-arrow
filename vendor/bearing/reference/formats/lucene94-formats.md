# Lucene 9.4 Formats — Field Infos

Byte-level format documentation for the Lucene 9.4 Field Infos format (`.fnm` files).

**Source of truth**: `Lucene94FieldInfosFormat.java` write() method. See `encoding-primitives.md` for shared encoding details.

---

## Constants

```
CODEC_NAME = "Lucene94FieldInfos"
EXTENSION = "fnm"
FORMAT_START = 0
FORMAT_PARENT_FIELD = 1
FORMAT_DOCVALUE_SKIPPER = 2
FORMAT_CURRENT = 2
```

Verified: `Lucene94FieldInfosFormat.java` lines 55-59

---

## File Layout

```
IndexHeader ("Lucene94FieldInfos", version=2, segmentID, segmentSuffix)

VInt:   fieldsCount (number of fields)

For each field (fieldsCount times):
  String:  fieldName
  VInt:    fieldNumber
  Byte:    fieldBits (flags, see below)
  Byte:    indexOptions (0-4)
  Byte:    docValuesType (0-5)
  Byte:    docValuesSkipIndexType (0-1)
  Long:    docValuesGen (LE, -1 if no updates)
  Map:     attributes (VInt count + [String key, String value] pairs)
  VInt:    pointDimensionCount (0 if not a point field)
  [If pointDimensionCount != 0]:
    VInt:  pointIndexDimensionCount
    VInt:  pointNumBytes
  VInt:    vectorDimension (0 if not a vector field)
  Byte:    vectorEncoding (0-1)
  Byte:    vectorSimilarityFunction (0-3)

CodecFooter
```

Verified: `Lucene94FieldInfosFormat.write()` lines 406-456

---

## Field Bits Byte

Single byte combining multiple boolean flags via bitwise OR:

| Bit | Mask | Flag | Meaning |
|---|---|---|---|
| 0 | `0x01` | `STORE_TERMVECTOR` | Field stores term vectors |
| 1 | `0x02` | `OMIT_NORMS` | Norms omitted for indexed field |
| 2 | `0x04` | `STORE_PAYLOADS` | Payloads stored for indexed field |
| 3 | `0x08` | `SOFT_DELETES_FIELD` | Field is a soft deletes marker |
| 4 | `0x10` | `PARENT_FIELD_FIELD` | Field is a parent field (format >= 1) |
| 5 | `0x20` | `DOCVALUES_SKIPPER` | Field has doc values skip index (format >= 2) |
| 6-7 | `0xC0` | Reserved | Must be 0 |

Verified: lines 61-69, validation at line 168

---

## IndexOptions Byte

| Byte | IndexOptions Enum |
|---|---|
| 0 | `NONE` — not indexed |
| 1 | `DOCS` — document IDs only |
| 2 | `DOCS_AND_FREQS` — with term frequencies |
| 3 | `DOCS_AND_FREQS_AND_POSITIONS` — with positions |
| 4 | `DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS` — with positions and character offsets |

Verified: `indexOptionsByte()` lines 458-475

---

## DocValuesType Byte

| Byte | DocValuesType Enum |
|---|---|
| 0 | `NONE` |
| 1 | `NUMERIC` |
| 2 | `BINARY` |
| 3 | `SORTED` |
| 4 | `SORTED_SET` |
| 5 | `SORTED_NUMERIC` |

Verified: `docValuesByte()` lines 477-496

---

## DocValuesSkipIndexType Byte

| Byte | DocValuesSkipIndexType Enum |
|---|---|
| 0 | `NONE` |
| 1 | `RANGE` |

Verified: `docValuesSkipIndexByte()` lines 498-509

---

## VectorEncoding Byte

Uses the enum's `ordinal()` value directly:

| Byte | VectorEncoding Enum |
|---|---|
| 0 | `BYTE` — signed 8-bit values per sample |
| 1 | `FLOAT32` — IEEE 32-bit float per sample |

Verified: line 446

---

## VectorSimilarityFunction Byte

**Warning**: This mapping is hardcoded to match Lucene 9.10's enum order, NOT the current enum's `ordinal()`. It uses a static `SIMILARITY_FUNCTIONS` list to avoid dependency on enum declaration order.

| Byte | VectorSimilarityFunction Enum |
|---|---|
| 0 | `EUCLIDEAN` |
| 1 | `DOT_PRODUCT` |
| 2 | `COSINE` |
| 3 | `MAXIMUM_INNER_PRODUCT` |

Verified: `SIMILARITY_FUNCTIONS` list lines 340-345, `distFuncToOrd()` lines 347-358

---

## Per-Field Record Summary

| # | Field | Type | Size | Conditional |
|---|---|---|---|---|
| 1 | name | String | variable | — |
| 2 | number | VInt | 1-5 | — |
| 3 | bits | Byte | 1 | — |
| 4 | indexOptions | Byte | 1 | — |
| 5 | docValuesType | Byte | 1 | — |
| 6 | docValuesSkipIndexType | Byte | 1 | — |
| 7 | docValuesGen | Long | 8 (LE) | — |
| 8 | attributes | Map | variable | — |
| 9 | pointDimensionCount | VInt | 1-5 | — |
| 10 | pointIndexDimensionCount | VInt | 1-5 | only if #9 != 0 |
| 11 | pointNumBytes | VInt | 1-5 | only if #9 != 0 |
| 12 | vectorDimension | VInt | 1-5 | — |
| 13 | vectorEncoding | Byte | 1 | — |
| 14 | vectorSimilarityFunction | Byte | 1 | — |

---

## Common Pitfalls

1. **fieldNumber is explicit**: Written as VInt, not implied by position. Fields can appear in any order.

2. **docValuesGen is always written**: Even when -1 (no updates), the full 8-byte Long is always present.

3. **Point fields are conditional**: `pointIndexDimensionCount` and `pointNumBytes` are only written when `pointDimensionCount != 0`. Readers must check before reading.

4. **Vector fields always write encoding/similarity**: Even when `vectorDimension == 0`, the encoding and similarity bytes are still written (both will be 0).

5. **Bit validation**: Bits 6-7 of the fieldBits byte must always be zero. Older format versions also require bits 4-5 to be zero.

6. **PerField format attributes are required**: Java's `PerFieldPostingsFormat` and `PerFieldDocValuesFormat` readers require `format` and `suffix` attributes in each field's attribute map (e.g., `"PerFieldPostingsFormat.format"`, `"PerFieldPostingsFormat.suffix"`). Without them, the reader silently skips the field — no error, just 0 terms.

---

## Java Source Files

| File | Purpose |
|---|---|
| `codecs/lucene94/Lucene94FieldInfosFormat.java` | Reader + writer for .fnm files |
