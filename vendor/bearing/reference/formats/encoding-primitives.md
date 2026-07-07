# Encoding Primitives

Shared encoding building blocks used across all Lucene file formats.

**Source of truth**: The Java implementation code, not Javadoc. Every claim below has been verified against the writer/reader implementation.

---

## Fixed-Width Integers

All fixed-width integers use **little-endian** byte order unless explicitly noted otherwise.

### `writeByte` / `readByte`
- 1 byte, unsigned
- Verified: `DataOutput.writeByte()` / `DataInput.readByte()`

### `writeShort` / `readShort` — LE 16-bit
- 2 bytes, little-endian
- Write order: `[bits 0-7], [bits 8-15]`
- Verified: `DataOutput.writeShort()` lines 86-89, `DataInput.readShort()` lines 82-86

### `writeInt` / `readInt` — LE 32-bit
- 4 bytes, little-endian
- Write order: `[bits 0-7], [bits 8-15], [bits 16-23], [bits 24-31]`
- Verified: `DataOutput.writeInt()` lines 73-78, `DataInput.readInt()` lines 94-100

### `writeLong` / `readLong` — LE 64-bit
- 8 bytes, little-endian
- Implemented as two LE `writeInt` calls: low 32 bits first, then high 32 bits
- Verified: `DataOutput.writeLong()` lines 223-226, `DataInput.readLong()` lines 153-155

### `writeBEInt` / `readBEInt` — BE 32-bit
- 4 bytes, **big-endian**
- Write order: `[bits 24-31], [bits 16-23], [bits 8-15], [bits 0-7]`
- **Used only in codec headers and footers** — not for data
- Verified: `CodecUtil.writeBEInt()` lines 653-658, `CodecUtil.readBEInt()` lines 667-672

### `writeBELong` / `readBELong` — BE 64-bit
- 8 bytes, **big-endian**
- Implemented as two BE `writeBEInt` calls: high 32 bits first, then low 32 bits
- **Used only in codec footers** (CRC value)
- Verified: `CodecUtil.writeBELong()` lines 661-664, `CodecUtil.readBELong()` lines 675-677

---

## Variable-Length Integers

### `writeVInt` / `readVInt` — Variable-Length 32-bit
- 1-5 bytes, unsigned (negative values technically work but are discouraged)
- Encoding: 7 bits of data per byte, **LSB first**. High bit (0x80) set = more bytes follow.
- Byte N carries bits `[7N .. 7N+6]` of the value

| Value | Bytes | Binary |
|---|---|---|
| 0 | 1 | `00000000` |
| 1 | 1 | `00000001` |
| 127 | 1 | `01111111` |
| 128 | 2 | `10000000 00000001` |
| 129 | 2 | `10000001 00000001` |
| 16383 | 2 | `11111111 01111111` |
| 16384 | 3 | `10000000 10000000 00000001` |

- Write algorithm (from `DataOutput.writeVInt()` lines 198-204):
  ```
  while (i & ~0x7F) != 0:
      emit (i & 0x7F) | 0x80
      i >>>= 7
  emit i
  ```
- Read algorithm (from `DataInput.readVInt()` lines 127-135):
  ```
  b = readByte(); i = b & 0x7F
  for shift = 7; (b & 0x80) != 0; shift += 7:
      b = readByte(); i |= (b & 0x7F) << shift
  ```

### `writeVLong` / `readVLong` — Variable-Length 64-bit
- 1-9 bytes, **non-negative only** (throws on negative input)
- Same encoding as VInt but for 64-bit values
- Verified: `DataOutput.writeVLong()` lines 236-241 delegates to `writeSignedVLong()` lines 244-250

---

## Zig-Zag Encoding

Used for efficiently encoding signed integers as unsigned variable-length values. Small absolute values (positive or negative) use fewer bytes.

### `writeZInt` / `readZInt` — Zig-Zag + VInt
- Encodes: `writeVInt(zigZagEncode(i))`
- Decodes: `zigZagDecode(readVInt())`
- Verified: `DataOutput.writeZInt()` lines 213-215, `DataInput.readZInt()` lines 143-145

### `writeZLong` / `readZLong` — Zig-Zag + VLong
- Encodes: `writeSignedVLong(zigZagEncode(i))` (note: uses signed VLong internally, not public writeVLong)
- 1-10 bytes (zig-zag encoding of negative values produces unsigned values requiring up to 10 bytes)
- Verified: `DataOutput.writeZLong()` lines 259-261

### Zig-Zag Algorithm
From `BitUtil.java` lines 283-305:

```
zigZagEncode(int i)  = (i >> 31) ^ (i << 1)    // arithmetic right shift
zigZagEncode(long l) = (l >> 63) ^ (l << 1)

zigZagDecode(int i)  = (i >>> 1) ^ -(i & 1)    // unsigned right shift
zigZagDecode(long l) = (l >>> 1) ^ -(l & 1)
```

| Signed | Zig-Zag Encoded |
|---|---|
| 0 | 0 |
| -1 | 1 |
| 1 | 2 |
| -2 | 3 |
| 2 | 4 |
| 2147483647 | 4294967294 |
| -2147483648 | 4294967295 |

---

## Strings

### `writeString` / `readString`
- Format: `VInt(byte_length) + UTF-8_bytes`
- The length is the **byte length** of the UTF-8 encoding, not the character count
- Verified: `DataOutput.writeString()` lines 271-275, `DataInput.readString()` lines 230-235

---

## Collections

### `writeMapOfStrings` / `readMapOfStrings`
- Format: `VInt(count) + [String(key) + String(value)] * count`
- Iteration order depends on the Map implementation passed to the writer
- Verified: `DataOutput.writeMapOfStrings()` lines 304-310, `DataInput.readMapOfStrings()` lines 261-276

### `writeSetOfStrings` / `readSetOfStrings`
- Format: `VInt(count) + String(value) * count`
- Verified: `DataOutput.writeSetOfStrings()` lines 321-326, `DataInput.readSetOfStrings()` lines 283-296

---

## Codec Header

Written at the start of every Lucene index file. All header/footer fields use **big-endian** byte order.

### Simple Header (`writeHeader`)
Format:
```
Magic:     BE Int32  = 0x3fd76c17  (CODEC_MAGIC)
CodecName: String    = VInt(len) + UTF-8 bytes
Version:   BE Int32  = codec version number
```
- Total length: `9 + codec_name.len()` bytes (4 magic + VInt(1) + name + 4 version, assuming name < 128 bytes so VInt is 1 byte)
- Verified: `CodecUtil.writeHeader()` lines 77-86, `CodecUtil.headerLength()` line 145

### Index Header (`writeIndexHeader`)
Format:
```
[Simple Header]
SegmentID:    byte[16]   = 16 raw bytes (StringHelper.ID_LENGTH = 16)
SuffixLength: byte       = 1 unsigned byte (0-255)
SuffixBytes:  byte[N]    = N raw bytes (ASCII suffix, e.g., file extension)
```
- Total length: `headerLength(codec) + 16 + 1 + suffix.len()`
- The suffix length is a single byte (not VInt), limiting suffix to 255 chars
- Verified: `CodecUtil.writeIndexHeader()` lines 121-135, `CodecUtil.indexHeaderLength()` lines 155-157

---

## Codec Footer

Written at the end of every Lucene index file. Always exactly **16 bytes**.

Format:
```
Magic:       BE Int32  = 0xc02893e8  (~CODEC_MAGIC, i.e., FOOTER_MAGIC)
AlgorithmID: BE Int32  = 0x00000000  (always 0, indicates zlib-CRC32)
Checksum:    BE Int64  = CRC-32 value zero-extended to 64 bits
```

- `FOOTER_MAGIC = ~CODEC_MAGIC = ~0x3fd76c17 = 0xc02893e8`
- The checksum covers all bytes from the start of the file through the AlgorithmID field (i.e., everything before the checksum itself)
- The CRC-32 value is stored in the low 32 bits of the BE Int64; high 32 bits must be zero
- CRC-32 polynomial: `0xEDB88320` (same as `java.util.zip.CRC32`, ISO 3309 / ITU-T V.42)
- Verified: `CodecUtil.writeFooter()` lines 409-413, `CodecUtil.footerLength()` returns 16 (line 422), `CodecUtil.writeCRC()` lines 643-650, `CodecUtil.validateFooter()` lines 560-598

---

## Common Pitfalls

1. **Endianness mismatch**: Data values (writeInt, writeLong, writeShort) are LE. Header/footer values (magic, version, CRC) are BE. This is the #1 source of bugs.

2. **VInt vs VLong sign handling**: `writeVLong` throws on negative input, but `writeVInt` silently encodes negatives (taking 5 bytes). `writeZLong` internally uses `writeSignedVLong` (private method) which does accept negatives.

3. **String length is byte length**: `writeString` writes the UTF-8 byte count, not the character count. For ASCII strings they're equal, but multi-byte UTF-8 characters will differ.

4. **Suffix length encoding**: In `writeIndexHeader`, the suffix length is a single raw byte (not VInt), limiting suffixes to 255 bytes.

5. **FOOTER_MAGIC value**: It's the bitwise complement of CODEC_MAGIC (`~0x3fd76c17`). Easy to confuse with just negation.

6. **CRC stored as 64 bits**: The CRC-32 value is written as a BE 64-bit long, with the upper 32 bits required to be zero. The reader validates this.

---

## Java Source Files

| File | Key Methods |
|---|---|
| `store/DataOutput.java` | `writeInt`, `writeShort`, `writeLong`, `writeVInt`, `writeVLong`, `writeZInt`, `writeZLong`, `writeString`, `writeMapOfStrings`, `writeSetOfStrings` |
| `store/DataInput.java` | `readInt`, `readShort`, `readLong`, `readVInt`, `readVLong`, `readZInt`, `readZLong`, `readString`, `readMapOfStrings`, `readSetOfStrings` |
| `codecs/CodecUtil.java` | `writeHeader`, `writeIndexHeader`, `writeFooter`, `writeBEInt`, `writeBELong`, `readBEInt`, `readBELong`, `writeCRC`, `readCRC` |
| `util/BitUtil.java` | `zigZagEncode(int)`, `zigZagEncode(long)`, `zigZagDecode(int)`, `zigZagDecode(long)` |
| `util/StringHelper.java` | `ID_LENGTH = 16` |
