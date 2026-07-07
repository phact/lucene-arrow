# Lucene 9.9 Formats — Segment Info

Byte-level format documentation for the Lucene 9.9 Segment Info format (`.si` files).

**Source of truth**: `Lucene99SegmentInfoFormat.java` write()/writeSegmentInfo() methods. See `encoding-primitives.md` for shared encoding details.

**Important**: Despite living in the `lucene99` package, this format uses codec name `"Lucene90SegmentInfo"` — the codec name was not bumped when the format moved packages.

---

## Constants

```
CODEC_NAME = "Lucene90SegmentInfo"
EXTENSION = "si"
VERSION_START = 0
VERSION_CURRENT = 0

SegmentInfo.YES = 1
SegmentInfo.NO  = -1
```

Verified: `Lucene99SegmentInfoFormat.java` line 83, `SegmentInfo.java` lines 45-48

---

## File Layout

All `writeInt`/`writeLong` calls use `DataOutput` methods (little-endian). See `encoding-primitives.md`.

```
IndexHeader ("Lucene90SegmentInfo", version=0, segmentID, suffix="")

--- Lucene Version ---
Int:    version.major (LE)
Int:    version.minor (LE)
Int:    version.bugfix (LE)

--- Minimum Version (conditional) ---
Byte:   hasMinVersion (1 = present, 0 = absent)
[If hasMinVersion == 1]:
  Int:  minVersion.major (LE)
  Int:  minVersion.minor (LE)
  Int:  minVersion.bugfix (LE)

--- Segment Metadata ---
Int:    maxDoc — document count in segment (LE)
Byte:   isCompoundFile (1 = YES, -1 = NO)
Byte:   hasBlocks (1 = YES, -1 = NO)

--- String Collections ---
Map:    diagnostics (VInt count + [String key, String value] pairs)
Set:    files (VInt count + String values)
Map:    attributes (VInt count + [String key, String value] pairs)

--- Index Sort (conditional) ---
VInt:   numSortFields (0 if unsorted)
[For each sort field]:
  String:  providerName (SortField provider class name)
  [provider-specific data via SortFieldProvider.write()]

CodecFooter
```

Verified: `Lucene99SegmentInfoFormat.writeSegmentInfo()` lines 185-235

---

## Field Details

### Lucene Version

Three LE Int32 values encoding the Lucene version that created this segment. On read, `major` must be >= 7.

### MinVersion

Optional — indicates the minimum Lucene version of any document in the segment. Present when documents may have been added by different Lucene versions (e.g., after merging). The `hasMinVersion` byte is strictly 0 or 1 (other values are corrupt).

### Compound File / Has Blocks Flags

Both use the `SegmentInfo.YES` (1) / `SegmentInfo.NO` (-1) convention, written as a single signed byte. On read, any value other than 1 or -1 indicates corruption.

### Diagnostics Map

Contains metadata about how the segment was created. Typical keys include:
- `"source"` — what created the segment (e.g., `"flush"`, `"merge"`)
- `"os.name"`, `"os.version"`, `"os.arch"` — platform info
- `"java.version"`, `"java.vendor"` — JVM info
- `"lucene.version"` — Lucene version string
- `"timestamp"` — creation time

### Files Set

Set of file names belonging to this segment (segment-name-stripped). Used for compound file construction and segment cleanup.

### Index Sort

If the segment has a sort order, `numSortFields` > 0 and each sort field is serialized by its `SortFieldProvider`. Our implementation typically has `numSortFields = 0` (unsorted).

---

## Common Pitfalls

1. **Codec name mismatch**: The codec name is `"Lucene90SegmentInfo"` even though the class is in the `lucene99` package. This is intentional — the format version didn't change, only the package location.

2. **YES/NO are not boolean**: They're signed bytes: YES=1, NO=-1 (not 0). Using 0 or any other value is treated as corruption.

3. **All ints are LE**: The version ints (major, minor, bugfix) and maxDoc use `DataOutput.writeInt()` which is little-endian. Do not confuse with `CodecUtil.writeBEInt()` used only in headers/footers.

4. **Files set ordering**: The set iteration order is not guaranteed. Readers should not depend on file ordering.

5. **Segment attributes are part of the codec contract**: Some codec readers expect specific keys in the `attributes` map (e.g., `"Lucene90StoredFieldsFormat.mode"` for stored fields). Missing entries cause `IllegalStateException` at open time — they aren't optional metadata.

---

## Java Source Files

| File | Purpose |
|---|---|
| `codecs/lucene99/Lucene99SegmentInfoFormat.java` | Reader + writer for .si files |
| `index/SegmentInfo.java` | YES/NO constants, segment metadata |
