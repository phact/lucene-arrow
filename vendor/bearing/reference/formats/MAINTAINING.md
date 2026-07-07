# Maintaining Format Reference Docs

## Purpose

These documents are **primarily consumed by Claude Code** as context when implementing or debugging codec writers. They exist to avoid re-reading 15+ Java files (1000+ lines each) every conversation.

**Keep them concise.** Every line costs context window space. Prefer terse byte layouts and tables over prose. Don't document implementation details that are obvious from reading the Java source once — focus on things that are easy to get wrong (endianness, conditional fields, write order, cross-file contracts).

## Trust Hierarchy

```
1. Java implementation code (Writer classes)     ← ONLY source of truth
2. Java implementation code (Reader classes)      ← Cross-reference
3. These reference docs                           ← Convenience, may have errors
4. Javadoc comments                               ← Frequently wrong or incomplete
```

**Never trust Javadoc alone.** We've been burned by Javadoc that disagreed with implementation. Always trace the actual `write*()` calls in the Writer class.

---

## Lucene 10.3.2 Codec Version Mapping

**This is the definitive mapping for our target codec (`Lucene103`).** When looking up a format, use this table to find the correct version.

| Component | Format Version | Codec Name | Files | Reference Doc |
|---|---|---|---|---|
| **Postings** | Lucene 103 | `Lucene103PostingsWriter*` | .doc, .pos, .pay, .psm | `lucene103-formats.md` |
| **Terms Dictionary** | Lucene 103 | `BlockTreeTermsDict/Index/Meta` | .tim, .tip, .tmd | `lucene103-formats.md` |
| **Field Infos** | Lucene 94 | `Lucene94FieldInfos` | .fnm | `lucene94-formats.md` |
| **Segment Info** | Lucene 99 | `Lucene90SegmentInfo` (!) | .si | `lucene99-formats.md` |
| **Stored Fields** | Lucene 90 | `Lucene90FieldsIndex*` | .fdt, .fdx, .fdm | `lucene90-formats.md` |
| **Doc Values** | Lucene 90 | `Lucene90DocValues*` | .dvd, .dvm | `lucene90-formats.md` |
| **Norms** | Lucene 90 | `Lucene90Norms*` | .nvd, .nvm | `lucene90-formats.md` |
| **Points/BKD** | Lucene 90 | `Lucene90PointsFormat*` | .kdd, .kdi, .kdm | `lucene90-formats.md` |
| **Compound** | Lucene 90 | `Lucene90Compound*` | .cfs, .cfe | `lucene90-formats.md` |
| **Segments File** | — | `segments` | segments_N | (standard across versions) |
| **Shared Encodings** | — | `DataOutput`, `CodecUtil`, `BitUtil` | — | `encoding-primitives.md` |
| **Utility Encoders** | — | `IndexedDISI`, `DirectMonotonic`, `DirectWriter` | — | `lucene90-formats.md` |

**Watch out for**: Segment Info uses codec name `"Lucene90SegmentInfo"` even though the format class is in the `lucene99` package. The codec name was not bumped when the class moved.

### How Lucene versioning works

Lucene only creates a new codec format version when the on-disk format actually changes. Components that haven't changed since 9.0 still use the Lucene 9.0 format — there's no "Lucene 103 stored fields" format because stored fields haven't changed. This is why our 10.3.2 codec references formats spanning Lucene 90 through 103.

When looking up encoding details, always check this table first to ensure you're reading the correct format version's documentation.

---

## How to Verify a Claim

When you suspect a doc is wrong, or when porting a new feature:

### Step 1: Find the Writer
Locate the Java Writer class for the format. The mapping above gives codec names — search for those in the Java source.

### Step 2: Trace the Write Path
Starting from the top-level write method (e.g., `flush()`, `write()`, `close()`), trace every `write*()` call in order. Note:
- Which `DataOutput` method is called (`writeInt` = LE, `writeBEInt` = BE)
- Whether the value is conditional
- What the value represents

### Step 3: Cross-Reference the Reader
Find the corresponding Reader class and verify the read sequence matches. If the reader does `readInt()` where the writer does `writeInt()`, they agree. If one uses BE and the other LE, you've found a bug (in the doc or the code).

### Step 4: Update the Doc
Fix the discrepancy. Add a note about what changed and which Java method was traced.

---

## Verification Checklist

When reviewing or updating a format doc, verify each claim against:

- [ ] **Endianness**: Is each integer LE (`writeInt`) or BE (`writeBEInt`)? This is the #1 bug source.
- [ ] **Byte count**: Does the stated size match the write method? (`writeInt` = 4, `writeLong` = 8, `writeByte` = 1, `writeVInt` = 1-5, `writeVLong` = 1-9)
- [ ] **Encoding choice**: VInt vs Int vs VLong? ZInt vs VInt? The wrong choice silently corrupts.
- [ ] **Conditional fields**: Is the field always present or only under certain conditions?
- [ ] **Magic values**: Verify hex constants against the Java source constants, not computed values.
- [ ] **Write order**: Fields must be documented in the exact order they're written. Out-of-order docs cause misalignment bugs.
- [ ] **Codec name**: Verify the exact string — copy from Java source, don't type from memory.
- [ ] **Version number**: Verify VERSION_CURRENT matches what our code writes.

---

## When to Update

- **Bug found**: If a Rust encoding bug traces back to a doc error, fix the doc immediately.
- **New feature ported**: When porting a new Lucene feature, document its format here.
- **Discrepancy discovered**: If you find the doc disagrees with Java source, fix it — the Java code wins.
- **Version upgrade**: If we upgrade to a newer Lucene version, audit all format docs against the new source.

---

## Document Structure Conventions

Each format doc follows this pattern for consistency:

1. **Constants section**: Codec names, extensions, version numbers — copy-pasted from Java source
2. **File layout**: Sequential byte-by-byte description with types and endianness
3. **Encoding details**: Strategy selection logic, conditional fields, algorithm specifics
4. **Common pitfalls**: Endianness traps, off-by-one patterns, surprising behaviors
5. **Java source files table**: Pointers to the writer and reader classes

Use this format:
```
Type:   fieldName (endianness if non-obvious)
```

Where Type is one of: `Byte`, `Short`, `Int`, `Long` (all LE unless noted), `VInt`, `VLong`, `ZInt`, `ZLong`, `String`, `Map`, `Set`, `Bytes`, or `BE Int`/`BE Long`.
