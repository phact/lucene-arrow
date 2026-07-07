# Reference Sources

This directory contains reference source code used during development.

## Setup

Run the download script to fetch all reference sources:

```bash
./reference/download-references.sh
```

This downloads:
- **Apache Lucene 10.3.2** source from the Apache archive into `reference/lucene-10.3.2/`
- **Assertables 9.8.6** source from crates.io into `reference/assertables/`

## Rules

- **Source of truth**: All Rust code should be derived from the Java source in `reference/lucene-10.3.2/lucene/`.
- **Do not modify**: Do not modify any files under `reference/lucene-10.3.2/` without explicit direction from the user.
- **Re-download**: If the source becomes corrupted or needs a clean state, delete `reference/lucene-10.3.2/` and re-run the download script.

## Format Reference Docs

`reference/formats/` contains byte-level encoding documentation for all Lucene file formats used by our target codec (Lucene103). These are a convenience reference — the Java implementation code is always the source of truth.

| Document | Contents |
|---|---|
| `MAINTAINING.md` | **Lucene 10.3.2 codec version mapping**, trust hierarchy, verification checklist |
| `encoding-primitives.md` | DataOutput/DataInput methods, VInt/VLong, ZigZag, CodecUtil headers/footers |
| `lucene90-formats.md` | Stored fields, doc values, norms, BKD, compound, IndexedDISI, DirectMonotonic, DirectWriter |
| `lucene94-formats.md` | Field infos (.fnm) — enum byte mappings, field bits flags |
| `lucene99-formats.md` | Segment info (.si) — version encoding, YES/NO flags |
| `lucene103-formats.md` | Postings, terms dictionary, FOR/PFor encoding, skip data |

Start with `MAINTAINING.md` for the codec version mapping table — it shows which format version each component uses and which doc to consult.

## Key Paths

| Path | Purpose |
|---|---|
| `reference/lucene-10.3.2/lucene/core/src/java/org/apache/lucene/` | Lucene main source code |
| `reference/lucene-10.3.2/lucene/core/src/test/java/org/apache/lucene/` | Lucene test source code |
| `reference/assertables/src/` | Assertables macro source code |

## Java Package Guide

Key packages under `.../src/java/org/apache/lucene/`:

| Package | Purpose |
|---|---|
| `index` | Core indexing: `IndexWriter`, `IndexReader`, `DocumentsWriter`, `SegmentInfo`, `Terms`, `PostingsEnum` |
| `document` | `Document`, `Field`, `FieldType`, `IndexOptions`, `DocValuesType` |
| `store` | `Directory`, `IndexInput`, `IndexOutput`, `Lock`, `IOContext` |
| `codecs` | `Codec` and all sub-format interfaces for encoding/decoding index data |
| `analysis` | `Analyzer`, `TokenStream`, `Tokenizer`, `TokenFilter` |
| `util` | Shared utilities: `BytesRef`, `Bits`, `FixedBitSet`, `InfoStream`, packed integers, FST, automaton |
