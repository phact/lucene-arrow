# Known Issues

Outstanding problems and optimization gaps in the indexing pipeline.

---

## 1. Peak RSS Higher Than Java

**Severity:** Medium — Rust peak RSS is higher than Java for the same workload

Rust peak RSS during indexing is larger than Java's. Two known contributing factors:

1. **Memory fragmentation** — partially addressed by switching to jemalloc, which returns freed pages to the OS more aggressively than glibc malloc.
2. **Higher default RAM buffer** — Rust defaults to 64 MB vs Java's 16 MB. Lowering the Rust default would reduce peak RSS but dramatically increases segment count because Rust does not yet merge segments during indexing.

**Next steps:** Implement segment merging during indexing (matching Java's `IndexWriter` merge policy). Once merging is in place, reduce the default RAM buffer to match Java's 16 MB. Then remeasure and profile for additional memory optimization opportunities.

---

## 2. Incomplete SmallFloat Port

**Severity:** Low — missing methods not yet needed

`src/util/small_float.rs` is a partial port of `o.a.l.util.SmallFloat`. The integer encoding methods (`longToInt4`, `int4ToLong`, `intToByte4`, `byte4ToInt`) are ported and working. Four float encoding methods are missing:

- `floatToByte(float, numMantissaBits, zeroExp)` — generic float-to-byte encoder using IEEE 754 bit manipulation
- `byteToFloat(byte, numMantissaBits, zeroExp)` — the reverse
- `floatToByte315(float)` — specialization with mantissa=3, zeroExp=15
- `byte315ToFloat(byte)` — the reverse

Additionally, `intToByte4` silently returns 0 on negative input instead of panicking (Java throws `IllegalArgumentException`).

**When this matters:** The missing float methods are used by the similarity/scoring layer (e.g., `BM25Similarity`) to encode and decode norm values as floats. They will be needed when porting similarity implementations for query-time scoring. The integer methods used by norms during indexing are already correct.

**Fix:** Complete the port following porting rules. Port the corresponding tests from `o.a.l.util.TestSmallFloat`.

---

## 3. No Payload Support in Write Path

**Severity:** Feature gap — fields with payloads cannot be indexed

The postings writer (`PostingsWriter`) does not support writing payloads. There is no payload length buffering, payload byte accumulation, or payload encoding in the `.pay` file. Offset support was added but payload branches remain unimplemented. The `payloadByteUpto` field in skip data is hardcoded to 0.

This affects any field that uses payloads (e.g., `TermVectorOptions` with payloads).

**Fix:** Port the payload branches from Java's `Lucene103PostingsWriter` — `payloadLengthBuffer` buffering, payload byte accumulation, and the `.pay` payload encoding in `addPosition`/`finishTerm`. The `.pay` file infrastructure and skip data plumbing are already in place from offset support.

---

## 4. Decomposed IndexOptions Booleans

**Severity:** Low — code clarity, no functional impact

Several structs store decomposed `IndexOptions` as individual booleans (`has_freq`, `has_prox`, `has_offsets`, `write_freqs`, `write_positions`) instead of a single `IndexOptions` enum. This allows invalid states (e.g., offsets without positions) and requires adding a new boolean parameter each time a new indexing tier is supported.

Known locations:

- `BlockFlushState` (`postings_writer.rs`) — `write_freqs`, `write_positions`
- `encode_term` (`postings_writer.rs`) — `write_positions`, `write_offsets` parameters
- `FreqProxPostingsArray::new` (`terms_hash.rs`) — `write_freqs`, `write_offsets` parameters
- `FreqProxTermsWriterPerField` (`terms_hash.rs`) — `has_freq`, `has_prox`, `has_offsets` (cached derivations alongside `index_options` field)
- `FieldInfoGlobal` (`field_infos.rs`) — `has_freq`, `has_prox`, `has_offsets`
- `StatsWriter` (`blocktree_writer.rs`) — `write_freqs`

**Fix:** Replace boolean fields/parameters with `IndexOptions` and use `has_freqs()`/`has_positions()`/`has_offsets()` at call sites. Hot-path code may cache derived booleans in local variables or function parameters, but not in struct fields — struct fields add storage overhead and duplicate state.
