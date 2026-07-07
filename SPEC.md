# SPEC: `lucene-arrow` — GPU-accelerated Lucene segment ⇄ Arrow serde

> Working name, change it. A Rust library that reads and writes Apache Lucene
> index segments directly to/from Apache Arrow, outside the JVM, with GPU
> acceleration for both directions and an Arrow Flight front door.
> Single source of truth: scope, contracts, GPU design, milestones.

---

## 1. What this is

Lucene's on-disk segment format is the storage layer underneath Elasticsearch
and OpenSearch. This project treats a Lucene segment as a **serialization
format** — like Parquet — and provides fast, JVM-free, GPU-accelerated
conversion between segments and Arrow:

- **Reader**: segment files → Arrow RecordBatches, streamed over Arrow Flight,
  for analytics in cuDF/cuPy/Arrow-native tools. Includes scan-once over
  segment sets larger than GPU memory.
- **Writer**: Arrow → finished, `CheckIndex`-clean segment files. An offline
  ingest path that bypasses the Elasticsearch/OpenSearch `_bulk` pipeline.

Read and write are **co-equal**. The CPU executor is the correctness reference
and portability fallback; the GPU path is the performance story in both
directions.

### Why it's tractable
Lucene is already half-columnar. Doc values are a column store; flat vectors
are contiguous float32 blobs. Those map onto Arrow almost directly. We
deliberately avoid the parts that don't (full-text postings/analyzers, BKD
trees) in v1.

### The one hard rule about merges
**Merges are out of scope. Permanently.** Merge-like work happens in
Arrow/GPU land downstream (that's the point of the project). This library
never implements `IndexWriter` semantics, merge policy, or a merge scheduler.
If a task starts to look like "reconcile two segments into one," stop —
that's the consumer's job.

### Target ecosystem reality (OpenSearch)
- OpenSearch 3.x runs on **Lucene 10**, so doc values / stored fields /
  postings / points in modern OS segments are standard Lucene 10 format.
- OS vector fields depend on the k-NN engine: **Lucene engine** = native
  Lucene vector format (our normal path); **Faiss engine (the OS default)** =
  native Faiss index files wrapped in an OS custom codec — *not* Lucene
  format. Handled by a separate read-only `faiss-source` module (§7.6).
  NMSLIB is dead (removed in OS 3.0); ignore it.
- OS 3.0 "derived source" strips vectors from `_source`; vector values live
  in the vector storage, which is where we read them anyway.

---

## 2. Non-goals (v1)

Do not build these. Each is a tar pit we are intentionally skipping:
- **Full-text postings + analyzers.** Matching ES/OS analyzers byte-for-byte
  is open-ended. Keyword/numeric/vector fields need no analysis; those are
  our world. (Postings *reading* as a text-analytics relation: post-v1, §7.8.)
- **BKD points.** The tree is the value; flattening discards the index.
- **HNSW graph construction.** v1 reads/writes **flat** vectors only. CAGRA →
  Lucene-HNSW build is a documented stretch goal (§13, P6).
- **OpenSearch-Faiss write path.** That means reimplementing the OS k-NN
  custom codec + training-model metadata. Never in scope.
- **Merge policy / scheduling / IndexWriter semantics / deletes / updates.**
- **Cluster/shard-allocation concerns.** We operate on segment directories.
- **ES mapping semantics** (`_source` reconstruction, aliases, runtime
  fields). We work at the Lucene field level.

---

## 3. Hard constraints (decided)

1. **Language: Rust.**
2. **Codec pin: Lucene 10.x, `Lucene103`** — matches OpenSearch 3.x and the
   Bearing project. Verify Bearing's tracked minor before coding. A format
   port on every Lucene major bump is the standing maintenance cost (same tax
   cqlite pays pinning to Cassandra 'oa'). Document it, price it in.
3. **Reuse Bearing for codec container framing** (headers, CRC footers,
   compound files, `segments_N`/`.si`/`.fnm`, FST term dicts). Our value-add
   is the Arrow bridge and GPU execution, not re-deriving the byte format.
   Assess Bearing's maturity/bus-factor early; budget for forking it.
4. **IO is bandwidth-agnostic.** Decode/encode kernels never know whether
   bytes arrived via mmap, a KvikIO compatibility-mode bounce buffer, or true
   GPUDirect Storage. They see device pointers, full stop.
5. **GPU is feature-gated.** The library builds and passes all correctness
   tests with no GPU present; CPU executors are the reference implementation.
6. **Dev-hardware reality:** consumer GPUs (e.g. RTX 5090) only ever run GDS
   compatibility mode (driver lockout, Linux-only cuFile). Treat true GDS as
   a deployment feature validated on rented data-center hardware; the 5090 is
   the build/correctness machine.
7. **GPU-native dependencies via pixi.** RAPIDS native libraries (`libcuvs`,
   later possibly `libcudf`) are pinned through a committed `pixi.toml` +
   `pixi.lock` (rapidsai + conda-forge channels, per-CUDA-version envs) —
   the Sirius pattern. No system installs, no source builds in CI; Rust
   links against the pixi env prefix.

---

## 4. Design stance & decision taxonomy

Decisions are graded to avoid locking in the wrong thing early:
- **[CONTRACT]** — stable and versioned; breaking it means a `frame_version`
  bump. Kept to the minimum needed for interop.
- **[DEFAULT]** — shipped behavior selectable per request; changeable later
  without breakage because the server echoes resolved config (§8.3).
- **[OPEN(gate)]** — intentionally undecided; a named experiment decides it.
  Current lean recorded, carries no authority. Register in §14.

Anti-lock-in mechanics: every stream carries `lucene.frame_version`; all
metadata in a reserved `lucene.*` namespace; clients tolerate unknown keys;
servers echo effective options so nothing depends on implicit defaults;
decode/encode plans are internal and versioned separately from frames.

---

## 5. Architecture

```
            ┌──────────────────────────────────────────────────┐
            │                   lucene-arrow                     │
            │                                                    │
 segments ─►│ SegmentSource ─► metadata parse (CPU)              │
            │      (§9)             │                            │
            │                       ▼                            │
            │              Decode/Encode Plan (§6)               │
            │                /               \                   │
            │        CPU executor        GPU executor            │
            │        (reference)         (feature: gpu)          │
            │                \               /                   │
            │                 ▼             ▼                    │
            │           Arrow arrays (host / device)             │
            │                       │                            │
            │            cross-segment unify (§7.3, §10.5)       │
            │                       │                            │
            └───────────────────────┼────────────────────────────┘
                                    ▼
                     Arrow Flight (DoGet / DoPut / DoAction)
```
Write is the same diagram mirrored: Arrow → EncodePlan → CPU/GPU encoders →
Bearing container assembly → segment files.

**Execution pattern (the cuIO pattern):** CPU parses the small sequential
block metadata (bit widths, per-block bases, offsets) into a plan; bulk
payload is then decoded/encoded in one pass (CPU or GPU) per the plan. Never
interleave metadata parsing with bulk work.

### Repository layout
```
lucene-arrow/
├── crates/
│   ├── core/          # SegmentSource, plans, metadata parse, extents
│   ├── docvalues/     # doc values ⇄ Arrow
│   ├── vectors/       # flat vectors ⇄ Arrow
│   ├── faiss-source/  # OS Faiss-engine sidecar reader (read-only)
│   ├── cpu/           # reference executors
│   ├── gpu/           # feature "gpu": kernels + KvikIO source
│   ├── flight/        # Flight server (DoGet/DoPut/DoAction)
│   └── codec/         # thin wrapper over Bearing Lucene103 framing
├── harness/           # Java/Lucene golden-file generator + CheckIndex gate
├── benches/
└── SPEC.md
```

---

## 6. The Decode/Encode Plan (shared core — build first)

A `DecodePlan` is a serializable description of how to turn byte ranges into a
typed column, with no Lucene knowledge leaking into executors:

```rust
struct DecodePlan {
    column: FieldId,
    arrow_type: DataType,
    blocks: Vec<BlockDecode>,
    sparse: Option<DisiPlan>,      // IndexedDISI jump table
}
enum BlockDecode {
    Direct      { offset: u64, len: u64, bit_width: u8 },
    DeltaPacked { offset: u64, len: u64, bit_width: u8, base: i64 },
    GcdPacked   { offset: u64, len: u64, bit_width: u8, base: i64, gcd: i64 },
    Table       { offset: u64, len: u64, bit_width: u8, table_off: u64 },
    Monotonic   { offset: u64, len: u64, bit_width: u8, base: i64, avg: f32 },
    Ordinals    { offset: u64, len: u64, bit_width: u8 },
    Raw         { offset: u64, len: u64 },   // flat vectors
}
```
`EncodePlan` is the inverse: per-block encoding choice + payload emission +
the metadata the codec framing needs. CPU and GPU executors consume identical
plans and must produce **bit-identical** output. This abstraction is the
spine: reader, writer, CPU, GPU all hang off it.

---

## 7. Read frame shapes

### 7.1 Core contract
- **Segment-scoped batches [CONTRACT v1].** A RecordBatch covers a contiguous
  local-docid range `[doc_lo, doc_hi)` of exactly one segment; all columns
  doc-aligned (row i = same doc). Makes docid bases, deletes, sparse
  validity, dictionaries locally solvable; cross-segment unification is
  metadata + ordinal remap, not physical stitching. Recorded falsifier: if
  tiny-segment shards make per-batch overhead untenable *at the frame layer*
  (IO coalescing already handles transfers, §10.2), a future frame_version
  may add multi-segment batches with a `_seg` run-length column.
- **System columns [CONTRACT v1]:** `_seg: Int32`, `_doc: Int32`,
  `_global_doc: Int64` (= server `doc_base + _doc`). Deselectable.
  `(_seg,_doc)` is the stable row address for hydration and sidecar joins.
- **Row modes — caller must choose (no implicit default):**
  `compact` = deletes dropped server-side (analytics default instinct);
  `positional` = one row per docid + `_live: Boolean`, preserving positional
  joins against docid-ordered sidecars (Faiss vectors). Null semantics are
  reserved for "field absent for this doc" (sparse validity); deletion is
  never expressed as null. [CONTRACT]
- **Batch metadata [CONTRACT keys]:** `lucene.segment`, `.segment_ord`,
  `.doc_lo/hi`, `.doc_base`, `.max_doc`, `.live_applied`, `.codec`,
  `.frame_version`. Stream-level: shard manifest JSON (ordered segments,
  bases, files, schema fingerprint) + echoed effective config.
- **Batch sizing [DEFAULT]:** 128Ki rows capped 128 MiB encoded, clamped by
  vector width; `batch_rows` request param; bench-swept.

### 7.2 Shapes by component ([CONTRACT v1] shapes; mode selection [DEFAULT])
| Lucene | Arrow | notes |
|---|---|---|
| NUMERIC | `Int64` / `Float64` (+validity iff sparse) | fused DIRECT/delta/GCD/table/monotonic + bit-cast |
| SORTED_NUMERIC | `List<Int64>` | offsets = monotonic-decoded addresses block; single-valued degrades to NUMERIC with `lucene.multi=false` |
| SORTED | `Dictionary<Int32, Utf8>` (+validity) | dict modes §7.3 |
| SORTED_SET | `List<Dictionary<Int32, Utf8>>` (+validity) | ditto |
| BINARY | `Binary` (+validity) | |
| flat vector | `FixedSizeList<Float32\|Int8, dim>` (+validity) | `lucene.vector.similarity`, `.quant={scheme,scale,bias}`; server dequant on request; `Raw` plan → DMA |

### 7.3 Dictionaries: whose ordinals? (per column per stream)
- **`global`** — server builds an OrdinalMap across the stream's segments,
  emits one dictionary batch up front, remaps ordinals on the fly. One
  coherent column across batches.
- **`segment`** — per-segment dicts via Arrow IPC dictionary replacement at
  segment boundaries; no merge cost; clients handle swaps.
- **`none`** — materialize `Utf8`/`Binary` via term-byte gather.
Field metadata: `lucene.dict = global|segment|none` (+cardinality).
Selection heuristic **[OPEN(gate: OrdinalMap bench on real high-cardinality
shards)]**; lean: global for small dicts, segment for large.

### 7.4 Stored fields — hydration RPC, never in the scan stream
`DoAction: hydrate` takes `(_seg,_doc)` pairs → requested stored columns.
Row-oriented block-compressed data; analytic role is hydrating filtered
results. GPU path (nvCOMP batched LZ4/zstd) **[OPEN(gate: hydration-volume
profile)]**; CPU LZ4 until then.

### 7.5 Live docs, norms, field infos
`.liv` → validity/compaction machinery (never skip: aggregates over
tombstones are silent corruption). Norms: small numeric column, expose
on request. `.fnm` field infos: the Arrow schema source.

### 7.6 Faiss sidecar vectors (OS Faiss engine) — `faiss-source` module
Not Lucene data; parses OS k-NN native index files. Read-only. Two modes:
- **`vectors`** — extract flat/IVF-flat storage, emit §7.2 vector shapes
  keyed to docid (engine-invisible downstream). PQ exposed as codes+codebook
  only; no reconstruction.
- **`handle`** — no decode; manifest pointing at the index file for direct
  cuVS consumption (cuVS reads Faiss natively — often the better ANN path).
This is the only module tracking OS's custom k-NN codec.

### 7.7 Analytical surface (what each component buys; scoping rationale)
Strong + columnar (v1): NUMERIC/SORTED* doc values → filters, aggregates,
group-by, top-N on pre-dictionary-encoded keys; flat vectors → exact KNN,
similarity, clustering on GPU (independent of any HNSW graph). Different
shape (post-v1): postings/term-vectors → term-doc matrix analytics (§7.8).
Retrieval-only: stored fields (hydration). Ignorable for analytics: BKD
(numeric fields are also doc values), HNSW graph (serving concern; flat
vectors give exact answers).

### 7.8 Postings relation (post-v1, text-analytics mode)
Separate relation, same segment-scoped invariant:
`term (dict) | doc Int32 | freq Int32 | pos List<Int32>?` — COO term-doc
matrix for GPU sparse linalg. Requires FST term dict access (CPU).

---

## 8. Flight protocol

### 8.1 Read
```
GetFlightInfo  descriptor.cmd = ReadRequest (JSON):
 { "path": ..., "segments": [...]|"all", "columns": [...],
   "row_mode": "compact"|"positional",          // required
   "dict": {"col": mode, "*": mode},
   "batch_rows": N, "executor": "auto"|"cpu"|"gpu", "frame_version": 1 }
 → reconciled schema + shard manifest + endpoints (segments grouped to
   ≥ ~1 GiB payload per endpoint for parallel DoGet [DEFAULT]).
DoGet(ticket) → [DictionaryBatch*] then RecordBatches, segment order, docid order.
DoAction("hydrate", {pairs, columns}) → stored-field batches.
DoAction("stats", segments) → per-field decode-cost estimates (planner food).
```
Field names [CONTRACT]; values [DEFAULT].

### 8.2 Schema reconciliation [CONTRACT]
Field present-in-some-segments → column all-null where absent. True type
conflict → fail GetFlightInfo loudly, offending segments named. No coercion.

### 8.3 Echoed effective config [CONTRACT]
The server resolves every unspecified option and echoes the full resolved
request in FlightInfo/PutResult app_metadata. Clients are forbidden (by
documented contract) from relying on unechoed defaults — this is what makes
every [DEFAULT] changeable without a compatibility event.

---

## 9. IO: `SegmentSource`

```rust
trait SegmentSource { fn open(&self, file: &str) -> Result<Box<dyn ByteRange>>; }
trait ByteRange {
    /// Fetch [offset, offset+len) into host or device memory.
    /// The impl decides DMA vs bounce vs mmap.
    fn read_into(&self, offset: u64, len: u64, dst: BufferTarget) -> Result<()>;
}
```
Impls: `MmapSource` (default, no GPU); `KvikioSource` (true GDS on supported
hardware; transparent pinned-bounce compat mode on consumer GPUs — same code
path). Write side mirrors with `write_from` (cuFileWrite where available,
pinned staging otherwise).

---

## 10. Write frame shapes

### 10.1 Job framing (field names [CONTRACT]; values [DEFAULT])
```
DoPut  descriptor.cmd = WriteJob:
 { "output_dir": ..., "codec": "Lucene103",
   "segment_max_docs": N, "segment_max_bytes": B,
   "index_sort": [{"field":...,"order":...}],     // optional
   "coercion": "strict"|"auto",
   "compound": bool, "executor": ..., "frame_version": 1 }
 ← PutResult per flushed segment:
   SegmentManifest{name, max_doc, files:[{name,bytes,crc32}], field_stats, wall_ms}
```
Docids assigned in arrival order; a batch belongs to exactly one segment
(write-side mirror of §7.1) [CONTRACT]. Segments cut on thresholds, never
mid-batch. Output segments are **all-live: no deletes, no updates, no
merges** [CONTRACT — scope]. `index_sort` sorts per segment only (GPU sort,
§11.8); global order across segments is the client's job — consistent with
"merges live upstream."

### 10.2 Arrow → Lucene: field mapping & acceptance matrix

Lucene-directed intent rides on Arrow field metadata [CONTRACT keys v1]:
`lucene.type = numeric|sorted|sorted_set|sorted_numeric|binary|vector`;
`lucene.field`; `lucene.vector.similarity`; `lucene.vector.encoding =
float32|int8`; `lucene.points` (post-v1, default false). When `lucene.type`
is absent it is inferred from the Arrow type per the matrix below.

Real Arrow data (cuDF output, Parquet loads) rarely arrives in canonical
shapes, so the writer runs a **normalization stage** before encode. Coercion
policy is per-job (`"coercion": "strict"|"auto"` [DEFAULT: auto], echoed per
§8.3): `strict` accepts canonical shapes only; `auto` also applies the
lossless coercions. Lossy conversions are never automatic — explicit metadata
only. Every coerced field records `lucene.source_type`; the round-trip
contract is `read(write(T)) == canonical(T)` — canonical inputs round-trip
exactly [CONTRACT], coerced inputs come back as their canonical type.

| Arrow input | Lucene target | class |
|---|---|---|
| `Int64`, `Float64`, `Dictionary<Int32,Utf8>`, `List<Int64>`, `List<Dictionary>`, `Binary`, `FixedSizeList<f32 or i8, d>` | per read table (§7.2) | canonical — pass-through |
| `Int8/16/32`, `UInt8/16/32`, `Boolean` (as 0/1) | NUMERIC (widen to i64) | lossless auto |
| `Float32` | NUMERIC (Float64) | lossless auto |
| `Timestamp(unit, tz)`, `Date32/64`, `Time32/64`, `Duration` | NUMERIC (i64; unit+tz kept in `lucene.source_type`) | lossless auto |
| `Utf8` / `LargeUtf8` | SORTED (dict-encode on the fly) | lossless auto |
| `List<Utf8>` | SORTED_SET (dict-encode) | lossless auto |
| `LargeBinary` | BINARY | lossless auto |
| `UInt64` | NUMERIC | explicit only (`lucene.allow_lossy`): values > i64::MAX unrepresentable |
| `Decimal128/256` | NUMERIC via scaled long | explicit only (`lucene.scale_factor`; ES scaled_float convention) |
| `FixedSizeList<f64 or f16, d>` | vector (cast to f32) | explicit only (representation change) |
| `Struct`, `Map`, `Union`, nested lists | — | rejected v1 (no doc-values analogue) |

Null semantics: Arrow nulls = "field absent for this doc" → sparse doc
values (DISI), exactly mirroring the read side. Unknown metadata keys and
rejected types fail at schema time, not mid-stream [CONTRACT].

### 10.2b What the written segments can serve (searchability)
v1 writes doc values + flat vectors only — no postings, no points, no norms.
Consequence for a cluster serving these segments: every field is sortable and
aggregatable; term/range **filtering works only via doc-values query
execution** (ES/OS can query non-indexed, doc_values-enabled fields —
correct but linear; no postings/BKD fast path); vector search is exact/flat
until P6 adds HNSW. That's the right trade for analytics-shaped data, but
document it loudly; `lucene.points = true` (post-v1) is the opt-in that
restores fast range filters. Mapping compatibility with the target index
(field names/types) is the operator's responsibility — we write Lucene
fields, not ES/OS mappings (§2).

### 10.3 Dictionary columns on write
Lucene SORTED requires byte-ordered term dicts; Arrow dicts are unordered.
Writer sorts the dictionary per segment (GPU string sort), remaps ordinals
through the permutation. Cross-batch dictionary deltas within a segment are
unioned at flush (write-side mini-OrdinalMap).

### 10.4 Vectors on write
v1: flat `Float32|Int8` → Lucene flat vector storage (GPU-packed buffers +
field metadata; optional fused quantize kernel). HNSW build = stretch P6
(cuVS CAGRA → Lucene HNSW; crib Elastic's cuVS plugin — flush-time GPU
build, gpu-ingest/cpu-serve tiering, ~12x index / ~7x force-merge reported).
No OS-Faiss write path (§2).

### 10.5 Container assembly & integrity
Encoders produce payload buffers + per-block metadata; Bearing's Lucene103
writers assemble files (headers, FST term dicts, footers, .si/.fnm, .cfs).
**Acceptance: Java `CheckIndex` passes on every output segment.** CRC32
placement (CPU-during-stream-out vs GPU slice-by-N + crc32_combine)
**[OPEN(gate: write profile — decide if assembly stage >10% of wall)]**,
lean CPU. FST construction stays CPU (inherently sequential; not open).

---

## 11. GPU execution design

### 11.0 Performance regimes — design for the envelope, not one point
| regime | effective input BW | bottleneck | kernels matter? |
|---|---|---|---|
| cold scan, 1× consumer NVMe | 7–14 GB/s | disk | overlap efficiency |
| cold scan, RAID / multi-NVMe / GDS array | 30–100+ GB/s | moving | yes |
| warm re-read (page cache) | 20–80 GB/s | memcpy/decode | yes |
| device-resident pipeline (decode feeds GPU compute in place) | VRAM-class | decode + compute share | emphatically |
| write path (encode → NVMe) | drive write BW | encode vs IO overlap | yes |

Design rule: kernels are written to be good everywhere (fused, single-pass,
batched launches); the pipeline hides whichever stage isn't the bottleneck.
Every bench reports **both** raw kernel GB/s (device-resident input, kernels
isolated) and sustained storage utilization. Neither alone is "the" metric.
[CONTRACT — bench methodology]

### 11.1 Execution model (both directions)
```
read:   metadata → DecodePlans → EXTENTS → [IO] → [decode] → [deliver]
write:  layout   → EncodePlans → [analyze] → [encode] → [assemble + IO out]
```
- 3-stage stream pipeline, events chaining stages, ≥4 extents in flight;
  pooled device allocator (RMM-style arena per ticket). Working set is
  O(pipeline_depth × extent), never O(shard) → larger-than-VRAM by
  construction.
- Per-extent kernel DAGs are shape-identical → capture once, replay with new
  pointers. Mechanism **[OPEN(gate: launch-overhead microbench on
  many-small-segment shards)]**: CUDA graphs vs persistent kernels vs plain
  batched launches. Lean: CUDA graphs.
- Write runs the pipeline in reverse with double-buffered assembly: GPU
  encodes extent N+1 while CPU frames/checksums/writes extent N.

### 11.2 Extents & coalescing (solves many-small-segments below the frame layer)
- **Extent** = contiguous byte range of one file: one column over one segment
  (or slice), including interleaved block metadata. 4–64 MiB [DEFAULT],
  4 KiB-aligned, one cuFileRead/pread (or write) per extent.
- Extents from different segments for the same column batch into **one**
  launch via a device-side descriptor table
  `{src, block_meta_off, n_blocks, dst_off, seg_ord}`; grid iterates
  descriptors (cuDF's batched-fragment trick). **Anti-goal: per-Lucene-block
  or per-segment kernel launches.**
- Compat-mode transfers use a pinned host ring (2× extent × depth), always
  `cudaMemcpyAsync`, never pageable, never synchronous.

### 11.3 Decode kernels (doc values) — fused single-pass
unpack → arithmetic epilogue → cast → store; one read of packed bytes, one
write of Arrow lanes.
- **Bit-unpack (PackedInts/DIRECT):** warp-cooperative; 128-bit `uint4`
  loads; funnel-shift extraction across lanes; compile-time specializations
  for widths {1,2,4,8,12,16,20,24,28,32,40,48,56,64} + generic fallback;
  width from block metadata.
- **Delta:** unpack + CUB BlockScan + per-block base. Lucene delta blocks are
  self-contained (base per block) → zero cross-block dependency;
  "sequential" decode is embarrassingly parallel.
- **GCD / table / monotonic / float:** pure fused epilogues
  (`v = base+gcd*x`; `v = table[x]`; `v = base+round(avg*i)+d_i`; int→float
  bit-cast). No extra passes.
- **Sparse (IndexedDISI):** DENSE blocks memcpy into Arrow validity words;
  SPARSE lists scatter; popcount prefix-sum gives value↔position mapping for
  gathers/offsets. No dense intermediate unless output is dense.
- **Ordinals:** unpack + optional fused gather (remap table in global mode;
  term-byte gather to Binary in none mode).
- **Multi-valued:** addresses block monotonic-decodes directly *as* the Arrow
  List offsets buffer.

### 11.4 Compact-mode gather
`.liv` → device; popc+scan → selection prefix; gather fused into the decode
store (`dst = sel[i]`, dead lanes skipped) — no separate compaction pass.

### 11.5 GPU OrdinalMap (dict=global)
Per-segment sorted term dicts on device → k-way merge-path (log₂k rounds of
segmented merges, device string compares vs shared byte blobs) → global
unique terms + per-segment `i32` remap tables, stream-resident. Cost ∝
dictionary size — hence the §7.3 gate on global-vs-segment crossover.

### 11.6 Vectors
Flat f32: DMA, no kernel. Quantized: fused dequant (`char4` loads) only when
Float32 requested; fused quantize on write. KNN/similarity is the consumer's
job (cuVS/cuBLAS); the P2 demo wires cuVS to our device buffers to prove
zero-copy handoff. **FlatKnn (our hand-rolled exact scorer) stays** as the
dependency-free reference; cuVS brute-force and CAGRA run against the same
device buffers and all three are benchmarked head-to-head (recall + QPS +
build time) in the §15 matrix. Tensor-core layout games are delegated to
those libraries.
**cuTile: not used** (Python-first, tensor-core-oriented; our kernels are
integer/memory-bound) — [OPEN(revisit: cuTile C++/Rust target ships)].

### 11.7 Encode kernels (write) — first-class mirrors of §11.3
- **Stats pass:** segmented reduce per block (min/max/GCD, distinct sketch)
  in one sweep. Encoding-selection policy placement **[OPEN(gate: write
  profile)]**, lean CPU-policy/GPU-execute for debuggability.
- **Bit-pack:** inverse funnel-shift kernel, same width specializations.
- **Delta encode:** adjacent-difference + per-block base, fused with pack.
- **Dictionary build:** device radix sort of dict entries + ordinal
  permutation remap (§10.3).
- **Index sort:** per-segment radix sort on key → one permutation → gathers
  fused into each column's pack-kernel load.
- **Normalization casts (§10.2):** widen / bit-cast / timestamp-to-i64 are
  trivial fused prologues on the pack kernels; `Utf8` → SORTED dict-encode on
  device = hash + radix sort + unique + ordinal scatter (same machinery as
  §10.3's dictionary build).
- **Sparse encode:** validity bitmap → DISI blocks (ALL/DENSE/SPARSE per
  block from popcounts).
- **On-GPU round-trip fuzzing:** `decode(encode(x)) == x` entirely on device
  as a fast CI property test — catches encoder/decoder drift cheaply;
  complements, never replaces, CheckIndex and golden files.

### 11.8 Deliberately CPU
FST term dict construction; file/container framing; codec metadata (all
sequential, cheap, and Bearing's). CRC32: open (§10.5).

---

## 12. Correctness strategy

Segment formats have no "close enough."
1. **Golden files:** `harness/` Java program (Lucene 10.x) writes segments
   with known values — numeric, sorted, sorted-set, binary, flat f32 and int8
   vectors, sparse fields, a segment with deletes, a multi-segment shard.
   Expected decoded values committed alongside.
2. **Round-trip:** `read(write(arrow)) == arrow` for canonical shapes;
   coerced inputs compare post-normalization (`== canonical(arrow)`, §10.2).
   Every written segment passes Java `CheckIndex`. Golden set includes one
   "messy" table (Int32, Timestamp, plain Utf8, Boolean, nulls) exercising
   the coercion matrix end to end.
3. **Differential:** CPU and GPU executors bit-identical on every golden
   (CI GPU runner; graceful skip without one). Plus §11.7 on-GPU fuzz.
4. **Cross-validate vs Bearing** where formats overlap.

---

## 13. Milestones (read+write in lockstep per component, GPU from the start)

- **P0 — open a segment:** parse `segments_N`/`.si`/`.fnm`/`.cfs` via codec
  wrapper; list fields, DV types; assert Lucene103. *Done:* print a real
  segment's field inventory.
- **P1 — numerics:** DV numeric **decode and encode**, CPU reference + GPU,
  plans + extents + pipeline skeleton. *Done:* round-trip green, CheckIndex
  green, bench rows (both metrics) recorded.
- **P2 — vectors:** flat vector read + write, DMA/dequant/quantize kernels,
  cuVS zero-copy KNN demo on a real shard's vectors, `faiss-source` handle
  mode. *Done:* "point at an OS shard, GPU KNN, no JVM" works on the 5090
  (compat mode fine).
- **P3 — dictionaries:** SORTED/SORTED_SET read (all three dict modes) +
  write; GPU OrdinalMap; resolve the dict-mode gate. *Done:* high-cardinality
  bench decides §7.3.
- **P4 — surface:** Flight DoGet/DoPut end-to-end, hydration RPC, config
  echo, multi-segment streams; Python cuDF client pulls a shard as one
  logical stream.
- **P5 — verdict:** run the full §15 matrix incl. kill-criterion baselines;
  resolve remaining OPEN gates from evidence.
- **P6 — GPU vector search + HNSW build (ACTIVE; P5 is green):**
  (a) pixi env with `libcuvs`; spike the `cuvs` Rust crate (resolves reg.
  #14). (b) cuVS brute-force + CAGRA over our device-resident vector
  buffers; keep FlatKnn as reference; three-way bench (recall/QPS/build).
  (c) **`.vem`/`.vex` writer fed by CAGRA** (Lucene99HnswVectorsFormat;
  CAGRA's single-layer graph is a valid HNSW instance — Elastic's
  conversion trick). *Done:* CheckIndex passes AND a Java Lucene KNN query
  over our written segment returns top-k matching our GPU search; perf
  compared against JVM HNSW flush-time indexing. This also un-defers
  vector *writing* generally (Java-servable segments need the graph).
- **P7 — postings relation (§7.8):** after P6. Sequential `.tim` block-tree
  term walk (no FST needed for full enumeration) → `.doc` FOR/PFor block
  decode (CPU reference via Bearing's public `pfor`, then GPU) → COO
  term-doc Arrow relation, segment-scoped per §7.1; golden-tested against
  Java-written text segments.

**Go/no-go after P2:** P0–P2 is a few-weeks proof. If the P5-style spot
benchmarks at that point show export-to-Parquet decisively winning realistic
workflows, stop and write it up — the learning was the floor return.

---

## 14. Decision register

| # | topic | status | current lean | decided by |
|---|-------|--------|--------------|-----------|
| 1 | segment-scoped batches | CONTRACT v1 | — | frame_version bump only |
| 2 | row mode | explicit param, no default | — | never needs deciding |
| 3 | dict mode heuristic | OPEN | global small / segment large | P3 OrdinalMap bench |
| 4 | batch sizing | DEFAULT | 128Ki / 128 MiB | bench sweep |
| 5 | CRC32 placement | OPEN | CPU | write profile (>10% assembly) |
| 6 | encode-policy placement | OPEN | CPU policy / GPU execute | write profile |
| 7 | launch mechanism | OPEN | CUDA graphs | launch microbench |
| 8 | nvCOMP stored fields | OPEN | defer | hydration-volume profile |
| 9 | cuTile | OPEN | not now | C++/Rust target ships |
| 10 | endpoint grouping | DEFAULT | ~1 GiB | bench sweep |
| 11 | no merges/deletes on write | CONTRACT (scope) | — | never |
| 12 | round-trip symmetric shapes | CONTRACT v1 | — | frame_version bump only |
| 13 | KvikIO binding (existing FFI vs own libcufile wrapper) | OPEN | evaluate first | P1 spike |
| 14 | cuVS from Rust (bindings vs C++ FFI shim) | OPEN | `cuvs` crate against pixi-provided `libcuvs` | P6a spike (in progress) |
| 15 | quantized vectors in P2 | OPEN | f32 first, int8 fast-follow | P2 scope check |
| 16 | write coercion policy | DEFAULT | `auto` (lossless only; lossy always explicit) | echoed config; `strict` available |

---

## 15. Bench matrix [CONTRACT — methodology]

Per column type × segment profile (many-small / few-large) × regime (§11.0)
× executor (CPU-SIMD, GPU-bounce, GPU-GDS):
1. Raw kernel decode/encode GB/s from device-resident input.
2. Sustained storage utilization (% of drive read/write BW converted).
3. End-to-end: open → Arrow (rows/s, GB/s); Arrow → CheckIndex-clean segment.
4. **Kill-criterion baselines:** (a) JVM Lucene DocValues scan;
   (b) scroll-export-to-Parquet + cuDF read — the competing read workflow,
   measured honestly including export; (c) JVM IndexWriter bulk ingest of
   the same Arrow data — the competing write workflow.
5. Correctness gates in the same harness (§12), every run.

---

## 16. Prior art (read before coding)

- **Bearing** (`toddfeak/bearing`) — Rust port of Lucene 10.3.x, Lucene103
  codec, byte-identical output, cross-validated vs Java. Our codec framing
  layer; read first; assess bus factor.
- **rucene** (`zhihu/rucene`) — Rust Lucene 6.2.1; served Zhihu's production
  search since 2018; proof the reimplementation path works, format ancient.
- **Elastic Search Labs cuVS posts** — GPU vector indexing blueprint:
  flush-time CAGRA build → HNSW convert, gpu-ingest/cpu-serve tiering,
  ~12x indexing / ~7x force-merge. Design doc for P6.
- **cuDF cuIO Parquet reader** — the CPU-plans/GPU-executes architecture and
  batched-fragment decode we copy.
- **cqlite** (`pmcfadin/cqlite`) — the structural analogue (SSTables outside
  the cluster); note what it punted on (writes, compaction) and why.
- **Tantivy** — Lucene-inspired, own format; Rust idioms only, no format
  compatibility.
- **jVector** (`datastax/jvector`) — Java ANN engine behind Cassandra 5
  SAI: HNSW-style hierarchy with Vamana (DiskANN) construction per layer,
  PQ-compressed vectors in memory + full-precision rerank from disk. The
  "graph as own on-disk format beside the store" pattern — same family as
  the OS-Faiss sidecar we read via §7.6.
- **Lance / LanceDB** (`lance-format/lance`) — ML-native columnar format
  (Rust): ~100× Parquet random access, adaptive encodings, IVF_PQ/HNSW
  indexes stored alongside data, dataset versioning. The strongest
  "export target" competitor for our §15(b) workflow comparisons — if
  export ever beats direct segment reads, it likely exports to Lance, not
  Parquet.
- **Amazon S3 Vectors** — storage-first serverless vector tier (GA 2026,
  up to ~2B vectors/index, ~100ms warm queries). Internals unpublished;
  the latency/economics profile implies partition-based (IVF-like)
  coarse search + rerank rather than graph traversal (object storage
  penalizes the random reads graphs need). Relevant as the "cold vector
  tier" competitor shape; also pairs with OpenSearch for hot serving —
  the same tiering story our segments-to-Arrow reader plays in.
