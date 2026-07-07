# lucene-arrow

GPU-accelerated Lucene segment ⇄ Apache Arrow serde, no JVM.

See concept at [`SPEC.md`](SPEC.md)

Lucene's on-disk segment format (the storage layer under Elasticsearch and
OpenSearch) is treated as a serialization format — like Parquet — with fast,
JVM-free conversion to and from Arrow. Doc values become Arrow columns; flat
vectors become `FixedSizeList` arrays; an Arrow Flight front door streams
shards to cuDF/cuPy. Leveraged bearing when possible for fast CPU multithreaded
Rust implementation. Measured against JVM for correctness.

## Status

| milestone | state |
|---|---|
| **P0 — open a segment** | ✅ `SegmentDirectory` parses `segments_N`/`.si`/`.fnm`/`.cfs` (Lucene103 pinned), typed field inventory, `segment_info` example prints a real segment |
| **P1 — numerics (CPU)** | ✅ NUMERIC doc values decode **and** encode. Decode: all Lucene90 encodings (constant, delta, GCD, table, **multi-block**, sparse DISI incl. DENSE rank blocks). Encode: **byte-identical to Bearing** (transitively byte-identical to Java Lucene) — proven in tests |
| **P1 — numerics (GPU)** | ✅ NVRTC funnel-shift kernel, descriptor-table batched launches (§11.2/11.3), bit-identical to CPU on synthetic sweeps **and** Java golden segments. 52–70 Gval/s (device-resident) on an RTX 5090 — see benches |
| **Golden files + CheckIndex** | ✅ real Java Lucene 10.3.2 segments in `harness/golden` (numerics, **multi-block**, deletes, multi-segment), all CheckIndex-clean, decoded & verified in `crates/codec/tests/golden.rs` |
| **P2 — flat vectors** | ✅ read/write of Lucene99FlatVectorsFormat (`.vemf`/`.vec`), float32 + int8, dense/sparse, doc-aligned `FixedSizeList` decode, GPU DMA payload path; validated against real Java `KnnFloatVectorField`/`KnnByteVectorField` segments. **Demo criterion met**: `--example gpu_knn` opens the Java-written golden segment and runs exact GPU KNN (CPU-verified) with no JVM — 1M×128d scan at 330 GB/s / 1.6 ms per query (`--bench knn_scale`). cuVS binding spike (reg. #14) and `faiss-source` still open |
| **`.fnm` self-parse** | ✅ first-party Lucene94FieldInfos parser — vector dim/encoding/similarity + skip-index flag (all discarded by Bearing) now populate `FieldMeta` |
| **P3 — dictionaries (read)** | ✅ SORTED / SORTED_SET (single + multi) / multi-valued SORTED_NUMERIC: LZ4 terms-dict materialization, ordinal decode, `Dictionary<Int32,Utf8>` / `List<Dictionary>` / `List<Int64>` — validated against Bearing **and** a real 3-segment Java index. ✅ `dict=global`: OrdinalMap k-way merge + remap fused into the Table epilogue (same kernel on GPU, differential-tested). **Register #3 resolved by bench**: merge runs 40–50 M terms/s/thread → default `global` below ~1 M total terms (≤ ~25 ms), `segment` above. ⬜ dictionary write path (GPU string sort, §10.3) |
| **P4 — Flight surface (read)** | ✅ `GetFlightInfo` + `DoGet` end-to-end over TCP: schema reconciliation (§8.2), effective-config echo (§8.3), segment-scoped batches with `_seg`/`_doc`/`_global_doc`, **both row modes against real tombstones** (`positional` + `_live`, `compact` server-side filter via `.liv`), `dict=segment` via IPC dictionary replacement, column projection, loud failures. e2e-tested against the Java golden shards. ✅ `DoAction: hydrate` — stored columns for `(_seg,_doc)` pairs, IPC-encoded, pair order preserved (§7.4). ✅ **GPU behind the engine** (`--features gpu`): `executor: auto|cpu|gpu` resolves honestly (echoed per §8.3), numeric columns decode on-device, wire output byte-identical to CPU (gated in `flight_gpu.rs`). ⬜ `DoPut`, stats action, multi-endpoint grouping |
| **`.liv` + BINARY** | ✅ first-party Lucene90LiveDocs reader (validated against Java tombstones, cardinality cross-checked) — a gap in Bearing too; BINARY doc values (fixed + variable length, sparse) → `Binary` columns |

### First numbers (RTX 5090, 64 Mi values, device-resident input, median of 10)

`cargo bench -p lucene-arrow-gpu --features gpu` — fused unpack+epilogue kernel,
one launch per column:

| bpv | GPU GB/s in | GPU Gval/s | CPU GB/s in (fused, 1 thread) | speedup |
|---|---|---|---|---|
| 1 | 8.8 | 70.1 | 0.17 | 51× |
| 8 | 65.2 | 65.3 | 1.37 | 48× |
| 16 | 122.1 | 61.0 | 2.75 | 45× |
| 32 | 229.0 | 57.3 | 5.55 | 41× |
| 64 | 413.0 | 51.6 | 11.08 | 37× |

Low widths are output-store-bound (~0.5 TB/s of i64 stores). The CPU
executor is a fused single pass (`for_each_unpacked` + inlined epilogue),
uniformly ~1.35 Gval/s on one thread; SIMD/multithreading is future §15
kill-criterion work.

**End-to-end** (`--bench e2e_decode`): 4 real-format columns × 32 Mi docs
(0.46 GB `.dvd`, warm host RAM → device-resident Arrow values; sparse
column uses the §11.4 fused-scatter kernel):

| config | wall | payload GB/s | Grows/s |
|---|---|---|---|
| CPU fused, 1 thread | 276 ms | 1.7 | 0.49 |
| GPU, pageable upload | 107 ms | 4.3 | 1.25 |
| GPU, pinned ring (32 MB × 4, copy/DMA overlap) | **31 ms** | **14.7** | **4.3** |

Kernels are 2 ms of that 31 ms — transfer-bound, at the throughput of a
PCIe-5 NVMe (§11.0 regime 1). Next rungs: reading storage straight into
the pinned ring (skip one host copy), then GDS.

**Encode kernels** (`--bench gpu_encode`, §11.7): the inverse bit-pack —
one thread per output word, no atomics, byte-identical to `direct::pack`
(differential-gated). 64 Mi values, device-resident: **52–151 Gval/s
(37–186× single-thread CPU)**, e.g. bpv 8 at 151 GB/s out, bpv 64 at
420 GB/s out. Policy stays CPU-side (register #6's lean, now with
evidence).

The encoder is now a trait (`NumericEncoder`): `CpuEncoder` (reference)
and `GpuPacker` (GPU **stats pass** — min/max/GCD-of-deltas plus *exact*
≤256-distinct table detection via a lock-free device hash — feeding the
pack kernel). Whole `.dvm`/`.dvd` entries from the GPU encoder are
**byte-identical to the CPU encoder** across GCD/table/constant/sparse/
`i64::MIN`-edge shapes (differential-gated). Wired into the DoPut
session's `executor` option (`cpu|gpu|auto`, echoed per manifest
`lucene.encoder`). Register #6 finding: the pack kernel is 37–186× in
isolation, but the session is currently *ingest-bound* (Arrow append +
file IO), so end-to-end write gains stay modest (5.8× → 6.4× JVM) until
the batch→buffer path goes zero-copy.

## Build & test

```bash
cargo test --workspace        # no GPU, no JVM required — 38 tests
cargo run -p lucene-arrow-codec --example make_demo_segment -- /tmp/demo-seg
cargo run -p lucene-arrow-codec --example segment_info -- /tmp/demo-seg
```

## Layout (SPEC §5)

```
crates/
├── core/          # DecodePlan/EncodePlan, SegmentSource/ByteRange, extents,
│                  # lucene.* metadata contract, CodecUtil framing + cursor
├── codec/         # Bearing wrapper: segments_N/.si/.fnm/.cfs → typed inventory
├── docvalues/     # Lucene90 doc values: .dvm → plans; values → .dvd/.dvm bytes
├── cpu/           # reference executors: plans + bytes → Arrow arrays
├── gpu/           # feature "gpu" (stub; SPEC §11)
├── vectors/       # flat vectors (stub; P2)
├── faiss-source/  # OS Faiss sidecar reader (stub; P2)
└── flight/        # Flight request/response vocabulary (server lands P4)
harness/           # Java golden-file generator + CheckIndex gate (JDK 21)
```

### Write path (SPEC §10)

✅ **Complete segment commits from our own writers, zero JVM**: doc values
(byte-identical encoders) + first-party `.fnm`/`.si`/stored-fields framing +
`segments_N` via Bearing's public commit. Output opens through Bearing's
readers (cross-parser check), round-trips through our decoder, and **passes
Java `CheckIndex -level 2`** — the §10.5 acceptance gate, run in
`crates/codec/tests/p5_write_segment.rs`.

✅ **DoPut end-to-end**: WriteJob + Arrow stream over Flight → numeric-family
§10.2 coercions (int widening, Boolean, Float32/64 bit-payload, temporals;
`lucene.source_type` recorded) → segments cut on thresholds at batch
boundaries → one commit → `SegmentManifest` PutResults (files, crc32,
wall_ms) → values read back via DoGet → **CheckIndex clean**.

✅ **SORTED write path** (§10.3): terms-dict writer **byte-identical to
Bearing/Java** (LZ4 blocks, reverse index — gated), ords through the same
executor trait (GPU-capable), and `Utf8`/`Dictionary<Int32,Utf8>` columns
accepted by DoPut (dict-encode on the fly, §10.2) — full string round-trip
DoPut → CheckIndex-clean segment → DoGet `Dictionary` column.

✅ **Every doc-values type writes**: BINARY (fixed + var-length),
multi-valued SORTED_NUMERIC, multi-valued SORTED_SET — all **byte-identical
to Bearing/Java** on a mixed segment (gated). Vector *writing* is
deliberately deferred: a Java-openable vector segment requires the HNSW
graph (P6 stretch); flat-vector write files exist for our own reader.
✅ **DoPut accepts the full §10.2 v1 matrix**: `List<Int64>` →
SORTED_NUMERIC, `List<Utf8>` → SORTED_SET, `Binary`/`LargeBinary` →
BINARY (empty lists = field absent), `coercion: "strict"` rejects
non-canonical shapes at schema time, and `DoAction: "stats"` serves
per-field decode-cost estimates (§8.1) — all e2e-tested with the
CheckIndex gate. ✅ explicit-lossy coercions: `UInt64` gated on
`lucene.allow_lossy = "true"` field metadata, `Decimal128` → scaled long
via `lucene.scale_factor` (ES `scaled_float` rounding) — never automatic,
rejected at schema time otherwise (§10.2 [CONTRACT]).

## P6 — cuVS integration (register #14 resolved)

pixi-managed `libcuvs 26.06` (SPEC §3.7; `pixi.toml` committed, env
gitignored). Build: `CONDA_PREFIX` → pixi env + its `bin` on PATH;
runtime: `LD_LIBRARY_PATH` → env `lib`. Three-way results
(`--bench knn_threeway`, 1M × 128-d f32, 64 queries, k=10):

| engine | build | search (64q) | recall vs FlatKnn |
|---|---|---|---|
| FlatKnn (ours, exact, zero deps) | — | 227 ms | 1.000 (ref) |
| cuVS brute force (exact) | — | 150 ms | 1.000 |
| cuVS CAGRA (default params) | **1.49 s** | 324 ms | 0.256* |

\* uniform-random vectors are the worst case for graph ANN and params are
untuned — the datum that matters for P6c is the **1.5 s GPU graph build
at 1M scale** (JVM HNSW builds take minutes). Exactness note: cuVS brute
force must use `L2Unexpanded` — the default expanded form catastrophically
cancels on near-duplicate vectors (our FlatKnn caught it returning a wrong
top-1; verified against f64).

### P6c — GPU-built graphs → Java-servable vector segments ✅

The full chain works: **NN-Descent on the 5090** (the CAGRA substrate;
162 ms for 4k×32) → conversion layer (symmetrize + deterministic-ring
connectivity — raw kNN digraphs are *disconnected* on clustered data;
Java's greedy search provably never left the entry cluster until we added
it) → **our `.vem`/`.vex` writer** (single-level HNSW, group-varint
neighbor lists, DirectMonotonic offsets) → segment with vector-aware
`.fnm` → **CheckIndex clean** and **Java `KnnFloatVectorQuery` top-10 ==
exact GPU top-10** (`crates/gpu/tests/p6_hnsw_write.rs`). Vector writing
is no longer deferred.

**Indexing head-to-head** (`--bench hnsw_build` vs `harness
BenchKnnIngest`, 200k × 128-d, identical clustered data): GPU pipeline
(NN-Descent 569 ms + navigable conversion 293 ms + files/commit 181 ms) =
**1.06 s total vs JVM HNSW flush 9.49 s — 9.0×**, output CheckIndex-clean.
The conversion step is single-threaded CPU and the gap widens with scale
(Elastic reports ~12× for the same architecture).

**Vectors through DoPut** (`--features cuvs` on the flight crate):
`FixedSizeList<Float32, d>` columns are canonical §10.2 — the write
session buffers them per segment and flush emits flat + graph files
(NN-Descent on GPU; exact CPU kNN for segments ≤ 4096 docs), similarity
via `lucene.vector.similarity` field metadata. Gate
(`flight_vectors.rs`): 6000×32-d through a real DoPut, exact float
round-trip over DoGet, CheckIndex vectors test green.

## P7 — postings relation (§7.8)

`crates/postings`: full-enumeration reader for the Lucene103 block-tree —
no FST traversal (in-order block descent + physical floor chaining), all
suffix compressions, singleton-RLE stats, metadata delta chains, level-0
and level-1 skip consumption, FOR/CONSECUTIVE/BITSET doc blocks, PFor
freqs, group-varint tails → CSR/COO `term | doc | freq`. Every posting
gated against Java-written goldens (`text`, `textbig`).

Throughput (4M docs, 110,973 terms, 12M postings; `BenchText scan` vs
`--bench csr_bench`, same checksum both sides):

| sweep | Mpostings/s |
|---|---|
| JVM TermsEnum+PostingsEnum (warm best) | 360 |
| ours, iterate+sum (same loop shape) | **456–462** |
| ours, full CSR materialization | 254 |
| **GPU doc-block kernel (5090, device-resident)** | **~37,000** |

GPU postings: the level-0 skip data carries every 128-doc block's base
doc id, so blocks decode independently — the CPU plans descriptors
(2.9 ms for 60,919 blocks) and one kernel fans the whole `.doc` file out
(`--bench postings_gpu`; 7.8M packed docs verified bit-identical to the
CPU scan).

Found upstream: `bearing::encoding::pfor::for_delta_decode` panics on
debug add-overflow for bpv ≤ 10 (Java relies on silent i32 wraparound in
the collapsed-lane prefix sums); wrapping-correct port vendored in
`crates/postings/src/pfor.rs`. Flight surface shipped: `relation: "postings"` +
`postings_field` in the DoGet ticket streams
`term: Dictionary(Int32, Binary) | doc: UInt32 | freq: UInt32` batches
per segment (raw relation; join `_live` for deletes). Remaining: GPU
FOR-block kernel, positions.

## P8 — jVector serialization

`vectors::jvector::write_index` emits jVector `OnDiskGraphIndex` **v5**
(big-endian; 288-byte CommonHeader; inline f32 vectors; dense fixed-size
L0 records; footer with authoritative header copy) from the same
GPU-built adjacency as the Lucene HNSW writer. Gate
(`crates/gpu/tests/p8_jvector_write.rs`): the real jVector
4.0.0-beta.6 library opens the file and its graph search returns the
exact GPU top-10, 10/10. One graph build on the 5090 now serves three
targets: Lucene `.vem`/`.vex`, jVector, and (via cuVS interop) Faiss.
Note: repo main is already at format v6 (feature list replaces bitset;
fused features); we pin v5 = what the shipped reader accepts. Post-v1:
PQ/fused features, multi-layer output.

## P9 — markdown → BM25 segments (in progress)

P9a (correctness) ✅: raw markdown → standard-lite analyzer
(`postings::text`, documented contract) → CPU hash-aggregate
(`postings::build`) → **Bearing's block-tree writer** (vendored at
`vendor/bearing` with a one-line visibility patch; Apache-2.0) + our
Lucene90 norms writer (`codec::norms`, SmallFloat field lengths) + our
`.fnm`/`.si` (index-options byte, `PerFieldPostingsFormat` routing) →
complete segments. Three gates in `p9_bm25_segment.rs`: our P7 reader
round-trips every posting; **CheckIndex validates postings + norms**;
**Java's live BM25 `TermQuery` scores match our formula to 1e-4**.
P9b ingest bench (300k Zipf markdown docs, 21M postings, 455k terms;
`BenchMdIngest` vs `--bench bm25_ingest`, identical corpus file):

| pipeline | wall | kdocs/s |
|---|---|---|
| JVM `IndexWriter` (TextField, single flush) | 5.72 s | 52 |
| ours, serial (tokenize 2.1s + sort/RLE 0.45s + write 1.1s) | 3.64 s | 82 |
| ours, parallel build (`build_parallel`, byte-identical to serial) | **2.82 s** | **107** |

After the zero-alloc pass (byte-level ASCII tokenizer fast path,
`entry_ref` interning, bucket-parallel sort+RLE — all identity-gated
against the serial reference): synthetic 2.51 s (2.28×). On **real
markdown** (600 arXiv papers converted by pdf_oxide, 46 MB):

| pipeline | wall | MB/s |
|---|---|---|
| JVM `IndexWriter` | 1.33 s | 35 |
| ours | **0.205 s** | **227** — 6.5× |

(8.0M tokens, 57.5k terms, CheckIndex-clean.) Real long-form documents
favor us much more than short synthetic ones: per-doc overheads amortize
and natural vocab makes interning nearly free.

2.06× → 2.28× on synthetic, output CheckIndex-clean. Executor decision (register #6 style,
Amdahl-driven): the aggregation executor is parallel CPU — with
tokenization parallelized, sort+RLE is ~0.3 s of a 2.8 s pipeline, so a
GPU radix sort buys <10%; the GPU's P9 job is **scoring** (P9c) where
the P7 kernel numbers show ~100× headroom. Remaining hotspot: per-token
String interning (allocator-bound) and the serial segment write.

P9c — GPU BM25 scoring over CSR (`gpu::bm25`, `--bench bm25_query`).
Correctness: GPU scores equal live Java Lucene scoring of the same
segment for every hit (1e-4; `p9_bm25_gpu.rs`). Throughput, 256 3-term
OR-queries on the 300k-doc corpus:

| queries | JVM BooleanQuery (warm) | GPU exhaustive |
|---|---|---|
| selective (random vocab, ~rare terms) | **19.6k qps** | 4.8k qps |
| heavy (top-500-df terms, ~65k rows/query) | 4.3k qps | **5.5k qps**, 379 Mrows/s |

The signature: GPU qps is flat w.r.t. query weight (fixed ~180 µs/query:
alloc + launch + score-array download), JVM degrades 4.5× — exhaustive
GPU scoring wins the analytics shape (heavy terms, full-corpus ranking,
score-everything joins), Lucene's impact-skipping wins selective top-k.
Both recorded as the executor policy. GPU headroom not yet taken:
query batching into one launch, device-side top-k, pinned score
buffers. Note: float atomicAdd makes exact tie order nondeterministic
across runs (scores still match to 1e-4).

## P10 — GPU text ingest (in progress)

`gpu::text_ingest`: tokenize + hash + exact vocabulary (byte-verified
open addressing) + `(term, doc)` pair emission on-device; hybrid Unicode
(pure-ASCII tokens on GPU, non-ASCII spans re-analyzed on CPU, merged) —
**byte-identical to `build_parallel`** on both corpora incl. 391k dirty
spans (`p10_gpu_ingest.rs`).

After iteration 2 (device table compaction, pinned D2H staging —
pageable ran at ~2 GB/s, pinned ~14 — and parallel dirty-span handling):

| corpus | GPU e2e | CPU build | kernel alone |
|---|---|---|---|
| arXiv 46 MB | **95 ms** (493 MB/s; ~14× JVM build) | 151 ms | **2.5 ms (~21 GB/s)** |
| synthetic 127 MB | **505 ms** | 1461 ms | 5.0 ms |

Remaining Amdahl (arXiv): csr 20 + pair-download 18 + remap 13 + dirty 12
+ vocab 8 + kernel 2.5 (+ ~20 alloc/upload). A device counting-sort/RLE
(only the final CSR crosses PCIe) would take e2e toward ~50 ms — real,
but the ~43 ms segment write already bounds the full job, so it's
recorded as headroom rather than assumed. Full-job today: ~138 ms
(build+write) vs JVM 1.33 s ≈ **9.6×** on real markdown.

## P5 verdict — kill-criterion baselines (SPEC §15.4)

Same machine, same data: the 16M-doc × 4-numeric-field index written by
**Java Lucene itself** (`harness BenchIngest`, 209 MB, 60M values), warm
page cache. Reproduce with `harness/src/Bench*.java`,
`--bench scan_dir`, `--bench write_bench`.

| workflow | throughput | vs JVM |
|---|---|---|
| **read**: JVM `NumericDocValues` scan (baseline a) | 316 Mvals/s | 1× |
| read: ours, CPU fused, 1 thread | 460 Mvals/s | 1.45× |
| read: ours, **GPU end-to-end** (pinned ring + kernels, device-resident out) | **4,006 Mvals/s** | **12.7×** |
| read: ours, GPU kernels only (device-resident in) | 48.7 Gvals/s | 154× |
| **write**: JVM `IndexWriter` bulk ingest (baseline c) | 4.09 Mdocs/s | 1× |
| write: ours, DoPut write job (CPU encode) | 23.7 Mdocs/s | 5.8× |
| write: ours, DoPut write job (**GPU stats+pack**, `executor: auto`) | **26.0 Mdocs/s** | **6.4×** |

Baseline (b), scroll-export-to-Parquet: any export workflow costs *at
least* the JVM scan above before serializing, writing, and re-reading —
our direct GPU path is 12.7× that floor, so the competing workflow loses
before its Parquet costs are even counted.

**Kill criterion not triggered** (SPEC §13 go/no-go): direct
segment⇄Arrow decisively beats both competing workflows. Still open for
the full §15 matrix: cold-scan/NVMe and GDS regimes, many-small-segment
profiles, multi-threaded CPU baseline.

## Correctness story today (SPEC §12)

1. **Round-trip**: `decode(encode(x)) == x` across encodings, sparse
   patterns, and DISI block shapes (`crates/cpu/tests/roundtrip.rs`).
2. **Cross-validation**: segments written by [Bearing]'s `IndexWriter`
   (cross-validated upstream against Java Lucene, byte-identical output)
   are decoded and compared against source values
   (`crates/codec/tests/p1_docvalues.rs`).
3. **Byte identity**: our `.dvm`/`.dvd` encoder output is compared
   byte-for-byte against Bearing's for the same input — dense, sparse,
   constant, table, and DENSE-DISI shapes all match exactly.
4. **Golden files + CheckIndex** (pending JDK 21): `harness/run.sh golden`
   writes Java-Lucene segments incl. shapes Bearing never emits
   (multi-block numerics, deletes); `harness/run.sh check` is the
   CheckIndex acceptance gate for our written segments.

## Bearing notes (SPEC §3.3, §16)

[Bearing]: https://github.com/toddfeak/bearing

`bearing 0.1.0-alpha.5` supplies directory/compound handling and
`segments_N`/`.si`/`.fnm` parsing (all public, reused). Its disk doc-values
reader, however, discards all block metadata and panics (`todo!`) on value
access, and its packed-ints/framing primitives are `pub(crate)` — so the
doc-values format lives here in `crates/docvalues`, written against
Bearing's byte-level format docs (`reference/formats/`) and verified by the
byte-identity tests above. Fork risk priced in per SPEC: if Bearing stalls,
what we'd vendor next is `.fnm`/`.si`/`segments_N` parsing (~small) — the
hard format work is already ours.

Known P1 gaps, documented in code: doc-values **skip index** fields
(Lucene 10 opt-in, off by default in ES/OS) need the caller to flag them
(`DvField::has_skip_index`) until we parse `.fnm` bits ourselves; `.liv`
(deletes) reading is not yet implemented (Bearing has none either);
multi-valued SORTED_NUMERIC → `List<Int64>` is planned post-P1.
