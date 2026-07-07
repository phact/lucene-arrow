# Status & benchmark log

Phase-by-phase record and full benchmark methodology. The [`README`](../README.md)
carries the clean build/use/test/perf summary; this is the detailed log.
Concept and contracts live in [`SPEC.md`](../SPEC.md).

All benchmarks: RTX 5090, CUDA 12.9, warm page cache, versus Java Lucene
10.3.2 on the same machine and data.

---

## Milestones

| milestone | state |
|---|---|
| P0 — open a segment | ✅ `SegmentDirectory` parses `segments_N`/`.si`/`.fnm`/`.cfs` (Lucene103), typed field inventory |
| P1 — numerics (CPU + GPU) | ✅ NUMERIC decode/encode, all Lucene90 encodings (constant/delta/GCD/table/multi-block/sparse DISI); encode byte-identical to Bearing; NVRTC funnel-shift kernel bit-identical to CPU on synthetic sweeps and Java goldens |
| P2 — flat vectors | ✅ Lucene99FlatVectorsFormat read/write (float32 + int8), doc-aligned `FixedSizeList`, GPU DMA payload; exact GPU KNN over a Java-written segment with no JVM |
| P3 — dictionaries (read) | ✅ SORTED / SORTED_SET / SORTED_NUMERIC, LZ4 terms-dict, ordinal decode, global `OrdinalMap` k-way merge fused into the Table epilogue |
| P4 — Flight (read) | ✅ `GetFlightInfo` + `DoGet`, both row modes against real tombstones, dict handling, projection, GPU behind the engine; `DoAction: hydrate` |
| P5 — kill-criterion verdict | ✅ recorded (below); direct segment⇄Arrow beats export-to-Parquet and JVM scan decisively |
| P6 — cuVS + GPU HNSW write | ✅ cuVS integrated; GPU-built graphs → Java-servable vector segments; vectors through DoPut |
| P7 — postings relation | ✅ block-tree reader, `term\|doc\|freq` relation over Flight, GPU doc-block decode |
| P8 — jVector serialization | ✅ `OnDiskGraphIndex` v5, verified by the real jVector library |
| P9 — markdown → BM25 | ✅ CheckIndex-clean BM25 segments, live Java score parity, GPU scoring |
| P10 — GPU text ingest | ✅ tokenize/hash/aggregate on device, byte-identical to CPU |

---

## P5 verdict — kill-criterion baselines (SPEC §15.4)

Same machine, same data: the 16M-doc × 4-numeric-field index written by
Java Lucene itself (`harness BenchIngest`, 209 MB, 60M values), warm cache.

| workflow | throughput | vs JVM |
|---|---|---|
| **read**: JVM `NumericDocValues` scan | 316 Mvals/s | 1× |
| read: ours, CPU fused, 1 thread | 460 Mvals/s | 1.45× |
| read: ours, **GPU end-to-end** (pinned ring + kernels) | **4,006 Mvals/s** | **12.7×** |
| read: ours, GPU kernels only (device-resident in) | 48.7 Gvals/s | 154× |
| **write**: JVM `IndexWriter` bulk ingest | 4.09 Mdocs/s | 1× |
| write: ours, DoPut (CPU encode) | 23.7 Mdocs/s | 5.8× |
| write: ours, DoPut (**GPU stats+pack**) | **26.0 Mdocs/s** | **6.4×** |

Export-to-Parquet (baseline b) costs *at least* the JVM scan above before
serializing/writing/re-reading, so it loses to the direct GPU path before
its own Parquet costs are counted. **Kill criterion not triggered.** Still
open for the full §15 matrix: cold-scan/NVMe and GDS regimes,
many-small-segment profiles, multi-threaded CPU baseline.

### Decode/encode kernel micro-benchmarks

Fused unpack+epilogue, one launch per column, 64 Mi values, device-resident
(`--bench gpu_decode`):

| bpv | GPU Gval/s | CPU Gval/s (1 thread) | speedup |
|---|---|---|---|
| 1 | 70.1 | 0.17 | 51× |
| 8 | 65.3 | 1.37 | 48× |
| 16 | 61.0 | 2.75 | 45× |
| 32 | 57.3 | 5.55 | 41× |
| 64 | 51.6 | 11.08 | 37× |

Low widths are output-store-bound (~0.5 TB/s of i64 stores). End-to-end
(`--bench e2e_decode`, 4 real columns × 32 Mi docs, 0.46 GB `.dvd`):
CPU fused 1-thread 276 ms; GPU pageable upload 107 ms; **GPU pinned ring
(32 MB × 4, copy/DMA overlap) 31 ms** (14.7 GB/s, 4.3 Grows/s) — of which
kernels are 2 ms, i.e. transfer-bound. Encode kernels (`--bench gpu_encode`)
hit 52–151 Gval/s (37–186× single-thread CPU), byte-identical to
`direct::pack`. Register #6 finding: the pack kernel is 37–186× in
isolation but DoPut is ingest-bound (Arrow append + file IO), so
end-to-end write stays at 6.4× until the batch→buffer path goes zero-copy.

---

## P6 — cuVS + GPU-built vector segments

pixi-managed `libcuvs 26.06` (SPEC §3.7). Three-way ANN
(`--bench knn_threeway`, 1M × 128-d f32, 64 queries, k=10):

| engine | build | search (64q) | recall vs FlatKnn |
|---|---|---|---|
| FlatKnn (ours, exact, zero deps) | — | 227 ms | 1.000 (ref) |
| cuVS brute force (exact) | — | 150 ms | 1.000 |
| cuVS CAGRA (default params) | **1.49 s** | 324 ms | 0.256* |

\* uniform-random vectors are the worst case for graph ANN and params are
untuned — the datum that matters is the 1.5 s GPU graph build at 1M scale
(JVM HNSW builds take minutes). cuVS brute force must use `L2Unexpanded`;
the expanded form catastrophically cancels on near-duplicate vectors (our
FlatKnn caught it returning a wrong top-1, verified against f64).

**GPU-built graphs → Java-servable segments.** NN-Descent on the 5090
(162 ms for 4k×32) → conversion layer (symmetrize + deterministic-ring
connectivity — raw kNN digraphs are disconnected on clustered data, and
Java's greedy search provably never left the entry cluster until this was
added) → our `.vem`/`.vex` writer → segment with vector-aware `.fnm` →
CheckIndex clean and **Java `KnnFloatVectorQuery` top-10 == exact GPU
top-10** (`crates/gpu/tests/p6_hnsw_write.rs`).

**Indexing head-to-head** (`--bench hnsw_build` vs `BenchKnnIngest`,
200k × 128, identical data): GPU pipeline (NN-Descent 569 ms + conversion
293 ms + files/commit 181 ms) = **1.06 s vs JVM HNSW flush 9.49 s = 9.0×**,
CheckIndex-clean. Conversion is single-threaded CPU; the gap widens with
scale (Elastic reports ~12× for the same architecture).

**Vectors through DoPut** (`cuvs` feature): `FixedSizeList<Float32, d>`
columns flush flat + graph files, similarity via `lucene.vector.similarity`
metadata (`flight_vectors.rs`: DoPut → exact float round-trip over DoGet →
CheckIndex vectors test green).

**Search throughput + the graph-quality fix** (`--bench vector_search`).
One GPU-built graph, searched three ways over 100k × 128-d, k=10, ef=100
(FlatKnn = exact GPU brute force; jVector / Lucene read the files we
wrote):

| engine | QPS | recall@10 |
|---|---|---|
| FlatKnn (ours, exact GPU) | 3,062 | 1.000 (reference) |
| jVector OnDiskGraphIndex | 3,576 | 0.969 |
| Lucene HNSW (KnnFloatVectorQuery) | 7,553 | 0.980 |

This bench exposed — and drove the fix for — a real graph-quality gap.
The write path originally built a **single-level** graph (cuVS NN-Descent
kNN edges + a deterministic connectivity ring). That passed the P6c/P8
*format* gates (4k, low-dim, exact-member queries → 100% recall) but at
100k scale gave only **~5% recall**: a single-level kNN graph is poorly
navigable for single-entry greedy search — greedy from the entry node
cannot escape its local component to reach a far query. Feeding cuVS
**CAGRA**'s search-optimized graph instead (extracted via
`cuvsCagraIndexGetGraph`) made it *worse* (~1%): CAGRA's graph, though
80% true-kNN accurate, is tuned for CAGRA's own multi-start search, not
Lucene/jVector greedy. The fix (`hnsw::small_world_from_cagra`) keeps
CAGRA's good local edges and adds a few **random long-range edges** per
node — small-world structure (O(log n) diameter) — which is exactly what
single-entry greedy needs: recall jumps to ~98%. Both the DoPut write
path and the bench use it.

**But our graph is still a hack — measured 3–5× worse than native
construction.** The bench also builds each engine's *own* graph on the
same data (jVector `GraphIndexBuilder`, Lucene `IndexWriter`) — at 100k,
ef=100:

| engine, graph source | QPS | recall@10 |
|---|---|---|
| FlatKnn (exact GPU) | 2,588 | 1.000 |
| jVector — our graph | 3,316 | 0.965 |
| jVector — native | **15,298** | **1.000** |
| Lucene HNSW — our graph | 7,653 | 0.975 |
| Lucene HNSW — native | **19,036** | **1.000** |

Both native builders beat our graph decisively on *both* speed and
recall — so it isn't a jVector-specific shape mismatch (Lucene is only
slightly more forgiving: 2.5× gap vs jVector's 4.6×). The reason: native
HNSW/Vamana do **diversity pruning** (non-redundant neighbor selection)
and build a **multi-layer hierarchy**; ours keeps raw CAGRA local edges +
*uniform*-random shortcuts, which waste degree and navigate worse. This
also corrects an earlier reading: with a *proper* graph, graph-ANN beats
exact FlatKnn even at 100k (Lucene native 19k vs FlatKnn 2.6k = 7×) — our
graph only looked "competitive with FlatKnn" because it was weak. Recall
at scale is governed by the search beam (ef) and degree, not the shortcut
count (scaling shortcuts up *lowers* recall — they steal local edges).
The real fix is proper multi-layer construction: either port
HNSW/Vamana-style build, or use cuVS `cuvsHnswFromCagra` (CAGRA →
hierarchy) and extend our `.vem`/`.vex` writer to multi-level. Post-v1.

**Update — done for Lucene.** Implemented exactly that: CAGRA →
`cuvsHnswFromCagra` (hierarchy=CPU → standard hnswlib) → `parse_hnswlib`
→ `HnswFilesBuilder::add_field_multi` (real multi-layer Lucene99
`.vem`/`.vex`). Gate `p_multilevel`: 4-level pyramid (5000/324/13/1),
CheckIndex clean, Java KNN exact. Result at 100k, ef=100:

| engine, graph source | QPS | recall@10 |
|---|---|---|
| Lucene HNSW — our multi-level | **16,329** | 0.978 |
| Lucene HNSW — native | 19,005 | 1.000 |

Then the jVector `OnDiskGraphIndex` writer got the same multi-layer
treatment (`write_index_multi`: layer table + dense L0 + sparse upper
levels), fed the same parsed hierarchy. Final five-way (100k, ef=100):

| engine, graph source | QPS | recall@10 |
|---|---|---|
| FlatKnn (exact GPU) | 2,692 | 1.000 |
| jVector — ours (multi-level) | **16,650** | 0.986 |
| jVector — native | 15,669 | 1.000 |
| Lucene HNSW — ours (multi-level) | 15,649 | 0.986 |
| Lucene HNSW — native | 18,772 | 1.000 |

Both our files now land **at native** — jVector-ours actually edges out
jVector's own builder on QPS; Lucene-ours is within ~1.2×, all at ~0.986
recall. From "3–5× worse hack" to native-quality graph-ANN files, by
letting cuVS build the hierarchy (`cuvsHnswFromCagra`) and faithfully
re-serializing it into both formats. Gate `p_multilevel`: CheckIndex +
Java `KnnFloatVectorQuery` (Lucene) and the real jVector library
(jVector) both open and search our multi-level files with exact top-1.
Remaining: wire the multi-level hierarchy into the DoPut write path
(currently single-level small-world) so ingested vectors get it too.

---

## P7 — postings relation (§7.8)

`crates/postings`: full-enumeration reader for the Lucene103 block-tree —
no FST traversal (in-order block descent + physical floor chaining), all
suffix compressions, singleton-RLE stats, metadata delta chains, level-0
and level-1 skip consumption, FOR/CONSECUTIVE/BITSET doc blocks, PFor
freqs, group-varint tails → CSR/COO `term | doc | freq`. Every posting
gated against Java-written goldens.

Throughput (4M docs, 110,973 terms, 12M postings; same checksum both sides):

| sweep | Mpostings/s |
|---|---|
| JVM TermsEnum+PostingsEnum (warm) | 360 |
| ours, iterate+sum | **456–462** |
| ours, full CSR materialization | 254 |
| GPU doc-block kernel (device-resident) | **~37,000** |

GPU postings: level-0 skip data carries every 128-doc block's base doc id,
so blocks decode independently — the CPU plans descriptors (2.9 ms for
60,919 blocks) and one kernel fans the whole `.doc` file out
(`--bench postings_gpu`; 7.8M packed docs bit-identical to the CPU scan).
Flight surface: `relation: "postings"` streams
`term: Dictionary(Int32, Binary) | doc: UInt32 | freq: UInt32` per segment.

Found upstream: `bearing::encoding::pfor::for_delta_decode` panics on
debug add-overflow for bpv ≤ 10 (Java relies on silent i32 wraparound in
the collapsed-lane prefix sums); wrapping-correct port vendored in
`crates/postings/src/pfor.rs`. Remaining: GPU FOR-block freq kernel,
positions.

---

## P8 — jVector serialization

`vectors::jvector::write_index` emits jVector `OnDiskGraphIndex` **v5**
(big-endian; 288-byte CommonHeader; inline f32 vectors; dense fixed-size L0
records; footer with authoritative header copy) from the same GPU-built
adjacency as the Lucene HNSW writer. Gate
(`crates/gpu/tests/p8_jvector_write.rs`): the real jVector 4.0.0-beta.6
library opens the file and its graph search returns the exact GPU top-10,
10/10. One graph build now serves three targets: Lucene `.vem`/`.vex`,
jVector, and (via cuVS interop) Faiss.

jVector's on-disk graphs are structurally immutable (only inline feature
bytes are rewritable in place); churn is handled LSM-style via the
`@Experimental` `OnDiskGraphIndexCompactor` (reuses source edges, one PQ
retrain per compaction) — the same immutable-segment + merge model our
architecture already follows. Repo main is at format v6 (feature list
replaces bitset; fused PQ); we pin v5 = what the shipped reader accepts.
Post-v1: PQ/fused features, multi-layer output.

---

## P9 — markdown → BM25 segments

Raw markdown → standard-lite analyzer (`postings::text`, documented
contract) → hash-aggregate (`postings::build`) → Bearing's block-tree
writer (vendored at `vendor/bearing` with a one-line visibility patch) +
our Lucene90 norms writer (`codec::norms`, SmallFloat field lengths) + our
`.fnm`/`.si` → complete segments. Three gates (`p9_bm25_segment.rs`): P7
reader round-trips every posting; CheckIndex validates postings + norms;
**live Java BM25 `TermQuery` scores match our formula to 1e-4**.

**Ingest** — real markdown (600 arXiv papers via pdf_oxide, 46 MB, 8.0M
tokens, 57.5k terms), `--bench bm25_ingest` vs `BenchMdIngest`:

| pipeline | wall | MB/s |
|---|---|---|
| JVM `IndexWriter` | 1.33 s | 35 |
| ours (parallel, zero-alloc, byte-identical to serial) | **0.205 s** | **227** — 6.5× |

Synthetic short-doc corpora show a smaller 2.28× — real long-form text
amortizes per-doc overhead and makes interning nearly free. Executor
decision (Amdahl-driven): after parallelizing tokenization, sort+RLE is a
small fraction of the wall, so a GPU radix sort buys little on the *build*
side; the GPU's job here is scoring and (P10) tokenization.

**Scoring** — GPU BM25 over CSR (`gpu::bm25`), scores equal live Java
scoring for every hit (1e-4; `p9_bm25_gpu.rs`). 256 3-term OR-queries,
300k-doc corpus:

| queries | JVM BooleanQuery | GPU exhaustive |
|---|---|---|
| selective (rare terms) | **19.6k qps** | 4.8k qps |
| heavy (top-500-df, ~65k rows/query) | 4.3k qps | **5.5k qps**, 379 Mrows/s |

GPU qps is flat w.r.t. query weight (fixed ~180 µs/query overhead); JVM
degrades 4.5×. Exhaustive GPU scoring wins the analytics shape (heavy
terms, full-corpus ranking, score-everything joins); Lucene's
impact-skipping wins selective top-k — recorded as executor policy.
Headroom not yet taken: query batching into one launch (fills GPU
occupancy — a single query barely occupies the 5090), device-side top-k,
pinned score buffers. Float atomicAdd makes exact tie order
nondeterministic (scores still match to 1e-4).

---

## P10 — GPU text ingest

`gpu::text_ingest`: tokenize + hash + exact vocabulary (byte-verified open
addressing) + `(term, doc)` pair emission on-device; hybrid Unicode
(pure-ASCII tokens on GPU, non-ASCII spans re-analyzed on CPU, merged) —
**byte-identical to `build_parallel`** on both corpora incl. 391k dirty
spans (`p10_gpu_ingest.rs`).

After device table compaction, pinned D2H staging (pageable ~2 GB/s →
pinned ~14), and parallel dirty-span handling:

| corpus | GPU e2e | CPU build | kernel alone |
|---|---|---|---|
| arXiv 46 MB | **95 ms** (493 MB/s; ~14× JVM build) | 151 ms | **2.5 ms (~21 GB/s)** |
| synthetic 127 MB | **505 ms** | 1461 ms | 5.0 ms |

Full job today: ~138 ms (build + write) vs JVM 1.33 s ≈ **9.6×** on real
markdown. Remaining Amdahl (arXiv e2e): csr 20 + pair-download 18 + remap
13 + dirty 12 + vocab 8 + kernel 2.5 ms. A device counting-sort/RLE (only
the final CSR crosses PCIe) would take e2e toward ~50 ms, but the ~43 ms
segment write (Bearing's serial encoder) already bounds the full job — so
it's recorded as headroom rather than assumed.

---

## Correctness (SPEC §12)

1. **Round-trip**: `decode(encode(x)) == x` across encodings, sparse
   patterns, DISI block shapes (`crates/cpu/tests/roundtrip.rs`).
2. **Cross-validation**: Bearing-written segments decoded and compared
   against source values (`crates/codec/tests/p1_docvalues.rs`).
3. **Byte identity**: our `.dvm`/`.dvd`/terms-dict/norms/vector output is
   byte-for-byte identical to Bearing's (transitively Java-identical).
4. **Differential**: every GPU kernel is bit-identical to the CPU
   reference on synthetic sweeps and Java goldens.
5. **CheckIndex**: written segments pass Java `CheckIndex -level 2`; graph
   and BM25 outputs are additionally verified by live Java/jVector queries.

## Bearing notes (SPEC §3.3, §16)

[Bearing](https://github.com/toddfeak/bearing) `0.1.0-alpha.5` supplies
directory/compound handling, `segments_N`/`.si`/`.fnm` parsing (public,
reused), the public block-tree terms writer (fed by our aggregated
postings, P9), and LZ4/group-varint/pfor encoding primitives. Its disk
doc-values reader discards block metadata and panics on value access, and
several packed-ints primitives are `pub(crate)` — so the doc-values format
is reimplemented in `crates/docvalues`, verified by the byte-identity tests
above. Vendored at `vendor/bearing` with one `pub(crate)` → `pub` patch on
`BlockTreeTermsWriter`. Fork risk priced in: if Bearing stalls, what we'd
vendor next is small (`.fnm`/`.si`/`segments_N` parsing) — the hard format
work is already ours.
