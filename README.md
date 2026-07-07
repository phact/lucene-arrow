# lucene-arrow

GPU-accelerated, JVM-free serde between Apache Lucene segments and Apache Arrow.

Lucene's on-disk segment format — the storage layer under Elasticsearch and
OpenSearch — is treated as a serialization format, like Parquet, with fast
conversion to and from Arrow and no JVM anywhere in the path. Doc values become
Arrow columns, flat vectors become `FixedSizeList` arrays, postings become a
`term | doc | freq` relation, and an Arrow Flight front door streams shards to
cuDF/cuPy. An RTX 5090 does the heavy lifting — decode, encode, ANN graph build,
BM25 scoring, text tokenization — while the CPU path is the correctness
reference every GPU kernel is checked bit-identical against.

Concept and contracts: [`SPEC.md`](SPEC.md). 

Full phase-by-phase status and benchmark methodology: [`docs/STATUS.md`](docs/STATUS.md).

## Capabilities

- **Read** — Lucene103 segment → Arrow: every Lucene90 doc-values type (numeric
  incl. multi-block, sorted / sorted-set / sorted-numeric, binary, sparse DISI),
  flat vectors, live docs, and the global ordinal map.
- **Write** — Arrow → CheckIndex-clean segment: the full §10.2 coercion matrix
  (numeric family, strings, lists, binary, strict + explicit-lossy), on CPU or
  GPU, byte-identical to Java Lucene.
- **Vectors** — flat vectors both directions; GPU-built HNSW graphs
  (NN-Descent / CAGRA) serialized to Lucene `.vem`/`.vex` and jVector
  `OnDiskGraphIndex`, each verified by the real Java / jVector library.
- **Postings & BM25** — full block-tree terms/postings reader → `term|doc|freq`
  relation; markdown → BM25-scored Lucene segments with live Java score parity;
  GPU tokenization and GPU BM25 scoring.
- **Flight** — Arrow Flight front door: `DoGet` (row modes, projection, dict
  handling, CPU/GPU executor), `DoPut` (Arrow → segments), `DoAction` (hydrate,
  stats), and the postings relation.

Correctness anchor: our encoders are byte-identical to
[Bearing](https://github.com/toddfeak/bearing), which is byte-identical to Java
Lucene; GPU output is differentially bit-identical to the CPU reference; written
segments pass Java `CheckIndex -level 2`.

## Build

A recent stable Rust toolchain. The core library builds with no GPU and no JVM:

```bash
cargo build --workspace
```

**GPU** (`gpu` feature) needs an NVIDIA GPU and CUDA 12.x runtime — kernels are
NVRTC-compiled at run time, so there is no build-time `nvcc` dependency.

```bash
cargo build -p lucene-arrow-gpu --features gpu
```

**cuVS** (`cuvs` feature — ANN graph build, GPU vector ingest) links `libcuvs`,
pinned via [pixi](https://pixi.sh) from RAPIDS. One-time setup:

```bash
pixi install                                  # fetch libcuvs 26.06 into .pixi/
export CONDA_PREFIX="$PWD/.pixi/envs/default"
export PATH="$CONDA_PREFIX/bin:$PATH"          # cmake discovery at build time
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib"     # linking at run time
cargo build -p lucene-arrow-gpu --features cuvs
```

**Harness** (Java golden segments, JVM baselines, CheckIndex) needs JDK 21 and
`lucene-core-10.3.2.jar` in `harness/lib/`. Only required to regenerate goldens
or run the JVM comparisons — the Rust tests ship with committed golden segments.

## Use

Nine library crates (SPEC §5):

| crate | role |
|---|---|
| `core` | plans, sources, cursor/framing, the `lucene.*` metadata contract |
| `docvalues` | Lucene90 doc-values format, read + write |
| `codec` | segment directory, `.fnm`/`.si`/`.liv`/norms writers, segment assembly |
| `cpu` | CPU decode executors (correctness reference) |
| `vectors` | flat vectors, Lucene HNSW + jVector graph writers |
| `postings` | block-tree postings reader, BM25 index build |
| `gpu` | CUDA executors (`gpu` / `cuvs` features) |
| `flight` | Arrow Flight service |
| `faiss-source` | Faiss-shard read path (stub) |

**Inspect a segment:**

```bash
cargo run -p lucene-arrow-codec --example make_demo_segment -- /tmp/seg
cargo run -p lucene-arrow-codec --example segment_info      -- /tmp/seg
```

**Serve over Flight** — `LuceneFlightService` is a tonic service you embed:

```rust
let svc = lucene_arrow_flight::LuceneFlightService::default().into_server();
tonic::transport::Server::builder().add_service(svc).serve(addr).await?;
```

Then `DoGet` a `ReadRequest` (path, columns, row_mode, executor) for
segment → Arrow; `DoPut` a `WriteJob` + Arrow stream for Arrow → segments; or
`DoGet` with `relation: "postings"` for the `term|doc|freq` relation. See
`crates/flight/tests/` for end-to-end examples of each surface.

**Exact GPU KNN over a Java-written segment, no JVM:**

```bash
cargo run -p lucene-arrow-gpu --features gpu --example gpu_knn -- harness/golden/vectors
```

## Test

```bash
cargo test --workspace                          # full suite — no GPU, no JVM needed
cargo test -p lucene-arrow-gpu --features gpu   # GPU differential gates
cargo clippy --workspace --all-targets          # lint

# cuVS gates (export the pixi env as in Build first):
cargo test -p lucene-arrow-gpu -p lucene-arrow-flight \
  --features lucene-arrow-gpu/cuvs,lucene-arrow-flight/cuvs
```

The Rust suite runs against committed golden segments (real Java Lucene 10.3.2).
To regenerate them or run the CheckIndex acceptance gate — Java's own
corruption checker; a pass means real Lucene / Elasticsearch / OpenSearch can
open the segment (needs JDK 21):

```bash
harness/run.sh golden harness/golden    # write fresh Java goldens
harness/run.sh check  <segment-dir>     # CheckIndex -level 2 on our output
```

## Performance

RTX 5090, warm cache, versus Java Lucene 10.3.2 on the same machine and data.
Every row is reproducible with the noted command (`--bench` = `cargo bench -p
lucene-arrow-gpu`; `Bench*` / `run.sh` = the JVM harness). Full methodology in
[`docs/STATUS.md`](docs/STATUS.md).

Reproduce the whole table in one shot — `scripts/bench-all.sh` detects
GPU / cuVS / JVM, generates any missing inputs, runs every bench and JVM
baseline, and prints a consolidated report (CPU rows run anywhere; the rest
fill in when available; `QUICK=1` for a fast smoke run).

| workflow | ours | JVM | speedup | reproduce |
|---|---|---|---|---|
| doc-values **read**, e2e (device-resident Arrow out) | 4,006 Mvals/s | 316 Mvals/s | **12.7×** | `--bench e2e_decode` / `BenchScan` |
| doc-values read, GPU kernels only | 48.7 Gvals/s | — | 154× | `--bench gpu_decode` |
| doc-values **write** (DoPut, GPU stats+pack) | 26.0 Mdocs/s | 4.09 Mdocs/s | **6.4×** | `--bench write_bench` / `BenchIngest` |
| **HNSW indexing** (200k×128, graph + segment) | 1.06 s | 9.49 s | **9.0×** | `--bench hnsw_build` / `BenchKnnIngest` |
| **postings scan** (12M postings, same checksum) | 462 Mpostings/s | 360 Mpostings/s | 1.27× | `--bench csr_bench` / `BenchText scan` |
| postings decode, GPU doc-block kernel | 37 Gdocs/s | — | ~100× | `--bench postings_gpu` |
| **BM25 ingest** (arXiv markdown, 46 MB), CPU | 227 MB/s | 35 MB/s | **6.5×** | `--bench bm25_ingest` / `BenchMdIngest` |
| BM25 ingest, GPU tokenize (full job) | ~138 ms | 1.33 s | **~9.6×** | `--bench gpu_ingest` |
| **BM25 scoring**, heavy queries | 5.5k qps (379 Mrows/s) | 4.3k qps | 1.27× | `--bench bm25_query` / `BenchBM25Query` |
| BM25 scoring, selective queries | 4.8k qps | 19.6k qps | 0.24× | `--bench bm25_query` |
