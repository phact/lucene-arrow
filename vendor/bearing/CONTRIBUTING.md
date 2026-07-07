# Contributing to Bearing

## Project Status

Bearing is a personal project — a Rust port of Apache Lucene. Pull requests are **not currently accepted**. You're welcome to fork and make it your own.

## Building & Testing

```bash
cargo build            # debug build
cargo test             # run all tests
cargo clippy           # lint (warnings should be zero)
cargo fmt              # format code
```

Full end-to-end test (requires Java 21+; Gradle handles the Lucene dependency):

```bash
./tests/e2e_indexfiles.sh
```

## Coding Conventions

See [`CLAUDE.md`](CLAUDE.md) for the authoritative style guide, including error handling, module layout, byte order rules, and naming conventions.

## Porting Methodology

Bearing is a port of **Apache Lucene 10.3.2**. Lucene is the source of truth for **behavior and storage formats** — the on-disk output must be compatible. However, the Rust code should be idiomatic Rust, not a transliteration of Java. Prefer traits, enums, and Rust's ownership model over mirroring Java's object hierarchy. Codec version naming (e.g., `lucene90`, `lucene103`) follows the Java convention.

To set up the reference sources:

```bash
./reference/download-references.sh
```

When porting a feature, locate the corresponding Java source under `reference/lucene-10.3.2/lucene/` and port both the implementation and its tests.

## Dependency Policy

Bearing uses very limited runtime dependencies and we strive to avoid adding more. At the same time, try not to reimplement common libraries.

## Test Data

`testdata/docs/` has 4 small files for quick tests. `testdata/impact-docs/` has 150 documents for impact/feature testing. Generate a larger corpus for benchmarking:

```bash
python3 testdata/gen_docs.py -n 2000    # generates to /tmp/perf-docs/
```

## E2E Tests

Roundtrip tests: Bearing writes indexes, Java Lucene reads and validates them.

```bash
./tests/e2e_all.sh              # run all e2e tests
./tests/e2e_indexfiles.sh       # basic indexing roundtrip
./tests/e2e_doc_values.sh       # doc values byte-level comparison
./tests/e2e_verify_impacts.sh   # feature field / impact encoding
./tests/e2e_verify_tim_compression.sh  # terms dictionary compression
```

Requires Java 21+. Gradle handles the Lucene dependency automatically.

## Performance Comparison

Compare indexing speed and correctness between Java Lucene and Rust:

```bash
./tests/compare_index_perf.sh -release --threads 8
./tests/compare_index_perf.sh -docs /tmp/perf-docs -release
```

Compare query performance across Java and Rust on the same index:

```bash
./tests/compare_query_perf.sh -docs /tmp/gutenberg-small-2000
```

## If PRs Open in the Future

Contributions would need to:

- Pass `cargo test` (all tests)
- Pass `cargo clippy -- -D warnings` (no lint warnings)
- Be formatted with `cargo fmt`
- Include tests for new functionality
