# CLAUDE.md

## Project

`github.com/toddfeak/bearing` — A port of Apache Lucene from Java to Rust.

The Java implementation of **Apache Lucene 10.3.2** is the canonical source. All Rust code should be derived from the Lucene 10.3.2 Java source in `reference/lucene-10.3.2/lucene/`. See `reference/CLAUDE.md` for details.

## Directory Layout

| Directory | Purpose |
|---|---|
| `src/` | Library and binary source code |
| `tests/` | Integration tests, E2E scripts, Java validation utilities (see `tests/CLAUDE.md`) |
| `testdata/` | Test corpus and data generation scripts |
| `reference/` | Reference sources and format documentation (see `reference/CLAUDE.md`) |
| `docs/` | Roadmap and planning documents |
| `docs/backlog/` | Known issues and gaps to address in future work |

## Current State

- Target codec: **Lucene103**
- Multi-threaded `IndexWriter` with SegmentWorker pool (no merging, no deletes)
- Fourteen field types: `KeywordField`, `LongField`, `TextField`, `StringField`, `IntField`, `FloatField`, `DoubleField`, `StoredField`, `LatLonPoint`, `FeatureField`, `IntRange`, `LongRange`, `FloatRange`, `DoubleRange`

## Build & Test

```bash
cargo build                    # debug build
cargo test                     # run all tests
cargo fmt                      # format code after editing
cargo clippy --all-targets     # lint (lib, tests, integration tests)
./tests/e2e_all.sh             # all e2e tests (indexing, impacts, compression)
```

Project-specific clippy lints are configured in `Cargo.toml` under `[lints.clippy]`. Currently enforced: `absolute_paths = "warn"` (requires `use` imports instead of inline qualified paths).

**Before any commit**, run `cargo test`, `cargo fmt`, `cargo clippy --all-targets`, and `./tests/e2e_all.sh`. All must pass.

## Conventions

- Write idiomatic Rust types (traits, enums, `Result`/`Option`, ownership) — but do not restructure algorithms. See **Porting Rules** below.
- **Iterative development**: Use `#[expect(dead_code)]` on code added in one commit but not referenced until a later commit. Never use `#[allow(dead_code)]`.
- Runtime dependencies extremely limited. Avoid adding more without explicit request, but also avoid reimplementing common libraries.
- Unimplemented methods should use `todo!("description")` or return `Err(...)`.
- **Byte order**: Little-endian (LE) for data, big-endian (BE) for codec headers/footers. This is the #1 source of bugs — always verify against the Java source.
- **CRC32**: Must match `java.util.zip.CRC32` (polynomial `0xEDB88320`).
- **Codec module naming**: Codec implementations live under version-named modules matching the Java package (e.g., `codecs::lucene90`, `codecs::lucene103`). Version-agnostic utilities stay directly under `codecs`.
- **Logging**: Use `log::debug!` at semantic boundaries (codec headers/footers, flush decisions). Do not log in hot loops.
- **Rustdoc**: Keep `///` and `//!` documentation up to date when changing public API. All public items must be documented.
- Only mention that something is ported from Lucene if it is very specific to Lucene, like a custom compression algorithm. The whole project is a port, we don't need to repeat it in comments.
- **Lazy reader pattern**: Codec readers must follow Lucene's lazy loading strategy. Constructors read only lightweight metadata (headers, per-field stats, file pointers) and open file handles for later use. Actual data (posting lists, stored fields, BKD leaves, etc.) is read on demand during queries. This keeps index opening fast and memory footprint small. Each reader should do exactly what Lucene's corresponding Java reader does at construction time — no more, no less.

## Porting Rules

- Every port starts by reading the COMPLETE Java source method/class first.
- Port the algorithm line-by-line. Preserve the same control flow, variable names (adjusted to snake_case), branch structure, and loop structure.
- Do NOT restructure conditionals, flatten loops, merge branches, reorder operations, or "simplify" logic — even if the result looks non-idiomatic in Rust.
- "Idiomatic Rust" means types (Result/Option/traits/enums) and ownership, NOT algorithm restructuring. The algorithm is not yours to change.
- When the port is done, show a side-by-side summary: Java line ranges → Rust line ranges, so the port can be verified as complete and correctly ordered.
- If you believe a structural change is necessary (not just stylistic), STOP and ask. Explain what Java does and what you want to change and why.

## Testing

- All code changes must include corresponding tests.
- When porting Java Lucene code, port the applicable tests from `reference/lucene-10.3.2/lucene/core/src/test/java/org/apache/lucene/`.
- Use the in-memory `Directory` implementation for unit tests.
- Prefer [Assertables](https://docs.rs/assertables) macros over plain `assert!` for more expressive and descriptive test assertions (e.g., `assert_lt!`, `assert_in_delta!`, `assert_len_eq_x!`, `assert_matches!`, `assert_none!`). Reference source is in `reference/assertables/src/`.
- Integration tests in `tests/*.rs` use only `pub` items from `bearing::*`.
- See `tests/CLAUDE.md` for E2E tests, Java utilities, performance comparison, and profiling.
