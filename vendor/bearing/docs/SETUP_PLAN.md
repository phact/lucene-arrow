# Bearing — GitHub & Project Setup Plan

## Context

Bearing is a Rust port of Apache Lucene, evolving from the existing `toddfeak/rust-lucene-indexer` project (104 commits, ~18K LOC, 367 tests, Lucene 10.3.2 write path). The scope is expanding from indexing-only to full Lucene support. This plan covers everything needed to establish Bearing as a properly structured open-source project on GitHub, initially private under the `toddfeak` account.

---

## 1. GitHub Repository Setup

**Dependencies**: None (do this first)

- [x] Create new `toddfeak/bearing` repository on GitHub
- [x] Initialize with a fresh commit of the current project state (no history from `rust-lucene-indexer`)
- [x] Set description: "A Rust port of Apache Lucene"
- [x] Add topics: `rust`, `lucene`, `search`, `indexing`, `information-retrieval`, `search-engine`, `full-text-search`, `apache-lucene`
- [x] Keep repository **private** until Pre-Public Checklist (Section 12) is complete
- [x] Disable wiki and discussions (enable discussions later if community grows)
- [x] Enable issues
- [x] Confirm default branch is `main`

**Note**: This is a clean break from `rust-lucene-indexer` — no fork relationship, no commit history carried over. The NOTICE file and README provide sufficient provenance.

---

## 2. License & Legal

**Dependencies**: Section 1

- [x] Create `LICENSE` file with full Apache License 2.0 text
- [x] Create `NOTICE` file (required by Apache 2.0 Section 4d for derivative works):
  ```
  Bearing
  Copyright 2025-2026 Todd Feak

  This product includes software derived from Apache Lucene
  (https://lucene.apache.org/), licensed under the Apache License 2.0.

  Apache Lucene
  Copyright 2001-2025 The Apache Software Foundation
  ```
- [x] Use SPDX short identifier (`// SPDX-License-Identifier: Apache-2.0`) as file header convention — add incrementally, not in bulk
- [x] Verify no existing files contain conflicting license statements
- [x] Confirm the Java test utilities (`VerifyIndex.java`, `IndexAllFields.java`) are original work, not copied from Lucene source

**Files to create**: `LICENSE`, `NOTICE`

---

## 3. Project Identity

**Dependencies**: Section 2

- [x] Update `Cargo.toml`:
  - `name = "bearing"`
  - `description = "A Rust port of Apache Lucene"`
  - `license = "Apache-2.0"`
  - `repository = "https://github.com/toddfeak/bearing"`
  - `homepage = "https://github.com/toddfeak/bearing"`
  - `readme = "README.md"`
  - `keywords = ["lucene", "search", "indexing", "full-text-search"]`
  - `categories = ["text-processing"]`
  - `rust-version = "1.85"` (edition 2024 requires 1.85+)
  - `exclude = ["reference/", "testdata/", "tests/*.sh", "tests/*.java", "rust-index/", ".claude/", "docs/"]`
- [x] Create `README.md`:
  - Project name and one-line description
  - Current status (what works: write path, 8 field types, multi-threaded indexing)
  - Lucene version target (10.3.2, Lucene103 codec)
  - Performance highlights (2x-4.4x faster than Java Lucene)
  - Build instructions (`cargo build`, `cargo test`)
  - Roadmap summary (tiers, expanded for full Lucene support)
  - Contributing link
  - License note with Apache Lucene attribution
- [x] Create `rust-toolchain.toml`:
  ```toml
  [toolchain]
  channel = "stable"
  components = ["rustfmt", "clippy"]
  ```
- [x] Verify `Cargo.lock` regenerates after rename

**Files to create**: `README.md`, `rust-toolchain.toml`
**Files to modify**: `Cargo.toml`

---

## 4. Community Files

**Dependencies**: Section 1

- [x] Create `CONTRIBUTING.md`:
  - Build and test instructions
  - Link to `CLAUDE.md` for coding conventions
  - Porting methodology (Java Lucene is the canonical source)
  - How to set up reference source (`./reference/download-lucene.sh`)
  - How to run the full test suite including e2e
  - PR expectations (tests required, clippy clean, formatted)
  - Dependency policy (minimal deps)
- [x] Create `SECURITY.md`:
  - Supported versions
  - Vulnerability reporting via email (not public issues)
  - Expected response timeline
- [x] Create `.github/ISSUE_TEMPLATE/bug_report.md`:
  - Rust version, OS, Bearing version
  - Steps to reproduce, expected vs actual behavior
  - Lucene compatibility context if relevant
- [x] Create `.github/ISSUE_TEMPLATE/feature_request.md`:
  - Which Lucene feature/API this relates to
  - Use case
  - Java Lucene class reference if applicable
- [x] Create `.github/ISSUE_TEMPLATE/config.yml`
- [x] Create `.github/PULL_REQUEST_TEMPLATE.md`:
  - What this PR does
  - Java Lucene source reference (if porting)
  - Test coverage
  - Checklist: `cargo test`, `cargo clippy`, `cargo fmt`

**Files to create**: `CONTRIBUTING.md`, `SECURITY.md`, `.github/ISSUE_TEMPLATE/bug_report.md`, `.github/ISSUE_TEMPLATE/feature_request.md`, `.github/ISSUE_TEMPLATE/config.yml`, `.github/PULL_REQUEST_TEMPLATE.md`

---

## 5. CI/CD

**Dependencies**: Section 3 (Cargo.toml must have correct name and MSRV)

- [x] Create `.github/workflows/ci.yml`:
  - Trigger on push to `main` and on pull requests
  - Job: **test** — `cargo test` on ubuntu-latest
  - Job: **clippy** — `cargo clippy -- -D warnings`
  - Job: **fmt** — `cargo fmt -- --check`
  - Job: **msrv** — test against Rust 1.85 (MSRV) in addition to stable
  - Cache via `Swatinem/rust-cache`
- [x] Create `.github/workflows/e2e.yml` (separate, heavier workflow):
  - Trigger: `workflow_dispatch` (manual) and weekly schedule
  - Install Java 21+, download Lucene, run `./tests/e2e_indexfiles.sh`
  - Separate because it requires Java + Lucene JAR
- [x] Create `.github/dependabot.yml`:
  - Weekly updates for `cargo` and `github-actions` ecosystems
- [x] Consider adding `deny.toml` for `cargo-deny`:
  - License auditing (important for Apache 2.0 — ensures dependency licenses are compatible)
  - Security advisory DB checks
- [x] Consider `cargo llvm-cov` in CI (adds ~2min, useful for visibility, don't block on thresholds)

**Files to create**: `.github/workflows/ci.yml`, `.github/workflows/e2e.yml`, `.github/dependabot.yml`
**Optional**: `deny.toml`

---

## 6. Repository Settings

**Dependencies**: Section 5 (CI checks must exist before requiring them)

- [x] Branch protection on `main` — **deferred to Section 12** (requires GitHub Pro for private repos):
  - Require status checks to pass: `test`, `clippy`, `fmt`
  - Require branches to be up to date before merging
  - Allow force pushes: no
  - Allow deletions: no
  - Note: As sole admin, you can bypass when needed. Protection is for discipline and future contributors.
- [x] Merge strategy:
  - Enable squash merging (default for PRs)
  - Enable rebase merging
  - Disable merge commits (keep history linear)
  - Auto-delete head branches after merge
- [x] Issue labels:
  - **Component**: `codec`, `indexing`, `search`, `analysis`, `store`, `util`, `cli`
  - **Type**: `bug`, `enhancement`, `documentation`, `infrastructure`
  - **Priority**: `priority:high`, `priority:low`
  - **Status**: `good first issue`, `help wanted`, `blocked`
  - **Lucene**: `lucene-compat`
- [x] Clean up default labels that don't apply (kept all defaults, they're all useful)

---

## 7. Code Cleanup for Rename

**Dependencies**: Section 3 (Cargo.toml name change)

All references to `rust-lucene-indexer` / `rust_lucene_indexer`:

- [x] `Cargo.toml` — `name = "bearing"` (done in Section 3)
- [x] `src/bin/indexfiles.rs` — 6 `use rust_lucene_indexer::` imports → `use bearing::`
- [x] `src/index/documents_writer_per_thread.rs:95` — `"rust-lucene-indexer"` string → `"bearing"` (segment diagnostics metadata)
- [x] `CLAUDE.md:7` — repo URL → `github.com/toddfeak/bearing`
- [x] `PLAN.md:1` — title update (moved to `docs/PLAN.md`)
- [x] `Cargo.lock` — regenerated by `cargo build`

Additional cleanup:
- [x] Delete empty `rust-index/` directory
- [x] Verify `cargo build` succeeds
- [x] Verify all 353 tests pass
- [x] Verify `cargo clippy` and `cargo fmt` clean

**Note**: The diagnostics string change means new indexes will have `"bearing"` in `.si` metadata instead of `"rust-lucene-indexer"`. Informational only, not functional.

---

## 8. Test Infrastructure

**Dependencies**: Section 7 (rename must be complete)

- [x] Keep unit tests in `#[cfg(test)] mod tests` within source files (idiomatic Rust, no change needed)
- [x] Keep existing e2e tests:
  - `tests/e2e_indexfiles.sh` — roundtrip: Bearing writes, Java Lucene reads
  - `tests/compare_java_rust.sh` — performance comparison
  - `tests/VerifyIndex.java` and `tests/IndexAllFields.java` — Java utilities
- [x] Consider adding Rust integration tests in `tests/*.rs` (tests against public API surface)
- [x] Roundtrip test strategy (document now, implement as capabilities are added):
  - **Bearing writes → Lucene reads**: Already working via `VerifyIndex.java`
  - **Lucene writes → Bearing reads**: Blocked until read path exists (Tier 3)
  - **Bearing writes → Bearing reads**: Blocked until read path exists
- [x] Lucene test fixtures:
  - Investigate extracting test indexes from Lucene's test suite as golden fixtures
  - Consider `testdata/fixtures/` for binary index files written by Java Lucene
  - Defer implementation until read path exists
- [x] Future: property-based testing (`proptest`) for codec round-trips

---

## 9. Documentation

**Dependencies**: Section 7

- [x] Adapt `CLAUDE.md`:
  - Update project name and repo URL
  - Expand scope: "A Rust port of Apache Lucene" (not just indexing)
  - Streamlined: removed Known Limitations, moved operational details to `tests/CLAUDE.md` and `reference/CLAUDE.md`
  - Added Directory Layout table pointing to subdirectory CLAUDE.md files
- [x] Rewrite `PLAN.md` for expanded Bearing scope:
  - Rename to "Bearing — Roadmap"
  - Keep performance summary
  - Expand Tier 3 (search) into multiple tiers covering the full Lucene surface
  - Add tiers for: advanced query types, analyzers, faceting, spatial, suggesters
  - Add "Non-Goals" section
  - Add version compatibility matrix
- [x] Review `reference/formats/MAINTAINING.md` — no repo name references found, clean
- [x] Create `CHANGELOG.md`:
  - Use [Keep a Changelog](https://keepachangelog.com/) format
  - Start with `[Unreleased]` section (no versioned entries — project is pre-release)

**Files to modify**: `CLAUDE.md`, `PLAN.md`
**Files to create**: `CHANGELOG.md`

---

## 10. Lucene Version Compatibility Strategy

**Dependencies**: Section 9

- [x] Document compatibility policy in `COMPATIBILITY.md`:
  - Current target: **Apache Lucene 10.3.2**
  - Codec target: **Lucene103** (constituents: Lucene90, Lucene94, Lucene99)
  - Guarantee: Bearing-written indexes MUST be readable by Java Lucene 10.3.2
  - Aspiration: Bearing should eventually read Java Lucene 10.3.2 indexes
- [x] ~~Add version constant in code~~ — skipped, not useful enough to justify
- [x] Define upgrade strategy in `COMPATIBILITY.md`:
  - Minor codec changes (Lucene103 → Lucene104): new codec version module
  - Major version changes: new codec modules, maintain backwards compat for reading
  - Track Lucene releases via GitHub issues
- [x] Surface codec version mapping in `COMPATIBILITY.md`
- [x] Future: golden index tests in CI — documented in `COMPATIBILITY.md`, deferred until read path exists

---

## 11. Style & Tooling

**Dependencies**: Section 5

- [x] Evaluate `rustfmt.toml` — not needed; `rustfmt` reads edition from `Cargo.toml` automatically
- [x] Skip `clippy.toml` — CI enforces `-D warnings`, CLAUDE.md says don't suppress warnings
- [x] `rust-toolchain.toml` created in Section 3

---

## 12. Pre-Public Checklist

**Dependencies**: ALL previous sections

- [x] **Secrets audit**: Scan codebase for `.env`, `.pem`, `.key`, credentials (fresh repo has no history to audit)
- [x] **Content audit**: No hardcoded local paths (`/home/rfeak/...`), no inappropriate TODOs, no personal info beyond git author
- [x] **Legal audit**: LICENSE exists, NOTICE exists, Cargo.toml has `license = "Apache-2.0"`, no verbatim Lucene copies without attribution
- [x] **Branch protection**: Enable branch protection on `main` (deferred from Section 6 — requires GitHub Pro or public repo)
- [x] **CI verification**: All checks pass on `main`, branch protection active
- [x] **Documentation verification**: README complete, CONTRIBUTING exists, no "rust-lucene-indexer" references remain
- [x] **Build verification**: `cargo build`, `cargo test`, `cargo clippy -- -D warnings`, `cargo fmt -- --check` all clean
- [x] **History review**: Fresh repo with no prior history — verify initial commit is clean
- [x] **Flip to public**: Settings > Danger Zone > Change visibility

---

## 13. Post-Public / Future (not blocking)

- [ ] Release strategy: define when to tag `v0.1.0`, adopt semver
- [ ] Annotated git tags (`git tag -a v0.1.0 -m "Initial release"`)
- [ ] GitHub Releases with changelog excerpts
- [ ] Crates.io publishing (after API surface stabilizes)
- [x] README badges: CI status, crates.io, docs.rs, license, MSRV
- [x] Ensure `cargo doc` builds cleanly for docs.rs
- [ ] Enable GitHub Discussions when community grows
- [ ] `criterion` benchmark suite with CI tracking
- [ ] `cargo fuzz` for codec edge cases
- [ ] Cross-platform CI (macOS, Windows) for index portability testing

---

## Dependency Graph

```
1. GitHub Repo Setup
├── 2. License & Legal
│   └── 3. Project Identity
│       └── 7. Code Cleanup for Rename
│           ├── 8. Test Infrastructure
│           └── 9. Documentation
│               └── 10. Lucene Version Compat
├── 4. Community Files (parallel with 2, 5)
├── 5. CI/CD (parallel with 2, 4)
│   ├── 6. Repository Settings
│   └── 11. Style & Tooling
└── 12. Pre-Public Checklist (depends on ALL above)
     └── 13. Post-Public
```

Sections 2, 4, and 5 can proceed in parallel after Section 1. Sections 3 → 7 → 8/9 are sequential. Section 12 is the final gate.
