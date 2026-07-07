# Merge Implementation Phases

Phases for implementing segment merging, derived from `merge_design.md`.
Each phase is a single-commit unit of work.

## Porting Rules

All merge infrastructure is ported from Java Lucene 10.3.2. The standard
porting rules apply: read the complete Java source first, port line-by-line
preserving control flow and naming, show Java→Rust line mapping when done.
Do not restructure algorithms.

## Current State

The codec writer abstraction work (`codec_writer_abstractions.md`) is
substantially complete:

- **Done**: `NormsProducer`, `DocValuesProducer`, `PointsProducer`, `FieldTerms`,
  `PostingsEnumProducer`, `NormsLookup` traits. `BufferedNormsProducer`,
  `BufferedFieldTerms`, `BufferedPointsProducer`, `BufferedNormsLookup` structs.
  All codec writers accept `&dyn Producer` traits.
- **Done**: `StoredFieldsWriter` trait, `TermVectorsWriter` trait (streaming
  write-side interfaces).
- **Done**: `SegmentReader` with all codec readers. `SegmentCommitInfo` struct.
  `segment_infos::read()` and `write_flushed_segments()`.
- **Missing from readers**: `TermVectorsReader` has no per-document read API
  (can open files but cannot retrieve vectors for a given doc). `PointsReader`
  has metadata accessors but no point-value visitor/iteration API.

---

## Phase 1: DocMap and DocIDMerger

Foundation types used by every codec merge path.

**Java source**:
- `MergeState.java` — `DocMap` interface, `buildDeletionDocMaps()`
- `DocIDMerger.java` — sequential and heap-based merge iteration

**Deliverables**:

- `DocMap` struct with `get(doc_id: i32) -> i32`. Two construction paths:
  without deletes (`doc_base + id`) and with deletes (cumulative offset via
  `PackedLongValues`). The delete path can use a simple `Vec<i32>` initially
  since we have no deletes — the important thing is that the interface exists.
- `DocIDMerger<S: MergeSub>` with sequential iteration mode. Iterates one
  segment at a time in order, calling `doc_map.get()` to remap IDs. Each sub
  yields its docs in order; the merger advances to the next sub when one is
  exhausted.
- `MergeSub` trait: `next_doc(&mut self) -> io::Result<i32>` and
  `mapped_doc_id(&self) -> i32`.
- Unit tests: merge 3 segments with known doc counts, verify output doc IDs
  are sequential `0..total`. Test with `live_docs = None` (all live).

**File placement**: `src/index/merge_state.rs` for `DocMap`, `MergeState`.
`src/index/doc_id_merger.rs` for `DocIDMerger` and `MergeSub`.

## Phase 2: MergeState and FieldInfos Merging

Build the `MergeState` struct that carries all per-segment readers and merged
metadata through the merge pipeline.

**Java source**:
- `MergeState.java` — constructor, field arrays
- `FieldInfos.java` — `FieldNumbers`, merge logic in `SegmentMerger.mergeFieldInfos()`

**Deliverables**:

- `MergeState` struct as sketched in `merge_design.md`. Borrows from
  `SegmentReader` instances. Fields: `doc_maps`, `field_infos` (per-segment),
  `merge_field_infos`, `max_docs`, `live_docs`, per-component reader slices,
  `segment_info` (output).
- `merge_field_infos()` function: union of all source `FieldInfos`, assigning
  merged field numbers. Lucene's `FieldNumbers` tracks global field numbering.
  Port the merge path from `SegmentMerger.mergeFieldInfos()`.
- Field number remapping: when source segments have different field number
  assignments, merge must map by field name. Build a `field_number_map:
  Vec<HashMap<u32, u32>>` (per-segment old→new mapping) or rely on name-based
  lookup during streaming writes.
- Construction from a `Vec<&SegmentReader>`: extract each reader's
  `field_infos()`, codec readers, and `max_doc()` into the `MergeState` arrays.
- Unit tests: build `MergeState` from 2-3 `SegmentReader`s over in-memory
  indexes. Verify merged field infos contain the union of all fields with
  correct numbering.

**File placement**: extend `src/index/merge_state.rs`.

## Phase 3: Stored Fields Merge

The simplest codec merge path. Validates the full pipeline from
`MergeState` through codec writer.

**Java source**:
- `StoredFieldsWriter.java` — `merge(MergeState)` default method
- `StoredFieldsWriter.MergeVisitor` — field number remapping visitor

**Deliverables**:

- `StoredFieldsMergeSub` implementing `MergeSub`: wraps a `StoredFieldsReader`
  reference and a `DocMap`. `next_doc()` advances through the source segment's
  docs, skipping deleted docs via `DocMap`.
- `merge_stored_fields()` function: creates `DocIDMerger<StoredFieldsMergeSub>`,
  opens a `StoredFieldsWriter` for the output segment, iterates merged docs in
  order. For each doc: `writer.start_document()`, read fields from source
  reader via `StoredFieldsReader::document(doc_id)`, write each field with
  remapped field number, `writer.finish_document()`.
- Field number remapping: when writing a field from a source segment, look up
  the field name in the merged `FieldInfos` to get the output field number.
  If all source segments share the same field numbering (common when they come
  from the same `IndexWriter`), skip remapping.
- Integration test: write 2 small indexes to the same directory with different
  segments, merge them, open the merged segment with `SegmentReader`, verify
  all stored fields are present with correct values.

**Note**: `StoredFieldsReader::document()` currently returns
`Vec<StoredField>`. This allocates per-document. Acceptable for correctness;
a future optimization could add a visitor-based API to avoid the intermediate
`Vec`.

## Phase 4: Norms Merge

**Java source**:
- `NormsConsumer.java` — `merge(MergeState)`, `mergeNormsField()`

**Deliverables**:

- `MergedNormsProducer` struct: implements `NormsProducer` over multiple source
  `NormsReader`s. `get_norms(field_info)` returns a synthetic
  `NumericDocValues` that uses `DocIDMerger` internally to iterate across
  segments, remapping doc IDs via `DocMap`. For each source segment, looks up
  the field by name in that segment's `FieldInfos` and calls the source
  reader's `get_norms()`.
- `merge_norms()` function: constructs `MergedNormsProducer`, calls
  `norms::write()` with it. The existing writer sees the same `&dyn
  NormsProducer` interface it uses during flush.
- Unit tests: merge segments where different fields have norms in different
  segments. Verify merged norms file is readable and values match.

## Phase 5: Doc Values Merge

**Java source**:
- `DocValuesConsumer.java` — `merge()`, `mergeNumericField()`,
  `mergeBinaryField()`, `mergeSortedField()`, `mergeSortedSetField()`,
  `mergeSortedNumericField()`

**Deliverables**:

- `MergedDocValuesProducer` struct: implements `DocValuesProducer` over
  multiple source `DocValuesReader`s. Each `get_*()` method returns a
  synthetic iterator using `DocIDMerger` for doc iteration with `DocMap`
  remapping.
- Numeric and binary: straightforward `DocIDMerger`-based iteration, same
  pattern as norms.
- Sorted and sorted-set: require ordinal remapping. Port Lucene's
  `OrdinalMap` — builds a global ordinal space by deduplicating term bytes
  across segments. Each segment's local ordinals map to global ordinals via
  `OrdinalMap.getGlobalOrds()`. This is the most complex part of DV merging.
- Sorted-numeric: port the singleton optimization from Lucene — when all
  source segments are single-valued, treat as plain numeric for efficiency.
- `merge_doc_values()` function: constructs `MergedDocValuesProducer`, calls
  `doc_values::write()`.
- Unit tests per DV type: merge segments with overlapping and disjoint field
  sets, verify output via `DocValuesReader`.

**Complexity note**: `OrdinalMap` is substantial (~300 lines in Java). It may
warrant its own sub-commit. The numeric/binary merge can land first since they
don't need ordinal remapping.

## Phase 6: Points Merge

**Java source**:
- `PointsWriter.java` — `merge(MergeState)`, `mergeOneField()`
- `PointValues` visitor pattern

**Prerequisites**: `PointsReader` needs a point-value visitor API. Currently
it has metadata accessors (`point_count`, `doc_count`, `num_leaves`) but no
way to iterate actual point values. This must be added first.

**Deliverables**:

- Extend `PointsReader` with a visitor/iteration API for reading point values
  from a segment. Port from Lucene's `PointValues.getPointTree()` /
  `IntersectVisitor` pattern, or a simpler flat iterator if the BKD writer
  accepts that.
- `merge_points()` function: for each field with points, visit all source
  segments' point values, remap doc IDs via `DocMap`, feed into `points::write()`.
  Lucene wraps this as a synthetic `PointsReader` passed to `writeField()`.
- The BKD writer (`points::write()`) rebuilds the tree from scratch over the
  merged point set. No bulk-copy optimization.
- Unit tests: merge segments with `LatLonPoint` and range fields, verify
  merged segment point counts and BKD tree structure.

## Phase 7: Postings Merge

The most complex merge path. Merges term dictionaries and postings lists
across segments.

**Java source**:
- `FieldsConsumer.java` — `merge(MergeState, NormsProducer)`
- `MultiFields.java`, `MappedMultiFields.java` — multi-segment term iteration
- `MultiTerms.java`, `MappedMultiTerms.java` — per-field merged terms
- `MultiPostingsEnum.java` — merged postings with doc ID remapping

**Deliverables**:

- `MergedFieldTerms` implementing `FieldTerms`: presents the merged term
  dictionary for one field across all source segments. Internally performs a
  k-way sorted merge of each segment's `TermsEnum` iterators. For each unique
  term, produces a `PostingsEnumProducer` that merges postings from all
  segments containing that term, remapping doc IDs via `DocMap`.
- The k-way merge: each source segment provides a sorted term iterator.
  Advance all to their first term, maintain a min-heap (or equivalent) by
  term bytes. The "current" term is the heap minimum. All segments with that
  same term contribute postings.
- `MergedPostingsEnumProducer`: wraps multiple per-segment
  `PostingsEnumProducer`s for the same term. Yields docs in merged ID order
  by iterating each segment's postings and applying `DocMap`. Since we have
  no deletes and no index sort, segment postings are already in order — just
  concatenate with rebased doc IDs.
- Random-access term bytes for `FieldTerms`: the k-way merge must materialize
  all merged terms before the block tree writer can build blocks (it needs
  random access to all term bytes for prefix computation). Two approaches:
  1. Collect all merged term bytes into a `Vec<Vec<u8>>` — simple, one copy
     per term. Terms are typically small (< 100 bytes).
  2. Use a `ByteBlockPool`-like allocator for the merged terms and return
     slices — zero-copy within the pool but more complex.
  Start with approach 1. Profile later to determine if the copy matters.
- Norms for competitive impacts: `merge_terms()` opens the just-written
  merged norms file as a `NormsReader`, wraps it in a `NormsLookup`, and
  passes it to the block tree writer. This matches Lucene's
  `SegmentMerger.mergeTerms()`.
- `merge_terms()` function: builds `MergedFieldTerms` per field, calls
  `BlockTreeTermsWriter::write_field()` for each.
- Unit tests: merge segments with overlapping terms (same term in multiple
  segments), disjoint terms, and varying index options (DOCS vs
  DOCS_AND_FREQS_AND_POSITIONS). Verify merged postings contain correct doc
  IDs and positions.

**Complexity note**: This phase has the most new code. The k-way merge, merged
postings enum, and `FieldTerms` adapter are all non-trivial. Consider splitting
into sub-commits: (a) `MergedPostingsEnumProducer`, (b) k-way term merge
iterator, (c) `MergedFieldTerms` + `FieldTerms` impl, (d) `merge_terms()`
wiring.

## Phase 8: Term Vectors Merge

**Java source**:
- `TermVectorsWriter.java` — `merge(MergeState)`, `addAllDocVectors()`

**Prerequisites**: `TermVectorsReader` needs a per-document read API. Currently
it can open files but has no method to retrieve term vectors for a specific
document. Port from Lucene's `TermVectorsReader.get(docID)`.

**Deliverables**:

- Extend `TermVectorsReader` with `get(doc_id: i32)` returning per-document
  term vector data (fields, terms, frequencies, positions, offsets). Port the
  decompression and per-document seeking from Lucene's
  `CompressingTermVectorsReader.get()`.
- `merge_term_vectors()` function: same pattern as stored fields merge.
  `DocIDMerger`-based iteration, push each document's vectors through the
  `TermVectorsWriter` streaming API (`start_document`, `start_field`,
  `start_term`, `finish_term`, `finish_field`, `finish_document`).
- Field number remapping via merged `FieldInfos` name lookup.
- Unit tests: merge segments with term vectors enabled, verify merged vectors
  contain correct terms and positions.

## Phase 9: SegmentMerger Orchestration

Wire all component merge functions into a single `SegmentMerger`.

**Java source**:
- `SegmentMerger.java` — constructor, `merge()`, `shouldMerge()`

**Deliverables**:

- `SegmentMerger` struct: takes `Vec<&SegmentReader>`, output `SegmentInfo`,
  output `Directory`. Builds `MergeState` from the readers.
- `merge()` method: calls component merge functions in order (matching
  Lucene's `SegmentMerger.merge()`). Each step is conditional on the merged
  `FieldInfos` having the relevant data type. Returns the final `MergeState`.
- After all components: write merged `.fnm` via `field_infos_format::write()`,
  write `.si` via `segment_info_format::write()`, optionally create compound
  file.
- `should_merge()` method: returns false if all source segments are empty
  (0 docs).
- Integration test: create a multi-segment in-memory index, run
  `SegmentMerger::merge()`, open the merged segment with `SegmentReader`,
  verify all data types are present and correct.

**File placement**: `src/index/segment_merger.rs`.

## Phase 10: TieredMergePolicy

**Java source**:
- `TieredMergePolicy.java` — `findMerges()`, `findForcedMerges()`, `score()`
- `MergePolicy.java` — base class, `OneMerge`, `MergeSpecification`

**Deliverables**:

- `MergePolicy` trait: `find_merges()`, `find_forced_merges()`,
  `use_compound_file()`.
- `MergeTrigger` enum: `SegmentFlush`, `FullFlush`, `Explicit`, `Closing`.
- `OneMerge` struct, `MergeSpecification` struct.
- `TieredMergePolicy` struct implementing `MergePolicy`. Port the
  `findMerges()` algorithm: segment sorting, budget computation, candidate
  evaluation, scoring. Port `findForcedMerges()` for explicit merge requests.
- Configuration: `max_merge_at_once`, `max_merged_segment_mb`,
  `floor_segment_mb`, `segs_per_tier`, `deletes_pct_allowed` with Lucene
  defaults.
- Unit tests: verify merge selection with known segment size distributions.
  Test edge cases: single segment (no merge), all segments below floor
  (aggressive merge), segments at size cap (excluded).

**File placement**: `src/index/merge_policy.rs`.

## Phase 11: SerialMergeScheduler and IndexWriter Integration

Wire merge policy and scheduler into the `IndexWriter` commit path.

**Java source**:
- `SerialMergeScheduler.java`
- `IndexWriter.java` — `maybeMerge()`, `merge()`, `mergeInit()`,
  `mergeMiddle()`, `commitMerge()`

**Deliverables**:

- `MergeScheduler` trait: `merge(&self, source: &mut dyn MergeSource)`.
- `MergeSource` trait: `next_merge()`, `do_merge()`.
- `SerialMergeScheduler`: loops `next_merge()` / `do_merge()` until no
  merges remain.
- `IndexWriter` integration:
  - Add `MergePolicy` and `MergeScheduler` to `IndexWriterConfig` (with
    `TieredMergePolicy` and `SerialMergeScheduler` as defaults).
  - `commit()` gains a `maybe_merge()` call after writing `segments_N`.
    Opens `SegmentReader` for each segment, calls `merge_policy.find_merges()`,
    schedules via `merge_scheduler.merge()`.
  - `do_merge()` implements: `merge_init()` (open readers, build
    `MergeState`), `merge_middle()` (run `SegmentMerger::merge()`),
    `commit_merge()` (update `SegmentInfos`, write new `segments_N`,
    delete old segment files).
  - `force_merge(max_segments)` public API: calls
    `merge_policy.find_forced_merges()`, same execution path.
- `SegmentInfos` updates: after merge, remove old `SegmentCommitInfo` entries,
  add the new merged entry, write new `segments_N+1`. Old segment files are
  deleted only after the new commit point is written.
- Integration test: index enough documents to create multiple segments (use
  small `ram_buffer_size_mb`), commit, verify segment count decreased.
  `force_merge(1)` test: verify single output segment with all documents.

## Phase 12: Standalone Merge Tool

A command-line binary for merging an existing index without indexing.

**Deliverables**:

- New binary target `bearing-merge` (in `src/bin/merge.rs` or similar).
- CLI: `bearing-merge --index <path> [--max-segments N]`. Default
  `max-segments` is 1 (full merge).
- Implementation: open `FSDirectory`, read `segments_N`, open
  `SegmentReader` per segment, run `find_forced_merges()`, execute merges
  via `SerialMergeScheduler`, write new `segments_N`.
- Safety: write new segment files first, commit atomically, delete old files
  last. If process crashes mid-merge, old `segments_N` remains valid.
- E2E test: create a multi-segment index via `IndexWriter`, run
  `bearing-merge`, verify resulting index has expected segment count and all
  documents are intact (cross-validate with Java `CheckIndex` if available).

## Phase 13: Validation and E2E

Final validation across all merge paths.

**Deliverables**:

- E2E merge test in `tests/`: index a corpus with all field types (text,
  keyword, numeric, stored, doc values, points, term vectors), produce
  multiple segments, merge to 1, verify all data is intact via
  `SegmentReader`.
- Java cross-validation: if the Java test harness supports it, create a
  multi-segment index in Java, merge in Rust (or vice versa), verify output.
- Performance baseline: measure merge time and peak memory for the e2e
  corpus. Not an optimization phase — just establish the baseline for future
  work.
- Update `docs/backlog/unsupported_indexing_features.md` to reflect merge
  support.

---

## Dependency Graph

```
Phase 1: DocMap, DocIDMerger
Phase 2: MergeState, FieldInfos merging ← depends on 1
Phase 3: Stored fields merge            ← depends on 2
Phase 4: Norms merge                    ← depends on 2
Phase 5: Doc values merge               ← depends on 2
Phase 6: Points merge                   ← depends on 2, needs PointsReader extension
Phase 7: Postings merge                 ← depends on 2
Phase 8: Term vectors merge             ← depends on 2, needs TermVectorsReader extension
Phase 9: SegmentMerger                  ← depends on 3-8
Phase 10: TieredMergePolicy             ← independent, can parallel with 3-8
Phase 11: IndexWriter integration       ← depends on 9, 10
Phase 12: Standalone merge tool         ← depends on 9, 10
Phase 13: Validation                    ← depends on 11, 12
```

Phases 3-8 can be implemented in any order. Phases 10 and 3-8 are independent
and can proceed in parallel. Phase 6 and 8 each require reader extensions as
prerequisites.

## What This Does NOT Include

- `ConcurrentMergeScheduler` (background merge threads)
- Bulk-copy optimizations (stored fields block copy, BKD subtree splicing)
- Live docs / delete support (the `DocMap` interface supports it; wiring
  `.liv` file reading is separate work)
- Index-time sorting (heap-based `DocIDMerger` mode)
- Merge progress tracking / abort
- I/O throttling
