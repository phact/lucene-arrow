# Segment Merging Design

## Overview

Segment merging combines multiple small segments into fewer larger segments.
This reduces file handle count, improves query performance (fewer segments to
search), and reclaims space from deleted documents. Lucene triggers merging
automatically after flushes and on explicit `force_merge()` calls.

The design has three layers:

1. **MergePolicy** — decides *which* segments to merge
2. **MergeScheduler** — decides *when* and *how* merges execute (threading)
3. **SegmentMerger** — orchestrates codec-level merge of all data components

Codec writers already accept producer traits (`&dyn NormsProducer`,
`&dyn FieldTerms`, etc.) thanks to the abstractions in `codec_writer_abstractions.md`.
Merging plugs file-backed readers into those same traits. The writers don't know
whether data comes from in-memory buffers (flush) or on-disk segments (merge).

## Lucene Reference

All Java sources are in `reference/lucene-10.3.2/lucene/core/src/java/org/apache/lucene/index/`.

| Class | Purpose |
|---|---|
| `MergePolicy` | Abstract base; `findMerges()` returns merge candidates |
| `TieredMergePolicy` | Default policy; selects least-cost merge from tiered segments |
| `MergeScheduler` | Abstract base; `merge()` executes pending merges |
| `SerialMergeScheduler` | Synchronous; merges on calling thread |
| `ConcurrentMergeScheduler` | Background thread pool with configurable limits |
| `SegmentMerger` | Orchestrates codec merge: fields, norms, postings, DV, points, TV |
| `MergeState` | Carries per-segment readers, doc maps, and merged field infos |
| `MergePolicy.OneMerge` | Represents a single merge: source segments + target segment |

## Architecture

### Data Flow

```
IndexWriter
  │
  ├─ flush() produces new segment
  │    └─ maybeMerge(SEGMENT_FLUSH)
  │         └─ MergePolicy::find_merges(segment_infos) → Vec<OneMerge>
  │              └─ MergeScheduler::merge(pending_merges)
  │                   └─ for each OneMerge:
  │                        ├─ merge_init(): open SegmentReaders, build MergeState
  │                        ├─ merge_middle(): SegmentMerger::merge()
  │                        │    ├─ merge_field_infos()
  │                        │    ├─ merge_stored_fields()
  │                        │    ├─ merge_norms()
  │                        │    ├─ merge_terms()
  │                        │    ├─ merge_doc_values()
  │                        │    ├─ merge_points()
  │                        │    ├─ merge_term_vectors()
  │                        │    └─ write_field_infos()
  │                        └─ commit_merge(): swap old segments for new in SegmentInfos
  │
  ├─ force_merge(max_segments)
  │    └─ MergePolicy::find_forced_merges(...)
  │
  └─ commit()
       └─ maybeMerge(FULL_FLUSH)
```

### MergeState

Central struct passed through all codec merge methods. Carries per-segment
readers and the doc ID mapping needed to renumber documents in the merged output.

```rust
pub struct MergeState<'a> {
    /// Per-segment doc ID remapping. `doc_maps[i].get(old_doc) -> new_doc`.
    /// Returns -1 for deleted documents (future: when live docs exist).
    pub doc_maps: Vec<DocMap>,

    /// Per-segment field metadata.
    pub field_infos: Vec<&'a FieldInfos>,

    /// Merged field metadata for the output segment.
    pub merge_field_infos: FieldInfos,

    /// Per-segment max doc count.
    pub max_docs: Vec<i32>,

    /// Per-segment live docs bitsets. `None` means all docs are live.
    pub live_docs: Vec<Option<&'a FixedBitSet>>,

    /// Per-segment codec readers (borrowed from SegmentReaders).
    pub stored_fields_readers: Vec<Option<&'a StoredFieldsReader>>,
    pub norms_producers: Vec<Option<&'a NormsReader>>,
    pub doc_values_producers: Vec<Option<&'a DocValuesReader>>,
    pub term_vectors_readers: Vec<Option<&'a TermVectorsReader>>,
    pub points_readers: Vec<Option<&'a PointsReader>>,
    pub fields_producers: Vec<Option<&'a BlockTreeTermsReader>>,

    /// Output segment info.
    pub segment_info: SegmentInfo,
}
```

### DocMap

Maps old doc IDs to new positions in the merged segment.

```rust
pub struct DocMap {
    doc_base: i32,
    live_docs: Option<FixedBitSet>,
    // When live_docs is None: get(id) = doc_base + id
    // When live_docs is Some: get(id) = doc_base + (id - deleted_before(id)), or -1 if deleted
}

impl DocMap {
    pub fn get(&self, doc_id: i32) -> i32 { ... }
}
```

Without deletes (our current state), `get(id)` is simply `doc_base + id` where
`doc_base` is the cumulative doc count from prior segments. The `live_docs` path
exists for future delete support but costs nothing when `None`.

### DocIDMerger

Coordinates iteration across multiple segments in doc ID order. Used by stored
fields, term vectors, norms, and doc values merge paths.

```rust
pub struct DocIDMerger<S: MergeSub> {
    subs: Vec<S>,
    current: usize,
}

pub trait MergeSub {
    fn next_doc(&mut self) -> io::Result<i32>;
    fn mapped_doc_id(&self) -> i32;
}
```

Lucene's `DocIDMerger` handles both sequential (no index sort) and heap-based
(with index sort) modes. Since we don't support index-time sorting, the
sequential mode is sufficient: iterate each segment's docs in order, applying
`DocMap` to rebase IDs.

## Merge Policy

### TieredMergePolicy

The default policy. Selects merges that minimize total merge cost across a
tiered segment size model.

**Algorithm (from `findMerges()`):**

1. Sort segments by decreasing size (pro-rated by delete percentage)
2. Remove segments larger than `max_merged_segment_mb / 2` (too large to merge)
3. Compute allowed segment count via logarithmic tiering:
   each tier holds `segs_per_tier` segments, tier size increases by
   `merge_factor = min(max_merge_at_once, segs_per_tier)`
4. If actual count <= allowed count, no merge needed
5. Otherwise, evaluate candidate merges starting from each eligible segment:
   - Greedily add adjacent segments up to `max_merge_at_once`
   - Segments below `floor_segment_mb` can exceed `max_merge_at_once`
   - Skip candidates whose merged size would exceed `max_merged_segment_mb`
6. Score each candidate: `skew * size^0.05 * (1 - reclaimed_ratio)^2`
   - Skew: ratio of largest segment to total (lower = more balanced = better)
   - Size: gentle penalty against very large merges
   - Reclaim: strong bonus for merges that recover deleted space
7. Select lowest-score candidate, register it, repeat until within budget

**Default parameters:**

| Parameter | Default | Purpose |
|---|---|---|
| `max_merge_at_once` | 10 | Max segments per merge |
| `max_merged_segment_mb` | 5120 | Output size cap |
| `floor_segment_mb` | 16 | Small segments merge aggressively below this |
| `segs_per_tier` | 8.0 | Segments allowed per tier level |
| `deletes_pct_allowed` | 20 | Trigger threshold for delete reclamation |

### Trait Design

```rust
pub trait MergePolicy: fmt::Debug + Send + Sync {
    fn find_merges(
        &self,
        trigger: MergeTrigger,
        segment_infos: &SegmentInfos,
    ) -> io::Result<Option<MergeSpecification>>;

    fn find_forced_merges(
        &self,
        segment_infos: &SegmentInfos,
        max_segment_count: u32,
    ) -> io::Result<Option<MergeSpecification>>;

    fn use_compound_file(&self, merged_info: &SegmentInfo) -> bool;
}

pub enum MergeTrigger {
    SegmentFlush,
    FullFlush,
    Explicit,
    Closing,
}

pub struct MergeSpecification {
    pub merges: Vec<OneMerge>,
}

pub struct OneMerge {
    pub segments: Vec<SegmentCommitInfo>,
    pub info: Option<SegmentCommitInfo>,  // set during merge_init
    pub estimated_merge_bytes: u64,
}
```

## Merge Scheduler

### SerialMergeScheduler

The initial implementation. Merges execute synchronously on the calling thread.

```rust
pub trait MergeScheduler: fmt::Debug + Send + Sync {
    fn merge(&self, source: &mut dyn MergeSource) -> io::Result<()>;
}

pub trait MergeSource {
    fn next_merge(&mut self) -> Option<OneMerge>;
    fn do_merge(&mut self, merge: OneMerge) -> io::Result<()>;
}
```

`SerialMergeScheduler` is a loop:

```rust
impl MergeScheduler for SerialMergeScheduler {
    fn merge(&self, source: &mut dyn MergeSource) -> io::Result<()> {
        while let Some(merge) = source.next_merge() {
            source.do_merge(merge)?;
        }
        Ok(())
    }
}
```

This blocks indexing during merges. Acceptable for the initial implementation
and for the standalone merge tool.

### ConcurrentMergeScheduler (future)

Spawns background threads for merge execution. Parameters: `max_thread_count`,
`max_merge_count`. Not needed initially — serial merging is correct and
sufficient for validating the merge pipeline.

## SegmentMerger

Orchestrates the merge of all data components. Owns the `MergeState` and
delegates to codec-specific merge methods.

**Merge order** (matches Lucene's `SegmentMerger.merge()`):

1. `merge_field_infos()` — build merged `FieldInfos` from all source segments
2. `merge_stored_fields()` — stored fields via streaming writer
3. `merge_norms()` — norms via `NormsProducer` wrapping
4. `merge_terms()` — postings/terms via `FieldsConsumer`
5. `merge_doc_values()` — doc values via `DocValuesProducer` wrapping
6. `merge_points()` — BKD points via visitor pattern
7. `merge_term_vectors()` — term vectors via streaming writer
8. Write merged `FieldInfos` to `.fnm`
9. Write segment info to `.si`
10. Optionally create compound file (`.cfs`/`.cfe`)

Each step is conditional — only runs if the merged `FieldInfos` indicates the
relevant data type exists.

## Codec Merge Patterns

Each codec component follows one of three patterns for merging. These align
with the producer trait patterns from `codec_writer_abstractions.md`.

### Pattern 1: Producer Wrapping (Norms, Doc Values)

The merge method creates a synthetic producer that wraps multiple source
readers into a single logical view. The existing codec writer consumes this
wrapped producer identically to how it consumes a `Buffered*Producer` during
flush.

```
Source SegmentReaders   →   Wrapped MergeProducer   →   Codec Writer
   NormsReader[0]    ─┐
   NormsReader[1]    ─┼→  MergedNormsProducer  →  norms::write()
   NormsReader[2]    ─┘
```

The `MergedNormsProducer` implements `NormsProducer` and internally uses
`DocIDMerger` to iterate docs across segments, applying `DocMap` for
renumbering. From the writer's perspective, it looks like a single segment's
worth of norms data.

Doc values follow the same pattern but additionally need ordinal remapping for
`SORTED` and `SORTED_SET` types (deduplicating term ordinals across segments).

### Pattern 2: Streaming Push (Stored Fields, Term Vectors)

Merging iterates source readers document by document and pushes each document
through the streaming writer trait.

```
Source SegmentReaders   →   DocIDMerger   →   Streaming Writer
   StoredFieldsReader[0]  ─┐
   StoredFieldsReader[1]  ─┼→  iterate docs  →  writer.start_document()
   StoredFieldsReader[2]  ─┘                    writer.write_field(...)
                                                 writer.finish_document()
```

The merge path calls the same `StoredFieldsWriter` / `TermVectorsWriter` trait
methods that the flush path uses. No new writer code needed.

Field number remapping: if field numbers differ across source segments (they
can when segments were written by different `IndexWriter` sessions with
different field registration order), the merge must remap field numbers to the
merged `FieldInfos` numbering. This is a lookup by field name.

### Pattern 3: Multi-Segment Terms (Postings)

Postings merge wraps multiple source `FieldsProducer` instances into a single
`MappedMultiFields` view. This merges term dictionaries across segments (terms
that appear in multiple segments are combined) and remaps doc IDs in postings
lists.

```
Source SegmentReaders   →   MappedMultiFields   →   BlockTreeTermsWriter.write_field()
   FieldsProducer[0]  ─┐
   FieldsProducer[1]  ─┼→  merged field/term iteration
   FieldsProducer[2]  ─┘     with DocMap applied to postings
```

The merged view implements `FieldsProducer` (field iteration) and `FieldTerms`
(per-field term/postings access). For each term, postings from all segments
containing that term are merged into a single stream with remapped doc IDs.

This is the most complex merge path. It requires:
- Multi-segment term merging (sorted merge of term dictionaries)
- Doc ID remapping within postings lists
- A `FieldTerms` implementation over the merged view (random-access term bytes
  for block tree construction)

#### Merge vs Flush for Postings

During flush, `BufferedFieldTerms` borrows term bytes zero-copy from the
`ByteBlockPool`. During merge, the `FieldTerms` implementation reads terms from
the block tree index (`.tim` files) of source segments. The block tree writer
sees the same `FieldTerms` trait either way.

Key difference: flush has all terms pre-sorted in memory. Merge must perform a
sorted merge of multiple already-sorted term streams. This is a standard
k-way merge.

#### Norms During Postings Merge

The postings writer needs norms for competitive impact computation (BM25
scoring metadata). During merge, `merge_terms()` opens the just-written merged
norms file and passes it to the postings writer — the same pattern as flush.
See Lucene's `SegmentMerger.mergeTerms()` which opens a `NormsProducer` over
the merged segment's norms output.

### Pattern 4: Visitor (Points)

Points/BKD merge uses a visitor pattern rather than `DocIDMerger`. Each source
segment's point tree is visited, doc IDs are remapped via `DocMap`, and the
remapped points are fed into the BKD tree builder.

```
Source SegmentReaders   →   Visitor with DocMap   →   BKD Builder
   PointsReader[0]  ─┐
   PointsReader[1]  ─┼→  visit all points, remap doc IDs  →  points::write()
   PointsReader[2]  ─┘
```

This rebuilds the BKD tree from scratch. No bulk-copy optimization.

## Performance Considerations

### Zero-Copy and Minimal Copy

The merge path should avoid unnecessary copies:

- **Stored fields**: Read compressed blocks from source, re-compress into
  output. No intermediate owned `Vec<u8>` per document; pass field values as
  slices through the visitor.
- **Norms / doc values**: The merged producer returns values by reference or
  via streaming iterators. Writers consume values one at a time.
- **Postings**: `PostingsEnumProducer` streams doc/freq/position data. No
  pre-collection into `Vec`.
- **Term bytes**: The `FieldTerms` merge implementation should provide term
  bytes as `&[u8]` slices from the source `.tim` data. For the k-way merge,
  only the current front of each segment's iterator needs to be live.
- **Points**: Visitor passes `&[u8]` packed values directly.

### Separate Code Paths

Some codec components may benefit from merge-specific optimizations that differ
from the flush path. Lucene allows this via overridable `merge()` methods on
codec consumers.

Potential merge-specific optimizations (not for initial implementation):

- **Stored fields bulk copy**: When source and target use the same compression
  format and no field remapping is needed, copy compressed blocks directly
  without decompressing. Lucene's `CompressingStoredFieldsMergeSub` supports
  this.
- **Points bulk merge**: When segments have non-overlapping value ranges, BKD
  subtrees could be spliced without revisiting individual points.

For the initial implementation, all merge paths should go through the standard
codec writer APIs. Optimize later with profiling data.

## Standalone Merge Tool

A command-line tool that merges an existing index without indexing new documents.
Equivalent to Lucene's `IndexWriter.forceMerge()`.

```
bearing-merge --index /path/to/index --max-segments 1
```

**Design:**

1. Open the directory and read `segments_N`
2. Open `SegmentReader` for each segment
3. Run `MergePolicy::find_forced_merges(segment_infos, max_segments)`
4. For each `OneMerge`:
   a. Build `MergeState` from source `SegmentReader`s
   b. Run `SegmentMerger::merge()`
   c. Update `SegmentInfos` (remove old, add new)
5. Write new `segments_N+1`
6. Delete old segment files

This tool reuses the same `MergePolicy`, `SegmentMerger`, and codec writer
infrastructure as in-IndexWriter merging. The only difference is the entry
point: instead of `maybeMerge()` after a flush, it's an explicit
`find_forced_merges()` call.

**Index safety:**

- The tool must not corrupt the index on failure. Write new segment files
  first, then atomically update `segments_N`, then delete old files.
- Old segment files are only deleted after the new `segments_N` is committed.
  If the process crashes mid-merge, the old `segments_N` still points to valid
  segments.

## Future Compatibility

### Deletes

Deletes are handled via `MergeState.live_docs` and `DocMap`. When `live_docs`
is `Some`, deleted documents are skipped during merge (DocMap returns -1).
The current design carries `live_docs: Vec<Option<&FixedBitSet>>` — `None`
means all docs are live. Adding delete support later requires:

1. Reading `.liv` files into `FixedBitSet`
2. Passing them into `MergeState`
3. Implementing the deletion-compacting `DocMap` (cumulative offset tracking)

No merge infrastructure changes needed.

### Index-Time Sorting

`DocIDMerger` in Lucene supports a heap-based mode for sorted indexes (where
merged doc order follows the sort rather than segment order). The design
supports this by having `DocIDMerger` accept a `needs_index_sort` flag. Not
needed now, but the `DocMap` and `DocIDMerger` abstractions accommodate it.

### ConcurrentMergeScheduler

The `MergeScheduler` trait abstracts scheduling. Swapping
`SerialMergeScheduler` for `ConcurrentMergeScheduler` later requires no
changes to merge logic — only the scheduling layer.

### Append Mode

`IndexWriter` with `OpenMode::Append` or `CreateOrAppend` must read
`segments_N` to discover existing segments. Merging naturally follows: the
appended segments participate in merge selection alongside existing ones. The
merge infrastructure is agnostic to how segments were created.

## Prerequisites

Before implementing merge, the following must be in place:

1. **SegmentReader** — exists (`src/index/segment_reader.rs`), opens all codec readers
2. **Producer traits on codec writers** — done for norms, doc values, points, postings
   (see `codec_writer_abstractions.md` phases 1-3)
3. **segment_infos read/write** — reading exists; writing a new `segments_N` after
   merge may need additions
4. **SegmentCommitInfo** — metadata struct tracking per-segment commit state
   (del count, DV generation, etc.)

## What This Does NOT Include

- NRT (near-real-time) reader integration
- Intra-merge parallelism (Lucene's `intraMergeTaskExecutor`)
- I/O throttling (`RateLimitedIndexOutput`)
- Merge progress tracking / abort
- Soft deletes
