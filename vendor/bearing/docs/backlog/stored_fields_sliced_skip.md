# Stored Fields: LZ4 partial decompression

## Problem

The Rust `StoredFieldsReader` always decompresses the entire chunk, even when only one document's bytes are needed. This applies to both sliced and non-sliced chunks.

For sliced chunks, all sub-blocks are decompressed even when the target document only spans a subset. For non-sliced chunks, the entire block is decompressed when only `[offset, offset+length)` bytes are needed.

## Lucene behavior

Java's `Decompressor.decompress(input, totalLength, offset, length, bytes)` supports partial decompression:
- **Sliced chunks**: skips sub-blocks before the target document's byte offset, only decompressing from the first relevant sub-block onward
- **Non-sliced chunks**: decompresses from `offset` for `length` bytes rather than the full `totalLength`

## Impact

Low for typical workloads with BlockState caching (added in the current implementation). The cache means consecutive reads in the same block avoid re-decompression entirely. Partial decompression would primarily help random access to large blocks where only one small document is needed.

## Fix

Add `offset` and `length` parameters to the Rust LZ4 decompressor (`src/encoding/lz4.rs`) so it can skip unneeded bytes. For sliced chunks, track cumulative sub-block sizes and skip past sub-blocks that end before the target offset. For non-sliced chunks, decompress only the needed byte range.
