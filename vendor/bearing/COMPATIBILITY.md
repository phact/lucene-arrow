# Bearing — Lucene Version Compatibility

## Current Target

- **Apache Lucene**: 10.3.2
- **Codec**: Lucene103
- **Guarantee**: Indexes written by Bearing are readable by Java Lucene 10.3.2

## Codec Version Mapping

Lucene only bumps a codec format version when its on-disk format changes. Components unchanged since Lucene 9.0 still use the Lucene 90 format. This is why our 10.3.2 codec references formats spanning Lucene 90 through 103.

| Component | Format Version | Files |
|---|---|---|
| Postings | Lucene 103 | `.doc`, `.pos`, `.pay`, `.psm` |
| Terms Dictionary | Lucene 103 | `.tim`, `.tip`, `.tmd` |
| Field Infos | Lucene 94 | `.fnm` |
| Segment Info | Lucene 99 | `.si` |
| Stored Fields | Lucene 90 | `.fdt`, `.fdx`, `.fdm` |
| Doc Values | Lucene 90 | `.dvd`, `.dvm` |
| Norms | Lucene 90 | `.nvd`, `.nvm` |
| Points/BKD | Lucene 90 | `.kdd`, `.kdi`, `.kdm` |
| Compound | Lucene 90 | `.cfs`, `.cfe` |
| Segments File | — | `segments_N` |

## Version Strategy

### Phase 1 — Lucene 10.3.2 (current)

Complete the write path, then build the read/query path, all targeting 10.3.2 exclusively. No multi-version complexity until the single-version implementation is solid.

### Phase 2 — Track Newer Versions

Once Bearing has full index + query support for 10.3.2, adopt newer Lucene releases as they ship. This requires architecture for:

- Selecting the codec version on write
- Detecting the codec version on read/query
- Running compatibility tests across versions

### Phase 3 — Older Version Support

Consider adding support for older Lucene index formats (both read and write).

## Upgrade Mechanics

When Lucene introduces a new codec version (e.g., Lucene 104):

1. **New module**: Add `src/codecs/lucene104/` for the changed components
2. **Unchanged components**: Continue using the existing format modules (e.g., Lucene 90 stored fields)
3. **Read compatibility**: Maintain older codec modules so Bearing can read indexes written by prior versions
4. **Track via issues**: Monitor Lucene releases and create GitHub issues for format changes

## Future Work

- **Golden index tests**: CI tests using indexes written by specific Lucene versions that Bearing must read correctly. Blocked until the read path exists.
- **Cross-version roundtrip**: Bearing writes with version X, Java Lucene X reads; Java Lucene X writes, Bearing reads.
