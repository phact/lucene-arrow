# harness — Java golden-file generator + CheckIndex gate (SPEC §12)

Two correctness gates only Java Lucene itself can provide:

1. **Golden segments** (`GenerateGolden.java`): writes Lucene 10.3.2 segments
   with known values — the shapes our reader must decode (dense/sparse
   numerics, table/GCD-friendly distributions, a large field that triggers
   Java's *multi-block* numeric encoding, deletes, multi-segment commits) —
   plus an `expected.json` with the per-doc values.
2. **CheckIndex gate** (`check_index.sh`): every segment our writer produces
   must pass `org.apache.lucene.index.CheckIndex`.

## Requirements

- **JDK 21+** (Lucene 10.x requires it; this machine currently has JDK 17 —
  install a Temurin 21 to run these).
- Network access to Maven Central on first run (downloads
  `lucene-core-10.3.2.jar`).

## Usage

```bash
./run.sh golden /path/to/golden-output      # generate golden segments + expected.json
./run.sh check  /path/to/segment-directory  # CheckIndex gate
```

Until this runs in CI, the interim correctness story is in
`crates/codec/tests/p1_docvalues.rs`: segments are written by Bearing
(byte-identical to Java Lucene, cross-validated upstream), our decode is
checked against source values, and our encode is checked **byte-for-byte**
against Bearing's output. The Java harness upgrades that to a direct
first-party gate, and adds the shapes Bearing never emits (multi-block
numerics, deletes).
