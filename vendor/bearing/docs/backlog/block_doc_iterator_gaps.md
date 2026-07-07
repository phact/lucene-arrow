# BlockPostingsEnum gaps

Deferred features in the read-side posting list iterator
(`src/codecs/lucene103/postings_reader.rs::BlockPostingsEnum`).

## Positions not decoded at query time

- **No `next_position()` / position iteration**: `BlockPostingsEnum` opens
  `.pos` only to validate its header at construction, then drops the handle.
  `refill_full_block`, `skip_level0_to`, `skip_level1_to`, and
  `do_move_to_next_level0_block` all skip past pos metadata ("positions not
  supported yet" comments). The `PostingsEnumProducer` trait defines
  `next_position`, but the only implementor is the write-path
  `BufferedPostingsEnum`. Needed for phrase queries, span queries, and
  hit highlighting.

## Offsets not decoded at query time

- **No offset values returned at query time**: `BlockPostingsEnum` does not
  open the `.pay` file, decode PFOR offset blocks, or return offset values.
  The reader correctly skips over offset metadata in `.psm`, term state, and
  skip data (no errors), but callers requesting offsets get nothing. Needed
  for highlighter support. The write path is complete — PFOR offset blocks
  in `.pay` and VInt offset tail in `.pos` are written correctly.

## Payloads not decoded at query time

- **No payload values returned at query time**: Same root cause as offsets —
  `.pay` is never opened at read time. Less urgent than positions/offsets
  because the write path does not support payloads either (see
  `known_issues.md` item 3); both sides must be completed together.
