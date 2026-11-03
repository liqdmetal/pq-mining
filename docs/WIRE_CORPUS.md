# Fixed-profile witness codec and parity corpus

`tests/consensus_vectors_v1.json` has schema version 1 and fixes
`n_columns=512`, `syndrome_bits=256`, `w_max=32`. It does not approve that
profile for deployment or change its mathematical acceptance rule.

The mathematical `verify_share` accepts any ordering of a nonempty set of
distinct in-range indices with weight at most 32. `encode_pq_witness` and
`decode_pq_witness` require strictly ascending order: bytes `CHITINPQ`, one
unsigned weight byte, then exactly that many little-endian uint16 indices.
Length must be 11..73 bytes. Invalid magic, empty/oversize support, trailing
bytes, truncation, duplicates, unsorted indices and indices >=512 fail closed
before deriving any columns. Miners sort explicitly; parsers never sort.

`verify_pq_witness` decodes first, then checks the mathematical relation.
This is the existing Rust CHITINPQ codec, not a new chain header format.
Header bytes in the corpus are opaque; instance input is
`u32be(header_length) || header || u64be(nonce)`. SHAKE-256 domains remain
`CHITIN-POW-H-V2` and `CHITIN-POW-S-V2`. The verifier derives only selected
columns, preserving the original word padding and matched-prefix definition.

Corpus arrays:

- `derivation`: `id`, `header_hex`, uint64 `nonce`, `target_hex`, and
  `columns` containing `{index,value_hex}` for pinned column values.
- `verify`: `id`, `header_hex`, `nonce`, `matched_bits`, `support`,
  `witness_hex`, `canonical`, `math_expected`, `wire_expected`. A permutation
  can have math true and wire false. For a Rust helper that itself enforces
  ascending support, sort a copy for the math comparison, but do not
  deduplicate or normalize the actual wire bytes. Invalid difficulty values
  are rejection cases and must be checked before constructing strict params.
- `wire`: `id`, `witness_hex`, `canonical`, and nullable `expected_support`.

Run `python tools/gen_consensus_vectors.py` to regenerate deterministically;
it writes only this repository. Cross-repo fixture copying is an explicit
review step. Keep the schema, parameter tuple, full file and SHA-256 pinned
in the importing repository. Regeneration is not permission to change a
consensus verdict. Existing `golden_vectors.json` remains unchanged.

`test_consensus_vectors.py` checks both corpora against an independent copy
of the pre-optimization full-table rule, randomized non-word-aligned
profiles, canonical parsing and bounded derivation counts. No expensive
mining or synthetic high-difficulty valid witness is used.

Reference search supports an absolute monotonic `deadline`, cooperative
`cancelled` callback, arena/column/step limits and a collision-retry budget.
Its `stop_reason` distinguishes success, nonce exhaustion, cancellation,
deadline and retry exhaustion. Interrupted work is inconclusive. Cancellation
is checked between setup phases and every 64 walk steps; a single bounded
SHAKE/allocation operation is not preempted.
