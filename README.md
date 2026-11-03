# Chitin-PoW — research-only code-based PoW experiment

**Status:** research repository. No deployment, network, wallet keys,
real balances, consensus integration, or certified security level.

## Lineage

This is the **current PoW research track** of the Chitin family. The older
confidential-transactions lab lives in
[`chitin-pqct`](https://github.com/liqdmetal/chitin-pqct). The chain-side
integration of this verifier (versioned proof language, `PQ_MAX_WEIGHT=32`
matching the golden vectors here) is maintained out-of-band; feed only what
these specs and vectors pin.

## Current design: option 2

The current branch experiments with a bounded-weight syndrome-decoding
verifier:

- `n_columns = 512`
- `syndrome_bits = 256`
- `w_max = 32`
- difficulty matches a retargetable prefix of the syndrome
- supports are sets of distinct column indices
- the instance is bound to both the block header and nonce
- verification XORs at most `w_max` header/nonce-derived columns

The verifier accepts a share only when the support is non-empty, distinct,
in-range, within `w_max`, and its XOR matches the target on the matched bits.
The nonce binding is consensus-critical: changing a nonce derives a new
instance and requires rechecking (at low difficulty accidental matches remain
possible). Mathematical support order is irrelevant; the fixed-profile wire
codec requires strictly ascending indices. See [wire corpus](docs/WIRE_CORPUS.md).

## Security status

The intended hardness basis is code-based syndrome decoding, but the composed
construction is **not certified**. The current bounded-language two-list MITM
calculation reports an **82.45-bit attack upper bound** at
`n=512, syndrome_bits=256, w_max=32, matched_bits=164`; the best cut is 215
with target weight 32. This is not a security claim; a better bounded-weight
or walk-structure attack may be faster.

The reference walk reaches a structured subset of supports. The restricted-
support question remains open even though the current toy audit found no gross
column bias, syndrome concentration, effective-weight collapse, or nonce-path
convergence.

Synthetic per-unit measurements and linear projections favour GPUs. They do
not establish a whole-chip ratio or an upper bound on hardware advantage.
The verifier does not require the reference walk. This project
does not claim ASIC-proof, FPGA-proof, CPU-only, or equal CPU/GPU mining.

## Search implementations

- `pqmine/search_share.py`: slow CPU reference search that uses the exact
  verifier derivation and returns a self-verified `Share`.
- `pqmine/decoupled_trace.py`: shared reference trace for the search and
  restricted-support audit.
- `pqmine/decoupled.py`: CUDA/Numba benchmark path using synthetic H; it is
  not yet a complete consensus miner.
- `pqmine/restriction_attack.py`: toy-scale structural screen, not a proof.
- `pqmine/bounded_work.py`: conservative symbolic two-list attack upper bound.
- `pqmine/partial_isd.py`: bounded CPU reference partial-syndrome solver; toy-validated, not a production solver.
- `pqmine/bdd_attack.py`: symbolic multi-partition attack family; not a complete BDD/ISD implementation.
- `pqmine/verifier.py`: consensus-rule reference implementation.
- `tests/consensus_vectors_v1.json`: versioned 512/256/32 derivation, math,
  canonical wire, and malformed-input corpus for Python/Rust parity.
- `pqmine/testnet.py`: loopback-only block/chain/node PoC; not DERO or production.

## Loopback testnet PoC

Run the isolated two-node toy chain:

```text
python tools/run_loopback_testnet.py --blocks 3 --max-nonces 128 --matched-bits 8
```

It binds only to `127.0.0.1`, uses toy parameters, mines with the bounded CPU
reference solver, and exercises real block validation plus HTTP broadcast. See
`docs/LOOPBACK_TESTNET.md`. It is not a DERO testnet and proves no PQ security.

## Verification

```text
cd <repo>
python -m pytest tests/ -q
```

Latest verified local result: see the current `pytest` run; historical counts are not security evidence.

See `docs/RESTRICTION_AUDIT.md`; the current option-2 specification is
`docs/POW_SPEC.md`. Questions best answered in `FAQ.md`.

## What remains before any deployment discussion

- independent coding-theory review of the bounded partial-syndrome model;
- analysis of walk-reachable support structure and better structured attacks;
- an actual optimized miner producing verifier-compatible supports;
- reproducible cross-hardware measurements with the resident Qwen workload
  protected;
- production difficulty calibration beyond the research-only timestamp-window controller;
- independent coding-theory review of the bounded-language cap and work model;
- executable toy cross-check of partition enumeration against the canonical verifier;
- machine-checkable two-list certificate covers all `511` splits × `32` accepted weights, pins the winning fields, and has a chunked CPU executor with explicit inconclusive status, peak-memory telemetry, accounting-tamper rejection, and real-instance probes;
- exact two-list bound regression is now checked against every column split;
- broader timestamp-manipulation analysis and cross-implementation review;
- cross-branch timestamp-grinding analysis beyond the current bounded testnet harness;
- serialization, pool/share protocol, and network design;
- independent implementation, parser/security review, and external
  cryptanalysis.

Passing tests demonstrate exercised behavior only. They do not establish
post-quantum security, ASIC resistance, or production readiness.

`quantum_work.py` reports generic oracle queries separately from the
Kachigar–Tillich asymptotic exponent extrapolation. Neither certifies a
finite-instance security floor. The n=2304 candidate remains unapproved;
`passes_numeric_screen` is bookkeeping, and `passes_modeled_floor` is false.
