# Chitin-PoW v2 — Release Artifact (for review & miner preparation)

**Status:** pre-launch research artifact. Published BEFORE Noumenon chain
mainnet so external reviewers and miner implementers can act on the real
algorithm — this is **not** a certification that the algorithm is PQ-secure
or consensus-ready. The consensus-critical rules below are pinned by
`tests/golden_vectors.json` (30 cross-language vectors; 240 tests + 20
subtests pass on the reference verifier as of 2026-10-18).

## 0. Must-fix before this artifact is authoritative

The chain integration (`noumenon-core/src/chitin.rs`) currently pins
`PQ_MAX_WEIGHT = 60`, while the reference verifier, all 30 golden vectors,
and this spec pin `w_max = 32`. The chain value is a leftover from the
unapproved `candidate_128_profile` (which uses `w_max=60` **only** at
`n=2304`). At `n=512` a `w_max=60` language has volume `2^284.7`, so the
calibrated difficulty `t=164` would admit ~2^120 expected supports — the
retarget becomes meaningless. **Consensus value must be `w_max = 32` at
`n=512`.** The chain constant is a blocker, not a style nit.

## 1. One-line honest pitch

A post-quantum proof-of-work based on bounded-weight syndrome decoding, with
a bounded, **measured GPU advantage of ~5–8×** at the recommended operating
point. It is in Ethash's territory, tamer than SHA-256's ASIC gap by orders
of magnitude, and **not** RandomX (not CPU-favoured).

## 2. What this is

A PoW whose puzzle is a bounded-weight syndrome-decoding instance derived
from the block header **and** the nonce:

> Find a set `e` of at most `w_max` distinct column indices of a
> (header, nonce)-derived parity-check matrix `H` such that the XOR of those
> columns matches a (header, nonce)-derived syndrome `s` on its top `t` bits.

Syndrome decoding is the hardness basis of Classic McEliece (NIST PQC) and
is NP-complete (Berlekamp–McEliece–van Tilborg, 1978). This research
construction uses that assumption as its intended PQ basis; it does **not**
certify that the composed, walk-restricted construction inherits generic
SDP security.

## 3. Consensus parameters (pinned)

```
n_columns     = 512      # columns of H
syndrome_bits = 256      # n - k, rate 1/2
w_max         = 32       # bounded weight  (MUST match chain PQ_MAX_WEIGHT)
matched_bits  = t        # DIFFICULTY, retargetable in [1, 256]
```

Only the verifier (§4) is consensus. The search procedure is not: acceptance
depends solely on the XOR check, so any miner (CPU/GPU/ASIC) cannot fork the
chain by using a different search.

### Why rate ½
`msis`-style balance: rate ½ maximises the Gilbert–Varshamov radius per
column and is the natural point where the syndrome is half the code length.

### Why `w_max = 32`, not larger
Dense errors are unsolvable at rate ½: a walk accumulating a column per step
gives weight `w ≈ K`; `C(512,460) = C(512,52)`, and at rate ½ the expected
solution count is `2^(238.6 − 256) = 2^−17` — essentially every header
unsolvable. SDP hardness lives at *small* `w` below the GV bound; `w_max=32`
sits in the satisfiable, hard regime. Chain length `K` and error weight `w`
are therefore decoupled: the walk is long (memory-hard search) but selects
only `w ≪ K` columns via a hash-bit gate.

## 4. Verifier (consensus)

Accept a share `(header, nonce, support)` iff:

1. `support` is a **set** of distinct indices in `[0, n_columns)` — a
   repeated column cancels over F₂ (`x ⊕ x = 0`), so without this a miner
   pads weight for free. Rejected before any XOR.
2. `|support| ≤ w_max` (bounded, not exact).
3. `⊕_{j∈support} H[:,j]` equals `s` on the top `t` bits.
4. `H` and `s` are derived from the bound `(header, nonce)` instance via
   SHAKE-256 — never supplied by the miner.

Cost: ≤ 32 XORs of 256-bit words. The K-step walk is never re-executed — it
is the search, not the proof.

## 5. Difficulty / retargeting

Hash PoW retargets for free (leading zeros). SDP does not: hardness depends
on `(n,k,w)` through ISD complexity, and instances can be unsatisfiable.
Retarget on **matched syndrome bits** `t ∈ [1, 256]`:

- Expected solutions of weight ≤ `w_max`: `2^(log2 Σ_{w≤32} C(512,w) − t)`
- Satisfiability cap: `t_cap = 165` at `w_max=32` (log2(E) = 4.0086)
- Each `t` step is exactly 1 bit — same granularity as a hash target

## 6. Attack accounting (upper bounds, not security claims)

At `n=512, w_max=32, t=164`:

| attack | bits |
|---|---:|
| classical two-list MITM (cut 215/297, weights 17/15) | **82.4461** |
| naive Grover oracle-query model (t/2) | 82.000 |
| quantum ISD bound (0.05869·n) | **30.049** ← binding |

The quantum ISD bound is the binding constraint: **this profile does not
support a 128-bit PQ claim.** The `n=2304` candidate (~135 quantum bits /
~197 classical MITM) exists in `pqmine/profiles.py` but is **not approved**:
gate-level resources, concrete memory/time, implementation security, and
independent coding-theory review are all missing.

## 7. Hardware profile (measured, not asserted)

5900X (12C/24T) + RTX 3090 via the fraction test (1 core vs 1 fully-occupied
SM, scaled linearly; linear scaling is an upper bound on the GPU —
conservative in the CPU's favour):

| K (walk length) | 1 core vs 1 SM | chip bound (82 SM / 12 core) |
|---|---:|---:|
| 1024 | SM 2.39× | 16.3× GPU |
| 2048 | SM 1.23× | 8.4× GPU |
| 8192 | SM 1.15× | 7.9× GPU |

**Recommended `K ≥ 2048` → GPU advantage ~8×.** Long chains narrow the
per-unit gap toward parity but never below it.

Why not CPU-favoured (both tried and measured): (a) latency-bound pointer
chasing — a single GPU thread is 9.9× slower than a core, but mining runs one
chain per nonce and 16k GPU threads amortise the latency away; (b) per-nonce
mutable scratchpad — CPU wins 2–3× **only** when the working set fits L1,
and the ~1.5× per-unit win is far short of the ~6.8× needed to overcome the
unit-count deficit. CPU-only would require RandomX's random-program VM, which
abandons the PQ assumption. Option 2 keeps the assumption and accepts the
GPU.

## 8. Reproduce

```text
cd C:/Users/Administrator/.hermes/pq-mining
python -m pytest tests/ -q                       # 240 passed, 20 subtests
python -c "from pqmine.calibration import *; ..."  # §5 numbers
python -c "from pqmine.fraction_test import fraction_report; fraction_report(2048)"  # §7
```

Golden vectors: `tests/golden_vectors.json` — pinned to `pqmine.verifier`
by `tests/test_golden_vectors.py`; consumed by the Rust integration tests.
Regenerate both with `PYTHONPATH=. python tools/gen_golden_vectors.py` only
if the consensus rule deliberately changes.

## 9. Open items that gate mainnet consensus

- **External review of the restricted-support decoding problem.** The old
  full-syndrome Stern/BJMM comparison is retired and does not validate the
  partial-syndrome verifier. The walk-reachable support structure has not
  been reviewed as a distinct decoding problem. Esser–Santini (CRYPTO'24)
  showed regular SDP admits better attacks; structure added for search
  convenience can weaken the instance. **This is the item that could sink
  the design.**
- Verifier-side `w_max` vs the walk's observed max-weight tail (§7).
- Fix the chain `PQ_MAX_WEIGHT = 60` → `32` divergence (§0).
- No network, no difficulty-adjustment controller, no chain integration yet.
- Only one paper on code-based PoW exists (ePrint 2026/1526, preliminary);
  this is a first implementation, not adoption of proven technology.

## 10. What this artifact is for

1. **Review:** give coding theorists the exact rules (§3–§6) and the pinned
   vectors before mainnet so the "could sink the design" item is resolved
   with the real algorithm, not a paraphrase of it.
2. **Miner preparation:** give implementers the exact verifier + parameters
   and the measured hardware profile so genesis difficulty can be calibrated
   against real hashrate. A new PoW cannot calibrate genesis difficulty if
   nobody has had the algorithm.
3. **Honesty:** the one-line pitch and §6/§7 must travel with the artifact.
   No 128-bit PQ claim, no CPU-favoured claim, no ASIC-proof claim.
