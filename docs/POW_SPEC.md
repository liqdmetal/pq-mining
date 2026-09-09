# Chitin-PoW v2 — Specification (Option 2)

**Status:** research specification. Parameters are fixed and justified by
measurement below; nothing here is deployed. Every number is reproducible
from `pqmine/` and its tests.

## 1. What this is

A proof-of-work whose puzzle is a **bounded-weight syndrome decoding**
instance derived from the block header **and the nonce**:

> Find a set `e` of at most `w_max` distinct column indices of a
> **(header, nonce)-derived** parity-check matrix `H` such that the XOR of
> those columns matches a **(header, nonce)-derived** syndrome `s` on its top
> `t` bits.

Syndrome decoding is the hardness assumption behind **Classic McEliece**
(NIST PQC) and has been studied since 1978 (Berlekamp–McEliece–van Tilborg;
NP-complete). This research construction uses that code-based assumption as
its intended PQ hardness basis; it does **not** certify that the composed,
walk-restricted construction inherits generic SDP security.

## 1b. What is verified, what is not

The **verifier** is the only consensus-critical piece. It independently
recomputes `H` and `s` from the bound instance and checks only the
distinct, bounded-support XOR relation. The **search** (the K-step walk) is
not re-executed; a share is a witness, not a transcript. A miner may use any
search implementation; only §4 is consensus.

## 2. What it is NOT — read this first

| claim | status | evidence |
|---|---|---|
| code-based PQ assumption used | **yes, uncertified** | syndrome decoding, §1 |
|| satisfiable at recommended target | **modeled yes** | §5, expected-solutions model; not every block guaranteed |
|| smoothly retargetable | **yes mechanically** | §5, 1-bit steps; security calibration is open |
|| cheap verification | **yes** | §4, `w_max` XORs |
|| bounded-weight accounting | **yes** | verifier accepts `1..w_max` |
|| ASIC-proof | **no** | any fixed function can be cast in silicon |
|| **CPU-favoured** | **no** | §6 — GPU wins **4.5–16×**, measured |
|| certified security level | **no** | bounded MITM is only an attack upper bound |

The honest one-line pitch: **a post-quantum PoW with a bounded, measured
GPU advantage of ~5–8× at the recommended operating point.** That is in
Ethash's territory and tamer than SHA-256's ASIC gap by orders of
magnitude. It is not RandomX.

## 3. Consensus parameters

```
n_columns     = 512      # columns of H
syndrome_bits = 256      # n - k, rate 1/2
w_max         = 32       # bounded weight
matched_bits  = t        # DIFFICULTY, retargetable in [1, 256]
```

Only the verifier (§4) is consensus. The search procedure (§7) is not: a
miner may use any algorithm, and a CPU/GPU miner disagreement cannot fork
the chain because acceptance depends solely on the XOR check.

### Why rate ½
`msis`-style balance: for a code of rate `R`, ISD work grows with the
distance from the Gilbert–Varshamov bound, and rate ½ maximises the GV
radius per column. Also the natural point where `k = n − k`, so the
syndrome is exactly half the code length.

### Why `w_max = 32`, not larger
This is the finding that reshaped the design. The earlier composed miner
walked `K = 400–460` steps and accumulated a column per step, giving
weight `w ≈ K`. **That is a DENSE error**: `C(512,460) = C(512,52)` by
symmetry, so weight-460 is the complement of weight-52, and at rate ½ the
expected number of solutions is `2^{238.6 − 256} = 2^{−17}` — essentially
every header **unsolvable**. SDP hardness lives at *small* `w` below the
GV bound. `w_max = 32` is inside the satisfiable, hard regime (§5).

Chain length `K` and error weight `w` are therefore **decoupled**
(`pqmine/decoupled.py`): the walk is long (for the memory-hard search) but
selects only `w ≪ K` columns via a hash-bit gate.

## 4. Verifier (consensus) — `pqmine/verifier.py`

Accept a share `(header, nonce, support)` iff:

1. `support` is a **set** of distinct indices in `[0, n_columns)`.
   Over F₂ a repeated column cancels (`x ⊕ x = 0`), so without this a
   miner pads weight for free. Rejected **before any XOR**.
2. `|support| ≤ w_max`. Bounded, not exact: a heavier error is a weaker
   claim, so the verifier accepts at-or-under and difficulty assumes the cap.
3. `⊕_{j∈support} H[:,j]` equals `s` on the top `t` bits.
4. `H` and `s` are derived from the header (`SHAKE-256`), never supplied.

Cost: ≤ 32 XORs of 256-bit words. **The K-step walk is never re-executed**
— it is the search, not the proof. 11 adversarial tests including the
duplicate-column forgery.

## 5. Difficulty — `pqmine/calibration.py`

Hash PoW gets smooth retargeting for free (leading zeros). SDP does not:
difficulty depends on `(n, k, w)` through ISD complexity, instances can be
**unsatisfiable**, and the first three parameter picks in this project
were exactly that (one at `2^{−6.3}` expected solutions).

Solution: retarget on **matched syndrome bits** `t ∈ [1, 256]`.

- Expected solutions of weight ≤ `w_max`: `2^{log₂ Σ_{w≤32} C(512,w) − t}`
- Satisfiability cap: `t_cap = max t with ≥ 2^4 expected solutions`
  → **`t_cap = 164`** at `w_max=32`, `P(sat) = 1.000`
- Each `t` step is exactly 1 bit — same granularity as a hash target

The old full-syndrome Stern/BJMM comparison is **not applicable** after
retargeting to a partial syndrome and accepting all weights `1..w_max`.
It is retired, not silently reused.

The current conservative published attack is the bounded-language two-list
meet-in-the-middle calculation in `pqmine/bounded_work.py`:

- `n=512`, `syndrome_bits=256`, `w_max=32`, `t=164`;
- satisfiability: `log2(E[accepted supports]) = 5.0086`;
- simple MITM attack: **83.06 bits of work**, using two lists of weight 16;
- memory for the larger list: **2^83.06 entries** before implementation
  compression and representation costs.

This is an **attack upper bound on security**, not a security claim. A better
bounded-weight or structured-support attack may be faster. No certified
security level is assigned until a coding theorist reviews the model.

## 6. Hardware profile — measured, not asserted

All numbers from the 5900X (12C/24T) + RTX 3090, via the **fraction test**
(`pqmine/fraction_test.py`): 1 CPU core vs 1 fully-occupied SM, scaled
linearly to each chip. Linear scaling ignores shared-memory contention on
both sides, so it is an **upper bound on the GPU** — conservative in the
CPU's favour.

| K (walk length) | 1 core vs 1 SM | chip bound (82 SM / 12 core) |
|---|---|---|
| 1024 | SM 2.39× | **16.3× GPU** |
| 2048 | SM 1.23× | 8.4× |
| 8192 | SM 1.15× | **7.9× GPU** |

Long chains narrow the per-unit gap toward parity but the SM never falls
below the core, and 82 vs 12 units keeps the chip GPU-favoured.

**Recommended `K ≥ 2048`** → GPU advantage ~8×.

### Why not CPU-favoured (and what would be)
Two things were tried and measured. **Latency-bound pointer chasing**
(PoSME): a single GPU thread is 9.9× *slower* than a core — confirmed —
but mining runs one chain per nonce, nonces are independent, and 16k GPU
threads amortise the latency away. PoSME's headline is correct and
irrelevant to PoW. **Per-nonce mutable scratchpad**: the CPU wins 2–3×
**only when the working set fits L1**; past L2 both spill to DRAM and GPU
bandwidth wins. Mutation + L1-fit is the CPU lever (this is *why* RandomX
sizes its hot scratchpad to L1), and it gives ~1.5× per unit here — far
short of the ~6.8× needed to overcome the unit-count deficit. Reaching
CPU-only requires RandomX's random-program VM, which abandons the PQ
assumption in the PoW. Option 2 keeps the assumption and accepts the GPU.

## 7. Search procedure (non-consensus) — `pqmine/decoupled.py`

Per nonce: an L1-sized mutable scratchpad drives a K-step data-dependent
walk; at each step a hash bit decides whether to select a column (distinctness
enforced via a used-set); the walk's accumulated syndrome is checked against
`s` on `t` bits.

The CUDA/Numba `decoupled.py` path is a **benchmark component**, not yet a
complete miner: it uses synthetic H and returns a syndrome/weight. The slow
CPU `search_share.py` reference path now derives H and s through the
consensus verifier, retains the selected support, constructs a real
`Share(header, nonce, support)`, and self-verifies it before reporting success.
It is test-scale only; it is not a production search path.

The selector has binomial spread rather than exact weight. A full run measured
mean 31.7 and maximum 49 for a target of 32; therefore the shipped verifier
must retain the explicit `w_max` cap, and the expected-weight statement is
not a proof that all walk outputs are admissible. Miners may replace the
search entirely; only §4 is consensus.

## 8. Quantum accounting boundary

No quantum-security number is certified for this construction. The old
asymptotic SDP/hash/SVP comparison is not a validated security baseline for
the current partial-syndrome, bounded-weight, walk-restricted verifier and
has been removed from the specification.

Known qualitative facts only:

- generic quantum speedups and nonce-search effects require a separate model;
- the memory-hard walk's reversible-oracle cost has not been analyzed;
- the walk-restricted support language may admit attacks that invalidate any
  generic SDP comparison.

## 9. Open items

- **External review of the restricted-support decoding problem.** The old
  full-syndrome Stern/BJMM comparison is retired and does not validate the
  current partial-syndrome verifier. The walk-reachable support structure has
  not been reviewed as a distinct decoding problem. Esser–Santini (CRYPTO'24)
  showed regular SDP admits better attacks; structure added for search
  convenience can weaken the instance. This is the item that could sink the
  design.
- Verifier-side `w_max` versus the walk's observed max-weight tail (§7).
- No network, no difficulty-adjustment controller, no chain integration.
- Only one paper on code-based PoW exists (ePrint 2026/1526, preliminary).
  This is a first implementation, not adoption of proven technology.

## 10. Reproduce

```
python -m pytest tests/ -q                       # currently 148 tests + 8 subtests
python -c "from pqmine.calibration import *; ..."  # §5 numbers
python -c "from pqmine.fraction_test import fraction_report; fraction_report(2048)"  # §6
```

The fraction test allocates ~8 MiB VRAM and runs beside a resident LLM.
