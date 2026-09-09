# Chitin-PoW v2 — Specification (Option 2)

**Status:** research specification. Parameters are fixed and justified by
measurement below; nothing here is deployed. Every number is reproducible
from `pqmine/` and its tests.

## 1. What this is

A proof-of-work whose puzzle is a **bounded-weight syndrome decoding**
instance derived from the block header:

> Find a set `e` of at most `w_max` distinct column indices of a
> header-derived parity-check matrix `H` such that the XOR of those
> columns matches a header-derived syndrome `s` on its top `t` bits.

Syndrome decoding is the hardness assumption behind **Classic McEliece**
(NIST PQC) and has been unbroken since 1978 (Berlekamp–McEliece–van Tilborg;
NP-complete). The PoW is therefore **post-quantum in the strict sense**:
its hardness assumption is itself a PQ assumption, not merely a hash.

## 2. What it is NOT — read this first

| claim | status | evidence |
|---|---|---|
| post-quantum assumption | **yes** | syndrome decoding, §1 |
| satisfiable every block | **yes** | §5, P(sat)=1.000 |
| smoothly retargetable | **yes** | §5, 1-bit steps |
| cheap verification | **yes** | §4, `w_max` XORs |
| exact-weight accounting | **yes** | verifier rule (1) |
| ASIC-proof | **no** | any fixed function can be cast in silicon |
| **CPU-favoured** | **no** | §6 — GPU wins **4.5–16×**, measured |
| certified security level | **no** | ISD model, not external review |

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

Attacker model: work is scored against **Stern** (what honest miners run)
and **BJMM** (the attacker's edge, published as a factor). May–Ozerov is
deliberately excluded — its `2^{0.0966n}` only beats Stern at astronomical
`n` (a "galactic algorithm") and would understate work by orders of
magnitude. Model validated: McEliece `n=1024, k=524, w=50` → **65.4 bits**
vs literature 60–65.

At the recommended point (`w_max=32`, `t=164`): **attacker edge 2.67 bits**
(BJMM over Stern). That is the number to publish.

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
walk; at each step a hash bit decides whether to select the addressed
column (distinctness enforced via a used-set); the walk's accumulated
syndrome is checked against `s` on `t` bits. Weight lands on target with
binomial spread (mean 31.7 at `w=32`, max ≤ 49, so `w_max` should be set
≥ the observed max or the target lowered). Miners may replace this
entirely; only §4 is consensus.

## 8. Quantum accounting

At **equal classical security** `2^100`:

| PoW type | quantum bits | lost |
|---|---|---|
| hash | 50.0 | 50.0 |
| **SDP (this)** | **60.8** | **39.2** |
| SVP (lattice) | 90.8 | 9.2 |

SDP is better than hash and markedly worse than lattice for quantum
resistance specifically. The defensible reason to choose SDP is
**cryptanalytic stability** (1978 → now, incremental ISD only) versus
lattice reduction's active frontier, not quantum resistance. Grover on the
nonce search is further blunted by memory-hardness (the oracle must be a
reversible circuit over the scratchpad).

## 9. Open items

- **External review of the ISD calibration.** The Stern/BJMM model is
  validated on one literature point; the restricted-support structure of
  §7's walk (only walk-reachable supports are found) has not been reviewed
  as a distinct decoding problem. Esser–Santini (CRYPTO'24) showed regular
  SDP admits better attacks; structure added for search convenience can
  weaken the instance. This is the item that would sink the design.
- Verifier-side `w_max` vs walk's observed max-weight tail (§7).
- No network, no difficulty-adjustment controller, no chain integration.
- Only one paper on code-based PoW exists (ePrint 2026/1526, months old,
  self-described preliminary). This is a first implementation, not
  adoption of proven tech.

## 10. Reproduce

```
python -m pytest tests/ -q                       # 128 tests
python -c "from pqmine.calibration import *; ..."  # §5 numbers
python -c "from pqmine.fraction_test import fraction_report; fraction_report(2048)"  # §6
```

The fraction test allocates ~8 MiB VRAM and runs beside a resident LLM.
