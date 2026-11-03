# Chitin-PoW v2 — Specification (Option 2)

**Status:** research specification. The fixed verifier profile is an experimental
interoperability target, not a security certification. Historical measurements
and symbolic calculations below have different scopes; neither approves deployment.

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
|| **CPU-favoured** | **not established** | §6 — synthetic per-unit measurements and projections |
|| certified security level | **no** | bounded MITM is only an attack upper bound |

A code-based PoW research experiment with unapproved security parameters.
Synthetic per-unit benchmarks favour GPUs; no whole-chip hardware-advantage
bound or production valid-share throughput is established.

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
4. `H` and `s` are derived from the bound `(header, nonce)` instance (`SHAKE-256`), never supplied by the miner.

Cost: at most 32 column SHAKE derivations, one target derivation, and 32
256-bit XORs. The walk is never re-executed. Mathematical supports are
order-independent; canonical CHITINPQ wire witnesses must be strictly ascending.
See [wire corpus](WIRE_CORPUS.md) for the exact codec and fixed-profile parity cases.

## 5. Difficulty — `pqmine/calibration.py`

Hash PoW gets smooth retargeting for free (leading zeros). SDP does not:
difficulty depends on `(n, k, w)` through ISD complexity, instances can be
**unsatisfiable**, and the first three parameter picks in this project
were exactly that (one at `2^{−6.3}` expected solutions).

Solution: retarget on **matched syndrome bits** `t ∈ [1, 256]`.

- Expected solutions of weight ≤ `w_max`: `2^{log₂ Σ_{w≤32} C(512,w) − t}`
- Satisfiability cap: `t_cap = max t with ≥ 2^4 expected solutions`
  → **`t_cap = 165`** at `w_max=32`, `log₂(E) = 4.0086` (bounded-language cap)
- Each `t` step is exactly 1 bit — same granularity as a hash target

The old full-syndrome Stern/BJMM comparison is **not applicable** after
retargeting to a partial syndrome and accepting all weights `1..w_max`.
It is retired, not silently reused.

The current conservative published attack is the bounded-language two-list
meet-in-the-middle calculation in `pqmine/bounded_work.py`:

- `n=512`, `syndrome_bits=256`, `w_max=32`, `t=164`;
- satisfiability: `log2(E[accepted supports]) = 5.0086`;
- simple MITM attack: **82.4461 bits of work**, using a `215/297` column split
  and support weights `17/15`;
- memory for the larger list: **2^82.4461 entries** before implementation
  compression and representation costs;
- walk-restricted nonce scan: **173.434 conditional bits** in a 256-nonce
  sample, with 148/256 supports under `w_max`; this is weaker than the generic
  route and assumes a uniform matched-prefix target.

The generic MITM number remains an **attack upper bound on security**, not a
security claim. The repository now emits a machine-checkable accounting
certificate for this construction: all `511` column splits and all `32`
accepted support weights are swept, and the winning fields are independently
recomputed. It also contains a chunked bounded CPU executor for the certificate's
winning split; toy runs are checked through the canonical verifier, while the
large-parameter executor is deliberately hard-capped and therefore
inconclusive when capped. It reports peak indexed entries, chunks, and left/right
candidate work. Real `derive_instance()` columns and targets are used for both the recommended and candidate profiles; a capped probe is evidence only of bounded execution, not of a solved large instance. The executor independently re-derives list volumes,

work, and pair-excess accounting before it searches, so tampered certificate
fields are rejected. The certificate records the winning target weight `32`,
split weights `17/15`, list exponents `82.4448/82.4461`, expected pair excess
`0.8909` bits, and memory exponent `82.4461`. That certificate validates
arithmetic and coverage only; it does
not certify cryptanalytic optimality, practical large-parameter execution, or a
security level. The walk estimate is only a conditional workload model because
the verifier accepts arbitrary bounded supports. A better bounded-weight,
partial-syndrome, quantum, or structured-support attack may be faster. No
certified security level is assigned until a coding theorist reviews the model.

### Quantum accounting boundary

`pqmine/quantum_work.py` reports generic amplitude-amplification oracle-query
counts separately from asymptotic ISD extrapolations. At t=164 the generic
query count is 82 bits; this excludes reversible SHAKE, circuit depth,
qubits, QRAM, fault tolerance and constants, and is not a security level.

[Kachigar–Tillich](https://arxiv.org/abs/1703.00263v2) report a worst-case
asymptotic exponent 0.05869n in their decoding regime. Substituting n=512
produces 30.049 numerically, but does not establish a concrete 30-bit attack
or a finite-instance security lower bound for our weight/prefix parameters.
The applicability and cost of concrete quantum ISD remain open.

### Candidate n=2304 profile — unapproved

`candidate_128_profile()` is a historical API name for an investigation
candidate: n=2304, syndrome_bits=1152, w_max=60, matched_bits=392. The name
expresses a target, not achieved security. Increasing n until 0.05869n
exceeds 128 cannot certify a 128-bit floor; attack upper bounds never prove
security lower bounds. `passes_numeric_screen` only compares symbolic
numbers; `passes_modeled_floor` and `security_floor_established` are false.

No candidate parameters are promoted into the fixed 512/256/32 corpus.
Concrete attacks, quantum resource analysis, implementation review, and
independent cryptanalysis are required. Print the research report with
`python tools/show_profile.py`; do not use it as a live mining configuration.

## 6. Hardware profile — measured units and projected chips

All numbers from the 5900X (12C/24T) + RTX 3090, via the **fraction test**
(`pqmine/fraction_test.py`): 1 CPU core vs 1 fully-occupied SM, scaled
linearly to each chip. The ratio is a projection, not a measured whole-chip
ratio or upper bound. Even two absolute throughput upper bounds do not bound
their ratio. Contention, clocks, occupancy and power differ between devices.
These synthetic kernels are not complete canonical valid-share miners.

| K (walk length) | 1 core vs 1 SM | chip projection (82 SM / 12 core) |
|---|---|---|
| 1024 | SM 2.39× | **16.3× GPU** |
| 2048 | SM 1.23× | 8.4× |
| 8192 | SM 1.15× | **7.9× GPU** |

These samples narrow the per-unit gap with longer chains. Extrapolated
chip rates favour the GPU for the listed configurations; other hardware,
software or solver choices have not been bounded.

At K ≥ 2048 these historical per-unit measurements project roughly 8×.
This is not a guaranteed hardware advantage or evidence that the walk is
required: arbitrary search algorithms are accepted by the verifier.

### CPU-favour was not established

The earlier pointer-chasing and per-nonce scratchpad experiments did not
establish CPU-favour for this design. Serial steps within a nonce do not
prevent GPUs from processing independent nonces concurrently. Cache-fitting
microbenchmarks do not establish the economics of complete miners. No
claim is made that a particular VM is necessary or sufficient for CPU-favour,
nor that the optional walk enforces a hardware cost on every accepted share.

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
- Only an isolated loopback harness and research difficulty controller; no
  production network or calibrated consensus deployment is established here.
- The research construction needs current independent literature and attack
  review; no novelty or completeness claim is made by this artifact.

## 10. Reproduce

```
python -m pytest tests/ -q                       # full suite (adversarial verifier suite included)
python -c "from pqmine.calibration import *; ..."  # §5 numbers
python -c "from pqmine.fraction_test import fraction_report; fraction_report(2048)"  # §6
```

The fraction test allocates ~8 MiB VRAM and runs beside a resident LLM.
