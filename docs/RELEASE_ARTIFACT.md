# Chitin-PoW v2 — review and miner-preparation artifact

**Status: research only, no approved production/PQ security level.** This
artifact distinguishes verifier facts, symbolic models and historical
benchmarks. Passing tests is evidence only of the exercised behavior.

## Fixed interoperability profile

```
n_columns = 512
syndrome_bits = 256
w_max = 32
matched_bits = t  # syntax accepts 1..256; calibration/satisfiability is separate
```

The chain must explicitly pin this parameter tuple and proof language.
A historical chain/reference w_max drift motivated shared fixtures; do not
assume a current chain defect without checking the exact integration commit.
The n=2304 research candidate uses different parameters and is unapproved.

## Acceptance and serialization

A mathematical support must be nonempty, distinct, in range, at most 32
columns, and XOR to the target's matched-bit prefix. Its order is irrelevant.
H and s derive from opaque header bytes and uint64 nonce using the unchanged
CHITIN-POW-H-V2 and CHITIN-POW-S-V2 SHAKE domains. Verification derives only
selected columns and the target: at most 33 SHAKE operations and 32 XORs.
The search walk is not verified and can be replaced or bypassed by miners.

The wire witness adds canonical order: CHITINPQ || u8(weight) || ascending
u16le indices. Exact length, weight, range and strict ordering are checked
before hashing. See [wire specification and schema](WIRE_CORPUS.md).

`tests/consensus_vectors_v1.json` is the versioned, fixed-profile Python/Rust
corpus. It separates derivation, mathematical validity and canonical-wire
validity; permutations can be math-valid but wire-invalid. Existing
`tests/golden_vectors.json` remains an unchanged broad research fixture.

## Difficulty and attack accounting

Expected bounded-support solutions are modeled as
2^(log2(sum(C(512,w), w=1..32)) - t). The expected-solutions cap t=165
corresponds to roughly 2^4 expected solutions; it does not guarantee every
instance is solvable or calibrate real hashrate.

The included classical two-list MITM model gives about 82.4461 work bits
at t=164 (cut 215/297, weights 17/15). This is symbolic attack accounting,
not a measured full-size attack, security lower bound or certificate.
Generic Grover accounting gives t/2=82 oracle-query bits before oracle
implementation, qubit, memory and error-correction costs.

[Kachigar–Tillich](https://arxiv.org/abs/1703.00263v2) give a worst-case
asymptotic exponent 0.05869n in their decoding regime. Its extrapolation
at n=512 is 30.049 numerically, not a concrete finite-instance attack cost
or proven security floor here. It is reported separately from Grover queries.

The historical `candidate_128_profile()` name refers to n=2304/1152,
w_max=60, t=392. It is research-only-unapproved. Crossing 128 in a symbolic
calculation does not establish 128-bit security. Profile reports explicitly
set `security_floor_established=false`; no automatic approval gate exists.

## Hardware evidence

The historical 5900X/RTX3090 fraction test measures one CPU core against one
occupied GPU SM on synthetic kernels. Scaling those rates by 12 cores and
82 SMs yields projections of 16.3× at K=1024, 8.4× at K=2048 and 7.9× at
K=8192. These are not measured whole-chip valid-share rates or upper bounds
on hardware advantage. Dividing two projected throughputs does not bound
the true ratio. Alternative solvers need not perform the benchmarked walk.

The project makes no CPU-only, ASIC-resistance or bounded GPU-advantage claim.
A deployable hardware comparison needs reproducible complete miners emitting
canonical verified shares, fixed difficulty and parameters, device/power/
software metadata, and independent replication. Do not disrupt a resident
workload to obtain an unapproved benchmark.

## Verification and remaining gates

```
python -m pytest tests/ -q
python tools/gen_consensus_vectors.py
python tools/show_profile.py
```

Regeneration writes only this repo; intentional consensus changes require
review of both implementations and pinned fixture hashes. The reference
search has nonce, memory/step, deadline/cancellation and collision budgets;
a capped result is inconclusive.

Deployment remains gated on finite-instance coding-theory/quantum analysis,
search-structure attacks, optimized verifier-compatible miners, profile-aware
chain work/retargeting, authenticated transport and parser review, and
independent consensus and implementation audits. The existing loopback
network and research controller do not satisfy those production gates.
