# FAQ — Chitin-PQ proof of work (research profile)

**Is this production consensus?**
No. Research/testnet liveness only. It is a bounded-weight syndrome-decoding
experiment with an explicit specification but no certified security level,
no production difficulty calibration, and no deployment.

**Is Chitin-PQ proof of work integrated into Noumenon?**
Yes, as a research implementation: the versioned verifier policy and
`PQ_MAX_WEIGHT=32` are tied to the golden vectors in this repository. That
does not establish production readiness or post-quantum security.

**Is it ASIC-proof / FPGA-proof / CPU-only?**
No claim on any of those. Synthetic per-unit results favour GPUs. Scaling
both devices linearly is a projection, not a measured whole-chip ratio or
a bound on the advantage of arbitrary miners. The verifier permits search
algorithms that bypass the reference walk entirely.

**Is it post-quantum secure?**
No certified PQ security level. The `0.05869·n` expression is an asymptotic
extrapolation from a particular decoding analysis, not a concrete attack
cost or security floor for this finite partial-syndrome problem. Increasing
n until that number exceeds 128 does not establish 128-bit security. The
n=2304 candidate is unapproved. Generic Grover query counts also omit the
oracle implementation and do not exclude better attacks.

**What is the verified attack bound?**
Bounded-language two-list MITM: **82.45-bit attack upper bound** at
`n=512, syndrome_bits=256, w_max=32, matched_bits=164`; best cut 215 with
target weight 32. That is an upper bound from the enclosed symbolic
calculator (`pqmine/bounded_work.py`), not a guarantee — a better
bounded-weight or walk-structure attack may be faster.

**Does it mine real DERO or anything live?**
No. `pqmine/testnet.py` is a loopback-only toy chain bound to `127.0.0.1`.
There is no network, no wallet, no real coin.

**What keeps the verifier honest?**
The verifier accepts a share only when the support is non-empty, distinct,
in-range, within `w_max`, and its XOR matches the target prefix; the nonce
is bound into the instance, so another nonce requires a fresh check.
Golden vectors in `tests/golden_vectors.json` pin the acceptance set;
chain-side constants must match them (`PQ_MAX_WEIGHT=32` is the canonical
example — the two drifted once and the retarget became meaningless).
The fixed-profile corpus `tests/consensus_vectors_v1.json` additionally
separates order-independent mathematical witnesses from ascending canonical
CHITINPQ wire witnesses. Unsorted wire witnesses are rejected, not normalized.

**Is the walk publicly replayable?**
Yes. The walk is deterministic from the header; it feeds the column table.
This is documented — the design makes no secrecy claim about the walk.

**What would make this a deployment candidate?**
Independent coding-theory review, a verifier-compatible optimized miner,
reproducible cross-hardware measurements, production difficulty
calibration, network/serialization design, and external cryptanalysis.
Passing tests demonstrate exercised behavior only.
