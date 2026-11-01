# FAQ — Chitin-PoW (research profile)

**Is this production consensus?**
No. Research/testnet liveness only. It is a bounded-weight syndrome-decoding
experiment with a certified-looking spec but no certified security level,
no production difficulty calibration, and no deployment.

**Is it ASIC-proof / FPGA-proof / CPU-only?**
No claim on any of those. Measured hardware results are GPU-favoured
(4.5–16× GPU advantage at bench scale). The honest one-line pitch is
"a post-quantum PoW with a bounded, measured GPU advantage" — nothing more.

**Is it post-quantum secure?**
No 128-bit PQ claim. The quantum ISD bound `0.05869·n` binds: at `n=512`
that is ~30 bits. Clearing 128 bits needs `n ≥ 2181`; that profile exists
but is not approved. The naive Grover `t/2` count is _not_ a conservative
PQ bound — quantum ISD exploits code structure.

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
is bound into the instance, so replaying a support at another nonce fails.
Golden vectors in `tests/golden_vectors.json` pin the acceptance set;
chain-side constants must match them (`PQ_MAX_WEIGHT=32` is the canonical
example — the two drifted once and the retarget became meaningless).

**Is the walk publicly replayable?**
Yes. The walk is deterministic from the header; it feeds the column table.
This is documented — the design makes no secrecy claim about the walk.

**What would make this a deployment candidate?**
Independent coding-theory review, a verifier-compatible optimized miner,
reproducible cross-hardware measurements, production difficulty
calibration, network/serialization design, and external cryptanalysis.
Passing tests demonstrate exercised behavior only.
