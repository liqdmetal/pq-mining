# Chitin-PoW: memory-hard lattice proof-of-work

**Research prototype.** Toy parameters, pure-Python reference, no consensus
integration, no network. `pqmine/` + 26 tests.

## What it is, and the overclaim it avoids

A valid share is a short kernel vector of a header-derived Module-SIS
instance:

```
A in Z_q^{n x m}  from  SHAKE256(header, n, m, q)
find z != 0  with  A z = 0 mod q  and  ||z||_inf <= beta
```

Verification is one matrix-vector product plus a norm check -- microseconds,
deterministic, no replay of the search. That asymmetry is what makes it
usable as consensus work.

**The honest framing:** hash-based PoW is already fairly quantum-resistant.
Grover gives only a quadratic speedup on preimage search, handled by
doubling the search space -- unlike signatures, where Shor is catastrophic.
So "SHA-256 mining is quantum-broken" is **not** the motivation and is not
true. The real motivations are ASIC resistance via memory bandwidth, and a
work function whose hardness connects to lattice cryptanalysis rather than
hash heuristics. Quantum sieving improves the classical `2^{0.292n}` to
about `2^{0.265n}` -- a constant-factor change in the exponent, not a square
root. That is a modest claim, and it is the correct one.

## The CPU+GPU thesis was FALSE as first designed

The design was supposed to split into a GPU-shaped generation phase and a
CPU-shaped collision phase, neither sufficient alone. Measurement refuted it:

| n | q | m | difficulty | generate % | collide % |
|---|---|---|---|---|---|
| 2 | 97 | 32 | 6.6 | 96.4 | 3.6 |
| 2 | 257 | 32 | 8.0 | 96.7 | 3.3 |
| 3 | 97 | 48 | 9.9 | 97.8 | 2.2 |
| 2 | 1021 | 32 | 10.0 | 96.5 | 3.5 |
| 3 | 257 | 48 | 12.0 | 97.8 | 2.2 |
| 4 | 97 | 64 | 13.2 | 98.2 | 1.8 |

Generation is ~97% of the work at every setting, so a GPU-only miner wins
and there is no cooperative property.

**Why, structurally:** Phase A costs `n*m` modular multiplies per candidate
(256 at n=4, m=64); Phase B costs one hash-table probe. Two orders of
magnitude apart *by construction* -- no parameter choice fixes it, because
the imbalance is in the operation counts. Worse, candidates are generated
deterministically from an index, so Phase A is independent of Phase B's
output and an accelerator can pipeline it indefinitely -- precisely the
property the design was meant to forbid.

## The fix, and what it measured

`pqmine/balanced_miner.py` makes two changes:

1. **Amortise Phase A.** Flipping one nonzero of `z` changes the syndrome by
   a single column: `s' = s +/- A[:,j]`. That is `n` additions instead of
   `n*m` multiplies -- a factor-`m` cut, moving cost to the table.
   (`test_flip_matches_full_recomputation` proves the incremental update
   equals a full recomputation.)
2. **Force dependent access.** The next position to flip is chosen from the
   bucket contents Phase B returned, so Phase A' cannot run ahead of Phase
   B. The walk becomes a data-dependent random walk through the table.

Measured result:

| n | q | m | steps | generate % | collide % | found |
|---|---|---|---|---|---|---|
| 2 | 97 | 32 | 72 | 65.9 | 34.1 | yes |
| 2 | 257 | 32 | 213 | 74.3 | 25.7 | yes |
| 3 | 97 | 48 | 5729 | 73.8 | 26.2 | yes |
| 3 | 257 | 48 | 60000 | 73.2 | 26.8 | **no** |
| 4 | 97 | 64 | 60000 | 72.1 | 27.9 | **no** |

**97/3 -> ~73/27**, roughly an 8x shift into the memory-bound phase, shares
still verify. Change 1 rebalances the arithmetic; change 2 is what creates
the dependence -- without it, "balanced" would just mean two pipelines
rather than cooperation.

**The tradeoff, stated:** two settings now fail to find a share within the
step budget. A data-dependent walk covers the search space less efficiently
than independent sampling, so balance was bought with search efficiency.
Whether that trade is acceptable depends on tuning `table_bits` and the walk
rule, and it is not yet resolved.

## What an ASIC can still do

An ASIC can implement the syndrome arithmetic more efficiently than a GPU.
What it cannot cheaply escape is the table: the working set is chosen to
exceed practical on-die SRAM, so an ASIC must attach external DRAM/HBM and
then competes on the same memory bandwidth as commodity hardware -- a small
constant margin rather than the ~1000x of SHA-256 ASICs.

This is the Cuckoo Cycle / Equihash argument and it has a known limit:
memory-hardness **narrows** the ASIC gap, it does not eliminate it. As SRAM
budgets grow, `table_bits` must grow with them. **Anyone claiming ASIC-proof
is selling something.**

## Verification gates (adversarially tested)

| attack | rejected by |
|---|---|
| zero vector | nonzero gate |
| wrong dimension | dimension gate |
| share for another header | matrix is header-derived |
| perturbed vector | kernel gate |
| **scaled kernel vector** | norm gate (still in kernel, norm too large) |
| `beta >= q/2` parameters | refused at construction (PoW would be free) |

## Not done

- No CUDA/OpenCL port. Phase A' is written so a GPU port replaces one
  function body, but the claim that a GPU helps is **untested**.
- No multi-threaded CPU phase; the reference is single-threaded pure Python,
  so absolute timings say nothing about achievable hashrate.
- No difficulty retargeting, no block/chain integration, no pool protocol,
  no share grading.
- No cryptanalysis. The search here is a deliberately simple sieve, not a
  state-of-the-art one; a better sieve changes the work exponent and
  therefore the difficulty calibration.
