"""Measured phase balance -- and the design change the measurement forced.

WHAT THE FIRST MEASUREMENT SHOWED
---------------------------------
`pqmine/miner.py`'s docstring claims a cooperative CPU+GPU property: a
GPU-shaped generation phase and a CPU-shaped collision phase, neither
sufficient alone. The first benchmark refuted that:

    n  q     m    diff  gen%   collide%
    2     97   32    6.6  96.4      3.6
    2    257   32    8.0  96.7      3.3
    3     97   48    9.9  97.8      2.2
    2   1021   32   10.0  96.5      3.5
    3    257   48   12.0  97.8      2.2
    4     97   64   13.2  98.2      1.8

Generation is ~97% of the work at every parameter setting. A GPU-only miner
that implements Phase A and does Phase B badly would still win, so the
"cooperative" claim was FALSE as designed.

WHY, STRUCTURALLY
-----------------
Cost per candidate:
  * Phase A: one n x m matrix-vector product = n*m modular multiplies.
    At n=4, m=64 that is 256 multiplies.
  * Phase B: one dictionary insert/probe = one hash plus a few compares.

So Phase A is two orders of magnitude more arithmetic per candidate BY
CONSTRUCTION. Sparsity does not save Phase B; it saves Phase A a little.
No parameter choice inside this design rebalances it, because the imbalance
is in the operation counts, not the constants.

Worse, the syndromes are computed on candidates that are generated
deterministically from an index. That makes Phase A embarrassingly parallel
AND independent of Phase B's output, so an accelerator can pipeline it
indefinitely -- exactly the property the design was supposed to forbid.

THE FIX: MAKE THE TABLE THE BOTTLENECK
--------------------------------------
Two changes, implemented in `balanced_miner.py`:

1. AMORTISE PHASE A. Generate candidates by INCREMENTAL UPDATE instead of
   from scratch. Flipping one nonzero position of z changes the syndrome by
   a single column of A: s' = s +/- A[:,j]. That is n additions instead of
   n*m multiplies, cutting Phase A by a factor of m and moving the cost to
   the table.

2. FORCE DEPENDENT ACCESS. Choose the NEXT position to flip from the bucket
   contents found in Phase B. Then Phase A' cannot run ahead of Phase B --
   the walk through candidate space is a data-dependent random walk through
   the table, which is the pointer-chasing pattern DRAM latency punishes and
   SRAM-rich ASICs cannot avoid without holding the whole table on die.

Change 1 alone rebalances the arithmetic. Change 2 is what actually creates
the dependence; without it, balance just means "two pipelines" rather than
"cooperation".

HONEST LIMIT
------------
Even with both changes this is a latency-bound random-walk PoW, which is the
Cuckoo-Cycle family. Its ASIC resistance rests on the working set exceeding
practical on-die SRAM, and that is a moving target: as SRAM budgets grow,
table_bits must grow with them. Anyone claiming a permanent ASIC advantage
is wrong; the honest claim is a narrowed, maintainable margin.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .miner import PhaseTiming
from .pow import (
    PoWError,
    PoWParameters,
    Share,
    derive_matrix,
    generate_candidate,
    infinity_norm,
    syndrome,
    verify_share,
)


def column(matrix: tuple, index: int) -> tuple:
    """Column `index` of A, the incremental-update step."""
    return tuple(row[index] for row in matrix)


def apply_flip(current_syndrome: tuple, matrix: tuple, position: int,
               sign: int, q: int) -> tuple:
    """s' = s + sign * A[:, position] mod q -- n additions, not n*m mults."""
    col = column(matrix, position)
    return tuple((s + sign * c) % q for s, c in zip(current_syndrome, col))


@dataclass
class WalkState:
    """One walker: a sparse vector and its syndrome, updated incrementally."""

    vector: list
    syndrome: tuple

    def copy(self) -> "WalkState":
        return WalkState(vector=list(self.vector), syndrome=self.syndrome)


def mine_balanced(header: bytes, params: PoWParameters, *,
                  max_steps: int = 200_000, prefix: int | None = None,
                  nonce: int = 0) -> tuple:
    """Data-dependent random walk. Returns (share_or_None, timing, steps).

    Phase A' (incremental syndrome update) and Phase B (table probe)
    alternate with the NEXT flip position chosen from what the table
    returned, so neither phase can run ahead of the other.
    """
    if max_steps <= 0:
        raise PoWError("max_steps must be positive")
    width = params.n if prefix is None else prefix
    if not 1 <= width <= params.n:
        raise PoWError("prefix must lie in [1, n]")

    matrix = derive_matrix(header, params)
    timing = PhaseTiming()
    table: dict = {}

    # Seed the walk from a deterministic candidate.
    seed_vector = list(generate_candidate(header, nonce, 0, params))
    state = WalkState(
        vector=seed_vector,
        syndrome=syndrome(matrix, seed_vector, params.q),
    )

    position = 0
    for step in range(max_steps):
        # ---- Phase A': incremental update (GPU-shaped arithmetic, now cheap)
        started = time.perf_counter()
        sign = 1 if (step & 1) == 0 else -1
        new_syndrome = apply_flip(
            state.syndrome, matrix, position, sign, params.q
        )
        state.vector[position] = (state.vector[position] + sign) % params.q
        state.syndrome = new_syndrome
        timing.generate_seconds += time.perf_counter() - started
        timing.candidates += 1

        # ---- Phase B: table probe (CPU-shaped: random access + branches)
        started = time.perf_counter()
        key = tuple(state.syndrome[:width])
        bucket = table.get(key)
        found = None
        if bucket is None:
            table[key] = [tuple(state.vector)]
        else:
            for other in bucket:
                difference = tuple(
                    (a - b) % params.q
                    for a, b in zip(state.vector, other)
                )
                if any(v != 0 for v in difference):
                    found = difference
                    break
            bucket.append(tuple(state.vector))
        timing.collide_seconds += time.perf_counter() - started
        timing.probes += 1

        if found is not None and infinity_norm(found, params.q) <= params.beta:
            share = Share(header=header, nonce=nonce, vector=found)
            if verify_share(share, params).valid:
                return share, timing, step + 1

        # ---- DATA DEPENDENCE: the next position comes from the table state.
        # An accelerator cannot precompute this without Phase B's result.
        bucket_len = len(table.get(key, ()))
        position = (
            position + 1 + bucket_len + (state.syndrome[0] % params.m)
        ) % params.m

    return None, timing, max_steps


__all__ = ["WalkState", "apply_flip", "column", "mine_balanced"]
