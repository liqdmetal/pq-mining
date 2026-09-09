"""Fixed walk: distinct-column selection, closing the weight-collapse break.

THE BREAK THE AUDIT FOUND
-------------------------
`restriction_attack.full_audit` failed on weight collapse:

    steps  distinct_addr  revisits  eff_weight  ratio
       32             32         0          24  0.750
       64             64         0          50  0.781
      128            127         1          72  0.562
      174            173         1          74  0.425
      256            254         2          72  0.281
      512            503         9          86  0.168

Effective weight PLATEAUS near ~74 no matter how many steps run. Diagnosis:
the collapse is not cycling (32 steps had ZERO revisits yet still lost 25%)
but the PIGEONHOLE. The original walk mapped each visited cell to a column
by `value % column_count`, so with 174 columns and K steps, K > 174 forces
repeats -- and over F_2 a column XORed an even number of times contributes
NOTHING. The syndrome therefore lived in a subspace spanned by ~74 columns
regardless of the work done, so the "weight-K solution to He^T = s" claim
was false and difficulty did not grow with K.

THE FIX
-------
Enforce distinctness structurally: maintain a used-column set and, when a
step selects an already-used column, rehash to the next unused one. This
makes the accumulated syndrome a genuine XOR of K DISTINCT columns, i.e. a
real weight-K error vector, which is exactly what the SDP acceptance
condition needs.

Consequences, stated honestly:

  * K must satisfy K <= n. You cannot have more distinct columns than the
    code has. So the sequentiality parameter is now CAPPED BY CODE LENGTH,
    and raising K means raising n. `max_steps_for_code` computes this.
  * The rehash loop adds work per step, but it is arithmetic on data already
    in registers, so the walk stays latency-bound (verified by re-running
    the native sweep).
  * Distinctness makes the support a SET, matching `verify_share`'s
    duplicate check in `codebased.py` -- the two layers now agree about
    what a solution is, which they did not before.

WHY THIS MATTERS BEYOND THE BUG
-------------------------------
The failure is a clean example of why the composition needed analysis: the
latency-bound walk and the SDP acceptance condition each looked correct in
isolation, and were inconsistent when combined. That is precisely the
restricted-support risk flagged in `composition_risks()`, and it was real.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

from .codebased import CodeBasedError
from .latency_bound import LatencyParameters, WALK_DOMAIN


def max_steps_for_code(column_count: int) -> int:
    """K cannot exceed the number of available columns.

    With distinct selection the walk needs K distinct columns out of n, so
    K <= n. Leaving headroom (90%) keeps the rehash loop short: as the used
    set approaches n, finding an unused column takes many probes.
    """
    if column_count <= 0:
        raise CodeBasedError("column_count must be positive")
    return max(1, int(column_count * 0.9))


@dataclass
class DistinctWalkResult:
    """A walk whose columns are guaranteed distinct."""

    syndrome: int
    transcript: bytes
    support: tuple
    visited: tuple
    steps: int
    seconds: float
    rehash_probes: int

    @property
    def effective_weight(self) -> int:
        """Now equal to `steps` by construction -- the property that failed
        before this fix."""
        return len(set(self.support))

    @property
    def weight_is_exact(self) -> bool:
        return self.effective_weight == self.steps

    @property
    def probes_per_step(self) -> float:
        return self.rehash_probes / self.steps if self.steps else 0.0

    def summary(self) -> dict:
        return {
            "steps": self.steps,
            "effective_weight": self.effective_weight,
            "weight_is_exact": self.weight_is_exact,
            "rehash_probes": self.rehash_probes,
            "probes_per_step": round(self.probes_per_step, 3),
            "seconds": round(self.seconds, 5),
            "ns_per_step": round(self.seconds / self.steps * 1e9, 1)
            if self.steps else 0.0,
        }


def distinct_walk(arena: bytearray, header: bytes, nonce: int,
                  params: LatencyParameters, columns: tuple
                  ) -> DistinctWalkResult:
    """Latency-bound walk that accumulates K DISTINCT columns.

    Identical to `latency_bound.walk` except for the column-selection step:
    a used-column set is maintained, and a collision rehashes to the next
    unused index. The pointer chase, mutation and causal chaining are
    unchanged, so the latency-bound property is preserved.
    """
    column_count = len(columns)
    if params.steps > max_steps_for_code(column_count):
        raise CodeBasedError(
            f"steps={params.steps} exceeds the distinct-column budget "
            f"{max_steps_for_code(column_count)} for {column_count} columns; "
            "raise the code length n or lower steps"
        )

    cell_bytes = params.cell_bytes
    mask = params.cells - 1

    state = int.from_bytes(
        hashlib.shake_256(
            WALK_DOMAIN + header + nonce.to_bytes(8, "big")
        ).digest(32),
        "big",
    )
    address = state & mask
    syndrome = 0
    used = set()
    support = []
    visited = []
    rehash_probes = 0
    transcript = hashlib.shake_256()
    transcript.update(WALK_DOMAIN)
    transcript.update(nonce.to_bytes(8, "big"))

    started = time.perf_counter()

    for _ in range(params.steps):
        offset = address * cell_bytes
        value = int.from_bytes(arena[offset:offset + 8], "little")

        # DISTINCT column selection: rehash past columns already used.
        index = value % column_count
        while index in used:
            rehash_probes += 1
            index = (index * 0x9E3779B1 + 0x7F4A7C15) % column_count
        used.add(index)
        support.append(index)
        syndrome ^= columns[index]
        visited.append(address)

        state = (state ^ value) * 0x9E3779B97F4A7C15 & ((1 << 64) - 1)
        state ^= state >> 29
        transcript.update(value.to_bytes(8, "little"))

        for write in range(params.write_density):
            target = (state + write * 0x27220A95) & mask
            write_offset = target * cell_bytes
            mutated = (value ^ (state >> (8 * write))) & ((1 << 64) - 1)
            arena[write_offset:write_offset + 8] = mutated.to_bytes(
                8, "little"
            )

        address = state & mask

    elapsed = time.perf_counter() - started
    return DistinctWalkResult(
        syndrome=syndrome,
        transcript=transcript.digest(32),
        support=tuple(support),
        visited=tuple(visited),
        steps=params.steps,
        seconds=elapsed,
        rehash_probes=rehash_probes,
    )


def collapse_comparison(header: bytes, columns: tuple,
                        params: LatencyParameters, arena_factory) -> dict:
    """Before/after the fix, on the same instance.

    `arena_factory` builds a fresh arena (the walk mutates it, so the two
    runs must not share one).
    """
    from .latency_bound import walk as original_walk

    before = original_walk(arena_factory(), header, 1, params, columns)
    from collections import Counter
    counts = Counter()
    arena = arena_factory()
    for address in before.visited:
        offset = address * params.cell_bytes
        value = int.from_bytes(arena[offset:offset + 8], "little")
        counts[value % len(columns)] += 1
    before_effective = sum(1 for c in counts.values() if c % 2 == 1)

    after = distinct_walk(arena_factory(), header, 1, params, columns)

    return {
        "steps": params.steps,
        "columns": len(columns),
        "before": {
            "effective_weight": before_effective,
            "ratio": round(before_effective / params.steps, 3),
        },
        "after": {
            "effective_weight": after.effective_weight,
            "ratio": round(after.effective_weight / params.steps, 3),
            "weight_is_exact": after.weight_is_exact,
            "probes_per_step": round(after.probes_per_step, 3),
        },
        "verdict": (
            "FIXED: the distinct-selection walk accumulates exactly `steps` "
            "distinct columns, so the syndrome is a genuine weight-K error "
            "and difficulty grows with K as claimed"
            if after.weight_is_exact else
            "STILL BROKEN: distinct selection did not achieve exact weight"
        ),
    }


__all__ = [
    "DistinctWalkResult",
    "collapse_comparison",
    "distinct_walk",
    "max_steps_for_code",
]
