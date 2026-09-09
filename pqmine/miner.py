"""The two-phase sieve miner: GPU-shaped generation, CPU-shaped collision.

STRUCTURE
---------
Phase A (GPU-shaped): generate N sparse candidates and their syndromes.
    Independent per candidate, arithmetic-dense, streams linearly. This is
    the part a GPU or an ASIC does well.

Phase B (CPU-shaped): bucket the syndromes and probe for near-collisions.
    Random access into a table larger than on-die SRAM, branch-heavy
    hit/miss handling, pointer chasing. This is the part that resists
    fixed-function hardware.

Phase A' (data-dependent refill): the buckets where collisions were found
    determine which candidates to regenerate. This is what forbids running
    the phases as independent pipelines -- an accelerator cannot precompute
    Phase A for the next round without knowing Phase B's output.

The reference implementation here is pure-Python and single-threaded, so it
is a CORRECTNESS and STRUCTURE reference, not a performance claim. Numbers
from it say nothing about achievable hashrate; `tools/bench.py` measures
the phase split so the CPU/GPU thesis can be checked rather than asserted.

WHY NEAR-COLLISIONS GIVE SHORT VECTORS
--------------------------------------
If z_i and z_j are sparse with ||z||_inf <= s and their syndromes agree,
then A(z_i - z_j) = 0 mod q exactly, and ||z_i - z_j||_inf <= 2s. So an
exact syndrome collision between two sparse candidates yields a kernel
vector whose norm is bounded by construction. Difficulty is tuned by how
many syndrome coordinates must agree: requiring all n is hardest, requiring
a prefix is easier and is how the difficulty knob works.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field

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


@dataclass
class PhaseTiming:
    """Wall-clock split between the two phases -- the CPU/GPU evidence."""

    generate_seconds: float = 0.0
    collide_seconds: float = 0.0
    candidates: int = 0
    probes: int = 0

    @property
    def total_seconds(self) -> float:
        return self.generate_seconds + self.collide_seconds

    @property
    def generate_fraction(self) -> float:
        total = self.total_seconds
        return self.generate_seconds / total if total else 0.0

    @property
    def collide_fraction(self) -> float:
        total = self.total_seconds
        return self.collide_seconds / total if total else 0.0

    def summary(self) -> dict:
        return {
            "candidates": self.candidates,
            "probes": self.probes,
            "generate_seconds": round(self.generate_seconds, 4),
            "collide_seconds": round(self.collide_seconds, 4),
            "generate_fraction": round(self.generate_fraction, 3),
            "collide_fraction": round(self.collide_fraction, 3),
            "note": (
                "generate_fraction is the GPU-shaped share of the work; "
                "collide_fraction is the CPU-shaped share. A design where "
                "one fraction is ~1.0 has no cooperative property."
            ),
        }


@dataclass
class MiningResult:
    """Outcome of a bounded mining attempt."""

    share: Share | None
    attempts: int
    rounds: int
    timing: PhaseTiming
    found: bool = field(init=False)

    def __post_init__(self) -> None:
        self.found = self.share is not None


def syndrome_key(values, prefix: int) -> tuple:
    """Bucket key: the first `prefix` syndrome coordinates.

    prefix == n is the full syndrome (hardest). Smaller prefixes accept
    partial collisions, which is the difficulty knob: a shorter prefix means
    more colliding pairs, so less work per share.
    """
    return tuple(values[:prefix])


def phase_a_generate(header: bytes, nonce: int, start: int, count: int,
                     matrix: tuple, params: PoWParameters):
    """GPU-SHAPED: independent candidate + syndrome generation.

    Every iteration is independent of every other, touches no shared state,
    and is pure arithmetic -- the shape that vectorises and offloads. A CUDA
    or OpenCL port replaces this function body and nothing else.
    """
    out = []
    for index in range(start, start + count):
        vector = generate_candidate(header, nonce, index, params)
        out.append((index, vector, syndrome(matrix, vector, params.q)))
    return out


def phase_b_collide(batch, table, prefix: int, params: PoWParameters):
    """CPU-SHAPED: probe a large table for syndrome collisions.

    Random access, unpredictable branches, and a working set that grows past
    any on-die SRAM. Returns (pairs, probe_count) where each pair is two
    candidate vectors whose syndromes agree on the prefix.
    """
    pairs = []
    probes = 0
    for index, vector, values in batch:
        key = syndrome_key(values, prefix)
        probes += 1
        bucket = table.get(key)
        if bucket is None:
            table[key] = [(index, vector, values)]
            continue
        for other in bucket:
            pairs.append((vector, other[1]))
        bucket.append((index, vector, values))
    return pairs, probes


def reduce_pair(left, right, params: PoWParameters) -> tuple:
    """z_left - z_right, which lies in the kernel when syndromes matched."""
    return tuple(
        (a - b) % params.q for a, b in zip(left, right)
    )


def mine(header: bytes, params: PoWParameters, *, max_candidates: int = 200_000,
         batch_size: int = 2_048, prefix: int | None = None,
         nonce: int = 0) -> MiningResult:
    """Search for a valid share, alternating the two phases.

    `prefix` defaults to the full syndrome width, which is the correct
    setting: a shorter prefix produces pairs whose difference is NOT in the
    kernel, so they fail verification. Partial prefixes are useful only for
    pool share-grading, never for block validity.
    """
    if max_candidates <= 0 or batch_size <= 0:
        raise PoWError("candidate and batch counts must be positive")
    width = params.n if prefix is None else prefix
    if not 1 <= width <= params.n:
        raise PoWError("prefix must lie in [1, n]")

    matrix = derive_matrix(header, params)
    table: dict = {}
    timing = PhaseTiming()
    rounds = 0
    produced = 0

    while produced < max_candidates:
        count = min(batch_size, max_candidates - produced)

        started = time.perf_counter()
        batch = phase_a_generate(header, nonce, produced, count, matrix, params)
        timing.generate_seconds += time.perf_counter() - started

        started = time.perf_counter()
        pairs, probes = phase_b_collide(batch, table, width, params)
        timing.collide_seconds += time.perf_counter() - started

        produced += count
        timing.candidates = produced
        timing.probes += probes
        rounds += 1

        for left, right in pairs:
            vector = reduce_pair(left, right, params)
            if infinity_norm(vector, params.q) > params.beta:
                continue
            share = Share(header=header, nonce=nonce, vector=vector)
            if verify_share(share, params).valid:
                return MiningResult(share, produced, rounds, timing)

    return MiningResult(None, produced, rounds, timing)


def expected_candidates_for_collision(params: PoWParameters,
                                      prefix: int | None = None) -> float:
    """Birthday estimate: sqrt(pi/2 * q^prefix) candidates for one collision."""
    import math

    width = params.n if prefix is None else prefix
    space = float(params.q) ** width
    return math.sqrt(math.pi / 2.0 * space)


__all__ = [
    "MiningResult",
    "PhaseTiming",
    "expected_candidates_for_collision",
    "mine",
    "phase_a_generate",
    "phase_b_collide",
    "reduce_pair",
    "syndrome_key",
]
