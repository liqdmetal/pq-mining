"""Small reference search that is actually compatible with the v2 verifier.

The CUDA/Numba decoupled benchmark intentionally uses a synthetic H and returns
only a syndrome/weight.  That is useful for hardware measurements, but it is
not a miner for the consensus verifier.  This module closes that boundary for
reference experiments only:

* derive H and s through ``verifier.py`` with the same header+nonce binding;
* retain the selected column support, not only its XOR;
* enforce distinctness and reject a walk whose support exceeds w_max;
* return a real ``Share`` and verify it before reporting success.

It is deliberately slow and CPU-only.  It is not a production miner, an ISD
solver, or evidence that the restricted walk is cryptographically sound.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .decoupled_trace import trace_decoupled_walk
from .latency_bound import LatencyParameters, build_arena
from .verifier import (
    PoWParams,
    Share,
    Verdict,
    derive_instance,
    verify_share,
)


class SearchShareError(ValueError):
    """Invalid reference-search parameters."""

@dataclass(frozen=True)
class ReferenceSearchResult:
    """One bounded reference search over consecutive nonces."""

    share: Share | None
    verdict: Verdict | None
    nonces_tested: int
    steps_per_nonce: int
    selected_weight: int
    seconds: float

    @property
    def found(self) -> bool:
        return self.share is not None and self.verdict is not None and self.verdict.valid

    def summary(self) -> dict:
        return {
            "found": self.found,
            "nonces_tested": self.nonces_tested,
            "steps_per_nonce": self.steps_per_nonce,
            "selected_weight": self.selected_weight,
            "seconds": round(self.seconds, 6),
            "verdict": self.verdict.reason if self.verdict else "not found",
        }


def _search_nonce(
    header: bytes,
    nonce: int,
    pow_params: PoWParams,
    latency_params: LatencyParameters,
    target_weight: int,
) -> tuple[Share | None, int]:
    """Run one decoupled walk and return a verifier-compatible support."""
    if len(header) == 0:
        raise SearchShareError("header must be non-empty")
    if not (0 <= nonce < 2**64):
        raise SearchShareError("nonce must be a uint64")
    if not (0 < target_weight <= latency_params.steps):
        raise SearchShareError("target_weight must lie in (0, steps]")

    columns, target = derive_instance(header, pow_params, nonce)
    trace = trace_decoupled_walk(
        build_arena(header, latency_params), nonce, latency_params,
        columns, target_weight,
    )
    if trace.weight > pow_params.w_max:
        return None, trace.weight

    total_bits = 64 * ((pow_params.syndrome_bits + 63) // 64)
    mask = (1 << pow_params.matched_bits) - 1
    mask <<= total_bits - pow_params.matched_bits
    if trace.selected_columns and (trace.syndrome & mask) == (target & mask):
        share = Share(header, nonce, tuple(sorted(trace.selected_columns)))
        verdict = verify_share(share, pow_params)
        if verdict.valid:
            return share, trace.weight
    return None, trace.weight


def search_reference(
    header: bytes,
    pow_params: PoWParams,
    latency_params: LatencyParameters,
    *,
    target_weight: int,
    first_nonce: int = 0,
    max_nonces: int = 256,
) -> ReferenceSearchResult:
    """Search consecutive nonces and return a real, self-verified ``Share``.

    Use tiny ``matched_bits`` and a bounded nonce count in tests.  At chain
    difficulty the expected search is intentionally expensive; this function
    must not be called as a production miner.
    """
    if max_nonces <= 0:
        raise SearchShareError("max_nonces must be positive")
    if not (0 <= first_nonce < 2**64):
        raise SearchShareError("first_nonce must be a uint64")

    started = time.perf_counter()
    last_weight = 0
    nonces_tested = 0
    for offset in range(max_nonces):
        nonce = first_nonce + offset
        if nonce >= 2**64:
            break
        nonces_tested += 1
        share, last_weight = _search_nonce(
            header, nonce, pow_params, latency_params, target_weight,
        )
        if share is not None:
            verdict = verify_share(share, pow_params)
            return ReferenceSearchResult(
                share=share,
                verdict=verdict,
                nonces_tested=nonces_tested,
                steps_per_nonce=latency_params.steps,
                selected_weight=last_weight,
                seconds=time.perf_counter() - started,
            )
    return ReferenceSearchResult(
        share=None,
        verdict=None,
        nonces_tested=nonces_tested,
        steps_per_nonce=latency_params.steps,
        selected_weight=last_weight,
        seconds=time.perf_counter() - started,
    )


__all__ = [
    "ReferenceSearchResult",
    "SearchShareError",
    "search_reference",
]
