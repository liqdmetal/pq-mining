"""Composite attack accounting for the current bounded-weight verifier.

This module keeps two distinct attack models separate:

1. ``mitm_attack_work`` models a generic attacker against the verifier's
   accepted language (all distinct supports of weights 1..w_max, partial
   syndrome match).
2. The walk model treats the decoupled selector as a nonce-indexed candidate
   generator: one support per nonce, with an observed probability of landing
   at weight <= w_max. Since H and s are nonce-bound, a candidate's syndrome
   match is modeled as a t-bit event for that nonce.

The walk estimate is an empirical workload model, not a security proof. The
verifier does not require the walk, so a miner can always choose the generic
MITM route or another algorithm. A passing walk audit cannot raise the generic
security bound.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .bounded_work import MITMAttack, mitm_attack_work
from .codebased import CodeBasedError, SDPParameters
from .decoupled_trace import trace_decoupled_walk
from .latency_bound import LatencyParameters, build_arena
from .verifier import PoWParams, derive_instance


class RestrictedWorkError(ValueError):
    """Invalid restricted-support attack parameters."""


@dataclass(frozen=True)
class WalkScanEstimate:
    """Measured one-candidate-per-nonce workload estimate."""

    samples: int
    admissible_samples: int
    w_max: int
    matched_bits: int
    steps: int

    @property
    def admissibility_probability(self) -> float:
        return self.admissible_samples / self.samples

    @property
    def success_probability_bits(self) -> float:
        """Negative log2 success probability per nonce.

        This assumes the selected support's matched syndrome prefix is
        independent and uniform after conditioning on its admissible weight.
        That assumption is explicit and requires external review.
        """
        probability = self.admissibility_probability
        return self.matched_bits - math.log2(probability)

    @property
    def step_work_bits(self) -> float:
        """Success work plus the serial K-step cost per nonce."""
        return self.success_probability_bits + math.log2(self.steps)

    def summary(self) -> dict:
        return {
            "samples": self.samples,
            "admissible_samples": self.admissible_samples,
            "admissibility_probability": round(self.admissibility_probability, 6),
            "w_max": self.w_max,
            "matched_bits": self.matched_bits,
            "steps": self.steps,
            "success_probability_bits": round(self.success_probability_bits, 3),
            "step_work_bits": round(self.step_work_bits, 3),
            "assumption": (
                "conditional on nonce-bound H,s behaving as a uniform random "
                "matched-prefix target for the walk-generated support"
            ),
        }


@dataclass(frozen=True)
class CompositeAttack:
    """The strongest result among the modeled generic and walk routes."""

    generic: MITMAttack
    walk: WalkScanEstimate

    @property
    def best_work_bits(self) -> float:
        return min(self.generic.work_bits, self.walk.step_work_bits)

    @property
    def best_route(self) -> str:
        return (
            "generic-bounded-mitm"
            if self.generic.work_bits <= self.walk.step_work_bits
            else "walk-restricted-nonce-scan"
        )

    def summary(self) -> dict:
        return {
            "generic_mitm": self.generic.summary(),
            "walk_scan": self.walk.summary(),
            "best_modeled_route": self.best_route,
            "best_modeled_work_bits": round(self.best_work_bits, 3),
            "security_interpretation": (
                "upper bound only; the verifier accepts arbitrary bounded "
                "supports and the walk is not consensus-required"
            ),
        }


def _validate_walk_inputs(
    sdp: SDPParameters,
    params: LatencyParameters,
    nonces: int,
    target_weight: int,
    w_max: int,
) -> None:
    if nonces <= 0:
        raise RestrictedWorkError("nonces must be positive")
    if not (0 < target_weight <= params.steps):
        raise RestrictedWorkError("target_weight must lie in (0, steps]")
    if target_weight > sdp.n:
        raise RestrictedWorkError("target_weight cannot exceed code columns")
    if not (0 < w_max < sdp.n):
        raise RestrictedWorkError("w_max must lie in (0, n)")


def sample_walk_scan(
    header: bytes,
    sdp: SDPParameters,
    params: LatencyParameters,
    *,
    matched_bits: int,
    w_max: int,
    nonces: int = 256,
    target_weight: int | None = None,
) -> WalkScanEstimate:
    """Measure the walk's admissible-support fraction over nonce samples."""
    target = sdp.weight if target_weight is None else target_weight
    _validate_walk_inputs(sdp, params, nonces, target, w_max)
    if not (0 < matched_bits <= sdp.syndrome_bits):
        raise RestrictedWorkError("matched_bits must lie in (0, syndrome_bits]")

    pow_params = PoWParams(
        n_columns=sdp.n,
        syndrome_bits=sdp.syndrome_bits,
        w_max=w_max,
        matched_bits=matched_bits,
    )
    admissible = 0
    for nonce in range(nonces):
        columns, _target = derive_instance(header, pow_params, nonce)
        trace = trace_decoupled_walk(
            build_arena(header, params), nonce, params, columns, target,
        )
        if 0 < trace.weight <= w_max:
            admissible += 1
    return WalkScanEstimate(
        samples=nonces,
        admissible_samples=admissible,
        w_max=w_max,
        matched_bits=matched_bits,
        steps=params.steps,
    )


def composite_attack(
    header: bytes,
    sdp: SDPParameters,
    params: LatencyParameters,
    *,
    matched_bits: int,
    w_max: int,
    nonces: int = 256,
    target_weight: int | None = None,
) -> CompositeAttack:
    """Compare generic bounded MITM with the measured walk scan."""
    walk = sample_walk_scan(
        header, sdp, params, matched_bits=matched_bits, w_max=w_max,
        nonces=nonces, target_weight=target_weight,
    )
    generic = mitm_attack_work(
        sdp.n, sdp.syndrome_bits, w_max, matched_bits,
    )
    return CompositeAttack(generic=generic, walk=walk)


__all__ = [
    "CompositeAttack",
    "RestrictedWorkError",
    "WalkScanEstimate",
    "composite_attack",
    "sample_walk_scan",
]
