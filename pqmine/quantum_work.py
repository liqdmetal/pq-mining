"""Symbolic quantum query accounting for the bounded PoW language.

This is a generic amplitude-amplification model, not a quantum ISD
implementation. It treats candidate-support generation and the verifier's
matched-prefix check as a coherent oracle. The result counts oracle queries
only; it omits reversible SHAKE cost, qubits, QRAM, circuit depth, error
correction, and implementation constants.

For comparison, a Grover search over a
uniform candidate oracle has a quadratic query speedup over the 2**t random
prefix-search cost, giving approximately 2**(t/2) queries in the satisfiable
regime. Known quantum ISD algorithms may give a different or cheaper concrete
route, so this is an upper bound on the attacker's quantum work, not a security
level.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .calibration import log2_binom

# Kachigar--Tillich (https://arxiv.org/abs/1703.00263v2) report this
# worst-case asymptotic exponent in their decoding regime. Multiplication by
# our finite n ignores rate/weight/prefix-regime assumptions, polynomial
# factors and quantum resources. It is an illustrative extrapolation, NOT a
# concrete attack cost or a lower bound certifying security for these shares.
QUANTUM_ISD_EXPONENT = 0.05869


class QuantumWorkError(ValueError):
    """Invalid quantum-work parameters."""


@dataclass(frozen=True)
class QuantumAttack:
    """Generic quantum candidate-search accounting.

    ``naive_grover_query_bits`` is the pure amplitude-amplification query
    count (t/2) over the uniform candidate oracle. ``grover_query_bits`` is
    the same query-only quantity, not a security floor. The legacy field
    ``quantum_isd_bits`` contains an asymptotic extrapolation for comparison
    only; it must not be combined with finite-instance costs or certified.
    """

    n: int
    max_weight: int
    matched_bits: int
    candidate_space_bits: float
    expected_solutions_bits: float
    naive_grover_query_bits: float
    grover_query_bits: float
    quantum_isd_bits: float
    satisfiable_regime: bool
    model: str = "Grover amplitude amplification over bounded supports"

    def summary(self) -> dict:
        return {
            "n": self.n,
            "max_weight": self.max_weight,
            "matched_bits": self.matched_bits,
            "candidate_space_bits": round(self.candidate_space_bits, 6),
            "expected_solutions_bits": round(self.expected_solutions_bits, 6),
            "naive_grover_query_bits": round(self.naive_grover_query_bits, 6),
            "grover_query_bits": round(self.grover_query_bits, 6),
            "quantum_isd_asymptotic_extrapolation_bits": round(self.quantum_isd_bits, 6),
            "finite_instance_quantum_security_bits": None,
            "security_floor_established": False,
            "satisfiable_regime": self.satisfiable_regime,
            "model": self.model,
            "warning": (
                "not a PQ security claim; generic oracle-query accounting and "
                "an asymptotic ISD extrapolation, not a finite-instance attack "
                "estimate or security lower bound; no gate/qubit/QRAM model"
            ),
        }


def _log2_sum(values: list[float]) -> float:
    if not values:
        raise QuantumWorkError("candidate space is empty")
    pivot = max(values)
    return pivot + math.log2(sum(2.0 ** (value - pivot) for value in values))


def grover_bounded_support_work(
    n: int,
    max_weight: int,
    matched_bits: int,
    *,
    syndrome_bits: int | None = None,
) -> QuantumAttack:
    """Return generic Grover query accounting for weights 1..max_weight.

    ``syndrome_bits`` is optional and only supplies a validation bound. The
    matched-prefix oracle is the verifier relation; lower accepted weights are
    included exactly as in ``bounded_work.mitm_attack_work``.
    """
    if n <= 1:
        raise QuantumWorkError("n must be greater than one")
    if not (0 < max_weight < n):
        raise QuantumWorkError("max_weight must lie in (0,n)")
    if syndrome_bits is not None and not (0 < matched_bits <= syndrome_bits):
        raise QuantumWorkError("matched_bits must lie in (0,syndrome_bits]")
    if matched_bits <= 0:
        raise QuantumWorkError("matched_bits must be positive")

    candidate_space_bits = _log2_sum(
        [log2_binom(n, weight) for weight in range(1, max_weight + 1)]
    )
    expected_solutions_bits = candidate_space_bits - matched_bits
    # In the intended satisfiable regime, N candidates with N/2^t expected
    # marked items require sqrt(N/(N/2^t)) = 2^(t/2) oracle queries.
    naive_grover_query_bits = matched_bits / 2.0
    # Display separately: this asymptotic extrapolation is not in the same
    # calibrated finite-instance cost model as the generic oracle queries.
    quantum_isd_bits = QUANTUM_ISD_EXPONENT * n
    return QuantumAttack(
        n=n,
        max_weight=max_weight,
        matched_bits=matched_bits,
        candidate_space_bits=candidate_space_bits,
        expected_solutions_bits=expected_solutions_bits,
        naive_grover_query_bits=naive_grover_query_bits,
        grover_query_bits=naive_grover_query_bits,
        quantum_isd_bits=quantum_isd_bits,
        satisfiable_regime=expected_solutions_bits >= 0.0,
    )


def best_modeled_quantum_work(
    n: int,
    syndrome_bits: int,
    max_weight: int,
    matched_bits: int,
    classical_attack_bits: float,
) -> dict:
    """Compare symbolic routes, displaying asymptotic ISD separately.

    Query counts omit oracle implementation cost. Even their minimum is not
    a certified finite-instance quantum security level.
    """
    attack = grover_bounded_support_work(
        n, max_weight, matched_bits, syndrome_bits=syndrome_bits,
    )
    best = min(float(classical_attack_bits), attack.grover_query_bits)
    return {
        "classical_attack_bits": round(float(classical_attack_bits), 6),
        "naive_grover_query_bits": round(attack.naive_grover_query_bits, 6),
        "grover_query_bits": round(attack.grover_query_bits, 6),
        "quantum_isd_asymptotic_extrapolation_bits": round(attack.quantum_isd_bits, 6),
        "security_floor_established": False,
        "best_modeled_quantum_work_bits": round(best, 6),
        "selected_route": (
            "generic-grover" if attack.grover_query_bits <= classical_attack_bits
            else "classical-mitm"
        ),
        "satisfiable_regime": attack.satisfiable_regime,
        "warning": (
            "not a PQ security claim; symbolic query counts only; asymptotic "
            "ISD extrapolation is not a finite-instance cost or security floor"
        ),
    }


__all__ = [
    "QuantumAttack",
    "QuantumWorkError",
    "best_modeled_quantum_work",
    "grover_bounded_support_work",
]
