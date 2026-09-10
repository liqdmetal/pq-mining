"""Explicit PoW parameter profiles and conservative attack gates.

Profiles are engineering candidates, not security claims. A profile is only
"screen-passing" when it clears the currently implemented symbolic classical
MITM and generic Grover-query models. Quantum ISD, gate-level resources,
structured-support attacks, and external review remain outside this gate.
"""

from __future__ import annotations

from dataclasses import dataclass

from .bounded_work import MITMAttack, log2_expected_bounded_solutions, mitm_attack_work
from .quantum_work import QuantumAttack, grover_bounded_support_work
from .verifier import PoWParams


@dataclass(frozen=True)
class ProfileScreen:
    """Computed screen for one candidate profile."""

    name: str
    params: PoWParams
    expected_solutions_bits: float
    classical_attack: MITMAttack
    quantum_attack: QuantumAttack
    classical_floor_bits: float
    quantum_floor_bits: float

    @property
    def passes_modeled_floor(self) -> bool:
        return (
            self.expected_solutions_bits >= 4.0
            and self.classical_attack.work_bits >= self.classical_floor_bits
            and self.quantum_attack.grover_query_bits >= self.quantum_floor_bits
        )

    def summary(self) -> dict:
        return {
            "name": self.name,
            "params": {
                "n_columns": self.params.n_columns,
                "syndrome_bits": self.params.syndrome_bits,
                "w_max": self.params.w_max,
                "matched_bits": self.params.matched_bits,
            },
            "expected_solutions_bits": round(self.expected_solutions_bits, 6),
            "classical_attack_bits": round(self.classical_attack.work_bits, 6),
            "classical_split_index": self.classical_attack.split_index,
            "classical_attack_weight": self.classical_attack.target_weight,
            "quantum_grover_query_bits": round(
                self.quantum_attack.grover_query_bits, 6,
            ),
            "passes_current_modeled_floor": self.passes_modeled_floor,
            "warning": (
                "candidate only; not a PQ security claim. Quantum ISD, gate-level "
                "resources, structured-support attacks, and external review remain."
            ),
        }


def screen_profile(
    name: str,
    params: PoWParams,
    *,
    classical_floor_bits: float = 128.0,
    quantum_floor_bits: float = 128.0,
) -> ProfileScreen:
    """Compute all current attack screens for a profile."""
    expected = log2_expected_bounded_solutions(
        params.n_columns,
        params.syndrome_bits,
        params.w_max,
        params.matched_bits,
    )
    classical = mitm_attack_work(
        params.n_columns,
        params.syndrome_bits,
        params.w_max,
        params.matched_bits,
    )
    quantum = grover_bounded_support_work(
        params.n_columns,
        params.w_max,
        params.matched_bits,
        syndrome_bits=params.syndrome_bits,
    )
    return ProfileScreen(
        name=name,
        params=params,
        expected_solutions_bits=expected,
        classical_attack=classical,
        quantum_attack=quantum,
        classical_floor_bits=classical_floor_bits,
        quantum_floor_bits=quantum_floor_bits,
    )


def candidate_128_profile() -> ProfileScreen:
    """Return the first profile passing the current 128-bit modeled screens."""
    params = PoWParams(
        n_columns=512,
        syndrome_bits=256,
        w_max=60,
        matched_bits=256,
    )
    return screen_profile("candidate-128-modeled-floor", params)


__all__ = [
    "ProfileScreen",
    "candidate_128_profile",
    "screen_profile",
]
