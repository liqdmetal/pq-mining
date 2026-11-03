"""Explicit research profiles and noncertifying symbolic attack screens.

Profiles are engineering candidates, not security claims. A profile is only
"numerically screened" when it clears the implemented symbolic calculations.
Attack-cost upper bounds cannot establish a security lower bound. Quantum ISD
finite-instance costs, gate-level resources and external review remain open.
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
        """Compatibility API: no implemented model establishes a security floor."""
        return False

    @property
    def passes_numeric_screen(self) -> bool:
        return (
            self.expected_solutions_bits >= 4.0
            and self.classical_attack.work_bits >= self.classical_floor_bits
            and self.quantum_attack.grover_query_bits >= self.quantum_floor_bits
            and self.quantum_attack.quantum_isd_bits >= self.quantum_floor_bits
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
            "quantum_naive_grover_query_bits": round(
                self.quantum_attack.naive_grover_query_bits, 6,
            ),
            "quantum_grover_query_bits": round(
                self.quantum_attack.grover_query_bits, 6,
            ),
            "quantum_isd_asymptotic_extrapolation_bits": round(
                self.quantum_attack.quantum_isd_bits, 6,
            ),
            "passes_current_modeled_floor": self.passes_modeled_floor,
            "passes_numeric_screen": self.passes_numeric_screen,
            "security_floor_established": False,
            "status": "research-only-unapproved",
            "warning": (
                "candidate only; not a PQ security claim. A numeric screen of "
                "attack upper bounds and asymptotic extrapolations cannot prove "
                "a finite-instance security floor. Concrete ISD and review remain."
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
    """Historical name for the unapproved n=2304 research candidate.

    The name expresses an investigation target, not achieved 128-bit security.
    In particular 0.05869*n is not a finite-instance security certificate and
    crossing 128 in that extrapolation does not approve a parameter profile.
    """
    params = PoWParams(
        n_columns=2304,
        syndrome_bits=1152,
        w_max=60,
        matched_bits=392,
    )
    return screen_profile("candidate-2304-unreviewed", params)


__all__ = [
    "ProfileScreen",
    "candidate_128_profile",
    "screen_profile",
]
