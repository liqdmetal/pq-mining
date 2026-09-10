"""Conservative attack bound for the verifier's *bounded-weight* language.

The verifier accepts any support of weight 1..w_max.  A calibration that
scores only exact weight w_max, or that subtracts 256-t bits from a full
syndrome ISD estimate, does not describe that language at partial difficulty.
The latter can even produce negative "work" bits, which is an invalid security
statement.

This module supplies a deliberately simple generic attack upper bound: split
the columns into two halves, enumerate supports of weights a and b, index one
list by the matched syndrome prefix, and scan the other.  It is not claimed
to be the best attack.  The construction gives an attack costing at most the
reported work, so a faster attack may exist; the number is an upper bound on
security, not a lower bound.

At n=512, w_max=32, t=164 it gives 82.4461 bits of classical work. This retires
the old invalid 2.67-bit Stern/BJMM edge story for the partial-syndrome
verifier until a coding theorist supplies a proper bounded-language model.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

from .calibration import log2_binom


class BoundedWorkError(ValueError):
    """Invalid bounded-weight attack parameters."""


@dataclass(frozen=True)
class MITMExecutionResult:
    """Result of a capped executable two-list search."""

    support: tuple[int, ...] | None
    candidates_enumerated: int
    capped: bool

    @property
    def found(self) -> bool:
        return self.support is not None

    def summary(self) -> dict:
        return {
            "found": self.found,
            "support": list(self.support) if self.support is not None else None,
            "candidates_enumerated": self.candidates_enumerated,
            "capped": self.capped,
            "warning": (
                "toy/reference execution only; capped results are inconclusive"
            ),
        }


@dataclass(frozen=True)
class MITMAttack:
    """One two-list attack choice and its conservative work estimate."""

    target_weight: int
    left_weight: int
    right_weight: int
    left_list_bits: float
    right_list_bits: float
    matched_bits: int
    work_bits: float
    expected_pair_excess_bits: float
    split_index: int

    @property
    def memory_bits(self) -> float:
        return max(self.left_list_bits, self.right_list_bits)

    def summary(self) -> dict:
        return {
            "target_weight": self.target_weight,
            "left_weight": self.left_weight,
            "right_weight": self.right_weight,
            "left_list_bits": round(self.left_list_bits, 3),
            "right_list_bits": round(self.right_list_bits, 3),
            "matched_bits": self.matched_bits,
            "work_bits_upper_bound": round(self.work_bits, 3),
            "memory_bits": round(self.memory_bits, 3),
            "expected_pair_excess_bits": round(self.expected_pair_excess_bits, 3),
            "split_index": self.split_index,
        }


@dataclass(frozen=True)
class MITMCertificate:
    """Machine-checkable accounting certificate for one MITM construction.

    This certifies the arithmetic and search coverage of the modeled
    construction. It does not certify cryptanalytic optimality or a practical
    implementation at large parameters.
    """

    n: int
    syndrome_bits: int
    max_weight: int
    matched_bits: int
    attack: MITMAttack
    split_count: int
    weight_count: int

    def summary(self) -> dict:
        return {
            "certificate_type": "bounded-weight-two-list-mitm-accounting",
            "parameters": {
                "n": self.n,
                "syndrome_bits": self.syndrome_bits,
                "max_weight": self.max_weight,
                "matched_bits": self.matched_bits,
            },
            "coverage": {
                "column_splits": self.split_count,
                "accepted_weights": self.weight_count,
                "split_range": "1..n-1",
                "weight_range": "1..max_weight",
            },
            "attack": self.attack.summary(),
            "verified": True,
            "warning": (
                "certifies this construction's arithmetic and coverage only; "
                "not cryptanalytic optimality or a security level"
            ),
        }


@dataclass(frozen=True)
class MITMAttack:
    """One two-list attack choice and its conservative work estimate."""

    target_weight: int
    left_weight: int
    right_weight: int
    left_list_bits: float
    right_list_bits: float
    matched_bits: int
    work_bits: float
    expected_pair_excess_bits: float
    split_index: int

    @property
    def memory_bits(self) -> float:
        return max(self.left_list_bits, self.right_list_bits)

    def summary(self) -> dict:
        return {
            "target_weight": self.target_weight,
            "left_weight": self.left_weight,
            "right_weight": self.right_weight,
            "left_list_bits": round(self.left_list_bits, 3),
            "right_list_bits": round(self.right_list_bits, 3),
            "matched_bits": self.matched_bits,
            "work_bits_upper_bound": round(self.work_bits, 3),
            "memory_bits": round(self.memory_bits, 3),
            "expected_pair_excess_bits": round(self.expected_pair_excess_bits, 3),
            "split_index": self.split_index,
        }


def _validate(n: int, syndrome_bits: int, max_weight: int,
              matched_bits: int) -> None:
    if n <= 1 or syndrome_bits <= 0:
        raise BoundedWorkError("n and syndrome_bits must be positive")
    if not (0 < max_weight < n):
        raise BoundedWorkError("max_weight must lie in (0,n)")
    if not (0 < matched_bits <= syndrome_bits):
        raise BoundedWorkError("matched_bits must lie in (0,syndrome_bits]")


def execute_mitm_instance(
    columns: tuple[int, ...],
    target: int,
    syndrome_bits: int,
    matched_bits: int,
    max_weight: int,
    attack: MITMAttack,
    *,
    max_candidates: int = 1_000_000,
) -> MITMExecutionResult:
    """Execute the certified two-list construction on one concrete instance.

    This bounded CPU executor builds one list, indexes by the matched prefix,
    and scans the other list. A capped result is inconclusive.
    """
    _validate(len(columns), syndrome_bits, max_weight, matched_bits)
    if not (0 < attack.split_index < len(columns)):
        raise BoundedWorkError("attack split does not match columns")
    if attack.matched_bits != matched_bits:
        raise BoundedWorkError("attack matched_bits mismatch")
    if attack.target_weight > max_weight:
        raise BoundedWorkError("attack weight exceeds max_weight")
    if attack.target_weight != attack.left_weight + attack.right_weight:
        raise BoundedWorkError("attack weights do not sum to target weight")
    if not (0 <= attack.left_weight <= attack.split_index):
        raise BoundedWorkError("left attack weight is out of range")
    if not (0 <= attack.right_weight <= len(columns) - attack.split_index):
        raise BoundedWorkError("right attack weight is out of range")
    expected_left = log2_binom(attack.split_index, attack.left_weight)
    expected_right = log2_binom(
        len(columns) - attack.split_index, attack.right_weight,
    )
    expected_work = max(
        max(expected_left, expected_right),
        matched_bits - min(expected_left, expected_right),
    )
    expected_excess = expected_left + expected_right - matched_bits
    if not math.isclose(
        attack.left_list_bits, expected_left, rel_tol=0.0, abs_tol=1e-9,
    ):
        raise BoundedWorkError("left list accounting mismatch")
    if not math.isclose(
        attack.right_list_bits, expected_right, rel_tol=0.0, abs_tol=1e-9,
    ):
        raise BoundedWorkError("right list accounting mismatch")
    if not math.isclose(
        attack.work_bits, expected_work, rel_tol=0.0, abs_tol=1e-9,
    ):
        raise BoundedWorkError("work accounting mismatch")
    if not math.isclose(
        attack.expected_pair_excess_bits, expected_excess,
        rel_tol=0.0, abs_tol=1e-9,
    ):
        raise BoundedWorkError("pair-excess accounting mismatch")
    if attack.memory_bits != max(attack.left_list_bits, attack.right_list_bits):
        raise BoundedWorkError("memory accounting mismatch")
    if max_candidates <= 0:
        raise BoundedWorkError("max_candidates must be positive")
    total_bits = 64 * ((syndrome_bits + 63) // 64)
    mask = ((1 << matched_bits) - 1) << (total_bits - matched_bits)
    target_prefix = target & mask
    left = tuple(range(attack.split_index))
    right = tuple(range(attack.split_index, len(columns)))
    left_map: dict[int, tuple[int, ...]] = {}
    checked = 0
    for support in itertools.combinations(left, attack.left_weight):
        if checked >= max_candidates:
            return MITMExecutionResult(None, checked, True)
        accumulator = 0
        for index in support:
            accumulator ^= columns[index]
        left_map.setdefault(accumulator & mask, support)
        checked += 1
    for support in itertools.combinations(right, attack.right_weight):
        if checked >= max_candidates:
            return MITMExecutionResult(None, checked, True)
        accumulator = 0
        for index in support:
            accumulator ^= columns[index]
        needed = target_prefix ^ (accumulator & mask)
        candidate = left_map.get(needed)
        checked += 1
        if candidate is not None:
            return MITMExecutionResult(candidate + support, checked, False)
    return MITMExecutionResult(None, checked, False)


def mitm_attack_work(
    n: int,
    syndrome_bits: int,
    max_weight: int,
    matched_bits: int,
    *,
    split_index: int | None = None,
) -> MITMAttack:
    """Return the cheapest two-list attack over accepted weights and cuts.

    If ``split_index`` is omitted, every cut from 1 through ``n-1`` is
    evaluated. This closes the old midpoint-only accounting gap. The returned
    value is still only an attack upper bound for this specific two-list model.
    """
    _validate(n, syndrome_bits, max_weight, matched_bits)
    if split_index is not None and not (0 < split_index < n):
        raise BoundedWorkError("split_index must lie in (0,n)")
    splits = range(1, n) if split_index is None else (split_index,)
    best: MITMAttack | None = None
    for cut in splits:
        left_size = cut
        right_size = n - cut
        for total_weight in range(1, max_weight + 1):
            left_min = max(0, total_weight - right_size)
            left_max = min(left_size, total_weight)
            for left_weight in range(left_min, left_max + 1):
                right_weight = total_weight - left_weight
                left_bits = log2_binom(left_size, left_weight)
                right_bits = log2_binom(right_size, right_weight)
                if not math.isfinite(left_bits) or not math.isfinite(right_bits):
                    continue
                expected_excess = left_bits + right_bits - matched_bits
                work_bits = max(
                    max(left_bits, right_bits),
                    matched_bits - min(left_bits, right_bits),
                )
                candidate = MITMAttack(
                    target_weight=total_weight,
                    left_weight=left_weight,
                    right_weight=right_weight,
                    left_list_bits=left_bits,
                    right_list_bits=right_bits,
                    matched_bits=matched_bits,
                    work_bits=work_bits,
                    expected_pair_excess_bits=expected_excess,
                    split_index=cut,
                )
                if best is None or candidate.work_bits < best.work_bits:
                    best = candidate
    if best is None:
        raise BoundedWorkError("no valid support split")
    return best


def max_matched_bits_bounded(
    n: int,
    syndrome_bits: int,
    max_weight: int,
    surplus_bits: float = 4.0,
) -> int:
    """Return the highest prefix difficulty retaining bounded-language surplus.

    The verifier accepts every support weight from 1 through ``max_weight``.
    Therefore the cap is derived from the sum of those support volumes, not from
    the exact ``max_weight`` term alone.
    """
    _validate(n, syndrome_bits, max_weight, 1)
    if surplus_bits < 0:
        raise BoundedWorkError("surplus_bits must be non-negative")
    total_terms = [log2_binom(n, weight) for weight in range(1, max_weight + 1)]
    pivot = max(total_terms)
    log2_volume = pivot + math.log2(
        sum(2.0 ** (term - pivot) for term in total_terms)
    )
    return max(0, min(syndrome_bits, math.floor(log2_volume - surplus_bits)))


def mitm_split_sweep(
    n: int, syndrome_bits: int, max_weight: int, matched_bits: int,
) -> list[MITMAttack]:
    """Return the best attack at every possible split position."""
    return [
        mitm_attack_work(
            n, syndrome_bits, max_weight, matched_bits, split_index=cut,
        )
        for cut in range(1, n)
    ]


def certify_mitm_attack(
    n: int, syndrome_bits: int, max_weight: int, matched_bits: int,
) -> MITMCertificate:
    """Build a machine-checkable certificate for the full MITM sweep.

    The certificate proves that the reported construction was minimized over
    every column split and every accepted support weight. It does not prove
    that this is the best cryptanalytic algorithm or a security level.
    """
    _validate(n, syndrome_bits, max_weight, matched_bits)
    attack = mitm_attack_work(n, syndrome_bits, max_weight, matched_bits)
    return MITMCertificate(
        n=n,
        syndrome_bits=syndrome_bits,
        max_weight=max_weight,
        matched_bits=matched_bits,
        attack=attack,
        split_count=n - 1,
        weight_count=max_weight,
    )


def verify_mitm_certificate(certificate: MITMCertificate) -> bool:
    """Independently verify a full-sweep MITM accounting certificate."""
    expected = certify_mitm_attack(
        certificate.n,
        certificate.syndrome_bits,
        certificate.max_weight,
        certificate.matched_bits,
    )
    if expected.split_count != certificate.split_count:
        return False
    if expected.weight_count != certificate.weight_count:
        return False
    return expected.attack == certificate.attack


def log2_expected_bounded_solutions(n: int, syndrome_bits: int,
                                    max_weight: int,
                                    matched_bits: int) -> float:
    """log2 expected accepted supports of every weight 1..max_weight."""
    _validate(n, syndrome_bits, max_weight, matched_bits)
    terms = [log2_binom(n, w) - matched_bits
             for w in range(1, max_weight + 1)]
    pivot = max(terms)
    return pivot + math.log2(sum(2.0 ** (term - pivot) for term in terms))


__all__ = [
    "BoundedWorkError",
    "MITMAttack",
    "MITMCertificate",
    "MITMExecutionResult",
    "certify_mitm_attack",
    "execute_mitm_instance",
    "log2_expected_bounded_solutions",
    "max_matched_bits_bounded",
    "mitm_attack_work",
    "mitm_split_sweep",
    "verify_mitm_certificate",
]
