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

At n=512, w_max=32, t=164 it gives 83.06 bits of classical work. This retires
the old invalid 2.67-bit Stern/BJMM edge story for the partial-syndrome
verifier until a coding theorist supplies a proper bounded-language model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .calibration import log2_binom


class BoundedWorkError(ValueError):
    """Invalid bounded-weight attack parameters."""


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
    "log2_expected_bounded_solutions",
    "mitm_attack_work",
    "mitm_split_sweep",
]
