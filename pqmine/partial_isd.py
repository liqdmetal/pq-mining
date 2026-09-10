"""Bounded reference solver for the verifier's partial-syndrome language.

This is an executable two-list information-set-decoding baseline. It is useful
for validating the symbolic attack accounting on toy instances and for checking
that a candidate support is accepted by the real verifier. It is deliberately
bounded: the recommended parameters are not meant to be solved by this Python
reference path.

No security claim follows from a successful toy solve or from the returned
candidate count. Better ISD/BDD, quantum, and structured-support attacks may
exist.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Sequence

from .verifier import PoWParams, Share, derive_instance, syndrome_mask, verify_share


class PartialISDError(ValueError):
    """Invalid or resource-bounded partial-ISD request."""


@dataclass(frozen=True)
class PartialISDParams:
    """Limits and verifier-language parameters for the reference solver."""

    syndrome_bits: int
    matched_bits: int
    max_weight: int
    max_candidates: int = 1_000_000

    def validate(self, n_columns: int) -> None:
        if n_columns <= 1:
            raise PartialISDError("at least two columns are required")
        if self.syndrome_bits <= 0:
            raise PartialISDError("syndrome_bits must be positive")
        if not (0 < self.matched_bits <= self.syndrome_bits):
            raise PartialISDError("matched_bits must be in (0, syndrome_bits]")
        if not (0 < self.max_weight < n_columns):
            raise PartialISDError("max_weight must be in (0, n_columns)")
        if self.max_candidates <= 0:
            raise PartialISDError("max_candidates must be positive")


@dataclass(frozen=True)
class PartialISDResult:
    """Result and measured work from one bounded two-list solve."""

    support: tuple[int, ...] | None
    weight: int | None
    candidates_enumerated: int
    left_entries: int
    right_entries: int
    total_weight: int | None
    left_weight: int | None
    right_weight: int | None
    matched_bits: int
    capped: bool

    @property
    def found(self) -> bool:
        return self.support is not None

    @property
    def work_log2(self) -> float:
        """log2 of enumerated candidate entries, or -inf for no work."""
        if self.candidates_enumerated <= 0:
            return float("-inf")
        return math.log2(self.candidates_enumerated)

    def summary(self) -> dict:
        return {
            "found": self.found,
            "support": list(self.support) if self.support is not None else None,
            "weight": self.weight,
            "candidates_enumerated": self.candidates_enumerated,
            "work_log2": round(self.work_log2, 3),
            "left_entries": self.left_entries,
            "right_entries": self.right_entries,
            "total_weight": self.total_weight,
            "left_weight": self.left_weight,
            "right_weight": self.right_weight,
            "matched_bits": self.matched_bits,
            "capped": self.capped,
            "warning": (
                "toy/reference accounting only; not a certified security level"
            ),
        }


def _mask(syndrome_bits: int, matched_bits: int) -> int:
    total_bits = 64 * ((syndrome_bits + 63) // 64)
    return ((1 << matched_bits) - 1) << (total_bits - matched_bits)


def _enumerate_weighted(
    columns: Sequence[int],
    weight: int,
    mask: int,
    max_entries: int,
):
    """Yield ``(support, masked_xor)`` for one partition and weight."""
    if weight < 0 or weight > len(columns):
        return
    for support in itertools.combinations(range(len(columns)), weight):
        if max_entries <= 0:
            raise PartialISDError("candidate cap reached")
        acc = 0
        for index in support:
            acc ^= columns[index]
        yield support, acc & mask
        max_entries -= 1


def solve_partial_mitm(
    columns: Sequence[int],
    target: int,
    params: PartialISDParams,
) -> PartialISDResult:
    """Find a bounded support whose XOR matches the target prefix.

    The columns are split at the midpoint. For every total weight from 1 to
    ``max_weight``, one side is indexed by its matched syndrome prefix and the
    other side is scanned for the complementary prefix. All supports are sets
    because each side is generated with ``itertools.combinations``.

    ``max_candidates`` is a hard cap over generated list entries. A capped
    result is not evidence of failure; it means the reference solver stopped.
    """
    columns = tuple(int(value) for value in columns)
    params.validate(len(columns))
    if any(value < 0 for value in columns):
        raise PartialISDError("columns must be non-negative integers")
    mask = _mask(params.syndrome_bits, params.matched_bits)
    target_prefix = int(target) & mask
    midpoint = len(columns) // 2
    left = columns[:midpoint]
    right = columns[midpoint:]
    candidates = 0
    left_entries = 0
    right_entries = 0

    for total_weight in range(1, params.max_weight + 1):
        left_min = max(0, total_weight - len(right))
        left_max = min(len(left), total_weight)
        for left_weight in range(left_min, left_max + 1):
            right_weight = total_weight - left_weight
            table: dict[int, list[tuple[int, ...]]] = {}
            try:
                for support, prefix in _enumerate_weighted(
                    left, left_weight, mask, params.max_candidates - candidates,
                ):
                    table.setdefault(prefix, []).append(support)
                    candidates += 1
                    left_entries += 1
            except PartialISDError:
                return PartialISDResult(
                    None, None, candidates, left_entries, right_entries,
                    None, None, None, params.matched_bits, True,
                )

            try:
                for support, prefix in _enumerate_weighted(
                    right, right_weight, mask, params.max_candidates - candidates,
                ):
                    candidates += 1
                    right_entries += 1
                    needed = target_prefix ^ prefix
                    for left_support in table.get(needed, ()):
                        combined = left_support + tuple(
                            midpoint + index for index in support
                        )
                        if not combined:
                            continue
                        return PartialISDResult(
                            combined, total_weight, candidates,
                            left_entries, right_entries, total_weight,
                            left_weight, right_weight,
                            params.matched_bits, False,
                        )
            except PartialISDError:
                return PartialISDResult(
                    None, None, candidates, left_entries, right_entries,
                    None, None, None, params.matched_bits, True,
                )

    return PartialISDResult(
        None, None, candidates, left_entries, right_entries,
        None, None, None, params.matched_bits, False,
    )


def solve_and_verify(
    header: bytes,
    nonce: int,
    pow_params: PoWParams,
    attack_params: PartialISDParams,
) -> tuple[PartialISDResult, Share | None]:
    """Run the reference solver on the nonce-bound verifier instance."""
    columns, target = derive_instance(header, pow_params, nonce)
    result = solve_partial_mitm(columns, target, attack_params)
    if result.support is None:
        return result, None
    share = Share(header, nonce, result.support)
    verdict = verify_share(share, pow_params)
    if not verdict.valid:
        raise PartialISDError(
            f"internal solver result rejected by verifier: {verdict.reason}"
        )
    return result, share


__all__ = [
    "PartialISDError",
    "PartialISDParams",
    "PartialISDResult",
    "solve_and_verify",
    "solve_partial_mitm",
]
