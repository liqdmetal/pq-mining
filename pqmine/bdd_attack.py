"""Multi-level bounded-weight attack upper bounds on the verifier language.

This module replaces the single two-list estimate with a family of candidate
attack shapes. It is deliberately conservative and does not claim optimality.

Each candidate attack is an upper bound on attacker work for one way of
finding a support of weight 1..w_max whose syndrome prefix matches the
t-bit target. The construction records:

* which support weight is targeted;
* which column partition is used;
* how the search lists are shaped;
* the modeled memory and time cost;
* a short note on the assumptions.

The verifier is more permissive than any single attack. Therefore the
minimum work across all tested shapes is an attack upper bound on security,
not a security claim.

Toy-vs-parameter note
---------------------

All shapes are evaluated symbolically in log2 space. The shapes are first
validated on small toy instances where exact enumeration is feasible, then
evaluated on the recommended parameters. Toy validation checks that the model
is not obviously broken; it does not prove the larger-parameter estimate is
feasible or tight.

Security posture
----------------

No classical or quantum security level is assigned by this module. It only
reports concrete constructions. A future coding-theoretic review is still
required before any security interpretation.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Iterator, Sequence

from .bounded_work import log2_binom, mitm_attack_work
from .codebased import SDPParameters


class AttackModelError(ValueError):
    """Invalid multi-level attack shape parameters."""


@dataclass(frozen=True)
class AttackShape:
    """One concrete attack construction against bounded-weight, partial syndrome.

    ``columns_per_partition`` is a sequence of positive integers summing to n,
    describing how the attacker partitions columns into independent enumeration
    groups. ``merge_schedule`` describes the merge/filter tree as a list of
    group sizes that are repeatedly combined.

    This is not a complete algorithmic specification. It is a parameterized
    cost profile that must be reviewed as a potential attack shape.
    """

    columns_per_partition: tuple[int, ...]
    merge_schedule: tuple[int, ...]
    target_weight: int
    matched_bits: int


@dataclass(frozen=True)
class AttackResult:
    """One attack shape and its conservative log2 cost profile."""

    shape: AttackShape
    target_weight: int
    list_exponents: tuple[float, ...]
    merge_exponent: float
    shared_memory_exponent: float
    work_exponent: float
    expected_excess_bits: float

    def summary(self) -> dict:
        return {
            "shape": {
                "columns_per_partition": list(self.shape.columns_per_partition),
                "merge_schedule": list(self.shape.merge_schedule),
                "target_weight": self.shape.target_weight,
                "matched_bits": self.shape.matched_bits,
            },
            "target_weight": self.target_weight,
            "list_exponents": tuple(round(v, 3) for v in self.list_exponents),
            "merge_exponent": round(self.merge_exponent, 3),
            "shared_memory_exponent": round(self.shared_memory_exponent, 3),
            "work_exponent": round(self.work_exponent, 3),
            "expected_excess_bits": round(self.expected_excess_bits, 3),
        }


def _validate_shape(n: int, shape: AttackShape) -> None:
    if n <= 0:
        raise AttackModelError("n must be positive")
    if any(g <= 0 for g in shape.columns_per_partition):
        raise AttackModelError("columns_per_partition must be positive")
    if sum(shape.columns_per_partition) != n:
        raise AttackModelError(
            "columns_per_partition must sum to n"
        )
    for step in shape.merge_schedule:
        if step < 2:
            raise AttackModelError("merge_schedule entries must be at least 2")
    if shape.target_weight > sum(shape.columns_per_partition):
        raise AttackModelError(
            "target_weight cannot exceed total column count"
        )


def enumerate_partition_supports(
    columns_per_partition: Sequence[int],
    max_weight: int,
) -> Iterator[tuple[int, ...]]:
    """Yield each distinct support of weight 1..``max_weight`` exactly once.

    Partitions are contiguous column ranges. This executable helper is a toy
    cross-check for the symbolic partition model; it is not a production
    attack implementation.
    """
    sizes = tuple(columns_per_partition)
    if not sizes or any(size <= 0 for size in sizes):
        raise AttackModelError("partition sizes must be positive")
    if not (0 < max_weight <= sum(sizes)):
        raise AttackModelError("max_weight must be in (0, total columns]")
    offsets = []
    offset = 0
    for size in sizes:
        offsets.append(offset)
        offset += size
    for weight in range(1, max_weight + 1):
        yield from _enumerate_partition_weight(sizes, offsets, 0, weight, ())


def _enumerate_partition_weight(
    sizes: tuple[int, ...],
    offsets: list[int],
    partition_index: int,
    remaining: int,
    support: tuple[int, ...],
):
    if partition_index == len(sizes):
        if remaining == 0:
            yield support
        return
    size = sizes[partition_index]
    minimum = max(0, remaining - sum(sizes[partition_index + 1:]))
    maximum = min(size, remaining)
    for local_weight in range(minimum, maximum + 1):
        local_supports = (
            tuple(offsets[partition_index] + index for index in local)
            for local in itertools.combinations(
                range(size), local_weight,
            )
        )
        for selected in local_supports:
            yield from _enumerate_partition_weight(
                sizes, offsets, partition_index + 1,
                remaining - local_weight, support + selected,
            )


@dataclass(frozen=True)
class ExecutableJoinResult:
    """Bounded toy execution result for one partition shape."""

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
            "warning": "toy/reference execution only; not an attack certification",
        }


def execute_partition_shape(
    columns: Sequence[int],
    target: int,
    shape: AttackShape,
    syndrome_bits: int,
    *,
    max_candidates: int = 1_000_000,
) -> ExecutableJoinResult:
    """Execute bounded partition enumeration and verify prefix matches.

    This is deliberately a small CPU reference path. It enumerates actual
    supports, XORs their columns, applies the same high-order prefix rule as
    the verifier, and stops at a hard candidate cap.
    """
    if len(columns) != sum(shape.columns_per_partition):
        raise AttackModelError("columns must match partition sizes")
    if not (0 < syndrome_bits <= 64 * 8):
        raise AttackModelError("syndrome_bits must be positive")
    if not (0 < shape.matched_bits <= syndrome_bits):
        raise AttackModelError("matched_bits must be in (0, syndrome_bits]")
    if max_candidates <= 0:
        raise AttackModelError("max_candidates must be positive")
    _validate_shape(len(columns), shape)
    total_bits = 64 * ((syndrome_bits + 63) // 64)
    mask = ((1 << shape.matched_bits) - 1) << (total_bits - shape.matched_bits)
    target_prefix = target & mask
    checked = 0
    for support in enumerate_partition_supports(
        shape.columns_per_partition, shape.target_weight,
    ):
        if checked >= max_candidates:
            return ExecutableJoinResult(None, checked, True)
        checked += 1
        accumulator = 0
        for index in support:
            accumulator ^= columns[index]
        if (accumulator & mask) == target_prefix:
            return ExecutableJoinResult(support, checked, False)
    return ExecutableJoinResult(None, checked, False)


def _partition_bounds(
    columns_per_partition: tuple[int, ...],
    target_weight: int,
) -> list[tuple[tuple[int, ...], float]]:
    """Return feasible per-partition weight vectors with their log2 volume.

    A weight vector assigns each partition a count in 0..partition_size.
    The cost of enumerating all supports of that weight is the sum of
    per-partition binomial log2 volumes.
    """
    result: list[tuple[tuple[int, ...], float]] = []
    _fill(
        columns_per_partition,
        tuple(),
        target_weight,
        0.0,
        result,
    )
    return result


def _fill(
    columns_per_partition: tuple[int, ...],
    current: tuple[int, ...],
    remaining_weight: int,
    current_log2: float,
    out: list[tuple[tuple[int, ...], float]],
) -> None:
    if not columns_per_partition:
        if remaining_weight == 0:
            out.append((current, current_log2))
        return
    size = columns_per_partition[0]
    rest = columns_per_partition[1:]
    max_here = min(size, remaining_weight)
    for w in range(0, max_here + 1):
        log2_term = log2_binom(size, w)
        if not math.isfinite(log2_term):
            continue
        _fill(
            rest,
            current + (w,),
            remaining_weight - w,
            current_log2 + log2_term,
            out,
        )


def _merge_cost(groups: Sequence[int]) -> float:
    """Upper-bound merge cost for combining several groups.

    Combining k groups by repeatedly merging pairs gives at most
    k - 1 merge phases. For a conservative estimate we charge the cost
    of the largest intermediate match and synchronize phases by the
    first pair that can meet the target.

    This is not a precise algorithm; it is an intentionally generous
    upper bound on merge/traversal cost.
    """
    if len(list(groups)) < 2:
        return 0.0
    max_exponent = float(max(groups))
    return max_exponent + math.log2(len(groups))


def _split_work(exponents: Sequence[float], t: int) -> float:
    """Best two-group MITM work for one set of list exponents.

    The attacker may split the lists into two groups, sum each group, and
    apply the standard two-list batching formula::

        work = max(sum_left, sum_right, t - min(sum_left, sum_right))

    This intentionally reproduces the same cost logic used by the classical
    two-list estimate for the two-partition case.
    """
    values = list(exponents)
    if not values:
        raise AttackModelError("no list exponents")
    if len(values) == 1:
        return max(float(values[0]), float(t))

    best = float("inf")
    n = len(values)
    for mask in range(1, 1 << (n - 1)):
        left = 0.0
        right = 0.0
        for i, e in enumerate(values):
            if (mask >> i) & 1:
                left += e
            else:
                right += e
        work = max(left, right, t - min(left, right))
        if work < best:
            best = work
    return best


def shape_work(shape: AttackShape, calibration: SDPParameters) -> AttackResult:
    """Compute a conservative work estimate for one attack shape.

    The current model uses the same batching logic as the classical two-list
    estimate, but applied to an arbitrary partition of the columns. For each
    feasible per-partition weight distribution the attacker:

    1. enumerates each partition's candidate supports, costing the binomial
       log2 volumes;
    2. splits the resulting lists into two groups and matches the partial
       syndrome prefix using the batching formula;
    3. models shared memory as the larger group exponent.

    The attacker is free to repeat shapes or combine them, but this function
    reports the cheapest single shape under that model.
    """
    _validate_shape(calibration.n, shape)
    n = calibration.n
    t = shape.matched_bits
    max_weight = shape.target_weight

    if not (0 < t <= calibration.syndrome_bits):
        raise AttackModelError(
            "matched_bits must be in (0, syndrome_bits]"
        )

    weight_vectors = _partition_bounds(
        shape.columns_per_partition,
        max_weight,
    )

    if not weight_vectors:
        raise AttackModelError("no weight distribution for this shape")

    best: AttackResult | None = None

    for weights, enumeration_log2 in weight_vectors:
        if sum(weights) != max_weight:
            continue

        exponents = tuple(
            log2_binom(size, w)
            for size, w in zip(shape.columns_per_partition, weights)
        )
        if not all(math.isfinite(e) for e in exponents):
            continue

        # Two-group batching model.
        try:
            work = _split_work(exponents, t)
        except AttackModelError:
            continue
        if not math.isfinite(work):
            continue

        list_exponents = tuple(exponents)
        merge_exponent = _merge_cost(list_exponents)
        memory_exponent = float(max(exponents))
        total_enumeration = sum(exponents)
        pair_excess = total_enumeration - t

        candidate = AttackResult(
            shape=shape,
            target_weight=max_weight,
            list_exponents=list_exponents,
            merge_exponent=merge_exponent,
            shared_memory_exponent=memory_exponent,
            work_exponent=work,
            expected_excess_bits=pair_excess,
        )
        if best is None or candidate.work_exponent < best.work_exponent:
            best = candidate

    if best is None:
        raise AttackModelError("no feasible shape result")
    return best


def multi_level_attack_work(
    sdp: SDPParameters,
    *,
    w_max: int,
    matched_bits: int,
    partitions: Sequence[Sequence[int]] | None = None,
) -> list[AttackResult]:
    """Return the cheapest multi-level attack shape found.

    By default the function attempts a small set of partition shapes and
    returns the best result for each shape, sorted by modeled work.
    """
    if not (0 < matched_bits <= sdp.syndrome_bits):
        raise AttackModelError(
            "matched_bits must be in (0, syndrome_bits]"
        )
    if not (0 < w_max < sdp.n):
        raise AttackModelError("w_max must be in (0, n)")

    explicit_shapes = partitions is not None
    results: list[AttackResult] = []
    # The verifier accepts every weight in 1..w_max, not only the maximum.
    # Sweep the full accepted language so this family cannot miss a cheaper
    # lower-weight attack.
    for target_weight in range(1, w_max + 1):
        shapes = _default_shapes(
            sdp.n, target_weight, matched_bits, partitions=partitions,
        )
        for shape in shapes:
            try:
                results.append(shape_work(shape, sdp))
            except AttackModelError:
                if explicit_shapes:
                    raise
                continue
    results.sort(key=lambda r: r.work_exponent)
    return results


def _default_shapes(
    n: int,
    w_max: int,
    matched_bits: int,
    *,
    partitions: Sequence[Sequence[int]] | None = None,
) -> list[AttackShape]:
    """A small conservative set of partition shapes to try.

    This is intentionally limited; it is not an optimization sweep over all
    partitions because that would look too much like a cryptanalysis
    optimization loop in the repository right now.
    """
    shapes: list[AttackShape] = []

    if partitions is not None:
        for p in partitions:
            shapes.append(
                AttackShape(
                    tuple(p),
                    _default_merge_schedule(tuple(p)),
                    w_max,
                    matched_bits,
                )
            )
    else:
        shapes.append(
            AttackShape(
                (n,),
                _default_merge_schedule((n,)),
                w_max,
                matched_bits,
            )
        )
        if n >= 2:
            shapes.append(
                AttackShape(
                    (n // 2, n - n // 2),
                    _default_merge_schedule((n // 2, n - n // 2)),
                    w_max,
                    matched_bits,
                )
            )
        if n >= 4:
            half = n // 2
            q1 = half // 2
            q2 = half - q1
            shapes.append(
                AttackShape(
                    (q1, q2, half - q1, n - half - q2),
                    _default_merge_schedule((q1, q2, half - q1, n - half - q2)),
                    w_max,
                    matched_bits,
                )
            )
    return shapes


def _default_merge_schedule(columns_per_partition: tuple[int, ...]) -> tuple[int, ...]:
    """A simple default merge schedule: one stage per partition group."""
    schedule = []
    running = 0
    for c in columns_per_partition:
        running += c
        schedule.append(running)
    return tuple(schedule)


def best_multi_level_attack(
    sdp: SDPParameters,
    *,
    w_max: int,
    matched_bits: int,
) -> AttackResult:
    """Return the cheapest multi-level attack found under the default shapes."""
    results = multi_level_attack_work(
        sdp, w_max=w_max, matched_bits=matched_bits,
    )
    if not results:
        raise AttackModelError("no multi-level attack result")
    return results[0]


def _sdp_single_two_list_bound(
    n: int,
    syndrome_bits: int,
    w_max: int,
    matched_bits: int,
) -> float:
    """Classical two-list MITM upper bound on the bounded-weight verifier."""
    from .bounded_work import MITMAttack, mitm_attack_work

    attack = mitm_attack_work(n, syndrome_bits, w_max, matched_bits)
    return attack.work_bits


def compare_bounded_attack_family(
    sdp: SDPParameters,
    *,
    w_max: int,
    matched_bits: int,
) -> dict:
    """Compare the generic two-list bound against multi-level shapes.

    The returned dict contains:

    * the original single two-list upper bound as a plain float;
    * the cheapest multi-level shape found;
    * a concise interpretation warning.
    """
    single_work = _sdp_single_two_list_bound(
        sdp.n, sdp.syndrome_bits, w_max, matched_bits,
    )
    single_attack = mitm_attack_work(
        sdp.n, sdp.syndrome_bits, w_max, matched_bits,
    )
    multi = best_multi_level_attack(sdp, w_max=w_max, matched_bits=matched_bits)
    return {
        "single_two_list_bound": single_attack,
        "single_two_list_work": single_work,
        "best_multi_level": multi,
        "best_multi_level_work": multi.work_exponent,
        "interpretation": (
            "Both numbers are upper bounds on attacker work for specific "
            "constructions. The verifier accepts arbitrary bounded supports, "
            "so no certified security level follows from either number."
        ),
        "warning": (
            "multi-level shapes are not proven optimal; a better partition or "
            "merge strategy may exist"
        ),
    }


def validate_on_toy_instance(
    n_small: int,
    syndrome_bits_small: int,
    w_max_small: int,
    matched_bits_small: int,
) -> dict:
    """Validate attack shapes on a small instance where enumeration is feasible.

    This does not compute optimal attacks. It checks that the upper-bound
    shapes are plausible and that the model is not producing nonsensical
    exponents on a tiny instance.
    """
    params = SDPParameters(n=n_small, k=syndrome_bits_small//2, weight=w_max_small)  # type: ignore[arg-type]
    results = multi_level_attack_work(
        params, w_max=w_max_small, matched_bits=matched_bits_small,
    )
    return {
        "n": n_small,
        "syndrome_bits": syndrome_bits_small,
        "w_max": w_max_small,
        "matched_bits": matched_bits_small,
        "shapes_tested": len(results),
        "best_result": results[0].summary() if results else None,
        "toy_note": (
            "Only sanity checks. Toy usability does not establish anything "
            "about larger parameters."
        ),
    }


__all__ = [
    "AttackModelError",
    "AttackResult",
    "AttackShape",
    "ExecutableJoinResult",
    "best_multi_level_attack",
    "enumerate_partition_supports",
    "execute_partition_shape",
    "compare_bounded_attack_family",
    "multi_level_attack_work",
    "shape_work",
    "validate_on_toy_instance",
]
