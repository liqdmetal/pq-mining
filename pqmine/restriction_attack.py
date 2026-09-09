"""Attacking the walk-restriction risk: is the SDP instance still hard?

THE RISK, STATED PRECISELY
--------------------------
`latency_bound.py` accepts a share when the syndrome accumulated along a
K-step walk equals the target. But the walk visits only addresses its own
data-dependent chain produces, so the set of REACHABLE column multisets is
a vanishing fraction of all weight-K supports. If that restricted set has
exploitable structure, an attacker decodes faster than generic ISD and the
"Classic McEliece assumption" claim collapses.

This is the same failure mode as Esser-Santini (CRYPTO 2024) on REGULAR
syndrome decoding: adding structure to make a problem convenient made it
easier to attack. I flagged it as UNANALYSED; this module analyses it.

FOUR CONCRETE ATTACKS TESTED HERE
---------------------------------
1. COLUMN BIAS. If the walk's column-selection is non-uniform, some columns
   appear far more often than others, and an attacker can restrict search to
   the frequent ones. Measured with a chi-square test against uniform, the
   same check PoSME applies to its address distribution (they report
   chi2/df = 1.0004 at 2^24).

2. SHORT CYCLES. A pointer chase over a finite arena must eventually cycle.
   If cycles are short, the walk revisits the same cells, and over F_2 a
   column XORed twice CANCELS -- so a short cycle collapses the effective
   weight and the syndrome lands in a tiny subspace. This is the most
   dangerous of the four because it is a hard structural break, not a
   statistical bias.

3. SYNDROME CONCENTRATION. If accumulated syndromes cluster instead of
   spreading over the (n-k)-bit space, difficulty is not what the parameters
   claim: an attacker aims at the dense region.

4. PREFIX REUSE. Walks for different nonces share no prefix (each nonce
   seeds a different start), but if they CONVERGE onto a common path an
   attacker amortises work across nonces -- breaking the per-nonce
   independence that difficulty calculations assume.

WHAT A PASS AND A FAIL MEAN
---------------------------
A pass here does NOT prove the composition secure -- these are the attacks I
thought of, run at toy scale. A fail is conclusive in the other direction:
it means the design is broken and must change. Negative results are the
point.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import dataclass

from .codebased import CodeBasedError, SDPParameters
from .decoupled_trace import trace_decoupled_walk
from .latency_bound import LatencyParameters, build_arena
from .verifier import PoWParams, derive_columns as derive_v2_columns


def _v2_columns(header: bytes, sdp: SDPParameters, nonce: int) -> tuple:
    """Derive the exact consensus columns for one header/nonce."""
    params = PoWParams(
        n_columns=sdp.n,
        syndrome_bits=sdp.syndrome_bits,
        w_max=max(sdp.weight, 1),
        matched_bits=sdp.syndrome_bits,
    )
    return derive_v2_columns(header, params, nonce)


@dataclass(frozen=True)
class BiasReport:
    """Chi-square uniformity test on selected columns."""

    samples: int
    categories: int
    chi_square: float
    degrees_of_freedom: int
    chi_square_per_df: float
    max_frequency_ratio: float

    @property
    def uniform(self) -> bool:
        """chi2/df near 1.0 indicates uniformity.

        Bounds chosen loosely (0.5-2.0): a toy-scale sample is noisy, and
        the question is whether there is GROSS bias an attacker could use,
        not whether the mixer is a perfect PRF.
        """
        return 0.5 <= self.chi_square_per_df <= 2.0

    def summary(self) -> dict:
        return {
            "samples": self.samples,
            "categories": self.categories,
            "chi_square_per_df": round(self.chi_square_per_df, 4),
            "max_frequency_ratio": round(self.max_frequency_ratio, 3),
            "uniform": self.uniform,
            "verdict": (
                "no gross column bias detected"
                if self.uniform else
                "BIASED: an attacker can restrict search to frequent columns"
            ),
        }


def column_bias(header: bytes, sdp: SDPParameters,
                params: LatencyParameters, nonces: int = 64,
                target_weight: int | None = None) -> BiasReport:
    """Chi-square test on columns selected by the v2 decoupled trace."""
    if nonces <= 0:
        raise CodeBasedError("nonces must be positive")
    target = sdp.weight if target_weight is None else target_weight
    counts = Counter()

    for nonce in range(nonces):
        columns = _v2_columns(header, sdp, nonce)
        arena = build_arena(header, params)
        result = trace_decoupled_walk(
            arena, nonce, params, columns, target,
        )
        for column in result.selected_columns:
            counts[column] += 1

    total = sum(counts.values())
    categories = len(columns)
    expected = total / categories
    chi_square = sum(
        (counts.get(index, 0) - expected) ** 2 / expected
        for index in range(categories)
    )
    degrees = categories - 1
    frequencies = [counts.get(i, 0) for i in range(categories)]
    lowest = max(1, min(frequencies))

    return BiasReport(
        samples=total,
        categories=categories,
        chi_square=chi_square,
        degrees_of_freedom=degrees,
        chi_square_per_df=chi_square / degrees if degrees else 0.0,
        max_frequency_ratio=max(frequencies) / lowest,
    )


@dataclass(frozen=True)
class CycleReport:
    """Revisit statistics for one walk -- the dangerous structural attack."""

    steps: int
    selected_weight: int
    distinct_addresses: int
    revisits: int
    longest_run_without_revisit: int
    effective_weight: int

    @property
    def revisit_rate(self) -> float:
        return self.revisits / self.steps if self.steps else 0.0

    @property
    def weight_collapse_ratio(self) -> float:
        """effective/selected weight after F_2 parity cancellation.

        The decoupled selector intentionally selects only about ``w`` columns
        during ``K`` memory-hard steps. Comparing effective weight with K was
        an audit bug: it called the deliberate selector sparse weight collapse.
        """
        return (
            self.effective_weight / self.selected_weight
            if self.selected_weight else 0.0
        )

    @property
    def healthy(self) -> bool:
        return self.selected_weight > 0 and self.weight_collapse_ratio > 0.5

    def summary(self) -> dict:
        return {
            "steps": self.steps,
            "selected_weight": self.selected_weight,
            "distinct_addresses": self.distinct_addresses,
            "revisits": self.revisits,
            "revisit_rate": round(self.revisit_rate, 4),
            "effective_weight": self.effective_weight,
            "weight_collapse_ratio": round(self.weight_collapse_ratio, 4),
            "healthy": self.healthy,
            "verdict": (
                "no significant weight collapse"
                if self.healthy else
                "COLLAPSED: XOR cancellation from revisits destroys the "
                "claimed error weight; the syndrome lives in a small "
                "subspace and the SDP instance is not what it claims"
            ),
        }


def cycle_analysis(header: bytes, sdp: SDPParameters,
                   params: LatencyParameters, nonce: int = 0,
                   target_weight: int | None = None) -> CycleReport:
    """Measure address revisits and selected-column parity in the decoupled trace."""
    target = sdp.weight if target_weight is None else target_weight
    columns = _v2_columns(header, sdp, nonce)
    arena = build_arena(header, params)
    result = trace_decoupled_walk(arena, nonce, params, columns, target)

    seen = set()
    revisits = 0
    current_run = 0
    longest_run = 0
    for address in result.visited:
        if address in seen:
            revisits += 1
            longest_run = max(longest_run, current_run)
            current_run = 0
        else:
            seen.add(address)
            current_run += 1
    longest_run = max(longest_run, current_run)

    # The selector already rehashes duplicate columns, so effective selected
    # weight equals the selected support size. Keep parity as a diagnostic.
    column_counts = Counter(result.selected_columns)
    effective = sum(1 for count in column_counts.values() if count % 2 == 1)

    return CycleReport(
        steps=result.steps,
        selected_weight=result.weight,
        distinct_addresses=len(seen),
        revisits=revisits,
        longest_run_without_revisit=longest_run,
        effective_weight=effective,
    )


@dataclass(frozen=True)
class ConcentrationReport:
    """Are accumulated syndromes spread or clustered?"""

    samples: int
    distinct_syndromes: int
    most_common_count: int
    syndrome_bits: int

    @property
    def collision_rate(self) -> float:
        return 1.0 - (self.distinct_syndromes / self.samples) \
            if self.samples else 0.0

    @property
    def expected_distinct(self) -> float:
        """Birthday expectation for `samples` draws from 2^bits.

        ``expm1`` is required: at 256 bits, ``1-exp(-samples/space)``
        cancels to zero in binary64 even though the expectation is about the
        sample count.
        """
        space = 2.0 ** self.syndrome_bits
        return space * (-math.expm1(-self.samples / space))

    @property
    def spread(self) -> bool:
        """Observed distinct count should be near the birthday expectation."""
        return self.distinct_syndromes >= 0.9 * self.expected_distinct

    def summary(self) -> dict:
        return {
            "samples": self.samples,
            "distinct_syndromes": self.distinct_syndromes,
            "expected_distinct": round(self.expected_distinct, 1),
            "most_common_count": self.most_common_count,
            "collision_rate": round(self.collision_rate, 5),
            "spread": self.spread,
            "verdict": (
                "syndromes spread as expected for a random map"
                if self.spread else
                "CONCENTRATED: syndromes cluster, so real difficulty is "
                "lower than the parameters claim"
            ),
        }


def syndrome_concentration(header: bytes, sdp: SDPParameters,
                           params: LatencyParameters,
                           nonces: int = 512,
                           target_weight: int | None = None) -> ConcentrationReport:
    """Sample decoupled-walk syndromes across nonces."""
    target = sdp.weight if target_weight is None else target_weight
    syndromes = Counter()
    for nonce in range(nonces):
        columns = _v2_columns(header, sdp, nonce)
        arena = build_arena(header, params)
        result = trace_decoupled_walk(
            arena, nonce, params, columns, target,
        )
        syndromes[result.syndrome] += 1

    return ConcentrationReport(
        samples=nonces,
        distinct_syndromes=len(syndromes),
        most_common_count=syndromes.most_common(1)[0][1],
        syndrome_bits=sdp.syndrome_bits,
    )


@dataclass(frozen=True)
class ConvergenceReport:
    """Do walks for different nonces merge onto shared paths?"""

    pairs_compared: int
    max_shared_suffix: int
    mean_shared_suffix: float
    steps: int

    @property
    def independent(self) -> bool:
        """Sharing more than a tenth of the path would let an attacker
        amortise work across nonces."""
        return self.max_shared_suffix < max(2, self.steps // 10)

    def summary(self) -> dict:
        return {
            "pairs_compared": self.pairs_compared,
            "steps": self.steps,
            "max_shared_suffix": self.max_shared_suffix,
            "mean_shared_suffix": round(self.mean_shared_suffix, 2),
            "independent": self.independent,
            "verdict": (
                "walks stay independent across nonces"
                if self.independent else
                "CONVERGENT: nonces share long path suffixes, so work "
                "amortises across nonces and per-nonce difficulty is wrong"
            ),
        }


def path_convergence(header: bytes, sdp: SDPParameters,
                     params: LatencyParameters,
                     nonces: int = 24,
                     target_weight: int | None = None) -> ConvergenceReport:
    """Compare decoupled walk paths pairwise for shared suffixes."""
    target = sdp.weight if target_weight is None else target_weight
    paths = []
    for nonce in range(nonces):
        columns = _v2_columns(header, sdp, nonce)
        arena = build_arena(header, params)
        trace = trace_decoupled_walk(
            arena, nonce, params, columns, target,
        )
        paths.append(trace.visited)

    longest = 0
    total = 0
    pairs = 0
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            left, right = paths[i], paths[j]
            shared = 0
            for a, b in zip(reversed(left), reversed(right)):
                if a != b:
                    break
                shared += 1
            longest = max(longest, shared)
            total += shared
            pairs += 1

    return ConvergenceReport(
        pairs_compared=pairs,
        max_shared_suffix=longest,
        mean_shared_suffix=total / pairs if pairs else 0.0,
        steps=params.steps,
    )


def full_audit(header: bytes, sdp: SDPParameters,
               params: LatencyParameters,
               target_weight: int | None = None) -> dict:
    """Run all four attacks against the actual decoupled selector."""
    target = sdp.weight if target_weight is None else target_weight
    bias = column_bias(header, sdp, params, nonces=16,
                       target_weight=target)
    cycles = cycle_analysis(header, sdp, params, target_weight=target)
    concentration = syndrome_concentration(
        header, sdp, params, nonces=256, target_weight=target,
    )
    convergence = path_convergence(
        header, sdp, params, nonces=16, target_weight=target,
    )

    failures = []
    if not bias.uniform:
        failures.append("column bias")
    if not cycles.healthy:
        failures.append("weight collapse from revisits")
    if not concentration.spread:
        failures.append("syndrome concentration")
    if not convergence.independent:
        failures.append("path convergence across nonces")

    return {
        "column_bias": bias.summary(),
        "cycles": cycles.summary(),
        "concentration": concentration.summary(),
        "convergence": convergence.summary(),
        "failures": failures,
        "passed": not failures,
        "caveat": (
            "these are the attacks I thought of, at toy scale. Passing does "
            "NOT establish security -- the restricted-support question needs "
            "a profile-aware ISD analysis by a coding theorist. Failing IS "
            "conclusive: it means the construction is broken."
        ),
    }


__all__ = [
    "BiasReport",
    "ConcentrationReport",
    "ConvergenceReport",
    "CycleReport",
    "column_bias",
    "cycle_analysis",
    "full_audit",
    "path_convergence",
    "syndrome_concentration",
]
