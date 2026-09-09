"""Difficulty calibration for the code-based PoW: the blocker for option 2.

WHY THIS IS HARD (and why hash PoW never had this problem)
----------------------------------------------------------
Hash PoW difficulty is a leading-zero count. It is smooth (one bit = 2x),
monotone, and EVERY nonce search has solutions. SDP difficulty has none of
those properties for free:

  1. It depends on (n, k, w) jointly through ISD complexity, which is
     exponential in w with a rate-dependent exponent -- not a single knob.
  2. Instances can be UNSATISFIABLE. A syndrome s may have NO weight-w
     preimage. A miner then burns the whole block interval on an
     impossible puzzle. Measured earlier: 2^-6.3 expected solutions at one
     naive parameter pick, i.e. ~99% of headers unsolvable.
  3. Several ISD algorithms exist with different exponents. Calibrating
     against the wrong one hands a hidden advantage to whoever runs the
     right one -- the "secret ASIC with no fab" problem.

THE CALIBRATION MODEL
---------------------
Two quantities per parameter set, both computed here:

  E[solutions] = C(n, w) / 2^(n-k)
      expected number of weight-w vectors hitting a uniform syndrome.
      REQUIRE  log2 E[sol] >= surplus (default 4 bits, ~94% satisfiable
      by a Poisson tail; 6 bits -> 99.8%).

  work = ISD cost to find ONE solution, in bit-operations, under the
      algorithm an attacker ACTUALLY runs. Three are modelled:
        Prange 1962   : the floor -- if you can't beat Prange you can't
                        mine at all
        Stern 1989    : what the reference miner implements (birthday
                        split); the HONEST calibration target
        BJMM 2012     : the best practical algorithm; the attacker's edge
      May-Ozerov 2015 (2^0.0966n) is NOT used: it is a galactic algorithm
      whose constants only pay off at astronomical n (the SDP-PoW paper
      says so explicitly). Calibrating against it would UNDERSTATE the
      real work by orders of magnitude.

  attacker_edge = work_Stern / work_BJMM
      how much faster an optimal attacker mines than the reference miner.
      This is the number a PoW must publish. If it exceeds ~2-4x the
      difficulty is effectively set by BJMM, not by what honest miners
      run, and the chain should ship BJMM in the reference miner instead.

DIFFICULTY RETARGET
-------------------
Difficulty must move SMOOTHLY, so it cannot be "pick n,k,w" per block --
those are discrete and the work jumps in big steps. Instead:

  * fix (n, k) and the rate k/n for an epoch;
  * hold w fixed (it sets the ISD exponent -- the security knob);
  * retarget by requiring the SYNDROME to match on only the top t bits of
    the n-k available, i.e. accept He^T == s on t <= n-k bits. This
    multiplies E[solutions] by 2^(n-k-t) and divides work by the same,
    giving a smooth 1-bit-per-step ladder EXACTLY like hash PoW, while
    the underlying instance stays a full-weight SDP instance.

  So difficulty D in [0, n-k]: harder = more syndrome bits must match.
  The satisfiability surplus is then  log2 C(n,w) - t  and the retarget
  must never push t past  log2 C(n,w) - surplus.

WHAT THIS DOES NOT DO
---------------------
It does not prove the walk-restricted instance (latency_bound /
distinct_composed) is as hard as plain SDP -- only walk-reachable
supports are searchable. That is the standing composition caveat and it
applies to the work numbers here: they are for PLAIN SDP.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


class CalibrationError(ValueError):
    """Bad parameters."""


def log2_binom(n: int, k: int) -> float:
    if k < 0 or k > n:
        return float("-inf")
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)) / math.log(2)


def log2_expected_solutions(n: int, k: int, w: int, matched_bits: int | None = None) -> float:
    """log2 E[#weight-w e with He^T = s on `matched_bits` syndrome bits].

    matched_bits defaults to the full n-k. Fewer matched bits = more
    solutions = easier; this is the retarget lever.
    """
    t = (n - k) if matched_bits is None else matched_bits
    if not (0 <= t <= n - k):
        raise CalibrationError("matched_bits must be in [0, n-k]")
    return log2_binom(n, w) - t


def satisfiable_probability(log2_expected: float) -> float:
    """P[at least one solution] under a Poisson model: 1 - exp(-E)."""
    if log2_expected > 60:
        return 1.0
    return 1.0 - math.exp(-(2.0 ** log2_expected))


# ---------------------------------------------------------------------------
# ISD work models. All return log2(bit-operations) to find one solution.
# Iteration counts use the standard success-probability-per-iteration
# formulas; per-iteration cost is Gaussian elimination + the algorithm's
# own search. These are the textbook estimates (Peters 2010 / Baldi et al.),
# not tight, but they are what the field calibrates against.
# ---------------------------------------------------------------------------

def _gauss_cost(n: int, k: int) -> float:
    """log2 cost of one (n-k) x n Gaussian elimination, ~ (n-k)^2 * n / 2."""
    return math.log2(max(1.0, (n - k) ** 2 * n / 2))


def work_prange(n: int, k: int, w: int) -> float:
    """Prange: succeed when all w errors land outside the k-column info set.
    P = C(n-k, w) / C(n, w). Work = gauss / P."""
    log_p = log2_binom(n - k, w) - log2_binom(n, w)
    return _gauss_cost(n, k) - log_p


def work_stern(n: int, k: int, w: int, p: int = 2, ell: int | None = None) -> float:
    """Stern: split info set in two, p errors in each half, ell-bit filter.
    P = C(k/2,p)^2 * C(n-k-ell, w-2p) / C(n,w).
    Per-iteration cost = gauss + 2*C(k/2,p) list build + collisions."""
    if ell is None:
        ell = max(1, int(math.log2(max(2, math.comb(k // 2, p)))))
    if w < 2 * p:
        return work_prange(n, k, w)
    half = k // 2
    log_p = (2 * log2_binom(half, p)
             + log2_binom(n - k - ell, w - 2 * p)
             - log2_binom(n, w))
    list_size = log2_binom(half, p)
    # collisions expected: L^2 / 2^ell
    collisions = 2 * list_size - ell
    per_iter = math.log2(2 ** _gauss_cost(n, k) + 2 * 2 ** list_size + 2 ** max(0.0, collisions))
    return per_iter - log_p


def work_bjmm(n: int, k: int, w: int) -> float:
    """BJMM 2012 asymptotic-style estimate: the practical best.

    Full BJMM has a 3-level representation tree with tuned (p, ell, eps);
    reproducing the optimiser is out of scope. We use the standard rule of
    thumb that BJMM beats Stern by a factor ~2^(0.02n) in the exponent at
    rates near 1/2 (its asymptotic exponent 0.1019n vs Stern's 0.1166n),
    scaled by the actual rate via the ratio of exponents. This is an
    ESTIMATE of the attacker's edge, labelled as such.
    """
    rate = k / n
    # asymptotic exponents at the given rate for full decoding distance are
    # not closed-form; use the well-known rate-1/2 values and scale by the
    # Prange-normalised ratio. Honest label: approximation.
    stern_exp_half = 0.1166
    bjmm_exp_half = 0.1019
    advantage_frac = 1.0 - bjmm_exp_half / stern_exp_half     # ~12.6%
    stern = work_stern(n, k, w)
    prange = work_prange(n, k, w)
    # BJMM's gain applies to the exponential part above Gaussian elimination
    exponential_part = max(0.0, stern - _gauss_cost(n, k))
    return _gauss_cost(n, k) + exponential_part * (1.0 - advantage_frac * (rate / 0.5))


@dataclass(frozen=True)
class Calibration:
    n: int
    k: int
    w: int
    matched_bits: int
    surplus_bits: float

    @property
    def log2_solutions(self) -> float:
        return log2_expected_solutions(self.n, self.k, self.w, self.matched_bits)

    @property
    def satisfiable(self) -> float:
        return satisfiable_probability(self.log2_solutions)

    @property
    def work_prange(self) -> float:
        return work_prange(self.n, self.k, self.w) - (self.n - self.k - self.matched_bits)

    @property
    def work_stern(self) -> float:
        return work_stern(self.n, self.k, self.w) - (self.n - self.k - self.matched_bits)

    @property
    def work_bjmm(self) -> float:
        return work_bjmm(self.n, self.k, self.w) - (self.n - self.k - self.matched_bits)

    @property
    def attacker_edge_bits(self) -> float:
        """log2(work_Stern / work_BJMM): how much an optimal attacker beats
        the reference miner. Must be published; >2 bits means the chain
        should ship BJMM as the reference."""
        return self.work_stern - self.work_bjmm

    def summary(self) -> dict:
        return {
            "n": self.n, "k": self.k, "w": self.w, "rate": round(self.k / self.n, 3),
            "matched_syndrome_bits": self.matched_bits,
            "of_available": self.n - self.k,
            "log2_expected_solutions": round(self.log2_solutions, 2),
            "satisfiable_probability": round(self.satisfiable, 4),
            "meets_surplus": self.log2_solutions >= self.surplus_bits,
            "work_bits": {
                "prange": round(self.work_prange, 1),
                "stern (reference miner)": round(self.work_stern, 1),
                "bjmm (attacker)": round(self.work_bjmm, 1),
            },
            "attacker_edge_bits": round(self.attacker_edge_bits, 2),
            "attacker_edge_factor": round(2 ** self.attacker_edge_bits, 2),
        }


def max_matched_bits(n: int, k: int, w: int, surplus_bits: float = 4.0) -> int:
    """Hardest retarget that keeps the instance satisfiable with the
    required surplus. The retarget must never exceed this."""
    ceiling = int(math.floor(log2_binom(n, w) - surplus_bits))
    return max(0, min(n - k, ceiling))


def difficulty_ladder(n: int, k: int, w: int, surplus_bits: float = 4.0) -> list:
    """Every retarget step from easiest (t=0) to the satisfiability cap.
    Each step is exactly one bit of work, like hash PoW."""
    cap = max_matched_bits(n, k, w, surplus_bits)
    return [Calibration(n, k, w, t, surplus_bits) for t in range(0, cap + 1)]


def calibrate(n: int, k: int, w: int, target_work_bits: float,
              surplus_bits: float = 4.0) -> Calibration:
    """Pick the retarget t whose Stern work is closest to the target,
    refusing any t past the satisfiability cap."""
    if not (0 < k < n) or not (0 < w <= n):
        raise CalibrationError("need 0 < k < n and 0 < w <= n")
    ladder = difficulty_ladder(n, k, w, surplus_bits)
    if not ladder:
        raise CalibrationError("no satisfiable retarget exists for these parameters")
    best = min(ladder, key=lambda c: abs(c.work_stern - target_work_bits))
    return best


__all__ = [
    "Calibration",
    "CalibrationError",
    "calibrate",
    "difficulty_ladder",
    "log2_binom",
    "log2_expected_solutions",
    "max_matched_bits",
    "satisfiable_probability",
    "work_bjmm",
    "work_prange",
    "work_stern",
]
