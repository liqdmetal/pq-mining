"""Parameter selection for the code-based PoW -- so instances are solvable.

THE PROBLEM THE FIRST RUN EXPOSED
---------------------------------
Three trial parameter sets gave:

    n=64  k=44 w=4  syndrome=20b  E[solutions]=2^-0.7  -> found
    n=96  k=68 w=4  syndrome=28b  E[solutions]=2^-6.3  -> NOT found
    n=128 k=96 w=6  syndrome=32b  E[solutions]=2^+0.3  -> NOT found

The n=96 failure is not a miner bug: with E[solutions] = 2^-6.3 the instance
almost certainly has NO weight-4 error hitting the target, so no amount of
searching helps. A PoW whose instances are frequently unsatisfiable stalls
the chain -- miners burn work on impossible puzzles.

The n=128 failure is different and is a real miner limitation: the instance
IS satisfiable on average (2^0.3 solutions) but a balanced birthday split can
only find solutions that place EXACTLY w/2 errors in each half. A random
weight-6 solution splits 3/3 with probability only

    C(64,3)*C(64,3) / C(128,6) ~ 0.32

so ~2/3 of solutions are invisible to a single balanced split. Real ISD fixes
this by randomising the split (or the information set) between attempts --
which is exactly what the `restarts` loop in mine_sdp_multi does.

WHAT THIS MODULE PROVIDES
-------------------------
`select_parameters` picks (n, k, w) with a target solution surplus, so
instances are satisfiable with high probability, and reports the birthday
memory that difficulty then implies. `split_hit_probability` quantifies the
balanced-split blindness above so the restart count can be chosen rather
than guessed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .codebased import CodeBasedError, SDPParameters


def split_hit_probability(n: int, weight: int) -> float:
    """P(a uniform weight-w support splits exactly w/2 into each half).

    A balanced birthday split sees only these solutions, so the expected
    number of restarts before a given solution becomes visible is 1/p.
    """
    if n <= 0 or weight <= 0 or weight % 2 != 0 or weight > n:
        raise CodeBasedError("need even 0 < weight <= n")
    half = n // 2
    if weight // 2 > half:
        return 0.0
    favourable = math.comb(half, weight // 2) * math.comb(n - half, weight // 2)
    return favourable / math.comb(n, weight)


def expected_restarts_for_visibility(n: int, weight: int) -> float:
    probability = split_hit_probability(n, weight)
    return float("inf") if probability == 0 else 1.0 / probability


@dataclass(frozen=True)
class ParameterChoice:
    params: SDPParameters
    log2_expected_solutions: float
    split_probability: float
    expected_restarts: float
    log2_birthday_memory: float
    birthday_bytes_estimate: int

    def summary(self) -> dict:
        return {
            "n": self.params.n,
            "k": self.params.k,
            "weight": self.params.weight,
            "syndrome_bits": self.params.syndrome_bits,
            "log2_expected_solutions": round(self.log2_expected_solutions, 2),
            "split_probability": round(self.split_probability, 4),
            "expected_restarts": round(self.expected_restarts, 1),
            "log2_birthday_memory": round(self.log2_birthday_memory, 1),
            "birthday_mib_estimate": round(
                self.birthday_bytes_estimate / (1024 * 1024), 2
            ),
        }


def select_parameters(
    *,
    syndrome_bits: int,
    weight: int,
    solution_surplus_bits: float = 4.0,
    bytes_per_entry: int = 48,
) -> ParameterChoice:
    """Choose n (hence k) so E[solutions] ~ 2^solution_surplus_bits.

    We need log2 C(n,w) - syndrome_bits >= surplus, so grow n until the
    search space exceeds the syndrome width by the requested margin. A
    surplus of 4 bits means ~16 expected solutions: high probability of
    satisfiability without making the puzzle trivially easy.

    `bytes_per_entry` estimates real table cost (syndrome key + support +
    hashtable overhead); 48 B is a realistic figure for a compact native
    open-addressed table, not the Python dict used in the reference miner.
    """
    if syndrome_bits < 8:
        raise CodeBasedError("syndrome_bits must be at least 8")
    if weight <= 0 or weight % 2 != 0:
        raise CodeBasedError("weight must be a positive even integer")
    if solution_surplus_bits < 0:
        raise CodeBasedError("surplus must be non-negative")

    target = syndrome_bits + solution_surplus_bits
    n = weight
    while n <= 1 << 20:
        if math.log2(math.comb(n, weight)) >= target:
            break
        n += 2
    else:
        raise CodeBasedError("no n found; raise weight or lower surplus")

    k = n - syndrome_bits
    if k <= 0:
        raise CodeBasedError(
            f"syndrome_bits={syndrome_bits} too wide for n={n}; "
            "raise weight so a smaller n suffices"
        )

    params = SDPParameters(n=n, k=k, weight=weight)
    probability = split_hit_probability(n, weight)
    log2_memory = syndrome_bits / 2.0
    return ParameterChoice(
        params=params,
        log2_expected_solutions=params.log2_expected_solutions,
        split_probability=probability,
        expected_restarts=expected_restarts_for_visibility(n, weight),
        log2_birthday_memory=log2_memory,
        birthday_bytes_estimate=int((2.0 ** log2_memory) * bytes_per_entry),
    )


def difficulty_ladder(weight: int = 8, start: int = 16, stop: int = 48,
                      step: int = 4) -> list:
    """Parameter sets across a difficulty range, with memory cost each.

    Shows the design's central property: difficulty and MEMORY rise together,
    because birthday work is 2^(syndrome_bits/2) entries. Raising difficulty
    forces attackers to buy RAM, not just gates.
    """
    ladder = []
    for bits in range(start, stop + 1, step):
        try:
            choice = select_parameters(syndrome_bits=bits, weight=weight)
        except CodeBasedError:
            continue
        ladder.append(choice.summary())
    return ladder


__all__ = [
    "ParameterChoice",
    "difficulty_ladder",
    "expected_restarts_for_visibility",
    "select_parameters",
    "split_hit_probability",
]
