"""Stern-style ISD miner for the code-based PoW.

THE ALGORITHM
-------------
Plain birthday split (the core of Stern/Dumer ISD, without the Gaussian
elimination step -- for a PoW we do not need to decode an arbitrary syndrome,
only to find SOME weight-w error hitting the target):

  1. Split the n positions into two halves A and B.
  2. Enumerate weight-(w/2) supports from A; store XOR-syndrome -> support
     in a table.
  3. Enumerate weight-(w/2) supports from B; for each, look up
     target ^ syndrome_B in the table. A hit means
     syndrome_A ^ syndrome_B == target, i.e. a full solution of weight w.

Why this shape matters for the PoW thesis:

  * Step 2/3 cost per candidate is w/2 XORs of (n-k)-bit words -- almost
    nothing. The ASIC-friendly dense-arithmetic phase that dominated the
    M-SIS design (97% of runtime) is gone.
  * The table must hold ~2^((n-k)/2) entries for a hit to be likely, so
    TIME AND MEMORY ARE THE SAME QUANTITY. Halving memory forces roughly
    quadratically more time (the standard time-memory tradeoff for
    birthday search), which is Equihash's asymmetry argument on a
    post-quantum assumption.

WHAT THIS IS NOT
----------------
Not a state-of-the-art ISD. BJMM/May-Ozerov use nested birthday levels and
representations to do better asymptotically, and a real deployment must
calibrate difficulty against those, not against this. This miner exists to
(a) prove the PoW is solvable and verifiable end-to-end, and (b) measure the
phase balance, which is the whole reason for moving off lattices.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from itertools import combinations

from .codebased import (
    CodeBasedError,
    SDPParameters,
    SDPShare,
    derive_columns,
    derive_target_syndrome,
    syndrome_of,
    verify_share,
)


@dataclass
class ISDTiming:
    """Phase split: XOR arithmetic vs table access."""

    xor_seconds: float = 0.0
    table_seconds: float = 0.0
    candidates: int = 0
    probes: int = 0
    table_entries: int = 0

    @property
    def total_seconds(self) -> float:
        return self.xor_seconds + self.table_seconds

    @property
    def xor_fraction(self) -> float:
        total = self.total_seconds
        return self.xor_seconds / total if total else 0.0

    @property
    def table_fraction(self) -> float:
        total = self.total_seconds
        return self.table_seconds / total if total else 0.0

    def summary(self) -> dict:
        return {
            "candidates": self.candidates,
            "probes": self.probes,
            "table_entries": self.table_entries,
            "xor_seconds": round(self.xor_seconds, 4),
            "table_seconds": round(self.table_seconds, 4),
            "xor_fraction": round(self.xor_fraction, 3),
            "table_fraction": round(self.table_fraction, 3),
            "note": (
                "xor_fraction is the ASIC/GPU-friendly arithmetic share; "
                "table_fraction is the memory-bound share. Compare against "
                "the M-SIS design's 0.97 arithmetic share."
            ),
        }


@dataclass
class ISDResult:
    share: SDPShare | None
    timing: ISDTiming
    found: bool


def mine_sdp(header: bytes, params: SDPParameters, *,
             max_left: int = 200_000, max_right: int = 200_000) -> ISDResult:
    """Birthday-split search for a weight-w error hitting the target syndrome.

    `max_left` / `max_right` bound the enumeration so the search terminates
    on unsatisfiable or under-provisioned instances instead of hanging.
    """
    if max_left <= 0 or max_right <= 0:
        raise CodeBasedError("enumeration bounds must be positive")

    columns = derive_columns(header, params)
    target = derive_target_syndrome(header, params)
    half_weight = params.weight // 2
    split = params.n // 2
    left_positions = range(0, split)
    right_positions = range(split, params.n)

    timing = ISDTiming()
    table: dict = {}

    # ---- Build side: enumerate weight-(w/2) supports from the left half.
    for count, support in enumerate(combinations(left_positions, half_weight)):
        if count >= max_left:
            break
        started = time.perf_counter()
        value = syndrome_of(support, columns)
        timing.xor_seconds += time.perf_counter() - started
        timing.candidates += 1

        started = time.perf_counter()
        bucket = table.get(value)
        if bucket is None:
            table[value] = [support]
        else:
            bucket.append(support)
        timing.table_seconds += time.perf_counter() - started
    timing.table_entries = len(table)

    # ---- Probe side: enumerate the right half and look for completions.
    for count, support in enumerate(combinations(right_positions, half_weight)):
        if count >= max_right:
            break
        started = time.perf_counter()
        value = syndrome_of(support, columns)
        needed = value ^ target
        timing.xor_seconds += time.perf_counter() - started
        timing.candidates += 1

        started = time.perf_counter()
        matches = table.get(needed)
        timing.table_seconds += time.perf_counter() - started
        timing.probes += 1

        if not matches:
            continue
        for left_support in matches:
            combined = tuple(sorted(left_support + support))
            # Halves are disjoint by construction, so weight is exact; still
            # verify rather than trust the construction.
            share = SDPShare(header=header, support=combined)
            if verify_share(share, params).valid:
                return ISDResult(share, timing, True)

    return ISDResult(None, timing, False)


def mine_sdp_multi(header: bytes, params: SDPParameters, *,
                   restarts: int = 8, max_left: int = 200_000,
                   max_right: int = 200_000) -> ISDResult:
    """Randomised-split search: the fix for balanced-split blindness.

    A single balanced split only finds solutions placing exactly w/2 errors
    in each half. For n=128, w=6 that is ~32% of solutions, so one split
    misses two thirds of them and reports failure on a satisfiable instance.

    Real ISD randomises the information set between iterations; the PoW
    analogue is to reshuffle which positions belong to which half. Each
    restart is derived deterministically from the header and the attempt
    index, so verification stays a single XOR check -- the verifier never
    needs to know which split found the solution.
    """
    if restarts <= 0:
        raise CodeBasedError("restarts must be positive")

    columns = derive_columns(header, params)
    target = derive_target_syndrome(header, params)
    half_weight = params.weight // 2
    combined_timing = ISDTiming()

    for attempt in range(restarts):
        # Deterministic per-attempt permutation of the position set.
        seed = hashlib.shake_256(
            b"CHITIN-SDP-SPLIT-V1" + header + attempt.to_bytes(4, "big")
        ).digest(32)
        order = list(range(params.n))
        state = int.from_bytes(seed, "big")
        for index in range(params.n - 1, 0, -1):
            state = (state * 6364136223846793005 + 1442695040888963407) & (
                (1 << 128) - 1
            )
            swap = state % (index + 1)
            order[index], order[swap] = order[swap], order[index]

        split = params.n // 2
        left_positions = order[:split]
        right_positions = order[split:]

        table: dict = {}
        for count, support in enumerate(
            combinations(left_positions, half_weight)
        ):
            if count >= max_left:
                break
            started = time.perf_counter()
            value = syndrome_of(support, columns)
            combined_timing.xor_seconds += time.perf_counter() - started
            combined_timing.candidates += 1

            started = time.perf_counter()
            table.setdefault(value, []).append(support)
            combined_timing.table_seconds += time.perf_counter() - started
        combined_timing.table_entries = max(
            combined_timing.table_entries, len(table)
        )

        for count, support in enumerate(
            combinations(right_positions, half_weight)
        ):
            if count >= max_right:
                break
            started = time.perf_counter()
            value = syndrome_of(support, columns)
            needed = value ^ target
            combined_timing.xor_seconds += time.perf_counter() - started
            combined_timing.candidates += 1

            started = time.perf_counter()
            matches = table.get(needed)
            combined_timing.table_seconds += time.perf_counter() - started
            combined_timing.probes += 1

            if not matches:
                continue
            for left_support in matches:
                combined = tuple(sorted(left_support + support))
                share = SDPShare(header=header, support=combined)
                if verify_share(share, params).valid:
                    return ISDResult(share, combined_timing, True)

    return ISDResult(None, combined_timing, False)


def expected_left_entries(params: SDPParameters) -> float:
    """How many left-half entries the birthday bound wants."""
    return 2.0 ** (params.syndrome_bits / 2.0)


def memory_time_tradeoff(params: SDPParameters, memory_fraction: float
                         ) -> dict:
    """Time penalty for using only `memory_fraction` of the ideal table.

    Birthday search with a table of size L*f needs about 1/f times more
    probes to accumulate the same number of collisions, so shrinking memory
    costs time linearly at best -- and the constant-memory variant degrades
    to full enumeration, 2^(n-k) instead of 2^((n-k)/2).
    """
    if not 0 < memory_fraction <= 1:
        raise CodeBasedError("memory_fraction must lie in (0, 1]")
    ideal = expected_left_entries(params)
    return {
        "ideal_entries": int(ideal),
        "actual_entries": int(ideal * memory_fraction),
        "probe_multiplier": round(1.0 / memory_fraction, 2),
        "log2_full_enumeration": params.syndrome_bits,
        "log2_birthday": params.syndrome_bits / 2.0,
        "note": (
            "a memory-free attacker falls back to full enumeration, "
            f"2^{params.syndrome_bits} instead of "
            f"2^{params.syndrome_bits / 2.0} -- that gap is the "
            "memory-hardness argument"
        ),
    }


__all__ = [
    "ISDResult",
    "ISDTiming",
    "expected_left_entries",
    "memory_time_tradeoff",
    "mine_sdp",
]
