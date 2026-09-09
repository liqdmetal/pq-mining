"""Native distinct-column walk: does the collapse fix survive being fast?

WHY THIS MEASUREMENT IS NECESSARY
---------------------------------
`distinct_walk.py` fixed a real break -- effective weight had collapsed to
~74 columns regardless of K, so the syndrome lived in a small subspace and
the "weight-K solution to He^T = s" claim was false. The fix maintains a
used-column set and rehashes past collisions.

But that fix adds work INSIDE the step loop, and option B's entire value is
that the step is dominated by DRAM latency (measured: 18.5x cache-to-DRAM
ratio, plateauing at ~93-109 ns). If the rehash loop costs more than a
memory stall, the walk becomes compute-bound and the ASIC/GPU-resistance
argument evaporates -- the fix would have traded one break for another.

The pure-Python measurement cannot answer this: it reported ~1,750 ns/step
at EVERY arena size, i.e. it was measuring the interpreter, not the memory
system. So this must be compiled, exactly as `native_walk.py` was.

WHAT IS MEASURED
----------------
The same arena sweep as `native_walk.latency_sweep`, but with distinct
column selection active, plus the rehash probe count. Pass condition: the
cache-to-DRAM ratio stays well above 1 (memory still dominates) and the
DRAM-resident step cost stays in the tens-to-low-hundreds of ns.

A NOTE ON THE USED-SET REPRESENTATION
-------------------------------------
A Python `set` is not available in nopython numba, and it would also be the
wrong choice: a hash set lookup is itself a random memory access, which
would pollute the very measurement being taken. A flat uint8 bitmap over
the column space is used instead -- O(1), cache-resident (n bytes for n
columns, ~512 B here), and it does not add DRAM traffic. That keeps the
arena the only large random-access structure, which is what the design
intends.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

try:
    import numpy as np
    from numba import njit
    NATIVE_AVAILABLE = True
except Exception:                                    # noqa: BLE001
    np = None
    NATIVE_AVAILABLE = False


class NativeDistinctError(RuntimeError):
    """Raised when the native backend is unavailable."""


if NATIVE_AVAILABLE:

    @njit(cache=True, nogil=True)
    def _native_distinct_walk(arena, columns, used, seed, steps, mask,
                              write_density, column_count):  # pragma: no cover
        """Serial walk accumulating `steps` DISTINCT columns.

        `used` is a caller-supplied uint8 bitmap, cleared per walk. It is a
        flat array rather than a hash set so the distinctness check costs a
        cache hit, not another DRAM round trip.
        """
        for i in range(column_count):
            used[i] = 0

        state = seed
        address = state & mask
        syndrome = np.uint64(0)
        probes = 0

        for _ in range(steps):
            value = arena[address]

            index = value % column_count
            while used[index] == 1:
                probes += 1
                index = (index * np.uint64(0x9E3779B1)
                         + np.uint64(0x7F4A7C15)) % column_count
            used[index] = 1
            syndrome ^= columns[index]

            state = (state ^ value) * np.uint64(0x9E3779B97F4A7C15)
            state ^= state >> np.uint64(29)

            for write in range(write_density):
                target = (state + np.uint64(write) * np.uint64(0x27220A95)) & mask
                arena[target] = value ^ (state >> np.uint64(8 * write))

            address = state & mask

        return syndrome, probes


@dataclass
class DistinctMeasurement:
    arena_bits: int
    arena_mib: float
    steps: int
    seconds: float
    probes: int

    @property
    def ns_per_step(self) -> float:
        return self.seconds / self.steps * 1e9 if self.steps else 0.0

    @property
    def probes_per_step(self) -> float:
        return self.probes / self.steps if self.steps else 0.0

    def summary(self) -> dict:
        return {
            "arena_bits": self.arena_bits,
            "arena_mib": round(self.arena_mib, 2),
            "steps": self.steps,
            "ns_per_step": round(self.ns_per_step, 1),
            "probes_per_step": round(self.probes_per_step, 3),
        }


def native_distinct_rate(arena_bits: int, steps: int, *,
                         column_count: int = 512, write_density: int = 2,
                         warmup: bool = True) -> DistinctMeasurement:
    """Time a compiled distinct-column walk."""
    if not NATIVE_AVAILABLE:
        raise NativeDistinctError("numba/numpy unavailable")
    if not 8 <= arena_bits <= 30:
        raise NativeDistinctError("arena_bits must lie in [8, 30]")
    if steps > int(column_count * 0.9):
        raise NativeDistinctError(
            f"steps={steps} exceeds the distinct-column budget "
            f"{int(column_count * 0.9)} for {column_count} columns"
        )

    cells = 1 << arena_bits
    rng = np.random.default_rng(20260909)
    arena = rng.integers(0, 2 ** 63, size=cells, dtype=np.uint64)
    columns = rng.integers(0, 2 ** 63, size=column_count, dtype=np.uint64)
    used = np.zeros(column_count, dtype=np.uint8)
    mask = np.uint64(cells - 1)
    seed = np.uint64(0x243F6A8885A308D3)

    if warmup:
        _native_distinct_walk(arena, columns, used, seed, min(64, steps),
                              mask, write_density, column_count)

    started = time.perf_counter()
    _syndrome, probes = _native_distinct_walk(
        arena, columns, used, seed, steps, mask, write_density, column_count
    )
    elapsed = time.perf_counter() - started

    return DistinctMeasurement(
        arena_bits=arena_bits,
        arena_mib=cells * 8 / (1024 * 1024),
        steps=steps,
        seconds=elapsed,
        probes=int(probes),
    )


def repeated_walk_sweep(arena_bits_range=(12, 16, 20, 22, 24, 26),
                        steps: int = 400, repeats: int = 2000,
                        column_count: int = 512) -> dict:
    """Sweep arena size, running many walks so timing is not noise-dominated.

    One 400-step walk is far too short to time reliably, so each arena size
    runs `repeats` walks and the per-step cost is averaged. This mirrors
    real mining, which performs many walks per header.

    PERFORMANCE NOTE: the arena is allocated and randomly filled ONCE per
    arena size, not once per walk. An earlier version called
    `native_distinct_rate` in the loop, which re-allocated and re-filled a
    2^26-cell (512 MiB) array 1500 times -- the allocation dwarfed the
    measurement and the sweep took minutes. Reusing the arena is also more
    faithful to mining: a miner fills the arena once per header and then
    walks it repeatedly.
    """
    if not NATIVE_AVAILABLE:
        raise NativeDistinctError("numba/numpy unavailable")
    if steps > int(column_count * 0.9):
        raise NativeDistinctError(
            f"steps={steps} exceeds the distinct-column budget "
            f"{int(column_count * 0.9)} for {column_count} columns"
        )

    rng = np.random.default_rng(20260909)
    columns = rng.integers(0, 2 ** 63, size=column_count, dtype=np.uint64)
    used = np.zeros(column_count, dtype=np.uint8)

    rows = []
    for bits in arena_bits_range:
        cells = 1 << bits
        arena = rng.integers(0, 2 ** 63, size=cells, dtype=np.uint64)
        mask = np.uint64(cells - 1)

        # Warm the JIT once per shape, outside the timed region.
        _native_distinct_walk(arena, columns, used,
                              np.uint64(1), min(64, steps), mask, 2,
                              column_count)

        total_probes = 0
        started = time.perf_counter()
        for index in range(repeats):
            _syndrome, probes = _native_distinct_walk(
                arena, columns, used, np.uint64(0x243F6A88 + index),
                steps, mask, 2, column_count,
            )
            total_probes += int(probes)
        elapsed = time.perf_counter() - started

        total_steps = steps * repeats
        rows.append({
            "arena_bits": bits,
            "arena_mib": round(cells * 8 / (1024 * 1024), 2),
            "ns_per_step": round(elapsed / total_steps * 1e9, 1),
            "probes_per_step": round(total_probes / total_steps, 3),
        })

    smallest = rows[0]["ns_per_step"]
    largest = rows[-1]["ns_per_step"]
    ratio = largest / smallest if smallest > 0 else 0.0

    return {
        "rows": rows,
        "cache_to_dram_ratio": round(ratio, 2),
        "still_latency_bound": ratio > 3.0,
        "verdict": (
            f"STILL LATENCY-BOUND after the distinctness fix: ns/step rises "
            f"{ratio:.1f}x from cache to DRAM, so the rehash loop did not "
            "turn the walk compute-bound. The collapse fix and option B's "
            "ASIC-resistance argument coexist."
            if ratio > 3.0 else
            f"REGRESSION: ratio fell to {ratio:.1f}x, so the distinctness "
            "check now dominates the step and the walk is compute-bound. "
            "The collapse fix bought correctness at the cost of the "
            "property the design exists for -- both cannot be claimed."
        ),
    }


__all__ = [
    "NATIVE_AVAILABLE",
    "DistinctMeasurement",
    "NativeDistinctError",
    "native_distinct_rate",
    "repeated_walk_sweep",
]
