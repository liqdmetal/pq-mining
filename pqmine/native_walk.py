"""Native latency-bound walk, because Python cannot measure DRAM latency.

WHY THIS FILE EXISTS
--------------------
The pure-Python walk in `latency_bound.py` reports ~1,750 ns/step and, more
tellingly, reports the SAME ~1,750 ns whether the arena is 0.5 MiB (fits in
L2) or 128 MiB (guaranteed DRAM). Real DRAM random access is ~80 ns and L2
is ~4 ns, so a 40x hardware difference showed up as 0%: the measurement was
entirely interpreter overhead.

That is the same class of error as the first GPU/CPU comparison, which used
CPython for the CPU side and reported a meaningless 3,700x. The fix is the
same: compile it.

WHAT THIS MEASURES
------------------
A numba-compiled pointer chase over a uint64 arena, single-threaded (the
walk is inherently serial, so threads do not apply within one walk), with
the arena size swept from cache-resident to DRAM-resident. If the design is
genuinely latency-bound, ns/step must RISE with arena size and plateau near
DRAM latency. If it stays flat, the walk is compute-bound and option B has
failed on its own premise.

The CPU thread cap from `cuda_codebased` still applies to any parallel
benchmarks; a single walk uses one thread by construction.
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


class NativeError(RuntimeError):
    """Raised when the native backend is unavailable."""


if NATIVE_AVAILABLE:

    @njit(cache=True, nogil=True)
    def _native_walk(arena, columns, seed, steps, mask, write_density,
                     column_count):                          # pragma: no cover
        """Serial data-dependent walk with in-place mutation.

        Deliberately single-threaded: the address of step i+1 depends on the
        value read at step i, so there is no parallelism to exploit WITHIN a
        walk. That dependency is the entire ASIC/GPU-resistance argument.
        """
        state = seed
        address = state & mask
        syndrome = np.uint64(0)

        for _ in range(steps):
            value = arena[address]

            # accumulate the selected column of H
            syndrome ^= columns[value % column_count]

            # next address depends on the value just read
            state = (state ^ value) * np.uint64(0x9E3779B97F4A7C15)
            state ^= state >> np.uint64(29)

            # mutate so the arena cannot be recomputed on demand
            for write in range(write_density):
                target = (state + np.uint64(write) * np.uint64(0x27220A95)) & mask
                arena[target] = value ^ (state >> np.uint64(8 * write))

            address = state & mask

        return syndrome, state


@dataclass
class NativeWalkMeasurement:
    arena_bits: int
    arena_mib: float
    steps: int
    seconds: float

    @property
    def ns_per_step(self) -> float:
        return self.seconds / self.steps * 1e9 if self.steps else 0.0

    @property
    def steps_per_second(self) -> float:
        return self.steps / self.seconds if self.seconds > 0 else 0.0

    def summary(self) -> dict:
        return {
            "arena_bits": self.arena_bits,
            "arena_mib": round(self.arena_mib, 2),
            "steps": self.steps,
            "ns_per_step": round(self.ns_per_step, 1),
            "steps_per_second": round(self.steps_per_second, 1),
        }


def native_walk_rate(arena_bits: int, steps: int, *, write_density: int = 2,
                     column_count: int = 174, warmup: bool = True
                     ) -> NativeWalkMeasurement:
    """Time a compiled walk over a 2^arena_bits x uint64 arena."""
    if not NATIVE_AVAILABLE:
        raise NativeError("numba/numpy unavailable")
    if not 8 <= arena_bits <= 30:
        raise NativeError("arena_bits must lie in [8, 30]")

    cells = 1 << arena_bits
    rng = np.random.default_rng(20260909)
    arena = rng.integers(0, 2 ** 63, size=cells, dtype=np.uint64)
    columns = rng.integers(0, 2 ** 63, size=column_count, dtype=np.uint64)
    mask = np.uint64(cells - 1)
    seed = np.uint64(0x243F6A8885A308D3)

    if warmup:
        _native_walk(arena, columns, seed, 1000, mask, write_density,
                     column_count)

    started = time.perf_counter()
    _native_walk(arena, columns, seed, steps, mask, write_density,
                 column_count)
    elapsed = time.perf_counter() - started

    return NativeWalkMeasurement(
        arena_bits=arena_bits,
        arena_mib=cells * 8 / (1024 * 1024),
        steps=steps,
        seconds=elapsed,
    )


def latency_sweep(steps: int = 2_000_000,
                  bits_range=(12, 16, 20, 22, 24, 26)) -> dict:
    """Sweep arena size and report whether the walk is latency-bound.

    The signature of a latency-bound design is ns/step rising with arena
    size as the working set escapes L2, then L3, then plateauing at DRAM
    latency. A flat curve means compute-bound.
    """
    rows = []
    for bits in bits_range:
        measurement = native_walk_rate(bits, steps)
        rows.append(measurement.summary())

    smallest = rows[0]["ns_per_step"]
    largest = rows[-1]["ns_per_step"]
    ratio = largest / smallest if smallest > 0 else 0.0

    return {
        "rows": rows,
        "cache_to_dram_ratio": round(ratio, 2),
        "latency_bound": ratio > 3.0,
        "verdict": (
            f"LATENCY-BOUND: ns/step rises {ratio:.1f}x from a "
            "cache-resident arena to a DRAM-resident one, so the walk is "
            "dominated by memory latency rather than computation. This is "
            "the property PoSME reports and the property the previous "
            "independent-probe design lacked."
            if ratio > 3.0 else
            f"NOT LATENCY-BOUND: ns/step changed only {ratio:.1f}x across "
            "the sweep, so computation dominates and the design has failed "
            "its own premise. Reducing per-step work is required before any "
            "ASIC-resistance claim."
        ),
    }


def theoretical_gpu_penalty() -> dict:
    """Why a GPU loses at THIS workload, unlike the independent-probe one.

    Numbers from PoSME (arXiv:2604.15751) and vendor specifications; this
    function records the reasoning rather than measuring it, because the
    GPU on this machine hosts a resident LLM and must not be loaded.
    """
    return {
        "independent_probes": {
            "gpu_advantage": "6.1x (MEASURED on this machine)",
            "why": (
                "thousands of resident warps hide 400-600 ns latency behind "
                "other work, so the workload is bandwidth-bound and the GPU "
                "wins on bandwidth (936 GB/s vs ~50 GB/s)"
            ),
        },
        "dependent_walk": {
            "gpu_advantage": "0.05-0.07x, i.e. 14-19x SLOWER (PoSME, 4 GPU archs)",
            "why": (
                "one walk issues one load at a time, so no parallelism is "
                "available to hide latency. The GPU pays its full ~400-600 "
                "ns per step against the CPU's ~80 ns, and HBM3 is only "
                "~1.3x faster than DDR5 at random access."
            ),
        },
        "caveat": (
            "a GPU can still run many INDEPENDENT walks concurrently. What "
            "protects the design is that VRAM capacity caps the number of "
            "resident arenas: at 128 MiB per arena a 24 GB card holds ~190, "
            "while its 10,496 cores sit idle. PoSME reports exactly this -- "
            "GPU throughput choked by VRAM capacity."
        ),
        "asic_bound": (
            "~2x per PoSME, versus 8-16x for Argon2id and 2-5x for RandomX, "
            "because an ASIC cannot beat commodity DRAM latency -- it would "
            "have to invent faster DRAM, not faster logic."
        ),
    }


__all__ = [
    "NATIVE_AVAILABLE",
    "NativeError",
    "NativeWalkMeasurement",
    "latency_sweep",
    "native_walk_rate",
    "theoretical_gpu_penalty",
]
