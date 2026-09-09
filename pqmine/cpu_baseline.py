"""Fair CPU baseline for Phase A: JIT-compiled, multi-threaded native code.

WHY THIS EXISTS
---------------
The first GPU/CPU comparison used pure-Python for the CPU side and reported
a ~3,700x speedup. That number is meaningless as a hardware statement:
CPython is roughly 50-100x slower than native code for tight integer loops,
and it uses one core out of 24. Comparing a CUDA kernel to CPython measures
Python, not silicon.

This module provides the honest baseline -- numba-JIT'd, parallel across all
cores, same algorithm, same SplitMix64 candidate derivation -- so the
measured ratio reflects the actual hardware asymmetry the PoW design lives
or dies by.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except Exception:  # noqa: BLE001
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):  # type: ignore
        def wrap(fn):
            return fn
        return wrap

    prange = range  # type: ignore

GOLDEN = np.uint64(0x9E3779B97F4A7C15)
MIX1 = np.uint64(0xBF58476D1CE4E5B9)
MIX2 = np.uint64(0x94D049BB133111EB)


@njit(cache=True, nogil=True)
def _splitmix(z):
    x = z
    x = (x ^ (x >> np.uint64(30))) * MIX1
    x = (x ^ (x >> np.uint64(27))) * MIX2
    return x ^ (x >> np.uint64(31))


@njit(cache=True, parallel=True, nogil=True)
def _phase_a_parallel(matrix, seed, count, n, m, q, sparsity, out):
    """Same kernel as cuda_phase_a, JIT'd and spread over all cores."""
    for tid in prange(count):
        z = np.uint64(seed) + GOLDEN * np.uint64(tid + 1)
        acc = np.zeros(n, dtype=np.int64)
        for _ in range(sparsity):
            z = z + GOLDEN
            x = _splitmix(z)
            position = np.int64(x % np.uint64(m))
            sign = -1 if (x >> np.uint64(63)) & np.uint64(1) else 1
            for i in range(n):
                acc[i] += sign * matrix[i, position]
        for i in range(n):
            r = acc[i] % q
            out[tid, i] = r


@dataclass(frozen=True)
class CpuMeasurement:
    device: str
    candidates: int
    seconds: float
    threads: int

    @property
    def candidates_per_second(self) -> float:
        return self.candidates / self.seconds if self.seconds else 0.0

    def summary(self) -> dict:
        return {
            "device": self.device,
            "candidates": self.candidates,
            "seconds": round(self.seconds, 4),
            "candidates_per_second": round(self.candidates_per_second, 1),
            "threads": self.threads,
        }


def native_cpu_phase_a(matrix_rows, seed: int, count: int, n: int, m: int,
                       q: int, sparsity: int, warmup: bool = True):
    """Timed native CPU Phase A. Returns (syndromes, CpuMeasurement)."""
    if not NUMBA_AVAILABLE:
        raise RuntimeError("numba is required for the native CPU baseline")
    import numba

    matrix = np.asarray(matrix_rows, dtype=np.int64)
    out = np.zeros((count, n), dtype=np.int64)

    if warmup:
        # Exclude JIT compilation from the measurement.
        small = np.zeros((256, n), dtype=np.int64)
        _phase_a_parallel(matrix, seed, 256, n, m, q, sparsity, small)

    started = time.perf_counter()
    _phase_a_parallel(matrix, seed, count, n, m, q, sparsity, out)
    elapsed = time.perf_counter() - started
    return out, CpuMeasurement(
        "cpu-numba-parallel", count, elapsed, numba.get_num_threads()
    )


def verify_against_reference(matrix_rows, seed: int, n: int, m: int, q: int,
                             sparsity: int, count: int = 2048) -> dict:
    """Cross-check the JIT path against the pure-Python reference."""
    from .cuda_phase_a import equivalent_cpu_syndromes

    out, _ = native_cpu_phase_a(
        matrix_rows, seed, count, n, m, q, sparsity
    )
    reference, _ = equivalent_cpu_syndromes(
        matrix_rows, seed, count, n, m, q, sparsity
    )
    matches = sum(
        1 for row, ref in zip(out.tolist(), reference)
        if tuple(row) == tuple(ref)
    )
    return {
        "checked": count,
        "matches": matches,
        "all_match": matches == count,
    }


__all__ = [
    "CpuMeasurement",
    "NUMBA_AVAILABLE",
    "native_cpu_phase_a",
    "verify_against_reference",
]
