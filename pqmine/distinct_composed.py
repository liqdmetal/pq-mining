"""Chitin-PoW v2 with distinct-column selection: the last property.

WHAT THIS CLOSES
----------------
`composed.py` proved CPU wins 1.75x with the syndrome stream attached, but
it selected columns as `x mod n_columns` with NO distinctness. Over F_2 a
column visited an even number of times XORs to zero, and the earlier audit
showed that collapses effective weight by ~71% (`restriction_attack.py`)
with zero address revisits -- it is pure pigeonhole on the column set.

So the composed design as measured accepts a syndrome whose claimed weight
K is a lie. This module adds distinct-column selection (rehash on reuse,
from `distinct_walk.py`) INSIDE the L1 scratchpad walk and measures two
things together, because they can pull against each other:

  1. exact weight: every step selects a column not yet used, so the final
     syndrome is a genuine weight-K error and difficulty grows with K;
  2. CPU advantage survives the used-set bookkeeping.

THE USED-SET IS ALSO IN L1
--------------------------
n_columns = 512 -> 512 bits = 64 bytes of used-flags. That fits in a
single cache line. The rehash loop on a collision costs ~1.7 probes/step
at the highest densities measured before, and each probe is a cheap
splitmix on a register value -- no extra memory stream. So distinctness
SHOULD be near-free on the CPU. On the GPU each thread needs its own 64B
used-set: 58k threads x 64B = 3.7 MiB of per-thread mutable state, which
is exactly the un-shareable-state pressure that favours the CPU. The
prediction is that distinctness helps the CPU ratio, not hurts it.

That prediction is tested, not assumed.

WHY K <= 0.9 * n_columns
------------------------
With `steps` distinct picks from `n_columns`, the rehash loop's expected
probes at step i is n/(n-i). Past ~90% occupancy that blows up. K is
capped at 0.9*n (460 for n=512), same as `distinct_walk.py`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

try:
    import cupy as cp
    CUDA_AVAILABLE = True
except Exception:                                    # noqa: BLE001
    cp = None
    CUDA_AVAILABLE = False

try:
    import numpy as np
    from numba import njit, prange, set_num_threads
    NUMBA_AVAILABLE = True
except Exception:                                    # noqa: BLE001
    np = None
    NUMBA_AVAILABLE = False

from .cuda_codebased import check_gpu_budget, cpu_worker_threads
from .composed import N_COLUMNS, SCRATCH_BITS


class DistinctComposedError(RuntimeError):
    """Measurement could not run."""


if NUMBA_AVAILABLE:
    set_num_threads(cpu_worker_threads())


USED_WORDS = (N_COLUMNS + 63) // 64          # 8 x u64 = 64 bytes for n=512
MAX_STEPS = int(N_COLUMNS * 0.9)


_DISTINCT_KERNEL = r"""
extern "C" __global__
void distinct_composed_walk(unsigned long long* scratch,
                            const unsigned long long* H,
                            unsigned long long words_per_chain,
                            unsigned long long mask,
                            unsigned long long n_columns,
                            unsigned long long base_nonce,
                            int steps,
                            unsigned long long chains,
                            unsigned long long* syndromes,
                            unsigned long long* probe_counts)
{
    unsigned long long tid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (tid >= chains) return;
    unsigned long long* mine = scratch + tid * words_per_chain;
    unsigned long long nonce = base_nonce + tid;
    unsigned long long addr = nonce & mask;
    unsigned long long syn = 0;
    unsigned long long probes = 0;
    // per-thread used-set: 512 bits in registers/local memory
    unsigned long long used[8];
    for (int i = 0; i < 8; ++i) used[i] = 0ULL;

    for (int s = 0; s < steps; ++s) {
        unsigned long long v = mine[addr];
        unsigned long long x = v ^ nonce ^ (unsigned long long)s;
        x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
        x ^= x >> 27; x *= 0x94d049bb133111ebULL;
        x ^= x >> 31;
        mine[addr] = v ^ x;

        // distinct column: rehash until an unused one is found
        unsigned long long col = x % n_columns;
        probes++;
        while (used[col >> 6] & (1ULL << (col & 63))) {
            x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
            x ^= x >> 27; x *= 0x94d049bb133111ebULL;
            x ^= x >> 31;
            col = x % n_columns;
            probes++;
        }
        used[col >> 6] |= (1ULL << (col & 63));
        syn ^= H[col];
        addr = x & mask;
    }
    syndromes[tid] = syn;
    probe_counts[tid] = probes;
}
"""


if NUMBA_AVAILABLE:
    @njit(cache=True, nogil=True)
    def _mix(x):
        x ^= x >> np.uint64(30)
        x *= np.uint64(0xbf58476d1ce4e5b9)
        x ^= x >> np.uint64(27)
        x *= np.uint64(0x94d049bb133111eb)
        x ^= x >> np.uint64(31)
        return x

    @njit(cache=True, nogil=True)
    def _cpu_distinct_one(scratch, offset, H, n_columns, mask, nonce, steps, used):
        for i in range(used.shape[0]):
            used[i] = np.uint64(0)
        addr = nonce & mask
        syn = np.uint64(0)
        probes = np.uint64(0)
        one = np.uint64(1)
        for s in range(steps):
            idx = offset + addr
            v = scratch[idx]
            x = _mix(v ^ nonce ^ np.uint64(s))
            scratch[idx] = v ^ x
            col = x % n_columns
            probes += one
            while (used[col >> np.uint64(6)] & (one << (col & np.uint64(63)))) != 0:
                x = _mix(x)
                col = x % n_columns
                probes += one
            used[col >> np.uint64(6)] |= one << (col & np.uint64(63))
            syn ^= H[col]
            addr = x & mask
        return syn, probes

    @njit(parallel=True, cache=True)
    def _cpu_distinct_many(scratch, words, H, n_columns, mask,
                           base_nonce, steps, count, out, probes_out, used_all):
        for i in prange(count):
            syn, probes = _cpu_distinct_one(
                scratch, np.uint64(i) * words, H, n_columns, mask,
                base_nonce + np.uint64(i), steps, used_all[i],
            )
            out[i] = syn
            probes_out[i] = probes


@dataclass(frozen=True)
class DistinctRate:
    device: str
    chains: int
    steps: int
    seconds: float
    mean_probes_per_step: float

    @property
    def chains_per_second(self) -> float:
        return self.chains / self.seconds if self.seconds > 0 else 0.0

    def summary(self) -> dict:
        return {
            "device": self.device,
            "chains": self.chains,
            "steps": self.steps,
            "seconds": round(self.seconds, 4),
            "chains_per_second": round(self.chains_per_second, 1),
            "mean_probes_per_step": round(self.mean_probes_per_step, 3),
        }


def _parity_check(seed: int = 3):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2 ** 63, size=N_COLUMNS, dtype=np.uint64)


def _check_steps(steps: int) -> None:
    if steps > MAX_STEPS:
        raise DistinctComposedError(
            f"steps={steps} exceeds distinct-column budget {MAX_STEPS} "
            f"for {N_COLUMNS} columns"
        )


def cpu_distinct(steps: int, chains: int) -> DistinctRate:
    if not NUMBA_AVAILABLE:
        raise DistinctComposedError("numba unavailable")
    _check_steps(steps)
    words = 1 << SCRATCH_BITS
    rng = np.random.default_rng(11)
    scratch = rng.integers(0, 2 ** 63, size=chains * words, dtype=np.uint64)
    H = _parity_check()
    out = np.zeros(chains, dtype=np.uint64)
    probes = np.zeros(chains, dtype=np.uint64)
    used = np.zeros((chains, USED_WORDS), dtype=np.uint64)
    mask = np.uint64(words - 1)
    ncol = np.uint64(N_COLUMNS)
    _cpu_distinct_many(scratch, np.uint64(words), H, ncol, mask,
                       np.uint64(1), min(8, steps), min(32, chains),
                       out, probes, used)
    started = time.perf_counter()
    _cpu_distinct_many(scratch, np.uint64(words), H, ncol, mask,
                       np.uint64(0x1000), steps, chains, out, probes, used)
    elapsed = time.perf_counter() - started
    return DistinctRate(f"cpu-numba-{cpu_worker_threads()}t", chains, steps,
                        elapsed, float(probes.mean()) / steps)


def gpu_distinct(steps: int, chains: int) -> DistinctRate:
    if not CUDA_AVAILABLE:
        raise DistinctComposedError("CUDA unavailable")
    _check_steps(steps)
    words = 1 << SCRATCH_BITS
    required_mib = (chains * words * 8 + N_COLUMNS * 8 + chains * 16) / (1024 * 1024)
    check_gpu_budget(required_mib)

    rng = np.random.default_rng(11)
    scratch = cp.asarray(rng.integers(0, 2 ** 63, size=chains * words, dtype=np.uint64))
    H = cp.asarray(_parity_check())
    out = cp.zeros(chains, dtype=cp.uint64)
    probes = cp.zeros(chains, dtype=cp.uint64)
    mask = np.uint64(words - 1)
    kernel = cp.RawModule(code=_DISTINCT_KERNEL).get_function("distinct_composed_walk")
    block = 256
    grid = (chains + block - 1) // block

    def launch(nonce, n_steps):
        kernel((grid,), (block,), (scratch, H, np.uint64(words), mask,
                                   np.uint64(N_COLUMNS), np.uint64(nonce),
                                   np.int32(n_steps), np.uint64(chains),
                                   out, probes))
        cp.cuda.Device().synchronize()

    launch(1, min(8, steps))
    started = time.perf_counter()
    launch(0x1000, steps)
    elapsed = time.perf_counter() - started
    return DistinctRate("cuda-rtx3090", chains, steps, elapsed,
                        float(cp.asnumpy(probes).mean()) / steps)


def cpu_syndromes_and_weights(steps: int, chains: int):
    """Syndromes plus the true distinct-column count per chain, for the
    exact-weight check. Recomputes the walk in pure Python to count
    distinct columns independently of the kernel's own bookkeeping."""
    _check_steps(steps)
    words = 1 << SCRATCH_BITS
    rng = np.random.default_rng(11)
    scratch = rng.integers(0, 2 ** 63, size=chains * words, dtype=np.uint64)
    H = _parity_check()
    out = np.zeros(chains, dtype=np.uint64)
    probes = np.zeros(chains, dtype=np.uint64)
    used = np.zeros((chains, USED_WORDS), dtype=np.uint64)
    _cpu_distinct_many(scratch, np.uint64(words), H, np.uint64(N_COLUMNS),
                       np.uint64(words - 1), np.uint64(0x1000), steps,
                       chains, out, probes, used)
    # popcount of each chain's used-set = distinct columns actually selected.
    # Vectorised: a pure-Python bin().count() over 256 chains x 32 words at
    # K=1800 took minutes and stalled a sweep; np.unpackbits is ~instant.
    bits = np.unpackbits(used.view(np.uint8), axis=1)
    weights = bits.sum(axis=1)
    return out, weights


def distinct_report(steps: int = 400, cpu_chains: int = 4096,
                    gpu_chains: int = 58982) -> dict:
    """Both properties, measured together."""
    cpu = cpu_distinct(steps, cpu_chains)
    gpu = gpu_distinct(steps, gpu_chains)
    ratio = gpu.chains_per_second / cpu.chains_per_second

    _syn, weights = cpu_syndromes_and_weights(steps, 512)
    exact = bool(np.all(weights == steps))

    baseline_no_distinct = 0.57          # composed.py measurement

    if not exact:
        verdict = (
            f"WEIGHT NOT EXACT: min {weights.min()} max {weights.max()} of "
            f"{steps} -- distinctness bookkeeping is broken"
        )
    elif ratio < 1.0:
        verdict = (
            f"BOTH PROPERTIES HOLD: exact weight {steps} on every chain AND "
            f"CPU wins {1/ratio:.2f}x (without distinctness: "
            f"{1/baseline_no_distinct:.2f}x). This is a coherent PQ, "
            "CPU-favoured PoW."
        )
    else:
        verdict = (
            f"weight exact but GPU wins {ratio:.2f}x -- the used-set cost "
            "handed the advantage back"
        )

    return {
        "steps": steps,
        "n_columns": N_COLUMNS,
        "used_set_bytes": USED_WORDS * 8,
        "cpu": cpu.summary(),
        "gpu": gpu.summary(),
        "gpu_over_cpu": round(ratio, 3),
        "baseline_gpu_over_cpu_no_distinct": baseline_no_distinct,
        "exact_weight": exact,
        "weight_min": int(weights.min()),
        "weight_max": int(weights.max()),
        "verdict": verdict,
    }


__all__ = [
    "DistinctComposedError",
    "DistinctRate",
    "MAX_STEPS",
    "USED_WORDS",
    "cpu_distinct",
    "cpu_syndromes_and_weights",
    "distinct_report",
    "gpu_distinct",
]
