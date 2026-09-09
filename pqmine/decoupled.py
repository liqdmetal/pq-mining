"""Decoupled walk: chain length K and error weight w as independent knobs.

THE CONFLICT THIS RESOLVES
--------------------------
Two measured requirements pointed opposite directions:

  * memory-hardness wants LONG chains (K >= 1024): the fraction test showed
    the GPU's per-chain fixed cost amortises over short walks (16x GPU at
    K=100) and the CPU only pulls ahead per-unit at K >= 400;
  * SDP hardness wants SMALL weight (w ~ 16-48 for n=512): the calibrator
    showed the composed design's w = K = 460 is a DENSE error (complement
    of weight 52), giving E[solutions] = 2^-17 at rate 1/2 -- unsolvable.

The earlier design tied them together: every walk step accumulated one
column, so w == K. That coupling is what made it incoherent.

THE DECOUPLING
--------------
Every step still does the full L1-scratchpad mutation (the memory-hard
part, K steps). But a column is accumulated into the syndrome ONLY when
the step's mixed value passes a selection test:

    x = mix(v ^ nonce ^ step)
    scratch[addr] = v ^ x                       every step (memory-hard)
    if (x >> 56) < threshold:                   selection, rate ~ w/K
        col = distinct_column(x)                target w total
        syndrome ^= H[col]
    addr = x & mask

with threshold chosen so E[selected] = w over K steps. The syndrome is
then a weight-~w error over a walk of length K. Two knobs, independent:

    K  -> memory-hardness / GPU-vs-CPU balance
    w  -> SDP hardness / difficulty

WHY EXACT WEIGHT STILL MATTERS AND HOW IT IS HANDLED
----------------------------------------------------
Selection gives weight ~w with binomial spread, not exactly w. The
verifier does NOT need exact w -- it needs w <= w_max (an error heavier
than the bound is a weaker claim, and the SDP acceptance is "find a
weight <= w_max preimage"). So: accept iff popcount(used) <= w_max AND
syndrome matches. Distinctness still applies to the SELECTED columns
(rehash on reuse), so popcount(used) IS the true weight -- the F_2
cancellation hole stays closed.

A share therefore proves: a weight-<= w_max solution to He^T = s, found
by a K-step memory-hard walk. That is a genuine bounded-weight SDP
witness with a memory-hard search.

WHAT THIS MODULE MEASURES
-------------------------
  1. The weight distribution actually lands near w (selection works).
  2. Fraction test (1 core vs 1 SM) at LONG K with SMALL w, to see whether
     the memory-hard chain length recovers the CPU's per-unit lead now
     that w no longer forces K short.
  3. Satisfiability at the chosen (n, k, w_max) via the calibrator.
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
    from numba import njit, set_num_threads
    NUMBA_AVAILABLE = True
except Exception:                                    # noqa: BLE001
    np = None
    NUMBA_AVAILABLE = False

from .cuda_codebased import check_gpu_budget
from .composed import N_COLUMNS, SCRATCH_BITS
from .fraction_test import CPU_CORES, GPU_SMS, THREADS_PER_SM


class DecoupledError(RuntimeError):
    """Measurement could not run."""


USED_WORDS = (N_COLUMNS + 63) // 64


def selection_threshold(target_weight: int, steps: int) -> int:
    """8-bit threshold so that E[selected over K steps] = target_weight.

    x>>56 is uniform on [0,256); select when below threshold, so
    P(select) = threshold/256 and E = K*threshold/256.
    """
    if not (0 < target_weight <= steps):
        raise DecoupledError("target_weight must be in (0, steps]")
    threshold = round(256 * target_weight / steps)
    return max(1, min(255, threshold))


_DECOUPLED_KERNEL = r"""
extern "C" __global__
void decoupled_walk(unsigned long long* scratch,
                    const unsigned long long* H,
                    unsigned long long words_per_chain,
                    unsigned long long mask,
                    unsigned long long n_columns,
                    unsigned long long base_nonce,
                    int steps,
                    unsigned long long threshold,
                    unsigned long long chains,
                    unsigned long long* syndromes,
                    unsigned long long* weights)
{
    unsigned long long tid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (tid >= chains) return;
    unsigned long long* mine = scratch + tid * words_per_chain;
    unsigned long long nonce = base_nonce + tid;
    unsigned long long addr = nonce & mask;
    unsigned long long syn = 0;
    unsigned long long weight = 0;
    unsigned long long used[8];
    for (int i = 0; i < 8; ++i) used[i] = 0ULL;

    for (int s = 0; s < steps; ++s) {
        unsigned long long v = mine[addr];
        unsigned long long x = v ^ nonce ^ (unsigned long long)s;
        x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
        x ^= x >> 27; x *= 0x94d049bb133111ebULL;
        x ^= x >> 31;
        mine[addr] = v ^ x;                       // every step: memory-hard

        if ((x >> 56) < threshold) {              // selection: rate w/K
            unsigned long long col = x % n_columns;
            unsigned long long y = x;
            while (used[col >> 6] & (1ULL << (col & 63))) {
                y ^= y >> 30; y *= 0xbf58476d1ce4e5b9ULL;
                y ^= y >> 27; y *= 0x94d049bb133111ebULL;
                y ^= y >> 31;
                col = y % n_columns;
            }
            used[col >> 6] |= (1ULL << (col & 63));
            syn ^= H[col];
            weight++;
        }
        addr = x & mask;
    }
    syndromes[tid] = syn;
    weights[tid] = weight;
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
    def _cpu_decoupled_one(scratch, offset, H, n_columns, mask, nonce,
                           steps, threshold, used):
        for i in range(used.shape[0]):
            used[i] = np.uint64(0)
        addr = nonce & mask
        syn = np.uint64(0)
        weight = np.uint64(0)
        one = np.uint64(1)
        for s in range(steps):
            idx = offset + addr
            v = scratch[idx]
            x = _mix(v ^ nonce ^ np.uint64(s))
            scratch[idx] = v ^ x
            if (x >> np.uint64(56)) < threshold:
                col = x % n_columns
                y = x
                while (used[col >> np.uint64(6)] & (one << (col & np.uint64(63)))) != 0:
                    y = _mix(y)
                    col = y % n_columns
                used[col >> np.uint64(6)] |= one << (col & np.uint64(63))
                syn ^= H[col]
                weight += one
            addr = x & mask
        return syn, weight


@dataclass(frozen=True)
class DecoupledRate:
    device: str
    unit: str
    chains: int
    steps: int
    target_weight: int
    seconds: float
    mean_weight: float
    max_weight: int

    @property
    def chains_per_second(self) -> float:
        return self.chains / self.seconds if self.seconds > 0 else 0.0

    def summary(self) -> dict:
        return {
            "device": self.device, "unit": self.unit, "chains": self.chains,
            "steps": self.steps, "target_weight": self.target_weight,
            "seconds": round(self.seconds, 4),
            "chains_per_second": round(self.chains_per_second, 1),
            "mean_weight": round(self.mean_weight, 2),
            "max_weight": self.max_weight,
        }


def _parity_check(seed: int = 3):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2 ** 63, size=N_COLUMNS, dtype=np.uint64)


def one_core(steps: int, target_weight: int, chains: int = 256) -> DecoupledRate:
    if not NUMBA_AVAILABLE:
        raise DecoupledError("numba unavailable")
    set_num_threads(1)
    words = 1 << SCRATCH_BITS
    rng = np.random.default_rng(11)
    scratch = rng.integers(0, 2 ** 63, size=chains * words, dtype=np.uint64)
    H = _parity_check()
    used = np.zeros(USED_WORDS, dtype=np.uint64)
    mask = np.uint64(words - 1)
    ncol = np.uint64(N_COLUMNS)
    thr = np.uint64(selection_threshold(target_weight, steps))
    _cpu_decoupled_one(scratch, np.uint64(0), H, ncol, mask, np.uint64(1),
                       min(8, steps), thr, used)
    weights = []
    started = time.perf_counter()
    for i in range(chains):
        _, w = _cpu_decoupled_one(scratch, np.uint64(i) * np.uint64(words), H,
                                  ncol, mask, np.uint64(0x1000 + i), steps,
                                  thr, used)
        weights.append(int(w))
    elapsed = time.perf_counter() - started
    return DecoupledRate("cpu-5900X", "1 core", chains, steps, target_weight,
                         elapsed, sum(weights) / len(weights), max(weights))


def one_sm(steps: int, target_weight: int, chains: int = THREADS_PER_SM) -> DecoupledRate:
    if not CUDA_AVAILABLE:
        raise DecoupledError("CUDA unavailable")
    words = 1 << SCRATCH_BITS
    required_mib = (chains * words * 8 + N_COLUMNS * 8 + chains * 16) / (1024 * 1024)
    check_gpu_budget(required_mib)
    rng = np.random.default_rng(11)
    scratch = cp.asarray(rng.integers(0, 2 ** 63, size=chains * words, dtype=np.uint64))
    H = cp.asarray(_parity_check())
    syn = cp.zeros(chains, dtype=cp.uint64)
    wts = cp.zeros(chains, dtype=cp.uint64)
    mask = np.uint64(words - 1)
    thr = np.uint64(selection_threshold(target_weight, steps))
    kernel = cp.RawModule(code=_DECOUPLED_KERNEL).get_function("decoupled_walk")

    def launch(nonce, n_steps):
        kernel((1,), (chains,), (scratch, H, np.uint64(words), mask,
                                 np.uint64(N_COLUMNS), np.uint64(nonce),
                                 np.int32(n_steps), thr, np.uint64(chains),
                                 syn, wts))
        cp.cuda.Device().synchronize()

    launch(1, min(8, steps))
    started = time.perf_counter()
    launch(0x1000, steps)
    elapsed = time.perf_counter() - started
    w = cp.asnumpy(wts)
    return DecoupledRate("cuda-rtx3090", "1 SM", chains, steps, target_weight,
                         elapsed, float(w.mean()), int(w.max()))


def decoupled_report(steps: int, target_weight: int) -> dict:
    core = one_core(steps, target_weight)
    sm = one_sm(steps, target_weight)
    per_unit = sm.chains_per_second / core.chains_per_second
    chip = (GPU_SMS * sm.chains_per_second) / (CPU_CORES * core.chains_per_second)
    return {
        "steps": steps, "target_weight": target_weight,
        "one_core": core.summary(), "one_sm": sm.summary(),
        "per_unit_gpu_over_cpu": round(per_unit, 3),
        "chip_gpu_over_cpu_bound": round(chip, 3),
        "weight_landed_near_target": abs(core.mean_weight - target_weight) < 0.25 * target_weight,
    }


__all__ = [
    "DecoupledError", "DecoupledRate", "USED_WORDS",
    "decoupled_report", "one_core", "one_sm", "selection_threshold",
]
