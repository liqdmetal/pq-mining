"""Chitin-PoW v2: L1-sized mutable scratchpad driving a code-based syndrome walk.

WHAT THE MEASUREMENTS TAUGHT, IN ORDER
--------------------------------------
1. Lattice sieve, independent probes      -> GPU 3.15x   (arithmetic-bound)
2. Code-based sieve, independent probes   -> GPU 6-9x    (bandwidth-bound)
3. Dependent walk, SHARED read-only arena -> GPU 16.9x   (parallel across nonces)
4. Dependent walk, PRIVATE MUTABLE state:
       1-8 KiB/chain  (fits L1)  -> CPU 2-3x
       64+ KiB/chain  (spills)   -> GPU 3x

The lever is not latency and it is not size. It is a MUTABLE working set
small enough to live in a CPU core's L1 (~32 KB) and large enough that a
GPU thread cannot keep it in registers/shared memory. Mutation is what
makes it un-shareable across the GPU's thousands of threads; L1-fit is
what lets the CPU service it at ~1 ns instead of ~90 ns. Large state
spills BOTH devices to DRAM, where GPU bandwidth wins -- which is why
RandomX sizes its hot scratchpad to L1/L2, not to gigabytes.

THE COMPOSITION
---------------
Each chain owns an 8 KiB mutable scratchpad (1024 x u64). Each step:

    v      = scratch[addr]                 read own state
    x      = mix(v ^ nonce ^ step)         data-dependent
    scratch[addr] = v ^ x                  MUTATE own state (un-shareable)
    column = x mod n_columns               select a parity-check column
    syndrome ^= H[column]                  accumulate SDP syndrome
    addr   = x & (words-1)                 next address depends on x

After K steps the accumulated syndrome must match the header-derived
target: He^T = s for the weight-K error e whose support is the columns
visited. That is the syndrome decoding problem -- Classic McEliece's
assumption, the most cryptanalytically stable PQ family -- and it is what
makes a share a PQ proof rather than a hash preimage.

The scratchpad walk is the SEARCH (CPU-favoured); the syndrome check is the
ACCEPTANCE (post-quantum). Neither alone gives both properties; this
module is the composition, and the test is whether the CPU advantage from
measurement 4 SURVIVES adding the syndrome accumulation.

WHAT COULD GO WRONG
-------------------
* The H[column] lookup is a second memory stream. H for n=512 columns of
  u64 is 4 KiB -- it fits alongside the scratchpad in L1, so it should be
  cheap on the CPU. On the GPU it is a shared read-only table, which the
  GPU handles well. If the syndrome stream dominates, the GPU could claw
  back the advantage. Measured below, not assumed.
* Distinct-column selection (from distinct_walk.py) is NOT applied here.
  Over F_2 a column visited twice cancels, which the earlier audit showed
  collapses effective weight. That fix costs a used-set; adding it is the
  next step once this composition's balance is known.
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


class ComposedError(RuntimeError):
    """Measurement could not run."""


if NUMBA_AVAILABLE:
    set_num_threads(cpu_worker_threads())


SCRATCH_BITS = 10           # 1024 x u64 = 8 KiB: measured CPU-win regime
N_COLUMNS = 512             # parity-check columns; H = 4 KiB, fits L1 too


_COMPOSED_KERNEL = r"""
extern "C" __global__
void composed_walk(unsigned long long* scratch,
                   const unsigned long long* H,
                   unsigned long long words_per_chain,
                   unsigned long long mask,
                   unsigned long long n_columns,
                   unsigned long long base_nonce,
                   int steps,
                   unsigned long long chains,
                   unsigned long long* syndromes)
{
    unsigned long long tid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (tid >= chains) return;
    unsigned long long* mine = scratch + tid * words_per_chain;
    unsigned long long nonce = base_nonce + tid;
    unsigned long long addr = nonce & mask;
    unsigned long long syn = 0;
    for (int s = 0; s < steps; ++s) {
        unsigned long long v = mine[addr];
        unsigned long long x = v ^ nonce ^ (unsigned long long)s;
        x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
        x ^= x >> 27; x *= 0x94d049bb133111ebULL;
        x ^= x >> 31;
        mine[addr] = v ^ x;
        syn ^= H[x % n_columns];
        addr = x & mask;
    }
    syndromes[tid] = syn;
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
    def _cpu_composed_one(scratch, offset, H, n_columns, mask, nonce, steps):
        addr = nonce & mask
        syn = np.uint64(0)
        for s in range(steps):
            idx = offset + addr
            v = scratch[idx]
            x = _mix(v ^ nonce ^ np.uint64(s))
            scratch[idx] = v ^ x
            syn ^= H[x % n_columns]
            addr = x & mask
        return syn

    @njit(parallel=True, cache=True)
    def _cpu_composed_many(scratch, words, H, n_columns, mask,
                           base_nonce, steps, count, out):
        for i in prange(count):
            out[i] = _cpu_composed_one(
                scratch, np.uint64(i) * words, H, n_columns, mask,
                base_nonce + np.uint64(i), steps,
            )


@dataclass(frozen=True)
class ComposedRate:
    device: str
    chains: int
    steps: int
    seconds: float

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
        }


def _parity_check(seed: int = 3) -> "np.ndarray":
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2 ** 63, size=N_COLUMNS, dtype=np.uint64)


def cpu_composed(steps: int, chains: int,
                 scratch_bits: int = SCRATCH_BITS) -> ComposedRate:
    if not NUMBA_AVAILABLE:
        raise ComposedError("numba unavailable")
    words = 1 << scratch_bits
    rng = np.random.default_rng(11)
    scratch = rng.integers(0, 2 ** 63, size=chains * words, dtype=np.uint64)
    H = _parity_check()
    out = np.zeros(chains, dtype=np.uint64)
    mask = np.uint64(words - 1)
    ncol = np.uint64(N_COLUMNS)
    _cpu_composed_many(scratch, np.uint64(words), H, ncol, mask,
                       np.uint64(1), min(8, steps), min(32, chains), out)
    started = time.perf_counter()
    _cpu_composed_many(scratch, np.uint64(words), H, ncol, mask,
                       np.uint64(0x1000), steps, chains, out)
    elapsed = time.perf_counter() - started
    return ComposedRate(f"cpu-numba-{cpu_worker_threads()}t", chains, steps, elapsed)


def gpu_composed(steps: int, chains: int,
                 scratch_bits: int = SCRATCH_BITS) -> ComposedRate:
    if not CUDA_AVAILABLE:
        raise ComposedError("CUDA unavailable")
    words = 1 << scratch_bits
    required_mib = (chains * words * 8 + N_COLUMNS * 8 + chains * 8) / (1024 * 1024)
    check_gpu_budget(required_mib)

    rng = np.random.default_rng(11)
    scratch = cp.asarray(rng.integers(0, 2 ** 63, size=chains * words, dtype=np.uint64))
    H = cp.asarray(_parity_check())
    out = cp.zeros(chains, dtype=cp.uint64)
    mask = np.uint64(words - 1)
    module = cp.RawModule(code=_COMPOSED_KERNEL)
    kernel = module.get_function("composed_walk")
    block = 256
    grid = (chains + block - 1) // block
    args = (scratch, H, np.uint64(words), mask, np.uint64(N_COLUMNS),
            np.uint64(1), np.int32(min(8, steps)), np.uint64(chains), out)
    kernel((grid,), (block,), args)
    cp.cuda.Device().synchronize()
    started = time.perf_counter()
    args = (scratch, H, np.uint64(words), mask, np.uint64(N_COLUMNS),
            np.uint64(0x1000), np.int32(steps), np.uint64(chains), out)
    kernel((grid,), (block,), args)
    cp.cuda.Device().synchronize()
    elapsed = time.perf_counter() - started
    return ComposedRate("cuda-rtx3090", chains, steps, elapsed)


def composed_report(steps: int = 2048, cpu_chains: int = 4096,
                    gpu_chains: int = 58982) -> dict:
    """Does the CPU advantage survive adding the syndrome stream?

    Baseline from the scratchpad sweep at 8 KiB: CPU 1,076,054 ch/s vs GPU
    522,961 ch/s = 0.49x (CPU wins 2.05x). If the composed ratio stays near
    0.5x the syndrome accumulation is free; if it drifts toward 1x+ the
    second memory stream handed the GPU its advantage back.
    """
    cpu = cpu_composed(steps, cpu_chains)
    gpu = gpu_composed(steps, gpu_chains)
    ratio = gpu.chains_per_second / cpu.chains_per_second
    baseline_ratio = 522961 / 1076054

    if ratio < 1.0:
        verdict = (
            f"CPU WINS by {1/ratio:.2f}x WITH the syndrome stream "
            f"(baseline without it: {1/baseline_ratio:.2f}x). The composition "
            "keeps the CPU advantage: PQ acceptance + CPU-favoured search."
        )
    else:
        verdict = (
            f"GPU WINS by {ratio:.2f}x once the syndrome stream is added "
            f"(baseline without it: CPU {1/baseline_ratio:.2f}x). The H "
            "lookup handed the advantage back; the composition fails."
        )

    return {
        "scratch_kib_per_chain": (1 << SCRATCH_BITS) * 8 / 1024,
        "parity_check_kib": N_COLUMNS * 8 / 1024,
        "steps": steps,
        "cpu": cpu.summary(),
        "gpu": gpu.summary(),
        "gpu_over_cpu": round(ratio, 3),
        "baseline_gpu_over_cpu_no_syndrome": round(baseline_ratio, 3),
        "syndrome_stream_cost": round(ratio / baseline_ratio, 3),
        "verdict": verdict,
    }


__all__ = [
    "ComposedError",
    "ComposedRate",
    "N_COLUMNS",
    "SCRATCH_BITS",
    "composed_report",
    "cpu_composed",
    "gpu_composed",
]
