"""Per-nonce mutable state: the knob that decides whether option B can be saved.

WHAT THE GPU-PENALTY MEASUREMENT PROVED
---------------------------------------
    single chain : GPU 9.86x SLOWER than CPU   (PoSME confirmed)
    aggregate    : GPU 16.9x FASTER than CPU   (option B FAILED)

The dependent walk defeats per-chain latency hiding, but mining runs one
chain per nonce and nonces are independent, so 16,384 GPU threads each
walk a chain and amortise the latency away. Latency-boundness protects a
chain; PoW is parallel across chains. That is the flaw in the premise.

THE FIX, AND WHY IT IS THE RIGHT ONE
------------------------------------
The shared read-only arena was the bug. It lets every GPU thread share
one 128 MiB block, so 16k chains cost 128 MiB total. RandomX survives
because each chain needs its OWN ~2 MB scratchpad that it MUTATES: 16k
chains would need 32 GB of fast per-thread memory, which no GPU has as
register/shared memory, so the GPU is forced to run chains out of slow
global memory a few hundred at a time. The lever is:

    per-chain private mutable working set  x  number of concurrent chains
    must exceed what the GPU can serve at full speed.

This module gives each chain a private scratchpad it READS AND WRITES,
sized by a parameter, and measures aggregate throughput on both devices
as the scratchpad grows. The prediction: CPU throughput barely moves (a
CPU core has ~1 MB L2 + big L3 per chain anyway), while GPU throughput
collapses once total scratchpad exceeds what its memory system can keep
hot for thousands of threads.

WHY MUTATION MATTERS (not just size)
------------------------------------
A read-only per-chain block can be shared or recomputed. A block the
chain WRITES to must be private -- two chains cannot share a buffer one of
them is mutating. So the scratchpad is written every step, and the next
address depends on a value written earlier in the SAME chain. This is
PoSME's "mutable arena" principle, which the shared-arena benchmark
skipped for convenience. That convenience was the whole bug.

RESOURCE DISCIPLINE
-------------------
Total GPU allocation = chains x scratchpad. Capped by check_gpu_budget so
this can never crowd the resident LLM; the sweep stops at the budget and
reports how many chains fit rather than allocating past it.
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

from .cuda_codebased import GPU_BUDGET_MIB, check_gpu_budget, cpu_worker_threads


class ScratchpadError(RuntimeError):
    """Measurement could not run."""


if NUMBA_AVAILABLE:
    set_num_threads(cpu_worker_threads())


# Each thread owns scratch[tid*words .. (tid+1)*words). Every step:
#   v = scratch[addr]            (read own scratchpad)
#   scratch[addr] = v ^ mix      (WRITE own scratchpad -- forces privacy)
#   addr = hash(v) & mask        (next address depends on the value read)
# The write is what makes the buffer un-shareable between chains.
_SCRATCH_KERNEL = r"""
extern "C" __global__
void scratch_walk(unsigned long long* scratch,
                  unsigned long long words_per_chain,
                  unsigned long long mask,
                  unsigned long long base_nonce,
                  int steps,
                  unsigned long long chains,
                  unsigned long long* out)
{
    unsigned long long tid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    // grid is rounded up to whole blocks; threads past the last chain must
    // not touch memory or they index off the end of the scratch buffer.
    if (tid >= chains) return;
    unsigned long long* mine = scratch + tid * words_per_chain;
    unsigned long long nonce = base_nonce + tid;
    unsigned long long addr = nonce & mask;
    unsigned long long acc = 0;
    for (int s = 0; s < steps; ++s) {
        unsigned long long v = mine[addr];
        unsigned long long x = v ^ nonce ^ (unsigned long long)s;
        x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
        x ^= x >> 27; x *= 0x94d049bb133111ebULL;
        x ^= x >> 31;
        mine[addr] = v ^ x;          // mutate: this chain's state changes
        acc ^= x;
        addr = x & mask;
    }
    out[tid] = acc;
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
    def _cpu_scratch_one(scratch, offset, mask, nonce, steps):
        addr = nonce & mask
        acc = np.uint64(0)
        for s in range(steps):
            idx = offset + addr
            v = scratch[idx]
            x = _mix(v ^ nonce ^ np.uint64(s))
            scratch[idx] = v ^ x
            acc ^= x
            addr = x & mask
        return acc

    @njit(parallel=True, cache=True)
    def _cpu_scratch_many(scratch, words, mask, base_nonce, steps, count, out):
        for i in prange(count):
            out[i] = _cpu_scratch_one(
                scratch, np.uint64(i) * words, mask,
                base_nonce + np.uint64(i), steps,
            )


@dataclass(frozen=True)
class ScratchRate:
    device: str
    chains: int
    steps: int
    scratch_kib_per_chain: float
    seconds: float

    @property
    def chains_per_second(self) -> float:
        return self.chains / self.seconds if self.seconds > 0 else 0.0

    def summary(self) -> dict:
        return {
            "device": self.device,
            "chains": self.chains,
            "scratch_kib_per_chain": self.scratch_kib_per_chain,
            "total_mib": round(self.chains * self.scratch_kib_per_chain / 1024, 1),
            "seconds": round(self.seconds, 4),
            "chains_per_second": round(self.chains_per_second, 1),
        }


def _words(scratch_bits: int) -> int:
    return 1 << scratch_bits


def cpu_scratch(scratch_bits: int, steps: int, chains: int) -> ScratchRate:
    """CPU: each chain mutates its own scratchpad. 80% of cores."""
    if not NUMBA_AVAILABLE:
        raise ScratchpadError("numba unavailable")
    words = _words(scratch_bits)
    rng = np.random.default_rng(11)
    scratch = rng.integers(0, 2 ** 63, size=chains * words, dtype=np.uint64)
    out = np.zeros(chains, dtype=np.uint64)
    mask = np.uint64(words - 1)
    _cpu_scratch_many(scratch, np.uint64(words), mask, np.uint64(1),
                      min(8, steps), min(32, chains), out)
    started = time.perf_counter()
    _cpu_scratch_many(scratch, np.uint64(words), mask, np.uint64(0x1000),
                      steps, chains, out)
    elapsed = time.perf_counter() - started
    return ScratchRate(f"cpu-numba-{cpu_worker_threads()}t", chains, steps,
                       words * 8 / 1024, elapsed)


def gpu_chains_that_fit(scratch_bits: int, budget_mib: float = GPU_BUDGET_MIB) -> int:
    """How many private scratchpads fit in the benchmark VRAM budget."""
    per_chain_bytes = _words(scratch_bits) * 8
    return int(budget_mib * 1024 * 1024 * 0.9 // per_chain_bytes)


def gpu_scratch(scratch_bits: int, steps: int, chains: int) -> ScratchRate:
    """GPU: one private scratchpad per thread. Budget-checked."""
    if not CUDA_AVAILABLE:
        raise ScratchpadError("CUDA unavailable")
    words = _words(scratch_bits)
    required_mib = (chains * words * 8 + chains * 8) / (1024 * 1024)
    check_gpu_budget(required_mib)

    rng = np.random.default_rng(11)
    scratch = cp.asarray(
        rng.integers(0, 2 ** 63, size=chains * words, dtype=np.uint64)
    )
    out = cp.zeros(chains, dtype=cp.uint64)
    mask = np.uint64(words - 1)
    module = cp.RawModule(code=_SCRATCH_KERNEL)
    kernel = module.get_function("scratch_walk")
    block = 256
    grid = (chains + block - 1) // block

    kernel((grid,), (block,), (scratch, np.uint64(words), mask, np.uint64(1),
                               np.int32(min(8, steps)), np.uint64(chains), out))
    cp.cuda.Device().synchronize()
    started = time.perf_counter()
    kernel((grid,), (block,), (scratch, np.uint64(words), mask,
                               np.uint64(0x1000), np.int32(steps),
                               np.uint64(chains), out))
    cp.cuda.Device().synchronize()
    elapsed = time.perf_counter() - started
    return ScratchRate("cuda-rtx3090", chains, steps, words * 8 / 1024, elapsed)


def scratchpad_sweep(scratch_bits_range=(10, 13, 16, 18, 20),
                     steps: int = 2048, cpu_chains: int = 2048) -> dict:
    """Aggregate throughput vs per-chain scratchpad size, both devices.

    The GPU runs as many chains as fit in the VRAM budget at each size
    (the honest maximum -- more chains would exceed the budget). The CPU
    runs a fixed count, since its per-chain working set is irrelevant to
    how many it can run: it just runs them 19 at a time.

    Throughput is compared per-chain-per-second so the differing chain
    counts do not distort the ratio.
    """
    rows = []
    for bits in scratch_bits_range:
        kib = _words(bits) * 8 / 1024
        gpu_n = gpu_chains_that_fit(bits)
        if gpu_n < 256:
            rows.append({
                "scratch_kib": kib, "gpu_chains_fit": gpu_n,
                "note": "fewer than 256 chains fit the VRAM budget; GPU "
                        "cannot fill even one block of warps -- skipped",
            })
            continue
        gpu_n = min(gpu_n, 65536)
        cpu = cpu_scratch(bits, steps, cpu_chains)
        gpu = gpu_scratch(bits, steps, gpu_n)
        ratio = gpu.chains_per_second / cpu.chains_per_second
        rows.append({
            "scratch_kib": kib,
            "gpu_chains_fit": gpu_n,
            "cpu_chains_per_s": round(cpu.chains_per_second, 1),
            "gpu_chains_per_s": round(gpu.chains_per_second, 1),
            "gpu_over_cpu": round(ratio, 3),
        })

    measured = [r for r in rows if "gpu_over_cpu" in r]
    if not measured:
        verdict = "no size fit the budget; nothing measured"
    else:
        first, last = measured[0], measured[-1]
        if last["gpu_over_cpu"] < 1.0:
            verdict = (
                f"CROSSOVER REACHED: at {last['scratch_kib']:.0f} KiB/chain "
                f"the CPU wins ({1/last['gpu_over_cpu']:.2f}x). Private "
                "mutable state is the lever that saves option B."
            )
        elif last["gpu_over_cpu"] < first["gpu_over_cpu"] / 2:
            verdict = (
                f"TRENDING: GPU advantage fell from {first['gpu_over_cpu']:.1f}x "
                f"to {last['gpu_over_cpu']:.1f}x as scratchpad grew. The "
                "lever works; larger scratchpads (beyond this VRAM budget) "
                "would likely cross over."
            )
        else:
            verdict = (
                f"NO EFFECT: GPU advantage {first['gpu_over_cpu']:.1f}x -> "
                f"{last['gpu_over_cpu']:.1f}x. Private state alone does not "
                "rescue option B at these sizes."
            )
    return {"steps": steps, "rows": rows, "verdict": verdict}


__all__ = [
    "ScratchRate",
    "ScratchpadError",
    "cpu_scratch",
    "gpu_chains_that_fit",
    "gpu_scratch",
    "scratchpad_sweep",
]
