"""GPU penalty on the latency-bound distinct walk: the test that decides option B.

THE CLAIM UNDER TEST
--------------------
PoSME reports GPUs are 14-19x SLOWER than a consumer CPU at latency-bound
pointer chasing. Our lattice and code-based designs both LOST to the GPU
(3.15x and 6-9x) because their probes were INDEPENDENT -- the GPU hid
latency behind thousands of warps. The distinct walk makes each step
depend on the previous one's value, so there is nothing to hide latency
behind. If PoSME is right, the GPU should now LOSE.

This measures it on the same RTX 3090 that beat the earlier designs.

WHY A SINGLE GPU THREAD IS THE HONEST COMPARISON
------------------------------------------------
A miner runs many INDEPENDENT walks (one per nonce). So the fair GPU
strategy is one walk per thread, thousands of threads, and the relevant
metric is aggregate walks/second, not single-walk latency. That is the
number that decides mining economics. We measure BOTH:

  * single-walk latency (ns/step): how long one dependent chain takes.
    The GPU will be terrible here -- a single GPU thread has ~400-600 ns
    DRAM latency vs ~90 ns on the CPU.
  * aggregate throughput (walks/s) with many parallel walks: the GPU's
    thousands of threads amortise latency across independent chains.

The design goal is that aggregate GPU throughput does NOT exceed CPU
throughput by more than the GPU's power/cost ratio. If GPU wins aggregate
by 10x, it's a GPU coin. If ~1x or the CPU wins, it's a CPU coin.

RESOURCE DISCIPLINE
-------------------
Arena sizes are capped so the GPU allocation stays under the shared-LLM
budget (check_gpu_budget). A 128 MiB arena x 1 is fine; we do NOT
allocate one arena per thread -- all threads walk the same arena, which is
also what a real miner does (one arena per header, many nonces).
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


class PenaltyError(RuntimeError):
    """Measurement could not run."""


if NUMBA_AVAILABLE:
    set_num_threads(cpu_worker_threads())


# The dependent walk, identical on both devices. Each step:
#   value = arena[addr]; addr = hash(value ^ nonce ^ step) & mask
# The read of arena[addr] must complete before the next addr is known.
# This is the latency-bound chain; nothing can be prefetched.
_WALK_KERNEL = r"""
extern "C" __global__
void dependent_walk(const unsigned long long* arena,
                    unsigned long long mask,
                    unsigned long long base_nonce,
                    int steps,
                    unsigned long long* out)
{
    unsigned long long tid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    unsigned long long nonce = base_nonce + tid;
    unsigned long long addr = nonce & mask;
    unsigned long long acc = 0;
    for (int s = 0; s < steps; ++s) {
        unsigned long long v = arena[addr];
        acc ^= v;
        // splitmix64 finaliser: next address depends on the value just read
        unsigned long long x = v ^ nonce ^ (unsigned long long)s;
        x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
        x ^= x >> 27; x *= 0x94d049bb133111ebULL;
        x ^= x >> 31;
        addr = x & mask;
    }
    out[tid] = acc;
}
"""


if NUMBA_AVAILABLE:
    @njit(cache=True, nogil=True)
    def _splitmix(x):
        x ^= x >> np.uint64(30)
        x *= np.uint64(0xbf58476d1ce4e5b9)
        x ^= x >> np.uint64(27)
        x *= np.uint64(0x94d049bb133111eb)
        x ^= x >> np.uint64(31)
        return x

    @njit(cache=True, nogil=True)
    def _cpu_walk_one(arena, mask, nonce, steps):
        addr = nonce & mask
        acc = np.uint64(0)
        for s in range(steps):
            v = arena[addr]
            acc ^= v
            addr = _splitmix(v ^ nonce ^ np.uint64(s)) & mask
        return acc

    @njit(parallel=True, cache=True)
    def _cpu_walk_many(arena, mask, base_nonce, steps, count, out):
        for i in prange(count):
            out[i] = _cpu_walk_one(arena, mask, base_nonce + np.uint64(i), steps)


@dataclass(frozen=True)
class WalkRate:
    device: str
    walks: int
    steps: int
    seconds: float

    @property
    def walks_per_second(self) -> float:
        return self.walks / self.seconds if self.seconds > 0 else 0.0

    @property
    def ns_per_step_per_walk(self) -> float:
        """Latency of ONE chain, if walks were serial. For the GPU with many
        threads this is NOT the wall-clock per step -- it's the amortised
        figure. Reported for the single-thread case only."""
        total_steps = self.walks * self.steps
        return self.seconds / total_steps * 1e9 if total_steps else 0.0

    def summary(self) -> dict:
        return {
            "device": self.device,
            "walks": self.walks,
            "steps": self.steps,
            "seconds": round(self.seconds, 4),
            "walks_per_second": round(self.walks_per_second, 1),
            "amortised_ns_per_step": round(self.ns_per_step_per_walk, 1),
        }


def _make_arena_cpu(arena_bits: int, seed: int = 7):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2 ** 63, size=1 << arena_bits, dtype=np.uint64)


def cpu_walks(arena_bits: int, steps: int, walks: int) -> WalkRate:
    """Many independent dependent-walks on the CPU, 80% of cores."""
    if not NUMBA_AVAILABLE:
        raise PenaltyError("numba unavailable")
    arena = _make_arena_cpu(arena_bits)
    mask = np.uint64((1 << arena_bits) - 1)
    out = np.zeros(walks, dtype=np.uint64)
    # warm JIT
    _cpu_walk_many(arena, mask, np.uint64(1), min(16, steps), min(64, walks), out)
    started = time.perf_counter()
    _cpu_walk_many(arena, mask, np.uint64(0x1000), steps, walks, out)
    elapsed = time.perf_counter() - started
    return WalkRate(f"cpu-numba-{cpu_worker_threads()}t", walks, steps, elapsed)


def cpu_single_thread_latency(arena_bits: int, steps: int, walks: int = 64) -> WalkRate:
    """One thread, serial walks: the true single-chain latency."""
    if not NUMBA_AVAILABLE:
        raise PenaltyError("numba unavailable")
    arena = _make_arena_cpu(arena_bits)
    mask = np.uint64((1 << arena_bits) - 1)
    _cpu_walk_one(arena, mask, np.uint64(1), min(16, steps))
    started = time.perf_counter()
    for i in range(walks):
        _cpu_walk_one(arena, mask, np.uint64(0x2000 + i), steps)
    elapsed = time.perf_counter() - started
    return WalkRate("cpu-single-thread", walks, steps, elapsed)


def gpu_walks(arena_bits: int, steps: int, walks: int) -> WalkRate:
    """Many independent dependent-walks on the GPU, one per thread.

    Budget-checked: one arena shared by all threads plus the output
    array. A 2^24 arena is 128 MiB; with the 512 MiB budget that is the
    practical ceiling here.
    """
    if not CUDA_AVAILABLE:
        raise PenaltyError("CUDA unavailable")
    required_mib = ((1 << arena_bits) * 8 + walks * 8) / (1024 * 1024)
    check_gpu_budget(required_mib)

    rng = np.random.default_rng(7)
    arena_host = rng.integers(0, 2 ** 63, size=1 << arena_bits, dtype=np.uint64)
    arena = cp.asarray(arena_host)
    out = cp.zeros(walks, dtype=cp.uint64)
    mask = np.uint64((1 << arena_bits) - 1)

    module = cp.RawModule(code=_WALK_KERNEL)
    kernel = module.get_function("dependent_walk")
    block = 256
    grid = (walks + block - 1) // block

    # warm
    kernel((grid,), (block,), (arena, mask, np.uint64(1), np.int32(min(16, steps)), out))
    cp.cuda.Device().synchronize()

    started = time.perf_counter()
    kernel((grid,), (block,), (arena, mask, np.uint64(0x1000), np.int32(steps), out))
    cp.cuda.Device().synchronize()
    elapsed = time.perf_counter() - started
    return WalkRate("cuda-rtx3090", walks, steps, elapsed)


def gpu_single_thread_latency(arena_bits: int, steps: int) -> WalkRate:
    """ONE GPU thread walking alone: the raw single-chain DRAM latency on
    the GPU, with no parallelism to hide it. This is the number PoSME's
    14-19x refers to."""
    if not CUDA_AVAILABLE:
        raise PenaltyError("CUDA unavailable")
    required_mib = ((1 << arena_bits) * 8 + 8) / (1024 * 1024)
    check_gpu_budget(required_mib)

    rng = np.random.default_rng(7)
    arena = cp.asarray(rng.integers(0, 2 ** 63, size=1 << arena_bits, dtype=np.uint64))
    out = cp.zeros(1, dtype=cp.uint64)
    mask = np.uint64((1 << arena_bits) - 1)
    module = cp.RawModule(code=_WALK_KERNEL)
    kernel = module.get_function("dependent_walk")
    kernel((1,), (1,), (arena, mask, np.uint64(1), np.int32(min(16, steps)), out))
    cp.cuda.Device().synchronize()
    started = time.perf_counter()
    kernel((1,), (1,), (arena, mask, np.uint64(0x3000), np.int32(steps), out))
    cp.cuda.Device().synchronize()
    elapsed = time.perf_counter() - started
    return WalkRate("cuda-single-thread", 1, steps, elapsed)


def penalty_report(arena_bits: int = 24, steps: int = 4096,
                   parallel_walks: int = 16384) -> dict:
    """Both comparisons, with the verdict that matters for mining."""
    cpu_lat = cpu_single_thread_latency(arena_bits, steps, walks=32)
    gpu_lat = gpu_single_thread_latency(arena_bits, steps)
    cpu_agg = cpu_walks(arena_bits, steps, parallel_walks)
    gpu_agg = gpu_walks(arena_bits, steps, parallel_walks)

    latency_ratio = gpu_lat.ns_per_step_per_walk / cpu_lat.ns_per_step_per_walk
    throughput_ratio = gpu_agg.walks_per_second / cpu_agg.walks_per_second

    if throughput_ratio < 1.0:
        verdict = (
            f"CPU WINS aggregate throughput by {1/throughput_ratio:.2f}x: the "
            "dependent walk defeats GPU latency-hiding even with "
            f"{parallel_walks} parallel chains. This is a CPU-favoured PoW."
        )
    elif throughput_ratio < 3.0:
        verdict = (
            f"NEAR PARITY: GPU leads aggregate by only {throughput_ratio:.2f}x, "
            "within the GPU's power/cost premium. Cooperative or CPU-viable."
        )
    else:
        verdict = (
            f"GPU WINS aggregate by {throughput_ratio:.2f}x: parallel chains "
            "still amortise latency enough that this is a GPU coin. Option B "
            "does not achieve its goal at these parameters."
        )

    return {
        "arena_mib": (1 << arena_bits) * 8 / (1024 * 1024),
        "steps_per_walk": steps,
        "single_chain_latency": {
            "cpu": cpu_lat.summary(),
            "gpu": gpu_lat.summary(),
            "gpu_slower_by": round(latency_ratio, 2),
            "posme_reported_range": "14-19x",
        },
        "aggregate_throughput": {
            "cpu": cpu_agg.summary(),
            "gpu": gpu_agg.summary(),
            "gpu_over_cpu": round(throughput_ratio, 3),
        },
        "verdict": verdict,
        "note": (
            "Single-chain latency is where PoSME's 14-19x lives and the GPU "
            "should lose badly. Aggregate throughput is what mining pays "
            "for; the GPU's thousands of threads each run a chain, so the "
            "question is whether latency-hiding across INDEPENDENT chains "
            "recovers the deficit. That is the number that decides option B."
        ),
    }


__all__ = [
    "PenaltyError",
    "WalkRate",
    "cpu_single_thread_latency",
    "cpu_walks",
    "gpu_single_thread_latency",
    "gpu_walks",
    "penalty_report",
]
