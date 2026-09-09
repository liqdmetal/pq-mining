"""CUDA port of the code-based PoW phases, to test the CPU/GPU split claim.

THE HYPOTHESIS BEING TESTED
---------------------------
The code-based design has two phases with OPPOSITE hardware preferences,
which is exactly the "both devices needed" shape the M-SIS design failed to
deliver:

  Phase A -- syndrome computation: XOR of w bit-packed columns.
      Pure ALU, no branches, fully independent per candidate.
      -> should favour the GPU heavily (thousands of cores, XOR is free)

  Phase B -- table build/probe: random access into a 2^((n-k)/2)-entry table.
      Dependent, latency-bound, irregular.
      -> should favour the CPU (bigger per-thread cache, ~80ns latency vs
         GPU's ~400-600ns for uncoalesced global access)

If that holds, neither device dominates: a GPU-only miner stalls on probes,
a CPU-only miner stalls on syndromes, and the efficient configuration uses
both. That is a STRONGER claim than the lattice design could make, where
Phase A was 97% of the work and a GPU simply won.

WHAT THIS MODULE MEASURES (not asserts)
---------------------------------------
  * gpu_syndrome_rate      -- Phase A on the RTX 3090
  * cpu_syndrome_rate      -- Phase A on the 5900X (numba, 24 threads)
  * gpu_probe_rate         -- Phase B random gather on the GPU
  * cpu_probe_rate         -- Phase B random gather on the CPU
and reports the ratio for EACH phase, so the cooperative claim either
survives contact with the hardware or it does not.

HONEST NOTE ON THE PROBE BENCHMARK
----------------------------------
A real miner's Phase B is a hash-table insert/lookup with collision
handling. Both benchmarks here use a plain open-addressed gather into a
power-of-two table, which is the memory-access CORE of that operation
without the bookkeeping. It measures the latency behaviour that decides the
CPU/GPU split; it is not a full hashtable implementation.
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


# RESOURCE DISCIPLINE (user directive, learned the hard way)
#
# This machine runs a local qwen model on the same RTX 3090, and the CPU
# must stay responsive. Two hard caps, applied as DEFAULTS so no call site
# has to remember them:
#
#   * CPU: 80% of logical cores (19 of 24 on the 5900X).
#   * GPU: benchmarks allocate at most GPU_BUDGET_MIB and refuse to run if
#     that would not fit in currently-free VRAM. An earlier version of this
#     module allocated an 8 GB table and had to be killed mid-run because
#     it was competing with the resident LLM. Do not remove this guard.
CPU_THREAD_FRACTION = 0.80
GPU_BUDGET_MIB = 512


def cpu_worker_threads(fraction: float = CPU_THREAD_FRACTION) -> int:
    """Logical cores to use: 80% by default, at least 1."""
    import os

    total = os.cpu_count() or 1
    return max(1, int(total * fraction))


def free_gpu_mib() -> int:
    """Currently-free VRAM in MiB, or -1 when CUDA is unavailable."""
    if not CUDA_AVAILABLE:
        return -1
    free_bytes, _total = cp.cuda.runtime.memGetInfo()
    return int(free_bytes // (1024 * 1024))


def check_gpu_budget(required_mib: float) -> None:
    """Refuse an allocation that would crowd the resident LLM.

    Raises rather than degrading silently: a benchmark that swaps against
    another process reports meaningless numbers anyway.
    """
    if required_mib > GPU_BUDGET_MIB:
        raise PortError(
            f"allocation {required_mib:.0f} MiB exceeds the "
            f"{GPU_BUDGET_MIB} MiB benchmark budget (a local LLM shares "
            "this GPU); lower table_bits or count"
        )
    available = free_gpu_mib()
    if available >= 0 and required_mib > available * 0.5:
        raise PortError(
            f"allocation {required_mib:.0f} MiB would use more than half of "
            f"the {available} MiB free VRAM; refusing to compete with the "
            "resident model"
        )


if NUMBA_AVAILABLE:
    try:
        set_num_threads(cpu_worker_threads())
    except Exception:                                # noqa: BLE001
        pass


class PortError(RuntimeError):
    """Raised when a required backend is unavailable."""


# ---------------------------------------------------------------- Phase A GPU

_SYNDROME_KERNEL = r"""
extern "C" __global__
void syndrome_batch(const unsigned long long* __restrict__ columns,
                    const unsigned int* __restrict__ supports,
                    unsigned long long* __restrict__ out,
                    const int count, const int weight)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= count) return;

    // XOR the w columns named by this candidate's support. This is the
    // ENTIRE work function -- contrast the M-SIS kernel, which did
    // n*m modular multiply-accumulates per candidate.
    unsigned long long acc = 0ULL;
    for (int j = 0; j < weight; ++j) {
        acc ^= columns[supports[idx * weight + j]];
    }
    out[idx] = acc;
}
"""

# --------------------------------------------------------------- Phase B GPU

_PROBE_KERNEL = r"""
extern "C" __global__
void table_probe(const unsigned long long* __restrict__ table,
                 const unsigned long long* __restrict__ keys,
                 unsigned int* __restrict__ hits,
                 const int count, const unsigned int mask)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= count) return;

    // Random-address gather: the access pattern that decides the CPU/GPU
    // split. Deliberately uncoalesced -- consecutive threads touch
    // unrelated cache lines, exactly as a syndrome-keyed table does.
    unsigned long long key = keys[idx];
    unsigned int slot = (unsigned int)(key ^ (key >> 29)) & mask;
    hits[idx] = (table[slot] == key) ? 1u : 0u;
}
"""


@dataclass
class PhaseRate:
    """One measurement: operations per second on one device."""

    device: str
    phase: str
    operations: int
    seconds: float

    @property
    def rate(self) -> float:
        return self.operations / self.seconds if self.seconds > 0 else 0.0

    def summary(self) -> dict:
        return {
            "device": self.device,
            "phase": self.phase,
            "operations": self.operations,
            "seconds": round(self.seconds, 4),
            "ops_per_second": round(self.rate, 1),
        }


def _require_cuda() -> None:
    if not CUDA_AVAILABLE:
        raise PortError("CuPy/CUDA unavailable")


def _require_numba() -> None:
    if not NUMBA_AVAILABLE:
        raise PortError("numba/numpy unavailable")


def gpu_syndrome_rate(columns, weight: int, count: int,
                      *, warmup: bool = True) -> PhaseRate:
    """Phase A on the GPU: XOR-reduce `weight` columns per candidate."""
    _require_cuda()
    module = cp.RawModule(code=_SYNDROME_KERNEL)
    kernel = module.get_function("syndrome_batch")

    d_columns = cp.asarray(columns, dtype=cp.uint64)
    rng = cp.random.default_rng(12345)
    supports = rng.integers(0, len(columns), size=count * weight,
                            dtype=cp.uint32)
    out = cp.zeros(count, dtype=cp.uint64)

    threads = 256
    blocks = (count + threads - 1) // threads
    args = (d_columns, supports, out, count, weight)

    if warmup:
        kernel((blocks,), (threads,), args)
        cp.cuda.Stream.null.synchronize()

    start = time.perf_counter()
    kernel((blocks,), (threads,), args)
    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - start
    return PhaseRate("cuda-rtx3090", "syndrome-xor", count, elapsed)


def gpu_probe_rate(table_bits: int, count: int,
                   *, warmup: bool = True) -> PhaseRate:
    """Phase B on the GPU: uncoalesced random gather.

    Budget-checked: table is 8 B/entry plus 12 B/probe of keys+hits.
    """
    _require_cuda()
    required_mib = ((1 << table_bits) * 8 + count * 12) / (1024 * 1024)
    check_gpu_budget(required_mib)

    module = cp.RawModule(code=_PROBE_KERNEL)
    kernel = module.get_function("table_probe")

    size = 1 << table_bits
    rng = cp.random.default_rng(999)
    table = rng.integers(0, 2 ** 63, size=size, dtype=cp.uint64)
    keys = rng.integers(0, 2 ** 63, size=count, dtype=cp.uint64)
    hits = cp.zeros(count, dtype=cp.uint32)

    threads = 256
    blocks = (count + threads - 1) // threads
    args = (table, keys, hits, count, cp.uint32(size - 1))

    if warmup:
        kernel((blocks,), (threads,), args)
        cp.cuda.Stream.null.synchronize()

    start = time.perf_counter()
    kernel((blocks,), (threads,), args)
    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - start
    return PhaseRate("cuda-rtx3090", "table-probe", count, elapsed)


if NUMBA_AVAILABLE:

    @njit(cache=True, parallel=True, nogil=True)
    def _cpu_syndrome(columns, supports, out, count, weight):   # pragma: no cover
        for i in prange(count):
            acc = np.uint64(0)
            for j in range(weight):
                acc ^= columns[supports[i * weight + j]]
            out[i] = acc

    @njit(cache=True, parallel=True, nogil=True)
    def _cpu_probe(table, keys, hits, count, mask):             # pragma: no cover
        for i in prange(count):
            key = keys[i]
            slot = (key ^ (key >> np.uint64(29))) & mask
            hits[i] = 1 if table[slot] == key else 0


def cpu_syndrome_rate(columns, weight: int, count: int,
                      *, warmup: bool = True) -> PhaseRate:
    """Phase A on the CPU: same algorithm, native, 80% of cores."""
    _require_numba()
    arr = np.asarray(columns, dtype=np.uint64)
    rng = np.random.default_rng(12345)
    supports = rng.integers(0, len(columns), size=count * weight,
                            dtype=np.uint32)
    out = np.zeros(count, dtype=np.uint64)

    if warmup:
        _cpu_syndrome(arr, supports[:weight * 64], out[:64], 64, weight)

    start = time.perf_counter()
    _cpu_syndrome(arr, supports, out, count, weight)
    elapsed = time.perf_counter() - start
    return PhaseRate("cpu-numba-24t", "syndrome-xor", count, elapsed)


def cpu_probe_rate(table_bits: int, count: int,
                   *, warmup: bool = True) -> PhaseRate:
    """Phase B on the CPU: same uncoalesced random gather."""
    _require_numba()
    size = 1 << table_bits
    rng = np.random.default_rng(999)
    table = rng.integers(0, 2 ** 63, size=size, dtype=np.uint64)
    keys = rng.integers(0, 2 ** 63, size=count, dtype=np.uint64)
    hits = np.zeros(count, dtype=np.uint32)
    mask = np.uint64(size - 1)

    if warmup:
        _cpu_probe(table, keys[:64], hits[:64], 64, mask)

    start = time.perf_counter()
    _cpu_probe(table, keys, hits, count, mask)
    elapsed = time.perf_counter() - start
    return PhaseRate("cpu-numba-24t", "table-probe", count, elapsed)


def verify_gpu_matches_cpu(columns, weight: int, count: int = 4096) -> dict:
    """Cross-validate the CUDA syndrome kernel against the CPU path.

    A speedup on wrong arithmetic is worthless, so this runs first.
    """
    _require_cuda()
    _require_numba()

    rng = np.random.default_rng(4242)
    supports = rng.integers(0, len(columns), size=count * weight,
                            dtype=np.uint32)
    host_columns = np.asarray(columns, dtype=np.uint64)

    expected = np.zeros(count, dtype=np.uint64)
    _cpu_syndrome(host_columns, supports, expected, count, weight)

    module = cp.RawModule(code=_SYNDROME_KERNEL)
    kernel = module.get_function("syndrome_batch")
    d_out = cp.zeros(count, dtype=cp.uint64)
    threads = 256
    blocks = (count + threads - 1) // threads
    kernel((blocks,), (threads,),
           (cp.asarray(host_columns), cp.asarray(supports), d_out,
            count, weight))
    cp.cuda.Stream.null.synchronize()

    matches = int((cp.asnumpy(d_out) == expected).sum())
    return {
        "checked": count,
        "matches": matches,
        "all_match": matches == count,
    }


def cooperative_verdict(gpu_syndrome: PhaseRate, cpu_syndrome: PhaseRate,
                        gpu_probe: PhaseRate, cpu_probe: PhaseRate) -> dict:
    """Does each device win a different phase? That is the whole question."""
    syndrome_ratio = gpu_syndrome.rate / cpu_syndrome.rate
    probe_ratio = gpu_probe.rate / cpu_probe.rate

    gpu_wins_syndrome = syndrome_ratio > 1.0
    cpu_wins_probe = probe_ratio < 1.0
    cooperative = gpu_wins_syndrome and cpu_wins_probe

    if cooperative:
        verdict = (
            "COOPERATIVE: the GPU wins syndrome generation and the CPU wins "
            "table probing, so neither device alone is efficient -- a "
            "GPU-only miner stalls on probes and a CPU-only miner stalls on "
            "syndromes. This is the 'both devices needed' property."
        )
    elif gpu_wins_syndrome and not cpu_wins_probe:
        verdict = (
            "GPU-DOMINATED: the GPU wins BOTH phases, so a GPU-only miner is "
            "optimal and the cooperative claim fails."
        )
    elif not gpu_wins_syndrome and cpu_wins_probe:
        verdict = (
            "CPU-DOMINATED: the CPU wins both phases, which satisfies the "
            "'CPU-only precedent' requirement but not the cooperative one."
        )
    else:
        verdict = "MIXED: each device wins the phase the other prefers."

    return {
        "syndrome_gpu_over_cpu": round(syndrome_ratio, 2),
        "probe_gpu_over_cpu": round(probe_ratio, 2),
        "gpu_wins_syndrome": gpu_wins_syndrome,
        "cpu_wins_probe": cpu_wins_probe,
        "cooperative": cooperative,
        "verdict": verdict,
    }


__all__ = [
    "CUDA_AVAILABLE",
    "NUMBA_AVAILABLE",
    "PhaseRate",
    "PortError",
    "cooperative_verdict",
    "cpu_probe_rate",
    "cpu_syndrome_rate",
    "gpu_probe_rate",
    "gpu_syndrome_rate",
    "verify_gpu_matches_cpu",
]
