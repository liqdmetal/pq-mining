"""Fraction test: 1 CPU core vs 1 GPU SM, so measurement never disturbs qwen.

WHY FRACTIONS
-------------
Full-chip comparisons need the whole GPU, which means unloading the
resident LLM every time. That is slow and easy to forget. A per-unit
comparison uses one execution unit on each device and scales linearly:

    CPU chip rate ~= 12 cores x per-core rate
    GPU chip rate ~= 82 SMs  x per-SM rate       (RTX 3090 has 82 SMs)

so the chip-level GPU/CPU ratio is  (82 * sm_rate) / (12 * core_rate).
The per-unit run needs one small scratch allocation (a few MiB) and one
core, so it can run at any time beside the LLM.

WHAT IS THE FAIR UNIT
---------------------
NOT one GPU thread. One thread leaves 81 SMs and ~2000 threads of its own
SM idle and measures only latency -- we already have that figure (9.86x
slower). The unit that corresponds to "one CPU core, fully used" is ONE
SM, fully occupied: launch exactly one block sized to fill the SM's
resident-thread capacity (1024 threads on Ampere), each running its own
chain. That is a real 1/82 of the GPU, the same way one core is 1/12 of
the CPU.

SCALING LIMITS
--------------
The chip ratio is a linear extrapolation, not a measured whole-chip ratio
or an upper bound on that ratio. Even if each device's absolute projection
were an upper bound, dividing two upper bounds does not bound their ratio.
Cache/memory contention, clocks, occupancy, power and the resident workload
can affect the two devices differently. This synthetic benchmark also does
not measure a complete miner emitting canonical verifier-valid witnesses.
"""

from __future__ import annotations

import os
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
from .distinct_composed import _DISTINCT_KERNEL, _cpu_distinct_one, _parity_check


class FractionError(RuntimeError):
    """Measurement could not run."""


CPU_CORES = 12                  # 5900X physical cores
GPU_SMS = 82                    # RTX 3090
THREADS_PER_SM = 1024           # Ampere max resident threads per SM


@dataclass(frozen=True)
class UnitRate:
    device: str
    unit: str
    chains: int
    steps: int
    seconds: float

    @property
    def chains_per_second(self) -> float:
        return self.chains / self.seconds if self.seconds > 0 else 0.0

    def summary(self) -> dict:
        return {
            "device": self.device,
            "unit": self.unit,
            "chains": self.chains,
            "steps": self.steps,
            "seconds": round(self.seconds, 4),
            "chains_per_second": round(self.chains_per_second, 1),
        }


def one_core(steps: int, chains: int = 512) -> UnitRate:
    """ONE CPU core, serial chains. numba pinned to a single thread."""
    if not NUMBA_AVAILABLE:
        raise FractionError("numba unavailable")
    set_num_threads(1)
    words = 1 << SCRATCH_BITS
    rng = np.random.default_rng(11)
    scratch = rng.integers(0, 2 ** 63, size=chains * words, dtype=np.uint64)
    H = _parity_check()
    used = np.zeros((N_COLUMNS + 63) // 64, dtype=np.uint64)
    mask = np.uint64(words - 1)
    ncol = np.uint64(N_COLUMNS)
    # warm
    _cpu_distinct_one(scratch, np.uint64(0), H, ncol, mask, np.uint64(1),
                      min(8, steps), used)
    started = time.perf_counter()
    for i in range(chains):
        _cpu_distinct_one(scratch, np.uint64(i) * np.uint64(words), H, ncol,
                          mask, np.uint64(0x1000 + i), steps, used)
    elapsed = time.perf_counter() - started
    return UnitRate("cpu-5900X", "1 core", chains, steps, elapsed)


def one_sm(steps: int, chains: int = THREADS_PER_SM) -> UnitRate:
    """ONE SM, fully occupied: a single block of 1024 threads.

    Launching exactly one block guarantees the work lands on one SM. 1024
    threads is Ampere's resident-thread cap, so the SM is as busy as it
    can be -- the honest counterpart to a fully used CPU core.
    """
    if not CUDA_AVAILABLE:
        raise FractionError("CUDA unavailable")
    if chains > THREADS_PER_SM:
        raise FractionError(f"one block cannot exceed {THREADS_PER_SM} threads")
    words = 1 << SCRATCH_BITS
    required_mib = (chains * words * 8 + N_COLUMNS * 8 + chains * 16) / (1024 * 1024)
    check_gpu_budget(required_mib)              # ~8 MiB: trivially safe

    rng = np.random.default_rng(11)
    scratch = cp.asarray(rng.integers(0, 2 ** 63, size=chains * words, dtype=np.uint64))
    H = cp.asarray(_parity_check())
    out = cp.zeros(chains, dtype=cp.uint64)
    probes = cp.zeros(chains, dtype=cp.uint64)
    mask = np.uint64(words - 1)
    kernel = cp.RawModule(code=_DISTINCT_KERNEL).get_function("distinct_composed_walk")

    def launch(nonce, n_steps):
        # grid=(1,) block=(chains,): ONE block, ONE SM.
        kernel((1,), (chains,), (scratch, H, np.uint64(words), mask,
                                 np.uint64(N_COLUMNS), np.uint64(nonce),
                                 np.int32(n_steps), np.uint64(chains),
                                 out, probes))
        cp.cuda.Device().synchronize()

    launch(1, min(8, steps))
    started = time.perf_counter()
    launch(0x1000, steps)
    elapsed = time.perf_counter() - started
    return UnitRate("cuda-rtx3090", "1 SM (1024 threads)", chains, steps, elapsed)


def fraction_report(steps: int = 400) -> dict:
    """Per-unit rates and the chip-level ratio they imply."""
    core = one_core(steps)
    sm = one_sm(steps)
    per_unit_ratio = sm.chains_per_second / core.chains_per_second
    chip_ratio = (GPU_SMS * sm.chains_per_second) / (CPU_CORES * core.chains_per_second)
    verdict = (
        f"Linear extrapolation: {chip_ratio:.2f}x GPU/CPU; not a whole-chip "
        "measurement, hardware advantage bound, or consensus-miner throughput result."
    )

    return {
        "steps": steps,
        "one_core": core.summary(),
        "one_sm": sm.summary(),
        "per_unit_gpu_over_cpu": round(per_unit_ratio, 3),
        "scaling": {
            "cpu_cores": CPU_CORES,
            "gpu_sms": GPU_SMS,
            "chip_gpu_over_cpu_linear_projection": round(chip_ratio, 3),
            "is_measured_whole_chip_ratio": False,
            "is_hardware_advantage_bound": False,
            "note": (
                "linear per-unit scaling ignores shared L2/memory contention "
                "on both chips; dividing projected rates does not bound the "
                "actual ratio. Full valid-share benchmarks remain required"
            ),
        },
        "gpu_mib_touched": round(
            (THREADS_PER_SM * (1 << SCRATCH_BITS) * 8 + N_COLUMNS * 8) / (1024 * 1024), 1
        ),
        "verdict": verdict,
    }


__all__ = [
    "CPU_CORES",
    "GPU_SMS",
    "THREADS_PER_SM",
    "FractionError",
    "UnitRate",
    "fraction_report",
    "one_core",
    "one_sm",
]
