"""CUDA port of Phase A, to measure the GPU advantage instead of asserting it.

WHAT THIS ANSWERS
-----------------
`README.md` claims Phase A (candidate generation + syndrome computation) is
"GPU-favourable" and Phase B (table probing) is "CPU-favourable". Those were
claims about hardware, made without hardware measurement. This module ports
Phase A to CUDA so the speedup can be MEASURED. If the GPU wins Phase A by a
large factor, the design is GPU-dominated and cannot be CPU-only. If Phase B
does not move to the GPU as easily, the ratio between them is the real
cooperative-mining signal.

IMPLEMENTATION NOTES
--------------------
The kernel is written to be the shape an ASIC/FPGA would also like -- dense
int32 modular multiply-accumulate over a small matrix, one thread per
candidate, fully independent. That is deliberate: if this shape runs well on
a GPU it will run better on fixed-function silicon, and measuring it is how
we find out whether the work function is defensible at all.

Candidate generation uses a counter-mode hash (SplitMix64-style, since
SHAKE256 is impractical per-thread) so the GPU can derive candidates without
host transfer. That CHANGES the candidate distribution relative to
`pqmine.pow.generate_candidate`, so this module is a PERFORMANCE probe, not
a drop-in consensus component -- a real deployment needs the same PRF on
both sides. `equivalent_cpu_syndromes` recomputes the same candidates on the
CPU so the two paths can be checked against each other.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

try:
    import cupy as cp

    CUDA_AVAILABLE = True
except Exception:  # noqa: BLE001 - absence is a normal condition
    cp = None
    CUDA_AVAILABLE = False


class CudaUnavailableError(RuntimeError):
    """Raised when a CUDA path is requested without a working GPU."""


# One thread per candidate. Derives a sparse vector from a counter, then
# accumulates its syndrome. Matrix is read from global memory; for the small
# n*m used here it lands in L2/constant-ish cache and streams well.
_KERNEL_SOURCE = r"""
extern "C" __global__
void syndromes(
    const int* __restrict__ matrix,   // n x m, row-major
    const unsigned long long seed,
    const int n,
    const int m,
    const int q,
    const int sparsity,
    const int count,
    int* __restrict__ out_syndromes,  // count x n
    int* __restrict__ out_vectors     // count x sparsity (positions+signs)
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;

    // SplitMix64 on (seed ^ tid): cheap, decorrelated, no host transfer.
    unsigned long long z = seed + 0x9E3779B97F4A7C15ULL * (unsigned long long)(tid + 1);

    int acc[32];                       // n <= 32 for this probe
    for (int i = 0; i < n; ++i) acc[i] = 0;

    for (int s = 0; s < sparsity; ++s) {
        z += 0x9E3779B97F4A7C15ULL;
        unsigned long long x = z;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
        x = x ^ (x >> 31);

        int position = (int)(x % (unsigned long long)m);
        int sign = (x & 0x8000000000000000ULL) ? -1 : 1;

        out_vectors[tid * sparsity + s] = position * 2 + (sign > 0 ? 1 : 0);

        // acc += sign * matrix[:, position]
        for (int i = 0; i < n; ++i) {
            int v = matrix[i * m + position];
            acc[i] += sign * v;
        }
    }

    for (int i = 0; i < n; ++i) {
        int r = acc[i] % q;
        if (r < 0) r += q;
        out_syndromes[tid * n + i] = r;
    }
}
"""

_kernel_cache = {}


def _get_kernel():
    if not CUDA_AVAILABLE:
        raise CudaUnavailableError("cupy/CUDA is not available")
    if "syndromes" not in _kernel_cache:
        module = cp.RawModule(code=_KERNEL_SOURCE, options=("-std=c++14",))
        _kernel_cache["syndromes"] = module.get_function("syndromes")
    return _kernel_cache["syndromes"]


@dataclass(frozen=True)
class PhaseAMeasurement:
    """Timing for one Phase A batch on one device."""

    device: str
    candidates: int
    seconds: float
    n: int
    m: int

    @property
    def candidates_per_second(self) -> float:
        return self.candidates / self.seconds if self.seconds else 0.0

    @property
    def modular_mults(self) -> int:
        """Useful arithmetic performed: sparsity is folded into n adds."""
        return self.candidates * self.n * self.m

    def summary(self) -> dict:
        return {
            "device": self.device,
            "candidates": self.candidates,
            "seconds": round(self.seconds, 4),
            "candidates_per_second": round(self.candidates_per_second, 1),
        }


def gpu_phase_a(matrix_rows, seed: int, count: int, n: int, m: int, q: int,
                sparsity: int, threads: int = 256):
    """Run Phase A on the GPU. Returns (syndromes, vectors, seconds)."""
    kernel = _get_kernel()
    if n > 32:
        raise CudaUnavailableError("this probe kernel supports n <= 32")

    flat = cp.asarray(
        [value for row in matrix_rows for value in row], dtype=cp.int32
    )
    out_syndromes = cp.empty((count, n), dtype=cp.int32)
    out_vectors = cp.empty((count, sparsity), dtype=cp.int32)

    blocks = (count + threads - 1) // threads
    cp.cuda.Stream.null.synchronize()
    started = time.perf_counter()
    kernel(
        (blocks,), (threads,),
        (flat, cp.uint64(seed), cp.int32(n), cp.int32(m), cp.int32(q),
         cp.int32(sparsity), cp.int32(count), out_syndromes, out_vectors),
    )
    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - started
    return out_syndromes, out_vectors, elapsed


def equivalent_cpu_syndromes(matrix_rows, seed: int, count: int, n: int,
                             m: int, q: int, sparsity: int):
    """Recompute the SAME candidates on the CPU, for cross-validation.

    Mirrors the kernel's SplitMix64 derivation exactly. If this disagrees
    with the GPU output, the kernel is wrong and any speedup is meaningless.
    """
    mask = (1 << 64) - 1
    golden = 0x9E3779B97F4A7C15
    syndromes = []
    vectors = []
    for tid in range(count):
        z = (seed + golden * (tid + 1)) & mask
        acc = [0] * n
        positions = []
        for _ in range(sparsity):
            z = (z + golden) & mask
            x = z
            x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & mask
            x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & mask
            x = x ^ (x >> 31)
            position = x % m
            sign = -1 if (x & 0x8000000000000000) else 1
            positions.append(position * 2 + (1 if sign > 0 else 0))
            for i in range(n):
                acc[i] += sign * matrix_rows[i][position]
        syndromes.append(tuple(value % q for value in acc))
        vectors.append(tuple(positions))
    return syndromes, vectors


def cpu_phase_a_timed(matrix_rows, seed: int, count: int, n: int, m: int,
                      q: int, sparsity: int) -> PhaseAMeasurement:
    started = time.perf_counter()
    equivalent_cpu_syndromes(matrix_rows, seed, count, n, m, q, sparsity)
    elapsed = time.perf_counter() - started
    return PhaseAMeasurement("cpu-python", count, elapsed, n, m)


def gpu_phase_a_timed(matrix_rows, seed: int, count: int, n: int, m: int,
                      q: int, sparsity: int) -> PhaseAMeasurement:
    _, _, elapsed = gpu_phase_a(
        matrix_rows, seed, count, n, m, q, sparsity
    )
    return PhaseAMeasurement("cuda-rtx3090", count, elapsed, n, m)


def verify_kernel_matches_cpu(matrix_rows, seed: int, n: int, m: int, q: int,
                              sparsity: int, count: int = 4096) -> dict:
    """Cross-validate GPU against CPU on the same candidates."""
    gpu_syndromes, gpu_vectors, _ = gpu_phase_a(
        matrix_rows, seed, count, n, m, q, sparsity
    )
    cpu_syndromes, cpu_vectors = equivalent_cpu_syndromes(
        matrix_rows, seed, count, n, m, q, sparsity
    )
    gpu_s = cp.asnumpy(gpu_syndromes).tolist()
    gpu_v = cp.asnumpy(gpu_vectors).tolist()

    syndrome_matches = sum(
        1 for a, b in zip(gpu_s, cpu_syndromes) if tuple(a) == tuple(b)
    )
    vector_matches = sum(
        1 for a, b in zip(gpu_v, cpu_vectors) if tuple(a) == tuple(b)
    )
    return {
        "checked": count,
        "syndrome_matches": syndrome_matches,
        "vector_matches": vector_matches,
        "all_match": syndrome_matches == count and vector_matches == count,
    }


__all__ = [
    "CUDA_AVAILABLE",
    "CudaUnavailableError",
    "PhaseAMeasurement",
    "cpu_phase_a_timed",
    "equivalent_cpu_syndromes",
    "gpu_phase_a",
    "gpu_phase_a_timed",
    "verify_kernel_matches_cpu",
]
