"""Head-to-head Phase A benchmark: RTX 3090 vs 5900X, honestly baselined.

    python tools/bench_gpu_vs_cpu.py

Reports candidates/sec for the same algorithm on:
  * cpu-python           (reference, unfair -- shown only to expose the trap)
  * cpu-numba-parallel   (native, all 24 threads -- the HONEST baseline)
  * cuda-rtx3090         (the CUDA port)

and then measures Phase B (table probing) which does NOT port to the GPU
easily, so the ratio between them is the real cooperative-mining signal.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pqmine import PoWParameters, derive_matrix
from pqmine.cpu_baseline import NUMBA_AVAILABLE, native_cpu_phase_a
from pqmine.cuda_phase_a import (
    CUDA_AVAILABLE,
    cpu_phase_a_timed,
    gpu_phase_a,
    gpu_phase_a_timed,
    verify_kernel_matches_cpu,
)

COUNT = 8 << 20  # 8.4M candidates


def banner(text):
    print()
    print("=" * 72)
    print(text)
    print("=" * 72)


def phase_b_probe_rate(syndromes, params) -> float:
    """Measure table-probe throughput -- the memory-bound phase."""
    table: dict = {}
    rows = [tuple(row) for row in syndromes]
    started = time.perf_counter()
    for index, key in enumerate(rows):
        bucket = table.get(key)
        if bucket is None:
            table[key] = [index]
        else:
            bucket.append(index)
    elapsed = time.perf_counter() - started
    return len(rows) / elapsed if elapsed else 0.0


def main() -> int:
    banner("CHITIN-PoW  PHASE A: GPU vs CPU")
    params = PoWParameters(n=8, m=64, q=12289, beta=4, sparsity=8)
    matrix = derive_matrix(b"bench-header", params)
    print(f"  instance: n={params.n} m={params.m} q={params.q} "
          f"sparsity={params.sparsity}")
    print(f"  candidates per run: {COUNT:,}")
    print(f"  CUDA available: {CUDA_AVAILABLE}   numba: {NUMBA_AVAILABLE}")

    if CUDA_AVAILABLE:
        check = verify_kernel_matches_cpu(
            matrix, 1, params.n, params.m, params.q, params.sparsity, 4096
        )
        print(f"  kernel cross-validation: {check['syndrome_matches']}/"
              f"{check['checked']} exact -> {check['all_match']}")
        if not check["all_match"]:
            print("  ABORT: kernel disagrees with the CPU reference")
            return 1

    results = {}

    banner("1. cpu-python (UNFAIR baseline -- measures Python, not silicon)")
    small = 20_000
    cpu_py = cpu_phase_a_timed(
        matrix, 1, small, params.n, params.m, params.q, params.sparsity
    )
    results["cpu-python"] = cpu_py.candidates_per_second
    print(f"  {cpu_py.candidates_per_second:>18,.0f} cand/s  "
          f"({small:,} candidates, 1 thread)")

    if NUMBA_AVAILABLE:
        banner("2. cpu-numba-parallel (HONEST baseline: native, 24 threads)")
        _, cpu_native = native_cpu_phase_a(
            matrix, 1, COUNT, params.n, params.m, params.q, params.sparsity
        )
        results["cpu-native"] = cpu_native.candidates_per_second
        print(f"  {cpu_native.candidates_per_second:>18,.0f} cand/s  "
              f"({cpu_native.threads} threads)")
        print(f"  python was {cpu_native.candidates_per_second / results['cpu-python']:,.0f}x "
              "slower -- this is why the first comparison was meaningless")

    if CUDA_AVAILABLE:
        banner("3. cuda-rtx3090")
        gpu = gpu_phase_a_timed(
            matrix, 1, COUNT, params.n, params.m, params.q, params.sparsity
        )
        results["gpu"] = gpu.candidates_per_second
        print(f"  {gpu.candidates_per_second:>18,.0f} cand/s")

    banner("VERDICT: Phase A")
    if "gpu" in results and "cpu-native" in results:
        ratio = results["gpu"] / results["cpu-native"]
        print(f"  GPU / CPU(native, 24T) = {ratio:.2f}x")
        print()
        print(f"  vs cpu-python           = "
              f"{results['gpu'] / results['cpu-python']:,.0f}x  (MISLEADING)")
        print()
        if ratio > 10:
            print("  => Phase A is GPU-DOMINATED. A GPU/ASIC miner wins this")
            print("     phase outright; the work function is not CPU-only.")
        elif ratio > 2:
            print("  => Phase A favours the GPU by a modest factor. Cooperative")
            print("     mining is plausible but the GPU still leads.")
        else:
            print("  => Phase A is roughly PARITY. The arithmetic is not the")
            print("     GPU's advantage here; memory behaviour dominates.")

    banner("PHASE B: table probing (the memory-bound phase)")
    if CUDA_AVAILABLE:
        syndromes, _, _ = gpu_phase_a(
            matrix, 1, 1 << 20, params.n, params.m, params.q, params.sparsity
        )
        import cupy as cp

        host = cp.asnumpy(syndromes)
        rate = phase_b_probe_rate(host, params)
        print(f"  cpu table probe: {rate:>15,.0f} probes/s (python dict)")
        if "gpu" in results:
            print(f"  Phase A GPU is {results['gpu'] / rate:,.0f}x faster than")
            print("  this Phase B probe rate, so at equilibrium the miner is")
            print("  bottlenecked on the TABLE, not on syndrome arithmetic.")
            print()
            print("  NOTE: a python dict is a poor Phase B baseline too. A")
            print("  native open-addressed table would be much faster, so this")
            print("  ratio is an upper bound on Phase B's dominance.")

    banner("HONEST SUMMARY")
    print("  Measured, not asserted:")
    for name, rate in results.items():
        print(f"    {name:20s} {rate:>18,.0f} cand/s")
    print()
    print("  What this does NOT show: that the design is ASIC-resistant.")
    print("  The Phase A kernel is dense int32 MAC over a small matrix --")
    print("  the single most ASIC-friendly shape there is. Silicon would beat")
    print("  both numbers above by orders of magnitude.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
