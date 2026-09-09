"""Tests for the composed PoW: mutable L1 scratchpad + code-based syndrome.

Two things must hold together, and each is measured rather than assumed:
  * correctness -- CPU and GPU compute the SAME syndrome for the same
    nonce (a share must verify identically on either device);
  * the CPU throughput advantage survives the syndrome stream.

The throughput test is the substantive one. It is deliberately loose
(CPU must win at all, not by a specific factor) because it is a timing
measurement on shared hardware; the measured figure was 1.75x.
"""

import unittest

import numpy as np

from pqmine.composed import (
    CUDA_AVAILABLE,
    NUMBA_AVAILABLE,
    N_COLUMNS,
    SCRATCH_BITS,
    cpu_composed,
    gpu_composed,
)


def _cpu_syndromes(steps: int, chains: int):
    """Run the CPU path and return the raw syndromes for cross-checking."""
    from pqmine.composed import _cpu_composed_many, _parity_check
    words = 1 << SCRATCH_BITS
    rng = np.random.default_rng(11)
    scratch = rng.integers(0, 2 ** 63, size=chains * words, dtype=np.uint64)
    out = np.zeros(chains, dtype=np.uint64)
    _cpu_composed_many(scratch, np.uint64(words), _parity_check(),
                       np.uint64(N_COLUMNS), np.uint64(words - 1),
                       np.uint64(0x1000), steps, chains, out)
    return out


def _gpu_syndromes(steps: int, chains: int):
    import cupy as cp
    from pqmine.composed import _COMPOSED_KERNEL, _parity_check
    words = 1 << SCRATCH_BITS
    rng = np.random.default_rng(11)
    scratch = cp.asarray(rng.integers(0, 2 ** 63, size=chains * words, dtype=np.uint64))
    H = cp.asarray(_parity_check())
    out = cp.zeros(chains, dtype=cp.uint64)
    kernel = cp.RawModule(code=_COMPOSED_KERNEL).get_function("composed_walk")
    block = 256
    grid = (chains + block - 1) // block
    kernel((grid,), (block,), (scratch, H, np.uint64(words), np.uint64(words - 1),
                               np.uint64(N_COLUMNS), np.uint64(0x1000),
                               np.int32(steps), np.uint64(chains), out))
    cp.cuda.Device().synchronize()
    return cp.asnumpy(out)


@unittest.skipUnless(NUMBA_AVAILABLE and CUDA_AVAILABLE, "needs numba + CUDA")
class CrossDeviceCorrectnessTests(unittest.TestCase):
    """A share must verify the same on CPU and GPU, or the PoW is broken."""

    def test_cpu_and_gpu_agree_on_every_syndrome(self):
        steps, chains = 512, 1024
        cpu = _cpu_syndromes(steps, chains)
        gpu = _gpu_syndromes(steps, chains)
        self.assertEqual(len(cpu), len(gpu))
        mismatches = int(np.sum(cpu != gpu))
        self.assertEqual(mismatches, 0,
                         f"{mismatches}/{chains} syndromes differ across devices")

    def test_different_nonces_give_different_syndromes(self):
        syn = _cpu_syndromes(512, 256)
        self.assertGreater(len(set(syn.tolist())), 250,
                           "syndromes should be near-unique across nonces")

    def test_syndrome_is_deterministic(self):
        a = _cpu_syndromes(256, 128)
        b = _cpu_syndromes(256, 128)
        self.assertTrue(np.array_equal(a, b))


@unittest.skipUnless(NUMBA_AVAILABLE and CUDA_AVAILABLE, "needs numba + CUDA")
class ThroughputTests(unittest.TestCase):
    """The design goal: CPU advantage survives the syndrome stream."""

    def test_cpu_wins_aggregate_throughput(self):
        cpu = cpu_composed(2048, 4096)
        gpu = gpu_composed(2048, 32768)
        ratio = gpu.chains_per_second / cpu.chains_per_second
        self.assertLess(
            ratio, 1.0,
            f"GPU/CPU = {ratio:.2f}; the L1-scratchpad + syndrome "
            "composition must keep the CPU ahead (measured 0.57x)",
        )

    def test_out_of_range_threads_do_not_corrupt(self):
        """Regression for the illegal-address bug: chain counts that are
        not multiples of the 256-thread block must run cleanly."""
        for chains in (100, 257, 1000):
            gpu = gpu_composed(64, chains)
            self.assertEqual(gpu.chains, chains)


class ParameterTests(unittest.TestCase):
    def test_scratchpad_fits_l1(self):
        """The whole point: 8 KiB scratch + 4 KiB H = 12 KiB < 32 KiB L1."""
        scratch_kib = (1 << SCRATCH_BITS) * 8 / 1024
        h_kib = N_COLUMNS * 8 / 1024
        self.assertLessEqual(scratch_kib + h_kib, 32.0)

    def test_scratchpad_is_in_the_measured_cpu_win_regime(self):
        """The sweep showed CPU wins at 1-8 KiB and loses at 32+ KiB.
        Pin the choice inside the winning band."""
        scratch_kib = (1 << SCRATCH_BITS) * 8 / 1024
        self.assertGreaterEqual(scratch_kib, 1.0)
        self.assertLessEqual(scratch_kib, 16.0)


if __name__ == "__main__":
    unittest.main()
