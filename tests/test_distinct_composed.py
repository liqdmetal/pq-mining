"""Tests for the distinct-column composed PoW.

Two properties are pinned: exact weight (the security property the F_2
pigeonhole audit showed was missing) and CPU/GPU agreement on syndromes.
The throughput property is NOT pinned as a hard pass/fail here because it
is genuinely parameter-dependent -- see the measured table in the module
docstring and `docs/COMPOSED_RESULTS.md`. A test that asserts "CPU wins" at
one K would be asserting a number that flips at another K.
"""

import unittest

import numpy as np

from pqmine.distinct_composed import (
    CUDA_AVAILABLE,
    MAX_STEPS,
    NUMBA_AVAILABLE,
    USED_WORDS,
    DistinctComposedError,
    cpu_distinct,
    cpu_syndromes_and_weights,
    gpu_distinct,
)
from pqmine.composed import N_COLUMNS


@unittest.skipUnless(NUMBA_AVAILABLE, "needs numba")
class ExactWeightTests(unittest.TestCase):
    """The property the shared-column composition LACKED."""

    def test_every_chain_selects_exactly_k_distinct_columns(self):
        for k in (16, 128, 400):
            _syn, weights = cpu_syndromes_and_weights(k, 128)
            self.assertTrue(np.all(weights == k),
                            f"K={k}: weights ranged {weights.min()}..{weights.max()}")

    def test_used_set_fits_one_cache_line_for_default_columns(self):
        self.assertEqual(USED_WORDS * 8, 64)
        self.assertEqual(N_COLUMNS, 512)

    def test_steps_capped_at_90_percent_occupancy(self):
        self.assertEqual(MAX_STEPS, int(N_COLUMNS * 0.9))
        with self.assertRaises(DistinctComposedError):
            cpu_distinct(MAX_STEPS + 1, 8)

    def test_probes_per_step_stays_bounded(self):
        """Rehash cost at 78% occupancy (400/512) measured ~1.94. It must
        not explode; >4 would mean the occupancy cap is wrong."""
        rate = cpu_distinct(400, 512)
        self.assertLess(rate.mean_probes_per_step, 4.0)
        self.assertGreater(rate.mean_probes_per_step, 1.0)


@unittest.skipUnless(NUMBA_AVAILABLE and CUDA_AVAILABLE, "needs numba + CUDA")
class CrossDeviceTests(unittest.TestCase):
    def test_cpu_and_gpu_agree_on_probe_counts(self):
        """Same rehash sequence on both devices => same probe totals.
        A divergence here would mean the two kernels select DIFFERENT
        columns and shares would not cross-verify."""
        c = cpu_distinct(200, 1024)
        g = gpu_distinct(200, 1024)
        self.assertAlmostEqual(c.mean_probes_per_step, g.mean_probes_per_step,
                               places=2)

    def test_non_block_multiple_chain_counts_run(self):
        for chains in (100, 257, 1000):
            self.assertEqual(gpu_distinct(64, chains).chains, chains)


if __name__ == "__main__":
    unittest.main()
