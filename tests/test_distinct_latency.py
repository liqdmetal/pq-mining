"""Tests that the two properties in tension both hold.

The distinctness fix and the latency-bound property pull against each
other: the fix adds per-step work, and option B exists only because the
step is dominated by DRAM latency. These tests assert BOTH, so a future
change that trades one for the other fails loudly instead of silently
producing a design that is correct but ASIC-friendly, or fast but
cryptographically hollow.

Measured on this machine (5900X, DDR4), 400 steps x 1500 repeats:

    arena     ns/step   probes/step
    0.03 MiB     29.2        1.686
      8   MiB     45.5        1.680
    128   MiB    113.8        1.698
    512   MiB    134.9        1.693

    cache->DRAM ratio 4.62x  (was 18.5x before distinctness)
"""

from __future__ import annotations

import unittest

from pqmine.codebased import CodeBasedError, SDPParameters, derive_columns
from pqmine.distinct_walk import (
    distinct_walk,
    max_steps_for_code,
)
from pqmine.latency_bound import LatencyParameters, build_arena
from pqmine.native_distinct import (
    NATIVE_AVAILABLE,
    NativeDistinctError,
    native_distinct_rate,
    repeated_walk_sweep,
)


class WeightExactnessTests(unittest.TestCase):
    """Property 1: the syndrome is a genuine weight-K error vector."""

    def setUp(self):
        self.sdp = SDPParameters(n=512, k=480, weight=8)
        self.columns = derive_columns(b"exactness", self.sdp)

    def test_effective_weight_equals_steps(self):
        """The break this fixed: effective weight plateaued near 74
        regardless of K, so difficulty did not grow with work."""
        for steps in (32, 128, 256, 400):
            params = LatencyParameters(arena_bits=14, steps=steps)
            result = distinct_walk(
                build_arena(b"exactness", params), b"exactness", 1,
                params, self.columns,
            )
            self.assertTrue(result.weight_is_exact)
            self.assertEqual(result.effective_weight, steps)

    def test_support_has_no_duplicates(self):
        params = LatencyParameters(arena_bits=14, steps=200)
        result = distinct_walk(
            build_arena(b"dup", params), b"dup", 3, params, self.columns
        )
        self.assertEqual(len(set(result.support)), len(result.support))

    def test_steps_beyond_column_budget_rejected(self):
        """K <= n is a hard structural limit once columns must be distinct;
        exceeding it must raise rather than silently reuse columns."""
        params = LatencyParameters(arena_bits=14, steps=500)
        with self.assertRaises(CodeBasedError):
            distinct_walk(
                build_arena(b"over", params), b"over", 1, params,
                self.columns,
            )

    def test_budget_leaves_headroom(self):
        """The 90% cap keeps the rehash loop short: as the used set fills,
        finding an unused column takes many probes."""
        self.assertEqual(max_steps_for_code(512), 460)
        self.assertLess(max_steps_for_code(512), 512)

    def test_walk_remains_deterministic(self):
        params = LatencyParameters(arena_bits=14, steps=120)
        first = distinct_walk(
            build_arena(b"det", params), b"det", 9, params, self.columns
        )
        second = distinct_walk(
            build_arena(b"det", params), b"det", 9, params, self.columns
        )
        self.assertEqual(first.syndrome, second.syndrome)
        self.assertEqual(first.support, second.support)


@unittest.skipUnless(NATIVE_AVAILABLE, "numba unavailable")
class LatencyBoundStillHoldsTests(unittest.TestCase):
    """Property 2: memory latency still dominates the step."""

    def test_still_latency_bound_after_distinctness_fix(self):
        """THE test that keeps the two properties honest together.

        Repeats are modest here to keep the suite fast; the full sweep
        (1500 repeats) measured 4.62x.
        """
        sweep = repeated_walk_sweep(
            arena_bits_range=(12, 20, 24), steps=400, repeats=200
        )
        self.assertTrue(sweep["still_latency_bound"], sweep["verdict"])
        self.assertGreater(sweep["cache_to_dram_ratio"], 2.0)

    def test_rehash_cost_does_not_grow_with_arena(self):
        """probes/step must stay flat: it is a function of how full the
        used-set is, not of arena size. If it grew with the arena, the
        distinctness check would be competing for the memory system."""
        sweep = repeated_walk_sweep(
            arena_bits_range=(12, 24), steps=400, repeats=200
        )
        probes = [row["probes_per_step"] for row in sweep["rows"]]
        self.assertLess(abs(probes[0] - probes[-1]), 0.2)

    def test_dram_step_cost_in_expected_band(self):
        measurement = native_distinct_rate(24, 400, column_count=512)
        self.assertGreater(measurement.ns_per_step, 20.0)
        self.assertLess(measurement.ns_per_step, 600.0)

    def test_native_rejects_over_budget_steps(self):
        with self.assertRaises(NativeDistinctError):
            native_distinct_rate(20, 500, column_count=512)

    def test_native_rejects_bad_arena_size(self):
        for bad in (4, 64):
            with self.assertRaises(NativeDistinctError):
                native_distinct_rate(bad, 100)


if __name__ == "__main__":
    unittest.main()
