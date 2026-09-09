"""Tests for the latency-bound walk (option B).

The decisive test is test_walk_is_latency_bound: ns/step must rise sharply
as the arena outgrows cache. A flat curve would mean the walk is
compute-bound and the whole ASIC-resistance argument collapses -- so this
test is the design's own falsification criterion, not a formality.
"""

from __future__ import annotations

import unittest

from pqmine.codebased import CodeBasedError, SDPParameters, derive_columns
from pqmine.latency_bound import (
    LatencyParameters,
    build_arena,
    composition_risks,
    replay_walk,
    verification_cost_note,
    walk,
)
from pqmine.native_walk import (
    NATIVE_AVAILABLE,
    latency_sweep,
    native_walk_rate,
    theoretical_gpu_penalty,
)


class ParameterTests(unittest.TestCase):
    def test_rejects_degenerate_shapes(self):
        for bad in (
            dict(arena_bits=4, steps=10),        # too small
            dict(arena_bits=40, steps=10),       # absurdly large
            dict(arena_bits=16, steps=0),        # no work
        ):
            with self.assertRaises(CodeBasedError):
                LatencyParameters(**bad)
        with self.assertRaises(CodeBasedError):
            LatencyParameters(arena_bits=16, steps=10, write_density=0)
        with self.assertRaises(CodeBasedError):
            LatencyParameters(arena_bits=16, steps=10, cell_bytes=4)

    def test_arena_size_arithmetic(self):
        params = LatencyParameters(arena_bits=22, steps=100, cell_bytes=32)
        self.assertEqual(params.cells, 4_194_304)
        self.assertAlmostEqual(params.arena_mib, 128.0, places=1)
        self.assertTrue(params.exceeds_typical_l3)

    def test_small_arena_flagged_as_cache_resident(self):
        """An arena that fits in L3 defeats the design's premise."""
        params = LatencyParameters(arena_bits=18, steps=100, cell_bytes=32)
        self.assertAlmostEqual(params.arena_mib, 8.0, places=1)
        self.assertFalse(params.exceeds_typical_l3)


class WalkTests(unittest.TestCase):
    def setUp(self):
        self.sdp = SDPParameters(n=174, k=134, weight=8)
        self.columns = derive_columns(b"walk-tests", self.sdp)
        self.params = LatencyParameters(arena_bits=14, steps=400)

    def test_walk_is_deterministic(self):
        first = walk(build_arena(b"det", self.params), b"det", 5,
                     self.params, self.columns)
        second = walk(build_arena(b"det", self.params), b"det", 5,
                      self.params, self.columns)
        self.assertEqual(first.syndrome, second.syndrome)
        self.assertEqual(first.transcript, second.transcript)
        self.assertEqual(first.visited, second.visited)

    def test_nonce_changes_the_walk(self):
        base = walk(build_arena(b"n", self.params), b"n", 1,
                    self.params, self.columns)
        other = walk(build_arena(b"n", self.params), b"n", 2,
                     self.params, self.columns)
        self.assertNotEqual(base.syndrome, other.syndrome)
        self.assertNotEqual(base.visited, other.visited)

    def test_header_changes_the_walk(self):
        first = walk(build_arena(b"header-a", self.params), b"header-a", 1,
                     self.params, self.columns)
        second = walk(build_arena(b"header-b", self.params), b"header-b", 1,
                      self.params, self.columns)
        self.assertNotEqual(first.syndrome, second.syndrome)

    def test_replay_reproduces_the_walk(self):
        arena = build_arena(b"replay", self.params)
        mined = walk(arena, b"replay", 9, self.params, self.columns)
        replayed = replay_walk(b"replay", 9, self.params, self.columns)
        self.assertEqual(mined.syndrome, replayed.syndrome)
        self.assertEqual(mined.transcript, replayed.transcript)

    def test_arena_is_mutated(self):
        """Mutability is load-bearing: a static arena is recomputable in
        O(1), so an attacker need not store it at all."""
        arena = build_arena(b"mutate", self.params)
        before = bytes(arena)
        walk(arena, b"mutate", 3, self.params, self.columns)
        self.assertNotEqual(before, bytes(arena))

    def test_walk_visits_the_requested_number_of_steps(self):
        result = walk(build_arena(b"count", self.params), b"count", 1,
                      self.params, self.columns)
        self.assertEqual(len(result.visited), self.params.steps)
        self.assertEqual(result.steps, self.params.steps)

    def test_empty_header_rejected(self):
        with self.assertRaises(CodeBasedError):
            build_arena(b"", self.params)

    def test_addresses_stay_in_range(self):
        result = walk(build_arena(b"range", self.params), b"range", 1,
                      self.params, self.columns)
        for address in result.visited:
            self.assertGreaterEqual(address, 0)
            self.assertLess(address, self.params.cells)


@unittest.skipUnless(NATIVE_AVAILABLE, "numba unavailable")
class NativeLatencyTests(unittest.TestCase):
    """The measurements that decide whether option B succeeded."""

    def test_walk_is_latency_bound(self):
        """THE decisive test. ns/step must rise sharply as the arena leaves
        cache; a flat curve falsifies the design.

        Measured on this machine: 5.9 ns (L1/L2) -> 93.5 ns (128 MiB DRAM),
        an 18.5x rise, plateauing near DDR4 random-access latency.
        """
        sweep = latency_sweep(steps=300_000, bits_range=(12, 20, 24))
        self.assertTrue(sweep["latency_bound"], sweep["verdict"])
        self.assertGreater(sweep["cache_to_dram_ratio"], 3.0)

    def test_dram_step_cost_is_in_the_right_ballpark(self):
        """A DRAM-resident walk should cost tens to low hundreds of ns per
        step. Far below that means cache; far above means the measurement is
        dominated by something other than memory."""
        measurement = native_walk_rate(24, 300_000)
        self.assertGreater(measurement.ns_per_step, 20.0)
        self.assertLess(measurement.ns_per_step, 500.0)

    def test_cache_resident_walk_is_much_faster(self):
        small = native_walk_rate(12, 300_000)
        large = native_walk_rate(24, 300_000)
        self.assertLess(small.ns_per_step, large.ns_per_step / 3.0)

    def test_native_rejects_bad_arena_size(self):
        from pqmine.native_walk import NativeError

        for bad in (4, 64):
            with self.assertRaises(NativeError):
                native_walk_rate(bad, 1000)


class HonestyTests(unittest.TestCase):
    """The construction must state its own unresolved problems."""

    def test_verification_cost_problem_is_stated(self):
        note = verification_cost_note()
        self.assertIn("problem", note)
        self.assertIn("OPEN", note["status"])
        # Must not pretend verification is as cheap as hash PoW.
        self.assertIn("2^70", note["why_it_is_still_usable"])

    def test_composition_risks_name_the_restriction_problem(self):
        risks = composition_risks()
        self.assertIn("restricted_solution_space", risks)
        self.assertIn("UNANALYSED", risks["restricted_solution_space"])
        self.assertIn("Esser-Santini", risks["restricted_solution_space"])
        self.assertIn("OPEN", risks["status"])

    def test_gpu_penalty_records_both_regimes(self):
        """The measured loss AND the literature's expected win, so the
        reversal is documented rather than asserted."""
        penalty = theoretical_gpu_penalty()
        self.assertIn("MEASURED", penalty["independent_probes"]["gpu_advantage"])
        self.assertIn("SLOWER", penalty["dependent_walk"]["gpu_advantage"])
        self.assertIn("VRAM", penalty["caveat"])

    def test_asic_bound_is_compared_to_prior_art(self):
        penalty = theoretical_gpu_penalty()
        self.assertIn("RandomX", penalty["asic_bound"])
        self.assertIn("Argon2id", penalty["asic_bound"])


if __name__ == "__main__":
    unittest.main()
