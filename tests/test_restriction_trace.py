"""Regression tests for the restricted-support audit trace."""

import unittest
from collections import Counter

from pqmine.codebased import SDPParameters, derive_columns
from pqmine.decoupled_trace import trace_decoupled_walk
from pqmine.latency_bound import LatencyParameters, build_arena, walk
from pqmine.restriction_attack import ConcentrationReport, cycle_analysis


class ReadTimeTraceTests(unittest.TestCase):
    def test_selected_columns_are_recorded_at_read_time(self):
        sdp = SDPParameters(n=512, k=480, weight=8)
        params = LatencyParameters(arena_bits=12, steps=100)
        result = walk(build_arena(b"trace", params), b"trace", 7,
                      params, derive_columns(b"trace", sdp))
        self.assertEqual(len(result.selected_columns), params.steps)

    def test_cycle_analysis_uses_trace_and_matches_odd_parity(self):
        sdp = SDPParameters(n=512, k=480, weight=8)
        params = LatencyParameters(arena_bits=12, steps=100)
        result = trace_decoupled_walk(
            build_arena(b"cycle", params), 3, params,
            derive_columns(b"cycle", sdp), target_weight=8,
        )
        expected = sum(count % 2 for count in Counter(result.selected_columns).values())
        report = cycle_analysis(b"cycle", sdp, params, nonce=3)
        self.assertEqual(report.effective_weight, expected)

    def test_birthday_expectation_is_stable_at_256_bits(self):
        report = ConcentrationReport(
            samples=256, distinct_syndromes=256, most_common_count=1,
            syndrome_bits=256,
        )
        self.assertAlmostEqual(report.expected_distinct, 256.0, places=8)
        self.assertTrue(report.spread)


if __name__ == "__main__":
    unittest.main()
