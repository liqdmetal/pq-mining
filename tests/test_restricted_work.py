"""Tests for generic-vs-walk restricted attack accounting."""

import unittest

from pqmine.codebased import SDPParameters
from pqmine.latency_bound import LatencyParameters
from pqmine.restricted_work import (
    RestrictedWorkError,
    composite_attack,
    sample_walk_scan,
)


class RestrictedWorkTests(unittest.TestCase):
    def setUp(self):
        self.sdp = SDPParameters(n=64, k=32, weight=8)
        self.walk = LatencyParameters(
            arena_bits=8, steps=32, write_density=1, cell_bytes=8,
        )

    def test_sample_reports_admissibility(self):
        report = sample_walk_scan(
            b"restricted-work", self.sdp, self.walk,
            matched_bits=4, w_max=8, nonces=32, target_weight=4,
        )
        self.assertEqual(report.samples, 32)
        self.assertGreaterEqual(report.admissible_samples, 0)
        self.assertLessEqual(report.admissible_samples, report.samples)
        self.assertGreater(report.success_probability_bits, 0.0)

    def test_composite_includes_generic_route(self):
        report = composite_attack(
            b"restricted-work", self.sdp, self.walk,
            matched_bits=4, w_max=8, nonces=16, target_weight=4,
        )
        self.assertIn(report.best_route, {
            "generic-bounded-mitm", "walk-restricted-nonce-scan",
        })
        self.assertLessEqual(report.best_work_bits, report.generic.work_bits)

    def test_invalid_zero_samples_rejected(self):
        with self.assertRaises(RestrictedWorkError):
            sample_walk_scan(
                b"bad", self.sdp, self.walk,
                matched_bits=4, w_max=8, nonces=0, target_weight=4,
            )

    def test_target_above_columns_rejected(self):
        with self.assertRaises(RestrictedWorkError):
            sample_walk_scan(
                b"bad", self.sdp, self.walk,
                matched_bits=4, w_max=8, nonces=1, target_weight=65,
            )


if __name__ == "__main__":
    unittest.main()
