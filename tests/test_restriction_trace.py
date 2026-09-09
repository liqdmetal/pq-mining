"""Regression tests for the restricted-support audit trace.

These tests exercise the audit across nonce-derived parity-check columns so
that the restricted-support screens actually reflect the current verifier
construction, not a stale once-per-header matrix.
"""

import unittest
from collections import Counter

from pqmine.codebased import SDPParameters
from pqmine.decoupled_trace import trace_decoupled_walk
from pqmine.latency_bound import LatencyParameters, build_arena
from pqmine.verifier import PoWParams, derive_columns as derive_v2_columns
from pqmine.restriction_attack import ConcentrationReport, cycle_analysis


def derive_v2_columns_one(header: bytes, sdp: SDPParameters, nonce: int) -> tuple:
    params = PoWParams(
        n_columns=sdp.n,
        syndrome_bits=sdp.syndrome_bits,
        w_max=max(sdp.weight, 1),
        matched_bits=sdp.syndrome_bits,
    )
    return derive_v2_columns(header, params, nonce)


class TestV2ColumnParityCollapse(unittest.TestCase):
    """At toy parameters, effective weight must equal selected weight."""

    def test_parity_collapse_vanishes_at_target_weight(self):
        sdp = SDPParameters(n=512, k=256, weight=8)
        params = LatencyParameters(arena_bits=12, steps=400)
        for nonce in range(6):
            columns = derive_v2_columns_one(b"cycle-v2", sdp, nonce)
            result = trace_decoupled_walk(
                build_arena(b"cycle-v2", params), nonce, params,
                columns, sdp.weight,
            )
            expected = sum(count % 2 for count in Counter(result.selected_columns).values())
            self.assertEqual(result.weight, expected,
                             f"nonce {nonce}: effective {result.weight} != selected {expected}")


class TestConcentrationReportDisplay(unittest.TestCase):
    def test_concentration_report_has_distinct_syndromes_field(self):
        report = ConcentrationReport(
            samples=256,
            distinct_syndromes=256,
            most_common_count=1,
            syndrome_bits=256,
        )
        self.assertIsInstance(report.samples, int)
        self.assertIsInstance(report.distinct_syndromes, int)
        self.assertAlmostEqual(report.expected_distinct, 256.0, places=6)
        self.assertTrue(report.spread)


class TestCycleReportDisplay(unittest.TestCase):
    def test_cycle_report_has_steps_and_healthy_verdict(self):
        report = cycle_analysis(b"conv", SDPParameters(n=512, k=256, weight=8),
                                LatencyParameters(arena_bits=12, steps=400),
                                target_weight=8)
        self.assertIsInstance(report.steps, int)
        self.assertGreater(report.steps, 0)
        self.assertTrue(report.healthy)
        self.assertIsInstance(report.effective_weight, int)
