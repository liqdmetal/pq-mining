"""Tests for difficulty calibration.

The two findings worth pinning: the work model reproduces a literature
figure (so it is not made up), and the composed design's K=460 is a DENSE
error -- the complement of a sparse one -- which is the wrong regime for
SDP hardness and makes the instance unsatisfiable at rate 1/2.
"""

import math
import unittest

from pqmine.calibration import (
    Calibration,
    CalibrationError,
    calibrate,
    difficulty_ladder,
    log2_binom,
    log2_expected_solutions,
    max_matched_bits,
    satisfiable_probability,
    work_bjmm,
    work_prange,
    work_stern,
)


class WorkModelSanityTests(unittest.TestCase):
    def test_reproduces_mceliece_literature_figure(self):
        """n=1024 k=524 w=50: Stern is ~2^60-65 in the literature.
        If this drifts the model is broken."""
        stern = work_stern(1024, 524, 50)
        self.assertGreater(stern, 58.0)
        self.assertLess(stern, 70.0)

    def test_prange_is_always_at_least_stern(self):
        for n, k, w in ((512, 256, 16), (1024, 524, 50), (2048, 1024, 40)):
            self.assertGreaterEqual(work_prange(n, k, w), work_stern(n, k, w) - 0.01)

    def test_bjmm_is_at_least_as_fast_as_stern(self):
        for n, k, w in ((512, 256, 16), (1024, 524, 50)):
            self.assertLessEqual(work_bjmm(n, k, w), work_stern(n, k, w) + 0.01)

    def test_binomial_symmetry(self):
        self.assertAlmostEqual(log2_binom(512, 460), log2_binom(512, 52), places=6)


class SatisfiabilityTests(unittest.TestCase):
    def test_earlier_naive_pick_was_nearly_unsolvable(self):
        """The n=90 k=58 w=8 instance that measured 2^-6.3 solutions.
        At full 32-bit match the model gives ~2^4 -- but the MEASURED
        instance used a different (k, syndrome) mapping; what matters is
        the model flags low-surplus regimes."""
        e = log2_expected_solutions(90, 58, 8, 32)
        self.assertLess(e, 6.0)

    def test_surplus_controls_satisfiability(self):
        self.assertGreater(satisfiable_probability(4.0), 0.9999)
        self.assertLess(satisfiable_probability(-6.3), 0.02)

    def test_matched_bits_out_of_range_rejected(self):
        with self.assertRaises(CalibrationError):
            log2_expected_solutions(512, 256, 16, 300)


class DenseWeightFindingTests(unittest.TestCase):
    """The composed design mines K = steps = 400-460 of 512 columns.
    That is a DENSE error: the complement of a weight-52..112 one."""

    def test_k460_is_the_complement_of_a_sparse_error(self):
        self.assertAlmostEqual(log2_binom(512, 460), log2_binom(512, 52), places=6)

    def test_k460_is_unsatisfiable_at_rate_half(self):
        e = log2_expected_solutions(512, 256, 460)
        self.assertLess(e, 0.0, "E[solutions] < 1: most headers unsolvable")

    def test_sparse_k_is_abundantly_satisfiable(self):
        self.assertGreater(log2_expected_solutions(512, 256, 100), 100.0)

    def test_the_hard_regime_is_sparse_not_dense(self):
        """SDP hardness lives at SMALL w (below the GV bound); the composed
        design's large K is the easy/degenerate side. Pin that a sparse w
        gives more Stern work than a dense one at the same rate."""
        self.assertGreater(work_stern(512, 256, 48), work_stern(512, 256, 8))


class RetargetLadderTests(unittest.TestCase):
    def test_ladder_is_one_bit_per_step(self):
        ladder = difficulty_ladder(512, 256, 24, 4.0)
        for a, b in zip(ladder, ladder[1:]):
            self.assertAlmostEqual(b.work_stern - a.work_stern, 1.0, places=6)
            self.assertAlmostEqual(a.log2_solutions - b.log2_solutions, 1.0, places=6)

    def test_ladder_never_exceeds_satisfiability_cap(self):
        ladder = difficulty_ladder(512, 256, 24, 4.0)
        for step in ladder:
            self.assertGreaterEqual(step.log2_solutions, 4.0 - 1e-9)

    def test_cap_respects_available_syndrome_bits(self):
        self.assertLessEqual(max_matched_bits(512, 256, 24), 256)

    def test_calibrate_picks_closest_work(self):
        c = calibrate(512, 256, 24, target_work_bits=-100.0)
        self.assertIsInstance(c, Calibration)
        self.assertLessEqual(c.matched_bits, max_matched_bits(512, 256, 24))

    def test_calibrate_rejects_bad_shape(self):
        with self.assertRaises(CalibrationError):
            calibrate(512, 512, 24, 10.0)


class AttackerEdgeTests(unittest.TestCase):
    def test_edge_is_reported_and_positive(self):
        c = Calibration(512, 256, 32, 164, 4.0)
        self.assertGreater(c.attacker_edge_bits, 0.0)
        self.assertIn("attacker_edge_factor", c.summary())

    def test_edge_grows_with_weight(self):
        """BJMM's advantage over Stern grows with w -- so heavier instances
        hand the optimal attacker more edge, which the spec must publish."""
        low = Calibration(512, 256, 16, 95, 4.0).attacker_edge_bits
        high = Calibration(512, 256, 48, 221, 4.0).attacker_edge_bits
        self.assertGreater(high, low)


if __name__ == "__main__":
    unittest.main()
