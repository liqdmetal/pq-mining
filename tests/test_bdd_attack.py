"""Tests for multi-level bounded-weight attack shapes.

These tests validate the multi-level model on small instances and compare
it against the current two-list bounded-weight estimate, without claiming
optimality.
"""

from __future__ import annotations

import unittest

from pqmine.bdd_attack import (
    AttackModelError,
    AttackShape,
    best_multi_level_attack,
    compare_bounded_attack_family,
    multi_level_attack_work,
    shape_work,
    validate_on_toy_instance,
)
from pqmine.codebased import SDPParameters


class MultiLevelModelTests(unittest.TestCase):
    def test_toy_validation_runs(self):
        report = validate_on_toy_instance(
            n_small=32,
            syndrome_bits_small=16,
            w_max_small=4,
            matched_bits_small=8,
        )
        self.assertGreater(report["shapes_tested"], 0)
        self.assertIsNotNone(report["best_result"])

    def test_large_parameter_shape_exists(self):
        sdp = SDPParameters(n=512, k=256, weight=32)
        results = multi_level_attack_work(
            sdp, w_max=32, matched_bits=164,
        )
        self.assertGreater(len(results), 0)
        self.assertLessEqual(results[0].work_exponent, 83.1)
        self.assertTrue(any(r.target_weight < 32 for r in results))

    def test_lower_weight_is_not_omitted(self):
        sdp = SDPParameters(n=64, k=32, weight=8)
        results = multi_level_attack_work(
            sdp, w_max=8, matched_bits=16,
        )
        self.assertEqual(min(r.target_weight for r in results), 1)
        self.assertEqual(max(r.target_weight for r in results), 8)

    def test_explicit_invalid_shape_is_rejected(self):
        sdp = SDPParameters(n=16, k=8, weight=2)
        with self.assertRaises(AttackModelError):
            shape_work(AttackShape((), (), 2, 4), sdp)
        with self.assertRaises(AttackModelError):
            shape_work(AttackShape((2, 2, 2), (2, 3), 2, 4), sdp)

    def test_explicit_bad_partition_set_is_rejected(self):
        sdp = SDPParameters(n=16, k=8, weight=2)
        with self.assertRaises(AttackModelError):
            multi_level_attack_work(
                sdp, w_max=2, matched_bits=4, partitions=[[2, 2, 2]],
            )

    def test_invalid_matched_bits_rejected(self):
        sdp = SDPParameters(n=16, k=8, weight=2)
        with self.assertRaises(AttackModelError):
            multi_level_attack_work(sdp, w_max=2, matched_bits=0)

    def test_toy_shape_summary_has_warning_boundary(self):
        report = validate_on_toy_instance(32, 16, 4, 8)
        self.assertIn("toy_note", report)

    def test_best_multi_level_falls_back_to_two_list(self):
        sdp = SDPParameters(n=512, k=256, weight=32)
        result = best_multi_level_attack(
            sdp, w_max=32, matched_bits=164,
        )
        self.assertLessEqual(result.work_exponent, 83.1)

    def test_compare_includes_two_list_baseline(self):
        sdp = SDPParameters(n=512, k=256, weight=32)
        comparison = compare_bounded_attack_family(
            sdp, w_max=32, matched_bits=164,
        )
        self.assertIn("single_two_list_bound", comparison)
        self.assertIn("best_multi_level", comparison)
        # The canonical baseline now sweeps all two-list cuts. The symbolic
        # multi-partition family is a separate, narrower model.
        work_two = comparison["single_two_list_work"]
        work_multi = comparison["best_multi_level_work"]
        self.assertGreaterEqual(work_multi, work_two)
        self.assertAlmostEqual(work_two, 82.4461, places=3)

    def test_two_list_baseline_matches_full_split_sweep(self):
        from pqmine.bounded_work import mitm_attack_work, mitm_split_sweep

        baseline = mitm_attack_work(512, 256, 32, 164)
        sweep = mitm_split_sweep(512, 256, 32, 164)
        best_in_sweep = min(sweep, key=lambda attack: attack.work_bits)

        self.assertAlmostEqual(
            best_in_sweep.work_bits, baseline.work_bits, places=4,
        )
        self.assertEqual(best_in_sweep.split_index, baseline.split_index)

    def test_winning_two_list_fields_match_closed_form(self):
        from pqmine.bounded_work import log2_binom, mitm_attack_work

        attack = mitm_attack_work(512, 256, 32, 164)
        expected_left = log2_binom(attack.split_index, attack.left_weight)
        expected_right = log2_binom(
            512 - attack.split_index, attack.right_weight,
        )
        expected_work = max(
            max(expected_left, expected_right),
            164 - min(expected_left, expected_right),
        )

        self.assertAlmostEqual(attack.left_list_bits, expected_left)
        self.assertAlmostEqual(attack.right_list_bits, expected_right)
        self.assertAlmostEqual(attack.work_bits, expected_work)
        self.assertAlmostEqual(
            attack.expected_pair_excess_bits,
            expected_left + expected_right - 164,
        )

if __name__ == "__main__":
    unittest.main()
