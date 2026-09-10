"""Tests for the actual verifier language: weight 1..w_max, partial syndrome."""

import unittest

from pqmine.bounded_work import (
    log2_expected_bounded_solutions,
    mitm_attack_work,
)


class BoundedLanguageTests(unittest.TestCase):
    def test_lower_weights_are_included(self):
        bounded = log2_expected_bounded_solutions(512, 256, 32, 164)
        exact = 168.90954819506314 - 164.0
        self.assertGreater(bounded, exact)

    def test_mitm_never_reports_negative_work(self):
        attack = mitm_attack_work(512, 256, 32, 164)
        self.assertGreaterEqual(attack.work_bits, 0.0)
        self.assertLessEqual(attack.work_bits, 82.45)
        self.assertEqual(attack.split_index, 215)

    def test_attack_has_an_explicit_weight(self):
        attack = mitm_attack_work(512, 256, 32, 164)
        self.assertIn(attack.target_weight, range(1, 33))
        self.assertEqual(attack.target_weight,
                         attack.left_weight + attack.right_weight)

    def test_partial_syndrome_changes_the_attack(self):
        hard = mitm_attack_work(512, 256, 164, 164)
        easy = mitm_attack_work(512, 256, 32, 32)
        self.assertNotEqual(hard.work_bits, easy.work_bits)


if __name__ == "__main__":
    unittest.main()
