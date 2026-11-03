"""Tests for conservative quantum query accounting."""

import unittest

from pqmine.quantum_work import (
    QuantumWorkError,
    best_modeled_quantum_work,
    grover_bounded_support_work,
)


class QuantumWorkTests(unittest.TestCase):
    def test_asymptotic_extrapolation_is_not_concrete_grover_cost(self):
        attack = grover_bounded_support_work(
            512, 32, 164, syndrome_bits=256,
        )
        # These are different models, not two calibrated concrete attacks.
        self.assertAlmostEqual(attack.quantum_isd_bits, 30.04928, places=3)
        self.assertAlmostEqual(attack.grover_query_bits, 82.0, places=3)
        self.assertFalse(attack.summary()["security_floor_established"])
        self.assertIsNone(attack.summary()["finite_instance_quantum_security_bits"])
        self.assertTrue(attack.satisfiable_regime)
        self.assertGreater(attack.candidate_space_bits, 164.0)

    def test_lower_weights_are_in_candidate_space(self):
        attack = grover_bounded_support_work(32, 4, 8, syndrome_bits=16)
        self.assertGreater(attack.candidate_space_bits, 0.0)
        self.assertAlmostEqual(attack.expected_solutions_bits,
                               attack.candidate_space_bits - 8.0)

    def test_best_route_reports_quantum_ceiling(self):
        result = best_modeled_quantum_work(
            512, 256, 32, 164, classical_attack_bits=82.446094,
        )
        self.assertEqual(result["selected_route"], "generic-grover")
        self.assertAlmostEqual(result["best_modeled_quantum_work_bits"], 82.0, places=3)
        self.assertAlmostEqual(result["quantum_isd_asymptotic_extrapolation_bits"], 30.04928, places=3)
        self.assertFalse(result["security_floor_established"])
        self.assertIn("not a PQ security claim", result["warning"])

    def test_bad_parameters_rejected(self):
        with self.assertRaises(QuantumWorkError):
            grover_bounded_support_work(512, 32, 0, syndrome_bits=256)
        with self.assertRaises(QuantumWorkError):
            grover_bounded_support_work(512, 32, 300, syndrome_bits=256)


if __name__ == "__main__":
    unittest.main()
