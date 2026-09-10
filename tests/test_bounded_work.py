"""Tests for the actual verifier language: weight 1..w_max, partial syndrome."""

import unittest

from pqmine.bounded_work import (
    certify_mitm_attack,
    execute_mitm_instance,
    log2_expected_bounded_solutions,
    max_matched_bits_bounded,
    mitm_attack_work,
    verify_mitm_certificate,
)
from pqmine.verifier import PoWParams, Share, derive_instance, verify_share


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

    def test_bounded_satisfiability_cap_uses_all_accepted_weights(self):
        cap = max_matched_bits_bounded(512, 256, 32, surplus_bits=4.0)
        self.assertEqual(cap, 165)
        self.assertGreaterEqual(
            log2_expected_bounded_solutions(512, 256, 32, cap), 4.0,
        )
        self.assertLess(
            log2_expected_bounded_solutions(512, 256, 32, cap + 1), 4.0,
        )

    def test_bounded_cap_respects_syndrome_dimension(self):
        self.assertEqual(max_matched_bits_bounded(64, 16, 8), 16)

    def test_bounded_cap_rejects_negative_surplus(self):
        with self.assertRaises(ValueError):
            max_matched_bits_bounded(64, 16, 8, surplus_bits=-1.0)

    def test_mitm_certificate_verifies_full_parameter_sweep(self):
        certificate = certify_mitm_attack(32, 32, 4, 8)
        self.assertTrue(verify_mitm_certificate(certificate))
        self.assertEqual(certificate.split_count, 31)
        self.assertEqual(certificate.weight_count, 4)
        self.assertEqual(certificate.summary()["verified"], True)

    def test_mitm_certificate_detects_tampering(self):
        from dataclasses import replace

        certificate = certify_mitm_attack(32, 32, 4, 8)
        tampered_attack = replace(
            certificate.attack,
            split_index=certificate.attack.split_index + 1,
        )
        tampered = replace(certificate, attack=tampered_attack)
        self.assertFalse(verify_mitm_certificate(tampered))

    def test_large_parameter_certificate_has_explicit_coverage(self):
        certificate = certify_mitm_attack(512, 256, 32, 164)
        self.assertTrue(verify_mitm_certificate(certificate))
        self.assertEqual(certificate.split_count, 511)
        self.assertEqual(certificate.weight_count, 32)
        self.assertAlmostEqual(certificate.attack.work_bits, 82.4461, places=3)
        self.assertEqual(certificate.attack.target_weight, 32)
        self.assertEqual(certificate.attack.left_weight, 17)
        self.assertEqual(certificate.attack.right_weight, 15)
        self.assertAlmostEqual(certificate.attack.left_list_bits, 82.4448, places=3)
        self.assertAlmostEqual(certificate.attack.right_list_bits, 82.4461, places=3)
        self.assertAlmostEqual(certificate.attack.expected_pair_excess_bits, 0.8909, places=3)
        self.assertEqual(certificate.attack.memory_bits, certificate.attack.work_bits)

    def test_executable_mitm_finds_verifier_valid_toy_share(self):
        params = PoWParams(n_columns=16, syndrome_bits=8, w_max=4, matched_bits=3)
        header = b"mitm-toy"
        columns, target = derive_instance(header, params, nonce=7)
        certificate = certify_mitm_attack(16, 8, 4, 3)
        result = execute_mitm_instance(
            columns, target, 8, 3, 4, certificate.attack,
            max_candidates=10_000,
        )
        self.assertTrue(result.found, result.summary())
        self.assertFalse(result.capped)
        self.assertTrue(
            verify_share(Share(header, 7, result.support), params).valid,
        )

    def test_executable_mitm_hard_cap_is_inconclusive(self):
        columns = tuple(index << 56 for index in range(16))
        certificate = certify_mitm_attack(16, 8, 4, 3)
        result = execute_mitm_instance(
            columns, 1 << 63, 8, 3, 4, certificate.attack,
            max_candidates=1,
        )
        self.assertTrue(result.capped)
        self.assertFalse(result.found)

    def test_executable_mitm_rejects_tampered_attack_parameters(self):
        from dataclasses import replace

        certificate = certify_mitm_attack(16, 8, 4, 3)
        tampered = replace(certificate.attack, matched_bits=4)
        with self.assertRaises(ValueError):
            execute_mitm_instance(
                tuple(range(16)), 0, 8, 3, 4, tampered,
            )

    def test_executable_mitm_exhaustive_no_result_is_not_capped(self):
        certificate = certify_mitm_attack(16, 8, 4, 3)
        columns = tuple(index << 56 for index in range(16))
        result = execute_mitm_instance(
            columns, 1 << 63, 8, 3, 4, certificate.attack,
            max_candidates=10_000,
        )
        self.assertFalse(result.found)
        self.assertFalse(result.capped)
        self.assertGreater(result.candidates_enumerated, 0)

    def test_executable_mitm_validates_selected_split_weights(self):
        from dataclasses import replace

        certificate = certify_mitm_attack(16, 8, 4, 3)
        tampered = replace(certificate.attack, left_weight=0)
        with self.assertRaises(ValueError):
            execute_mitm_instance(
                tuple(range(16)), 0, 8, 3, 4, tampered,
            )

    def test_executable_mitm_large_parameter_run_is_explicitly_capped(self):
        certificate = certify_mitm_attack(512, 256, 32, 164)
        columns = tuple(range(512))
        result = execute_mitm_instance(
            columns, 0, 256, 164, 32, certificate.attack,
            max_candidates=1_000,
        )
        self.assertTrue(result.capped)
        self.assertFalse(result.found)
        self.assertEqual(result.summary()["capped"], True)


if __name__ == "__main__":
    unittest.main()
