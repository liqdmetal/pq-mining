"""Tests for explicit PoW candidate profile screens."""

import unittest

from pqmine.profiles import candidate_128_profile, screen_profile
from pqmine.verifier import PoWParams


class ProfileScreenTests(unittest.TestCase):
    def test_candidate_profile_clears_current_modeled_floor(self):
        screen = candidate_128_profile()
        self.assertTrue(screen.passes_modeled_floor, screen.summary())
        self.assertEqual(screen.params.w_max, 60)
        self.assertEqual(screen.params.matched_bits, 256)
        self.assertGreaterEqual(screen.classical_attack.work_bits, 128.0)
        self.assertEqual(screen.quantum_attack.grover_query_bits, 128.0)
        self.assertGreaterEqual(screen.expected_solutions_bits, 4.0)

    def test_old_toy_research_profile_does_not_pass(self):
        screen = screen_profile(
            "old-toy-research",
            PoWParams(n_columns=512, syndrome_bits=256,
                      w_max=32, matched_bits=164),
        )
        self.assertFalse(screen.passes_modeled_floor)
        self.assertLess(screen.quantum_attack.grover_query_bits, 128.0)

    def test_summary_exposes_nonclaims(self):
        summary = candidate_128_profile().summary()
        self.assertTrue(summary["passes_current_modeled_floor"])
        self.assertIn("not a PQ security claim", summary["warning"])


if __name__ == "__main__":
    unittest.main()
