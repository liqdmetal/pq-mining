"""Tests for the bounded executable partial-syndrome solver."""

import unittest

from pqmine.partial_isd import (
    PartialISDError,
    PartialISDParams,
    solve_and_verify,
    solve_partial_mitm,
)
from pqmine.verifier import PoWParams


class PartialISDTests(unittest.TestCase):
    def test_finds_toy_partial_prefix_and_uses_set_support(self):
        shift = 60
        columns = tuple(value << shift for value in (0b1000, 0b0100, 0b0010, 0b0001))
        result = solve_partial_mitm(
            columns, 0b1100 << shift,
            PartialISDParams(syndrome_bits=4, matched_bits=4, max_weight=2),
        )
        self.assertTrue(result.found, result.summary())
        self.assertEqual(len(result.support), len(set(result.support)))
        acc = 0
        for index in result.support:
            acc ^= columns[index]
        self.assertEqual(acc, 0b1100 << shift)
        self.assertFalse(result.capped)

    def test_sweeps_lower_weights(self):
        shift = 60
        columns = tuple(value << shift for value in (1, 2, 4, 8))
        result = solve_partial_mitm(
            columns, 1 << shift,
            PartialISDParams(syndrome_bits=4, matched_bits=4, max_weight=3),
        )
        self.assertTrue(result.found)
        self.assertEqual(result.weight, 1)
        self.assertEqual(result.support, (0,))

    def test_partial_prefix_is_not_full_syndrome(self):
        shift = 60
        columns = (0b1000 << shift, 0b0100 << shift)
        # The target has the same top two bits as column 0 but a different
        # lower payload; only the matched prefix is relevant.
        target = (0b10 << 62) | (0b0011 << shift)
        result = solve_partial_mitm(
            columns, target,
            PartialISDParams(syndrome_bits=4, matched_bits=2, max_weight=1),
        )
        self.assertTrue(result.found)
        self.assertEqual(result.support, (0,))

    def test_candidate_cap_is_hard(self):
        shift = 56
        columns = tuple(value << shift for value in range(1, 17))
        result = solve_partial_mitm(
            columns, 0b11111111 << shift,
            PartialISDParams(
                syndrome_bits=8, matched_bits=8, max_weight=1,
                max_candidates=3,
            ),
        )
        self.assertTrue(result.capped)
        self.assertLessEqual(result.candidates_enumerated, 3)

    def test_bad_parameters_rejected(self):
        with self.assertRaises(PartialISDError):
            solve_partial_mitm(
                (1, 2), 0,
                PartialISDParams(syndrome_bits=8, matched_bits=0, max_weight=1),
            )

    def test_real_verifier_instance_is_accepted(self):
        pow_params = PoWParams(
            n_columns=16, syndrome_bits=8, w_max=4, matched_bits=3,
        )
        result, share = solve_and_verify(
            b"partial-isd-toy", 0, pow_params,
            PartialISDParams(
                syndrome_bits=8, matched_bits=3, max_weight=4,
                max_candidates=100_000,
            ),
        )
        self.assertTrue(result.found, result.summary())
        self.assertIsNotNone(share)


if __name__ == "__main__":
    unittest.main()
