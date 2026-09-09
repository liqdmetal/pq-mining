"""Reference search must produce the same object the verifier accepts."""

import unittest

from pqmine.latency_bound import LatencyParameters
from pqmine.search_share import SearchShareError, search_reference
from pqmine.verifier import PoWParams, verify_share


class ReferenceSearchTests(unittest.TestCase):
    def setUp(self):
        self.pow = PoWParams(
            n_columns=64,
            syndrome_bits=32,
            w_max=8,
            matched_bits=4,
        )
        self.walk = LatencyParameters(
            arena_bits=8,
            steps=32,
            write_density=1,
            cell_bytes=8,
        )

    def test_found_share_is_verifier_compatible(self):
        result = search_reference(
            b"reference-search-test",
            self.pow,
            self.walk,
            target_weight=4,
            first_nonce=0,
            max_nonces=512,
        )
        self.assertTrue(result.found, result.summary())
        self.assertIsNotNone(result.share)
        self.assertTrue(verify_share(result.share, self.pow).valid)
        self.assertLessEqual(len(result.share.support), self.pow.w_max)
        self.assertEqual(len(set(result.share.support)), len(result.share.support))

    def test_nonce_is_part_of_the_verified_share(self):
        result = search_reference(
            b"nonce-binding-test",
            self.pow,
            self.walk,
            target_weight=4,
            max_nonces=512,
        )
        self.assertTrue(result.found, result.summary())
        moved = type(result.share)(
            result.share.header,
            result.share.nonce + 1,
            result.share.support,
        )
        self.assertFalse(verify_share(moved, self.pow).valid)

    def test_weight_cap_is_enforced_during_search(self):
        strict = PoWParams(
            n_columns=64,
            syndrome_bits=32,
            w_max=1,
            matched_bits=4,
        )
        result = search_reference(
            b"weight-cap-test",
            strict,
            self.walk,
            target_weight=8,
            max_nonces=4,
        )
        self.assertFalse(result.found)
        self.assertGreaterEqual(result.selected_weight, strict.w_max)

    def test_bad_inputs_rejected(self):
        with self.assertRaises(SearchShareError):
            search_reference(b"", self.pow, self.walk, target_weight=4)
        with self.assertRaises(SearchShareError):
            search_reference(
                b"x", self.pow, self.walk, target_weight=4, max_nonces=0,
            )


if __name__ == "__main__":
    unittest.main()
