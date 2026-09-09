"""Verifier tests: the consensus rule, attacked.

The verifier is the only consensus-critical code in the PoW. Each rule is
tested by its own forgery. The F_2 duplicate attack is the one that would
have been catastrophic: pad a weight-2 solution to weight-6 for free.
"""

import unittest

from pqmine.verifier import (
    PoWParams, Share, ShareError, derive_columns, derive_instance, derive_target,
    expected_work_log2, verify_share,
)


def find_share(header: bytes, params: PoWParams, max_tries: int = 200000):
    """Brute-force a genuine share for a SMALL difficulty, for tests."""
    import itertools
    cols = derive_columns(header, params)
    target = derive_target(header, params)
    total = 64 * ((params.syndrome_bits + 63) // 64)
    mask = ((1 << params.matched_bits) - 1) << (total - params.matched_bits)
    tries = 0
    for w in range(1, params.w_max + 1):
        for support in itertools.combinations(range(params.n_columns), w):
            acc = 0
            for j in support:
                acc ^= cols[j]
            if (acc & mask) == (target & mask):
                return Share(header, 0, tuple(support))
            tries += 1
            if tries > max_tries:
                return None
    return None


class VerifierRuleTests(unittest.TestCase):
    def setUp(self):
        # Tiny difficulty so a genuine share is findable by brute force.
        self.params = PoWParams(n_columns=64, syndrome_bits=64, w_max=4,
                                matched_bits=10)
        self.header = b"verifier-test-header"
        self.share = find_share(self.header, self.params)
        self.assertIsNotNone(self.share, "fixture must find a real share")

    def test_genuine_share_accepted(self):
        v = verify_share(self.share, self.params)
        self.assertTrue(v.valid, v.reason)
        self.assertEqual(v.matched, self.params.matched_bits)
        self.assertLessEqual(v.weight, self.params.w_max)

    def test_duplicate_column_rejected(self):
        """THE attack: over F_2, x^x=0, so a repeated column pads weight
        without changing the syndrome. Must be refused before any XOR."""
        j = self.share.support[0]
        padded = Share(self.header, 0, self.share.support + (j, j))
        v = verify_share(padded, self.params)
        self.assertFalse(v.valid)
        self.assertIn("duplicate", v.reason)

    def test_over_weight_rejected(self):
        heavy = PoWParams(n_columns=64, syndrome_bits=64, w_max=2, matched_bits=10)
        if len(self.share.support) > 2:
            v = verify_share(self.share, heavy)
            self.assertFalse(v.valid)
            self.assertIn("w_max", v.reason)

    def test_wrong_header_rejected(self):
        """H and s come from the bound header/nonce instance."""
        moved = Share(b"other-header", self.share.nonce, self.share.support)
        v = verify_share(moved, self.params)
        self.assertFalse(v.valid)
        self.assertIn("syndrome", v.reason)

    def test_nonce_replay_rejected(self):
        """A support valid for nonce n must not replay at nonce n+1."""
        moved = Share(self.header, self.share.nonce + 1, self.share.support)
        v = verify_share(moved, self.params)
        self.assertFalse(v.valid)
        self.assertIn("syndrome", v.reason)

    def test_derive_instance_matches_components_and_nonce_changes(self):
        cols, target = derive_instance(self.header, self.params, 0)
        self.assertEqual(cols, derive_columns(self.header, self.params, 0))
        self.assertEqual(target, derive_target(self.header, self.params, 0))
        cols_next, target_next = derive_instance(self.header, self.params, 1)
        self.assertNotEqual(cols, cols_next)
        self.assertNotEqual(target, target_next)

    def test_nonce_is_bound_in_the_instance(self):
        a = derive_columns(self.header, self.params, 0)
        b = derive_columns(self.header, self.params, 1)
        self.assertNotEqual(a, b)
        self.assertNotEqual(
            derive_target(self.header, self.params, 0),
            derive_target(self.header, self.params, 1),
        )

    def test_perturbed_support_rejected(self):
        sup = list(self.share.support)
        sup[0] = (sup[0] + 1) % self.params.n_columns
        if len(set(sup)) == len(sup):
            v = verify_share(Share(self.header, 0, tuple(sorted(sup))), self.params)
            self.assertFalse(v.valid)

    def test_out_of_range_rejected(self):
        v = verify_share(Share(self.header, 0, (0, 999)), self.params)
        self.assertFalse(v.valid)
        self.assertIn("range", v.reason)

    def test_empty_support_rejected(self):
        v = verify_share(Share(self.header, 0, ()), self.params)
        self.assertFalse(v.valid)

    def test_harder_difficulty_rejects_easy_share(self):
        """Raising matched_bits must not retroactively accept."""
        harder = PoWParams(n_columns=64, syndrome_bits=64, w_max=4, matched_bits=40)
        v = verify_share(self.share, harder)
        # Almost certainly fails the extra 30 bits.
        self.assertFalse(v.valid)


class ParamsTests(unittest.TestCase):
    def test_spec_defaults(self):
        p = PoWParams()
        self.assertEqual((p.n_columns, p.syndrome_bits, p.w_max), (512, 256, 32))
        self.assertEqual(p.matched_bits, 164)

    def test_invalid_params_rejected(self):
        with self.assertRaises(ShareError):
            PoWParams(matched_bits=0)
        with self.assertRaises(ShareError):
            PoWParams(matched_bits=300)
        with self.assertRaises(ShareError):
            PoWParams(w_max=600)

    def test_expected_work_is_matched_bits(self):
        self.assertEqual(expected_work_log2(PoWParams(matched_bits=164)), 164.0)


if __name__ == "__main__":
    unittest.main()
