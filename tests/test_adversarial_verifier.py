"""Adversarial verifier suite: every mutation class of a genuine share.

The consensus rule is attacked systematically rather than one-off:

  - every single index flip (differential vs an independent oracle);
  - every out-of-set index swap (differential);
  - duplicate padding at every position;
  - removal of every element (differential -- weight is "at or under the
    cap", so a removal that still XORs correctly is accepted by design);
  - order permutation (XOR commutes: the verifier is order-agnostic);
  - exact w_max boundary acceptance/rejection;
  - pre-allocation rejection of overweight supports at real params;
  - uint64 nonce boundaries, header type/length edges, params validation
    boundaries, malformed share shapes.

The oracle recomputes the relation from the public derivation API, so it
cannot silently agree with a broken verifier on the same bug.
"""

import time
import unittest

from pqmine.verifier import (
    PoWParams, Share, ShareError, derive_columns, derive_target,
    syndrome_mask, verify_share,
)
from tests.test_verifier import find_share


def oracle(header: bytes, nonce: int, support, params: PoWParams) -> bool:
    """Independent recomputation of the relation's validity."""
    try:
        support = tuple(support)
    except (TypeError, ValueError):
        return False
    if not support:
        return False
    if any(isinstance(j, bool) or not isinstance(j, int) for j in support):
        return False
    if len(set(support)) != len(support):
        return False
    if any(not (0 <= j < params.n_columns) for j in support):
        return False
    if len(support) > params.w_max:
        return False
    cols = derive_columns(header, params, nonce)
    target = derive_target(header, params, nonce)
    acc = 0
    for j in support:
        acc ^= cols[j]
    mask = syndrome_mask(params)
    return (acc & mask) == (target & mask)


class AdversarialVerifierTests(unittest.TestCase):
    def setUp(self):
        # Matched_bits=16 keeps the removal/swap differential deterministic
        # at flake odds below 1e-5 (no assertion is probabilistic: verdicts
        # are compared to the oracle, never guessed).
        self.params = PoWParams(n_columns=64, syndrome_bits=64, w_max=4,
                                matched_bits=16)
        self.header = b"adversarial-verifier-header"
        self.share = find_share(self.header, self.params)
        self.assertIsNotNone(self.share, "fixture must find a real share")
        self.assertEqual(verify_share(self.share, self.params).valid, True)
        self.assertEqual(oracle(self.header, 0, self.share.support, self.params), True)
        self.support = list(self.share.support)

    def check(self, support, expect_valid, reason_hint=None):
        v = verify_share(Share(self.header, 0, tuple(support)), self.params)
        self.assertEqual(v.valid, expect_valid, v.reason)
        self.assertEqual(
            v.valid, oracle(self.header, 0, support, self.params),
            f"verifier disagrees with oracle on {support!r}",
        )
        if reason_hint:
            self.assertIn(reason_hint, v.reason)
        return v

    # --- mutation campaigns -------------------------------------------

    def test_every_single_index_flip_rejected(self):
        """Swap each in-set index for one out-of-set neighbour."""
        for i, j in enumerate(self.support):
            cand = (j + 1) % self.params.n_columns
            if cand in self.support:
                continue  # would create a duplicate -- covered separately
            s = self.support.copy()
            s[i] = cand
            with self.subTest(flip=i, cand=cand):
                v = self.check(s, False)

    def test_every_out_of_set_swap_rejected(self):
        """Replace each in-set index with every possible out-of-set index."""
        in_set = set(self.support)
        out_of_set = [c for c in range(self.params.n_columns) if c not in in_set]
        for i, j in enumerate(self.support):
            for c in out_of_set:
                s = self.support.copy()
                s[i] = c
                with self.subTest(replace=j, with_=c):
                    self.check(s, False)

    def test_every_duplicate_padding_position_rejected(self):
        """F_2 cancellation attack: one repeated column at every position.
        Rejection is guaranteed; the reason is either "duplicate" (when the
        padded length stays within w_max) or the weight > w_max pre-check
        (the verifier rejects overweight by length before any processing)."""
        for j in set(self.support):
            for pos in range(len(self.support) + 1):
                s = self.support.copy()
                s.insert(pos, j)
                with self.subTest(column=j, pos=pos):
                    if len(s) <= self.params.w_max:
                        self.check(s, False, reason_hint="duplicate")
                    else:
                        self.check(s, False)

    def test_all_identical_columns_rejected(self):
        for j in set(self.support):
            with self.subTest(column=j):
                self.check((j,) * self.params.w_max, False,
                           reason_hint="duplicate")

    def test_duplicate_in_middle_rejected(self):
        """Non-adjacent duplicate: rejected — as duplicate when within the
        cap, otherwise by the overweight pre-check first (by design)."""
        s = self.support + [self.support[0]]
        if len(s) <= self.params.w_max:
            self.check(s, False, reason_hint="duplicate")
        else:
            self.check(s, False)

    def test_removal_of_every_element_differential(self):
        """Removal breaks the relation except with 2^-matched_bits odds;
        we do not guess -- verdict is compared against the oracle."""
        for i in range(len(self.support)):
            s = self.support.copy()
            del s[i]
            with self.subTest(removed=i):
                expected = oracle(self.header, 0, s, self.params)
                self.check(s, expected)

    def test_order_permutation_order_agnostic(self):
        """XOR commutes: shuffled supports must verify identically."""
        import random
        rng = random.Random(20261101)
        s = self.support.copy()
        for trial in range(8):
            rng.shuffle(s)
            with self.subTest(trial=trial):
                v = verify_share(Share(self.header, 0, tuple(s)), self.params)
                self.assertTrue(v.valid, v.reason)
                self.assertEqual(v.weight, len(s))
                self.assertEqual(v.matched, self.params.matched_bits)

    # --- boundary semantics --------------------------------------------

    def test_exact_w_max_accepted_at_cap(self):
        params = PoWParams(n_columns=64, syndrome_bits=64,
                           w_max=len(self.support), matched_bits=16)
        v = verify_share(self.share, params)
        self.assertTrue(v.valid, v.reason)
        self.assertEqual(v.weight, self.params.w_max)

    def test_over_w_max_by_one_rejected(self):
        params = PoWParams(n_columns=64, syndrome_bits=64,
                           w_max=len(self.support) - 1, matched_bits=16)
        v = verify_share(self.share, params)
        self.assertFalse(v.valid)
        self.assertIn("w_max", v.reason)

    def test_full_syndrome_verification_accepted(self):
        params = PoWParams(n_columns=64, syndrome_bits=64, w_max=4,
                           matched_bits=64)
        v = verify_share(self.share, params)
        self.assertFalse(v.valid)  # 16-bit share cannot satisfy 64 bits
        v2 = verify_share(self.share, PoWParams(n_columns=64, syndrome_bits=64,
                                                w_max=4, matched_bits=16))
        self.assertEqual(v2.valid, True)

    # --- DoS guards and shapes ------------------------------------------

    def test_real_params_overweight_rejected_before_work(self):
        """At production params a 33-element support must be rejected by
        length alone -- no hashing of 33 columns, no allocation blowup."""
        params = PoWParams(n_columns=512, syndrome_bits=256, w_max=32,
                           matched_bits=164)
        support = tuple(range(33))  # distinct, in range, one over the cap
        t0 = time.perf_counter()
        v = verify_share(Share(b"h", 0, support), params)
        elapsed = time.perf_counter() - t0
        self.assertFalse(v.valid)
        self.assertIn("w_max", v.reason)
        self.assertLess(elapsed, 0.5, "overweight check must not do work")

    def test_giant_support_no_allocation_blowup(self):
        params = PoWParams(n_columns=512, syndrome_bits=256, w_max=32,
                           matched_bits=164)
        t0 = time.perf_counter()
        v = verify_share(Share(b"h", 0, list(range(2**20))), params)
        self.assertLess(time.perf_counter() - t0, 0.5)
        self.assertFalse(v.valid)
        self.assertIn("w_max", v.reason)

    def test_non_iterable_support_shapes(self):
        params = self.params
        for bad in (None, 42, 3.5, b"x"):
            with self.subTest(bad=bad):
                v = verify_share(Share(self.header, 0, bad), params)
                self.assertFalse(v.valid)
                self.assertIn("not an iterable", v.reason)

    def test_bool_and_negative_and_fractional_indices(self):
        for bad in ((True,), (-1,), (1.5,), (2**63,)):
            with self.subTest(bad=bad):
                v = verify_share(Share(self.header, 0, bad), self.params)
                self.assertFalse(v.valid)

    # --- instance binding edges ------------------------------------------

    def test_nonce_uint64_boundaries(self):
        self.assertEqual(len(derive_columns(self.header, self.params, 2**64 - 1)), 64)
        for bad in (2**64, -1, 1.5, "7"):
            with self.subTest(nonce=bad):
                with self.assertRaises(ShareError):
                    derive_columns(self.header, self.params, bad)

    def test_header_type_and_length_edges(self):
        alpha = derive_target(b"x", self.params, 0)
        omega = derive_target(b"x" * 1000, self.params, 0)
        self.assertIsInstance(alpha, int)
        self.assertIsInstance(omega, int)
        for bad in (b"", "", 7):
            with self.subTest(header=bad):
                with self.assertRaises(ShareError):
                    derive_columns(bad, self.params, 0)

    def test_replay_at_far_nonce_rejected(self):
        moved = Share(self.header, 2**32, self.share.support)
        self.assertFalse(verify_share(moved, self.params).valid)

    # --- params validation boundaries ------------------------------------

    def test_params_validation_boundaries(self):
        with self.assertRaises(ShareError):
            PoWParams(n_columns=0, syndrome_bits=64, w_max=4, matched_bits=16)
        with self.assertRaises(ShareError):
            PoWParams(n_columns=64, syndrome_bits=64, w_max=0, matched_bits=16)
        with self.assertRaises(ShareError):
            PoWParams(n_columns=64, syndrome_bits=64, w_max=64, matched_bits=16)
        with self.assertRaises(ShareError):
            PoWParams(n_columns=64, syndrome_bits=64, w_max=4, matched_bits=0)
        with self.assertRaises(ShareError):
            PoWParams(n_columns=64, syndrome_bits=32, w_max=4, matched_bits=64)
        # legal edges: full-syndrome cap and the 4096-bit code-length cap
        PoWParams(n_columns=512, syndrome_bits=4096, w_max=32, matched_bits=4096)
        with self.assertRaises(ShareError):
            PoWParams(n_columns=512, syndrome_bits=4097, w_max=32, matched_bits=4096)

    # --- determinism ------------------------------------------------------

    def test_double_verification_deterministic(self):
        a = verify_share(self.share, self.params)
        b = verify_share(self.share, self.params)
        self.assertEqual(a, b)
        self.assertEqual((a.valid, a.weight, a.matched), (b.valid, b.weight, b.matched))


if __name__ == "__main__":
    unittest.main()
