"""Tests for Chitin-PoW: correctness, cheap verification, and phase balance.

The tests that matter most are the adversarial ones (a share must not verify
if anything about it is wrong) and the measured phase balance, because the
CPU+GPU thesis is a CLAIM about measurements and the first version of the
design failed it.
"""

import unittest

from pqmine import (
    PoWError,
    PoWParameters,
    Share,
    centered,
    derive_matrix,
    expected_candidates_for_collision,
    generate_candidate,
    infinity_norm,
    mine,
    syndrome,
    verify_share,
)
from pqmine.balanced_miner import apply_flip, column, mine_balanced


class ParameterTests(unittest.TestCase):
    def test_rejects_degenerate_shapes(self):
        with self.assertRaises(PoWError):
            PoWParameters(n=4, m=4, q=97, beta=2)      # m must exceed n
        with self.assertRaises(PoWError):
            PoWParameters(n=2, m=32, q=97, beta=0)     # beta positive
        with self.assertRaises(PoWError):
            PoWParameters(n=2, m=32, q=97, beta=60)    # beta >= q/2 trivial
        with self.assertRaises(PoWError):
            PoWParameters(n=0, m=32, q=97, beta=2)

    def test_beta_at_or_above_half_q_is_refused(self):
        """Otherwise almost any vector satisfies the norm bound and the PoW
        is free -- a real footgun, so it fails closed."""
        with self.assertRaises(PoWError):
            PoWParameters(n=2, m=32, q=101, beta=50)

    def test_difficulty_grows_with_n_and_q(self):
        base = PoWParameters(n=2, m=32, q=97, beta=4).difficulty_bits
        self.assertGreater(
            PoWParameters(n=4, m=64, q=97, beta=4).difficulty_bits, base
        )
        self.assertGreater(
            PoWParameters(n=2, m=32, q=1021, beta=4).difficulty_bits, base
        )


class MatrixTests(unittest.TestCase):
    def setUp(self):
        self.params = PoWParameters(n=3, m=16, q=97, beta=4, sparsity=4)

    def test_matrix_is_deterministic_and_correctly_shaped(self):
        first = derive_matrix(b"header", self.params)
        second = derive_matrix(b"header", self.params)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        self.assertTrue(all(len(row) == 16 for row in first))
        self.assertTrue(all(0 <= v < 97 for row in first for v in row))

    def test_different_headers_give_different_matrices(self):
        self.assertNotEqual(
            derive_matrix(b"header-a", self.params),
            derive_matrix(b"header-b", self.params),
        )

    def test_shape_is_bound_into_the_derivation(self):
        """A different (n,m,q) must not reuse the same expansion, or two
        chains with different parameters could share work."""
        other = PoWParameters(n=3, m=16, q=193, beta=4, sparsity=4)
        self.assertNotEqual(
            derive_matrix(b"header", self.params)[0][0:4],
            derive_matrix(b"header", other)[0][0:4],
        )

    def test_empty_header_refused(self):
        with self.assertRaises(PoWError):
            derive_matrix(b"", self.params)

    def test_centered_representative(self):
        self.assertEqual(centered(96, 97), -1)
        self.assertEqual(centered(1, 97), 1)
        self.assertEqual(centered(48, 97), 48)
        self.assertEqual(centered(49, 97), -48)


class MiningTests(unittest.TestCase):
    def setUp(self):
        self.params = PoWParameters(n=2, m=32, q=97, beta=4, sparsity=4)

    def test_mines_a_valid_share(self):
        result = mine(b"block-header-0001", self.params,
                      max_candidates=50_000, batch_size=512)
        self.assertTrue(result.found, "reference miner must find a share")
        verification = verify_share(result.share, self.params)
        self.assertTrue(verification.valid, verification.reason())

    def test_share_is_a_genuine_kernel_vector(self):
        result = mine(b"block-header-0001", self.params,
                      max_candidates=50_000, batch_size=512)
        matrix = derive_matrix(b"block-header-0001", self.params)
        residue = syndrome(matrix, result.share.vector, self.params.q)
        self.assertTrue(all(v == 0 for v in residue), "A z must be 0 mod q")

    def test_share_respects_the_norm_bound(self):
        result = mine(b"block-header-0001", self.params,
                      max_candidates=50_000, batch_size=512)
        self.assertLessEqual(
            infinity_norm(result.share.vector, self.params.q), self.params.beta
        )

    def test_share_is_nonzero(self):
        result = mine(b"block-header-0001", self.params,
                      max_candidates=50_000, batch_size=512)
        self.assertTrue(any(v != 0 for v in result.share.vector))

    def test_birthday_estimate_is_the_right_order(self):
        """The estimate should not be wildly off the observed attempts."""
        estimate = expected_candidates_for_collision(self.params)
        self.assertGreater(estimate, 1.0)
        self.assertLess(estimate, 10_000.0)

    def test_rejects_bad_search_bounds(self):
        with self.assertRaises(PoWError):
            mine(b"h", self.params, max_candidates=0)
        with self.assertRaises(PoWError):
            mine(b"h", self.params, prefix=99)


class VerificationAttackTests(unittest.TestCase):
    """A share must fail if ANYTHING about it is wrong."""

    def setUp(self):
        self.params = PoWParameters(n=2, m=32, q=97, beta=4, sparsity=4)
        self.header = b"attack-header"
        self.result = mine(self.header, self.params,
                           max_candidates=50_000, batch_size=512)
        self.assertTrue(self.result.found)

    def test_zero_vector_is_rejected(self):
        share = Share(self.header, 0, tuple([0] * self.params.m))
        verification = verify_share(share, self.params)
        self.assertFalse(verification.valid)
        self.assertIn("zero vector", verification.reason())

    def test_wrong_dimension_is_rejected(self):
        share = Share(self.header, 0, self.result.share.vector[:-1])
        verification = verify_share(share, self.params)
        self.assertFalse(verification.valid)
        self.assertIn("dimension", verification.reason())

    def test_share_for_another_header_is_rejected(self):
        """Work must not transfer between blocks."""
        share = Share(b"different-header", 0, self.result.share.vector)
        self.assertFalse(verify_share(share, self.params).valid)

    def test_perturbed_vector_leaves_the_kernel(self):
        vector = list(self.result.share.vector)
        vector[0] = (vector[0] + 1) % self.params.q
        verification = verify_share(
            Share(self.header, 0, tuple(vector)), self.params
        )
        self.assertFalse(verification.valid)

    def test_large_norm_vector_is_rejected_even_if_in_kernel(self):
        """Scaling a kernel vector keeps it in the kernel but breaks the
        norm bound -- the norm gate must catch it."""
        scaled = tuple(
            (v * 20) % self.params.q for v in self.result.share.vector
        )
        verification = verify_share(Share(self.header, 0, scaled), self.params)
        matrix = derive_matrix(self.header, self.params)
        self.assertTrue(
            all(v == 0 for v in syndrome(matrix, scaled, self.params.q)),
            "scaled vector should still be in the kernel",
        )
        self.assertFalse(verification.valid)
        self.assertIn("norm", verification.reason())

    def test_verification_is_cheap(self):
        """The asymmetry the scheme rests on: verify must be fast."""
        import time

        started = time.perf_counter()
        for _ in range(200):
            verify_share(self.result.share, self.params)
        elapsed = time.perf_counter() - started
        self.assertLess(elapsed / 200, 0.01, "verification should be ~ms")


class PhaseBalanceTests(unittest.TestCase):
    """The CPU+GPU thesis is a measurement, and version 1 FAILED it."""

    def test_naive_miner_is_generation_dominated(self):
        """Documents the refuted claim: ~97% of work is Phase A, so a
        GPU-only miner would win and there is no cooperative property."""
        params = PoWParameters(n=3, m=48, q=97, beta=4, sparsity=4)
        result = mine(b"balance", params, max_candidates=20_000,
                      batch_size=4096)
        fraction = result.timing.generate_fraction
        self.assertGreater(
            fraction, 0.85,
            "if this drops, the naive-miner finding has changed and the "
            "docs in balanced_miner.py need revisiting",
        )

    def test_balanced_miner_shifts_work_to_the_table(self):
        """Incremental syndrome updates plus a data-dependent walk move a
        real share of the cost into the memory-bound phase."""
        params = PoWParameters(n=3, m=48, q=97, beta=4, sparsity=4)
        _, timing, _ = mine_balanced(b"balance", params, max_steps=20_000)
        self.assertGreater(
            timing.collide_fraction, 0.15,
            "the balanced miner must put >15% of time in the table phase",
        )

    def test_balanced_miner_is_more_balanced_than_naive(self):
        params = PoWParameters(n=3, m=48, q=97, beta=4, sparsity=4)
        naive = mine(b"cmp", params, max_candidates=20_000, batch_size=4096)
        _, balanced, _ = mine_balanced(b"cmp", params, max_steps=20_000)
        self.assertGreater(
            balanced.collide_fraction, naive.timing.collide_fraction * 3,
            "the redesign should improve balance by a large factor",
        )

    def test_balanced_miner_still_produces_valid_shares(self):
        params = PoWParameters(n=2, m=32, q=97, beta=4, sparsity=4)
        share, _, _ = mine_balanced(b"valid", params, max_steps=60_000)
        self.assertIsNotNone(share, "balanced miner must still find shares")
        self.assertTrue(verify_share(share, params).valid)


class IncrementalUpdateTests(unittest.TestCase):
    """The incremental update must equal a full recomputation."""

    def setUp(self):
        self.params = PoWParameters(n=3, m=16, q=97, beta=4, sparsity=4)
        self.matrix = derive_matrix(b"inc", self.params)

    def test_flip_matches_full_recomputation(self):
        vector = list(generate_candidate(b"inc", 0, 0, self.params))
        base = syndrome(self.matrix, vector, self.params.q)
        for position in (0, 3, 7, 15):
            for sign in (1, -1):
                incremental = apply_flip(
                    base, self.matrix, position, sign, self.params.q
                )
                mutated = list(vector)
                mutated[position] = (mutated[position] + sign) % self.params.q
                full = syndrome(self.matrix, mutated, self.params.q)
                with self.subTest(position=position, sign=sign):
                    self.assertEqual(incremental, full)

    def test_column_extraction(self):
        col = column(self.matrix, 2)
        self.assertEqual(col, tuple(row[2] for row in self.matrix))


if __name__ == "__main__":
    unittest.main()
