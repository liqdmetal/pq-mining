"""Reference search must produce the same object the verifier accepts."""

import unittest
import time
from unittest.mock import patch

from pqmine.decoupled import selection_threshold
from pqmine.latency_bound import LatencyParameters
from pqmine.search_share import SearchShareError, search_reference
from pqmine.verifier import PoWParams, verify_share


class SelectionThresholdTests(unittest.TestCase):
    def test_full_target_selects_every_step(self):
        self.assertEqual(selection_threshold(32, 32), 256)
        self.assertEqual(selection_threshold(1, 1), 256)


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

    def test_nonce_boundary_reports_actual_coverage(self):
        result = search_reference(
            b"nonce-boundary-test",
            self.pow,
            self.walk,
            target_weight=4,
            first_nonce=(2**64) - 1,
            max_nonces=4,
        )
        self.assertLessEqual(result.nonces_tested, 1)
        self.assertEqual(result.nonces_tested, 1)

    def test_target_weight_above_columns_rejected(self):
        with self.assertRaises(SearchShareError):
            search_reference(
                b"too-many-columns", self.pow, self.walk,
                target_weight=self.pow.n_columns + 1,
                max_nonces=1,
            )

    def test_full_support_does_not_spin(self):
        from pqmine.decoupled_trace import trace_decoupled_walk
        from pqmine.latency_bound import build_arena

        tiny_walk = LatencyParameters(
            arena_bits=8, steps=64, write_density=1, cell_bytes=8,
        )
        trace = trace_decoupled_walk(
            build_arena(b"full-support-test", tiny_walk),
            0, tiny_walk, tuple(range(4)), target_weight=4,
        )
        self.assertEqual(len(trace.selected_columns), 4)
        self.assertEqual(len(set(trace.selected_columns)), 4)
        self.assertEqual(trace.steps, 64)


    def test_bad_inputs_rejected(self):
        with self.assertRaises(SearchShareError):
            search_reference(b"", self.pow, self.walk, target_weight=4)
        with self.assertRaises(SearchShareError):
            search_reference(
                b"x", self.pow, self.walk, target_weight=4, max_nonces=0,
            )

    def test_deadline_and_cancellation_stop_before_allocating(self):
        for kwargs, reason in (({"deadline": time.monotonic() - 1}, "deadline"),
                               ({"cancelled": lambda: True}, "cancelled")):
            with patch("pqmine.search_share.derive_instance") as derive:
                result = search_reference(b"x", self.pow, self.walk, target_weight=4, **kwargs)
                self.assertFalse(result.found)
                self.assertEqual(result.nonces_tested, 0)
                self.assertEqual(result.stop_reason, reason)
                derive.assert_not_called()

    def test_cancellation_during_walk_is_inconclusive(self):
        checks = 0
        def cancelled():
            nonlocal checks
            checks += 1
            return checks >= 3
        result = search_reference(b"x", self.pow, self.walk, target_weight=4,
                                  cancelled=cancelled)
        self.assertEqual(result.stop_reason, "cancelled")
        self.assertEqual(result.nonces_tested, 0)

    def test_resource_envelope_rejects_before_work(self):
        for kwargs in ({"max_arena_bytes": 1}, {"max_steps_per_nonce": 1},
                       {"max_columns": 1}, {"max_nonces": True},
                       {"first_nonce": True}, {"target_weight": True},
                       {"deadline": float("inf")}, {"cancelled": False},
                       {"max_rehashes_per_step": 0}):
            options = {"target_weight": 4, **kwargs}
            with self.assertRaises(SearchShareError):
                search_reference(b"x", self.pow, self.walk, **options)

    def test_collision_cycle_has_a_hard_bound(self):
        from pqmine.decoupled_trace import WalkInterrupted, trace_decoupled_walk
        from pqmine.latency_bound import build_arena
        with patch("pqmine.decoupled_trace.mix64", return_value=0):
            with self.assertRaisesRegex(WalkInterrupted, "collision_budget"):
                trace_decoupled_walk(build_arena(b"x", self.walk), 0,
                                     self.walk, tuple(range(64)), 4,
                                     max_rehashes_per_step=3)


if __name__ == "__main__":
    unittest.main()
