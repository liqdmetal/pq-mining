"""Tests for the loopback-only Chitin-PoW testnet PoC."""

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from pqmine.verifier import PoWParams, Share
from pqmine.testnet import (
    Block,
    BlockHeader,
    Chain,
    ChainParams,
    LocalNode,
    TestnetError,
    make_genesis,
    mine_next_block,
    validate_chain,
    validate_block,
)


class TestnetConsensusTests(unittest.TestCase):
    def setUp(self):
        self.params = ChainParams(
            pow=PoWParams(n_columns=32, syndrome_bits=32, w_max=4, matched_bits=8),
            max_future_seconds=3600,
        )
        self.chain = Chain(self.params, miner_id="alice")

    def test_genesis_and_mined_block_validate(self):
        block = mine_next_block(self.chain, self.params, miner_id="alice", max_nonces=128)
        self.assertTrue(validate_block(block, self.chain.tip, self.params))
        self.assertTrue(self.chain.add_block(block))
        self.assertEqual(self.chain.height, 1)
        self.assertTrue(validate_chain(self.chain.blocks, self.params))

    def test_wrong_nonce_replay_is_rejected(self):
        block = mine_next_block(self.chain, self.params, miner_id="alice", max_nonces=128)
        replay = Block(block.header, block.nonce + 1, block.support)
        self.assertFalse(validate_block(replay, self.chain.tip, self.params))

    def test_duplicate_support_is_rejected(self):
        header = BlockHeader(
            version=1, height=1, previous_hash=self.chain.tip.block_id,
            timestamp=1, miner_id="alice", matched_bits=self.params.pow.matched_bits,
        )
        bad = Block(header, nonce=0, support=(1, 1))
        self.assertFalse(validate_block(bad, self.chain.tip, self.params))

    def test_chain_rejects_wrong_parent_and_height(self):
        block = mine_next_block(self.chain, self.params, miner_id="alice", max_nonces=128)
        self.assertTrue(self.chain.add_block(block))
        wrong_header = BlockHeader(
            version=1, height=3, previous_hash="00" * 32,
            timestamp=2, miner_id="alice", matched_bits=self.params.pow.matched_bits,
        )
        wrong = Block(wrong_header, nonce=0, support=(1,))
        self.assertFalse(self.chain.add_block(wrong))

    def test_json_round_trip_preserves_block(self):
        block = mine_next_block(self.chain, self.params, miner_id="alice", max_nonces=128)
        decoded = Block.from_dict(block.to_dict())
        self.assertEqual(decoded, block)
        self.assertEqual(decoded.block_id, block.block_id)

    def test_competing_valid_branch_with_more_work_wins(self):
        first = mine_next_block(self.chain, self.params, miner_id="alice", max_nonces=128)
        self.assertTrue(self.chain.add_block(first))
        side_parent = self.chain.genesis
        side = mine_next_block(
            Chain.from_blocks(self.params, [side_parent]),
            self.params,
            miner_id="bob",
            max_nonces=128,
        )
        self.assertTrue(self.chain.add_block(side))
        self.assertEqual(self.chain.tip.block_id, max(first.block_id, side.block_id))
        self.assertEqual(self.chain.height, 1)

    def test_longer_side_branch_reorganizes_tip(self):
        first = mine_next_block(self.chain, self.params, miner_id="alice", max_nonces=128)
        self.assertTrue(self.chain.add_block(first))
        side_chain = Chain.from_blocks(self.params, [self.chain.genesis])
        side_one = mine_next_block(side_chain, self.params, miner_id="bob", max_nonces=128)
        self.assertTrue(side_chain.add_block(side_one))
        side_two = mine_next_block(side_chain, self.params, miner_id="bob", max_nonces=128)
        self.assertTrue(side_chain.add_block(side_two))
        self.assertTrue(self.chain.add_block(side_one))
        self.assertTrue(self.chain.add_block(side_two))
        self.assertEqual(self.chain.tip.block_id, side_two.block_id)
        self.assertEqual(
            [b.block_id for b in self.chain.blocks],
            [self.chain.genesis.block_id, side_one.block_id, side_two.block_id],
        )

    def test_equal_height_tie_uses_deterministic_block_id(self):
        left = mine_next_block(self.chain, self.params, miner_id="left", max_nonces=128)
        right = mine_next_block(
            Chain.from_blocks(self.params, [self.chain.genesis]),
            self.params,
            miner_id="right",
            max_nonces=128,
        )
        self.assertTrue(self.chain.add_block(left))
        self.assertTrue(self.chain.add_block(right))
        self.assertEqual(self.chain.tip.block_id, max(left.block_id, right.block_id))

    def test_equal_height_tie_is_stable_after_persistence(self):
        left = mine_next_block(self.chain, self.params, miner_id="left", max_nonces=128)
        right = mine_next_block(
            Chain.from_blocks(self.params, [self.chain.genesis]),
            self.params,
            miner_id="right",
            max_nonces=128,
        )
        self.assertTrue(self.chain.add_block(left))
        self.assertTrue(self.chain.add_block(right))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tie.json"
            self.chain.save(path)
            restored = Chain.load(path, self.params)
        self.assertEqual(restored.tip.block_id, self.chain.tip.block_id)
        self.assertEqual(restored.height, 1)

    def test_duplicate_block_is_rejected(self):
        block = mine_next_block(self.chain, self.params, miner_id="alice", max_nonces=128)
        self.assertTrue(self.chain.add_block(block))
        self.assertFalse(self.chain.add_block(block))

    def test_profile_id_is_bound_to_genesis_and_header(self):
        self.assertEqual(self.chain.profile_id, self.params.profile_id)
        self.assertIn(self.params.profile_id, self.chain.genesis.header.profile_id)
        block = mine_next_block(self.chain, self.params, miner_id="alice", max_nonces=128)
        self.assertEqual(block.header.profile_id, self.params.profile_id)

    def test_persistence_restart_round_trip(self):
        block = mine_next_block(self.chain, self.params, miner_id="alice", max_nonces=128)
        self.assertTrue(self.chain.add_block(block))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chain.json"
            self.chain.save(path)
            restored = Chain.load(path, self.params)
            self.assertEqual(restored.height, self.chain.height)
            self.assertEqual(restored.tip.block_id, self.chain.tip.block_id)

    def test_persistence_keeps_side_branch(self):
        first = mine_next_block(self.chain, self.params, miner_id="alice", max_nonces=128)
        self.assertTrue(self.chain.add_block(first))
        side = mine_next_block(
            Chain.from_blocks(self.params, [self.chain.genesis]),
            self.params, miner_id="bob", max_nonces=128,
        )
        self.assertTrue(self.chain.add_block(side))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chain.json"
            self.chain.save(path)
            restored = Chain.load(path, self.params)
            self.assertTrue(restored.has_block(first.block_id))
            self.assertTrue(restored.has_block(side.block_id))

    def test_profile_mismatch_is_rejected(self):
        other = ChainParams(
            pow=PoWParams(n_columns=32, syndrome_bits=32, w_max=4, matched_bits=7),
            max_future_seconds=3600,
        )
        block = mine_next_block(self.chain, self.params, miner_id="alice", max_nonces=128)
        self.assertFalse(Chain(other).add_block(block))


class DifficultyConsensusTests(unittest.TestCase):

    def setUp(self):
        self.params = ChainParams(
            pow=PoWParams(n_columns=32, syndrome_bits=32, w_max=4, matched_bits=8),
            max_future_seconds=3600,
            target_block_seconds=10,
            retarget_interval=2,
            max_retarget_step=4,
        )

    def test_header_commits_initial_difficulty(self):
        chain = Chain(self.params)
        block = mine_next_block(
            chain, self.params, miner_id="alice", max_nonces=128, timestamp=2,
        )
        self.assertEqual(block.header.matched_bits, 8)
        self.assertTrue(chain.add_block(block))

    def test_retarget_changes_difficulty_only_at_window_boundary(self):
        chain = Chain(self.params)
        first = mine_next_block(
            chain, self.params, miner_id="alice", max_nonces=128, timestamp=2,
        )
        self.assertTrue(chain.add_block(first))
        second = mine_next_block(
            chain, self.params, miner_id="alice", max_nonces=128, timestamp=4,
        )
        self.assertEqual(second.header.matched_bits, 8)
        self.assertTrue(chain.add_block(second))
        third = mine_next_block(
            chain, self.params, miner_id="alice", max_nonces=128, timestamp=6,
        )
        self.assertEqual(third.header.matched_bits, 9)
        self.assertTrue(chain.add_block(third))
        fourth = mine_next_block(
            chain, self.params, miner_id="alice", max_nonces=128, timestamp=8,
        )
        self.assertEqual(fourth.header.matched_bits, 9)

    def test_fast_window_increases_difficulty(self):
        chain = Chain(self.params)
        first = mine_next_block(
            chain, self.params, miner_id="alice", max_nonces=128, timestamp=1,
        )
        self.assertTrue(chain.add_block(first))
        second = mine_next_block(
            chain, self.params, miner_id="alice", max_nonces=128, timestamp=2,
        )
        self.assertTrue(chain.add_block(second))
        third = mine_next_block(
            chain, self.params, miner_id="alice", max_nonces=128, timestamp=3,
        )
        self.assertEqual(third.header.matched_bits, 9)

    def test_target_window_holds_difficulty(self):
        chain = Chain(self.params)
        first = mine_next_block(
            chain, self.params, miner_id="alice", max_nonces=128, timestamp=10,
        )
        self.assertTrue(chain.add_block(first))
        second = mine_next_block(
            chain, self.params, miner_id="alice", max_nonces=128, timestamp=30,
        )
        self.assertTrue(chain.add_block(second))
        third = mine_next_block(
            chain, self.params, miner_id="alice", max_nonces=128, timestamp=50,
        )
        self.assertEqual(third.header.matched_bits, 8)

    def test_non_monotonic_timestamp_is_rejected(self):
        chain = Chain(self.params)
        first = mine_next_block(
            chain, self.params, miner_id="alice", max_nonces=128, timestamp=10,
        )
        self.assertTrue(chain.add_block(first))
        second = Block(
            BlockHeader(
                version=1,
                height=2,
                previous_hash=chain.tip.block_id,
                timestamp=9,
                miner_id="alice",
                matched_bits=self.params.pow.matched_bits,
                profile_id=self.params.profile_id,
            ),
            nonce=0,
            support=(),
        )
        self.assertFalse(chain.add_block(second))

    def test_wrong_retargeted_difficulty_is_rejected(self):
        chain = Chain(self.params)
        first = mine_next_block(
            chain, self.params, miner_id="alice", max_nonces=128, timestamp=2,
        )
        self.assertTrue(chain.add_block(first))
        second = mine_next_block(
            chain, self.params, miner_id="alice", max_nonces=128, timestamp=4,
        )
        self.assertTrue(chain.add_block(second))
        expected = mine_next_block(
            chain, self.params, miner_id="alice", max_nonces=128, timestamp=6,
        )
        forged_header = BlockHeader(
            version=expected.header.version,
            height=expected.header.height,
            previous_hash=expected.header.previous_hash,
            timestamp=expected.header.timestamp,
            miner_id=expected.header.miner_id,
            profile_id=expected.header.profile_id,
            matched_bits=expected.header.matched_bits + 1,
        )
        forged = Block(forged_header, expected.nonce, expected.support)
        self.assertFalse(chain.add_block(forged))


class TestnetNodeTests(unittest.TestCase):

    def setUp(self):
        self.params = ChainParams(
            pow=PoWParams(n_columns=32, syndrome_bits=32, w_max=4, matched_bits=8),
            max_future_seconds=3600,
        )
        self.nodes = []

    def tearDown(self):
        for node in self.nodes:
            node.stop()

    def test_two_loopback_nodes_sync_a_mined_block(self):
        a = LocalNode(self.params, "alice")
        b = LocalNode(self.params, "bob")
        self.nodes = [a, b]
        a.start(port=0)
        b.start(port=0)
        a.peers = [b.url]
        block = a.mine_and_broadcast(max_nonces=128)
        self.assertEqual(a.chain.height, 1)
        self.assertEqual(b.chain.tip.block_id, block.block_id)
        self.assertFalse(b.receive_block(block))  # duplicate broadcast is rejected
        self.assertEqual(b.chain.height, 1)
        status = json.loads(urlopen(b.url + "/status", timeout=2).read())
        self.assertEqual(status["height"], 1)

    def test_http_chain_endpoint_is_loopback_testnet_only(self):
        node = LocalNode(self.params, "alice")
        self.nodes = [node]
        node.start(port=0)
        raw = urlopen(node.url + "/chain", timeout=2).read()
        payload = json.loads(raw)
        self.assertEqual(payload["network"], "chitin-pow-loopback-testnet")
        self.assertEqual(payload["height"], 0)
        self.assertEqual(payload["profile_id"], self.params.profile_id)

    def test_oversized_and_malformed_block_payloads_are_rejected(self):
        node = LocalNode(self.params, "alice")
        self.nodes = [node]
        node.start(port=0)
        for payload in (b"not-json", b"{}", b"x" * (node.params.max_payload_bytes + 1)):
            request = Request(
                node.url + "/block", data=payload,
                headers={"Content-Type": "application/json"}, method="POST",
            )
            with self.assertRaises(HTTPError) as error:
                urlopen(request, timeout=2)
            self.assertEqual(error.exception.code, 400)

    def test_http_profile_mismatch_is_rejected(self):
        node = LocalNode(self.params, "alice")
        self.nodes = [node]
        node.start(port=0)
        other = ChainParams(
            pow=PoWParams(n_columns=32, syndrome_bits=32, w_max=4, matched_bits=7),
            max_future_seconds=3600,
        )
        foreign = mine_next_block(Chain(other), other, miner_id="foreign", max_nonces=128)
        request = Request(
            node.url + "/block", data=json.dumps(foreign.to_dict()).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with self.assertRaises(HTTPError) as error:
            urlopen(request, timeout=2)
        self.assertEqual(error.exception.code, 400)

    def test_http_sync_imports_a_branch(self):
        source = LocalNode(self.params, "source")
        target = LocalNode(self.params, "target")
        self.nodes = [source, target]
        source.start(port=0)
        target.start(port=0)
        block = source.mine_and_broadcast(max_nonces=128)
        # Target did not receive it; import the full source chain explicitly.
        request = Request(
            target.url + "/sync", data=json.dumps(source.chain.to_dict()).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urlopen(request, timeout=2) as response:
            payload = json.loads(response.read())
        self.assertTrue(payload["accepted"])
        self.assertEqual(target.chain.tip.block_id, block.block_id)
        self.assertEqual(payload["height"], 1)

    def test_http_sync_rejects_wrong_profile(self):
        node = LocalNode(self.params, "alice")
        self.nodes = [node]
        node.start(port=0)
        other = ChainParams(
            pow=PoWParams(n_columns=32, syndrome_bits=32, w_max=4, matched_bits=7),
            max_future_seconds=3600,
        )
        foreign = LocalNode(other, "foreign")
        foreign.start(port=0)
        try:
            request = Request(
                node.url + "/sync", data=json.dumps(foreign.chain.to_dict()).encode(),
                headers={"Content-Type": "application/json"}, method="POST",
            )
            with self.assertRaises(HTTPError) as error:
                urlopen(request, timeout=2)
            self.assertEqual(error.exception.code, 400)
        finally:
            foreign.stop()

    def test_http_sync_payload_limit_applies(self):
        node = LocalNode(self.params, "alice")
        self.nodes = [node]
        node.start(port=0)
        payload = b"x" * (node.params.max_payload_bytes + 1)
        request = Request(
            node.url + "/sync", data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with self.assertRaises(HTTPError) as error:
            urlopen(request, timeout=2)
        self.assertEqual(error.exception.code, 400)

    def test_sync_import_is_transactional_on_orphan(self):
        source = LocalNode(self.params, "source")
        target = LocalNode(self.params, "target")
        self.nodes = [source, target]
        source.start(port=0)
        target.start(port=0)
        first = source.mine_and_broadcast(max_nonces=128)
        payload = source.chain.to_dict()
        payload["blocks"].append({
            "header": {
                "version": 1, "height": 99, "previous_hash": "11" * 32,
                "timestamp": 99, "miner_id": "orphan",
                "profile_id": self.params.profile_id,
            },
            "nonce": 0, "support": [0],
        })
        before = target.chain.tip.block_id
        self.assertFalse(target.receive_chain(payload))
        self.assertEqual(target.chain.tip.block_id, before)
        self.assertFalse(target.chain.has_block(first.block_id))

    def test_restart_loads_persisted_chain(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "node.json"
            first = LocalNode(self.params, "first", state_path=path)
            first.start(port=0)
            block = first.mine_and_broadcast(max_nonces=128)
            first.stop()
            restarted = LocalNode(self.params, "restarted", state_path=path)
            self.nodes.append(restarted)
            restarted.start(port=0)
            self.assertEqual(restarted.chain.tip.block_id, block.block_id)
            self.assertEqual(restarted.chain.height, 1)
            restarted.stop()

    def test_reconcile_peer_chain_after_missed_blocks(self):
        source = LocalNode(self.params, "source")
        target = LocalNode(self.params, "target")
        self.nodes = [source, target]
        source.start(port=0)
        target.start(port=0)
        source.mine_and_broadcast(max_nonces=128)
        source.mine_and_broadcast(max_nonces=128)
        self.assertEqual(target.chain.height, 0)
        self.assertTrue(target.reconcile_peer(source.url))
        self.assertEqual(target.chain.tip.block_id, source.chain.tip.block_id)

    def test_automatic_peer_sync_recovers_missed_blocks(self):
        source = LocalNode(self.params, "source")
        target = LocalNode(self.params, "target")
        self.nodes = [source, target]
        source.start(port=0)
        target.start(port=0, auto_sync_interval=0.05)
        target.peers = [source.url]
        source.mine_and_broadcast(max_nonces=128)
        source.mine_and_broadcast(max_nonces=128)
        deadline = time.time() + 3
        while time.time() < deadline and target.chain.height < source.chain.height:
            time.sleep(0.05)
        self.assertEqual(target.chain.height, source.chain.height)
        self.assertEqual(target.chain.tip.block_id, source.chain.tip.block_id)

    def test_peer_limit_is_enforced(self):
        node = LocalNode(self.params, "alice")
        self.assertEqual(node.max_peers, 8)
        for index in range(node.max_peers):
            node.add_peer(f"http://127.0.0.1:{40000 + index}")
        with self.assertRaises(TestnetError):
            node.add_peer("http://127.0.0.1:49999")

    def test_reconcile_rejects_oversized_peer_response(self):
        node = LocalNode(self.params, "alice")
        self.nodes = [node]
        node.start(port=0)
        node.max_sync_bytes = 256
        self.assertFalse(node.reconcile_peer(node.url))
        self.assertEqual(node.chain.height, 0)

    def test_dead_peer_backoff_prevents_hot_loop(self):
        node = LocalNode(self.params, "alice")
        self.nodes = [node]
        node.start(port=0, auto_sync_interval=0.05)
        node.add_peer("http://127.0.0.1:9")
        time.sleep(0.30)
        status = json.loads(urlopen(node.url + "/status", timeout=2).read())
        self.assertLessEqual(status["sync"]["attempts"], 3)
        self.assertGreaterEqual(status["sync"]["backoff_skips"], 1)


if __name__ == "__main__":
    unittest.main()
