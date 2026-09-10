"""Tests for the loopback-only Chitin-PoW testnet PoC."""

import json
import threading
import unittest
from urllib.request import Request, urlopen

from pqmine.verifier import PoWParams, Share
from pqmine.testnet import (
    Block,
    BlockHeader,
    Chain,
    ChainParams,
    LocalNode,
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
            timestamp=1, miner_id="alice",
        )
        bad = Block(header, nonce=0, support=(1, 1))
        self.assertFalse(validate_block(bad, self.chain.tip, self.params))

    def test_chain_rejects_wrong_parent_and_height(self):
        block = mine_next_block(self.chain, self.params, miner_id="alice", max_nonces=128)
        self.assertTrue(self.chain.add_block(block))
        wrong_header = BlockHeader(
            version=1, height=3, previous_hash="00" * 32,
            timestamp=2, miner_id="alice",
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
        self.assertEqual([b.block_id for b in self.chain.blocks], [self.chain.genesis.block_id, side_one.block_id, side_two.block_id])

    def test_duplicate_block_is_rejected(self):
        block = mine_next_block(self.chain, self.params, miner_id="alice", max_nonces=128)
        self.assertTrue(self.chain.add_block(block))
        self.assertFalse(self.chain.add_block(block))


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


if __name__ == "__main__":
    unittest.main()
