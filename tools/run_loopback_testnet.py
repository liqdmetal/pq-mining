"""Run a bounded two-node Chitin-PoW loopback testnet PoC.

This never binds outside 127.0.0.1 and uses toy parameters by default.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow direct execution from the repository's tools/ directory.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pqmine.testnet import ChainParams, LocalNode
from pqmine.verifier import PoWParams


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--max-nonces", type=int, default=128)
    parser.add_argument("--matched-bits", type=int, default=8)
    args = parser.parse_args()
    if args.blocks <= 0 or args.max_nonces <= 0:
        parser.error("--blocks and --max-nonces must be positive")

    params = ChainParams(
        pow=PoWParams(
            n_columns=32,
            syndrome_bits=32,
            w_max=4,
            matched_bits=args.matched_bits,
        ),
        max_future_seconds=3600,
    )
    alice = LocalNode(params, "alice")
    bob = LocalNode(params, "bob")
    try:
        alice.start(port=0)
        bob.start(port=0)
        alice.peers = [bob.url]
        blocks = []
        for _ in range(args.blocks):
            block = alice.mine_and_broadcast(max_nonces=args.max_nonces)
            blocks.append(block.block_id)
        output = {
            "network": params.network,
            "loopback_only": True,
            "alice_url": alice.url,
            "bob_url": bob.url,
            "blocks_mined": len(blocks),
            "block_ids": blocks,
            "alice_height": alice.chain.height,
            "bob_height": bob.chain.height,
            "same_tip": alice.chain.tip.block_id == bob.chain.tip.block_id,
            "pow": {
                "n_columns": params.pow.n_columns,
                "syndrome_bits": params.pow.syndrome_bits,
                "w_max": params.pow.w_max,
                "matched_bits": params.pow.matched_bits,
            },
            "warning": "research PoC; toy parameters; not DERO; not a security claim",
        }
        print(json.dumps(output, indent=2, sort_keys=True))
        return 0 if output["same_tip"] else 1
    finally:
        alice.stop()
        bob.stop()


if __name__ == "__main__":
    raise SystemExit(main())
