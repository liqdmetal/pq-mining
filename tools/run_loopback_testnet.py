"""Run a bounded two-node Chitin-PoW loopback testnet PoC.

This never binds outside 127.0.0.1. The default toy profile is mineable;
the candidate profile is opt-in and reports its gate status without pretending
the bounded reference miner can feasibly mine it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pqmine.profiles import candidate_128_profile
from pqmine.testnet import ChainParams, LocalNode, TestnetError
from pqmine.verifier import PoWParams


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--max-nonces", type=int, default=128)
    parser.add_argument("--matched-bits", type=int, default=8)
    parser.add_argument("--profile", choices=("toy", "candidate-128"), default="toy")
    parser.add_argument("--sync-only", action="store_true")
    args = parser.parse_args()
    if args.blocks <= 0 or args.max_nonces <= 0:
        parser.error("--blocks and --max-nonces must be positive")

    if args.profile == "candidate-128":
        screen = candidate_128_profile()
        if not args.sync_only:
            print(json.dumps({
                "error": "candidate profile is not mineable by the bounded reference miner",
                "profile": screen.summary(),
                "action": "use --sync-only for profile-bound node inspection",
            }, indent=2, sort_keys=True))
            return 2
        pow_params = screen.params
        profile_report = screen.summary()
    else:
        pow_params = PoWParams(
            n_columns=32,
            syndrome_bits=32,
            w_max=4,
            matched_bits=args.matched_bits,
        )
        profile_report = {"name": "toy", "warning": "fast local profile only"}

    params = ChainParams(pow=pow_params, max_future_seconds=3600)
    alice = LocalNode(params, "alice")
    bob = LocalNode(params, "bob")
    try:
        alice.start(port=0)
        bob.start(port=0)
        alice.peers = [bob.url]
        blocks = []
        if not args.sync_only:
            for _ in range(args.blocks):
                block = alice.mine_and_broadcast(max_nonces=args.max_nonces)
                blocks.append(block.block_id)
        output = {
            "network": params.network,
            "profile_id": params.profile_id,
            "profile": profile_report,
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
            "warning": "research PoC; not DERO; not a security claim",
        }
        print(json.dumps(output, indent=2, sort_keys=True))
        return 0 if output["same_tip"] else 1
    except TestnetError as exc:
        print(json.dumps({"error": str(exc), "profile": profile_report}, indent=2))
        return 1
    finally:
        alice.stop()
        bob.stop()


if __name__ == "__main__":
    raise SystemExit(main())
