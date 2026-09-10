"""Run the bounded two-node Chitin-PoW loopback testnet PoC.

This never binds outside 127.0.0.1. The default toy profile is mineable;
the candidate profile is opt-in and reports its gate status without pretending
the bounded reference miner can feasibly mine it.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
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
    parser.add_argument(
        "--restart-recovery", action="store_true",
        help="stop/restart Bob and reconcile missed blocks from Alice",
    )
    parser.add_argument(
        "--state-dir", type=Path,
        help="directory for persistent node state; otherwise a temporary directory is used",
    )
    parser.add_argument(
        "--serve", action="store_true",
        help="keep one loopback node running for manual inspection",
    )
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--miner-id", default="alice")
    parser.add_argument("--peer", action="append", default=[])
    args = parser.parse_args()
    if args.blocks <= 0 or args.max_nonces <= 0:
        parser.error("--blocks and --max-nonces must be positive")
    if args.sync_only and args.restart_recovery:
        parser.error("--sync-only and --restart-recovery are mutually exclusive")

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
    managed_temp = None
    if args.state_dir is None:
        managed_temp = tempfile.TemporaryDirectory(prefix="chitin-loopback-")
        state_dir = Path(managed_temp.name)
    else:
        state_dir = args.state_dir
        state_dir.mkdir(parents=True, exist_ok=True)

    if not (0 <= args.port <= 65535):
        parser.error("--port must be in [0, 65535]")
    for peer in args.peer:
        if not peer.startswith("http://127.0.0.1:"):
            parser.error("--peer must use http://127.0.0.1:<port>")
    if args.serve:
        node = LocalNode(params, args.miner_id, state_path=state_dir / f"{args.miner_id}.json")
        try:
            node.peers = list(args.peer)
            node.start(port=args.port)
            print(json.dumps({
                "mode": "serve",
                "url": node.url,
                "status": node.url + "/status",
                "chain": node.url + "/chain",
                "network": params.network,
                "profile_id": params.profile_id,
                "profile": profile_report,
                "loopback_only": True,
                "warning": "research PoC; stop with Ctrl+C; not DERO",
            }, indent=2, sort_keys=True), flush=True)
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                return 0
        finally:
            node.stop()
            if managed_temp is not None:
                managed_temp.cleanup()

    alice = LocalNode(params, "alice", state_path=state_dir / "alice.json")
    bob = LocalNode(params, "bob", state_path=state_dir / "bob.json")
    bob_restarted = False
    recovered = False
    blocks: list[str] = []
    try:
        alice.start(port=0)
        bob.start(port=0)
        alice.peers = [bob.url]
        if not args.sync_only:
            if args.restart_recovery:
                first = alice.mine_and_broadcast(max_nonces=args.max_nonces)
                blocks.append(first.block_id)
                bob.stop()
                alice.peers = []
                for _ in range(max(args.blocks - 1, 1)):
                    block = alice.mine_and_broadcast(max_nonces=args.max_nonces)
                    blocks.append(block.block_id)
                bob = LocalNode(params, "bob", state_path=state_dir / "bob.json")
                bob.start(port=0)
                bob_restarted = True
                bob.peers = [alice.url]
                recovered = bob.reconcile_peer(alice.url)
            else:
                for _ in range(args.blocks):
                    block = alice.mine_and_broadcast(max_nonces=args.max_nonces)
                    blocks.append(block.block_id)
        output = {
            "network": params.network,
            "profile_id": params.profile_id,
            "profile": profile_report,
            "loopback_only": True,
            "state_dir": str(state_dir),
            "alice_url": alice.url,
            "bob_url": bob.url,
            "bob_restarted": bob_restarted,
            "recovered_from_peer": recovered,
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
        success = output["same_tip"] and (not args.restart_recovery or recovered)
        return 0 if success else 1
    except TestnetError as exc:
        print(json.dumps({"error": str(exc), "profile": profile_report}, indent=2))
        return 1
    finally:
        alice.stop()
        bob.stop()
        if managed_temp is not None:
            managed_temp.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
