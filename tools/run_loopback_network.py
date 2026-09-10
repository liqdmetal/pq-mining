"""Run two bounded Chitin-PoW nodes as separate loopback processes.

Research-only harness: toy CPU profile, wallet-free, payout-free, and bound to
127.0.0.1. The parent owns the lifecycle; worker processes own node state.
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from queue import Empty, Queue
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pqmine.testnet import ChainParams, LocalNode, TestnetError
from pqmine.verifier import PoWParams


READY_TIMEOUT = 5.0


def _toy_params(matched_bits: int) -> ChainParams:
    return ChainParams(
        pow=PoWParams(
            n_columns=32,
            syndrome_bits=32,
            w_max=4,
            matched_bits=matched_bits,
        ),
        max_future_seconds=3600,
    )


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _reader(stream, queue: Queue[str]) -> None:
    for line in iter(stream.readline, ""):
        queue.put(line)
    stream.close()


def _start_worker(
    *,
    miner_id: str,
    state_dir: Path,
    port: int,
    peers: list[str],
    mine: bool,
    run_for: float,
    mine_interval: float,
    max_nonces: int,
    matched_bits: int,
) -> tuple[subprocess.Popen[str], str, Queue[str]]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--miner-id",
        miner_id,
        "--state-dir",
        str(state_dir),
        "--port",
        str(port),
        "--run-for",
        str(run_for),
        "--mine-interval",
        str(mine_interval),
        "--max-nonces",
        str(max_nonces),
        "--matched-bits",
        str(matched_bits),
    ]
    if mine:
        command.append("--mine")
    for peer in peers:
        command.extend(("--peer", peer))
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    lines: Queue[str] = Queue()
    threading.Thread(target=_reader, args=(process.stdout, lines), daemon=True).start()
    deadline = time.monotonic() + READY_TIMEOUT
    while time.monotonic() < deadline:
        try:
            line = lines.get(timeout=0.05)
        except Empty:
            if process.poll() is not None:
                break
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("ready") is True:
            return process, payload["url"], lines
    stderr = process.stderr.read() if process.stderr is not None else ""
    process.kill()
    raise RuntimeError(f"worker failed to start: {stderr.strip() or 'no ready message'}")


def _status(url: str) -> dict:
    with urlopen(url.rstrip("/") + "/status", timeout=2) as response:
        return json.loads(response.read())


def _stop(process: subprocess.Popen[str]) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)


def _worker(args: argparse.Namespace) -> int:
    params = _toy_params(args.matched_bits)
    state_path = args.state_dir / f"{args.miner_id}.json"
    node = LocalNode(params, args.miner_id, state_path=state_path)
    try:
        for peer in args.peer:
            node.add_peer(peer)
        node.start(port=args.port, auto_sync_interval=0.1)
        print(json.dumps({
            "ready": True,
            "url": node.url,
            "miner_id": args.miner_id,
            "network": params.network,
            "profile_id": params.profile_id,
            "loopback_only": True,
        }, sort_keys=True), flush=True)
        deadline = time.monotonic() + args.run_for
        while time.monotonic() < deadline:
            if args.mine:
                try:
                    node.mine_and_broadcast(max_nonces=args.max_nonces)
                except TestnetError:
                    pass
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(min(args.mine_interval, remaining))
        return 0
    finally:
        node.stop()


def _validate_peer(peer: str) -> None:
    if not peer.startswith("http://127.0.0.1:"):
        raise ValueError("--peer must use http://127.0.0.1:<port> (loopback only)")


def _parent(args: argparse.Namespace) -> int:
    for peer in args.peer:
        _validate_peer(peer)
    if args.run_for <= 0:
        raise ValueError("--run-for must be positive")
    if args.mine_interval <= 0 or args.max_nonces <= 0:
        raise ValueError("--mine-interval and --max-nonces must be positive")
    if args.state_dir is None:
        raise ValueError("--state-dir is required for the process runner")
    args.state_dir.mkdir(parents=True, exist_ok=True)

    port_a, port_b = _free_port(), _free_port()
    url_a_guess = f"http://127.0.0.1:{port_a}"
    url_b_guess = f"http://127.0.0.1:{port_b}"
    processes: list[subprocess.Popen[str]] = []
    try:
        alice, url_a, _ = _start_worker(
            miner_id="alice", state_dir=args.state_dir, port=port_a,
            peers=[url_b_guess], mine=True, run_for=args.run_for + READY_TIMEOUT,
            mine_interval=args.mine_interval, max_nonces=args.max_nonces,
            matched_bits=args.matched_bits,
        )
        processes.append(alice)
        bob, url_b, _ = _start_worker(
            miner_id="bob", state_dir=args.state_dir, port=port_b,
            peers=[url_a], mine=False, run_for=args.run_for + READY_TIMEOUT,
            mine_interval=args.mine_interval, max_nonces=args.max_nonces,
            matched_bits=args.matched_bits,
        )
        processes.append(bob)
        time.sleep(args.run_for)
        alice_status = _status(url_a)
        bob_status = _status(url_b)
        report = {
            "network": alice_status["network"],
            "profile_id": alice_status["profile_id"],
            "loopback_only": True,
            "processes_started": len(processes),
            "state_dir": str(args.state_dir),
            "alice": alice_status,
            "bob": bob_status,
            "same_tip": alice_status["tip"] == bob_status["tip"],
            "warning": "research PoC; not DERO; not a security claim",
        }
        print(json.dumps(report, sort_keys=True))
        return 0 if report["same_tip"] and alice_status["height"] >= 1 else 1
    finally:
        for process in reversed(processes):
            _stop(process)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--miner-id", default="alice")
    parser.add_argument("--state-dir", type=Path, required=False)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--run-for", type=float, default=5.0)
    parser.add_argument("--mine-interval", type=float, default=0.2)
    parser.add_argument("--max-nonces", type=int, default=128)
    parser.add_argument("--matched-bits", type=int, default=8)
    parser.add_argument("--mine", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--peer", action="append", default=[])
    args = parser.parse_args()
    if args.state_dir is None:
        parser.error("--state-dir is required")
    if not (0 <= args.port <= 65535):
        parser.error("--port must be in [0, 65535]")
    try:
        for peer in args.peer:
            _validate_peer(peer)
        return _worker(args) if args.worker else _parent(args)
    except (TestnetError, ValueError, OSError, RuntimeError) as exc:
        print(json.dumps({"error": str(exc), "loopback_only": True}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
