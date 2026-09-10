"""Loopback-only Chitin-PoW testnet proof of concept.

This is an isolated research harness, not DERO, not a production network, and
not evidence of post-quantum security. It reuses the existing nonce-bound PoW
verifier as the block acceptance rule and adds only local block/chain plumbing.

The default launcher/test profile uses toy parameters so the chain can mine on
CPU without touching the resident GPU workload. The HTTP server binds only to
127.0.0.1; there is no peer discovery, wallet, payout, or external network.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Iterable
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .partial_isd import PartialISDParams, solve_partial_mitm
from .verifier import PoWParams, Share, derive_instance, verify_share

NETWORK_NAME = "chitin-pow-loopback-testnet"
ZERO_HASH = "00" * 32


class TestnetError(ValueError):
    """Malformed testnet object or invalid consensus transition."""

    __test__ = False


@dataclass(frozen=True)
class ChainParams:
    pow: PoWParams
    max_future_seconds: int = 120
    network: str = NETWORK_NAME
    max_payload_bytes: int = 64 * 1024
    target_block_seconds: int = 60
    retarget_interval: int = 2016
    max_retarget_step: int = 4
    median_time_window: int = 11

    def __post_init__(self) -> None:
        if self.target_block_seconds <= 0:
            raise TestnetError("target_block_seconds must be positive")
        if self.retarget_interval <= 0:
            raise TestnetError("retarget_interval must be positive")
        if self.max_retarget_step <= 0:
            raise TestnetError("max_retarget_step must be positive")
        if self.median_time_window <= 0 or self.median_time_window % 2 == 0:
            raise TestnetError("median_time_window must be a positive odd integer")
        if not (0 < self.pow.matched_bits <= self.pow.syndrome_bits):
            raise TestnetError("initial matched_bits outside syndrome range")

    @property
    def profile_id(self) -> str:
        payload = {
            "network": self.network,
            "n_columns": self.pow.n_columns,
            "syndrome_bits": self.pow.syndrome_bits,
            "w_max": self.pow.w_max,
            "matched_bits": self.pow.matched_bits,
            "target_block_seconds": self.target_block_seconds,
            "retarget_interval": self.retarget_interval,
            "max_retarget_step": self.max_retarget_step,
            "median_time_window": self.median_time_window,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]
        return f"chitin-pow-{digest}"


@dataclass(frozen=True)
class BlockHeader:
    version: int
    height: int
    previous_hash: str
    timestamp: int
    miner_id: str
    matched_bits: int = 0
    profile_id: str = ""

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "height": self.height,
            "previous_hash": self.previous_hash,
            "timestamp": self.timestamp,
            "miner_id": self.miner_id,
            "matched_bits": self.matched_bits,
            "profile_id": self.profile_id,
        }

    @classmethod
    def from_dict(cls, value: dict) -> "BlockHeader":
        return cls(
            version=int(value["version"]),
            height=int(value["height"]),
            previous_hash=str(value["previous_hash"]),
            timestamp=int(value["timestamp"]),
            miner_id=str(value["miner_id"]),
            matched_bits=int(value.get("matched_bits", 0)),
            profile_id=str(value.get("profile_id", "")),
        )

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")


@dataclass(frozen=True)
class Block:
    header: BlockHeader
    nonce: int
    support: tuple[int, ...]

    @property
    def block_id(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def _body_dict(self) -> dict:
        """Canonical block fields; derived block_id is excluded."""
        return {
            "header": self.header.to_dict(),
            "nonce": self.nonce,
            "support": list(self.support),
        }

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self._body_dict(), sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")

    def to_dict(self) -> dict:
        payload = self._body_dict()
        payload["block_id"] = self.block_id
        return payload

    @classmethod
    def from_dict(cls, value: dict) -> "Block":
        block = cls(
            header=BlockHeader.from_dict(value["header"]),
            nonce=int(value["nonce"]),
            support=tuple(int(index) for index in value["support"]),
        )
        supplied = value.get("block_id")
        if supplied is not None and supplied != block.block_id:
            raise TestnetError("block_id does not match canonical block bytes")
        return block


def _header_bytes(header: BlockHeader, params: ChainParams) -> bytes:
    return params.network.encode("ascii") + b"\x00" + header.canonical_bytes()


def block_work(block: Block) -> int:
    """Return deterministic integer work for one accepted PoW block."""
    if block.header.height == 0:
        return 0
    return 1 << block.header.matched_bits


def make_genesis(params: ChainParams, miner_id: str = "genesis") -> Block:
    """Create the deterministic height-zero anchor; genesis has no PoW share."""
    return Block(
        header=BlockHeader(1, 0, ZERO_HASH, 0, miner_id, params.pow.matched_bits, params.profile_id),
        nonce=0,
        support=(),
    )


def expected_matched_bits(
    previous: Block,
    params: ChainParams,
    *,
    window_start_timestamp: int | None = None,
) -> int:
    """Return consensus difficulty using the completed timestamp window.

    A retarget is evaluated only at a window boundary. The caller must provide
    the timestamp of the first block in that validated window; silently using
    the parent timestamp would make the result miner-controlled.
    """
    current = previous.header.matched_bits or params.pow.matched_bits
    if previous.header.height == 0:
        return current
    if previous.header.height % params.retarget_interval != 0:
        return current
    if window_start_timestamp is None:
        raise TestnetError("difficulty window ancestry required")
    target = params.target_block_seconds * params.retarget_interval
    elapsed = previous.header.timestamp - window_start_timestamp
    delta = 1 if elapsed < target else -1 if elapsed > target else 0
    delta = max(-params.max_retarget_step, min(params.max_retarget_step, delta))
    return max(1, min(params.pow.syndrome_bits, current + delta))


def _window_start_timestamp(
    previous: Block,
    params: ChainParams,
    blocks: dict[str, Block],
) -> int | None:
    """Find the first timestamp in the completed retarget window."""
    if previous.header.height == 0:
        return None
    if previous.header.height % params.retarget_interval != 0:
        return None
    target_height = previous.header.height - params.retarget_interval + 1
    cursor = previous
    while cursor.header.height > target_height:
        cursor = blocks[cursor.header.previous_hash]
    return cursor.header.timestamp


def _expected_for_parent(
    previous: Block,
    params: ChainParams,
    blocks: dict[str, Block],
) -> int:
    return expected_matched_bits(
        previous,
        params,
        window_start_timestamp=_window_start_timestamp(previous, params, blocks),
    )


def median_time_past(
    previous: Block,
    params: ChainParams,
    blocks: dict[str, Block],
) -> int:
    """Return the upper median of the validated recent ancestor timestamps."""
    timestamps: list[int] = []
    cursor = previous
    for _ in range(params.median_time_window):
        timestamps.append(cursor.header.timestamp)
        if cursor.header.height == 0:
            break
        cursor = blocks[cursor.header.previous_hash]
    timestamps.sort()
    return timestamps[len(timestamps) // 2]


def validate_block(
    block: Block,
    previous: Block,
    params: ChainParams,
    *,
    now: int | None = None,
    difficulty_override: int | None = None,
    validation_blocks: dict[str, Block] | None = None,
) -> bool:
    """Validate linkage, serialization fields, difficulty, and nonce-bound PoW."""
    try:
        header = block.header
        if header.version != 1:
            return False
        if header.profile_id != params.profile_id:
            return False
        if previous.header.profile_id != params.profile_id:
            return False
        if header.height != previous.header.height + 1:
            return False
        if header.previous_hash != previous.block_id:
            return False
        expected_bits = (
            difficulty_override
            if difficulty_override is not None
            else expected_matched_bits(previous, params)
        )
        if header.matched_bits != expected_bits:
            return False
        if not header.miner_id or len(header.miner_id) > 128:
            return False
        blocks_for_time = validation_blocks or {previous.block_id: previous}
        if header.timestamp <= median_time_past(previous, params, blocks_for_time):
            return False
        current = int(time.time()) if now is None else now
        if header.timestamp > current + params.max_future_seconds:
            return False
        if not (0 <= block.nonce < 2**64):
            return False
        if tuple(sorted(block.support)) != block.support:
            return False
        pow_params = PoWParams(
            n_columns=params.pow.n_columns,
            syndrome_bits=params.pow.syndrome_bits,
            w_max=params.pow.w_max,
            matched_bits=header.matched_bits,
        )
        share = Share(
            _header_bytes(header, params), block.nonce, tuple(block.support),
        )
        return verify_share(share, pow_params).valid
    except (TypeError, ValueError, OverflowError):
        return False


def validate_chain(blocks: Iterable[Block], params: ChainParams) -> bool:
    """Validate genesis and every sequential transition."""
    sequence = list(blocks)
    if not sequence or sequence[0] != make_genesis(params, sequence[0].header.miner_id):
        return False
    known = {sequence[0].block_id: sequence[0]}
    for previous, block in zip(sequence, sequence[1:]):
        expected = _expected_for_parent(previous, params, known)
        if not validate_block(block, previous, params, difficulty_override=expected):
            return False
        known[block.block_id] = block
    return True


def _validate_block_in_chain(
    block: Block,
    previous: Block,
    params: ChainParams,
    blocks: dict[str, Block],
) -> bool:
    """Validate one block with ancestry-aware difficulty."""
    expected = _expected_for_parent(previous, params, blocks)
    return validate_block(
        block,
        previous,
        params,
        difficulty_override=expected,
        validation_blocks=blocks,
    )


def mine_next_block(
    chain: "Chain",
    params: ChainParams,
    *,
    miner_id: str,
    max_nonces: int = 100_000,
    timestamp: int | None = None,
) -> Block:
    """Mine one toy block using the bounded reference partial-ISD solver."""
    if max_nonces <= 0:
        raise TestnetError("max_nonces must be positive")
    previous = chain.tip
    stamp = int(time.time()) if timestamp is None else timestamp
    if stamp <= previous.header.timestamp:
        if timestamp is None:
            stamp = previous.header.timestamp + 1
        else:
            raise TestnetError("timestamp must exceed parent")
    header = BlockHeader(
        version=1,
        height=previous.header.height + 1,
        previous_hash=previous.block_id,
        timestamp=stamp,
        miner_id=miner_id,
        matched_bits=_expected_for_parent(previous, params, chain._blocks),
        profile_id=params.profile_id,
    )
    header_bytes = _header_bytes(header, params)
    pow_params = PoWParams(
        n_columns=params.pow.n_columns,
        syndrome_bits=params.pow.syndrome_bits,
        w_max=params.pow.w_max,
        matched_bits=header.matched_bits,
    )
    attack_params = PartialISDParams(
        syndrome_bits=params.pow.syndrome_bits,
        matched_bits=header.matched_bits,
        max_weight=params.pow.w_max,
        max_candidates=max_nonces,
    )
    # The PoW nonce is searched as the instance-binding nonce. Rebuild columns
    # for each nonce; this keeps the block rule identical to verify_share().
    for nonce in range(min(max_nonces, 2**64)):
        columns, target = derive_instance(header_bytes, pow_params, nonce)
        result = solve_partial_mitm(columns, target, attack_params)
        if result.found:
            block = Block(header, nonce, tuple(sorted(result.support or ())))
            if _validate_block_in_chain(
                    block, previous, params, chain._blocks):
                return block
    raise TestnetError("bounded miner exhausted max_nonces")


class Chain:
    """Branch-aware in-memory chain for the isolated PoC.

    Every valid block whose parent is known is retained. The exposed ``blocks``
    view is the deterministic best chain: greatest height first, then greatest
    block id on equal-height ties. This is cumulative-work selection for the
    current fixed-work toy profile; it must be replaced by per-block work before
    any production discussion.
    """

    def __init__(self, params: ChainParams, miner_id: str = "genesis") -> None:
        self.params = params
        genesis = make_genesis(params, miner_id)
        self._blocks: dict[str, Block] = {genesis.block_id: genesis}
        self._best_tip_id = genesis.block_id
        self._lock = threading.RLock()

    @classmethod
    def from_blocks(cls, params: ChainParams, blocks: list[Block]) -> "Chain":
        """Build a branch-aware chain from a validated contiguous branch."""
        if not blocks:
            raise TestnetError("blocks must not be empty")
        chain = cls(params, miner_id=blocks[0].header.miner_id)
        genesis = chain.genesis
        if blocks[0] != genesis:
            raise TestnetError("invalid genesis for chain parameters")
        for block in blocks[1:]:
            if not chain.add_block(block):
                raise TestnetError("invalid block in initial chain")
        return chain

    @property
    def genesis(self) -> Block:
        return self._blocks[next(
            block_id for block_id, block in self._blocks.items()
            if block.header.height == 0
        )]

    def _best_chain(self) -> list[Block]:
        tip = self._blocks[self._best_tip_id]
        reverse: list[Block] = []
        while True:
            reverse.append(tip)
            if tip.header.height == 0:
                break
            tip = self._blocks[tip.header.previous_hash]
        return list(reversed(reverse))

    def _chain_work(self, tip: Block) -> int:
        total = 0
        cursor = tip
        while cursor.header.height > 0:
            total += block_work(cursor)
            cursor = self._blocks[cursor.header.previous_hash]
        return total

    @property
    def work(self) -> int:
        """Cumulative work of the selected best chain."""
        with self._lock:
            return self._chain_work(self._blocks[self._best_tip_id])

    def _selection_key(self, block: Block) -> tuple[int, int, str]:
        return self._chain_work(block), block.header.height, block.block_id

    def _reselect_tip(self) -> None:
        self._best_tip_id = max(
            self._blocks,
            key=lambda block_id: self._selection_key(self._blocks[block_id]),
        )

    @property
    def blocks(self) -> list[Block]:
        with self._lock:
            return self._best_chain()

    @property
    def height(self) -> int:
        return self._blocks[self._best_tip_id].header.height

    @property
    def tip(self) -> Block:
        return self._blocks[self._best_tip_id]

    @property
    def profile_id(self) -> str:
        return self.params.profile_id

    def has_block(self, block_id: str) -> bool:
        with self._lock:
            return block_id in self._blocks

    def save(self, path) -> None:
        """Persist every known block and the selected tip as canonical JSON."""
        with self._lock:
            payload = self.to_dict(include_all=True)
        destination = os.fspath(path)
        directory = os.path.dirname(os.path.abspath(destination)) or "."
        fd, temporary = tempfile.mkstemp(prefix=".chain-", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        except BaseException:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise

    @classmethod
    def load(cls, path, params: ChainParams) -> "Chain":
        """Load and validate all persisted blocks and their selected tip."""
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("network") != params.network:
            raise TestnetError("network mismatch")
        if payload.get("profile_id") != params.profile_id:
            raise TestnetError("profile mismatch")
        raw_blocks = [Block.from_dict(item) for item in payload.get("blocks", [])]
        if not raw_blocks:
            raise TestnetError("persisted chain is empty")
        genesis = make_genesis(params, raw_blocks[0].header.miner_id)
        if raw_blocks[0] != genesis:
            raise TestnetError("persisted genesis mismatch")
        chain = cls(params, miner_id=raw_blocks[0].header.miner_id)
        pending = {block.block_id: block for block in raw_blocks[1:]}
        while pending:
            progressed = False
            for block_id, block in list(pending.items()):
                if chain.add_block(block):
                    del pending[block_id]
                    progressed = True
            if not progressed:
                raise TestnetError("persisted chain contains invalid or orphaned blocks")
        if payload.get("tip") != chain.tip.block_id:
            raise TestnetError("persisted tip mismatch")
        return chain

    def to_dict(self, *, include_all: bool = False) -> dict:
        with self._lock:
            blocks = (
                list(self._blocks.values())
                if include_all else self._best_chain()
            )
            blocks.sort(key=lambda block: (block.header.height, block.block_id))
            return {
                "network": self.params.network,
                "profile_id": self.params.profile_id,
                "height": self.height,
                "tip": self.tip.block_id,
                "blocks": [block.to_dict() for block in blocks],
            }

    def add_block(self, block: Block) -> bool:
        with self._lock:
            if block.block_id in self._blocks:
                return False
            parent = self._blocks.get(block.header.previous_hash)
            if parent is None or not _validate_block_in_chain(
                    block, parent, self.params, self._blocks):
                return False
            self._blocks[block.block_id] = block
            self._reselect_tip()
            return True


class LocalNode:
    """Loopback-only HTTP node; no external bind is permitted."""

    def __init__(self, params: ChainParams, miner_id: str, state_path=None) -> None:
        self.params = params
        self.miner_id = miner_id
        self.state_path = state_path
        if state_path is not None and os.path.exists(state_path):
            self.chain = Chain.load(state_path, params)
        else:
            self.chain = Chain(params, miner_id="genesis")
            if state_path is not None:
                self.chain.save(state_path)
        self.max_peers = 8
        self.max_sync_bytes = 64 * 1024
        self.peers: list[str] = []
        self._peer_failures: dict[str, int] = {}
        self._peer_retry_at: dict[str, float] = {}
        self.peer_backoff_base = 0.5
        self.peer_backoff_max = 30.0
        self.auto_sync_interval: float | None = None
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._sync_thread: threading.Thread | None = None
        self._sync_stop = threading.Event()
        self._sync_lock = threading.Lock()
        self.sync_attempts = 0
        self.sync_successes = 0
        self.sync_backoff_skips = 0
        self.last_sync_error: str | None = None

    @property
    def sync_running(self) -> bool:
        return self._sync_thread is not None and self._sync_thread.is_alive()

    def add_peer(self, peer_url: str) -> None:
        parsed = urlparse(peer_url)
        if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or not parsed.port:
            raise TestnetError("peer must be an http://127.0.0.1:<port> URL")
        normalized = peer_url.rstrip("/")
        if normalized in self.peers:
            return
        if len(self.peers) >= self.max_peers:
            raise TestnetError("peer limit reached")
        self.peers.append(normalized)

    def _peer_backoff(self, peer_url: str) -> float:
        failures = self._peer_failures.get(peer_url, 0)
        return min(self.peer_backoff_max, self.peer_backoff_base * (2 ** max(0, failures - 1)))

    def _peer_failed(self, peer_url: str, error: str) -> None:
        failures = self._peer_failures.get(peer_url, 0) + 1
        self._peer_failures[peer_url] = failures
        self._peer_retry_at[peer_url] = time.monotonic() + self._peer_backoff(peer_url)
        self.last_sync_error = error

    def _peer_succeeded(self, peer_url: str) -> None:
        self._peer_failures.pop(peer_url, None)
        self._peer_retry_at.pop(peer_url, None)
        self.last_sync_error = None

    @property
    def url(self) -> str:
        if self._server is None:
            raise TestnetError("node is not started")
        return f"http://127.0.0.1:{self._server.server_port}"

    def start(self, *, port: int = 0, auto_sync_interval: float | None = None) -> None:
        if self._server is not None:
            raise TestnetError("node already started")
        if auto_sync_interval is not None and auto_sync_interval < 0.05:
            raise TestnetError("auto_sync_interval must be at least 0.05 seconds")
        self.auto_sync_interval = auto_sync_interval
        self._sync_stop.clear()
        node = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, _format: str, *_args) -> None:
                return

            def _send(self, payload: dict, status: int = 200) -> None:
                body = json.dumps(payload, sort_keys=True).encode("utf-8")
                if len(body) > node.params.max_payload_bytes:
                    body = b'{"error":"response too large"}'
                    status = 413
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                if self.path == "/status":
                    self._send({
                        "network": node.params.network,
                        "profile_id": node.params.profile_id,
                        "height": node.chain.height,
                        "tip": node.chain.tip.block_id,
                        "miner_id": node.miner_id,
                        "sync": {
                            "running": node.sync_running,
                            "attempts": node.sync_attempts,
                            "successes": node.sync_successes,
                            "backoff_skips": node.sync_backoff_skips,
                            "last_error": node.last_sync_error,
                        },
                    })
                elif self.path == "/chain":
                    self._send(node.chain.to_dict())
                else:
                    self._send({"error": "not found"}, 404)

            def do_POST(self) -> None:
                if self.path not in ("/block", "/sync"):
                    self._send({"error": "not found"}, 404)
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if length <= 0 or length > node.params.max_payload_bytes:
                        if length > 0:
                            self.rfile.read(min(length, node.params.max_payload_bytes + 1))
                        self.close_connection = True
                        raise TestnetError("payload too large or empty")
                    body = self.rfile.read(length)
                    if len(body) != length:
                        raise TestnetError("truncated payload")
                    payload = json.loads(body)
                    if self.path == "/sync":
                        accepted = node.receive_chain(payload)
                        if accepted and node.state_path is not None:
                            node.chain.save(node.state_path)
                        self._send({"accepted": accepted, "height": node.chain.height}, 200 if accepted else 400)
                    else:
                        block = Block.from_dict(payload)
                        accepted = node.receive_block(block)
                        self._send({"accepted": accepted}, 200 if accepted else 400)
                except (ValueError, TypeError, KeyError, TestnetError, json.JSONDecodeError) as exc:
                    self._send({"accepted": False, "error": str(exc)}, 400)

        self._server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever, name=f"chitin-node-{self.miner_id}",
            daemon=True,
        )
        self._thread.start()
        if self.auto_sync_interval is not None:
            self._sync_thread = threading.Thread(
                target=self._sync_loop,
                name=f"chitin-sync-{self.miner_id}",
                daemon=True,
            )
            self._sync_thread.start()

    def _sync_loop(self) -> None:
        interval = self.auto_sync_interval or 1.0
        while not self._sync_stop.wait(interval):
            for peer in tuple(self.peers):
                if time.monotonic() < self._peer_retry_at.get(peer, 0.0):
                    with self._sync_lock:
                        self.sync_backoff_skips += 1
                    continue
                with self._sync_lock:
                    self.sync_attempts += 1
                try:
                    if self.reconcile_peer(peer):
                        with self._sync_lock:
                            self.sync_successes += 1
                        self._peer_succeeded(peer)
                    else:
                        self._peer_failed(peer, "peer returned no import")
                except Exception as exc:  # defensive boundary for peer input
                    with self._sync_lock:
                        self._peer_failed(peer, str(exc))

    def stop(self) -> None:
        self._sync_stop.set()
        if self._sync_thread is not None:
            self._sync_thread.join(timeout=2)
            self._sync_thread = None
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def receive_block(self, block: Block) -> bool:
        accepted = self.chain.add_block(block)
        if accepted and self.state_path is not None:
            self.chain.save(self.state_path)
        return accepted

    def reconcile_peer(self, peer_url: str) -> bool:
        """Fetch and transactionally import a bounded peer chain."""
        request = Request(peer_url.rstrip("/") + "/chain", method="GET")
        try:
            with urlopen(request, timeout=0.1) as response:
                if response.status != 200:
                    return False
                length = int(response.headers.get("Content-Length", "0"))
                if length > self.max_sync_bytes:
                    return False
                raw = response.read(self.max_sync_bytes + 1)
                if len(raw) > self.max_sync_bytes:
                    return False
                payload = json.loads(raw)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return False
        accepted = self.receive_chain(payload)
        if accepted and self.state_path is not None:
            self.chain.save(self.state_path)
        return accepted

    def reconcile_peers(self) -> int:
        """Attempt reconciliation with all configured peers; return imports."""
        imported = 0
        for peer in tuple(self.peers):
            if self.reconcile_peer(peer):
                imported += 1
        return imported

    def receive_chain(self, payload: dict) -> bool:
        """Import a peer chain transactionally after full validation."""
        if payload.get("network") != self.params.network:
            return False
        if payload.get("profile_id") != self.params.profile_id:
            return False
        raw_blocks = payload.get("blocks")
        if not isinstance(raw_blocks, list) or len(raw_blocks) > 10_000:
            return False
        try:
            blocks = [Block.from_dict(item) for item in raw_blocks]
        except (KeyError, TypeError, ValueError, TestnetError):
            return False
        if not blocks or blocks[0] != self.chain.genesis:
            return False

        with self.chain._lock:
            known = dict(self.chain._blocks)
            best = self.chain._best_tip_id
            pending = {block.block_id: block for block in blocks[1:]}
            while pending:
                progressed = False
                for block_id, block in list(pending.items()):
                    parent = known.get(block.header.previous_hash)
                    if parent is None or not _validate_block_in_chain(
                            block, parent, self.params, known):
                        continue
                    known[block_id] = block
                    del pending[block_id]
                    progressed = True
                if not progressed:
                    return False
            candidate = max(
                known.values(), key=lambda block: (block.header.height, block.block_id)
            )
            changed = candidate.block_id != best or len(known) != len(self.chain._blocks)
            self.chain._blocks = known
            self.chain._best_tip_id = candidate.block_id
            return changed

    def mine_and_broadcast(self, *, max_nonces: int = 100_000) -> Block:
        block = mine_next_block(
            self.chain, self.params, miner_id=self.miner_id,
            max_nonces=max_nonces,
        )
        if not self.chain.add_block(block):
            raise TestnetError("locally mined block failed local validation")
        if self.state_path is not None:
            self.chain.save(self.state_path)
        payload = json.dumps(block.to_dict()).encode("utf-8")
        for peer in self.peers:
            try:
                request = Request(
                    peer + "/block", data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urlopen(request, timeout=2) as response:
                    if response.status != 200:
                        raise TestnetError("peer rejected block")
            except OSError as exc:
                raise TestnetError(f"peer broadcast failed: {exc}") from exc
        return block


__all__ = [
    "Block",
    "BlockHeader",
    "Chain",
    "ChainParams",
    "LocalNode",
    "NETWORK_NAME",
    "TestnetError",
    "make_genesis",
    "mine_next_block",
    "validate_block",
    "validate_chain",
]
