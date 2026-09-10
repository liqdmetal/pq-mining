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
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Iterable
from urllib.request import Request, urlopen

from .partial_isd import PartialISDParams, solve_partial_mitm
from .verifier import PoWParams, Share, derive_instance, verify_share

NETWORK_NAME = "chitin-pow-loopback-testnet"
ZERO_HASH = "00" * 32


class TestnetError(ValueError):
    """Malformed testnet object or invalid consensus transition."""


@dataclass(frozen=True)
class ChainParams:
    pow: PoWParams
    max_future_seconds: int = 120
    network: str = NETWORK_NAME


@dataclass(frozen=True)
class BlockHeader:
    version: int
    height: int
    previous_hash: str
    timestamp: int
    miner_id: str

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "height": self.height,
            "previous_hash": self.previous_hash,
            "timestamp": self.timestamp,
            "miner_id": self.miner_id,
        }

    @classmethod
    def from_dict(cls, value: dict) -> "BlockHeader":
        return cls(
            version=int(value["version"]),
            height=int(value["height"]),
            previous_hash=str(value["previous_hash"]),
            timestamp=int(value["timestamp"]),
            miner_id=str(value["miner_id"]),
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


def make_genesis(params: ChainParams, miner_id: str = "genesis") -> Block:
    """Create the deterministic height-zero anchor; genesis has no PoW share."""
    return Block(
        header=BlockHeader(1, 0, ZERO_HASH, 0, miner_id),
        nonce=0,
        support=(),
    )


def validate_block(
    block: Block,
    previous: Block,
    params: ChainParams,
    *,
    now: int | None = None,
) -> bool:
    """Validate linkage, serialization fields, timestamp, and nonce-bound PoW."""
    try:
        header = block.header
        if header.version != 1:
            return False
        if header.height != previous.header.height + 1:
            return False
        if header.previous_hash != previous.block_id:
            return False
        if not header.miner_id or len(header.miner_id) > 128:
            return False
        if not (previous.header.timestamp <= header.timestamp):
            return False
        current = int(time.time()) if now is None else now
        if header.timestamp > current + params.max_future_seconds:
            return False
        if not (0 <= block.nonce < 2**64):
            return False
        if tuple(sorted(block.support)) != block.support:
            return False
        share = Share(
            _header_bytes(header, params), block.nonce, tuple(block.support),
        )
        return verify_share(share, params.pow).valid
    except (TypeError, ValueError, OverflowError):
        return False


def validate_chain(blocks: Iterable[Block], params: ChainParams) -> bool:
    """Validate genesis and every sequential transition."""
    sequence = list(blocks)
    if not sequence or sequence[0] != make_genesis(params, sequence[0].header.miner_id):
        return False
    for previous, block in zip(sequence, sequence[1:]):
        if not validate_block(block, previous, params):
            return False
    return True


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
    if stamp < previous.header.timestamp:
        stamp = previous.header.timestamp
    header = BlockHeader(
        version=1,
        height=previous.header.height + 1,
        previous_hash=previous.block_id,
        timestamp=stamp,
        miner_id=miner_id,
    )
    header_bytes = _header_bytes(header, params)
    attack_params = PartialISDParams(
        syndrome_bits=params.pow.syndrome_bits,
        matched_bits=params.pow.matched_bits,
        max_weight=params.pow.w_max,
        max_candidates=max_nonces,
    )
    columns, target = derive_instance(header_bytes, params.pow, 0)
    # The PoW nonce is searched as the instance-binding nonce. Rebuild columns
    # for each nonce; this keeps the block rule identical to verify_share().
    for nonce in range(min(max_nonces, 2**64)):
        columns, target = derive_instance(header_bytes, params.pow, nonce)
        result = solve_partial_mitm(columns, target, attack_params)
        if result.found:
            block = Block(header, nonce, tuple(sorted(result.support or ())))
            if validate_block(block, previous, params):
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

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "network": self.params.network,
                "height": self.height,
                "tip": self.tip.block_id,
                "blocks": [block.to_dict() for block in self._best_chain()],
            }

    def add_block(self, block: Block) -> bool:
        with self._lock:
            if block.block_id in self._blocks:
                return False
            parent = self._blocks.get(block.header.previous_hash)
            if parent is None or not validate_block(block, parent, self.params):
                return False
            self._blocks[block.block_id] = block
            current = self._blocks[self._best_tip_id]
            candidate_key = (block.header.height, block.block_id)
            current_key = (current.header.height, current.block_id)
            if candidate_key > current_key:
                self._best_tip_id = block.block_id
            return True


class LocalNode:
    """Loopback-only HTTP node; no external bind is permitted."""

    def __init__(self, params: ChainParams, miner_id: str) -> None:
        self.params = params
        self.miner_id = miner_id
        self.chain = Chain(params, miner_id="genesis")
        self.peers: list[str] = []
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        if self._server is None:
            raise TestnetError("node is not started")
        return f"http://127.0.0.1:{self._server.server_port}"

    def start(self, *, port: int = 0) -> None:
        if self._server is not None:
            raise TestnetError("node already started")
        node = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, _format: str, *_args) -> None:
                return

            def _send(self, payload: dict, status: int = 200) -> None:
                body = json.dumps(payload, sort_keys=True).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                if self.path == "/status":
                    self._send({
                        "network": node.params.network,
                        "height": node.chain.height,
                        "tip": node.chain.tip.block_id,
                        "miner_id": node.miner_id,
                    })
                elif self.path == "/chain":
                    self._send(node.chain.to_dict())
                else:
                    self._send({"error": "not found"}, 404)

            def do_POST(self) -> None:
                if self.path != "/block":
                    self._send({"error": "not found"}, 404)
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    block = Block.from_dict(json.loads(self.rfile.read(length)))
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

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def receive_block(self, block: Block) -> bool:
        return self.chain.add_block(block)

    def mine_and_broadcast(self, *, max_nonces: int = 100_000) -> Block:
        block = mine_next_block(
            self.chain, self.params, miner_id=self.miner_id,
            max_nonces=max_nonces,
        )
        if not self.chain.add_block(block):
            raise TestnetError("locally mined block failed local validation")
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
