"""Verifier for Chitin-PoW v2 shares: the spec's acceptance rule, in code.

WHY A SEPARATE VERIFIER
-----------------------
The miner tracks distinctness with a used-set and reports a weight. A
verifier must trust NEITHER. Security of the PoW rests on what the
verifier independently recomputes from the share alone:

  1. The support is a SET of distinct column indices in range.  (F_2
     pigeonhole: a repeated column cancels, so a dishonest miner could
     pad weight for free. The verifier rejects duplicates outright.)
  2. |support| <= w_max.  Bounded weight, NOT exact: a heavier error is a
     WEAKER claim (larger solution set), so the verifier accepts anything
     at or under the cap and difficulty is set assuming the cap.
  3. XOR of the selected columns of H equals the target syndrome on the
     matched bits.  This is the SDP relation He^T = s (restricted to the
     difficulty's matched-bit prefix); it is the whole proof of work.
  4. H and s are derived from the header, not supplied by the miner.

Cost: |support| XORs of 64-bit words -- microseconds. Verification is
asymmetric with search by the nonce count (see spec), not the step count.

The walk (K steps, scratchpad, chain) is NOT re-executed. It is the search
procedure, not the proof. A share is valid iff (1)-(4) hold; how the miner
found it is irrelevant to acceptance. This is deliberate: it means the
walk can be changed, optimised or replaced without a consensus change, and
it means a GPU/CPU miner disagreement can never fork the chain -- only the
XOR check is consensus.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


class ShareError(ValueError):
    """Malformed share."""


@dataclass(frozen=True)
class PoWParams:
    """Consensus parameters. Fixed for the chain; see docs/POW_SPEC.md."""

    n_columns: int = 512
    syndrome_bits: int = 256       # n - k for rate 1/2
    w_max: int = 32                # bounded weight
    matched_bits: int = 164        # DIFFICULTY: retargetable, <= syndrome_bits

    def __post_init__(self) -> None:
        if not (0 < self.matched_bits <= self.syndrome_bits <= 64 * 8):
            raise ShareError("matched_bits must be in (0, syndrome_bits]")
        if not (0 < self.w_max < self.n_columns):
            raise ShareError("w_max must be in (0, n_columns)")


@dataclass(frozen=True)
class Share:
    header: bytes
    nonce: int
    support: tuple            # sorted distinct column indices


@dataclass(frozen=True)
class Verdict:
    valid: bool
    reason: str
    weight: int
    matched: int


def _instance_prefix(header: bytes, nonce: int) -> bytes:
    """Canonical instance binding for one header/nonce pair."""
    if not isinstance(header, bytes) or not header:
        raise ShareError("header must be non-empty bytes")
    if not (0 <= nonce < 2**64):
        raise ShareError("nonce must be a uint64")
    return (len(header).to_bytes(4, "big") + header
            + nonce.to_bytes(8, "big"))


def derive_columns(header: bytes, params: PoWParams, nonce: int = 0) -> tuple:
    """Parity-check columns from the bound header/nonce instance."""
    words = (params.syndrome_bits + 63) // 64
    prefix = _instance_prefix(header, nonce)
    out = []
    for j in range(params.n_columns):
        digest = hashlib.shake_256(
            b"CHITIN-POW-H-V2" + prefix + j.to_bytes(4, "big")
        ).digest(8 * words)
        out.append(int.from_bytes(digest, "big"))
    return tuple(out)


def derive_target(header: bytes, params: PoWParams, nonce: int = 0) -> int:
    """Target syndrome from the bound header/nonce instance."""
    words = (params.syndrome_bits + 63) // 64
    prefix = _instance_prefix(header, nonce)
    return int.from_bytes(
        hashlib.shake_256(b"CHITIN-POW-S-V2" + prefix).digest(8 * words), "big"
    )


def derive_instance(header: bytes, params: PoWParams, nonce: int = 0) -> tuple[tuple[int, ...], int]:
    """Return the complete consensus instance ``(H, s)`` for one nonce.

    Keeping the two derivations behind one named helper prevents miners,
    audits, and tools from silently binding different fields or domains.
    """
    return derive_columns(header, params, nonce), derive_target(header, params, nonce)


def syndrome_mask(params: PoWParams) -> int:
    """Return the canonical top-``matched_bits`` syndrome mask."""
    total = 64 * ((params.syndrome_bits + 63) // 64)
    return ((1 << params.matched_bits) - 1) << (total - params.matched_bits)


# Private compatibility alias for existing internal callers.
_mask = syndrome_mask


def verify_share(share: Share, params: PoWParams) -> Verdict:
    """Rules (1)-(4). Never trusts anything the miner reported."""
    support = tuple(share.support)

    # (1) distinct, in range. Duplicates are the F_2 cancellation attack.
    if len(set(support)) != len(support):
        return Verdict(False, "duplicate column (F_2 cancellation attack)",
                       len(support), 0)
    if any(not (0 <= j < params.n_columns) for j in support):
        return Verdict(False, "column index out of range", len(support), 0)
    if not support:
        return Verdict(False, "empty support", 0, 0)

    # (2) bounded weight.
    if len(support) > params.w_max:
        return Verdict(False, f"weight {len(support)} > w_max {params.w_max}",
                       len(support), 0)

    # (3) syndrome relation on the matched bits. (4) H, s from header.
    columns = derive_columns(share.header, params, share.nonce)
    target = derive_target(share.header, params, share.nonce)
    acc = 0
    for j in support:
        acc ^= columns[j]
    mask = _mask(params)
    matched = (acc & mask) == (target & mask)
    matched_count = params.matched_bits if matched else 0
    if not matched:
        return Verdict(False, "syndrome mismatch on matched bits",
                       len(support), matched_count)
    return Verdict(True, "valid", len(support), matched_count)


def expected_work_log2(params: PoWParams) -> float:
    """Log2 of the honest random-candidate work, approximately ``t``.

    This is only the nonce/candidate sampling cost. It is not an attacker
    security estimate; the bounded-language attack analysis lives in
    ``pqmine.bounded_work`` and currently gives only an upper bound on
    security. No Stern/BJMM security claim is made here.
    """
    return float(params.matched_bits)


__all__ = [
    "PoWParams",
    "Share",
    "ShareError",
    "Verdict",
    "derive_columns",
    "derive_instance",
    "derive_target",
    "expected_work_log2",
    "syndrome_mask",
    "verify_share",
]
