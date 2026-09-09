"""Chitin-PoW: a memory-hard lattice proof-of-work with cheap verification.

HONEST FRAMING FIRST
--------------------
"Post-quantum mining" is easy to oversell, so: hash-based PoW is ALREADY
fairly quantum-resistant. Grover gives only a quadratic speedup on preimage
search, which is handled by doubling the search space -- unlike signatures,
where Shor is catastrophic. So the motivation for a lattice PoW is NOT that
SHA-256 mining is quantum-broken. It is not.

The real motivations are:

  1. ASIC/FPGA resistance through memory-bandwidth and branch dependence.
  2. The work is a recognised hard problem (short-vector search) rather than
     an arbitrary hash inversion, so the security argument connects to
     lattice cryptanalysis instead of to hash heuristics.
  3. Quantum sieving buys far less against lattices than Grover does
     against hashes -- best known quantum sieve exponents improve the
     classical 2^{0.292n} to about 2^{0.265n}, a small constant-factor
     change in the exponent rather than a square root.

Point 3 is the genuinely post-quantum part, and it is a modest claim.

THE WORK FUNCTION
-----------------
From a block header derive a Module-SIS instance A in Z_q^{n x m}. A valid
share is a short nonzero z with

    A z = 0 mod q      and      ||z||_inf <= beta

Verification is ONE matrix-vector product plus a norm check: microseconds,
deterministic, no replay of the search. That asymmetry is what makes it
usable as consensus work.

WHY IT IS MEMORY-HARD (the CPU+GPU thesis)
------------------------------------------
The efficient way to find such a z is the sieving approach: generate a large
list of random sparse candidates z_i, compute their syndromes s_i = A z_i
mod q, and look for NEAR-COLLISIONS s_i ~ s_j. Then z_i - z_j is short and
its syndrome is small, so a few reduction steps land on a solution.

That decomposes into two phases with genuinely different hardware affinity:

  PHASE A (GPU-favourable): generate candidates and compute syndromes.
    Wide, independent, arithmetic-dense, streams linearly. An RTX 3090's
    ~936 GB/s and thousands of lanes are ideal.

  PHASE B (CPU-favourable): probe a large hash table for near-collisions.
    Random access into a working set far larger than any on-chip SRAM,
    unpredictable branches on hit/miss, pointer chasing. Large caches,
    deep out-of-order execution and good branch prediction win; a 5900X's
    24 threads with 64 MB L3 are ideal.

Neither phase alone finds a share, and the phases ALTERNATE with data
dependence: which region of the table to refill depends on where collisions
were found. An accelerator that only does Phase A stalls on Phase B, and
vice versa. That is the cooperative-mining property.

WHAT AN ASIC COULD STILL DO -- stated plainly
---------------------------------------------
An ASIC can absolutely implement the syndrome arithmetic more efficiently
than a GPU. What it cannot cheaply escape is the table: the working set is
chosen to exceed practical on-die SRAM, so an ASIC must attach external
DRAM/HBM and then competes on the same memory bandwidth as commodity
hardware, where the margin is a small constant rather than the 1000x of
SHA-256 ASICs. This is the Cuckoo Cycle / Equihash argument, and it has a
known limit: memory-hardness narrows the ASIC gap, it does not eliminate it.
Anyone claiming ASIC-PROOF is selling something.

STATUS: research prototype. Toy parameters, a deliberately simple search
(not a state-of-the-art sieve), no consensus integration, no network.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

CHALLENGE_DOMAIN = b"CHITIN-POW-CHALLENGE-V1"
CANDIDATE_DOMAIN = b"CHITIN-POW-CANDIDATE-V1"


class PoWError(ValueError):
    """Raised on malformed proof-of-work parameters or shares."""


@dataclass(frozen=True)
class PoWParameters:
    """Work-function shape. n rows, m columns, modulus q, norm bound beta."""

    n: int
    m: int
    q: int
    beta: int
    table_bits: int = 20      # log2 of the candidate table size
    sparsity: int = 8         # nonzero entries per generated candidate

    def __post_init__(self) -> None:
        if min(self.n, self.m) <= 0:
            raise PoWError("n and m must be positive")
        if self.m <= self.n:
            raise PoWError("m must exceed n or no short kernel vector exists")
        if self.q <= 2:
            raise PoWError("modulus must exceed 2")
        if self.beta <= 0:
            raise PoWError("beta must be positive")
        if self.beta >= self.q // 2:
            raise PoWError(
                "beta >= q/2 makes the problem trivial: almost any vector "
                "satisfies the norm bound"
            )
        if not 4 <= self.table_bits <= 32:
            raise PoWError("table_bits outside a sane range")
        if not 1 <= self.sparsity <= self.m:
            raise PoWError("sparsity must lie in [1, m]")

    @property
    def table_size(self) -> int:
        return 1 << self.table_bits

    @property
    def difficulty_bits(self) -> float:
        """Rough work exponent: the syndrome space that must collide.

        A random sparse candidate's syndrome is n log2(q) bits wide. Birthday
        search over a table of size T finds a collision after about
        2^((n log2 q)/2) candidates, so halving the syndrome space halves the
        exponent. Reported for difficulty tuning, not as a security claim.
        """
        import math

        return 0.5 * self.n * math.log2(self.q)


def derive_matrix(header: bytes, params: PoWParameters) -> tuple:
    """Expand the block header into A in Z_q^{n x m}, deterministically.

    Every miner and verifier must agree on A, so it comes from a XOF over
    the header with an explicit domain separator and the shape bound in --
    otherwise a different (n, m, q) could reuse the same header bytes.
    """
    if not header:
        raise PoWError("header must not be empty")
    width = (params.q.bit_length() + 7) // 8 + 2  # oversample to reduce bias
    hasher = hashlib.shake_256()
    hasher.update(CHALLENGE_DOMAIN)
    hasher.update(len(header).to_bytes(4, "big"))
    hasher.update(header)
    hasher.update(params.n.to_bytes(4, "big"))
    hasher.update(params.m.to_bytes(4, "big"))
    hasher.update(params.q.to_bytes(8, "big"))
    stream = hasher.digest(width * params.n * params.m)

    rows = []
    offset = 0
    for _ in range(params.n):
        row = []
        for _ in range(params.m):
            chunk = stream[offset:offset + width]
            offset += width
            row.append(int.from_bytes(chunk, "big") % params.q)
        rows.append(tuple(row))
    return tuple(rows)


def syndrome(matrix: tuple, vector, q: int) -> tuple:
    """A z mod q."""
    return tuple(
        sum(a * b for a, b in zip(row, vector)) % q for row in matrix
    )


def centered(value: int, q: int) -> int:
    """Representative in (-q/2, q/2]."""
    value %= q
    return value - q if value > q // 2 else value


def infinity_norm(vector, q: int) -> int:
    return max((abs(centered(v, q)) for v in vector), default=0)


@dataclass(frozen=True)
class Share:
    """A candidate solution: header + nonce + the short vector."""

    header: bytes
    nonce: int
    vector: tuple

    def encode(self) -> bytes:
        parts = [
            len(self.header).to_bytes(4, "big"),
            self.header,
            self.nonce.to_bytes(8, "big"),
            len(self.vector).to_bytes(4, "big"),
        ]
        for value in self.vector:
            parts.append((value % (1 << 32)).to_bytes(4, "big"))
        return b"".join(parts)

    @property
    def share_id(self) -> bytes:
        hasher = hashlib.shake_256()
        hasher.update(b"CHITIN-POW-SHARE-V1")
        hasher.update(self.encode())
        return hasher.digest(32)


@dataclass(frozen=True)
class Verification:
    """Per-gate verification result -- never a bare bool."""

    kernel_ok: bool
    norm_ok: bool
    nonzero_ok: bool
    dimension_ok: bool
    measured_norm: int

    @property
    def valid(self) -> bool:
        return (
            self.dimension_ok
            and self.nonzero_ok
            and self.norm_ok
            and self.kernel_ok
        )

    def reason(self) -> str:
        if not self.dimension_ok:
            return "wrong vector dimension"
        if not self.nonzero_ok:
            return "zero vector is a trivial kernel element"
        if not self.norm_ok:
            return f"norm {self.measured_norm} exceeds beta"
        if not self.kernel_ok:
            return "A z != 0 mod q"
        return "valid"


def verify_share(share: Share, params: PoWParameters) -> Verification:
    """Cheap, deterministic verification: one mat-vec plus a norm check.

    This is the asymmetry the whole scheme rests on -- verifying is
    microseconds while finding is memory-hard work.
    """
    vector = tuple(int(v) for v in share.vector)
    dimension_ok = len(vector) == params.m
    if not dimension_ok:
        return Verification(False, False, False, False, 0)

    nonzero_ok = any(v % params.q != 0 for v in vector)
    norm = infinity_norm(vector, params.q)
    norm_ok = norm <= params.beta

    matrix = derive_matrix(share.header, params)
    residue = syndrome(matrix, vector, params.q)
    kernel_ok = all(value == 0 for value in residue)

    return Verification(
        kernel_ok=kernel_ok,
        norm_ok=norm_ok,
        nonzero_ok=nonzero_ok,
        dimension_ok=dimension_ok,
        measured_norm=norm,
    )


def generate_candidate(header: bytes, nonce: int, index: int,
                       params: PoWParameters) -> tuple:
    """One sparse candidate vector, deterministically from (header, nonce, i).

    Determinism matters for two reasons: a pool can re-derive a miner's
    candidate from an index instead of trusting a submitted vector, and the
    GPU/CPU phases can regenerate the same stream without shipping it.
    """
    hasher = hashlib.shake_256()
    hasher.update(CANDIDATE_DOMAIN)
    hasher.update(len(header).to_bytes(4, "big"))
    hasher.update(header)
    hasher.update(nonce.to_bytes(8, "big"))
    hasher.update(index.to_bytes(8, "big"))
    stream = hasher.digest(params.sparsity * 5)

    vector = [0] * params.m
    offset = 0
    for _ in range(params.sparsity):
        position = int.from_bytes(stream[offset:offset + 4], "big") % params.m
        sign = 1 if stream[offset + 4] & 1 else -1
        offset += 5
        vector[position] += sign
    return tuple(v % params.q for v in vector)


__all__ = [
    "CANDIDATE_DOMAIN",
    "CHALLENGE_DOMAIN",
    "PoWError",
    "PoWParameters",
    "Share",
    "Verification",
    "centered",
    "derive_matrix",
    "generate_candidate",
    "infinity_norm",
    "syndrome",
    "verify_share",
]
