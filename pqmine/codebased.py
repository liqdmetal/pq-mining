"""Code-based PoW: syndrome decoding over F_2, with the published complexity data.

PRIOR ART (checked before writing this)
---------------------------------------
Nash, Chimal-Dzul, Choo, "On the Suitability of Syndrome Decoding for
Proof-of-Work under Quantum Adversaries" (ePrint 2026/1526) is the first
SDP-based PoW construction. It is very recent and its own conclusion is that
"further research is needed" -- so this module is implementing a research
proposal, not a settled design. Related: Behnia et al. lattice PoW (2021),
Equihash (generalized birthday), Ding's multivariate PoW (2019).

THE NUMBER THAT COMPLICATES THE CHOICE (Table 1 of that paper)
--------------------------------------------------------------
    problem   best classical    best quantum      exponent ratio
    hash      2^n               2^(n/2)           0.500
    SDP       2^(0.0966 n)      2^(0.05869 n)     0.607
    SVP       2^(0.292 n)       2^(0.265 n)       0.908

The paper's "quantum speedup" column compares absolute exponents per unit n,
but n means different things per problem (difficulty bits / code length /
lattice dimension), so that comparison is not apples-to-apples. The right
question is: at EQUAL CLASSICAL SECURITY, how much does quantum buy?

    classical 2^100 work  ->  quantum work
    hash   : 2^50   (loses 50 bits)
    SDP    : 2^60.7 (loses 39 bits)   <- this module
    SVP    : 2^90.7 (loses  9 bits)

So code-based is markedly better than hash but MUCH WORSE than lattice for
quantum resistance specifically. `quantum_advantage_table()` computes this.

Choosing SDP anyway is defensible, but for a DIFFERENT reason than quantum
resistance: cryptanalytic stability. Syndrome decoding has been studied since
Prange 1962 / Berlekamp-McEliece-van Tilborg 1978 with only incremental ISD
gains, whereas lattice reduction is an active frontier where a miner running
a better sieve gains an invisible advantage. Pick SDP for stability, not for
a smaller quantum gap.

WHY BINARY CODES SUIT A MEMORY-HARD PoW
---------------------------------------
Two structural wins over the M-SIS design in `pow.py`:

1. SYNDROME COMPUTATION IS FREE. Over F_2 the syndrome of a weight-w error is
   the XOR of w columns of H. Bit-packed, a column is (n-k)/64 machine words,
   so a candidate costs w XORs -- versus n*m modular multiplies for M-SIS.
   Since arithmetic was 97% of the M-SIS miner's time, making it nearly free
   is exactly what shifts the bottleneck onto the memory-bound table. The
   ASIC-friendly part of the work function largely disappears.

2. THE WORK *IS* THE MEMORY. Birthday/Stern-style search needs two lists of
   size L with L^2 >= 2^(n-k), so L ~ 2^((n-k)/2). The time and the space
   requirement are the same quantity -- you cannot trade memory away without
   paying more than quadratically in time. That is Equihash's argument
   resting on a post-quantum assumption instead of a hash.

A RISK I AM NOT HIDING
----------------------
The paper's difficulty knob is a "structured profile constraint" on the error
support. That is close to REGULAR syndrome decoding, and Esser-Santini
(CRYPTO 2024, "Not just regular decoding") showed regular SDP admits BETTER
attacks than plain SDP. So the structure added for difficulty calibration may
weaken the instance. This module therefore implements PLAIN SDP with a fixed
weight and full-syndrome match, and exposes the profile variant only as an
explicitly-flagged experiment. See `profile_constraint_warning()`.

Also: May-Ozerov's 2^0.0966n is a "galactic algorithm" -- its constants only
pay off at astronomical code lengths. At real parameters Stern/BJMM are what
attackers run, so difficulty must be calibrated against THOSE, not against
the best asymptotic exponent.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

CODE_DOMAIN = b"CHITIN-SDP-CODE-V1"
SYNDROME_DOMAIN = b"CHITIN-SDP-SYNDROME-V1"

# Published exponents, for the honest comparison above.
CLASSICAL_EXPONENTS = {
    "hash": 1.0,
    "sdp_may_ozerov": 0.0966,
    "svp_sieve": 0.292,
}
QUANTUM_EXPONENTS = {
    "hash": 0.5,
    "sdp_kachigar_tillich": 0.05869,
    "svp_sieve": 0.265,
}


class CodeBasedError(ValueError):
    """Raised on malformed code-based PoW parameters or shares."""


@dataclass(frozen=True)
class SDPParameters:
    """A syndrome-decoding PoW instance shape.

    n      code length (number of columns / candidate positions)
    k      code dimension, so the syndrome is n-k bits wide
    weight target error weight w (exact, not <= t: exactness makes the
           verifier's job trivial and removes a degree of freedom an
           attacker could exploit by submitting lower-weight solutions)
    """

    n: int
    k: int
    weight: int

    def __post_init__(self) -> None:
        if self.n <= 0:
            raise CodeBasedError("code length must be positive")
        if not 0 < self.k < self.n:
            raise CodeBasedError("dimension must satisfy 0 < k < n")
        if not 0 < self.weight <= self.n:
            raise CodeBasedError("weight must lie in (0, n]")
        if self.syndrome_bits < 8:
            raise CodeBasedError(
                "syndrome narrower than 8 bits makes the PoW trivial"
            )
        if self.weight % 2 != 0:
            raise CodeBasedError(
                "weight must be even so the birthday split is balanced"
            )

    @property
    def syndrome_bits(self) -> int:
        return self.n - self.k

    @property
    def log2_search_space(self) -> float:
        """log2 C(n, w): how many weight-w vectors exist."""
        return math.log2(math.comb(self.n, self.weight))

    @property
    def log2_expected_solutions(self) -> float:
        """log2( C(n,w) / 2^(n-k) ) -- must be >= 0 or instances are often
        unsatisfiable, which stalls mining."""
        return self.log2_search_space - self.syndrome_bits

    @property
    def satisfiable(self) -> bool:
        return self.log2_expected_solutions >= 0.0

    @property
    def log2_birthday_work(self) -> float:
        """Half the syndrome width: L ~ 2^((n-k)/2) list entries needed."""
        return self.syndrome_bits / 2.0

    @property
    def birthday_memory_entries(self) -> float:
        return 2.0 ** self.log2_birthday_work

    def summary(self) -> dict:
        return {
            "n": self.n,
            "k": self.k,
            "weight": self.weight,
            "syndrome_bits": self.syndrome_bits,
            "log2_search_space": round(self.log2_search_space, 2),
            "log2_expected_solutions": round(self.log2_expected_solutions, 2),
            "satisfiable": self.satisfiable,
            "log2_birthday_work": round(self.log2_birthday_work, 2),
            "birthday_memory_entries": int(self.birthday_memory_entries),
        }


def derive_columns(header: bytes, params: SDPParameters) -> tuple:
    """Expand the header into H's n columns, each a bit-packed integer.

    Column-major is the right layout: the syndrome of an error is the XOR of
    the columns in its support, so columns are the unit of work. Each column
    is an (n-k)-bit integer, which Python XORs natively.
    """
    if not header:
        raise CodeBasedError("header must not be empty")
    width_bytes = (params.syndrome_bits + 7) // 8
    mask = (1 << params.syndrome_bits) - 1

    hasher = hashlib.shake_256()
    hasher.update(CODE_DOMAIN)
    hasher.update(len(header).to_bytes(4, "big"))
    hasher.update(header)
    hasher.update(params.n.to_bytes(4, "big"))
    hasher.update(params.k.to_bytes(4, "big"))
    hasher.update(params.weight.to_bytes(4, "big"))
    stream = hasher.digest(width_bytes * params.n)

    columns = []
    for index in range(params.n):
        chunk = stream[index * width_bytes:(index + 1) * width_bytes]
        columns.append(int.from_bytes(chunk, "big") & mask)
    return tuple(columns)


def derive_target_syndrome(header: bytes, params: SDPParameters) -> int:
    """The syndrome s the miner must hit, from the header."""
    if not header:
        raise CodeBasedError("header must not be empty")
    width_bytes = (params.syndrome_bits + 7) // 8
    hasher = hashlib.shake_256()
    hasher.update(SYNDROME_DOMAIN)
    hasher.update(len(header).to_bytes(4, "big"))
    hasher.update(header)
    hasher.update(params.n.to_bytes(4, "big"))
    hasher.update(params.k.to_bytes(4, "big"))
    return int.from_bytes(hasher.digest(width_bytes), "big") & (
        (1 << params.syndrome_bits) - 1
    )


def syndrome_of(support, columns: tuple) -> int:
    """XOR the columns in `support`. This is the entire work function."""
    accumulator = 0
    for position in support:
        accumulator ^= columns[position]
    return accumulator


@dataclass(frozen=True)
class SDPShare:
    """A candidate solution: the error support (sorted, distinct positions)."""

    header: bytes
    support: tuple

    def encode(self) -> bytes:
        parts = [len(self.header).to_bytes(4, "big"), self.header,
                 len(self.support).to_bytes(4, "big")]
        for position in self.support:
            parts.append(position.to_bytes(4, "big"))
        return b"".join(parts)

    @property
    def share_id(self) -> bytes:
        hasher = hashlib.shake_256()
        hasher.update(b"CHITIN-SDP-SHARE-V1")
        hasher.update(self.encode())
        return hasher.digest(32)


@dataclass(frozen=True)
class SDPVerification:
    """Per-gate verification -- never a bare bool."""

    weight_ok: bool
    distinct_ok: bool
    range_ok: bool
    syndrome_ok: bool
    measured_weight: int

    @property
    def valid(self) -> bool:
        return (
            self.range_ok
            and self.distinct_ok
            and self.weight_ok
            and self.syndrome_ok
        )

    def reason(self) -> str:
        if not self.range_ok:
            return "support position outside [0, n)"
        if not self.distinct_ok:
            return "duplicate position: support must be a set"
        if not self.weight_ok:
            return f"weight {self.measured_weight} != required weight"
        if not self.syndrome_ok:
            return "H e^T != s"
        return "valid"


def verify_share(share: SDPShare, params: SDPParameters) -> SDPVerification:
    """Verify in w XORs plus a set check. This is the asymmetry.

    Note the duplicate check: over F_2 a repeated position cancels itself
    (x ^ x == 0), so without it a miner could pad a low-weight solution up to
    the required weight for free. That is a real forgery, not a formality.
    """
    support = tuple(int(p) for p in share.support)
    range_ok = all(0 <= p < params.n for p in support)
    distinct_ok = len(set(support)) == len(support)
    measured_weight = len(set(support))
    weight_ok = measured_weight == params.weight

    if not (range_ok and distinct_ok and weight_ok):
        return SDPVerification(
            weight_ok=weight_ok, distinct_ok=distinct_ok, range_ok=range_ok,
            syndrome_ok=False, measured_weight=measured_weight,
        )

    columns = derive_columns(share.header, params)
    target = derive_target_syndrome(share.header, params)
    syndrome_ok = syndrome_of(support, columns) == target

    return SDPVerification(
        weight_ok=True, distinct_ok=True, range_ok=True,
        syndrome_ok=syndrome_ok, measured_weight=measured_weight,
    )


def quantum_advantage_table(classical_bits: float = 100.0) -> dict:
    """At equal CLASSICAL security, how many bits does quantum buy?

    This is the comparison the paper's Table 1 does not make directly, and
    it reverses the apparent ranking: lattice loses the LEAST to quantum.
    """
    rows = {}
    for label, (classical, quantum) in {
        "hash": (CLASSICAL_EXPONENTS["hash"], QUANTUM_EXPONENTS["hash"]),
        "sdp": (CLASSICAL_EXPONENTS["sdp_may_ozerov"],
                QUANTUM_EXPONENTS["sdp_kachigar_tillich"]),
        "svp": (CLASSICAL_EXPONENTS["svp_sieve"],
                QUANTUM_EXPONENTS["svp_sieve"]),
    }.items():
        dimension = classical_bits / classical
        quantum_bits = quantum * dimension
        rows[label] = {
            "classical_exponent": classical,
            "quantum_exponent": quantum,
            "exponent_ratio": round(quantum / classical, 4),
            "dimension_for_target": round(dimension, 1),
            "quantum_bits": round(quantum_bits, 1),
            "bits_lost_to_quantum": round(classical_bits - quantum_bits, 1),
        }
    return {
        "classical_target_bits": classical_bits,
        "rows": rows,
        "verdict": (
            "lattice loses the fewest bits to quantum; SDP is intermediate; "
            "hash loses the most. Choose SDP for CRYPTANALYTIC STABILITY, "
            "not for quantum resistance."
        ),
    }


def profile_constraint_warning() -> dict:
    """Why this module implements PLAIN SDP, not the paper's profile variant."""
    return {
        "paper_proposal": (
            "ePrint 2026/1526 adds a 'structured profile constraint' on the "
            "error support to control solution density and give extra "
            "difficulty-calibration parameters."
        ),
        "risk": (
            "That constraint is close to REGULAR syndrome decoding, and "
            "Esser-Santini (CRYPTO 2024) showed regular SDP admits strictly "
            "better attacks than plain SDP. Structure added for difficulty "
            "tuning can therefore reduce the security of the instance."
        ),
        "decision": (
            "implement plain SDP with exact weight and full-syndrome match; "
            "expose profile variants only behind an explicit experimental flag"
        ),
        "galactic_caveat": (
            "May-Ozerov's 2^0.0966n only beats Stern/BJMM at astronomical n, "
            "so difficulty must be calibrated against the algorithms an "
            "attacker actually runs, not the best asymptotic exponent."
        ),
        "status": "OPEN -- needs a profile-aware ISD analysis before use",
    }


__all__ = [
    "CLASSICAL_EXPONENTS",
    "CODE_DOMAIN",
    "CodeBasedError",
    "QUANTUM_EXPONENTS",
    "SDPParameters",
    "SDPShare",
    "SDPVerification",
    "SYNDROME_DOMAIN",
    "derive_columns",
    "derive_target_syndrome",
    "profile_constraint_warning",
    "quantum_advantage_table",
    "syndrome_of",
    "verify_share",
]
