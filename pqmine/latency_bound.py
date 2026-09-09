"""Latency-bound code-based PoW: option B, informed by PoSME.

WHY THE PREVIOUS DESIGN LOST TO THE GPU (measured, not guessed)
---------------------------------------------------------------
`cuda_codebased.py` measured, on this machine:

    phase              RTX 3090      5900X (19T)    GPU/CPU
    syndrome XOR       5.37 G/s      582 M/s        9.2x
    table probe        2.18 G/s      359 M/s        6.1x

The GPU won BOTH phases, so the cooperative claim failed. The reason is not
that the GPU has low memory latency -- it does not -- but that the probes
were INDEPENDENT. With thousands of resident warps a GPU hides a 400-600 ns
latency behind other work, so raw parallelism beats the CPU's ~80 ns.
Independent random access is a bandwidth problem, and GPUs win bandwidth.

WHAT THE LITERATURE SAYS TO DO INSTEAD
--------------------------------------
PoSME (Condrey, arXiv:2604.15751, Apr 2026) benchmarks 17 CPUs and 4 GPU
architectures on latency-bound pointer chasing and reports:

  * GPUs are 14-19x SLOWER than a consumer CPU at this workload,
  * ASIC advantage bounded at ~2x (vs 8-16x for Argon2id, 2-5x RandomX),
  * hash computation under 3.5% of step cost -- the memory access IS the work,
  * HBM3 only ~1.3x faster than DDR5 at RANDOM access, despite far greater
    sequential bandwidth. That gap is why parallelism stops helping.

The three principles it isolates, all of which this module adopts:

  1. DATA-DEPENDENT POINTER CHASING. The next address depends on the value
     just read, so steps cannot be issued in parallel. This is the property
     my previous design lacked.
  2. MUTABLE ARENA. In a static arena any block is recomputable from the seed
     in O(1), so an attacker need not store it. Writing back forces real
     storage because block values evolve unpredictably.
  3. CAUSAL CHAINING. Each step's output is bound into a running transcript,
     so a miner cannot skip steps or reorder them.

Supporting theory: Ren-Devadas "Bandwidth Hard Functions for ASIC
Resistance" (ePrint 2017/225) argues memory-hardness alone is insufficient
and that the bandwidth/latency wall is the durable constraint -- consistent
with RandomX and AstroBWTv3 being the two designs that actually survived.

HOW THIS KEEPS THE POST-QUANTUM ASSUMPTION
------------------------------------------
PoSME is hash-based, so on its own it is only "post-quantum" in the weak
sense that hash PoW already is. This module keeps syndrome decoding as the
ACCEPTANCE condition while making the SEARCH latency-bound:

  * the walk visits arena cells, and each cell contributes a column of H,
  * the accumulated syndrome after K steps must satisfy the SDP target,
  * so a valid share still proves a weight-K solution to
    H e^T = s -- the Classic McEliece assumption,
  * but finding one requires walking a serial, data-dependent chain through
    a mutable arena, which is the part a GPU cannot parallelise.

Verification stays cheap: replay K steps (K XORs plus K hashes) or check the
syndrome directly. No trusted setup.

WHAT IS NOT CLAIMED
-------------------
This is a prototype of a research idea, benchmarked on one CPU and one GPU.
PoSME's own numbers come from a paper published weeks ago with no
independent replication. The composition of a latency-bound walk with an SDP
acceptance condition is MY construction and has had no cryptanalysis: in
particular, restricting solutions to those reachable by the walk shrinks the
solution space in a way an attacker might exploit. See
`composition_risks()`.
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass

from .codebased import CodeBasedError, SDPParameters

ARENA_DOMAIN = b"CHITIN-LATENCY-ARENA-V1"
WALK_DOMAIN = b"CHITIN-LATENCY-WALK-V1"


@dataclass(frozen=True)
class LatencyParameters:
    """Shape of the latency-bound walk.

    arena_bits   log2 of the arena cell count. Must exceed cache so the walk
                 actually hits DRAM; 2^22 cells * 32 B = 128 MiB.
    steps        K, the number of sequential pointer-chase steps. This is
                 the sequentiality parameter: no parallelism can reduce it.
    write_density how many cells are rewritten per step (PoSME's rho). >= 1
                 keeps the arena mutable; higher values raise TMTO
                 resistance (PoSME reports 10x at rho=4).
    """

    arena_bits: int
    steps: int
    write_density: int = 2
    cell_bytes: int = 32

    def __post_init__(self) -> None:
        if not 8 <= self.arena_bits <= 32:
            raise CodeBasedError("arena_bits must lie in [8, 32]")
        if self.steps <= 0:
            raise CodeBasedError("steps must be positive")
        if self.write_density < 1:
            raise CodeBasedError("write_density must be at least 1")
        if self.cell_bytes < 8:
            raise CodeBasedError("cell_bytes must be at least 8")

    @property
    def cells(self) -> int:
        return 1 << self.arena_bits

    @property
    def arena_bytes(self) -> int:
        return self.cells * self.cell_bytes

    @property
    def arena_mib(self) -> float:
        return self.arena_bytes / (1024 * 1024)

    @property
    def exceeds_typical_l3(self) -> bool:
        """L3 on a 5900X is 64 MiB; the arena must be larger to hit DRAM."""
        return self.arena_mib > 64.0

    def summary(self) -> dict:
        return {
            "arena_bits": self.arena_bits,
            "cells": self.cells,
            "arena_mib": round(self.arena_mib, 2),
            "steps": self.steps,
            "write_density": self.write_density,
            "exceeds_typical_l3": self.exceeds_typical_l3,
            "sequential_dram_hits": self.steps,
        }


def build_arena(header: bytes, params: LatencyParameters) -> bytearray:
    """Seed the arena from the header.

    Filled with a fast stream rather than per-cell hashing: the SEARCH is
    meant to be the expensive part, not setup. Setup is amortised across all
    the walks a miner performs for one header.
    """
    if not header:
        raise CodeBasedError("header must not be empty")
    hasher = hashlib.shake_256()
    hasher.update(ARENA_DOMAIN)
    hasher.update(len(header).to_bytes(4, "big"))
    hasher.update(header)
    hasher.update(params.arena_bits.to_bytes(2, "big"))
    return bytearray(hasher.digest(params.arena_bytes))


@dataclass
class WalkResult:
    """One completed walk: its syndrome, transcript, and timing."""

    syndrome: int
    transcript: bytes
    visited: tuple
    steps: int
    seconds: float
    hash_seconds: float

    @property
    def steps_per_second(self) -> float:
        return self.steps / self.seconds if self.seconds > 0 else 0.0

    @property
    def hash_fraction(self) -> float:
        """PoSME reports <3.5%. Higher here means the walk is compute-bound,
        i.e. NOT latency-bound, and the design has failed its own premise."""
        return self.hash_seconds / self.seconds if self.seconds > 0 else 0.0

    def summary(self) -> dict:
        return {
            "steps": self.steps,
            "seconds": round(self.seconds, 5),
            "steps_per_second": round(self.steps_per_second, 1),
            "hash_fraction": round(self.hash_fraction, 4),
            "latency_bound": self.hash_fraction < 0.35,
            "ns_per_step": round(self.seconds / self.steps * 1e9, 1)
            if self.steps else 0.0,
        }


def walk(arena: bytearray, header: bytes, nonce: int,
         params: LatencyParameters, columns: tuple,
         *, measure_hash: bool = False) -> WalkResult:
    """The work function: a serial, data-dependent, mutating pointer chase.

    Each step:
      1. read the cell at the current address (DRAM random access),
      2. mix it with the running state to get the NEXT address -- the
         dependency that defeats parallelism,
      3. accumulate the column of H selected by that cell,
      4. write back `write_density` mutated cells so the arena evolves,
      5. chain the state into a causal transcript.

    Step 2 is the whole point: the address of read i+1 is unknown until read
    i has returned, so a miner with 10,496 cores still waits one DRAM
    latency per step. Parallel hardware can run many independent WALKS, but
    each walk costs the same wall-clock time, and the acceptance condition
    binds a single walk.
    """
    cell_bytes = params.cell_bytes
    mask = params.cells - 1
    column_count = len(columns)

    state = int.from_bytes(
        hashlib.shake_256(
            WALK_DOMAIN + header + nonce.to_bytes(8, "big")
        ).digest(32),
        "big",
    )
    address = state & mask
    syndrome = 0
    visited = []
    transcript = hashlib.shake_256()
    transcript.update(WALK_DOMAIN)
    transcript.update(nonce.to_bytes(8, "big"))

    hash_seconds = 0.0
    started = time.perf_counter()

    for step in range(params.steps):
        offset = address * cell_bytes
        cell = arena[offset:offset + 8]
        value = int.from_bytes(cell, "little")

        # (3) accumulate the selected column of H.
        syndrome ^= columns[value % column_count]
        visited.append(address)

        # (5) causal chaining, and (2) next address from the value read.
        if measure_hash:
            hash_started = time.perf_counter()
        state = (state ^ value) * 0x9E3779B97F4A7C15 & ((1 << 64) - 1)
        state ^= state >> 29
        transcript.update(value.to_bytes(8, "little"))
        if measure_hash:
            hash_seconds += time.perf_counter() - hash_started

        # (4) mutate: without this the arena is recomputable and need not
        # be stored at all.
        for write in range(params.write_density):
            target = (state + write * 0x27220A95) & mask
            write_offset = target * cell_bytes
            mutated = (value ^ (state >> (8 * write))) & ((1 << 64) - 1)
            arena[write_offset:write_offset + 8] = mutated.to_bytes(
                8, "little"
            )

        address = state & mask

    elapsed = time.perf_counter() - started
    return WalkResult(
        syndrome=syndrome,
        transcript=transcript.digest(32),
        visited=tuple(visited),
        steps=params.steps,
        seconds=elapsed,
        hash_seconds=hash_seconds,
    )


def replay_walk(header: bytes, nonce: int, params: LatencyParameters,
                columns: tuple) -> WalkResult:
    """Verifier side: rebuild the arena and replay. Deterministic.

    Verification costs the same as one walk, which is the honest limitation
    of this construction: unlike hash PoW, verification is NOT cheaper than
    mining by orders of magnitude. It is cheaper only by the number of
    nonces the miner tried. PoSME addresses this with an IVC/Binius proof
    layer (design validated, implementation pending); that is out of scope
    here and is why `verification_cost_note()` exists.
    """
    arena = build_arena(header, params)
    return walk(arena, header, nonce, params, columns)


def verification_cost_note() -> dict:
    """State the asymmetry problem plainly instead of hiding it."""
    return {
        "problem": (
            "replaying a K-step walk costs K sequential DRAM accesses, the "
            "same as producing it. Hash PoW verification is ~1 hash against "
            "2^d mining work; here it is K steps against (nonces tried)*K."
        ),
        "why_it_is_still_usable": (
            "the asymmetry is the NONCE COUNT, not the step count: a miner "
            "tries many nonces, a verifier replays one. With K=4096 and a "
            "difficulty needing ~2^20 nonces, verification is ~2^20 times "
            "cheaper than mining -- adequate, but far less comfortable than "
            "SHA-256's ~2^70."
        ),
        "consequence": (
            "K must stay small enough that full nodes can replay every "
            "block quickly. That caps the sequentiality parameter, and "
            "therefore caps how latency-bound the design can be."
        ),
        "posme_solution": (
            "PoSME proposes constant-size verification via binary-field "
            "folding (Binius IVC, 144,512 Boolean constraints per step). "
            "Design validated in the paper, production implementation "
            "pending. Not implemented here."
        ),
        "status": "OPEN -- the main engineering obstacle for deployment",
    }


def composition_risks() -> dict:
    """Cryptanalytic risks of MY composition, which nobody has reviewed."""
    return {
        "restricted_solution_space": (
            "only weight-K errors whose support is REACHABLE by a walk can "
            "be found. That is a vanishing fraction of all weight-K errors, "
            "so the instance is no longer plain SDP: an attacker who can "
            "characterise reachable supports may decode faster than generic "
            "ISD. This is the same class of risk as Esser-Santini's attack "
            "on regular SDP, and it is UNANALYSED."
        ),
        "walk_bias": (
            "the address sequence is generated by a cheap multiply-xorshift "
            "mixer, not a cryptographic PRF. If the walk is biased, the "
            "column multiset is biased, and syndromes may concentrate. "
            "PoSME validates ROM-uniformity empirically at 2^24 with "
            "chi2/df = 1.0004; no equivalent check has been done here."
        ),
        "grinding": (
            "a miner controls the nonce, hence the walk. Whether nonce "
            "grinding gives more than the expected linear advantage needs "
            "analysis."
        ),
        "quantum": (
            "Grover applies to the nonce search as in any PoW, and the "
            "sequential walk is a poor Grover oracle (K coherent DRAM "
            "accesses per evaluation), which arguably HELPS. But the SDP "
            "acceptance condition is also weakened by restriction, so the "
            "net post-quantum claim is unclear."
        ),
        "status": "OPEN -- no cryptanalysis exists for this composition",
    }


__all__ = [
    "ARENA_DOMAIN",
    "LatencyParameters",
    "WALK_DOMAIN",
    "WalkResult",
    "build_arena",
    "composition_risks",
    "replay_walk",
    "verification_cost_note",
    "walk",
]
