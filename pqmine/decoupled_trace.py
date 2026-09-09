"""Pure-Python trace of the decoupled selector.

This is the shared reference for two consumers:

* ``search_share`` turns the selected columns into a real verifier Share;
* ``restriction_attack`` measures the actual reachable-support distribution.

The CUDA/Numba benchmark must remain behaviorally equivalent to this trace.
It is intentionally slow and not a production miner.
"""

from __future__ import annotations

from dataclasses import dataclass

from .decoupled import selection_threshold
from .latency_bound import LatencyParameters

_MASK64 = (1 << 64) - 1


@dataclass(frozen=True)
class DecoupledTrace:
    syndrome: int
    selected_columns: tuple[int, ...]
    visited: tuple[int, ...]
    steps: int

    @property
    def weight(self) -> int:
        return len(self.selected_columns)


def mix64(x: int) -> int:
    """The SplitMix-style mixer used by the CUDA/Numba decoupled path."""
    x &= _MASK64
    x ^= x >> 30
    x = (x * 0xBF58476D1CE4E5B9) & _MASK64
    x ^= x >> 27
    x = (x * 0x94D049BB133111EB) & _MASK64
    x ^= x >> 31
    return x & _MASK64


def trace_decoupled_walk(
    arena: bytearray,
    nonce: int,
    params: LatencyParameters,
    columns: tuple[int, ...],
    target_weight: int,
) -> DecoupledTrace:
    """Trace the exact selector/state transition used by ``decoupled.py``."""
    if not (0 <= nonce < 2**64):
        raise ValueError("nonce must be a uint64")
    if not columns:
        raise ValueError("columns must not be empty")
    if not (0 < target_weight <= params.steps):
        raise ValueError("target_weight must lie in (0, steps]")
    if target_weight > len(columns):
        raise ValueError("target_weight cannot exceed available columns")
    if len(arena) != params.arena_bytes:
        raise ValueError("arena size does not match latency parameters")

    cell_bytes = params.cell_bytes
    mask = params.cells - 1
    threshold = selection_threshold(target_weight, params.steps)
    address = nonce & mask
    syndrome = 0
    visited: list[int] = []
    selected: list[int] = []
    used: set[int] = set()

    for step in range(params.steps):
        offset = address * cell_bytes
        value = int.from_bytes(arena[offset:offset + 8], "little")
        x = mix64(value ^ nonce ^ step)
        arena[offset:offset + 8] = (value ^ x).to_bytes(8, "little")
        visited.append(address)

        if (x >> 56) < threshold:
            # Once every column is selected, there is no new distinct support
            # element. Do not spin forever trying to rehash into one.
            if len(used) == len(columns):
                continue
            y = x
            column = y % len(columns)
            while column in used:
                y = mix64(y)
                column = y % len(columns)
            used.add(column)
            selected.append(column)
            syndrome ^= columns[column]

        address = x & mask

    return DecoupledTrace(
        syndrome=syndrome,
        selected_columns=tuple(selected),
        visited=tuple(visited),
        steps=params.steps,
    )


__all__ = ["DecoupledTrace", "mix64", "trace_decoupled_walk"]
