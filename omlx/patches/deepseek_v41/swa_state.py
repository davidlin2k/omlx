# SPDX-License-Identifier: Apache-2.0
"""SWA state persistence at 128-token granularity.

DeepInfer stores SWA KV only at positions that are multiples of the store grain.
After prefill, the store position reserves one token at the end of the sequence
and rounds down; less than one full unit stores nothing. During decode a replica
region takes the window at every grain crossing, and the attachment point at
finish rounds down to the grain. A copy can overshoot the boundary by at most
five tokens, absorbed by the six extra positions the ring has beyond the window.

Source of the rule: DeepInfer's V4.1 deployment Q&A, the "SWA KV cache" section.
"""

from __future__ import annotations

STORE_GRAIN = 128
RING_EXTRA_POSITIONS = 6
MAX_COPY_OVERSHOOT = 5


def store_position(offset: int, grain: int = STORE_GRAIN) -> int:
    """Rounded-down multiple after reserving one token at the sequence end."""
    if offset <= 0:
        return 0
    candidate = (offset - 1) // grain * grain
    return max(0, candidate)


def store_units(offset: int, grain: int = STORE_GRAIN) -> int:
    """How many full grain-sized units the sequence can store."""
    return store_position(offset, grain) // grain


def should_store(offset: int, grain: int = STORE_GRAIN) -> bool:
    """True only when at least one full unit exists to store."""
    return store_units(offset, grain) > 0


def crossing_boundaries(start: int, end: int, grain: int = STORE_GRAIN) -> list[int]:
    """Grain multiples crossed between two offsets, in ascending order."""
    if end <= start:
        return []
    first = (start // grain + 1) * grain
    return list(range(first, end + 1, grain))


def copy_position(offset: int, grain: int = STORE_GRAIN) -> int:
    """Attachment point of the most recent copy, rounded down to the grain."""
    return max(0, offset // grain * grain)


def overshoot_absorbed(copy_offset: int, grain: int = STORE_GRAIN) -> bool:
    """True when the copy's overshoot past its boundary fits the ring's extra slots."""
    overshoot = copy_offset % grain
    return 0 <= overshoot <= MAX_COPY_OVERSHOOT and RING_EXTRA_POSITIONS >= overshoot
