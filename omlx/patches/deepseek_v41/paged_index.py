# SPDX-License-Identifier: Apache-2.0
"""Paged KV index conversion: logical token index to physical slot.

DeepInfer's attention path converts a logical KV index into a physical slot
through the block table, with ``-1`` marking padding or an absent tail. omlx's
V4.1 model-level cache keeps contiguous per-request rows, so this conversion is
the missing piece for a paged attention path.
"""

from __future__ import annotations


def convert_kv_index(
    indices: list[int], block_tables: list[int], block_size: int
) -> list[int]:
    """Physical slot per logical index; a negative index stays ``-1``."""
    if block_size <= 0:
        raise ValueError("block size must be positive")
    out: list[int] = []
    for index in indices:
        if index < 0:
            out.append(-1)
            continue
        block, offset = divmod(index, block_size)
        if block >= len(block_tables):
            out.append(-1)
            continue
        out.append(block_tables[block] * block_size + offset)
    return out


def contiguous_index(start: int, count: int) -> list[int]:
    """A contiguous logical range, the non-paged case."""
    return [start + offset for offset in range(max(0, count))]


def ring_index(start: int, count: int, window: int) -> list[int]:
    """A ring expansion over a window, ``-1`` beyond the live tail."""
    if window <= 0:
        raise ValueError("window must be positive")
    out: list[int] = []
    for offset in range(max(0, count)):
        position = start + offset
        out.append(position % window if position >= 0 else -1)
    return out


def physical_slot_rows(
    indices: list[int], block_tables: list[int], block_size: int, rows: int
) -> list[int]:
    """Flatten converted slots into one row per query, ``-1`` padded."""
    converted = convert_kv_index(indices, block_tables, block_size)
    if rows <= 0:
        raise ValueError("rows must be positive")
    padded = converted + [-1] * max(0, rows - len(converted))
    return padded[:rows]
