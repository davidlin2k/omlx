# SPDX-License-Identifier: Apache-2.0
"""Context-parallel mirror split for long prefill, mapped from DeepInfer.

DeepInfer splits a sequence into mirror blocks so every rank gets the same row
count: each rank takes block ``r`` and block ``2 * cp - 1 - r``, which pairs the
head and the tail to the same rank. The equal row count is the balanced part.
This module is the split arithmetic, without the all-gather runtime.

Source of the rule: DeepInfer's tilelang ``prepare_cp_mirror`` and its
parallelism report.
"""

from __future__ import annotations

import math


def chunk_size(seq_len: int, cp_size: int) -> int:
    """Block size ``ceil(sq / (2 * cp)``."""
    if seq_len <= 0 or cp_size <= 0:
        raise ValueError("sequence length and cp size must be positive")
    return math.ceil(seq_len / (2 * cp_size))


def mirror_blocks(cp_size: int) -> tuple[tuple[int, int], ...]:
    """Block pair per rank, head paired with tail."""
    if cp_size <= 0:
        raise ValueError("cp size must be positive")
    return tuple((rank, 2 * cp_size - 1 - rank) for rank in range(cp_size))


def mirror_rows(seq_len: int, cp_size: int) -> tuple[tuple[int, ...], ...]:
    """Token indices per rank in mirror order, ``-1`` where a block overruns."""
    ps = chunk_size(seq_len, cp_size)
    rows = []
    for first, second in mirror_blocks(cp_size):
        entry: list[int] = []
        for block in (first, second):
            for offset in range(ps):
                index = block * ps + offset
                entry.append(index if index < seq_len else -1)
        rows.append(tuple(entry))
    return tuple(rows)


def mirror_row_count(seq_len: int, cp_size: int) -> int:
    """Rows every rank receives, identical across ranks."""
    return 2 * chunk_size(seq_len, cp_size)
