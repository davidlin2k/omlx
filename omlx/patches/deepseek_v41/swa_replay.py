# SPDX-License-Identifier: Apache-2.0
"""Encoder-side SWA bounded-replay policy.

DeepInfer's decoder replay is recomputation of a bounded suffix; its encoder
replay recomputes only the most recent window when the SWA KV cache misses for
the hit prefix, while the block KV cache is reused as-is. This module is the hit
policy, the part that decides the replay extent, without the engine integration.

Source of the rule: DeepInfer's V4.1 deployment Q&A, the "KV cache hits and
recomputation" and "SWA KV cache" sections.
"""

from __future__ import annotations

STORE_GRAIN = 128
DEFAULT_WINDOW = 128
DEFAULT_BACKTRACKING = 6


def replay_cap(data_len: int, window: int = DEFAULT_WINDOW, backtracking: int = DEFAULT_BACKTRACKING) -> int:
    """The longest hit under the truncation requirement, ``L - window - backtracking``."""
    return max(0, data_len - window - backtracking)


def hit_plan(
    data_len: int,
    block_hit: int,
    swa_present_in_last_window: bool,
    window: int = DEFAULT_WINDOW,
    backtracking: int = DEFAULT_BACKTRACKING,
    grain: int = STORE_GRAIN,
) -> int:
    """Hit length from the block cache, backed off one window when SWA misses.

    The block hit is a multiple of the store grain. If the SWA cache exists in
    ``[L1 - window, L1]`` the hit is ``L1``; otherwise it is ``L1 - window``.
    Either way the result is capped at ``L - window - backtracking``.
    """
    if data_len <= 0 or block_hit <= 0:
        return 0
    l1 = block_hit // grain * grain
    chosen = l1 if swa_present_in_last_window else max(0, l1 - window)
    return max(0, min(chosen, replay_cap(data_len, window, backtracking)))


def replay_needs_swa(data_len: int, block_hit: int, window: int = DEFAULT_WINDOW) -> bool:
    """True when the hit prefix is long enough that a window replay is required."""
    return data_len > 0 and block_hit >= window


def swa_visibility(
    positions: list[int],
    valid_start: int,
    rect_lo: int,
    window: int = DEFAULT_WINDOW,
) -> list[int]:
    """Visibility start per query: the max of window start, rectangle, and valid start.

    This is DeepInfer's bounded-replay visibility rule, which makes the kept
    range's attention values identical between a compact replay and a full pass.
    """
    return [
        max(position - window + 1, rect_lo, valid_start, 0) for position in positions
    ]
