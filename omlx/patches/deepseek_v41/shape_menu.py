# SPDX-License-Identifier: Apache-2.0
"""Fixed shape menu for decode submissions, the graph-capture analog.

MLX exposes no graph capture, so a fixed shape menu plus command-buffer batching
stands in for it. DeepInfer builds its CUDA-graph batch sizes as powers of two
below a fixed interval, then interval steps to the maximum, captured largest
first so one shared pool is sliced by smaller sizes.

Source of the rule: DeepInfer's ``build_graph_batch_sizes`` in its CUDA-graph
runner.
"""

from __future__ import annotations

DEFAULT_INTERVAL = 8


def graph_sizes(max_rows: int, interval: int = DEFAULT_INTERVAL) -> list[int]:
    """Menu entries, largest first: powers of two below the interval, then steps."""
    if max_rows <= 0:
        raise ValueError("max rows must be positive")
    if interval <= 0:
        raise ValueError("interval must be positive")
    sizes: set[int] = set()
    power = 1
    while power < interval and power <= max_rows:
        sizes.add(power)
        power *= 2
    step = interval
    while step <= max_rows:
        sizes.add(step)
        step += interval
    sizes.add(max_rows)
    return sorted(sizes, reverse=True)


def snap_to_menu(value: int, menu: list[int]) -> int:
    """Smallest menu entry at or above ``value``; the largest entry if none is."""
    if not menu:
        raise ValueError("menu must not be empty")
    for entry in sorted(menu):
        if entry >= value:
            return entry
    return max(menu)


def menu_size(menu: list[int]) -> int:
    """How many distinct shapes the menu allocates."""
    return len(set(menu))
