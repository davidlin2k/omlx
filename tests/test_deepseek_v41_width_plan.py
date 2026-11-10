# SPDX-License-Identifier: Apache-2.0
"""Unit 3 predicate: the JIT width kernel equals the policy on the same fixtures."""

from __future__ import annotations

import mlx.core as mx

from omlx.patches.deepseek_v41.width_plan import width_plan
from omlx.patches.deepseek_v41.width_policy import cumprod_rank_widths


def _as_mlx(remap, prev_rates):
    return mx.array(remap, mx.int32), mx.array(prev_rates, mx.float32)


FIXTURES = (
    ([0, 1], [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]], 6, 3),
    ([0, -1, 1], [[0.9, 0.8, 0.7], [0.4, 0.4, 0.4]], 7, 3),
    ([0, 1], [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], 8, 3),
    ([0, 1], [[0.3, 0.2, 0.1], [0.6, 0.5, 0.4]], 2, 3),
    ([-1, -1], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], 5, 3),
    ([0], [[0.7]], 1, 1),
    ([0, 1, 2, 3, 4, 5, 6, 7],
     [[0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99]] * 8, 64, 8),
)


def test_kernel_equals_policy_bit_for_bit():
    for remap, prev_rates, total_q, nextn in FIXTURES:
        remap_mx, rates_mx = _as_mlx(remap, prev_rates)
        got = width_plan(remap_mx, rates_mx, total_q).tolist()
        want = cumprod_rank_widths(remap, prev_rates, total_q, nextn)
        assert got == want, (remap, prev_rates, total_q, nextn, got, want)


def test_kernel_shape_and_dtype():
    remap_mx, rates_mx = _as_mlx([0, -1], [[0.9, 0.8], [0.1, 0.2]])
    widths = width_plan(remap_mx, rates_mx, 4)
    assert widths.shape == (2,)
    assert widths.dtype == mx.int32
    mx.eval(widths)


def test_kernel_rejects_an_out_of_range_remap():
    remap_mx, rates_mx = _as_mlx([0, 5], [[0.9], [0.1]])
    try:
        width_plan(remap_mx, rates_mx, 4)
    except ValueError:
        return
    raise AssertionError("an out-of-range remap must raise")


def test_kernel_caps_at_max_batch():
    remap = [0, 1, 2, 3, 4, 5, 6, 7]
    rates = [[0.99] * 8] * 8
    remap_mx, rates_mx = _as_mlx(remap, rates)
    widths = width_plan(remap_mx, rates_mx, 64, cap=8).tolist()
    assert all(1 <= width <= 8 for width in widths)
    assert sum(widths) >= len(remap)
