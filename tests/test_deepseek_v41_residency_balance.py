# SPDX-License-Identifier: Apache-2.0
"""Unit 10 predicate: greedy residency placement lowers slowest-slot over mean."""

from __future__ import annotations

from omlx.patches.deepseek_v41.residency_balance import (
    moe_lb,
    naive_placement,
    rebalance,
)


def _skewed_trace():
    # experts 0 and 3 are hot; the naive round-robin stacks both on slot 0.
    return [[100.0, 10.0, 10.0, 100.0, 10.0, 10.0]]


def test_greedy_placement_improves_slowest_slot_over_mean():
    trace = _skewed_trace()
    naive = naive_placement(6, 3)
    greedy = rebalance(trace, 3, 2)
    assert moe_lb(trace, naive, 3) == 2.5
    assert moe_lb(trace, greedy, 3) == 1.375
    assert moe_lb(trace, greedy, 3) < moe_lb(trace, naive, 3)


def test_placement_keeps_one_expert_per_slot_capacity():
    trace = _skewed_trace()
    placement = rebalance(trace, 3, 2)
    assert sorted(placement) == [0, 0, 1, 1, 2, 2]


def test_ties_prefer_the_lower_slot():
    trace = [[1.0, 1.0, 1.0, 1.0]]
    # Equal loads: the first-best (lower) slot keeps the tie, so the greedy
    # result is the round-robin layout, not a stacked one.
    assert rebalance(trace, 2, 2) == (0, 1, 0, 1)


def test_capacity_shortfall_is_rejected():
    try:
        rebalance([[1.0, 1.0, 1.0]], 1, 1)
    except ValueError:
        return
    raise AssertionError("a shortfall must raise")
