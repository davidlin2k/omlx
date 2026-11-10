# SPDX-License-Identifier: Apache-2.0
"""Unit 13 predicate: the fitted prefill cost model and its budget solver."""

from __future__ import annotations

from omlx.patches.deepseek_v41.prefill_cost import PrefillCost, fit, solve_budget


def _synthetic_samples():
    cost = PrefillCost(a=1e-7, b=0.01, c=1.0, d=0.0001)
    samples = []
    for q in (1, 512, 2048, 4096):
        for prefix in (0, 1024):
            samples.append((q, prefix, cost.ms(q, prefix)))
    return cost, samples


def test_fit_recovers_the_synthetic_model():
    cost, samples = _synthetic_samples()
    fitted = fit(samples)
    assert abs(fitted.a - cost.a) < 1e-9
    assert abs(fitted.b - cost.b) < 1e-7
    assert abs(fitted.c - cost.c) < 1e-4
    assert abs(fitted.d - cost.d) < 1e-9


def test_solver_picks_the_largest_query_inside_the_budget():
    cost = PrefillCost(a=1e-7, b=0.01, c=1.0, d=0.0001)
    # cost(1000, prefix 0) = 0.1 + 10 + 1 = 11.1 ms.
    assert solve_budget(cost, prefix_len=0, budget_ms=11.1, cap=4096) == 1000
    assert solve_budget(cost, prefix_len=0, budget_ms=1.0, cap=4096) == 0
    assert solve_budget(cost, prefix_len=0, budget_ms=1e9, cap=4096) == 4096


def test_solver_honours_the_prefix_term():
    cost = PrefillCost(a=1e-7, b=0.01, c=1.0, d=0.0001)
    # A 1024-token prefix costs 0.1024 ms more, so q drops.
    with_prefix = solve_budget(cost, prefix_len=1024, budget_ms=11.1, cap=4096)
    without = solve_budget(cost, prefix_len=0, budget_ms=11.1, cap=4096)
    assert with_prefix <= without


def test_linear_and_degenerate_models():
    linear = PrefillCost(a=0.0, b=0.01, c=1.0, d=0.0)
    assert solve_budget(linear, 0, 11.0, 4096) == 1000
    degenerate = PrefillCost(a=0.0, b=0.0, c=1.0, d=0.0)
    assert solve_budget(degenerate, 0, 1.0, 4096) == 4096
    assert solve_budget(degenerate, 0, 0.5, 4096) == 0
