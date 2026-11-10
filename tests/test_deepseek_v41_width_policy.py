# SPDX-License-Identifier: Apache-2.0
"""Unit 2 predicate: the width policy against DeepInfer's Python reference.

The oracle below is a line-by-line port of
``deepinfer/model_executor/layers/tilelang_ops/schedule_mtp_varlen.py:35-98``,
so the comparison is against the reference, not against the implementation.
"""

from __future__ import annotations

from omlx.patches.deepseek_v41.width_policy import (
    BudgetClass,
    Mode,
    WidthSource,
    cumprod_rank_widths,
    policy_row,
    widths_for_row,
)


def reference_widths(
    remap: list[int], prev_rates: list[list[float]], total_q: int, nextn: int
) -> list[int]:
    """Port of compute_seqlens_q_by_confidence_rates_ref."""
    num_seqs = len(remap)
    gathered: list[list[float]] = []
    for source in remap:
        if source == -1:
            gathered.append([0.0] * nextn)
        else:
            gathered.append(list(prev_rates[source]))
    cum_rates: list[list[float]] = []
    for row in gathered:
        product = 1.0
        running: list[float] = []
        for value in row:
            product *= value
            running.append(product)
        cum_rates.append(running)
    extra_budget = total_q - num_seqs
    flat = [value for row in cum_rates for value in row]
    order = sorted(range(len(flat)), key=lambda index: (-flat[index], index))
    ranks = [0] * len(flat)
    for rank, index in enumerate(order):
        ranks[index] = rank
    widths: list[int] = []
    for seq in range(num_seqs):
        wins = sum(
            1
            for position in range(nextn)
            if ranks[seq * nextn + position] < extra_budget
        )
        widths.append(1 + wins)
    return widths


FIXTURES = (
    # ties: identical rates across rows
    ([0, 1], [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]], 6, 3),
    # fresh rows mixed with remapped ones
    ([0, -1, 1], [[0.9, 0.8, 0.7], [0.4, 0.4, 0.4]], 7, 3),
    # a budget that reaches every position
    ([0, 1], [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], 8, 3),
    # a budget that reaches only the base token
    ([0, 1], [[0.3, 0.2, 0.1], [0.6, 0.5, 0.4]], 2, 3),
    # zero rates everywhere
    ([-1, -1], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], 5, 3),
    # nextn 1
    ([0], [[0.7]], 1, 1),
)


def test_policy_equals_reference_on_fixtures():
    for remap, prev_rates, total_q, nextn in FIXTURES:
        got = cumprod_rank_widths(remap, prev_rates, total_q, nextn)
        want = reference_widths(remap, prev_rates, total_q, nextn)
        assert got == want, (remap, prev_rates, total_q, nextn, got, want)


def test_clamped_width_equals_reference_capped():
    remap = [0, 1, 2, 3, 4, 5, 6, 7]
    prev_rates = [[0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99]] * 8
    total_q = 64
    nextn = 8
    reference = reference_widths(remap, prev_rates, total_q, nextn)
    capped = cumprod_rank_widths(remap, prev_rates, total_q, nextn, cap=8)
    assert capped == [min(width, 8) for width in reference]


def test_deterministic_rows_fix_width_across_compositions():
    row = policy_row(Mode.DETERMINISTIC, BudgetClass.BATCH)
    assert row.width_source is WidthSource.FIXED
    assert row.writes_curve is False
    one = widths_for_row(row, configured_depth=5, remap=[0])
    many = widths_for_row(row, configured_depth=5, remap=[0, 1, 2, 3])
    assert one == [5]
    assert many == [5, 5, 5, 5]


def test_turbo_batch_row_uses_confidence_redistribution():
    row = policy_row(Mode.TURBO, BudgetClass.BATCH)
    assert row.width_source is WidthSource.CONFIDENCE
    widths = widths_for_row(
        row,
        configured_depth=5,
        remap=[0, -1],
        prev_rates=[[0.9, 0.8, 0.7]],
        total_q=5,
        nextn=3,
    )
    assert widths == reference_widths([0, -1], [[0.9, 0.8, 0.7]], 5, 3)


def test_widths_never_escape_the_bound():
    for remap, prev_rates, total_q, nextn in FIXTURES:
        widths = cumprod_rank_widths(remap, prev_rates, total_q, nextn)
        assert widths
        assert all(1 <= width <= 8 for width in widths)
