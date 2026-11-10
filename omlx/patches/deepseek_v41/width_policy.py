# SPDX-License-Identifier: Apache-2.0
"""Verification-width policy for DeepSeek V4.1 DSpark cycles.

The domain here is one small decision table plus one redistribution rule. A
width is the number of verify tokens on one request row, bounded by
``decode_fused.MAX_BATCH``. The table selects the width source by mode and by
whether the cycle verifies one row or several, so the deterministic mode is a
row in the same table instead of a runtime switch that must stay in sync with
the turbo path.

The redistribution reproduces DeepInfer's
``compute_seqlens_q_by_confidence_rates``: cumulative product of the previous
step's confidence rates through a remap, then a stable descending rank over all
(sequence, position) pairs, and each sequence wins one base token plus every
position whose rank falls inside the extra budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

MAX_VERIFY_WIDTH = 8  # decode_fused.MAX_BATCH


class Mode(Enum):
    TURBO = "turbo"
    DETERMINISTIC = "deterministic"


class BudgetClass(Enum):
    SINGLE = "single"
    BATCH = "batch"


class WidthSource(Enum):
    FIXED = "fixed"
    CURVE = "curve"
    CONFIDENCE = "confidence"


class Redistribute(Enum):
    NONE = "none"
    CUMPROD_RANK = "cumprod_rank"


class CurveKey(Enum):
    NONE = "none"
    WIDTH = "width"
    WIDTH_ROWS = "width_rows"


@dataclass(frozen=True)
class PolicyRow:
    width_source: WidthSource
    redistribute: Redistribute
    floor: int
    cap: int
    curve_key: CurveKey
    writes_curve: bool


VERIFY_POLICY: dict[tuple[Mode, BudgetClass], PolicyRow] = {
    (Mode.TURBO, BudgetClass.SINGLE): PolicyRow(
        WidthSource.CURVE, Redistribute.NONE, 1, MAX_VERIFY_WIDTH, CurveKey.WIDTH, True
    ),
    (Mode.TURBO, BudgetClass.BATCH): PolicyRow(
        WidthSource.CONFIDENCE,
        Redistribute.CUMPROD_RANK,
        1,
        MAX_VERIFY_WIDTH,
        CurveKey.WIDTH_ROWS,
        True,
    ),
    (Mode.DETERMINISTIC, BudgetClass.SINGLE): PolicyRow(
        WidthSource.FIXED, Redistribute.NONE, 1, MAX_VERIFY_WIDTH, CurveKey.NONE, False
    ),
    (Mode.DETERMINISTIC, BudgetClass.BATCH): PolicyRow(
        WidthSource.FIXED, Redistribute.NONE, 1, MAX_VERIFY_WIDTH, CurveKey.NONE, False
    ),
}


def policy_row(mode: Mode, budget_class: BudgetClass) -> PolicyRow:
    return VERIFY_POLICY[(mode, budget_class)]


def cum_rates(
    remap: list[int], prev_rates: list[list[float]], nextn: int
) -> list[list[float]]:
    """Cumulative product of the previous rates through the remap.

    A fresh row (remap -1) has zero rates everywhere, so its cumulative product
    stays zero and it ranks last. This matches the reference, which fills a
    safe index then overrides the row with zeros.
    """
    out: list[list[float]] = []
    for source in remap:
        row = [0.0] * nextn if source == -1 else list(prev_rates[source])
        product = 1.0
        running: list[float] = []
        for value in row:
            product *= value
            running.append(product)
        out.append(running)
    return out


def cumprod_rank_widths(
    remap: list[int],
    prev_rates: list[list[float]],
    total_q: int,
    nextn: int,
    *,
    cap: int = MAX_VERIFY_WIDTH,
) -> list[int]:
    """Per-row widths from the extra budget and the ranked cumulative rates."""
    num_seqs = len(remap)
    cumulative = cum_rates(remap, prev_rates, nextn)
    extra_budget = total_q - num_seqs
    flat = [value for row in cumulative for value in row]
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
        widths.append(min(1 + wins, cap))
    return widths


def widths_for_row(
    row: PolicyRow,
    *,
    configured_depth: int,
    curve_width: int | None = None,
    remap: list[int] | None = None,
    prev_rates: list[list[float]] | None = None,
    total_q: int | None = None,
    nextn: int | None = None,
) -> list[int]:
    """Apply one policy row and return per-request widths."""
    if row.width_source is WidthSource.FIXED:
        count = max(row.floor, min(int(configured_depth), row.cap))
        rows = 1 if remap is None else len(remap)
        return [count] * rows
    if row.width_source is WidthSource.CURVE:
        count = max(row.floor, min(int(curve_width or configured_depth), row.cap))
        return [count]
    if remap is None or prev_rates is None or total_q is None or nextn is None:
        raise ValueError("confidence redistribution needs remap, rates, budget, nextn")
    widths = cumprod_rank_widths(remap, prev_rates, total_q, nextn, cap=row.cap)
    return [max(row.floor, width) for width in widths]
