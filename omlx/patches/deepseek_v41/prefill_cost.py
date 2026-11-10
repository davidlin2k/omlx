# SPDX-License-Identifier: Apache-2.0
"""Fitted prefill cost model and the budget solver.

DeepInfer's prefill scheduler gives each step a time budget and a fitted
quadratic ``cost(q, c)`` over the query count ``q`` and the already-computed
prefix length ``c``, then solves for the largest ``q`` that stays inside the
budget. This module is that model and solver, without the runtime.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PrefillCost:
    a: float
    b: float
    c: float
    d: float

    def ms(self, q: int, prefix_len: int = 0) -> float:
        return self.a * q * q + self.b * q + self.c + self.d * prefix_len


def fit(samples) -> PrefillCost:
    """Least-squares fit over samples of ``(q, prefix_len, ms)``."""
    if len(samples) < 4:
        raise ValueError("a quadratic with a prefix term needs four samples")
    design = np.array(
        [[q * q, q, 1.0, prefix_len] for q, prefix_len, _ in samples], dtype=float
    )
    target = np.array([ms for _, _, ms in samples], dtype=float)
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    return PrefillCost(*(float(value) for value in coef))


def solve_budget(cost: PrefillCost, prefix_len: int, budget_ms: float, cap: int) -> int:
    """Largest ``q <= cap`` whose predicted cost stays inside the budget."""
    if cap < 0:
        raise ValueError("cap must be non-negative")
    constant = cost.c + cost.d * prefix_len - budget_ms
    if cost.a > 0:
        disc = cost.b * cost.b - 4 * cost.a * constant
        if disc < 0:
            return 0
        root = (-cost.b + math.sqrt(disc)) / (2 * cost.a)
    elif cost.b > 0:
        root = -constant / cost.b
    else:
        return 0 if constant > 0 else cap
    if root < 0:
        return 0
    return int(max(0, min(cap, math.floor(root))))
