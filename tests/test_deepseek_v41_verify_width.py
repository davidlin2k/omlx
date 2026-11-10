# SPDX-License-Identifier: Apache-2.0
"""Unit 5 predicate: the verification-width scheduler."""

from __future__ import annotations

import mlx.core as mx

from omlx.patches.deepseek_v41.mtp import VerifyWidthScheduler
from omlx.patches.deepseek_v41.perf_curve import DecodeCurve
from omlx.patches.deepseek_v41.width_policy import Mode


def _curve(costs):
    curve = DecodeCurve(key="key-a")
    for width, cost in costs.items():
        curve.observe(width, 1, cost)
    return curve


def test_acceptance_fallback_without_curve_or_confidence():
    scheduler = VerifyWidthScheduler(3)
    widths = []
    for accepted in [0, 1, 2, 3, 1]:
        scheduler.observe(scheduler.cur, accepted, 0.01)
        widths.append(scheduler.width_for(None))
    assert widths == [1, 2, 3, 3, 2]


def test_acceptance_fallback_without_confidence_even_with_curve():
    scheduler = VerifyWidthScheduler(3, curve=_curve({1: 1.0, 4: 1.0}))
    scheduler.observe(3, 2, 0.01)
    assert scheduler.width_for(None) == 3


def test_curve_and_confidence_choose_the_best_ratio():
    scheduler = VerifyWidthScheduler(3, curve=_curve({1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}))
    confidence = mx.array([0.9, 0.9, 0.9])
    # expected = 1, 1.9, 2.71, 3.439 over equal costs -> width 4 wins.
    assert scheduler.width_for(confidence) == 4

    scheduler = VerifyWidthScheduler(
        3, curve=_curve({1: 1.0, 2: 1.0, 3: 1.0, 4: 100.0})
    )
    # the 100 ms shape loses, so width 3 wins.
    assert scheduler.width_for(confidence) == 3


def test_deterministic_mode_fixes_the_width():
    scheduler = VerifyWidthScheduler(
        3, curve=_curve({1: 1.0, 4: 1.0}), mode=Mode.DETERMINISTIC
    )
    confidence = mx.array([0.99, 0.99, 0.99])
    assert scheduler.width_for(confidence) == 3
    scheduler.observe(3, 0, 0.01)
    assert scheduler.width_for(confidence) == 1


def test_width_never_escapes_the_bound():
    scheduler = VerifyWidthScheduler(8, curve=_curve({w: 1.0 for w in range(1, 9)}))
    confidence = mx.array([0.99] * 8)
    width = scheduler.width_for(confidence)
    assert 1 <= width <= 8


def test_scheduler_is_timing_independent():
    a = VerifyWidthScheduler(3)
    b = VerifyWidthScheduler(3)
    for accepted in [0, 1, 2, 3]:
        a.observe(a.cur, accepted, 0.01)
        b.observe(b.cur, accepted, 100000.0, time_sample=False)
        assert a.cur == b.cur
        assert not a.should_exit() and not b.should_exit()
