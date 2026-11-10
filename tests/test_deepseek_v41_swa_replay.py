# SPDX-License-Identifier: Apache-2.0
"""Unit 11 predicate: the encoder-side SWA bounded-replay hit policy."""

from __future__ import annotations

from omlx.patches.deepseek_v41.swa_replay import (
    hit_plan,
    replay_cap,
    replay_needs_swa,
    swa_visibility,
)


def test_replay_cap_is_l_minus_window_and_backtracking():
    assert replay_cap(1024) == 1024 - 128 - 6
    assert replay_cap(134) == 0
    assert replay_cap(100) == 0


def test_hit_plan_keeps_the_block_hit_when_swa_is_present():
    # L1 = 768, cap = 890 -> the block hit stands.
    assert hit_plan(1024, 768, True) == 768


def test_hit_plan_backs_off_one_window_when_swa_misses():
    assert hit_plan(1024, 768, False) == 640


def test_hit_plan_clamps_to_the_replay_cap():
    # data 200, block hit 128, cap = 66 -> the chosen hit is clamped.
    assert hit_plan(200, 128, True) == 66
    assert hit_plan(200, 128, False) == 0


def test_hit_plan_is_zero_for_empty_inputs():
    assert hit_plan(0, 768, True) == 0
    assert hit_plan(1024, 0, True) == 0


def test_replay_need_tracks_the_window():
    assert replay_needs_swa(1024, 768) is True
    assert replay_needs_swa(1024, 0) is False


def test_visibility_is_the_max_of_window_rectangle_and_valid_start():
    positions = [0, 10, 200]
    got = swa_visibility(positions, valid_start=50, rect_lo=120, window=128)
    # pos 0 -> window start -127 clipped to 0; pos 10 -> -117 clipped; pos 200 -> 73
    # then each is raised to max(rect_lo, valid_start).
    assert got == [120, 120, 120]
    got = swa_visibility([200, 300], valid_start=0, rect_lo=0, window=128)
    assert got == [73, 173]
