# SPDX-License-Identifier: Apache-2.0
"""Unit 12 predicate: SWA state persistence at 128-token granularity."""

from __future__ import annotations

from omlx.patches.deepseek_v41.swa_state import (
    MAX_COPY_OVERSHOOT,
    STORE_GRAIN,
    copy_position,
    crossing_boundaries,
    overshoot_absorbed,
    should_store,
    store_position,
    store_units,
)


def test_store_position_reserves_one_token_and_rounds_down():
    assert store_position(129) == 128
    assert store_position(256) == 128
    assert store_position(257) == 256
    assert store_position(384) == 256


def test_less_than_one_full_unit_stores_nothing():
    # Reserving one token at 128 leaves 127, below one unit.
    assert store_position(128) == 0
    assert should_store(128) is False
    assert should_store(129) is True


def test_store_units_counts_full_grain_units():
    assert store_units(129) == 1
    assert store_units(256) == 1
    assert store_units(257) == 2
    assert store_units(1024) == 7


def test_crossing_boundaries_lists_grain_multiples():
    assert crossing_boundaries(0, 130) == [128]
    assert crossing_boundaries(100, 300) == [128, 256]
    assert crossing_boundaries(128, 128) == []


def test_copy_position_rounds_down_to_the_grain():
    assert copy_position(130) == 128
    assert copy_position(128) == 128
    assert copy_position(255) == 128


def test_overshoot_inside_the_ring_extra_positions():
    assert overshoot_absorbed(128 + MAX_COPY_OVERSHOOT) is True
    assert overshoot_absorbed(128 + MAX_COPY_OVERSHOOT + 1) is False
    assert overshoot_absorbed(128) is True
