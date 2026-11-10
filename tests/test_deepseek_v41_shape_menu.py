# SPDX-License-Identifier: Apache-2.0
"""Unit 21 predicate: the fixed shape menu for decode submissions."""

from __future__ import annotations

from omlx.patches.deepseek_v41.shape_menu import graph_sizes, menu_size, snap_to_menu


def test_menu_is_largest_first():
    assert graph_sizes(64) == [64, 56, 48, 40, 32, 24, 16, 8, 4, 2, 1]


def test_menu_holds_powers_of_two_below_the_interval():
    assert graph_sizes(1) == [1]
    assert graph_sizes(5) == [5, 4, 2, 1]
    assert graph_sizes(7) == [7, 4, 2, 1]
    assert graph_sizes(8) == [8, 4, 2, 1]


def test_menu_steps_by_the_interval_to_the_maximum():
    assert graph_sizes(32, interval=16) == [32, 16, 8, 4, 2, 1]
    assert graph_sizes(24, interval=8) == [24, 16, 8, 4, 2, 1]


def test_snap_picks_the_smallest_entry_at_or_above():
    menu = graph_sizes(64)
    assert snap_to_menu(1, menu) == 1
    assert snap_to_menu(5, menu) == 8
    assert snap_to_menu(50, menu) == 56
    assert snap_to_menu(65, menu) == 64


def test_menu_size_counts_distinct_shapes():
    assert menu_size(graph_sizes(64)) == 11


def test_invalid_inputs_raise():
    for call in (lambda: graph_sizes(0), lambda: graph_sizes(8, interval=0),
                 lambda: snap_to_menu(1, [])):
        try:
            call()
        except ValueError:
            continue
        raise AssertionError("invalid input must raise")
