# SPDX-License-Identifier: Apache-2.0
"""Unit 19 predicate: paged KV index conversion."""

from __future__ import annotations

from omlx.patches.deepseek_v41.paged_index import (
    contiguous_index,
    convert_kv_index,
    physical_slot_rows,
    ring_index,
)


def test_logical_to_physical_slot():
    # block size 4, blocks placed at physical blocks 7 and 8.
    got = convert_kv_index([0, 5, 7, -1], [7, 8], 4)
    assert got == [28, 33, 35, -1]


def test_block_table_overrun_maps_to_padding():
    assert convert_kv_index([9], [7, 8], 4) == [-1]


def test_contiguous_range():
    assert contiguous_index(3, 4) == [3, 4, 5, 6]
    assert contiguous_index(3, 0) == []


def test_ring_expansion_over_a_window():
    assert ring_index(0, 4, 4) == [0, 1, 2, 3]
    assert ring_index(2, 4, 4) == [2, 3, 0, 1]
    assert ring_index(-1, 2, 4) == [-1, 0]


def test_rows_pad_to_the_requested_count():
    assert physical_slot_rows([0, 5], [7, 8], 4, rows=3) == [28, 33, -1]
    assert physical_slot_rows([0], [7], 4, rows=1) == [28]


def test_invalid_inputs_raise():
    for call in (
        lambda: convert_kv_index([0], [7], 0),
        lambda: physical_slot_rows([0], [7], 4, rows=0),
        lambda: ring_index(0, 1, 0),
    ):
        try:
            call()
        except ValueError:
            continue
        raise AssertionError("invalid input must raise")
