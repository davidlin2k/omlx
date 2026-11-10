# SPDX-License-Identifier: Apache-2.0
"""Unit 18 predicate: the context-parallel mirror split."""

from __future__ import annotations

from omlx.patches.deepseek_v41.cp_mirror import (
    chunk_size,
    mirror_blocks,
    mirror_row_count,
    mirror_rows,
)


def test_chunk_size_is_half_of_the_two_way_split():
    assert chunk_size(10, 2) == 3
    assert chunk_size(8, 2) == 2
    assert chunk_size(1, 4) == 1


def test_blocks_pair_head_with_tail():
    assert mirror_blocks(2) == ((0, 3), (1, 2))
    assert mirror_blocks(4) == ((0, 7), (1, 6), (2, 5), (3, 4))


def test_rows_are_equal_across_ranks():
    rows = mirror_rows(10, 2)
    assert rows[0] == (0, 1, 2, 9, -1, -1)
    assert rows[1] == (3, 4, 5, 6, 7, 8)
    assert all(len(entry) == mirror_row_count(10, 2) for entry in rows)
    assert mirror_row_count(10, 2) == 6


def test_padding_marks_overrun_blocks():
    rows = mirror_rows(7, 2)
    # chunk 2, so blocks 0..3 cover 0..7; block 3 overruns by one token.
    assert rows[0] == (0, 1, 6, -1)
    assert rows[1] == (2, 3, 4, 5)


def test_empty_and_invalid_inputs_raise():
    for call in (lambda: chunk_size(0, 2), lambda: mirror_blocks(0)):
        try:
            call()
        except ValueError:
            continue
        raise AssertionError("invalid input must raise")
