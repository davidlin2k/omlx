# SPDX-License-Identifier: Apache-2.0
"""Unit 9 predicate: verify reductions equal the M=1 reference bitwise.

The existing ``decode_consistency`` contract arms a rowwise reduction so a
verify block reduces exactly as decoding one token at a time does. This test
extends it across widths 1 through 8, several row counts, and a ragged set of
widths where each row runs its own block.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from omlx.patches.deepseek_v4 import decode_consistency as dc

DIM, OUT = 32, 16


def _reference(x, call):
    """Per-token M=1 results, reshaped to the batched output's shape."""
    flat = x.reshape(-1, DIM)
    stacked = mx.stack([call(flat[index].reshape(1, DIM)) for index in range(flat.shape[0])])
    return stacked.reshape(*x.shape[:-1], stacked.shape[-1])


def test_matmul_armed_equals_per_token_m1_bitwise():
    weight_t = mx.random.normal((DIM, OUT))
    for width in range(1, 9):
        for rows in (1, 2, 4):
            x = mx.random.normal((rows, width, DIM))
            reference = _reference(x, lambda row: mx.matmul(row, weight_t))
            dc.set_armed(True)
            armed = dc.matmul(x, weight_t)
            dc.set_armed(False)
            np.testing.assert_array_equal(np.asarray(armed), np.asarray(reference))


def test_armed_linear_equals_per_token_m1_bitwise():
    assert dc.apply()
    layer = nn.Linear(DIM, OUT)
    for width in range(1, 9):
        for rows in (1, 2, 4):
            x = mx.random.normal((rows, width, DIM))
            reference = _reference(x, lambda row: layer(row))
            dc.set_armed(True)
            armed = layer(x)
            dc.set_armed(False)
            np.testing.assert_array_equal(np.asarray(armed), np.asarray(reference))


def test_ragged_widths_each_equal_their_own_m1():
    assert dc.apply()
    layer = nn.Linear(DIM, OUT)
    for width in (1, 3, 8):
        x = mx.random.normal((1, width, DIM))
        reference = _reference(x, lambda row: layer(row))
        dc.set_armed(True)
        armed = layer(x)
        dc.set_armed(False)
        np.testing.assert_array_equal(np.asarray(armed), np.asarray(reference))


def test_disarmed_batched_differs_so_the_test_is_sensitive():
    # A batched reduction is not the M=1 order in general; this proves the
    # equality above is caused by arming, not by a degenerate fixture.
    assert dc.apply()
    layer = nn.Linear(DIM, OUT)
    x = mx.random.normal((2, 4, DIM))
    reference = _reference(x, lambda row: layer(row))
    dc.set_armed(False)
    unarmed = layer(x)
    assert not np.array_equal(np.asarray(unarmed), np.asarray(reference))


def test_arming_toggles():
    dc.set_armed(False)
    assert dc.is_armed() is False
    dc.set_armed(True)
    assert dc.is_armed() is True
    dc.set_armed(False)
    assert dc.is_armed() is False
