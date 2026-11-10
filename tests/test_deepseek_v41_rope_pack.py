# SPDX-License-Identifier: Apache-2.0
"""Unit 25 predicate: fused rope plus pack equals the sequential pair bitwise.

The oracle is `fused_pack_activation(rope_activation(x, cos, sin))`, the pair the
fused kernel replaces.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from omlx.patches.deepseek_v41.activation import rope_activation
from omlx.patches.deepseek_v41.packed_kv import fused_pack_activation, fused_rope_pack


def _fixture(batch=1, length=3, dim=128, rope_dim=64, dtype=mx.bfloat16):
    mx.random.seed(9)
    x = mx.random.normal((batch, length, dim)).astype(dtype)
    cos = mx.random.normal((length, rope_dim // 2)).astype(mx.float32)
    sin = mx.random.normal((length, rope_dim // 2)).astype(mx.float32)
    return x, cos, sin


def test_fused_rope_pack_equals_the_pair_bitwise():
    x, cos, sin = _fixture()
    for bits, e4m3_scale in ((8, False), (8, True), (4, False), (4, True)):
        oracle = fused_pack_activation(
            rope_activation(x, cos, sin), bits=bits, e4m3_scale=e4m3_scale
        )
        fused = fused_rope_pack(x, cos, sin, bits=bits, e4m3_scale=e4m3_scale)
        np.testing.assert_array_equal(np.asarray(fused), np.asarray(oracle))


def test_fused_rope_pack_shape_and_dtype():
    x, cos, sin = _fixture()
    fused = fused_rope_pack(x, cos, sin, bits=8)
    assert fused.shape == (*x.shape[:-1], x.shape[-1] + x.shape[-1] // 32)
    assert fused.dtype == mx.uint8
    mx.eval(fused)


def test_fused_rope_pack_validates_inputs():
    x, cos, sin = _fixture()
    for call in (
        lambda: fused_rope_pack(x, cos[:1], sin[:1]),
        lambda: fused_rope_pack(x, cos, sin, bits=8, group_size=33),
    ):
        try:
            call()
        except (ValueError, ZeroDivisionError):
            continue
        raise AssertionError("invalid input must raise")


def test_fused_rope_pack_matches_with_a_larger_row():
    x, cos, sin = _fixture(batch=2, length=5, dim=256, rope_dim=64)
    oracle = fused_pack_activation(rope_activation(x, cos, sin), bits=8)
    fused = fused_rope_pack(x, cos, sin, bits=8)
    np.testing.assert_array_equal(np.asarray(fused), np.asarray(oracle))


def test_fused_rope_pack_matches_the_group_16_shape():
    # dim 32 with an 8-wide rope is the indexer and compressed-latent geometry;
    # a 16-element group spans two lanes, not four.
    x, cos, sin = _fixture(batch=1, length=4, dim=32, rope_dim=8)
    for bits, group_size, e4m3_scale in ((4, 16, True), (4, 32, False)):
        oracle = fused_pack_activation(
            rope_activation(x, cos, sin),
            bits=bits,
            group_size=group_size,
            e4m3_scale=e4m3_scale,
        )
        fused = fused_rope_pack(
            x, cos, sin, bits=bits, group_size=group_size, e4m3_scale=e4m3_scale
        )
        np.testing.assert_array_equal(np.asarray(fused), np.asarray(oracle))
