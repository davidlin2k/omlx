# SPDX-License-Identifier: Apache-2.0
"""Unit 7 predicate: counter-based deterministic draws.

The CPU reference in the module is the oracle, so the Metal kernels are checked
against arithmetic, not against themselves.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from omlx.patches.deepseek_v41.sampler_counter import (
    CASE_DRAFT,
    CASE_TARGET,
    CASE_VERIFY,
    cdf_tokens,
    counter_categorical,
    counter_categorical_fused,
    draw_u,
    draw_u_mx,
    seed_for,
    seeds_mx,
    splitmix64,
)


def test_splitmix_matches_published_constants():
    # SplitMix64's first outputs are a well-known sequence; checking the first
    # value pins the constants and the shifts.
    assert splitmix64(0) == 0xE220A8397B1DCDAF


def test_draw_is_a_function_of_the_tuple_only():
    tuples = [(0, 0, CASE_TARGET), (0, 0, CASE_DRAFT), (0, 0, CASE_VERIFY),
              (7, 3, CASE_TARGET), (7, 3, CASE_DRAFT), (2**63 - 1, 11, CASE_VERIFY)]
    for seq_seed, position, case in tuples:
        assert draw_u(seq_seed, position, case) == draw_u(seq_seed, position, case)
        assert 0.0 <= draw_u(seq_seed, position, case) < 1.0


def test_draft_and_target_use_distinct_seeds():
    assert seed_for(5, 2, CASE_DRAFT) != seed_for(5, 2, CASE_TARGET)
    assert seed_for(5, 2, CASE_VERIFY) != seed_for(5, 2, CASE_TARGET)


def test_metal_kernel_matches_the_reference_within_fp32():
    tuples = [
        (0, 0, CASE_TARGET), (1, 0, CASE_TARGET), (0, 1, CASE_TARGET),
        (7, 3, CASE_DRAFT), (7, 3, CASE_VERIFY), (123456789, 5, CASE_DRAFT),
        (2**63 - 1, 8, CASE_TARGET), (42, 63, CASE_VERIFY),
    ]
    seeds = [seed_for(*t) for t in tuples]
    got = draw_u_mx(seeds_mx(seeds)).tolist()
    want = [draw_u(*t) for t in tuples]
    # The kernel computes the fraction in fp32 while the reference uses Python
    # floats, so the values agree to fp32 rounding, not bit for bit.
    np.testing.assert_allclose(got, want, atol=1e-7, rtol=0)


def _reference_token(probs, u):
    cum = 0.0
    for index, prob in enumerate(probs):
        cum += float(prob)
        if u <= cum:
            return index
    return len(probs) - 1


def test_metal_kernel_selects_the_same_tokens_as_the_reference():
    probs = np.array([0.1, 0.2, 0.3, 0.4], np.float32)
    tuples = [(0, 0, CASE_TARGET), (0, 1, CASE_TARGET), (7, 3, CASE_DRAFT),
              (123456789, 5, CASE_VERIFY), (42, 63, CASE_VERIFY)]
    draws = draw_u_mx(seeds_mx([seed_for(*t) for t in tuples]))
    rows = mx.broadcast_to(mx.array(probs), (len(tuples), 4))
    got = cdf_tokens(draws, rows).tolist()
    want = [_reference_token(probs, draw_u(*t)) for t in tuples]
    assert got == want


def test_cdf_inversion_equals_the_reference():
    probs = np.array(
        [
            [0.25, 0.25, 0.25, 0.25],
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3, 0.4],
        ],
        np.float32,
    )
    draws = np.array([0.0, 0.34, 0.99], np.float32)
    got = cdf_tokens(mx.array(draws), mx.array(probs)).tolist()
    assert got == [0, 2, 3]


def test_categorical_is_composition_independent():
    row_a = mx.array([[0.1, 0.2, 0.3, 0.4]], mx.float32)
    row_b = mx.array([[0.4, 0.3, 0.2, 0.1]], mx.float32)
    solo_a = counter_categorical(row_a, seq_seed=99, sample_positions=mx.array([0]))
    solo_b = counter_categorical(row_b, seq_seed=99, sample_positions=mx.array([1]))
    together = counter_categorical(
        mx.concatenate([row_a, row_b]),
        seq_seed=99,
        sample_positions=mx.array([0, 1]),
    )
    assert solo_a.tolist()[0] == together.tolist()[0]
    assert solo_b.tolist()[0] == together.tolist()[1]


def test_fused_kernel_equals_the_two_dispatch_oracle():
    probs = mx.array(
        [
            [0.25, 0.25, 0.25, 0.25],
            [0.1, 0.2, 0.3, 0.4],
            [0.4, 0.3, 0.2, 0.1],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        mx.float32,
    )
    positions = mx.array([0, 1, 2, 3, 4], mx.int32)
    seeds = seeds_mx(seed_for(4242, position, CASE_TARGET) for position in positions.tolist())
    oracle = cdf_tokens(draw_u_mx(seeds), probs)
    fused = counter_categorical_fused(
        probs, seq_seed=4242, sample_positions=positions, case=CASE_TARGET
    )
    np.testing.assert_array_equal(np.asarray(oracle), np.asarray(fused))


def test_counter_categorical_uses_the_fused_path():
    probs = mx.array([[0.1, 0.2, 0.3, 0.4], [0.7, 0.1, 0.1, 0.1]], mx.float32)
    positions = mx.array([0, 1], mx.int32)
    a = counter_categorical(probs, seq_seed=11, sample_positions=positions)
    b = counter_categorical_fused(probs, seq_seed=11, sample_positions=positions)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
