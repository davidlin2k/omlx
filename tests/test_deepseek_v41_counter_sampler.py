# SPDX-License-Identifier: Apache-2.0
"""Unit 8 predicate: a seeded sampler draws without RNG state or composition.

The sampler factory takes an optional ``draw`` callable. With a seed, the V4.1
counter draw supplies it, so a request's tokens are a function of the tuple
``(seq_seed, position, case)`` alone.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from omlx.patches.deepseek_v41.sampler_counter import (
    CASE_DRAFT,
    CASE_TARGET,
    CounterDraw,
    counter_draw_for,
)
from omlx.utils.sampling import make_sampler


def _logits(rows=1, vocab=6, seed=3):
    mx.random.seed(seed)
    return mx.random.normal((rows, vocab))


def test_seeded_samplers_are_deterministic_across_instances():
    logits = _logits()
    a = make_sampler(temp=1.0, draw=counter_draw_for(1234))
    b = make_sampler(temp=1.0, draw=counter_draw_for(1234))
    first = [int(a(logits).tolist()[0]) for _ in range(6)]
    second = [int(b(logits).tolist()[0]) for _ in range(6)]
    assert first == second


def test_seeded_draws_survive_interleaved_rng_use():
    logits = _logits()
    plain = make_sampler(temp=1.0, draw=counter_draw_for(1234))
    interleaved = make_sampler(temp=1.0, draw=counter_draw_for(1234))
    baseline = [int(plain(logits).tolist()[0]) for _ in range(6)]
    other = []
    for _ in range(6):
        mx.random.categorical(logits)  # unrelated draws consume RNG state
        other.append(int(interleaved(logits).tolist()[0]))
    assert baseline == other


def test_unseeded_samplers_keep_the_rng_path():
    logits = _logits()
    sampler = make_sampler(temp=1.0)
    first = int(sampler(logits).tolist()[0])
    # The unseeded path reads RNG state, so re-seeding to a known state pins it.
    mx.random.seed(0)
    pinned = int(sampler(logits).tolist()[0])
    mx.random.seed(0)
    assert int(sampler(logits).tolist()[0]) == pinned
    assert first in range(logits.shape[-1])


def test_seeded_rowwise_call_matches_across_instances():
    logits = _logits(rows=3)
    a = make_sampler(temp=1.0, draw=counter_draw_for(99))
    b = make_sampler(temp=1.0, draw=counter_draw_for(99))
    np.testing.assert_array_equal(np.asarray(a(logits)), np.asarray(b(logits)))


def test_draft_and_target_seeds_differ():
    from omlx.patches.deepseek_v41.sampler_counter import seed_for

    assert seed_for(77, 0, CASE_DRAFT) != seed_for(77, 0, CASE_TARGET)


def test_greedy_path_needs_no_draw():
    logits = _logits()
    sampler = make_sampler(temp=0.0, draw=counter_draw_for(5))
    got = sampler(logits).tolist()
    assert got == [int(np.argmax(np.asarray(logits)))]


def test_counter_draw_requires_two_dimensional_logits():
    try:
        CounterDraw(1)(mx.zeros((4,)), 1.0)
    except ValueError:
        return
    raise AssertionError("one-dimensional logits must raise")
