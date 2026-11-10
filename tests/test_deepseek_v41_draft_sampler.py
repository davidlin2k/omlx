# SPDX-License-Identifier: Apache-2.0
"""Unit 8 draft wiring predicate: a seeded request's draft sampler.

The seed travels on the per-request sampler, which is the same mechanism that
already carries temp and top_p. The draft seam reuses it with the draft case
namespace, so drafts are a function of the tuple alone.
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from omlx.patches.deepseek_v41.sampler_counter import (
    CASE_DRAFT,
    CASE_TARGET,
    counter_draw_for,
    seed_for,
)
from omlx.patches.mlx_lm_mtp.batch_generator import _MtpState, _resolve_draft_sampler
from omlx.utils.sampling import make_sampler


def _logits(seed=5):
    mx.random.seed(seed)
    return mx.random.normal((1, 6))


def _batch(seed=None, temp=1.0):
    target = make_sampler(
        temp=temp,
        draw=counter_draw_for(seed, CASE_TARGET) if seed is not None else None,
    )
    if seed is not None:
        target.seed = seed
    return SimpleNamespace(samplers=[target], fallback_sampler=None)


def test_seeded_draft_sampler_is_deterministic_across_instances():
    logits = _logits()
    a = _resolve_draft_sampler(_batch(seed=7), _MtpState(uid=1))
    b = _resolve_draft_sampler(_batch(seed=7), _MtpState(uid=2))
    first = [int(a(logits).tolist()[0]) for _ in range(6)]
    second = [int(b(logits).tolist()[0]) for _ in range(6)]
    assert first == second


def test_draft_sampler_survives_interleaved_rng_use():
    logits = _logits()
    plain = _resolve_draft_sampler(_batch(seed=7), _MtpState(uid=1))
    interleaved = _resolve_draft_sampler(_batch(seed=7), _MtpState(uid=2))
    baseline = [int(plain(logits).tolist()[0]) for _ in range(6)]
    other = []
    for _ in range(6):
        mx.random.categorical(logits)
        other.append(int(interleaved(logits).tolist()[0]))
    assert baseline == other


def test_draft_case_differs_from_target_case():
    assert seed_for(7, 0, CASE_DRAFT) != seed_for(7, 0, CASE_TARGET)


def test_greedy_target_keeps_argmax_drafts():
    logits = _logits()
    batch = _batch(temp=0.0)
    state = _MtpState(uid=1)
    draft = _resolve_draft_sampler(batch, state)
    assert draft(logits).tolist() == [int(np.argmax(np.asarray(logits)))]


def test_unseeded_draft_sampler_keeps_the_rng_path():
    logits = _logits()
    batch = _batch(temp=1.0)
    draft = _resolve_draft_sampler(batch, _MtpState(uid=1))
    mx.random.seed(0)
    pinned = int(draft(logits).tolist()[0])
    mx.random.seed(0)
    assert int(draft(logits).tolist()[0]) == pinned
