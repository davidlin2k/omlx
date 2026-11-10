# SPDX-License-Identifier: Apache-2.0
"""Unit 4 predicate: confidence production on the runtime DSpark path.

The oracle is the published ``forward_spec`` path. The runtime helper must
produce the same raw projection from the same hidden and the same Markov
embeddings, and the transform must be ``sigmoid(clamp(z / temperature))``.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
from test_deepseek_v41 import load_reference_weights, tiny

from omlx.patches.deepseek_v41.dspark import confidence_logits, confidence_rates
from omlx.patches.deepseek_v41.language import LanguageModel


def mtp_config(**kwargs):
    return tiny(
        preserve_mtp=True,
        n_mtp_layers=3,
        dspark_block_size=3,
        dspark_noise_token_id=2,
        dspark_target_layer_ids=(2, 3, 4),
        dspark_n_routed_experts=2,
        dspark_n_activated_experts=1,
        dspark_markov_rank=32,
        compress_ratios=(0, 2, 2, 1, 1, 0, 0, 0),
        **kwargs,
    )


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def test_runtime_confidence_matches_forward_spec():
    model = LanguageModel(mtp_config())
    load_reference_weights(model)
    model.configure_mtp(True, 3)

    target = model.make_cache()
    draft_spec = model.make_dspark_cache()
    draft_runtime = model.make_dspark_cache()

    _, hidden0 = model(
        mx.array([[3, 4, 5, 6, 7]]), cache=target, return_dspark_hidden=True
    )
    assert model.forward_spec(mx.array([[10]]), hidden0, draft_spec) is None
    assert model.forward_spec(mx.array([[10]]), hidden0, draft_runtime) is None

    _, hidden1 = model(mx.array([[8]]), cache=target, return_dspark_hidden=True)
    published = model.forward_spec(mx.array([[11]]), hidden1, draft_spec)
    assert published is not None
    proposed, _biases, published_confidence = published

    logits, runtime_hidden = model.dspark_forward(
        hidden1, mx.array([[11]]), draft_runtime, draft_length=3
    )
    assert logits.shape[0] == 1

    # The Markov embeddings follow the published proposals, in the same order.
    # Tokens are passed as 1-D rows, exactly as the runtime draft loop does.
    depth = int(proposed.shape[1]) - 1
    embeddings = []
    previous = proposed[:, 0]
    for index in range(depth):
        _, embedding = model.dspark_markov(previous)
        embeddings.append(embedding)
        previous = proposed[:, index + 1]

    raw = model.dspark_confidence_raw(runtime_hidden, embeddings)
    np.testing.assert_array_equal(
        np.asarray(raw), np.asarray(published_confidence)
    )


def test_transform_is_sigmoid_clamped_temperature():
    head_hidden = mx.random.normal((1, 3, 32))
    embeddings = [mx.random.normal((1, 32)) for _ in range(3)]
    head = _FakeHead(mx.random.normal((1, 64)))
    raw = confidence_logits(head, head_hidden, embeddings)
    for temperature in (1.0, 0.5, 2.0):
        got = np.asarray(
            confidence_rates(head, head_hidden, embeddings, temperature=temperature)
        )
        want = _sigmoid(np.clip(np.asarray(raw) / temperature, -60.0, 60.0))
        np.testing.assert_allclose(got, want, atol=1e-6, rtol=1e-6)


def test_missing_head_returns_none():
    head_hidden = mx.zeros((1, 3, 4))
    embeddings = [mx.zeros((1, 2)) for _ in range(3)]
    assert confidence_logits(None, head_hidden, embeddings) is None
    assert confidence_rates(None, head_hidden, embeddings) is None


class _FakeHead:
    def __init__(self, weight):
        self.proj = type("P", (), {"weight": weight})()
