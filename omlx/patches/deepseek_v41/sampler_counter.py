# SPDX-License-Identifier: Apache-2.0
"""Counter-based deterministic draws for the DeepSeek V4.1 decode path.

A draw is a function of ``(seq_seed, sample_position, case)`` only, so two batch
compositions produce identical draws. This mirrors DeepInfer's SplitMix64 hash
plus CDF inversion, which is deterministic in the same tuple.

The CPU reference below is the oracle; the Metal kernels must equal it.
"""

from __future__ import annotations

from functools import cache

import mlx.core as mx
import numpy as np

_MASK64 = (1 << 64) - 1
_GOLDEN = 0x9E3779B97F4A7C15
_MIX1 = 0xBF58476D1CE4E5B9
_MIX2 = 0x94D049BB133111EB
_MTP_SEED_OFFSET = 1000000007
_53_BITS = 1 << 53

CASE_TARGET = 0
CASE_DRAFT = 1
CASE_VERIFY = 2

_SPLITMIX_SOURCE = r"""
    const uint i = thread_position_in_grid.x;
    if (i >= uint(COUNT)) return;
    ulong z = seeds[i] + 0x9E3779B97F4A7C15ul;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ul;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBul;
    z = z ^ (z >> 31);
    out[i] = float(z >> 11) * (1.0f / 9007199254740992.0f);
"""

_CDF_SOURCE = r"""
    const uint i = thread_position_in_grid.x;
    if (i >= uint(ROWS)) return;
    const float u = draws[i];
    float cum = 0.0f;
    int token = int(VOCAB) - 1;
    for (uint j = 0; j < VOCAB; ++j) {
        cum += probs[i * VOCAB + j];
        if (u <= cum) {
            token = int(j);
            break;
        }
    }
    tokens[i] = token;
"""

# One dispatch: hash the seed, then invert the CDF over the row's probabilities.
_FUSED_SOURCE = r"""
    const uint i = thread_position_in_grid.x;
    if (i >= uint(ROWS)) return;
    ulong z = seeds[i] + 0x9E3779B97F4A7C15ul;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ul;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBul;
    z = z ^ (z >> 31);
    const float u = float(z >> 11) * (1.0f / 9007199254740992.0f);
    float cum = 0.0f;
    int token = int(VOCAB) - 1;
    for (uint j = 0; j < VOCAB; ++j) {
        cum += probs[i * VOCAB + j];
        if (u <= cum) {
            token = int(j);
            break;
        }
    }
    tokens[i] = token;
"""


@cache
def _splitmix_uniform_kernel():
    return mx.fast.metal_kernel(
        name="deepseek_v41_splitmix_uniform",
        input_names=["seeds"],
        output_names=["out"],
        source=_SPLITMIX_SOURCE,
    )


@cache
def _cdf_invert_kernel():
    return mx.fast.metal_kernel(
        name="deepseek_v41_cdf_invert",
        input_names=["draws", "probs"],
        output_names=["tokens"],
        source=_CDF_SOURCE,
    )


@cache
def _counter_categorical_fused_kernel():
    return mx.fast.metal_kernel(
        name="deepseek_v41_counter_categorical",
        input_names=["seeds", "probs"],
        output_names=["tokens"],
        source=_FUSED_SOURCE,
    )


def splitmix64(seed: int) -> int:
    """Reference SplitMix64 over the unsigned 64-bit range."""
    z = (seed + _GOLDEN) & _MASK64
    z = ((z ^ (z >> 30)) * _MIX1) & _MASK64
    z = ((z ^ (z >> 27)) * _MIX2) & _MASK64
    return (z ^ (z >> 31)) & _MASK64


def seed_for(seq_seed: int, sample_position: int, case: int) -> int:
    """The draw's seed from the tuple alone. No batch or row term."""
    seed = (seq_seed ^ (sample_position * _GOLDEN) ^ (case * 3 + 1)) & _MASK64
    if case == CASE_DRAFT:
        seed ^= _MTP_SEED_OFFSET
    return seed & _MASK64


def uniform_from_seed(seed: int) -> float:
    """Uniform in [0, 1) from the top 53 bits, exactly as the kernel does."""
    return (splitmix64(seed) >> 11) / float(_53_BITS)


def draw_u(seq_seed: int, sample_position: int, case: int) -> float:
    """Reference uniform draw for one tuple."""
    return uniform_from_seed(seed_for(seq_seed, sample_position, case))


def seeds_mx(values) -> mx.array:
    """Build a uint64 seed vector.

    MLX cannot cast a Python list whose values exceed int64 range into uint64,
    so the conversion goes through numpy, which holds the full unsigned range.
    """
    return mx.array(np.array(list(values), dtype=np.uint64))


def draw_u_mx(seeds: mx.array) -> mx.array:
    """Uniform draws for a uint64 seed vector, on the GPU."""
    seeds = seeds.astype(mx.uint64)
    count = int(seeds.shape[0])
    if count == 0:
        return mx.zeros((0,), mx.float32)
    return _splitmix_uniform_kernel()(
        inputs=[seeds],
        template=[("COUNT", count)],
        grid=(count, 1, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(count,)],
        output_dtypes=[mx.float32],
    )[0]


def cdf_tokens(draws: mx.array, probs: mx.array) -> mx.array:
    """Token index per row by CDF inversion, on the GPU.

    ``token`` is the smallest index whose cumulative probability reaches ``u``,
    which equals ``sum(cumsum(probs) < u)`` from the reference.
    """
    if probs.ndim != 2:
        raise ValueError("probs must be [rows, vocab]")
    rows = int(probs.shape[0])
    vocab = int(probs.shape[1])
    if int(draws.shape[0]) != rows:
        raise ValueError("draws must match the probability rows")
    return _cdf_invert_kernel()(
        inputs=[draws.astype(mx.float32), probs.astype(mx.float32)],
        template=[("ROWS", rows), ("VOCAB", vocab)],
        grid=(rows, 1, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(rows,)],
        output_dtypes=[mx.int32],
    )[0]


def counter_categorical_fused(
    probs: mx.array,
    *,
    seq_seed: int,
    sample_positions: mx.array,
    case: int = CASE_TARGET,
) -> mx.array:
    """Deterministic token per row in one dispatch: hash the seed, invert the CDF."""
    if probs.ndim != 2:
        raise ValueError("probs must be [rows, vocab]")
    rows = int(probs.shape[0])
    vocab = int(probs.shape[1])
    if rows <= 0 or vocab <= 0:
        raise ValueError("empty probability table")
    if int(sample_positions.shape[0]) != rows:
        raise ValueError("sample positions must match the probability rows")
    seeds = seeds_mx(
        seed_for(int(seq_seed), int(position), int(case))
        for position in sample_positions.astype(mx.int32).tolist()
    )
    return _counter_categorical_fused_kernel()(
        inputs=[seeds, probs.astype(mx.float32)],
        template=[("ROWS", rows), ("VOCAB", vocab)],
        grid=(rows, 1, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(rows,)],
        output_dtypes=[mx.int32],
    )[0]


def counter_categorical(
    probs: mx.array,
    *,
    seq_seed: int,
    sample_positions: mx.array,
    case: int = CASE_TARGET,
) -> mx.array:
    """Deterministic token per row from the tuple and the row's probabilities.

    One fused dispatch. ``draw_u_mx`` plus ``cdf_tokens`` remain as the
    two-dispatch oracle the fused kernel is checked against.
    """
    return counter_categorical_fused(
        probs,
        seq_seed=seq_seed,
        sample_positions=sample_positions,
        case=case,
    )


class CounterDraw:
    """Per-request deterministic categorical draw.

    The position advances per call and per row, and the tuple ``(seq_seed,
    position, case)`` is the only input, so no batch composition or RNG state
    can change the draw. Rows of one call belong to the same request, so a
    row's position is the call's base plus its row index.
    """

    def __init__(self, seq_seed: int, case: int = CASE_TARGET):
        self.seq_seed = int(seq_seed)
        self.case = int(case)
        self.position = 0

    def __call__(self, logits: mx.array, temp: float) -> mx.array:
        if logits.ndim != 2:
            raise ValueError("the counter draw needs [rows, vocab] logits")
        rows = int(logits.shape[0])
        if rows <= 0:
            raise ValueError("empty logits row set")
        positions = mx.arange(self.position, self.position + rows, dtype=mx.int32)
        self.position += rows
        probs = mx.softmax(logits.astype(mx.float32) * (1 / temp), axis=-1)
        return counter_categorical(
            probs,
            seq_seed=self.seq_seed,
            sample_positions=positions,
            case=self.case,
        )


def counter_draw_for(seq_seed: int, case: int = CASE_TARGET) -> CounterDraw:
    """Build a per-request counter draw for the sampler factory."""
    return CounterDraw(seq_seed, case)
