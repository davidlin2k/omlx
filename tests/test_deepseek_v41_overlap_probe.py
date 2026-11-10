# SPDX-License-Identifier: Apache-2.0
"""Unit 23: substrate probe for cross-submission host and GPU overlap.

The two-stage decode design depends on one unmeasured property: whether host
Python runs concurrently with the GPU while an `mx.async_eval` submission is in
flight. This probe measures a GPU-only batch, a host-only busy loop, and both
together, then reports whether the combined time is additive.

A combined time near the sum means Metal serializes submission and host work, and
the pipeline shape gains nothing. A combined time near the maximum means the host
work hides inside the GPU work.
"""

from __future__ import annotations

import time

import mlx.core as mx


def _gpu_batch(k: int, size: int = 1024):
    a = mx.random.normal((size, size))
    b = mx.random.normal((size, size))
    return [mx.matmul(a, b) for _ in range(k)]


def _busy(seconds: float) -> float:
    """Host-only work for the requested duration; returns the actual time."""
    start = time.perf_counter()
    value = 0
    while time.perf_counter() - start < seconds:
        for _ in range(10_000):
            value += 1
    return time.perf_counter() - start


def _probe(k_gpu: int, host_seconds: float) -> dict[str, float]:
    mx.eval(_gpu_batch(4))  # warm the pipeline and the allocator

    start = time.perf_counter()
    mx.eval(_gpu_batch(k_gpu))
    mx.synchronize()
    gpu_only = time.perf_counter() - start

    start = time.perf_counter()
    host_only = _busy(host_seconds)

    ops = _gpu_batch(k_gpu)
    mx.async_eval(ops)
    start = time.perf_counter()
    host_inside = _busy(host_seconds)
    mx.synchronize()
    combined = time.perf_counter() - start

    return {
        "gpu_only_ms": gpu_only * 1000,
        "host_only_ms": host_only * 1000,
        "host_inside_ms": host_inside * 1000,
        "combined_ms": combined * 1000,
        "sum_ms": (gpu_only + host_only) * 1000,
        "max_ms": max(gpu_only, host_only) * 1000,
    }


def test_probe_reports_the_three_measurements(capsys):
    numbers = _probe(k_gpu=64, host_seconds=0.15)
    print("overlap probe", {key: round(value, 2) for key, value in numbers.items()})
    assert numbers["gpu_only_ms"] > 0
    assert numbers["host_only_ms"] > 0
    assert numbers["combined_ms"] > 0


def test_overlap_is_not_additive():
    numbers = _probe(k_gpu=64, host_seconds=0.15)
    hidden = numbers["sum_ms"] - numbers["combined_ms"]
    smaller = min(numbers["gpu_only_ms"], numbers["host_only_ms"])
    # Additive timing would hide nothing. The probe passes when at least half of
    # the smaller phase hides inside the larger one.
    assert hidden >= 0.5 * smaller, numbers


def test_host_work_inside_the_submission_matches_host_only():
    numbers = _probe(k_gpu=64, host_seconds=0.15)
    assert numbers["host_inside_ms"] <= 3 * numbers["host_only_ms"]
