# SPDX-License-Identifier: Apache-2.0
"""Unit 24: two-tick probe for the engine's settle-then-launch shape.

U23 proved that host work hides inside one GPU submission. The engine's tick has
a stricter shape: tick t launches its own GPU work while the host settles tick
t-1, which includes one blocking read of t-1's acceptance arrays, then does
deferred host work. This probe mimics that order with two submissions and one
`mx.eval` of the earlier tick's array, and reports whether the total is additive.
"""

from __future__ import annotations

import time

import mlx.core as mx


def _gpu_batch(k: int, size: int = 1024):
    a = mx.random.normal((size, size))
    b = mx.random.normal((size, size))
    return [mx.matmul(a, b) for _ in range(k)]


def _busy(seconds: float) -> float:
    start = time.perf_counter()
    value = 0
    while time.perf_counter() - start < seconds:
        for _ in range(10_000):
            value += 1
    return time.perf_counter() - start


def _tick_probe(k_gpu: int, host_seconds: float) -> dict[str, float]:
    mx.eval(_gpu_batch(4))  # warm

    start = time.perf_counter()
    mx.eval(_gpu_batch(k_gpu))
    mx.synchronize()
    gpu_tick = time.perf_counter() - start

    host_only = _busy(host_seconds)

    prev_ops = _gpu_batch(k_gpu)
    mx.async_eval(prev_ops)
    curr_ops = _gpu_batch(k_gpu)
    mx.async_eval(curr_ops)
    start = time.perf_counter()
    mx.eval(prev_ops[0])  # settle: one blocking read of tick t-1
    host_inside = _busy(host_seconds)  # deferred host work for tick t-1
    mx.synchronize()  # tick t completes
    total = time.perf_counter() - start

    additive = 2 * gpu_tick + host_only
    return {
        "gpu_tick_ms": gpu_tick * 1000,
        "host_only_ms": host_only * 1000,
        "host_inside_ms": host_inside * 1000,
        "two_tick_total_ms": total * 1000,
        "additive_ms": additive * 1000,
        "max_expected_ms": (gpu_tick + max(gpu_tick, host_only)) * 1000,
    }


def test_probe_reports_the_two_tick_measurements(capsys):
    numbers = _tick_probe(k_gpu=64, host_seconds=0.15)
    print("tick probe", {key: round(value, 2) for key, value in numbers.items()})
    assert numbers["two_tick_total_ms"] > 0


def test_two_tick_total_is_not_additive():
    numbers = _tick_probe(k_gpu=64, host_seconds=0.15)
    hidden = numbers["additive_ms"] - numbers["two_tick_total_ms"]
    smaller = min(numbers["gpu_tick_ms"], numbers["host_only_ms"])
    # Additive timing would hide nothing. The probe passes when at least half of
    # the smaller phase hides inside the larger one.
    assert hidden >= 0.5 * smaller, numbers


def test_two_tick_total_stays_near_the_maximum_expected():
    numbers = _tick_probe(k_gpu=64, host_seconds=0.15)
    # The settle read costs one GPU tick, then the host work should hide the
    # second GPU tick, so the total stays close to gpu + max(gpu, host).
    slack = 0.5 * numbers["gpu_tick_ms"]
    assert numbers["two_tick_total_ms"] <= numbers["max_expected_ms"] + slack, numbers
