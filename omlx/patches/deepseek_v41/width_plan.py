# SPDX-License-Identifier: Apache-2.0
"""One-dispatch verification-width plan for DeepSeek V4.1 DSpark cycles.

The whole decision runs on the GPU: cumulative product of the previous step's
confidence rates through a remap, a stable descending rank with a flat-index
tie rule, the extra-budget wins per row, and the clamp. This mirrors DeepInfer's
``compute_seqlens_q_by_confidence_rates`` kernel plus its C++ comparator, so the
host never transfers per-row rates to build widths.

JIT tier: the kernel compiles at runtime, so no ``make kernels`` build is needed
and the code lives beside the model patch.
"""

from __future__ import annotations

from functools import cache

import mlx.core as mx

from .width_policy import MAX_VERIFY_WIDTH

_SOURCE = r"""
    const uint seq = thread_position_in_grid.x;
    if (seq >= uint(B)) return;
    const int src = remap[seq];
    float cum[NEXTN];
    for (uint p = 0; p < NEXTN; ++p) {
        const float r = (src < 0) ? 0.0f : prev_rates[src * NEXTN + p];
        cum[p] = (p == 0) ? r : cum[p - 1] * r;
    }
    const int extra = total_q[0] - int(B);
    int wins = 0;
    for (uint p = 0; p < NEXTN; ++p) {
        const uint f = seq * NEXTN + p;
        uint rank = 0;
        for (uint f2 = 0; f2 < NEXTN * B; ++f2) {
            const uint s2 = f2 / NEXTN;
            const uint p2 = f2 % NEXTN;
            const int src2 = remap[s2];
            float c2 = 1.0f;
            for (uint j = 0; j <= p2; ++j) {
                c2 *= (src2 < 0) ? 0.0f : prev_rates[src2 * NEXTN + j];
            }
            if (c2 > cum[p] || (c2 == cum[p] && f2 < f)) rank += 1;
        }
        if (extra > 0 && rank < uint(extra)) wins += 1;
    }
    int width = wins + 1;
    if (width > CAP) width = CAP;
    if (width < 1) width = 1;
    widths[seq] = width;
"""


@cache
def _width_plan_kernel():
    return mx.fast.metal_kernel(
        name="deepseek_v41_width_plan",
        input_names=["remap", "prev_rates", "total_q"],
        output_names=["widths"],
        source=_SOURCE,
    )


def width_plan(
    remap: mx.array,
    prev_rates: mx.array,
    total_q: int,
    *,
    cap: int = MAX_VERIFY_WIDTH,
) -> mx.array:
    """Per-row widths for one cycle, computed on the GPU in one dispatch."""
    if remap.ndim != 1:
        raise ValueError("remap must be a flat int32 vector")
    if prev_rates.ndim != 2:
        raise ValueError("prev_rates must be [rows, nextn]")
    if cap < 1 or cap > MAX_VERIFY_WIDTH:
        raise ValueError("cap must be inside decode_fused.MAX_BATCH")
    rows = int(remap.shape[0])
    nextn = int(prev_rates.shape[1])
    if rows == 0 or nextn == 0:
        raise ValueError("empty remap or rates")
    remap32 = remap.astype(mx.int32)
    if int(mx.max(remap32, axis=-1).item()) >= int(prev_rates.shape[0]):
        raise ValueError("remap index exceeds the previous rate rows")
    widths = _width_plan_kernel()(
        inputs=[
            remap32,
            prev_rates.astype(mx.float32),
            mx.array([int(total_q)], mx.int32),
        ],
        template=[("NEXTN", nextn), ("CAP", cap), ("B", rows)],
        grid=(rows, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(rows,)],
        output_dtypes=[mx.int32],
    )[0]
    return widths
