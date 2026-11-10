# SPDX-License-Identifier: MIT
"""Fused reference-format FP8/FP4 KV packing; preserve rounding boundaries."""

from functools import cache
import mlx.core as mx

_SOURCE = r"""
 const uint d = thread_position_in_grid.x;
 const uint row = thread_position_in_grid.y;
 if (d >= D) return;
 const uint lane = thread_index_in_simdgroup;
 const float v = float(x[row * D + d]);
 auto power2 = [](int e) { return as_type<float>(uint(e + 127) << 23); };
 auto round8 = [&](float a) {
   float b = min(abs(a), 448.0f);
   // E4M3 binade from the exponent bits (exact), subnormal grid 2^-9 below 2^-6.
   int e = max(int((as_type<uint>(b) >> 23) & 255u) - 127 - 3, -9);
   float step = power2(e);
   return sign(a) * min(rint(b / step) * step, 448.0f);
 };
 auto code8 = [](float q) {
   uint b = as_type<uint>(q);
   uchar s = uchar((b >> 31) << 7);
   uint e = (b >> 23) & 255u;
   if (e <= 120u) return uchar(s | uchar(int(rint(abs(q) * 512.0f))));
   return uchar(s | uchar(((e - 120u) << 3) | ((b >> 20) & 7u)));
 };
 float amax = abs(v);
 for (uint offset = G / 2; offset > 0; offset >>= 1)
   amax = max(amax, simd_shuffle_xor(amax, offset));
 constexpr float limit = BITS == 8 ? 448.0f : 6.0f;
 amax = max(amax, as_type<float>(uint(MIN_BITS)));
 float scale;
 uchar sb;
 if (E4S) {
   // kernel.py fp4_quant_kernel: s = Cast(FP8, amax / fp4_max) -- a division, not * (1/6).
   scale = round8(amax / limit);
   sb = code8(scale);
 } else {
   // kernel.py fast_round_scale: 2^ceil(log2(amax * fp32(1/limit))) from the product's bits.
   const uint pb = as_type<uint>(amax * (1.0f / limit));
   int e = int((pb >> 23) & 255u) - 127 + int((pb & 0x7FFFFFu) != 0u);
   scale = power2(e);
   sb = uchar(e + 127);
 }
 float scaled = clamp(v / scale, -limit, limit);
 constexpr uint BYTES = D * BITS / 8;
 constexpr uint WIDTH = BYTES + D / G;
 if (BITS == 8) {
   y[row * WIDTH + d] = code8(round8(scaled));
 } else {
   float a = abs(scaled);
   uchar code = uchar(a > 0.25f) + uchar(a >= 0.75f) + uchar(a > 1.25f)
              + uchar(a >= 1.75f) + uchar(a > 2.5f) + uchar(a >= 3.5f)
              + uchar(a > 5.0f);
   code |= uchar(scaled < 0.0f) << 3;
   uint other = simd_shuffle_xor(uint(code), 1);
   if ((d & 1u) == 0u) y[row * WIDTH + d / 2] = code | uchar(other << 4);
 }
 if (lane % G == 0) y[row * WIDTH + BYTES + d / G] = sb;
"""


@mx.compile
def _reference_minimum(x, bits, e4m3_scale):
    # DeepSeek-V4.1-Flash kernel.py: act_quant max(amax, 1e-4); fp4_act_quant
    # 6*2^-9 (E4M3 scales) or 6*2^-126 (E8M0 scales).
    if bits == 8:
        return mx.maximum(x, 1e-4)
    return mx.maximum(x, 6.0 * (2.0**-9 if e4m3_scale else 2.0**-126))


@cache
def _minimum_bits(bits, e4m3_scale):
    # MLX compiled scalar literals can round differently from Metal constants.
    # Capture the reference value once, retaining even zero-group scale bytes.
    return int(
        _reference_minimum(mx.zeros((1,), mx.float32), bits, e4m3_scale)
        .view(mx.uint32)
        .item()
    )


@cache
def _kernel():
    return mx.fast.metal_kernel(
        name="v41_pack_kv_exact", input_names=["x"], output_names=["y"], source=_SOURCE
    )


def fused_pack_activation(x, bits=8, group_size=32, e4m3_scale=False):
    dim = x.shape[-1]
    rows = x.size // dim
    width = dim * bits // 8 + dim // group_size
    return _kernel()(
        inputs=[x],
        template=[
            ("D", dim),
            ("BITS", bits),
            ("G", group_size),
            ("E4S", e4m3_scale),
            ("MIN_BITS", _minimum_bits(bits, e4m3_scale)),
        ],
        grid=(dim, rows, 1),
        threadgroup=(min(256, dim), 1, 1),
        output_shapes=[(*x.shape[:-1], width)],
        output_dtypes=[mx.uint8],
    )[0]


# One dispatch: rotate the last RD dims, cast to the input dtype exactly as the
# rope kernel does, then quantize and pack exactly as the pack kernel does. The
# oracle is `fused_pack_activation(rope_activation(x, cos, sin))`.
_ROPE_PACK_SOURCE = r"""
    const uint i = thread_position_in_grid.x * 8;
    if (i >= uint(N)) return;
    const uint d = i % D;
    const uint row = i / D;
    float v[8];
    for (int j = 0; j < 8; ++j) v[j] = float(x[i + j]);
    if (d >= uint(D - RD)) {
        const uint p = ((i / ROW) % L) * (RD / 2) + (d - (D - RD)) / 2;
        for (int j = 0; j < 8; j += 2) {
            const float cs = cosv[p + j / 2], sn = sinv[p + j / 2];
            const float a = v[j], b = v[j + 1];
            v[j] = float(T(a * cs - b * sn));
            v[j + 1] = float(T(a * sn + b * cs));
        }
    }
    auto power2 = [](int e) { return as_type<float>(uint(e + 127) << 23); };
    auto round8 = [&](float a) {
        float b = min(abs(a), 448.0f);
        int e = max(int((as_type<uint>(b) >> 23) & 255u) - 127 - 3, -9);
        float step = power2(e);
        return sign(a) * min(rint(b / step) * step, 448.0f);
    };
    auto code8 = [](float q) {
        uint b = as_type<uint>(q);
        uchar s = uchar((b >> 31) << 7);
        uint e = (b >> 23) & 255u;
        if (e <= 120u) return uchar(s | uchar(int(rint(abs(q) * 512.0f))));
        return uchar(s | uchar(((e - 120u) << 3) | ((b >> 20) & 7u)));
    };
    float amax = 0.0f;
    for (int j = 0; j < 8; ++j) amax = max(amax, abs(v[j]));
    amax = max(amax, simd_shuffle_xor(amax, 1u));
    if (G == 32u) amax = max(amax, simd_shuffle_xor(amax, 2u));
    amax = max(amax, as_type<float>(uint(MIN_BITS)));
    constexpr float limit = BITS == 8 ? 448.0f : 6.0f;
    float scale;
    uchar sb;
    if (E4S) {
        scale = round8(amax / limit);
        sb = code8(scale);
    } else {
        const uint pb = as_type<uint>(amax * (1.0f / limit));
        int e = int((pb >> 23) & 255u) - 127 + int((pb & 0x7FFFFFu) != 0u);
        scale = power2(e);
        sb = uchar(e + 127);
    }
    constexpr uint BYTES = D * BITS / 8;
    constexpr uint WIDTH = BYTES + D / G;
    if (BITS == 8) {
        for (int j = 0; j < 8; ++j) {
            const float scaled = clamp(v[j] / scale, -limit, limit);
            y[row * WIDTH + d + j] = code8(round8(scaled));
        }
    } else {
        for (int j = 0; j < 8; j += 2) {
            const float s0 = clamp(v[j] / scale, -limit, limit);
            const float s1 = clamp(v[j + 1] / scale, -limit, limit);
            const float a0 = abs(s0), a1 = abs(s1);
            uchar c0 = uchar(a0 > 0.25f) + uchar(a0 >= 0.75f) + uchar(a0 > 1.25f)
                     + uchar(a0 >= 1.75f) + uchar(a0 > 2.5f) + uchar(a0 >= 3.5f)
                     + uchar(a0 > 5.0f);
            uchar c1 = uchar(a1 > 0.25f) + uchar(a1 >= 0.75f) + uchar(a1 > 1.25f)
                     + uchar(a1 >= 1.75f) + uchar(a1 > 2.5f) + uchar(a1 >= 3.5f)
                     + uchar(a1 > 5.0f);
            c0 |= uchar(s0 < 0.0f) << 3;
            c1 |= uchar(s1 < 0.0f) << 3;
            y[row * WIDTH + (d + j) / 2] = uchar(c0 | uchar(c1 << 4));
        }
    }
    if (d % G == 0) y[row * WIDTH + BYTES + d / G] = sb;
"""


@cache
def _rope_pack_kernel():
    return mx.fast.metal_kernel(
        name="v41_rope_pack_exact",
        input_names=["x", "cosv", "sinv"],
        output_names=["y"],
        source=_ROPE_PACK_SOURCE,
    )


def fused_rope_pack(x, cos, sin, bits=8, group_size=32, e4m3_scale=False):
    """RoPE then FP8/FP4 packing in one dispatch; see ``_ROPE_PACK_SOURCE``."""
    dim = x.shape[-1]
    rd = cos.shape[-1] * 2
    length = x.shape[1]
    row_width = x.size // (x.shape[0] * length)
    width = dim * bits // 8 + dim // group_size
    if rd > dim or dim % 8 or rd % 2:
        raise ValueError("invalid RoPE width or row alignment")
    if bits not in (4, 8):
        raise ValueError("bits must be 4 or 8")
    if group_size <= 0 or dim % group_size:
        raise ValueError("group size must divide the row width")
    if cos.shape[0] != length:
        raise ValueError("cos table length must match the sequence length")
    return _rope_pack_kernel()(
        inputs=[x, cos, sin],
        template=[
            ("T", x.dtype),
            ("N", x.size),
            ("D", dim),
            ("RD", rd),
            ("L", length),
            ("ROW", row_width),
            ("BITS", bits),
            ("G", group_size),
            ("E4S", e4m3_scale),
            ("MIN_BITS", _minimum_bits(bits, e4m3_scale)),
        ],
        grid=(x.size // 8, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(*x.shape[:-1], width)],
        output_dtypes=[mx.uint8],
    )[0]
