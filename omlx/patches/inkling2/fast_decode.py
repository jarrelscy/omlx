# SPDX-License-Identifier: Apache-2.0
"""Fused decode / speculative-verify path for Inkling 2.

A decode step visits 48 layers (looped core). The generic path issues ~90
MLX primitives per visit, and every one of them sits on a single dependent
chain, so batch-1 decode is bound by per-kernel latency rather than weight
bandwidth. This module replaces one visit with ~15 primitives:

  pre    attention-residual depth mix + RMSNorms (+ loop messages)
  [ctl]  loop controllers (low-rank gated message mix) for looped passes
  proj   attention projection(s) (MLX quantized matmul)
  epi    short convs, L2 clips, value norm, relative bias rows, sinks
  attn   split-K banded attention with per-query sinks + output gate
  o_proj (MLX quantized matmul)
  post   output conv + residual + MLP depth mix + MLP norm
  router (MLX matmul) + top-k
  moe    fused 4-bit expert matvecs (w13 + SwiGLU, w2 + combine + conv)

It handles one row (``B == 1``) with up to ``_MAX_L`` tokens (decode and
MTP verify) on a plain ``KVCache``; prefill and batched rows take the
reference path in ``model.py``. Numerics follow the reference rounding
points (BF16 after each projection/conv) to within an ulp.
"""

from __future__ import annotations

import logging
import os

import mlx.core as mx
from mlx_lm.models.cache import KVCache

logger = logging.getLogger(__name__)

_EPS = 1e-6
_TG = int(os.environ.get("OMLX_INKLING2_TG", "512"))  # row-kernel threads
_MAX_L = int(os.environ.get("OMLX_INKLING2_FAST_MAX_L", "8"))
_ENABLED = os.environ.get("OMLX_INKLING2_FAST_DECODE", "1") not in ("0", "false")
_DEBUG = os.environ.get("OMLX_INKLING2_FAST_DEBUG", "0") not in ("0", "false")

_HEADER = """
inline float bf(float x) { return float(bfloat(x)); }
constant float FP4_LUT[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                              -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
inline float e4m3(uint b) {
    uint e = (b >> 3) & 0xF;
    uint m = b & 7;
    float v = e == 0 ? float(m) * 0.001953125f : (1.0f + float(m) * 0.125f) * metal::exp2(float(int(e) - 7));
    return (b & 0x80) ? -v : v;
}
inline float fp4_dot16(uint2 w, thread const float* xv) {
    float a = 0.0f;
    for (int k = 0; k < 8; ++k) a += FP4_LUT[(w.x >> (4 * k)) & 0xF] * xv[k];
    for (int k = 0; k < 8; ++k) a += FP4_LUT[(w.y >> (4 * k)) & 0xF] * xv[8 + k];
    return a;
}
inline float u4_dot16(uint2 w, thread const float* xv) {
    float a = 0.0f;
    for (int k = 0; k < 8; ++k) a += float((w.x >> (4 * k)) & 0xF) * xv[k];
    for (int k = 0; k < 8; ++k) a += float((w.y >> (4 * k)) & 0xF) * xv[8 + k];
    return a;
}
"""

_kernels: dict = {}


def _kernel(key, inputs, outputs, source, contiguous=True):
    k = _kernels.get(key)
    if k is None:
        name = "omlx_ink2_" + "_".join(str(x) for x in key)
        name = "".join(ch if ch.isalnum() else "_" for ch in name)
        k = mx.fast.metal_kernel(
            name=name,
            input_names=inputs,
            output_names=outputs,
            source=source,
            header=_HEADER,
            ensure_row_contiguous=contiguous,
        )
        _kernels[key] = k
    return k


def _memo(fn):
    """Cache a kernel factory by its arguments (source generation is slow)."""
    cache = {}

    def wrapper(*args):
        key = (args, _TG)
        hit = cache.get(key)
        if hit is None:
            hit = cache[key] = fn(*args)
        return hit

    wrapper.__wrapped__ = fn
    wrapper.cache = cache
    _memo_caches.append(cache)
    return wrapper


_memo_caches: list = []


def _f(x: float) -> str:
    return repr(float(x)) + "f"


# --------------------------------------------------------------------------
# Row kernels: one threadgroup of _TG threads per token row over a hidden
# vector of H elements; thread t owns elements t, t + TG, ...

_ROW_PRELUDE = """
    constexpr uint TG = {TG};
    constexpr uint H = {H};
    constexpr uint L = {L};
    constexpr uint EPT = (H + TG - 1) / TG;
    constexpr uint NSG = TG / 32;
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    threadgroup float red[32 * NSG];
"""


def _reduce(vals, tag):
    lines = ["    threadgroup_barrier(mem_flags::mem_threadgroup);"]
    for i, v in enumerate(vals):
        lines.append(f"    {{ float t_ = simd_sum({v}); if (lane == 0) red[{i} * NSG + sg] = t_; }}")
    lines.append("    threadgroup_barrier(mem_flags::mem_threadgroup);")
    for i in range(len(vals)):
        lines.append(
            f"    float {tag}{i} = 0.0f; for (uint j_ = 0; j_ < NSG; ++j_) {tag}{i} += red[{i} * NSG + j_];"
        )
    return "\n".join(lines)


def _mix_code(srcs, nq, query, scale, out, tag):
    """Softmax depth mixing of per-thread FP32 source registers (partial last)."""
    ns = len(srcs)
    code = [f"    float {out}[{nq}][EPT];"]
    accs = []
    for s in range(ns):
        for q in range(nq):
            code.append(f"    float {tag}a{s}_{q} = 0.0f;")
            accs.append(f"{tag}a{s}_{q}")
    code.append("    for (uint e = 0; e < EPT; ++e) { uint i = tid + e * TG; if (i < H) {")
    for q in range(nq):
        code.append(f"        float qv{q} = {query}[{q} * H + i];")
        for s in range(ns):
            code.append(f"        {tag}a{s}_{q} += {srcs[s]}[e] * qv{q};")
    code.append("    }}")
    code.append(_reduce(accs, f"{tag}d"))
    for q in range(nq):
        code.append(f"    float {tag}m{q} = -INFINITY;")
        for s in range(ns):
            code.append(
                f"    float {tag}l{s}_{q} = {tag}d{s * nq + q} * {scale}; {tag}m{q} = max({tag}m{q}, {tag}l{s}_{q});"
            )
        code.append(f"    float {tag}z{q} = 0.0f;")
        for s in range(ns):
            code.append(
                f"    float {tag}w{s}_{q} = metal::exp({tag}l{s}_{q} - {tag}m{q}); {tag}z{q} += {tag}w{s}_{q};"
            )
        for s in range(ns):
            code.append(f"    {tag}w{s}_{q} /= {tag}z{q};")
        mix = " + ".join(f"{tag}w{s}_{q} * {srcs[s]}[e]" for s in range(ns))
        code.append(f"    for (uint e = 0; e < EPT; ++e) {out}[{q}][e] = {mix};")
    return "\n".join(code)


def _load_rows(names, reg_prefix):
    """Load FP32 [L, H] inputs (row ``row``) into registers ``{reg}{j}r``."""
    code = []
    for j, n in enumerate(names):
        code.append(f"    float {reg_prefix}{j}r[EPT];")
        code.append(
            f"    for (uint e = 0; e < EPT; ++e) {{ uint i = tid + e * TG; {reg_prefix}{j}r[e] = i < H ? {n}[row * H + i] : 0.0f; }}"
        )
    return "\n".join(code)


def _unit_rms_code(src, dst, tag):
    return "\n".join(
        [
            f"    float {tag}ss = 0.0f;",
            f"    for (uint e = 0; e < EPT; ++e) {tag}ss += {src}[e] * {src}[e];",
            _reduce([f"{tag}ss"], f"{tag}r"),
            f"    float {tag}inv = metal::rsqrt({tag}r0 / float(H) + {_f(_EPS)});",
            f"    float {dst}[EPT];",
            f"    for (uint e = 0; e < EPT; ++e) {dst}[e] = {src}[e] * {tag}inv;",
        ]
    )


def _norm_store(src, weight, out, out_off, tag, out32=None):
    """RMSNorm with a BF16 weight stored as BF16 (and optionally its FP32 value)."""
    lines = [
        f"    float {tag}ss = 0.0f;",
        f"    for (uint e = 0; e < EPT; ++e) {tag}ss += {src}[e] * {src}[e];",
        _reduce([f"{tag}ss"], f"{tag}r"),
        f"    float {tag}inv = metal::rsqrt({tag}r0 / float(H) + {_f(_EPS)});",
        "    for (uint e = 0; e < EPT; ++e) { uint i = tid + e * TG; if (i < H) {",
        f"        bfloat o_ = bfloat(float({weight}[i]) * ({src}[e] * {tag}inv));",
        f"        {out}[{out_off} + i] = o_;",
    ]
    if out32:
        lines.append(f"        {out32}[{out_off} + i] = float(o_);")
    lines.append("    }}")
    return "\n".join(lines)


def _store_z(src, out, off, tag):
    """BF16-rounded unit-RMS of ``src`` (a controller input)."""
    return "\n".join(
        [
            _unit_rms_code(src, f"{tag}u", tag),
            f"    for (uint e = 0; e < EPT; ++e) {{ uint i = tid + e * TG; if (i < H) {out}[{off} + i] = bf({tag}u[e]); }}",
        ]
    )


@_memo
def _pre_kernel(h, length, nb, nq, nbank):
    """Attention inputs for one token row (grid: L threadgroups).

    Without a bank: hn (FP32 [L, H]) and x (BF16 [nq, L, H], normalized).
    With a bank: hn, x (FP32 [nq, L, H] raw mixture), m (FP32 [4, L, H]
    messages) and z (FP32 [3, 2, L, H] controller inputs).
    """
    key = ("pre", h, length, nb, nq, nbank, _TG)
    blocks = [f"b{j}" for j in range(nb)]
    if nbank:
        bank = [f"k{j}" for j in range(nbank)] + ["kp"]
        inputs = ["h", *blocks, "query", *bank, "mquery", "scale"]
        outputs = ["hn", "x", "m", "z"]
    else:
        norms = ["w0"] if nq == 1 else ["w0", "w1", "w2"]
        inputs = ["h", *blocks, "query", *norms, "scale"]
        outputs = ["hn", "x"]
    src = [_ROW_PRELUDE.format(TG=_TG, H=h, L=length), "    uint row = threadgroup_position_in_grid.x;"]
    src.append(
        "    float hr[EPT];\n"
        "    for (uint e = 0; e < EPT; ++e) { uint i = tid + e * TG; hr[e] = i < H ? float(h[row * H + i]) : 0.0f; }"
    )
    src.append(_unit_rms_code("hr", "pn", "u"))
    src.append("    for (uint e = 0; e < EPT; ++e) { uint i = tid + e * TG; if (i < H) hn[row * H + i] = pn[e]; }")
    if nb == 0:
        # Layer 0: the attention input is h itself; RMSNorm of a BF16 input
        # rounds x * inv to BF16 before the weight.
        src.append(
            "    for (uint e = 0; e < EPT; ++e) { uint i = tid + e * TG; if (i < H)"
            " x[row * H + i] = bfloat(float(w0[i]) * bf(hr[e] * uinv)); }"
        )
        return _kernel(key, inputs, outputs, "\n".join(src))
    src.append(_load_rows(blocks, "b"))
    src.append(_mix_code([f"b{j}r" for j in range(nb)] + ["pn"], nq, "query", "scale[0]", "xo", "a"))
    if not nbank:
        for q in range(nq):
            w = "w0" if nq == 1 else f"w{q}"
            src.append(_norm_store(f"xo[{q}]", w, "x", f"({q} * L + row) * H", f"n{q}"))
    else:
        src.append(_load_rows(bank, "k"))
        src.append(_mix_code([f"k{j}r" for j in range(nbank + 1)], 4, "mquery", "scale[0]", "mo", "c"))
        src.append(
            "    for (uint e = 0; e < EPT; ++e) { uint i = tid + e * TG; if (i < H) {\n"
            + "".join(f"        x[({q} * L + row) * H + i] = xo[{q}][e];\n" for q in range(nq))
            + "".join(f"        m[({q} * L + row) * H + i] = mo[{q}][e];\n" for q in range(4))
            + "    }}"
        )
        for n in range(3):
            xs = f"xo[{n if nq == 3 else 0}]"
            src.append(_store_z(xs, "z", f"(({n} * 2) * L + row) * H", f"zx{n}"))
            src.append(_store_z(f"mo[{n}]", "z", f"(({n} * 2 + 1) * L + row) * H", f"zm{n}"))
    return _kernel(key, inputs, outputs, "\n".join(src))


# --------------------------------------------------------------------------
# Loop controllers.

_CTL_SPLIT = 8
_CTL_TG = 256


@_memo
def _ctl_low_kernel(h, r, length):
    """Partial down projections. Grid: N * L * SPLIT threadgroups.

    z: [N, 2, L, H] FP32 (BF16 values); down: [N, 2, H, r] BF16.
    part: [N, L, SPLIT, 2, r] FP32.
    """
    key = ("ctl_low", h, r, length)
    src = f"""
    constexpr uint H = {h};
    constexpr uint R = {r};
    constexpr uint L = {length};
    constexpr uint SPLIT = {_CTL_SPLIT};
    constexpr uint TG = {_CTL_TG};
    constexpr uint NSG = TG / 32;
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint g = threadgroup_position_in_grid.x;
    uint sp = g % SPLIT;
    uint row = (g / SPLIT) % L;
    uint n = g / (SPLIT * L);
    constexpr uint CH = (H + SPLIT - 1) / SPLIT;
    uint i0 = sp * CH;
    uint i1 = min(H, i0 + CH);
    threadgroup float red[2 * R * NSG];
    for (uint half_ = 0; half_ < 2; ++half_) {{
        float acc[R];
        for (uint j = 0; j < R; ++j) acc[j] = 0.0f;
        const device float* zz = z + ((n * 2 + half_) * L + row) * H;
        const device bfloat* dd = down + ((size_t)(n * 2 + half_) * H) * R;
        for (uint i = i0 + tid; i < i1; i += TG) {{
            float zv = zz[i];
            const device bfloat* rw = dd + (size_t)i * R;
            for (uint j = 0; j < R; ++j) acc[j] += zv * float(rw[j]);
        }}
        for (uint j = 0; j < R; ++j) {{
            float t = simd_sum(acc[j]);
            if (lane == 0) red[(half_ * R + j) * NSG + sg] = t;
        }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint o = tid; o < 2 * R; o += TG) {{
        float t = 0.0f;
        for (uint j = 0; j < NSG; ++j) t += red[o * NSG + j];
        part[(((n * L + row) * SPLIT + sp) * 2) * R + o] = t;
    }}
"""
    return _kernel(key, ["z", "down"], ["part"], src)


@_memo
def _ctl_mix_kernel(h, r, n, length, nnorm, out32, xs):
    """Gates, message mix and RMSNorm. Grid: N * L threadgroups.

    x: [N or 1, L, H] FP32; m: [4, L, H] FP32 (rows ``moff``..);
    gf: [N, r, 2H] BF16. y: [N, L, H] BF16 (+ FP32 copy).
    """
    key = ("ctl_mix", h, r, n, length, nnorm, out32, xs, _TG)
    norms = [f"w{i}" for i in range(nnorm)]
    wsel = "w0" if nnorm == 1 else "(n == 0 ? w0 : (n == 1 ? w1 : w2))"
    xrow = "row" if xs else "(n * L + row)"
    src = _ROW_PRELUDE.format(TG=_TG, H=h, L=length) + f"""
    constexpr uint R = {r};
    constexpr uint SPLIT = {_CTL_SPLIT};
    uint row = threadgroup_position_in_grid.x % L;
    uint n = threadgroup_position_in_grid.x / L;
    threadgroup float low[R];
    for (uint j = tid; j < R; j += TG) {{
        float s0 = 0.0f, s1 = 0.0f;
        for (uint sp = 0; sp < SPLIT; ++sp) {{
            s0 += part[(((n * L + row) * SPLIT + sp) * 2) * R + j];
            s1 += part[(((n * L + row) * SPLIT + sp) * 2 + 1) * R + j];
        }}
        low[j] = bf(s0 + s1);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const device bfloat* g = gf + (size_t)n * R * 2 * H;
    float yo[EPT];
    for (uint e = 0; e < EPT; ++e) {{
        uint i = tid + e * TG;
        yo[e] = 0.0f;
        if (i < H) {{
            float ga = 0.0f, fo = 0.0f;
            for (uint j = 0; j < R; ++j) {{
                float l = low[j];
                ga += l * float(g[(size_t)j * 2 * H + i]);
                fo += l * float(g[(size_t)j * 2 * H + H + i]);
            }}
            yo[e] = (1.0f - metal::precise::tanh(fo)) * x[{xrow} * H + i]
                + metal::precise::tanh(ga) * m[((moff[0] + n) * L + row) * H + i];
        }}
    }}
    const device bfloat* wn = {wsel};
"""
    src += _norm_store("yo", "wn", "y", "(n * L + row) * H", "q", "y32" if out32 else None)
    outputs = ["y", "y32"] if out32 else ["y"]
    return _kernel(key, ["part", "x", "m", "moff", "gf", *norms], outputs, src)


# --------------------------------------------------------------------------
# Projection epilogue: short convs, clips, value norm, relative bias rows.


def _epi_kernel(attn, full: bool, msg: bool, length: int):
    key = (id(attn), full, msg, length)
    hit = _epi_cache.get(key)
    if hit is None:
        hit = _epi_cache[key] = _epi_kernel_build(attn, full, msg, length)
    return hit


_epi_cache: dict = {}


def _epi_kernel_build(attn, full: bool, msg: bool, length: int, parts_only: bool = False):
    """Grid: HQ * L threadgroups of D threads (head h of token t).

    Outputs q [L, HQ, D], k/v [HKV, L, D] (BF16), rel [L, HQ, R] (BF16),
    sinks [L, HQ] (BF16) and the conv windows kxp/vxp [3 + L, DKV] FP32.
    """
    hq, hkv, d, dr = attn.n_heads, attn.n_kv, attn.head_dim, attn.d_rel
    r = attn.rel_extent
    kk = attn.conv_k
    assert kk == 4
    key = ("epi", full, msg, length, hq, hkv, d, dr, r, attn.log_floor, attn.log_alpha)
    dq = hq * d
    dkv = hkv * d
    qk_thr = float(attn._qk_clip.item())
    r_thr = float(attn._r_clip.item())
    if full:
        o_q, o_r, o_s = 0, dq, dq + dr * hq + dq
        wq = o_s + hq
        wk = dkv + hkv * kk
        inputs = ["pq", "pk", "pv", "kst", "vst", "ks", "vs", "qsc", "rsc", "relp", "dims"]
        q_at = f"pq[t * {wq} + {{o}}]"
        k_at = f"pk[(tok) * {wk} + {{o}}]"
        v_at = f"pv[(tok) * {wk} + {{o}}]"
    else:
        o_q, o_k, o_v = 0, dq, dq + dkv
        o_r = dq + 2 * dkv
        o_s = o_r + dr * hq + dq
        w = o_s + hq
        inputs = ["p", "kcv", "vcv", "ks", "vs", "qsc", "rsc", "relp", "dims"]
        q_at = f"p[t * {w} + {{o}}]"
        krow = length if msg else 0
        vrow = 2 * length if msg else 0
        k_at = f"p[({krow} + (tok)) * {w} + {o_k} + {{o}}]"
        v_at = f"p[({vrow} + (tok)) * {w} + {o_v} + {{o}}]"

    def raw(at, j):
        # Conv input j of the window [state(3), tokens]: FP32 state row or the
        # raw projection of token j - 3.
        return at

    if full:
        def conv(at, st, wname, out_name):
            xs = []
            for j in range(4):
                xs.append(
                    f"(t + {j} < 3 ? {st}[(t + {j}) * DKV + c] : float({at.format(o='c').replace('(tok)', f'(t + {j} - 3)')}))"
                )
            dyn = [f"float({at.format(o=f'DKV + h * 4 + {lag}').replace('(tok)', 't')})" for lag in range(4)]
            return f"""
            float x0 = {xs[0]}, x1 = {xs[1]}, x2 = {xs[2]}, x3 = {xs[3]};
            float o = x3 * (float({wname}[0 * DKV + c]) + {dyn[0]});
            o = o + x2 * (float({wname}[1 * DKV + c]) + {dyn[1]});
            o = o + x1 * (float({wname}[2 * DKV + c]) + {dyn[2]});
            o = o + x0 * (float({wname}[3 * DKV + c]) + {dyn[3]});
            {out_name} = bf((o + x3) * {_f(attn._dconv_scale)});
            {st}o[(3 + t) * DKV + c] = x3;
            if (t == 0) {{ {st}o[c] = {st}[c]; {st}o[DKV + c] = {st}[DKV + c]; {st}o[2 * DKV + c] = {st}[2 * DKV + c]; }}
"""
        kconv = conv(k_at, "ks", "kst", "kv_")
        vconv = conv(v_at, "vs", "vst", "vv_")
    else:
        def conv(at, st, wname, out_name):
            xs = []
            for j in range(4):
                xs.append(
                    f"(t + {j} < 3 ? {st}[(t + {j}) * DKV + c] : float({at.format(o='c').replace('(tok)', f'(t + {j} - 3)')}))"
                )
            return f"""
            float x0 = {xs[0]}, x1 = {xs[1]}, x2 = {xs[2]}, x3 = {xs[3]};
            float o = float({wname}[c * 4 + 0]) * x0;
            o = o + float({wname}[c * 4 + 1]) * x1;
            o = o + float({wname}[c * 4 + 2]) * x2;
            o = o + float({wname}[c * 4 + 3]) * x3;
            o = bf(o);
            {out_name} = bf((o + x3) * {_f(attn._sconv_scale)});
            {st}o[(3 + t) * DKV + c] = x3;
            if (t == 0) {{ {st}o[c] = {st}[c]; {st}o[DKV + c] = {st}[DKV + c]; {st}o[2 * DKV + c] = {st}[2 * DKV + c]; }}
"""
        kconv = conv(k_at, "ks", "kcv", "kv_")
        vconv = conv(v_at, "vs", "vcv", "vv_")
    if parts_only:
        return dict(
            inputs=inputs, q_at=q_at, kconv=kconv, vconv=vconv, o_q=o_q, o_r=o_r, o_s=o_s,
            qk_thr=qk_thr, r_thr=r_thr, hq=hq, hkv=hkv, d=d, dr=dr, r=r, dkv=dkv,
        )
    tau = ""
    tq = tr = ""
    if full:
        tau = f"float tau = 1.0f + {_f(attn.log_alpha)} * metal::log(max(float(dims[0] + t + 1) / {_f(attn.log_floor)}, 1.0f));"
        tq = "q_ = bf(q_) * tau;"
        tr = "rv = bf(rv) * tau;"
    src = f"""
    constexpr uint D = {d};
    constexpr uint DR = {dr};
    constexpr uint R = {r};
    constexpr uint DKV = {dkv};
    constexpr uint HQ = {hq};
    constexpr uint HKV = {hkv};
    constexpr uint L = {length};
    uint h = threadgroup_position_in_grid.x % HQ;
    uint t = threadgroup_position_in_grid.x / HQ;
    uint i = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    threadgroup float red[3][4];
    threadgroup float rr[DR];
    threadgroup float rsum;
    {tau}
    float q_ = float({q_at.format(o=f'{o_q} + h * D + i')}) * float(qsc[h * D + i]);
    float qs = simd_sum(q_ * q_);
    float rv = i < DR ? float({q_at.format(o=f'{o_r} + h * DR + i')}) * float(rsc[h * DR + i]) : 0.0f;
    float rs = simd_sum(rv * rv);
    if (lane == 0) red[0][sg] = qs;
    if (i == 0) rsum = rs;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float qn = metal::sqrt(red[0][0] + red[0][1] + red[0][2] + red[0][3]);
    float rn = metal::sqrt(rsum);
    q_ = bf(q_ * min({_f(qk_thr)} / (qn + 1e-6f), 1.0f));
    {tq}
    q[(t * HQ + h) * D + i] = bfloat(q_);
    if (i < DR) {{
        rv = bf(rv * min({_f(r_thr)} / (rn + 1e-6f), 1.0f));
        {tr}
        rr[i] = bf(rv);
    }}
    if (i == 0) sinks[t * HQ + h] = {q_at.format(o=f'{o_s} + h')};
    if (h < HKV) {{
        uint c = h * D + i;
        float kv_, vv_;
        {{
{kconv}
        }}
        {{
{vconv}
        }}
        float ks2 = simd_sum(kv_ * kv_);
        float vs2 = simd_sum(vv_ * vv_);
        if (lane == 0) {{ red[1][sg] = ks2; red[2][sg] = vs2; }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float kn = metal::sqrt(red[1][0] + red[1][1] + red[1][2] + red[1][3]);
        float vinv = metal::rsqrt((red[2][0] + red[2][1] + red[2][2] + red[2][3]) / float(D) + {_f(_EPS)});
        kout[(h * L + t) * D + i] = bfloat(kv_ * min({_f(qk_thr)} / (kn + 1e-6f), 1.0f));
        vout[(h * L + t) * D + i] = bfloat(vv_ * vinv);
    }} else {{
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Relative bias by backward distance (the reference's BF16 r @ rel_proj).
    for (uint dist = i; dist < R; dist += D) {{
        float b = 0.0f;
        for (uint u = 0; u < DR; ++u) b += rr[u] * float(relp[u * R + dist]);
        rel[(t * HQ + h) * R + dist] = bfloat(b);
    }}
"""
    return _kernel(key, inputs, ["q", "kout", "vout", "rel", "sinks", "kso", "vso"], src)


# --------------------------------------------------------------------------
# Attention: split-K over keys (one simdgroup per chunk, all G * L queries
# of a KV head), then a combine pass that adds the per-query sink and the
# sigmoid output gate.


# --------------------------------------------------------------------------
# Fused epilogue + attention: one threadgroup (8 simdgroups) per KV head.
# Phase A computes the convs/clips/norms for the KV head and its G query
# heads (writing k/v for the cache update, q and the relative-bias rows to
# scratch), phase B walks the keys (simdgroup s takes keys s, s + 8, ...;
# the newest L keys come from phase A), phase C merges the simdgroups, adds
# the per-query sink and applies the sigmoid output gate.

_FUSED_ATT_MAX_KEYS = int(os.environ.get("OMLX_INKLING2_FUSED_ATT_MAX_KEYS", "4096"))
_FA_SG = 8


def _attn_fused_kernel(attn, full: bool, msg: bool, length: int, gate_stride: int, gate_off: int):
    key = (id(attn), full, msg, length, gate_stride, gate_off)
    hit = _fa_cache.get(key)
    if hit is None:
        hit = _fa_cache[key] = _attn_fused_build(attn, full, msg, length, gate_stride, gate_off)
    return hit


_fa_cache: dict = {}


def _attn_fused_build(attn, full, msg, length, gate_stride, gate_off):
    P = _epi_kernel_build(attn, full, msg, length, parts_only=True)
    hq, hkv, d, dr, r, dkv = P["hq"], P["hkv"], P["d"], P["dr"], P["r"], P["dkv"]
    g = hq // hkv
    nq = g * length
    window = attn.window
    q_at = P["q_at"]
    gsrc = "pq" if full else "p"
    tau = tq = tr = ""
    if full:
        tau = f"float tau = 1.0f + {_f(attn.log_alpha)} * metal::log(max(float(dims[0] + t + 1) / {_f(attn.log_floor)}, 1.0f));"
        tq = "qv_ = bf(qv_) * tau;"
        tr = "rv = bf(rv) * tau;"
    src = f"""
    constexpr uint D = {d};
    constexpr uint DR = {dr};
    constexpr uint R = {r};
    constexpr uint DKV = {dkv};
    constexpr uint HQ = {hq};
    constexpr uint HKV = {hkv};
    constexpr uint G = {g};
    constexpr uint L = {length};
    constexpr uint NQ = {nq};
    constexpr uint NSG = {_FA_SG};
    constexpr int WINDOW = {window};
    constexpr uint PD = D / 32;
    uint kvh = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    int offset = dims[0];
    int lo = dims[1];
    int cap = dims[2];
    int S = offset + int(L);
    threadgroup float red[2 * L * 4 + 2 * NQ * 4];
    threadgroup float kvt[2][L][D];
    threadgroup float rr[NQ][DR];
    threadgroup float sinkv[NQ];
    threadgroup float cm[NSG][3][D + 2];

    // ---- Phase A1: k (threads 0..D-1) and v (D..2D-1) convs for all tokens.
    {{
        uint i = tid % D;
        uint h = kvh;
        uint c = h * D + i;
        bool isv = tid >= D;
        for (uint t = 0; t < L; ++t) {{
            float kv_ = 0.0f, vv_ = 0.0f;
            if (!isv) {{
{P["kconv"]}
                kvt[0][t][i] = kv_;
            }} else {{
{P["vconv"]}
                kvt[1][t][i] = vv_;
            }}
        }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {{
        uint i = tid % D;
        uint which = tid / D;
        for (uint t = 0; t < L; ++t) {{
            float x = kvt[which][t][i];
            float ss = simd_sum(x * x);
            if (lane == 0) red[(which * L + t) * 4 + (sg % 4)] = ss;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint t = 0; t < L; ++t) {{
            uint b = (which * L + t) * 4;
            float tot = red[b] + red[b + 1] + red[b + 2] + red[b + 3];
            float x = kvt[which][t][i];
            float o;
            if (which == 0) o = bf(x * min({_f(P["qk_thr"])} / (metal::sqrt(tot) + 1e-6f), 1.0f));
            else o = bf(x * metal::rsqrt(tot / float(D) + {_f(_EPS)}));
            kvt[which][t][i] = o;
            if (which == 0) kout[(kvh * L + t) * D + i] = bfloat(o);
            else vout[(kvh * L + t) * D + i] = bfloat(o);
        }}
    }}
    // ---- Phase A2: q clips (two (head, token) pairs per pass) and r clips.
    for (uint base = 0; base < NQ; base += 2) {{
        uint qi = base + tid / D;
        uint i = tid % D;
        uint gg = qi / L, t = qi % L;
        uint h = kvh * G + gg;
        float qv_ = 0.0f;
        if (qi < NQ) qv_ = float({q_at.format(o=f'{P["o_q"]} + h * D + i')}) * float(qsc[h * D + i]);
        float ss = simd_sum(qv_ * qv_);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0) red[2 * L * 4 + (tid / D) * 4 + (sg % 4)] = ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (qi < NQ) {{
            uint b = 2 * L * 4 + (tid / D) * 4;
            float qn = metal::sqrt(red[b] + red[b + 1] + red[b + 2] + red[b + 3]);
            {tau if full else ""}
            qv_ = bf(qv_ * min({_f(P["qk_thr"])} / (qn + 1e-6f), 1.0f));
            {tq}
            qs[(t * HQ + h) * D + i] = bfloat(qv_);
        }}
    }}
    for (uint w_ = tid; w_ < NQ * 16; w_ += NSG * 32) {{
        // 16-lane groups: (pair, u); DR == 16.
        uint qi = w_ / 16, u = w_ % 16;
        uint gg = qi / L, t = qi % L;
        uint h = kvh * G + gg;
        float rv = u < DR ? float({q_at.format(o=f'{P["o_r"]} + h * DR + u')}) * float(rsc[h * DR + u]) : 0.0f;
        float s2 = rv * rv;
        s2 += simd_shuffle_xor(s2, 8);
        s2 += simd_shuffle_xor(s2, 4);
        s2 += simd_shuffle_xor(s2, 2);
        s2 += simd_shuffle_xor(s2, 1);
        {tau if full else ""}
        rv = bf(rv * min({_f(P["r_thr"])} / (metal::sqrt(s2) + 1e-6f), 1.0f));
        {tr}
        if (u < DR) rr[qi][u] = bf(rv);
        if (u == 0) sinkv[qi] = float({q_at.format(o=f'{P["o_s"]} + h')});
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // ---- Phase A3: relative bias rows (BF16, distances below the key span).
    int span = min(int(R), S - lo);
    for (uint w_ = tid; w_ < NQ * uint(span); w_ += NSG * 32) {{
        uint qi = w_ / uint(span);
        uint dist = w_ % uint(span);
        uint gg = qi / L, t = qi % L;
        float b = 0.0f;
        for (uint u = 0; u < DR; ++u) b += rr[qi][u] * float(relp[u * R + dist]);
        rel[((t * HQ + kvh * G + gg)) * R + dist] = bfloat(b);
    }}
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    // ---- Phase B: keys [lo, S); the last L come from phase A.
    float qv[NQ][PD];
    float m[NQ], l[NQ], acc[NQ][PD];
    float scale = sc[0];
    for (uint qi = 0; qi < NQ; ++qi) {{
        uint gg = qi / L, t = qi % L;
        uint h = kvh * G + gg;
        for (uint c2 = 0; c2 < PD; ++c2) {{
            qv[qi][c2] = float(qs[(t * HQ + h) * D + lane * PD + c2]) * scale;
            acc[qi][c2] = 0.0f;
        }}
        m[qi] = -INFINITY;
        l[qi] = 0.0f;
    }}
    const device bfloat* kb = keys + (size_t)kvh * cap * D + lane * PD;
    const device bfloat* vb = values + (size_t)kvh * cap * D + lane * PD;
    for (int j = lo + int(sg); j < S; j += NSG) {{
        float kx[PD], vx[PD];
        if (j < offset) {{
            for (uint c2 = 0; c2 < PD; ++c2) {{ kx[c2] = float(kb[(size_t)j * D + c2]); vx[c2] = float(vb[(size_t)j * D + c2]); }}
        }} else {{
            uint tj = uint(j - offset);
            for (uint c2 = 0; c2 < PD; ++c2) {{ kx[c2] = kvt[0][tj][lane * PD + c2]; vx[c2] = kvt[1][tj][lane * PD + c2]; }}
        }}
        for (uint qi = 0; qi < NQ; ++qi) {{
            uint gg = qi / L, t = qi % L;
            int dist = (S - int(L) + int(t)) - j;
            bool vis = dist >= 0 && (WINDOW == 0 || dist < WINDOW);
            float s = 0.0f;
            for (uint c2 = 0; c2 < PD; ++c2) s += qv[qi][c2] * kx[c2];
            s = simd_sum(s);
            if (!vis) continue;
            if (dist < int(R)) s += float(rel[(t * HQ + kvh * G + gg) * R + dist]);
            float mn = max(m[qi], s);
            float f = metal::exp(m[qi] - mn);
            float pp = metal::exp(s - mn);
            l[qi] = l[qi] * f + pp;
            for (uint c2 = 0; c2 < PD; ++c2) acc[qi][c2] = acc[qi][c2] * f + pp * vx[c2];
            m[qi] = mn;
        }}
    }}
    // ---- Phase C: merge simdgroups three queries at a time, sink + gate.
    for (uint q0 = 0; q0 < NQ; q0 += 3) {{
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k3 = 0; k3 < 3; ++k3) {{
            uint qi = q0 + k3;
            if (qi >= NQ) break;
            for (uint c2 = 0; c2 < PD; ++c2) cm[sg][k3][lane * PD + c2] = acc[qi][c2];
            if (lane == 0) {{ cm[sg][k3][D] = m[qi]; cm[sg][k3][D + 1] = l[qi]; }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint w_ = tid; w_ < 3 * D; w_ += NSG * 32) {{
            uint k3 = w_ / D, i = w_ % D;
            uint qi = q0 + k3;
            if (qi >= NQ) continue;
            uint gg = qi / L, t = qi % L;
            uint h = kvh * G + gg;
            float sink = sinkv[qi];
            float M = sink;
            for (uint s_ = 0; s_ < NSG; ++s_) M = max(M, cm[s_][k3][D]);
            float Z = metal::exp(sink - M);
            float o = 0.0f;
            for (uint s_ = 0; s_ < NSG; ++s_) {{
                float wgt = metal::exp(cm[s_][k3][D] - M);
                Z += cm[s_][k3][D + 1] * wgt;
                o += cm[s_][k3][i] * wgt;
            }}
            float ob = bf(o / Z);
            float gt = float({gsrc}[t * {gate_stride} + {gate_off} + h * D + i]);
            out[t * (HQ * D) + h * D + i] = bfloat(ob * (1.0f / (1.0f + metal::exp(-gt))));
        }}
    }}
"""
    inputs = P["inputs"][:-1] + ["keys", "values", "dims", "sc"]
    outputs = ["out", "kout", "vout", "kso", "vso", "qs", "rel"]
    key = ("attf", full, msg, length, hq, hkv, d, r, window, gate_stride, gate_off, attn.log_floor)
    return _kernel(key, inputs, outputs, src)


_ATT_MAX_SPLIT = 64
_ATT_MIN_CHUNK = 32


@_memo
def _attn_pass1(hq, hkv, d, r, window, length):
    g = hq // hkv
    key = ("att1", hq, hkv, d, r, window, length)
    nqry = g * length
    src = f"""
    constexpr uint D = {d};
    constexpr uint HQ = {hq};
    constexpr uint G = {g};
    constexpr uint L = {length};
    constexpr uint NQ = {nqry};
    constexpr uint R = {r};
    constexpr int WINDOW = {window};
    constexpr uint PD = D / 32;
    uint split = threadgroup_position_in_grid.x;
    uint kvh = threadgroup_position_in_grid.y;
    uint nsplit = uint(dims[3]);
    uint lane = thread_index_in_simdgroup;
    int S = dims[0];
    int lo = dims[1];
    int cap = dims[2];
    float scale = sc[0];
    int nk = S - lo;
    int per = (nk + int(nsplit) - 1) / int(nsplit);
    int k0 = lo + int(split) * per;
    int k1 = min(S, k0 + per);
    float qv[NQ][PD];
    float m[NQ], l[NQ], acc[NQ][PD];
    for (uint qi = 0; qi < NQ; ++qi) {{
        uint gg = qi / L, t = qi % L;
        uint h = kvh * G + gg;
        for (uint c = 0; c < PD; ++c) {{
            qv[qi][c] = float(q[(t * HQ + h) * D + lane * PD + c]) * scale;
            acc[qi][c] = 0.0f;
        }}
        m[qi] = -INFINITY;
        l[qi] = 0.0f;
    }}
    const device bfloat* kb = keys + (size_t)kvh * cap * D + lane * PD;
    const device bfloat* vb = values + (size_t)kvh * cap * D + lane * PD;
    for (int j = k0; j < k1; ++j) {{
        float kx[PD], vx[PD];
        for (uint c = 0; c < PD; ++c) {{ kx[c] = float(kb[(size_t)j * D + c]); vx[c] = float(vb[(size_t)j * D + c]); }}
        for (uint qi = 0; qi < NQ; ++qi) {{
            uint gg = qi / L, t = qi % L;
            int dist = (S - int(L) + int(t)) - j;
            bool vis = dist >= 0 && (WINDOW == 0 || dist < WINDOW);
            float s = 0.0f;
            for (uint c = 0; c < PD; ++c) s += qv[qi][c] * kx[c];
            s = simd_sum(s);
            if (!vis) continue;
            if (dist < int(R)) s += float(rel[(t * HQ + kvh * G + gg) * R + dist]);
            float mn = max(m[qi], s);
            float f = metal::exp(m[qi] - mn);
            float p = metal::exp(s - mn);
            l[qi] = l[qi] * f + p;
            for (uint c = 0; c < PD; ++c) acc[qi][c] = acc[qi][c] * f + p * vx[c];
            m[qi] = mn;
        }}
    }}
    for (uint qi = 0; qi < NQ; ++qi) {{
        uint gg = qi / L, t = qi % L;
        uint row = (t * HQ + kvh * G + gg) * nsplit + split;
        for (uint c = 0; c < PD; ++c) pacc[row * D + lane * PD + c] = acc[qi][c];
        if (lane == 0) {{ pm[row] = m[qi]; pl[row] = l[qi]; }}
    }}
"""
    return _kernel(key, ["q", "keys", "values", "rel", "dims", "sc"], ["pacc", "pm", "pl"], src)


@_memo
def _attn_pass2(hq, d, gate_stride, gate_off, length):
    key = ("att2", hq, d, gate_stride, gate_off, length)
    src = f"""
    constexpr uint D = {d};
    constexpr uint HQ = {hq};
    uint qrow = threadgroup_position_in_grid.x;   // t * HQ + h
    uint t = qrow / HQ, h = qrow % HQ;
    uint i = thread_position_in_threadgroup.x;
    uint ns = nsplit[0];
    float sink = float(sinks[qrow]);
    float M = sink;
    for (uint s = 0; s < ns; ++s) M = max(M, pm[qrow * ns + s]);
    float Z = metal::exp(sink - M);
    float o = 0.0f;
    for (uint s = 0; s < ns; ++s) {{
        float w = metal::exp(pm[qrow * ns + s] - M);
        Z += pl[qrow * ns + s] * w;
        o += pacc[(qrow * ns + s) * D + i] * w;
    }}
    float ob = bf(o / Z);
    float gt = float(gate[t * {gate_stride} + {gate_off} + h * D + i]);
    out[t * (HQ * D) + h * D + i] = bfloat(ob * (1.0f / (1.0f + metal::exp(-gt))));
"""
    return _kernel(key, ["pacc", "pm", "pl", "sinks", "gate", "nsplit"], ["out"], src)


# --------------------------------------------------------------------------
# Attention epilogue: output conv, residual, MLP depth mix, MLP norm.


@_memo
def _post_kernel(h, length, nb, boundary, msg):
    """Grid: L threadgroups. Outputs res (FP32 [L, H]), the conv window
    s2xp (FP32 [3 + L, H]) and y (BF16 [L, H]) + y32, or with messages the
    raw mixture y (FP32 [L, H]) + z (FP32 [1, 2, L, H])."""
    key = ("post", h, length, nb, boundary, msg, _TG)
    blocks = [f"b{j}" for j in range(nb)]
    if msg:
        inputs = ["a", "aw", "s2", "hh", *blocks, "query", "m", "moff", "scale", "cscale"]
        outputs = ["res", "s2o", "y", "z"]
    else:
        inputs = ["a", "aw", "s2", "hh", *blocks, "query", "w", "scale", "cscale"]
        outputs = ["res", "s2o", "y", "y32"]
    resid = "float(a2)" if boundary else "float(hh[row * H + i]) + float(a2)"

    def xin(j):
        return f"(row + {j} < 3 ? s2[(row + {j}) * H + i] : float(a[(row + {j} - 3) * H + i]))"

    src = [_ROW_PRELUDE.format(TG=_TG, H=h, L=length), "    uint row = threadgroup_position_in_grid.x;"]
    src.append(
        f"""
    float rr_[EPT];
    for (uint e = 0; e < EPT; ++e) {{
        uint i = tid + e * TG;
        rr_[e] = 0.0f;
        if (i < H) {{
            float x0 = {xin(0)}, x1 = {xin(1)}, x2 = {xin(2)}, x3 = {xin(3)};
            float o = float(aw[i * 4 + 0]) * x0;
            o = o + float(aw[i * 4 + 1]) * x1;
            o = o + float(aw[i * 4 + 2]) * x2;
            o = o + float(aw[i * 4 + 3]) * x3;
            o = bf(o);
            bfloat a2 = bfloat((o + x3) * cscale[0]);
            float rv = {resid};
            rr_[e] = rv;
            res[row * H + i] = rv;
            s2o[(3 + row) * H + i] = x3;
            if (row == 0) {{ s2o[i] = s2[i]; s2o[H + i] = s2[H + i]; s2o[2 * H + i] = s2[2 * H + i]; }}
        }}
    }}
"""
    )
    src.append(_unit_rms_code("rr_", "pn", "u"))
    src.append(_load_rows(blocks, "b"))
    src.append(_mix_code([f"b{j}r" for j in range(nb)] + ["pn"], 1, "query", "scale[0]", "yo", "a"))
    if msg:
        src.append(
            "    float mr[EPT];\n"
            "    for (uint e = 0; e < EPT; ++e) { uint i = tid + e * TG; mr[e] = i < H ? m[(moff[0] * L + row) * H + i] : 0.0f;"
            " if (i < H) y[row * H + i] = yo[0][e]; }"
        )
        src.append(_store_z("yo[0]", "z", "row * H", "zx"))
        src.append(_store_z("mr", "z", "(L + row) * H", "zm"))
    else:
        src.append(_norm_store("yo[0]", "w", "y", "row * H", "n", "y32"))
    return _kernel(key, inputs, outputs, "\n".join(src))


# --------------------------------------------------------------------------
# Experts: fused 4-bit matvecs (NVFP4 group 16 or affine group 64).
#
# moe_up: one threadgroup covers 32 consecutive w13 rows (16 gate/up pairs)
# of one unique expert and computes them for every token routed to it, so
# a verify step reads each selected expert once.
# moe_down: one threadgroup covers 4 output rows for all tokens; its
# simdgroups walk the unique experts, then it combines, convolves (over the
# token window) and adds the residual.

_MOE_SG = 8
_MOE_ROWS = 4
_MOE_DOWN_ROWS = int(os.environ.get("OMLX_INKLING2_DOWN_ROWS", "4"))


def _group_dot(mode, row, k_words, k_groups, xv="xv", xsum="xsum"):
    if mode == "nvfp4":
        return f"""(e4m3(uint(wsc[(size_t)({row}) * {k_groups} + g])) * fp4_dot16(*(const device uint2*)(wq + (size_t)({row}) * {k_words} + g * 2), {xv}))"""
    return f"""(float(wsc[(size_t)({row}) * {k_groups // 4} + g / 4]) * u4_dot16(*(const device uint2*)(wq + (size_t)({row}) * {k_words} + g * 2), {xv}) + float(wbi[(size_t)({row}) * {k_groups // 4} + g / 4]) * {xsum})"""


_UNIQ = """
    uint nuniq = nu[0];
"""


@_memo
def _moe_up_kernel(h, inter, k, length, mode, gcap, ucap):
    """act[t * K + j, :] for slot j of token t. Grid: LK * TPE threadgroups
    (threadgroups past the unique-expert count exit)."""
    key = ("moe_up", h, inter, k, length, mode, gcap, ucap)
    k_words, k_groups = h // 8, h // 16
    rows_tg = _MOE_SG * _MOE_ROWS
    tpe = (2 * inter) // rows_tg
    scale = (
        "float s = bf(gs[ex] * (1.0f / 2688.0f)); gv = bf(gv * s); uv = bf(uv * s);"
        if mode == "nvfp4"
        else ""
    )
    src = f"""
    constexpr uint H = {h};
    constexpr uint I = {inter};
    constexpr uint NG = {k_groups};
    constexpr uint ROWS = {_MOE_ROWS};
    constexpr uint L = {length};
    constexpr uint LK = {length * k};
    uint tg = threadgroup_position_in_grid.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
{_UNIQ}
    uint u = tg / {tpe};
    if (u >= nuniq) return;
    uint ex = uexp[u];
    uint row0 = (tg % {tpe}) * {rows_tg} + sg * ROWS;
    const size_t ebase = (size_t)ex * (2 * I);
    // Tokens routed to this expert.
    uint toks[L];
    uint slots[L];
    uint nt = 0;
    for (uint a = 0; a < LK; ++a) if (idx[a] == ex) {{ toks[nt] = a / {k}; slots[nt] = a; nt++; }}
    float acc[L][ROWS];
    for (uint tt = 0; tt < L; ++tt) for (uint r = 0; r < ROWS; ++r) acc[tt][r] = 0.0f;
    for (uint g = lane; g < NG; g += 32) {{
        uint2 wv[ROWS];
        float sc_[ROWS];
        float bi_[ROWS];
        for (uint r = 0; r < ROWS; ++r) {{
            size_t rw = ebase + row0 + r;
            wv[r] = *(const device uint2*)(wq + rw * {k_words} + g * 2);
            {"sc_[r] = e4m3(uint(wsc[rw * " + str(k_groups) + " + g])); bi_[r] = 0.0f;" if mode == "nvfp4" else "sc_[r] = float(wsc[rw * " + str(k_groups // 4) + " + g / 4]); bi_[r] = float(wbi[rw * " + str(k_groups // 4) + " + g / 4]);"}
        }}
        for (uint tt = 0; tt < nt; ++tt) {{
            float xv[16];
            float xsum = 0.0f;
            const device bfloat* xp = x + toks[tt] * H + g * 16;
            for (int c = 0; c < 16; ++c) {{ xv[c] = float(xp[c]); xsum += xv[c]; }}
            for (uint r = 0; r < ROWS; ++r) {{
                {"acc[tt][r] += sc_[r] * fp4_dot16(wv[r], xv);" if mode == "nvfp4" else "acc[tt][r] += sc_[r] * u4_dot16(wv[r], xv) + bi_[r] * xsum;"}
            }}
        }}
    }}
    for (uint tt = 0; tt < nt; ++tt) {{
        float d0 = simd_sum(acc[tt][0]);
        float d1 = simd_sum(acc[tt][1]);
        float d2 = simd_sum(acc[tt][2]);
        float d3 = simd_sum(acc[tt][3]);
        if (lane == 0) {{
            float dd[4] = {{d0, d1, d2, d3}};
            for (uint r = 0; r < ROWS; r += 2) {{
                float gv = bf(dd[r]);
                float uv = bf(dd[r + 1]);
                {scale}
                float o = metal::precise::tanh(gv / {_f(gcap)}) * {_f(gcap)} * (1.0f / (1.0f + metal::precise::exp(-gv)))
                    * (metal::precise::tanh(uv / {_f(ucap)}) * {_f(ucap)});
                act[slots[tt] * I + (row0 + r) / 2] = bfloat(o);
            }}
        }}
    }}
"""
    inputs = ["x", "idx", "uexp", "nu", "wq", "wsc"] + (["gs"] if mode == "nvfp4" else ["wbi"])
    return _kernel(key, inputs, ["act"], src), length * k * tpe


@_memo
def _moe_down_kernel(h, inter, k, length, mode):
    key = ("moe_down", h, inter, k, length, mode, _MOE_DOWN_ROWS)
    k_words, k_groups = inter // 8, inter // 16
    lk = length * k
    wexpr = "wt[a] * (gs[idx[a]] * (1.0f / 2688.0f))" if mode == "nvfp4" else "wt[a]"
    src = f"""
    constexpr uint H = {h};
    constexpr uint I = {inter};
    constexpr uint NG = {k_groups};
    constexpr uint ROWS = {_MOE_DOWN_ROWS};
    constexpr uint L = {length};
    constexpr uint KE = {k};
    constexpr uint LK = {lk};
    uint tg = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint row0 = tg * ROWS;
{_UNIQ}
    threadgroup float dots[LK][ROWS];
    for (uint u = sg; u < nuniq; u += {_MOE_SG}) {{
        uint ex = uexp[u];
        const size_t ebase = (size_t)ex * H;
        for (uint a = 0; a < LK; ++a) {{
            if (idx[a] != ex) continue;
            const device bfloat* av = act + a * I;
            float acc[ROWS] = {{0.0f}};
            for (uint g = lane; g < NG; g += 32) {{
                float xv[16];
                float xsum = 0.0f;
                for (int c = 0; c < 16; ++c) {{ xv[c] = float(av[g * 16 + c]); xsum += xv[c]; }}
                        for (uint r = 0; r < ROWS; ++r) acc[r] += {_group_dot(mode, "ebase + row0 + r", k_words, k_groups)};
            }}
            for (uint r = 0; r < ROWS; ++r) {{
                float tsum = simd_sum(acc[r]);
                if (lane == 0) dots[a][r] = bf(tsum);
            }}
        }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < ROWS) {{
        uint i = row0 + tid;
        float xr[L];
        for (uint t = 0; t < L; ++t) {{
            float s = 0.0f;
            for (uint j = 0; j < KE; ++j) {{ uint a = t * KE + j; s += dots[a][tid] * ({wexpr}); }}
            xr[t] = bf(s);
        }}
        for (uint t = 0; t < L; ++t) {{
            float x0 = t + 0 < 3 ? s3[(t + 0) * H + i] : xr[t + 0 - 3];
            float x1 = t + 1 < 3 ? s3[(t + 1) * H + i] : xr[t + 1 - 3];
            float x2 = t + 2 < 3 ? s3[(t + 2) * H + i] : xr[t + 2 - 3];
            float x3 = xr[t];
            float o = float(mw[i * 4 + 0]) * x0;
            o = o + float(mw[i * 4 + 1]) * x1;
            o = o + float(mw[i * 4 + 2]) * x2;
            o = o + float(mw[i * 4 + 3]) * x3;
            o = bf(o);
            float m2 = bf((o + x3) * cscale[0]);
            hout[t * H + i] = bfloat(res[t * H + i] + m2);
            s3o[(3 + t) * H + i] = x3;
        }}
        s3o[i] = s3[i]; s3o[H + i] = s3[H + i]; s3o[2 * H + i] = s3[2 * H + i];
    }}
"""
    inputs = ["act", "idx", "uexp", "nu", "wt", "wq", "wsc"] + (["gs"] if mode == "nvfp4" else ["wbi"])
    inputs += ["s3", "mw", "res", "cscale"]
    return _kernel(key, inputs, ["hout", "s3o"], src), h // _MOE_DOWN_ROWS


# --------------------------------------------------------------------------
# Router logits: one simdgroup per expert row, all L tokens.


@_memo
def _router_kernel(h, e, length):
    key = ("router", h, e, length)
    src = f"""
    constexpr uint H = {h};
    constexpr uint L = {length};
    uint lane = thread_index_in_simdgroup;
    uint ex = threadgroup_position_in_grid.x * 8 + simdgroup_index_in_threadgroup;
    const device float4* wr = (const device float4*)(w + (size_t)ex * H);
    float acc[L];
    for (uint t = 0; t < L; ++t) acc[t] = 0.0f;
    for (uint i = lane; i < H / 4; i += 32) {{
        float4 wv = wr[i];
        for (uint t = 0; t < L; ++t) {{
            float4 xv = ((const device float4*)(y + t * H))[i];
            acc[t] += dot(wv, xv);
        }}
    }}
    for (uint t = 0; t < L; ++t) {{
        float s = simd_sum(acc[t]);
        if (lane == 0) logits[t * {e} + ex] = s;
    }}
"""
    return _kernel(key, ["y", "w"], ["logits"], src)


@_memo
def _topk_kernel(e, k, length):
    """Sigmoid top-k routing (one simdgroup per row) plus the unique expert
    list of all rows. Outputs inds [L, K] uint32, weights [L, K] FP32,
    uexp [L * K] uint32 and nuniq [1] uint32."""
    key = ("topk", e, k, length)
    src = f"""
    constexpr uint E = {e};
    constexpr uint K = {k};
    constexpr uint L = {length};
    constexpr uint LK = L * K;
    constexpr uint PER = E / 32;
    uint row = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    threadgroup uint chosen_all[LK];
    if (row < L) {{
        float sc[PER];
        float key_[PER];
        for (uint j = 0; j < PER; ++j) {{
            uint ex = lane + 32 * j;
            float s = 1.0f / (1.0f + metal::exp(-logits[row * E + ex]));
            sc[j] = s;
            key_[j] = s + bias[ex];
        }}
        float picked[K];
        uint chosen[K];
        float total = 0.0f;
        for (uint kk = 0; kk < K; ++kk) {{
            float best = -INFINITY;
            uint bj = 0;
            for (uint j = 0; j < PER; ++j) {{
                if (key_[j] > best) {{ best = key_[j]; bj = j; }}
            }}
            float top = simd_max(best);
            uint mine = (best == top) ? (lane + 32 * bj) : 0xFFFFFFFFu;
            uint win = simd_min(mine);
            float s = (lane == win % 32) ? sc[win / 32] : 0.0f;
            s = simd_sum(s);
            if (lane == win % 32) key_[win / 32] = -INFINITY;
            picked[kk] = s;
            chosen[kk] = win;
            total += s;
        }}
        if (lane == 0) {{
            float norm = route_scale[0] / (total + 1e-20f);
            for (uint kk = 0; kk < K; ++kk) {{
                inds[row * K + kk] = chosen[kk];
                weights[row * K + kk] = picked[kk] * norm;
                chosen_all[row * K + kk] = chosen[kk];
            }}
        }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_position_in_threadgroup.x == 0) {{
        uint n = 0;
        for (uint a = 0; a < LK; ++a) {{
            uint ea = chosen_all[a];
            bool seen = false;
            for (uint b = 0; b < n; ++b) seen = seen || (uexp[b] == ea);
            if (!seen) uexp[n++] = ea;
        }}
        for (uint b = n; b < LK; ++b) uexp[b] = 0xFFFFFFFFu;
        nuniq[0] = n;
    }}
"""
    return _kernel(key, ["logits", "bias", "route_scale"], ["inds", "weights", "uexp", "nuniq"], src)


def _fused_moe_ok(ex) -> bool:
    for p in (ex.w13, ex.w2):
        if p.bits != 4:
            return False
        if not ((p.mode == "nvfp4" and p.group_size == 16) or (p.mode == "affine" and p.group_size == 64)):
            return False
    return True


# --------------------------------------------------------------------------


def _prep(layer):
    """Per-layer constants (BF16 copies of the BF16-exact controller weights)."""
    c = getattr(layer, "_fd", None)
    if c is not None:
        return c
    c = {
        "scale": mx.array([layer.mlp_input.scale], dtype=mx.float32),
        "cscale": mx.array([layer._conv_scale], dtype=mx.float32),
        "ascale": mx.array([layer.attn.scale], dtype=mx.float32),
        "moe_ok": _fused_moe_ok(layer.mlp.experts),
        "rscale": mx.array([layer.mlp.route_scale], dtype=mx.float32),
    }
    ex = layer.mlp.experts
    c["gcap"] = float(ex._gcap.item())
    c["ucap"] = float(ex._ucap.item())
    if hasattr(layer, "loop_down"):
        down = layer.loop_down.astype(mx.bfloat16)
        gf = layer.loop_gain_forget.astype(mx.bfloat16)
        c["down_pre"] = mx.contiguous(down[:3])
        c["down_post"] = mx.contiguous(down[3:])
        c["gf_pre"] = mx.contiguous(gf[:3])
        c["gf_post"] = mx.contiguous(gf[3:])
        mx.eval(c["down_pre"], c["down_post"], c["gf_pre"], c["gf_post"])
        c["moff0"] = mx.array([0], dtype=mx.int32)
        c["moff3"] = mx.array([3], dtype=mx.int32)
    layer._fd = c
    return c


def _controller(layer, c, z, x, m, moff, n, length, post):
    hidden = x.shape[-1]
    down = c["down_post"] if post else c["down_pre"]
    gf = c["gf_post"] if post else c["gf_pre"]
    r = down.shape[-1]
    part = _ctl_low_kernel(hidden, r, length)(
        inputs=[z, down],
        grid=(n * length * _CTL_SPLIT * _CTL_TG, 1, 1),
        threadgroup=(_CTL_TG, 1, 1),
        output_shapes=[(n, length, _CTL_SPLIT, 2, r)],
        output_dtypes=[mx.float32],
    )[0]
    if post:
        norms = [layer.mlp_norm.weight]
    elif layer.sliding:
        norms = [layer.attn_norm.weight]
    else:
        norms = [layer.q_input_norm.weight, layer.k_input_norm.weight, layer.v_input_norm.weight]
    xs = x.shape[0] == 1
    return _ctl_mix_kernel(hidden, r, n, length, len(norms), post, xs)(
        inputs=[part, x, m, moff, gf, *norms],
        grid=(n * length * _TG, 1, 1),
        threadgroup=(_TG, 1, 1),
        output_shapes=[(n, length, hidden)] * (2 if post else 1),
        output_dtypes=[mx.bfloat16, mx.float32] if post else [mx.bfloat16],
    )


def fast_visit(layer, h, blocks, cache, bank):
    """One layer visit for B == 1, L <= _MAX_L tokens."""
    kv, conv = cache[0], cache[1]
    attn = layer.attn
    _, length, hidden = h.shape
    c = _prep(layer)
    states = [conv[i] for i in range(4)]
    nb = len(blocks)
    msg = bank is not None
    scale = c["scale"]
    nq = 1 if layer.sliding else 3
    row_grid = (length * _TG, 1, 1)
    row_tg = (_TG, 1, 1)
    f32, b16 = mx.float32, mx.bfloat16

    # 1. attention inputs
    if not msg:
        norms = (
            [layer.attn_norm.weight]
            if layer.sliding
            else [layer.q_input_norm.weight, layer.k_input_norm.weight, layer.v_input_norm.weight]
        )
        hn, x = _pre_kernel(hidden, length, nb, nq, 0)(
            inputs=[h, *blocks, layer.attn_input.query if nb else scale, *norms, scale],
            grid=row_grid,
            threadgroup=row_tg,
            output_shapes=[(1, length, hidden), (nq, length, hidden)],
            output_dtypes=[f32, b16],
        )
        m = None
    else:
        bblocks, bpart = bank
        hn, xr, m, z = _pre_kernel(hidden, length, nb, nq, len(bblocks))(
            inputs=[h, *blocks, layer.attn_input.query, *bblocks, bpart, layer.loop_message.query, scale],
            grid=row_grid,
            threadgroup=row_tg,
            output_shapes=[(1, length, hidden), (nq, length, hidden), (4, length, hidden), (3, 2, length, hidden)],
            output_dtypes=[f32] * 4,
        )
        x = _controller(layer, c, z, xr, m, c["moff0"], 3, length, False)[0]

    # 2. projections + epilogue
    hq, hkv, d = attn.n_heads, attn.n_kv, attn.head_dim
    r = attn.rel_extent
    offset = kv.offset
    epi_shapes = [(length, hq, d), (1, hkv, length, d), (1, hkv, length, d), (length, hq, r), (length, hq), (3 + length, attn.dkv), (3 + length, attn.dkv)]
    epi_dtypes = [b16] * 5 + [f32] * 2
    window = attn.window
    s_total = offset + length
    lo = max(0, s_total - length - window + 1) if window else 0
    nk = s_total - lo
    if layer.sliding:
        p = attn.in_proj(x)  # [L, W] or [3, L, W]
        gate_src, gate_stride, gate_off = p, p.shape[-1], attn._cuts[3]
        epi_in = [p, attn.k_conv, attn.v_conv, states[0], states[1], attn.q_scale, attn.r_scale, attn.rel_proj]
    else:
        pq = attn.q_proj(x[0])
        pk = attn.k_proj(x[1])
        pv = attn.v_proj(x[2])
        gate_src, gate_stride, gate_off = pq, pq.shape[-1], attn._cuts[1]
        epi_in = [pq, pk, pv, attn.k_static, attn.v_static, states[0], states[1], attn.q_scale, attn.r_scale, attn.rel_proj]
    if nk <= _FUSED_ATT_MAX_KEYS:
        cap = kv.keys.shape[2]
        fa = _attn_fused_kernel(attn, not layer.sliding, msg, length, gate_stride, gate_off)
        ao, k_new, v_new, kxp, vxp, _, _ = fa(
            inputs=[*epi_in, kv.keys, kv.values, mx.array([offset, lo, cap], dtype=mx.int32), c["ascale"]],
            grid=(hkv * _FA_SG * 32, 1, 1),
            threadgroup=(_FA_SG * 32, 1, 1),
            output_shapes=[(length, hq * d), (1, hkv, length, d), (1, hkv, length, d), (3 + length, attn.dkv), (3 + length, attn.dkv), (length, hq, d), (length, hq, r)],
            output_dtypes=[b16, b16, b16, f32, f32, b16, b16],
        )
        kv.update_and_fetch(k_new, v_new)
    else:
        dims = mx.array([offset], dtype=mx.int32)
        outs = _epi_kernel(attn, not layer.sliding, msg, length)(
            inputs=[*epi_in, dims],
            grid=(hq * length * d, 1, 1),
            threadgroup=(d, 1, 1),
            output_shapes=epi_shapes,
            output_dtypes=epi_dtypes,
        )
        q, k_new, v_new, rel, sinks, kxp, vxp = outs
        kv.update_and_fetch(k_new, v_new)
        nsplit = max(1, min(_ATT_MAX_SPLIT, (nk + _ATT_MIN_CHUNK - 1) // _ATT_MIN_CHUNK))
        cap = kv.keys.shape[2]
        adims = mx.array([s_total, lo, cap, nsplit], dtype=mx.int32)
        pacc, pm, pl = _attn_pass1(hq, hkv, d, r, window, length)(
            inputs=[q, kv.keys, kv.values, rel, adims, c["ascale"]],
            grid=(nsplit * 32, hkv, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[(length * hq * nsplit, d), (length * hq * nsplit,), (length * hq * nsplit,)],
            output_dtypes=[f32] * 3,
        )
        ao = _attn_pass2(hq, d, gate_stride, gate_off, length)(
            inputs=[pacc, pm, pl, sinks, gate_src, mx.array([nsplit], dtype=mx.uint32)],
            grid=(length * hq * d, 1, 1),
            threadgroup=(d, 1, 1),
            output_shapes=[(length, hq * d)],
            output_dtypes=[b16],
        )[0]
    a = attn.o_proj(ao)

    # 3. attention epilogue + MLP input
    if layer.is_boundary:
        blocks.append(hn)
    post_in = [a, layer.attn_conv, states[2], h, *blocks, layer.mlp_input.query]
    if not msg:
        res, s2xp, y, y32 = _post_kernel(hidden, length, len(blocks), layer.is_boundary, False)(
            inputs=[*post_in, layer.mlp_norm.weight, scale, c["cscale"]],
            grid=row_grid,
            threadgroup=row_tg,
            output_shapes=[(length, hidden), (3 + length, hidden), (length, hidden), (length, hidden)],
            output_dtypes=[f32, f32, b16, f32],
        )
    else:
        res, s2xp, yr, z = _post_kernel(hidden, length, len(blocks), layer.is_boundary, True)(
            inputs=[*post_in, m, c["moff3"], scale, c["cscale"]],
            grid=row_grid,
            threadgroup=row_tg,
            output_shapes=[(length, hidden), (3 + length, hidden), (length, hidden), (1, 2, length, hidden)],
            output_dtypes=[f32] * 4,
        )
        y, y32 = _controller(layer, c, z, yr.reshape(1, length, hidden), m, c["moff3"], 1, length, True)

    # 4. router + experts
    moe = layer.mlp
    ex = moe.experts
    n_exp = moe.router_weight.shape[0]
    if length == 1:
        logits = y32.reshape(1, hidden) @ moe.router_weight.T
    else:
        logits = _router_kernel(hidden, n_exp, length)(
            inputs=[y32, moe.router_weight],
            grid=(n_exp * 32, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(length, n_exp)],
            output_dtypes=[f32],
        )[0]
    inds, w, uexp, nuniq = _topk_kernel(n_exp, moe.top_k, length)(
        inputs=[logits, moe.router_bias, c["rscale"]],
        grid=(32 * length, 1, 1),
        threadgroup=(32 * length, 1, 1),
        output_shapes=[(length, moe.top_k), (length, moe.top_k), (length * moe.top_k,), (1,)],
        output_dtypes=[mx.uint32, f32, mx.uint32, mx.uint32],
    )
    topk = moe.top_k
    inter = ex.w2.weight.shape[-1] * (32 // ex.w2.bits)
    if c["moe_ok"]:
        up, n_up = _moe_up_kernel(hidden, inter, topk, length, ex.w13.mode, c["gcap"], c["ucap"])
        w13 = ex.w13
        flat = inds.reshape(-1)
        act = up(
            inputs=[y, flat, uexp, nuniq, w13.weight, w13.scales, w13.global_scale if w13.mode == "nvfp4" else w13.biases],
            grid=(n_up * _MOE_SG * 32, 1, 1),
            threadgroup=(_MOE_SG * 32, 1, 1),
            output_shapes=[(length * topk, inter)],
            output_dtypes=[b16],
        )[0]
        dn, n_dn = _moe_down_kernel(hidden, inter, topk, length, ex.w2.mode)
        w2p = ex.w2
        hout, s3xp = dn(
            inputs=[
                act,
                flat,
                uexp,
                nuniq,
                w.reshape(-1),
                w2p.weight,
                w2p.scales,
                w2p.global_scale if w2p.mode == "nvfp4" else w2p.biases,
                states[3],
                layer.mlp_conv,
                res,
                c["cscale"],
            ],
            grid=(n_dn * _MOE_SG * 32, 1, 1),
            threadgroup=(_MOE_SG * 32, 1, 1),
            output_shapes=[(1, length, hidden), (3 + length, hidden)],
            output_dtypes=[h.dtype, f32],
        )
    else:
        raise RuntimeError("unsupported expert carrier for the fused path")

    windows = (kxp, vxp, s2xp, s3xp)
    for slot, win in enumerate(windows):
        conv[slot] = win[length:].reshape(1, 3, -1)
    if length > 1:
        conv._omlx_verify_xp = {slot: win[None] for slot, win in enumerate(windows)}
    return hout


# --------------------------------------------------------------------------

_stats: dict = {}
_TIMING = os.environ.get("OMLX_INKLING2_TIMING", "0") not in ("0", "false")
_timing: dict = {}
_timing_n = [0]


def record_timing(n, wait, fwd):
    """Debug: synchronized forward time per input length (and the wait for
    GPU work queued before the forward, e.g. drafting)."""
    t = _timing.setdefault(n, [0, 0.0, 0.0])
    t[0] += 1
    t[1] += wait
    t[2] += fwd
    _timing_n[0] += 1
    if _timing_n[0] % 300 == 0:
        logger.info(
            "inkling2 forward timing: %s",
            " ".join(f"L{k}:n={v[0]},fwd={1000 * v[2] / v[0]:.1f}ms,wait={1000 * v[1] / v[0]:.1f}ms" for k, v in sorted(_timing.items())),
        )


def _why_not(layer, h, cache):
    if not _ENABLED:
        return "disabled"
    if cache is None:
        return "no-cache"
    if layer.layer_idx == 0 and (not layer.sliding or hasattr(layer, "loop_down")):
        return "layer0"
    if h.shape[0] != 1 or h.dtype != mx.bfloat16:
        return f"rows={h.shape[0]},{h.dtype}"
    if h.shape[1] > _MAX_L:
        return f"len>{_MAX_L}"
    kv, conv = cache[0], cache[1]
    if type(kv) is not KVCache:
        return f"cache={type(kv).__name__}"
    if kv.keys is None:
        return "empty-kv"
    if any(conv[i] is None for i in range(4)):
        return "no-conv-state"
    if not _prep(layer)["moe_ok"]:
        return "expert-format"
    if mx.default_device() != mx.gpu:
        return "cpu"
    return None


def eligible(layer, h, cache) -> bool:
    why = _why_not(layer, h, cache)
    if _DEBUG:
        k = why or f"fast-L{h.shape[1]}"
        _stats[k] = _stats.get(k, 0) + 1
        if sum(_stats.values()) % 4800 == 0:
            logger.info("inkling2 fast decode visits: %s", dict(sorted(_stats.items())))
    return why is None
