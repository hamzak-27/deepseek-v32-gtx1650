"""v2: dp4a kernels for Turing.

Why: Triton's int8 tl.dot on sm_75 has no fast path. Every v0/v1 kernel measured ~1.28 TOP/s on every shape,
the scalar integer rate, while cuBLAS INT8 reaches ~11.8 TOP/s = 512 ops / clock / SM on the GTX 1650 =
the dp4a peak. dp4a does 4 int8 products plus an add in one instruction. These kernels call it through inline
PTX, as outer products over K-major operands: one int32 word holds 4 consecutive K values, laid out
[K / 4, rows] so each step's loads are contiguous.

Measured on a plain 2048^3 dp4a GEMM: row-major operands 2.0 TOP/s, K-major 6.7 TOP/s (cuBLAS 9.3). The sweet
spot is ~128 int32 accumulators per thread; 256 spills registers and runs ~95x slower.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from kernels import GROUP, _bf16_bits_to_f32, _f32_to_bf16_bits


@triton.jit
def _dp4a(a, b, c):
    return tl.inline_asm_elementwise("dp4a.s32.s32 $0, $1, $2, $3;", "=r,r,r,r", [a, b, c],
                                     dtype=tl.int32, is_pure=True, pack=1)


# --------------------------------------------------------------------------------------------------------------
# act_quant writing the K-major packed layout
# --------------------------------------------------------------------------------------------------------------
@triton.jit
def _act_quant_kmajor_kernel(x_ptr, yT_ptr, s_ptr, M, N, NG,
                             BLOCK_M: tl.constexpr, G: tl.constexpr, ROUND_SCALE: tl.constexpr,
                             IN_BF16: tl.constexpr):
    """Same math as kernels._act_quant_kernel. Writes int8 element (m, k) at byte (k // 4) * 4M + 4m + k % 4,
    i.e. an int32 [N / 4, M] array whose word (k4, m) packs x[m, 4 k4 : 4 k4 + 4]."""
    pid_m, pid_g = tl.program_id(0), tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_g * G + tl.arange(0, G)
    mask = rows[:, None] < M
    x = tl.load(x_ptr + rows[:, None] * N + cols[None, :], mask=mask, other=0)
    x = _bf16_bits_to_f32(x) if IN_BF16 else x.to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-4)
    s = amax * (1.0 / 127.0)
    if ROUND_SCALE:
        s = libdevice.exp2(libdevice.ceil(libdevice.log2(s)))
    y = libdevice.rint(tl.math.div_rn(x, s[:, None]))
    y = tl.minimum(tl.maximum(y, -127.0), 127.0)
    off = (cols[None, :] // 4) * (4 * M) + rows[:, None] * 4 + cols[None, :] % 4
    tl.store(yT_ptr + off, y.to(tl.int8), mask=mask)
    tl.store(s_ptr + rows * NG + pid_g, s, mask=rows < M)


def act_quant_kmajor(x: torch.Tensor, round_scale: bool = False, block_m: int = 32):
    """x [..., N] -> (int32 [N // 4, M] K-major packed int8, fp32 scales [M, N // 128]), M = rows of x."""
    assert x.is_contiguous() and x.shape[-1] % GROUP == 0
    N = x.shape[-1]; x2 = x.view(-1, N); M = x2.shape[0]; NG = N // GROUP
    yT = torch.empty(N // 4, M, device=x.device, dtype=torch.int32)
    s = torch.empty(M, NG, device=x.device, dtype=torch.float32)
    bf16 = x.dtype == torch.bfloat16
    _act_quant_kmajor_kernel[(triton.cdiv(M, block_m), NG)](
        x2.view(torch.int16) if bf16 else x2, yT.view(torch.int8), s, M, N, NG,
        BLOCK_M=block_m, G=GROUP, ROUND_SCALE=round_scale, IN_BF16=bf16, num_warps=4)
    return yT, s


def kmajor(t: torch.Tensor) -> torch.Tensor:
    """int8 [rows, K] -> int32 [K // 4, rows] K-major packed. Done once, offline, for weights."""
    return t.contiguous().view(torch.int32).T.contiguous()


# --------------------------------------------------------------------------------------------------------------
# block-scaled GEMM
# --------------------------------------------------------------------------------------------------------------
@triton.jit
def _int8_gemm_dp4a_kernel(aT_ptr, bT_ptr, c_ptr, as_ptr, bs_ptr, M, N, KG,
                           BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, UNROLL: tl.constexpr,
                           GROUP_M: tl.constexpr, OUT_BF16: tl.constexpr):
    pid = tl.program_id(0)
    num_m, num_n = tl.cdiv(M, BLOCK_M), tl.cdiv(N, BLOCK_N)
    width = GROUP_M * num_n
    first_m = (pid // width) * GROUP_M
    size_m = tl.minimum(num_m - first_m, GROUP_M)
    pid_m = first_m + (pid % width) % size_m
    pid_n = (pid % width) // size_m
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mm, mn = rm < M, rn < N
    nb = (pid_n * BLOCK_N) // 128
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kb in range(0, KG):
        part = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
        for k0 in range(0, 32, UNROLL):  # 32 int32 words = one 128-wide scale block
            for u in tl.static_range(UNROLL):
                k4 = kb * 32 + k0 + u
                a = tl.load(aT_ptr + k4 * M + rm, mask=mm, other=0)
                b = tl.load(bT_ptr + k4 * N + rn, mask=mn, other=0)
                part = _dp4a(tl.broadcast_to(a[:, None], (BLOCK_M, BLOCK_N)),
                             tl.broadcast_to(b[None, :], (BLOCK_M, BLOCK_N)), part)
        sa = tl.load(as_ptr + rm * KG + kb, mask=mm, other=0.0)
        sb = tl.load(bs_ptr + nb * KG + kb)
        acc += part.to(tl.float32) * (sa * sb)[:, None]
    c = c_ptr + rm[:, None] * N + rn[None, :]
    out = _f32_to_bf16_bits(acc) if OUT_BF16 else acc.to(c_ptr.dtype.element_ty)
    tl.store(c, out, mask=mm[:, None] & mn[None, :])


_DP4A_CONFIGS = [triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "UNROLL": u, "GROUP_M": 8}, num_warps=w, num_stages=1)
                 for bm, bn, w in [(16, 128, 1), (32, 64, 1), (64, 32, 1), (64, 64, 1), (64, 64, 2), (32, 128, 1),
                                   (32, 128, 2), (64, 128, 2), (128, 64, 2), (64, 128, 4), (128, 128, 4)]
                 for u in (4, 8)]
_int8_gemm_dp4a = triton.autotune(configs=_DP4A_CONFIGS, key=["M", "N", "KG"])(_int8_gemm_dp4a_kernel)


def int8_gemm_v2(aT, a_s, bT, b_s, out_dtype=torch.float32):
    """aT int32 [K // 4, M] (act_quant_kmajor), a_s [M, K // 128], bT int32 [K // 4, N] (kmajor(weight)),
    b_s [N // 128, K // 128] -> [M, N]. Same math as kernels.int8_gemm."""
    K4, M = aT.shape; N = bT.shape[1]; KG = K4 * 4 // GROUP
    assert bT.shape[0] == K4 and N % GROUP == 0 and a_s.shape == (M, KG)
    c = torch.empty(M, N, device=aT.device, dtype=out_dtype)
    bf16 = out_dtype == torch.bfloat16
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
    _int8_gemm_dp4a[grid](aT, bT, c.view(torch.int16) if bf16 else c, a_s, b_s, M, N, KG, OUT_BF16=bf16)
    return c


# --------------------------------------------------------------------------------------------------------------
# lightning indexer
# --------------------------------------------------------------------------------------------------------------
@triton.jit
def _int8_index_dp4a_kernel(qT_ptr, w_ptr, kT_ptr, ks_ptr, o_ptr, M, Nk, start, R,
                            H: tl.constexpr, D4: tl.constexpr, BLOCK_Q: tl.constexpr, BLOCK_N: tl.constexpr,
                            UNROLL: tl.constexpr, CAUSAL: tl.constexpr):
    """qT int32 [b, D / 4, M * H] (row = m * H + h), kT int32 [b, D / 4, Nk]."""
    pid_n, pid_q, b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    q0 = pid_q * BLOCK_Q
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rq = q0 + tl.arange(0, BLOCK_Q)
    o = o_ptr + b * M * Nk + rq[:, None] * Nk + rn[None, :]
    omask = (rq[:, None] < M) & (rn[None, :] < Nk)
    if CAUSAL:  # skip tiles where every key is after every query
        if pid_n * BLOCK_N > start + q0 + BLOCK_Q - 1:
            tl.store(o, tl.full((BLOCK_Q, BLOCK_N), float("-inf"), tl.float32), mask=omask)
            return
    rows = q0 * H + tl.arange(0, BLOCK_Q * H)
    rmask = rows < R
    nmask = rn < Nk
    acc = tl.zeros((BLOCK_Q * H, BLOCK_N), dtype=tl.int32)
    for k0 in range(0, D4, UNROLL):
        for u in tl.static_range(UNROLL):
            qv = tl.load(qT_ptr + (b * D4 + k0 + u) * R + rows, mask=rmask, other=0)
            kv = tl.load(kT_ptr + (b * D4 + k0 + u) * Nk + rn, mask=nmask, other=0)
            acc = _dp4a(tl.broadcast_to(qv[:, None], (BLOCK_Q * H, BLOCK_N)),
                        tl.broadcast_to(kv[None, :], (BLOCK_Q * H, BLOCK_N)), acc)
    w = tl.load(w_ptr + b * R + rows, mask=rmask, other=0.0)
    logits = tl.maximum(acc.to(tl.float32), 0.0) * w[:, None]
    score = tl.sum(tl.reshape(logits, (BLOCK_Q, H, BLOCK_N)), axis=1)
    score = score * tl.load(ks_ptr + b * Nk + rn, mask=nmask, other=0.0)[None, :]
    if CAUSAL:
        score = tl.where(rn[None, :] <= start + rq[:, None], score, float("-inf"))
    tl.store(o, score, mask=omask)


def index_kmajor_q(q: torch.Tensor) -> torch.Tensor:
    """q int8 [b, m, h, d] -> int32 [b, d / 4, m * h]."""
    B, M, H, D = q.shape
    return q.contiguous().view(B, M * H, D).view(torch.int32).transpose(1, 2).contiguous()


def index_kmajor_k(k: torch.Tensor) -> torch.Tensor:
    """k int8 [b, n, d] -> int32 [b, d / 4, n], the layout a K-major key cache would keep."""
    return k.contiguous().view(torch.int32).transpose(1, 2).contiguous()


def int8_index_v2(qT, w, kT, k_s, H, causal=False, start=0, block_q=1, block_n=64, unroll=8, num_warps=1):
    """qT [b, d / 4, m * h] (index_kmajor_q), w [b, m, h], kT [b, d / 4, n] (index_kmajor_k), k_s [b, n] -> [b, m, n].
    Accumulators per thread = block_q * H * block_n / (32 * num_warps); ~128 is the measured sweet spot."""
    B, D4, R = qT.shape; M = R // H; Nk = kT.shape[2]
    assert w.is_contiguous() and k_s.is_contiguous() and kT.shape[:2] == (B, D4)
    o = torch.empty(B, M, Nk, device=qT.device, dtype=torch.float32)
    grid = (triton.cdiv(Nk, block_n), triton.cdiv(M, block_q), B)
    _int8_index_dp4a_kernel[grid](qT, w, kT, k_s, o, M, Nk, start, R, H=H, D4=D4, BLOCK_Q=block_q, BLOCK_N=block_n,
                                  UNROLL=unroll, CAUSAL=causal, num_warps=num_warps)
    return o
