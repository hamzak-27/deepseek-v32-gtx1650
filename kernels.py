"""INT8 ports of DeepSeek-V3.2-Exp's FP8 TileLang kernels (inference/kernel.py) for Turing GPUs (sm_75).

Turing has no FP8 hardware, and on the GTX 1650 measured INT8 matmul throughput is 6.4x FP32 while FP16 is
5x slower than FP32 (see roofline.py). The kernels keep DeepSeek's block-scaling scheme and swap FP8 for INT8:

- act_quant:  one scale per row per 128-column group (amax / 127), optional power-of-two (ue8m0) scales.
- int8_gemm:  C = sum over 128-wide K blocks of (A_kb @ B_kb^T) * a_s[m, kb] * b_s[n // 128, kb].
- int8_index: the lightning indexer, score[b, m, n] = k_s[b, n] * sum_h w[b, m, h] * relu(q[b, m, h] . k[b, n]).

Each kernel has a v0 that follows DeepSeek's tiling, and later versions that change it. bench.py compares them.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

GROUP = 128  # DeepSeek's quantization block size


# sm_75 has no bf16 convert instructions (cvt.bf16.f32 needs sm_80), so bf16 tensors are passed as int16 bits.
@triton.jit
def _bf16_bits_to_f32(bits):
    return (bits.to(tl.int32) << 16).to(tl.float32, bitcast=True)


@triton.jit
def _f32_to_bf16_bits(x):
    b = x.to(tl.int32, bitcast=True)
    b = b + 0x7FFF + ((b >> 16) & 1)  # round to nearest even
    return (b >> 16).to(tl.int16)


# --------------------------------------------------------------------------------------------------------------
# act_quant
# --------------------------------------------------------------------------------------------------------------
@triton.jit
def _act_quant_kernel(x_ptr, y_ptr, s_ptr, M, N, NG,
                      BLOCK_M: tl.constexpr, G: tl.constexpr, ROUND_SCALE: tl.constexpr, IN_BF16: tl.constexpr):
    pid_m, pid_g = tl.program_id(0), tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_g * G + tl.arange(0, G)
    mask = rows[:, None] < M
    x = tl.load(x_ptr + rows[:, None] * N + cols[None, :], mask=mask, other=0)
    x = _bf16_bits_to_f32(x) if IN_BF16 else x.to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-4)
    s = amax * (1.0 / 127.0)  # reciprocal multiply, as DeepSeek (amax * fp8_max_inv)
    if ROUND_SCALE:  # ue8m0: smallest power of two >= amax / 127, as DeepSeek's fast_round_scale
        s = libdevice.exp2(libdevice.ceil(libdevice.log2(s)))
    y = libdevice.rint(tl.math.div_rn(x, s[:, None]))
    y = tl.minimum(tl.maximum(y, -127.0), 127.0)
    tl.store(y_ptr + rows[:, None] * N + cols[None, :], y.to(tl.int8), mask=mask)
    tl.store(s_ptr + rows * NG + pid_g, s, mask=rows < M)


def act_quant(x: torch.Tensor, round_scale: bool = False, block_m: int = 32):
    """x [..., N] -> (int8 [..., N], fp32 scales [..., N // 128])."""
    assert x.is_contiguous() and x.shape[-1] % GROUP == 0
    N = x.shape[-1]; x2 = x.view(-1, N); M = x2.shape[0]; NG = N // GROUP
    y = torch.empty_like(x2, dtype=torch.int8); s = torch.empty(M, NG, device=x.device, dtype=torch.float32)
    bf16 = x.dtype == torch.bfloat16
    _act_quant_kernel[(triton.cdiv(M, block_m), NG)](x2.view(torch.int16) if bf16 else x2, y, s, M, N, NG,
                                                     BLOCK_M=block_m, G=GROUP, ROUND_SCALE=round_scale,
                                                     IN_BF16=bf16, num_warps=4)
    return y.view(x.shape), s.view(*x.shape[:-1], NG)


# --------------------------------------------------------------------------------------------------------------
# int8_gemm: block-scaled, B in nn.Linear layout [N, K]
# --------------------------------------------------------------------------------------------------------------
@triton.jit
def _int8_gemm_kernel(a_ptr, b_ptr, c_ptr, as_ptr, bs_ptr, M, N, K, KG,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                      GROUP_M: tl.constexpr, OUT_BF16: tl.constexpr):
    pid = tl.program_id(0)
    num_m, num_n = tl.cdiv(M, BLOCK_M), tl.cdiv(N, BLOCK_N)
    if GROUP_M > 1:  # grouped ordering: neighbouring programs share B tiles in L2
        width = GROUP_M * num_n
        first_m = (pid // width) * GROUP_M
        size_m = tl.minimum(num_m - first_m, GROUP_M)
        pid_m = first_m + (pid % width) % size_m
        pid_n = (pid % width) // size_m
    else:
        pid_m, pid_n = pid // num_n, pid % num_n
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    a_tile = a_ptr + rm[:, None] * K + rk[None, :]
    b_tile = b_ptr + rn[:, None] * K + rk[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    nb = (pid_n * BLOCK_N) // 128  # BLOCK_N divides 128, so the tile sits in one weight-scale block
    for kb in range(0, KG):
        a = tl.load(a_tile, mask=rm[:, None] < M, other=0)
        b = tl.load(b_tile, mask=rn[:, None] < N, other=0)
        part = tl.dot(a, tl.trans(b), out_dtype=tl.int32)
        sa = tl.load(as_ptr + rm * KG + kb, mask=rm < M, other=0.0)
        sb = tl.load(bs_ptr + nb * KG + kb)
        acc += part.to(tl.float32) * (sa * sb)[:, None]
        a_tile += BLOCK_K; b_tile += BLOCK_K
    c = c_ptr + rm[:, None] * N + rn[None, :]
    out = _f32_to_bf16_bits(acc) if OUT_BF16 else acc.to(c_ptr.dtype.element_ty)
    tl.store(c, out, mask=(rm[:, None] < M) & (rn[None, :] < N))


_GEMM_CONFIGS = [triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "GROUP_M": 8}, num_warps=w, num_stages=s)
                 for bm in (32, 64, 128) for bn in (64, 128) for w in (4, 8) for s in (2, 3)]
_int8_gemm_tuned = triton.autotune(configs=_GEMM_CONFIGS, key=["M", "N", "K"])(_int8_gemm_kernel)


def int8_gemm(a, a_s, b, b_s, out_dtype=torch.float32, version="v1"):
    """a [..., K] int8, a_s [..., K // 128], b [N, K] int8, b_s [N // 128, K // 128] -> [..., N].

    v0: DeepSeek's tile (32 x 128 x 128, 4 warps), row-major program order.
    v1: autotuned tile, warps and pipeline depth, grouped program order for L2 reuse.
    """
    assert a.is_contiguous() and b.is_contiguous() and a_s.is_contiguous() and b_s.is_contiguous()
    K = a.shape[-1]; N = b.shape[0]; M = a.numel() // K; KG = K // GROUP
    assert K % GROUP == 0 and N % GROUP == 0
    c = torch.empty(*a.shape[:-1], N, device=a.device, dtype=out_dtype)
    bf16 = out_dtype == torch.bfloat16; cv = c.view(torch.int16) if bf16 else c
    if version == "v0":
        grid = (triton.cdiv(M, 32) * triton.cdiv(N, 128),)
        _int8_gemm_kernel[grid](a, b, cv, a_s, b_s, M, N, K, KG, BLOCK_M=32, BLOCK_N=128, BLOCK_K=128,
                                GROUP_M=1, OUT_BF16=bf16, num_warps=4, num_stages=2)
    else:
        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
        _int8_gemm_tuned[grid](a, b, cv, a_s, b_s, M, N, K, KG, BLOCK_K=128, OUT_BF16=bf16)
    return c


# --------------------------------------------------------------------------------------------------------------
# int8_index: DeepSeek V3.2's lightning indexer
# --------------------------------------------------------------------------------------------------------------
@triton.jit
def _int8_index_kernel(q_ptr, w_ptr, k_ptr, ks_ptr, o_ptr, M, Nk, start,
                       H: tl.constexpr, D: tl.constexpr, BLOCK_Q: tl.constexpr, BLOCK_N: tl.constexpr,
                       CAUSAL: tl.constexpr):
    """One program: BLOCK_Q queries x BLOCK_N keys. The BLOCK_Q * H query heads are stacked as rows of one
    int8 dot, so each key tile loaded from memory is reused by all BLOCK_Q queries (DeepSeek's kernel = BLOCK_Q 1)."""
    pid_n, pid_q, b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    q0 = pid_q * BLOCK_Q
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rq = q0 + tl.arange(0, BLOCK_Q)
    o = o_ptr + b * M * Nk + rq[:, None] * Nk + rn[None, :]
    omask = (rq[:, None] < M) & (rn[None, :] < Nk)
    if CAUSAL:  # query at position start + m sees keys n <= start + m; skip tiles that are fully masked
        if pid_n * BLOCK_N > start + q0 + BLOCK_Q - 1:
            tl.store(o, tl.full((BLOCK_Q, BLOCK_N), float("-inf"), tl.float32), mask=omask)
            return
    rows = tl.arange(0, BLOCK_Q * H)  # row r = query q0 + r // H, head r % H
    rd = tl.arange(0, D)
    qmask = (q0 + rows // H) < M
    q = tl.load(q_ptr + (b * M + q0) * H * D + rows[:, None] * D + rd[None, :], mask=qmask[:, None], other=0)
    w = tl.load(w_ptr + (b * M + q0) * H + rows, mask=qmask, other=0.0)
    k = tl.load(k_ptr + (b * Nk + rn[:, None]) * D + rd[None, :], mask=rn[:, None] < Nk, other=0)
    logits = tl.dot(q, tl.trans(k), out_dtype=tl.int32).to(tl.float32)  # [BLOCK_Q * H, BLOCK_N]
    logits = tl.maximum(logits, 0.0) * w[:, None]
    score = tl.sum(tl.reshape(logits, (BLOCK_Q, H, BLOCK_N)), axis=1)  # [BLOCK_Q, BLOCK_N]
    ks = tl.load(ks_ptr + b * Nk + rn, mask=rn < Nk, other=0.0)
    score = score * ks[None, :]
    if CAUSAL:
        score = tl.where(rn[None, :] <= start + rq[:, None], score, float("-inf"))
    tl.store(o, score, mask=omask)


def int8_index(q, w, k, k_s, causal=False, start=0, block_q=None, block_n=None, num_warps=None):
    """q [b, m, h, d] int8, w [b, m, h] fp32 (head weights x q scale x softmax scale), k [b, n, d] int8,
    k_s [b, n] fp32 -> [b, m, n] fp32. causal: keys after start + m get -inf (prefill), and those tiles are skipped.

    Defaults: v1 (block_q 2). DeepSeek's layout is block_q=1, block_n=128.
    """
    B, M, H, D = q.shape; Nk = k.shape[1]
    assert q.is_contiguous() and k.is_contiguous() and w.is_contiguous() and k_s.is_contiguous()
    block_q = block_q or 2; block_n = block_n or 64; num_warps = num_warps or 4
    o = torch.empty(B, M, Nk, device=q.device, dtype=torch.float32)
    grid = (triton.cdiv(Nk, block_n), triton.cdiv(M, block_q), B)
    _int8_index_kernel[grid](q, w, k, k_s, o, M, Nk, start, H=H, D=D, BLOCK_Q=block_q, BLOCK_N=block_n,
                             CAUSAL=causal, num_warps=num_warps)
    return o
