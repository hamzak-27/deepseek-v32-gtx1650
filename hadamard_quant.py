"""Fused Hadamard rotation + INT8 quantization for the lightning indexer, writing the K-major layout int8_index_v2 reads.

V3.2's indexer rotates q and k with a 128-point Hadamard transform (scale 128^-0.5, `rotate_activation`) before
quantizing them, so outliers spread over all 128 dims and one scale per vector loses less. On this GPU there is no
fast_hadamard_transform; the rotation here is a 128x128 fp32 matmul inside the quantization kernel (tl.dot, one tile per
BLOCK_M rows), so the float vector is read once and only int8 + one scale are written.

rotate_quant_kmajor(x [B, R, 128]) -> (int32 [B, 32, R] K-major packed int8, fp32 scales [B, R]).
  q: x = q.view(B, M * H, 128) gives index_kmajor_q's layout; k: x = k [B, N, 128] gives index_kmajor_k's layout.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

D = 128


def hadamard_matrix(d=D, device="cuda"):
    h = torch.ones(1, 1)
    while h.shape[0] < d:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return (h / d ** 0.5).to(device=device, dtype=torch.float32).contiguous()


@triton.jit
def _rotate_quant_kernel(x_ptr, h_ptr, yT_ptr, s_ptr, R, TOTAL,
                         BLOCK_M: tl.constexpr, DD: tl.constexpr, ROTATE: tl.constexpr):
    """H128 = (1/sqrt2) [[H64, H64], [H64, -H64]] (Sylvester), so x H128 = [(a + b) H64, (a - b) H64] / sqrt2 for the
    halves a, b of x: one butterfly, then two 64-point products against the same 16 KB matrix (the 64 KB full matrix
    does not fit next to the tile in this GPU's shared memory). h_ptr holds H64 already scaled by 1/sqrt(128)."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)            # global row = b * R + r
    HALF: tl.constexpr = DD // 2
    c = tl.arange(0, HALF)
    mask = rows[:, None] < TOTAL
    a = tl.load(x_ptr + rows[:, None] * DD + c[None, :], mask=mask, other=0.0).to(tl.float32)
    bb = tl.load(x_ptr + rows[:, None] * DD + HALF + c[None, :], mask=mask, other=0.0).to(tl.float32)
    if ROTATE:
        hm = tl.load(h_ptr + c[:, None] * HALF + c[None, :])
        a, bb = tl.dot(a + bb, hm, input_precision="ieee"), tl.dot(a - bb, hm, input_precision="ieee")
    amax = tl.maximum(tl.maximum(tl.max(tl.abs(a), axis=1), tl.max(tl.abs(bb), axis=1)), 1e-4)
    s = amax * (1.0 / 127.0)
    b = rows // R; r = rows % R
    for half in tl.static_range(2):
        v = a if half == 0 else bb
        y = tl.minimum(tl.maximum(libdevice.rint(tl.math.div_rn(v, s[:, None])), -127.0), 127.0)
        col = c + half * HALF
        off = b[:, None] * (DD * R) + (col[None, :] // 4) * (4 * R) + r[:, None] * 4 + col[None, :] % 4
        tl.store(yT_ptr + off, y.to(tl.int8), mask=mask)
    tl.store(s_ptr + rows, s, mask=rows < TOTAL)


@triton.jit
def _stage(x, BLOCK_M: tl.constexpr, DD: tl.constexpr, A: tl.constexpr, Bs: tl.constexpr):
    """One butterfly stage: element i pairs with i ^ Bs (index = (a, pair, j) with j < Bs)."""
    v = tl.permute(tl.reshape(x, (BLOCK_M, A, 2, Bs)), (0, 1, 3, 2))  # pair dim last
    lo, hi = tl.split(v)
    return tl.reshape(tl.permute(tl.join(lo + hi, lo - hi), (0, 1, 3, 2)), (BLOCK_M, DD))


@triton.jit
def _fwht_quant_kernel(x_ptr, yT_ptr, s_ptr, R, TOTAL, BLOCK_M: tl.constexpr, DD: tl.constexpr, LOG: tl.constexpr):
    """Fast Walsh-Hadamard transform in registers: LOG butterfly stages of adds / subtracts (7 x 128 per row instead
    of a 128 x 128 product), then the same quantization and K-major store."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, DD)
    mask = rows[:, None] < TOTAL
    x = tl.load(x_ptr + rows[:, None] * DD + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    x = _stage(x, BLOCK_M, DD, DD // 2, 1); x = _stage(x, BLOCK_M, DD, DD // 4, 2); x = _stage(x, BLOCK_M, DD, DD // 8, 4)
    x = _stage(x, BLOCK_M, DD, DD // 16, 8); x = _stage(x, BLOCK_M, DD, DD // 32, 16); x = _stage(x, BLOCK_M, DD, DD // 64, 32)
    x = _stage(x, BLOCK_M, DD, 1, 64)  # 7 stages for DD = 128 (LOG)
    x = x * (1.0 / tl.sqrt(DD * 1.0))
    amax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-4)
    s = amax * (1.0 / 127.0)
    y = tl.minimum(tl.maximum(libdevice.rint(tl.math.div_rn(x, s[:, None])), -127.0), 127.0)
    b = rows // R; r = rows % R
    off = b[:, None] * (DD * R) + (cols[None, :] // 4) * (4 * R) + r[:, None] * 4 + cols[None, :] % 4
    tl.store(yT_ptr + off, y.to(tl.int8), mask=mask)
    tl.store(s_ptr + rows, s, mask=rows < TOTAL)


def fwht_quant_kmajor(x: torch.Tensor, block_m: int = 32, num_warps: int = 4):
    """Same output as rotate_quant_kmajor(x, rotate=True), with the butterfly transform."""
    assert x.is_contiguous() and x.shape[-1] == D and x.dim() == 3
    B, R, _ = x.shape; TOTAL = B * R
    yT = torch.empty(B, D // 4, R, device=x.device, dtype=torch.int32)
    s = torch.empty(B, R, device=x.device, dtype=torch.float32)
    _fwht_quant_kernel[(triton.cdiv(TOTAL, block_m),)](x, yT.view(torch.int8), s, R, TOTAL, BLOCK_M=block_m, DD=D, LOG=7, num_warps=num_warps)
    return yT, s


_H = {}
def rotate_quant_kmajor(x: torch.Tensor, rotate: bool = True, block_m: int = 32):
    assert x.is_contiguous() and x.shape[-1] == D and x.dim() == 3
    B, R, _ = x.shape; TOTAL = B * R
    if x.device not in _H: _H[x.device] = (hadamard_matrix(D // 2, device=x.device) * (D // 2) ** 0.5 / D ** 0.5).contiguous()
    yT = torch.empty(B, D // 4, R, device=x.device, dtype=torch.int32)
    s = torch.empty(B, R, device=x.device, dtype=torch.float32)
    _rotate_quant_kernel[(triton.cdiv(TOTAL, block_m),)](x, _H[x.device], yT.view(torch.int8), s, R, TOTAL,
                                                         BLOCK_M=block_m, DD=D, ROTATE=rotate, num_warps=4)
    return yT, s


def rotate_quant_ref(x: torch.Tensor, rotate: bool = True):
    """Reference: fp64 rotation, then the same per-vector INT8 rounding; returns int8 [B, R, 128] and scales [B, R]."""
    xr = x.double() @ hadamard_matrix(device=x.device).double() if rotate else x.double()
    s = xr.abs().amax(-1).clamp_min(1e-4) / 127.0
    return torch.clamp(torch.round(xr / s[..., None]), -127, 127).to(torch.int8), s.float()


def unpack_kmajor(yT: torch.Tensor) -> torch.Tensor:
    """int32 [B, 32, R] -> int8 [B, R, 128] (for tests)."""
    return yT.transpose(1, 2).contiguous().view(torch.int8).view(yT.shape[0], yT.shape[2], D)
