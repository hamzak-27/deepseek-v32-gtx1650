"""Correctness tests: every kernel against its float64 / exact reference, over odd shapes, masks and options.

Run: python -m pytest -q test_kernels.py
"""
import pytest
import torch

from kernels import act_quant, int8_gemm, int8_index
from reference import act_quant_ref, int8_gemm_ref, int8_index_ref

dev = "cuda"
torch.manual_seed(0)


def rel_err(x, ref):
    fin = torch.isfinite(ref)
    return ((x.double() - ref)[fin].abs().max() / ref[fin].abs().max().clamp_min(1e-30)).item()


# ---------------- act_quant: must match the reference bit for bit ----------------
@pytest.mark.parametrize("shape", [(1, 128), (7, 256), (33, 512), (4, 5, 384), (1000, 1024)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("round_scale", [False, True])
def test_act_quant_exact(shape, dtype, round_scale):
    x = (torch.randn(shape, device=dev) * 3).to(dtype)
    x.view(-1)[::97] *= 50  # outliers
    y, s = act_quant(x, round_scale)
    yr, sr = act_quant_ref(x, round_scale)
    assert y.dtype == torch.int8 and y.shape == x.shape and s.shape == (*shape[:-1], shape[-1] // 128)
    assert torch.equal(s, sr)
    assert torch.equal(y, yr)


def test_act_quant_zero_and_tiny():
    x = torch.zeros(4, 256, device=dev); x[1] = 1e-9; x[2, :5] = -7.0
    y, s = act_quant(x); yr, sr = act_quant_ref(x)
    assert torch.equal(y, yr) and torch.equal(s, sr)
    assert (s[0] == torch.tensor(1e-4, device=dev) * (1.0 / 127.0)).all()  # amax floor


def test_act_quant_roundtrip_error():
    x = torch.randn(64, 1024, device=dev)
    y, s = act_quant(x)
    deq = y.float().view(64, 8, 128) * s.unsqueeze(-1)
    assert (deq.view_as(x) - x).abs().max() <= s.max() / 2 + 1e-6  # error at most half a step


# ---------------- int8_gemm: integer dot is exact; only the fp32 accumulation of scaled blocks differs --------
def make_gemm(M, N, K):
    a = torch.randint(-127, 128, (M, K), device=dev, dtype=torch.int8)
    b = torch.randint(-127, 128, (N, K), device=dev, dtype=torch.int8)
    a_s = torch.rand(M, K // 128, device=dev) * 0.01 + 1e-4
    b_s = torch.rand(N // 128, K // 128, device=dev) * 0.01 + 1e-4
    return a, a_s, b, b_s


@pytest.mark.parametrize("M,N,K", [(1, 128, 128), (5, 256, 384), (37, 384, 256), (128, 128, 1024),
                                   (300, 640, 512), (1024, 1024, 1024)])
@pytest.mark.parametrize("version", ["v0", "v1"])
def test_int8_gemm(M, N, K, version):
    a, a_s, b, b_s = make_gemm(M, N, K)
    c = int8_gemm(a, a_s, b, b_s, version=version)
    assert c.shape == (M, N) and c.dtype == torch.float32
    assert rel_err(c, int8_gemm_ref(a, a_s, b, b_s)) < 1e-5


def test_int8_gemm_batched_and_dtypes():
    a, a_s, b, b_s = make_gemm(2 * 3 * 16, 256, 256)
    a3, as3 = a.view(2, 3, 16, 256), a_s.view(2, 3, 16, 2)
    ref = int8_gemm_ref(a3, as3, b, b_s)
    assert int8_gemm(a3, as3, b, b_s).shape == (2, 3, 16, 256)
    for dt, tol in [(torch.float16, 2e-3), (torch.bfloat16, 1e-2)]:
        assert rel_err(int8_gemm(a3, as3, b, b_s, out_dtype=dt), ref) < tol


def test_int8_gemm_end_to_end_linear():
    """Quantize a real fp32 linear layer with act_quant + per-block weight scales; compare with fp32 matmul."""
    M, N, K = 64, 512, 1024
    x = torch.randn(M, K, device=dev); W = torch.randn(N, K, device=dev) * 0.02
    xq, xs = act_quant(x)
    Wb = W.view(N // 128, 128, K // 128, 128)
    ws = (Wb.abs().amax((1, 3)).clamp_min(1e-4) / 127)
    Wq = torch.round(Wb / ws[:, None, :, None]).clamp(-127, 127).to(torch.int8).view(N, K)
    y = int8_gemm(xq, xs, Wq, ws)
    ref = x @ W.T
    assert ((y - ref).norm() / ref.norm()).item() < 0.02  # INT8 quantization error, not a kernel bug


# ---------------- int8_index -----------------------------------------------------------------------------------
def make_index(B, M, N, H=64, D=128):
    q = torch.randint(-127, 128, (B, M, H, D), device=dev, dtype=torch.int8)
    k = torch.randint(-127, 128, (B, N, D), device=dev, dtype=torch.int8)
    w = torch.randn(B, M, H, device=dev) * 1e-4
    k_s = torch.rand(B, N, device=dev) * 0.01
    return q, w, k, k_s


@pytest.mark.parametrize("B,M,N", [(1, 1, 128), (1, 1, 1000), (2, 3, 257), (1, 17, 512), (2, 64, 1024)])
@pytest.mark.parametrize("block_q,block_n", [(1, 128), (2, 64), (4, 32)])
def test_int8_index(B, M, N, block_q, block_n):
    q, w, k, k_s = make_index(B, M, N)
    o = int8_index(q, w, k, k_s, block_q=block_q, block_n=block_n)
    assert o.shape == (B, M, N)
    assert rel_err(o, int8_index_ref(q, w, k, k_s)) < 1e-5


@pytest.mark.parametrize("M,N,start", [(64, 64, 0), (100, 100, 0), (16, 300, 284), (33, 200, 50)])
def test_int8_index_causal(M, N, start):
    q, w, k, k_s = make_index(1, M, N)
    o = int8_index(q, w, k, k_s, causal=True, start=start)
    ref = int8_index_ref(q, w, k, k_s, causal=True, start=start)
    assert torch.equal(torch.isinf(o), torch.isinf(ref))
    assert rel_err(o, ref) < 1e-5


def test_int8_index_small_heads():
    q, w, k, k_s = make_index(1, 8, 256, H=16, D=64)
    assert rel_err(int8_index(q, w, k, k_s), int8_index_ref(q, w, k, k_s)) < 1e-5


def test_int8_index_topk_matches():
    """What the model uses: the top-k set must match the reference."""
    q, w, k, k_s = make_index(1, 32, 4096)
    o = int8_index(q, w, k, k_s); ref = int8_index_ref(q, w, k, k_s)
    a = o.topk(256, -1).indices.sort(-1).values; r = ref.topk(256, -1).indices.sort(-1).values
    assert (a == r).float().mean() > 0.999
