"""Correctness tests for the dp4a (v2) kernels. Run: python -m pytest -q test_kernels_v2.py"""
import pytest
import torch

from kernels import act_quant
from kernels_v2 import act_quant_kmajor, kmajor, int8_gemm_v2, int8_index_v2, index_kmajor_q, index_kmajor_k
from reference import int8_gemm_ref, int8_index_ref
from test_kernels import dev, rel_err, make_gemm, make_index


@pytest.mark.parametrize("shape", [(1, 128), (33, 512), (4, 5, 384), (1000, 1024)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("round_scale", [False, True])
def test_act_quant_kmajor_layout(shape, dtype, round_scale):
    x = (torch.randn(shape, device=dev) * 3).to(dtype)
    yT, s = act_quant_kmajor(x, round_scale)
    y, s0 = act_quant(x, round_scale)  # v0 already matches the reference bit for bit
    assert torch.equal(s.view_as(s0), s0)
    assert torch.equal(yT, kmajor(y.view(-1, shape[-1])))


@pytest.mark.parametrize("M,N,K", [(1, 128, 128), (5, 256, 384), (37, 384, 256), (300, 640, 512),
                                   (1024, 1024, 1024), (16, 7168, 2048)])
def test_int8_gemm_v2(M, N, K):
    a, a_s, b, b_s = make_gemm(M, N, K)
    c = int8_gemm_v2(kmajor(a), a_s, kmajor(b), b_s)
    assert c.shape == (M, N)
    assert rel_err(c, int8_gemm_ref(a, a_s, b, b_s)) < 1e-5


def test_int8_gemm_v2_pipeline_bf16():
    """bf16 activations -> act_quant_kmajor -> GEMM -> bf16 output, against the reference on the same integers."""
    x = torch.randn(64, 1024, device=dev, dtype=torch.bfloat16)
    _, _, b, b_s = make_gemm(64, 256, 1024)
    xT, xs = act_quant_kmajor(x); xq, xs0 = act_quant(x)
    ref = int8_gemm_ref(xq, xs0, b, b_s)
    assert rel_err(int8_gemm_v2(xT, xs, kmajor(b), b_s, out_dtype=torch.bfloat16), ref) < 1e-2


@pytest.mark.parametrize("B,M,N", [(1, 1, 128), (1, 1, 1000), (2, 3, 257), (1, 17, 512), (2, 64, 1024)])
@pytest.mark.parametrize("block_q,block_n,warps", [(1, 64, 1), (1, 128, 2), (2, 32, 1), (1, 32, 1)])
def test_int8_index_v2(B, M, N, block_q, block_n, warps):
    q, w, k, k_s = make_index(B, M, N)
    o = int8_index_v2(index_kmajor_q(q), w, index_kmajor_k(k), k_s, H=64,
                      block_q=block_q, block_n=block_n, num_warps=warps)
    assert rel_err(o, int8_index_ref(q, w, k, k_s)) < 1e-5


@pytest.mark.parametrize("M,N,start", [(64, 64, 0), (100, 100, 0), (16, 300, 284), (33, 200, 50)])
def test_int8_index_v2_causal(M, N, start):
    q, w, k, k_s = make_index(1, M, N)
    o = int8_index_v2(index_kmajor_q(q), w, index_kmajor_k(k), k_s, H=64, causal=True, start=start)
    ref = int8_index_ref(q, w, k, k_s, causal=True, start=start)
    assert torch.equal(torch.isinf(o), torch.isinf(ref)) and rel_err(o, ref) < 1e-5


def test_int8_index_v2_small_heads():
    q, w, k, k_s = make_index(1, 8, 256, H=16, D=64)
    o = int8_index_v2(index_kmajor_q(q), w, index_kmajor_k(k), k_s, H=16, block_n=128)
    assert rel_err(o, int8_index_ref(q, w, k, k_s)) < 1e-5
