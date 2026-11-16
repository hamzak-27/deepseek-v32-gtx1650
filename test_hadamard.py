"""Correctness tests for the fused Hadamard rotation + INT8 quantization. Run: python -m pytest -q test_hadamard.py"""
import pytest
import torch

from hadamard_quant import rotate_quant_kmajor, fwht_quant_kmajor, rotate_quant_ref, unpack_kmajor, hadamard_matrix
from kernels_v2 import index_kmajor_k, index_kmajor_q
from test_kernels import dev


def outliers(B, R):
    x = torch.randn(B, R, 128, device=dev); x[..., :4] *= 20; return x


def test_hadamard_is_orthonormal():
    h = hadamard_matrix(device=dev); assert torch.allclose(h @ h.T, torch.eye(128, device=dev), atol=1e-5)


@pytest.mark.parametrize("B,R", [(1, 1), (1, 4096), (4, 64), (2, 4097)])
@pytest.mark.parametrize("kernel", ["matrix", "butterfly"])
def test_rotate_quant_matches_fp64(B, R, kernel):
    x = outliers(B, R)
    yT, s = rotate_quant_kmajor(x) if kernel == "matrix" else fwht_quant_kmajor(x)
    y, (yr, sr) = unpack_kmajor(yT), rotate_quant_ref(x)
    d = (y.int() - yr.int()).abs()
    assert d.max() <= 1 and (d == 0).float().mean() > 0.9999  # off by one only where fp32 and fp64 round a tie apart
    assert ((s - sr).abs() / sr).max() < 1e-5


@pytest.mark.parametrize("B,R", [(1, 300), (3, 77)])
def test_no_rotation_is_plain_quant(B, R):
    x = outliers(B, R); yT, s = rotate_quant_kmajor(x, rotate=False); yr, sr = rotate_quant_ref(x, rotate=False)
    assert torch.equal(unpack_kmajor(yT), yr)


def test_layouts_match_indexer():
    q = outliers(2, 3 * 64); yT, _ = rotate_quant_kmajor(q)  # queries: [B, M * H, 128]
    assert torch.equal(yT, index_kmajor_q(unpack_kmajor(yT).view(2, 3, 64, 128)))
    k = outliers(2, 100); yT, _ = rotate_quant_kmajor(k)
    assert torch.equal(yT, index_kmajor_k(unpack_kmajor(yT)))
