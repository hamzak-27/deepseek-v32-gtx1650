# DeepSeek-V3.2 kernels on a GTX 1650 (Turing, sm_75)

INT8 ports of the three FP8 TileLang kernels in [DeepSeek-V3.2-Exp `inference/kernel.py`](../DeepSeek-V3.2-Exp/inference/kernel.py),
written in Triton for a 4 GB laptop GPU that has no FP8, no BF16 converts and no tensor cores.

| Kernel | What it does in V3.2 |
|---|---|
| `act_quant` | Quantize activations, one scale per 128 values (optional power-of-two "ue8m0" scales) |
| `int8_gemm` | Block-scaled matmul: per-token x per-128-K activation scales, per 128x128 weight-block scales |
| `int8_index` | The lightning indexer of DeepSeek Sparse Attention: `score[m,n] = k_s[n] * sum_h w[m,h] * relu(q[m,h] . k[n])`, 64 heads x 128 dims |

## What this GPU can do (`roofline.py`)

| Path | Measured | vs FP32 |
|---|---|---|
| FP32 matmul | 1.74 TFLOP/s | 1x |
| FP16 matmul | 0.33 TFLOP/s | 5x slower |
| BF16 matmul | 1.54 TFLOP/s | emulated |
| INT8 (`torch._int_mm`) | 11.2 TOP/s | 6.4x faster |
| Memory copy | 155 GB/s | |

INT8 = 512 ops / clock / SM, which is the `dp4a` peak. That decided the design.

## Versions

- **v0**: DeepSeek's tiling, INT8 instead of FP8 (`kernels.py`).
- **v1**: autotuned tiles + grouped program order; indexer stacks several queries per tile; causal tile skipping.
- **v2**: `dp4a` through inline PTX over K-major operands (`kernels_v2.py`). v0 and v1 sat at ~1.28 TOP/s on every
  shape because Triton's int8 `tl.dot` has no fast path on sm_75. Plain dp4a GEMM went 2.0 -> 6.7 TOP/s when the
  operands were stored K-major, and the best tiles hold ~128 int32 accumulators per thread (256 spill: 95x slower).

## Results (`python bench.py`, full table in `results/bench.md`)

| Kernel, shape | PyTorch baseline | v0 | best | Speedup vs baseline |
|---|---|---|---|---|
| act_quant 2048x7168 fp32 | 3.83 ms (eager) | 0.55 ms | 0.55 ms (134 GB/s) | **7.0x** |
| act_quant 2048x7168 bf16 | 4.47 ms | 0.39 ms | 0.39 ms | **11.5x** |
| GEMM expert up 7168->2048, M=4096 | 42.6 ms (fp32, unquantized) | 94.0 ms | 24.6 ms (v2) | **1.7x** (3.8x vs v0) |
| GEMM expert down 2048->7168, M=32 | 1.05 ms (fp32) | 0.75 ms | 0.22 ms (v2) | **4.7x**, beats unscaled cuBLAS INT8 (0.31 ms) |
| Indexer decode, 4 x 32k keys | 2.76 ms (fp32 eager) | 1.70 ms | 0.34 ms (v2) | **8.1x** |
| Indexer prefill causal 4096 x 4096 | 304 ms (fp32 eager), 220 ms (cuBLAS INT8) | 214 ms | 21.2 ms (v2 + causal skip) | **14.4x** (10.4x vs cuBLAS INT8) |

Every speedup keeps DeepSeek's block scales. `torch._int_mm` (cuBLAS INT8, 11.8 TOP/s) has no block scales, so it is a
ceiling and not a usable layer. The v2 GEMM reaches ~4.9 TOP/s, about 41% of that ceiling.

## Correctness (`python -m pytest -q`)

115 tests: 67 for v0/v1 in `test_kernels.py`, 48 for v2 in `test_kernels_v2.py`.
- `act_quant` equals the PyTorch reference bit for bit (fp32, fp16, bf16 inputs; both scale formats; outliers, zeros).
- GEMM and indexer within 1e-5 relative error of a float64 reference on the same integers; odd shapes, batches, bf16 out.
- Causal masks match exactly; the indexer's top-k set matches the reference.

## Known limits

- The K-major `act_quant` (v2) is 25-55% slower than the row-major one because its stores are scattered.
- One GPU, one driver; Triton 3.2 (triton-windows), PyTorch 2.6, CUDA 12.4.
- Fast Hadamard rotation (used before indexer quantization in V3.2) is not ported.
- Timings are kernel-only (`triton.testing.do_bench`), not a full model.

## Reproduce

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install "triton-windows<3.3" pytest
python roofline.py
python -m pytest -q
python bench.py
```
