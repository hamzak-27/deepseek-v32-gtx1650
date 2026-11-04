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

## Accuracy: does INT8 instead of FP8 change what the indexer selects? (`experiments/indexer_quant.py`)

V3.2 rotates the indexer's q and k with a Hadamard transform, then stores them in FP8 with one scale per vector.
Proxy setup: real post-RoPE q/k from Qwen2.5-0.5B (forward pass written from the weights, next-token loss 1.5 on
code, 3.2 on prose), 7 query heads per key group used as indexer heads, uniform head weights, 4 texts x 8,192 tokens,
256 queries in the second half of each. "Lost" = share of selected keys outside the full-precision top-k (tie-aware).
Averages over the 12 of 24 layers where top-2048 is not trivial (in the other 12, fewer than 2,048 keys score above 0).

| Format, top-2048 | No rotation | Hadamard |
|---|---|---|
| BF16 | 0.24% lost | 0.28% |
| FP8 e4m3 | 2.55% | **3.90%** (what V3.2 ships) |
| FP8, power-of-2 scales | 2.97% | 4.08% |
| INT8 | 1.53% | **0.75%** |
| INT6 | 5.68% | 2.95% |
| INT4 | 15.5% | 10.0% |

- INT8 beats FP8 in every pairing. INT8 + Hadamard is better than FP8 in 63 of 63 non-tied cells at k=2048
  (95% bootstrap CI of the recall gain: +0.6 to +1.3 points over all cells), identical across code and prose.
- Hadamard helps every integer format, never hurts INT8 in any cell, and is net negative for FP8. That FP8 loss
  comes mostly from layers 0 and 8; in other layers it is mixed. I have not explained why.
- The practical effect is small here: the keys a format drops are near-ties, so the share of the model's real
  attention mass inside the selection moves by at most 0.05 points for INT8 and FP8 (85.2% at k=2048; a sliding window
  of the last 2,048 tokens keeps 64.0%). INT4 + Hadamard keeps 84.6%.
- Limits: a proxy, not DeepSeek's trained indexer (its q/k, head weights and 128 dims may behave differently;
  it may also have been trained with FP8 in the loop); one 0.5B model; 4 texts; no end-to-end quality measurement.

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
