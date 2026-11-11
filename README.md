# DeepSeek-V3.2 kernels on a GTX 1650 (Turing, sm_75)

INT8 ports of the three FP8 TileLang kernels in [DeepSeek-V3.2-Exp `inference/kernel.py`](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/kernel.py),
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

## Accuracy: how much does the number format change what the indexer selects? (`experiments/`)

V3.2 rotates the indexer's q and k with a Hadamard transform, then stores them in FP8 with one scale per vector.
V3.2 itself cannot run here, so the proxy indexer uses real post-RoPE q/k from Qwen2.5-1.5B (128-dim heads, as V3.2)
and Qwen2.5-0.5B (64-dim), from a forward pass written out from the weights (`experiments/proxy.py`; next-token loss
1.3 / 1.5 on code, 2.9 / 3.2 on prose for 1.5B / 0.5B, as expected). Query heads of each key group act as indexer heads with uniform
weights; 4 texts x 8,192 tokens; 256 queries in the second half of each. "Lost" = share of selected keys outside the
full-precision top-k (tie-aware). Averages over layers where top-2048 is not trivial (13 of 28 and 12 of 24 layers;
in the rest, fewer than 2,048 keys score above 0).

| Top-2048 keys lost | 1.5B | 0.5B | 1.5B, keys centered |
|---|---|---|---|
| FP8 | 3.03% | 2.55% | 0.54% |
| FP8 + Hadamard (V3.2) | **2.87%** | **3.90%** | 0.49% |
| INT8 | 1.29% | 1.53% | 0.20% |
| INT8 + Hadamard | **0.62%** | **0.75%** | 0.12% |
| INT4 + Hadamard | 6.73% | 10.03% | 2.14% |

**1. INT8 + Hadamard loses 4-5x fewer keys than FP8 + Hadamard.** It is better in every one of the 119 non-trivial
(text, layer, key group) cells across both models. The gap holds with the outlier layer 0 removed (0.20% vs 0.93% on
1.5B, 0.48% vs 2.50% on 0.5B). With no offset at all (synthetic Gaussian keys) FP8 still loses ~4x more: 3 mantissa
bits give a step of up to 1/16 of each value, while INT8's step is 1/254 of the largest.

**2. Centering the keys helps every format.** Subtract a mean key (from the first 1,024 tokens) before quantizing,
keep it in fp32 and add `q . mean` back per query: the score function is unchanged and the cost is one dot product
per query and head. Lost keys fall 3-6x on 1.5B (better in 54 of 56 cells). Most of that is layer 0, whose keys share
an offset 40x the size of their spread; without layer 0 the gain is ~2x.

**3. Mechanism.** The size of the shared key offset predicts how much Hadamard helps INT8 (Spearman 0.92 on 1.5B,
0.89 on 0.5B) and how much centering helps (0.96, 0.93). In a synthetic test with the offset in one channel
(`indexer_mechanism.py`, chart 7), FP8 without rotation is unaffected, FP8 + Hadamard and INT8 degrade, and INT8 +
Hadamard barely moves: INT8's step is set by the largest value, FP8's by each value, and rotation spreads the offset
into every coordinate.

**Did not replicate:** on 0.5B, Hadamard made FP8 worse (2.55% -> 3.90%). On 1.5B it helps slightly (3.03% -> 2.87%,
CI crosses 0). The offset does not predict FP8's rotation penalty on real layers (Spearman 0.38 and -0.14).

**Practical size:** dropped keys are near-ties. The share of the model's real attention inside the selection moves
by at most 0.5 points (1.5B, FP8: 85.9% vs 86.4% full precision; a last-2,048-tokens window keeps 72.3%).

**Limits:** a proxy, not DeepSeek's trained indexer (its learned head weights may behave differently, and it may be
trained with FP8 in the loop); two small models from one family; 4 texts; no end-to-end quality measurement.

### Check on DeepSeek-V3.2-Exp's real weights (`ds_knorm.py`, `ds_layer0.py`, chart 8)

`ds_fetch.py` reads single tensors from the released checkpoint with HTTP range requests (safetensors headers give
each tensor's byte range), so about 100 MB is downloaded instead of the ~690 GB checkpoint.

**The key offset is a Qwen effect.** V3.2's indexer keys are `LayerNorm(wk x)`, so the LayerNorm bias would be a
shared offset. In all 61 layers plus the MTP layer it is ~1% of the key spread (|beta| / |gamma| on the no-RoPE dims:
median 0.009, max 0.026; gamma is ~1.0 everywhere).

**Real layer-0 indexer.** Layer 0's input is `RMSNorm(embedding)`, so its indexer (queries, keys, real per-head
weights, YaRN RoPE) is computed exactly in fp32 from the weights, on the same 4 windows (2 texts x 2) and queries:

| Top-2048 tokens lost, DeepSeek-V3.2-Exp layer 0 | all texts | per text |
|---|---|---|
| FP8 + Hadamard, ue8m0 scales (V3.2's setup) | **0.64%** | 0.58-0.69 |
| FP8 + Hadamard | 0.61% | 0.55-0.66 |
| INT8 | 0.20% | 0.18-0.22 |
| INT8 + Hadamard | **0.16%** | 0.14-0.17 |
| INT8 + Hadamard, keys centered | 0.14% | 0.13-0.14 |
| BF16 | 0.04% | |

INT8 + Hadamard loses 4x fewer than V3.2's own format, in every text. The key offset here is 0.7 (Qwen2.5-1.5B layer
0: 40), so centering gains only ~15%. Limits: layer 0 only; the fp32 reference follows DeepSeek's inference code but is
not checked against a run of their stack; no attention-mass metric (it would need layer 0's main attention).

### Layers 0-3, with the main attention (`ds_layers.py`, chart 9)

Layers 0-2 of V3.2 are dense, so the residual stream into layers 1-3 is computed exactly from ~1.8 GB of fp8 weights
(MLA attention over the fp32 indexer's top-2048, fp8 KV-cache simulation, dense SwiGLU MLP; RoPE and dequantization
checked against DeepSeek's code). Each window now starts with BOS. Mean over the same 4 windows, 256 queries each:

| | Layer 0 | Layer 1 | Layer 2 | Layer 3 |
|---|---|---|---|---|
| **Top-2048 tokens lost** | | | | |
| FP8 + Hadamard, ue8m0 (V3.2's setup) | 0.64% | 1.79% | 1.63% | 1.10% |
| INT8 + Hadamard | 0.16% | 0.43% | 0.39% | 0.27% |
| BF16 | 0.04% | 0.12% | 0.11% | 0.08% |
| **Attention output error** (real sparse attention vs fp32-indexer selection) | | | | |
| FP8 + Hadamard, ue8m0 | 0.21% | 0.40% | 1.03% | 0.59% |
| INT8 + Hadamard | 0.11% | 0.18% | 0.44% | 0.27% |
| BF16 | 0.05% | 0.08% | 0.21% | 0.13% |
| **Dense attention weight held by the 2048 selected tokens** | | | | |
| fp32 indexer | 97.2% | 94.4% | 71.4% | 84.0% |
| oracle (top-2048 by attention, per head) | 99.8% | 98.1% | 85.1% | 91.4% |
| last 2048 tokens | 41.7% | 54.2% | 65.0% | 55.4% |

- INT8 + Hadamard loses 3.9-4.3x fewer tokens than V3.2's format in all 16 layer-windows; the gap from layer 0 holds
  deeper in the model, where both lose about 2.5x more than at layer 0.
- The output error is about 2x lower with INT8 (lower in 15 of 16 layer-windows). The swapped tokens are near the
  cut-off: every format keeps the attention weight within 0.02 points of fp32.
- Context: all of this is small next to the indexer's own approximation. The fp32 indexer misses 3-14 points of
  attention weight that the oracle keeps, and using the last 2048 tokens instead costs 3.5-18% output error.
  So V3.2's FP8 choice costs little in absolute terms; INT8 is a cheaper format (on GPUs with fast INT8) that costs less.
- Sanity check of the forward pass: at every layer the fp32 indexer's selection holds far more attention weight than the
  last 2048 tokens, as an indexer trained to match the attention should; a wrong residual stream would collapse this.
- Limits: 4 of 61 layers (layer 4 on needs the 256 experts); fp32 activations in the linear layers (V3.2 serves them in
  fp8); one sequence length (8,192); output error before `o_proj`.

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

# accuracy experiments (need: pip install safetensors tokenizers pyarrow huggingface_hub transformers)
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct
huggingface-cli download Yelp/yelp_review_full --repo-type dataset
cd experiments
python indexer_quant.py Qwen2.5-0.5B-Instruct
python indexer_quant.py Qwen2.5-1.5B-Instruct
python indexer_mechanism.py 0.5b 1.5b
# DeepSeek-V3.2-Exp checks: weight index and tokenizer first (~17 MB), then tensors are fetched on demand
mkdir -p ds_weights && for f in model.safetensors.index.json:index.json tokenizer.json:tokenizer.json; do
  curl -sL -o ds_weights/${f#*:} https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/resolve/main/${f%%:*}; done
python ds_knorm.py
python ds_layer0.py
python ds_layers.py      # downloads ~1.8 GB of fp8 weights once; ~7 min on a GTX 1650
cd .. && python make_charts.py
```

The code text is `transformers/modeling_utils.py` from the installed package (read as a file, not imported), so exact
numbers depend on the transformers version (5.22.1 here).

## License

MIT. The kernels follow the structure of DeepSeek's TileLang kernels in DeepSeek-V3.2-Exp (MIT).
