"""Benchmark the INT8 kernels against PyTorch baselines on DeepSeek-V3.2 layer shapes.

Writes results/bench.json and results/bench.md. Run: python bench.py [--quick]
"""
import json, os, sys, time
import torch, triton

from kernels import act_quant, int8_gemm, int8_index
from kernels_v2 import act_quant_kmajor, kmajor, int8_gemm_v2, int8_index_v2, index_kmajor_q, index_kmajor_k
import reference as R

torch.backends.cuda.matmul.allow_tf32 = False
dev = "cuda"; QUICK = "--quick" in sys.argv
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"); os.makedirs(OUT, exist_ok=True)
rows = []


def bench(fn):
    return triton.testing.do_bench(fn, warmup=25, rep=100 if QUICK else 300)


def record(kernel, shape, impl, ms, work, unit):
    rate = work / ms / (1e6 if unit == "GB/s" else 1e9)  # work per ms -> per second, in G (bytes) or T (ops)
    rows.append(dict(kernel=kernel, shape=shape, impl=impl, ms=round(ms, 4), rate=round(rate, 2), unit=unit))
    print(f"{kernel:10s} {shape:28s} {impl:30s} {ms:9.3f} ms  {rate:8.2f} {unit}", flush=True)


# ---------------- act_quant: memory bound, report GB/s moved ----------------
for M in ([16, 4096] if QUICK else [16, 256, 2048, 8192]):
    for dt in (torch.float32, torch.bfloat16):
        x = torch.randn(M, 7168, device=dev, dtype=dt)
        moved = x.numel() * (x.element_size() + 1) + x.numel() // 128 * 4
        shp = f"{M}x7168 {str(dt)[6:]}"
        record("act_quant", shp, "torch eager", bench(lambda: R.act_quant_torch(x)), moved, "GB/s")
        record("act_quant", shp, "triton int8", bench(lambda: act_quant(x)), moved, "GB/s")
        record("act_quant", shp, "triton int8 K-major (v2)", bench(lambda: act_quant_kmajor(x)), moved, "GB/s")

# ---------------- int8_gemm on DeepSeek linear shapes (N, K) ----------------
gemm_shapes = {"q_a 7168->1536": (1536, 7168), "expert up 7168->2048": (2048, 7168),
               "expert down 2048->7168": (7168, 2048)}
for name, (N, K) in gemm_shapes.items():
    W = torch.randn(N, K, device=dev) * 0.02
    Wb = W.view(N // 128, 128, K // 128, 128); ws = Wb.abs().amax((1, 3)).clamp_min(1e-4) / 127
    Wq = torch.round(Wb / ws[:, None, :, None]).clamp(-127, 127).to(torch.int8).view(N, K)
    Wdeq = (Wq.float().view(N // 128, 128, K // 128, 128) * ws[:, None, :, None]).view(N, K)
    WqT = Wq.T.contiguous(); W16 = W.half(); WkT = kmajor(Wq)
    for M in ([32, 2048] if QUICK else [32, 128, 1024, 4096]):
        x = torch.randn(M, K, device=dev); xq, xs = act_quant(x); x16 = x.half(); xT = kmajor(xq)
        flops = 2 * M * N * K; shp = f"{name} M={M}"
        record("gemm", shp, "torch fp32 (unquantized)", bench(lambda: R.gemm_fp32(x, W)), flops, "TFLOP/s")
        record("gemm", shp, "torch fp16 (unquantized)", bench(lambda: x16 @ W16.T), flops, "TFLOP/s")
        record("gemm", shp, "torch dequant + fp32", bench(lambda: R.gemm_dequant_fp32(xq, xs, Wdeq)), flops, "TFLOP/s")
        record("gemm", shp, "torch _int_mm (no scales)", bench(lambda: R.int_mm_unscaled(xq, WqT)), flops, "TOP/s")
        record("gemm", shp, "triton int8 v0", bench(lambda: int8_gemm(xq, xs, Wq, ws, version="v0")), flops, "TOP/s")
        int8_gemm(xq, xs, Wq, ws, version="v1")  # autotune outside the timer
        record("gemm", shp, "triton int8 v1", bench(lambda: int8_gemm(xq, xs, Wq, ws, version="v1")), flops, "TOP/s")
        int8_gemm_v2(xT, xs, WkT, ws)
        record("gemm", shp, "triton int8 v2 (dp4a)", bench(lambda: int8_gemm_v2(xT, xs, WkT, ws)), flops, "TOP/s")

# ---------------- int8_index: lightning indexer, 64 heads x 128 dims ----------------
H, D = 64, 128
index_cases = ([("decode", 4, 1, 16384, False), ("prefill", 1, 2048, 2048, True)] if QUICK else
               [("decode", 4, 1, 4096, False), ("decode", 4, 1, 32768, False), ("decode", 1, 1, 131072, False),
                ("prefill", 1, 1024, 1024, True), ("prefill", 1, 4096, 4096, True)])
sweep = [(bq, bn, w) for bq in (1, 2, 4) for bn in (32, 64, 128) for w in (4, 8) if bq * H * bn <= 32768]
for kind, B, M, N, causal in index_cases:
    q = torch.randint(-127, 128, (B, M, H, D), device=dev, dtype=torch.int8)
    k = torch.randint(-127, 128, (B, N, D), device=dev, dtype=torch.int8)
    w = torch.randn(B, M, H, device=dev) * 1e-4; ks = torch.rand(B, N, device=dev) * 0.01
    ops = 2 * B * M * N * H * D; shp = f"{kind} B={B} M={M} N={N}"
    if M * N * H * 4 * B < 3e9 or M > 1:
        record("index", shp, "torch fp32 (chunked)", bench(lambda: R.index_torch_fp32(q, w, k, ks)), ops, "TOP/s")
    record("index", shp, "torch int8 _int_mm", bench(lambda: R.index_torch_int8(q, w, k, ks)), ops, "TOP/s")
    record("index", shp, "triton v0 (DeepSeek tile)", bench(lambda: int8_index(q, w, k, ks, block_q=1, block_n=128)), ops, "TOP/s")
    best = min(((bench(lambda: int8_index(q, w, k, ks, block_q=bq, block_n=bn, num_warps=wp)), (bq, bn, wp))
                for bq, bn, wp in sweep if not (M == 1 and bq > 1)))
    record("index", shp, f"triton v1 tuned q{best[1][0]} n{best[1][1]} w{best[1][2]}", best[0], ops, "TOP/s")
    if causal:
        bq, bn, wp = best[1]
        record("index", shp, "triton v1 + causal skip",
               bench(lambda: int8_index(q, w, k, ks, causal=True, block_q=bq, block_n=bn, num_warps=wp)), ops, "TOP/s")
    qT, kT = index_kmajor_q(q), index_kmajor_k(k)
    sweep2 = [(bq, bn, wp, u) for bq in (1, 2) for bn in (32, 64, 128) for wp in (1, 2, 4) for u in (4, 8)
              if 64 <= bq * H * bn // (32 * wp) <= 256 and not (M == 1 and bq > 1)]
    best2 = min(((bench(lambda: int8_index_v2(qT, w, kT, ks, H, block_q=bq, block_n=bn, unroll=u, num_warps=wp)),
                  (bq, bn, wp, u)) for bq, bn, wp, u in sweep2))
    bq, bn, wp, u = best2[1]
    record("index", shp, f"triton v2 dp4a q{bq} n{bn} w{wp} u{u}", best2[0], ops, "TOP/s")
    if causal:
        record("index", shp, "triton v2 dp4a + causal skip",
               bench(lambda: int8_index_v2(qT, w, kT, ks, H, causal=True, block_q=bq, block_n=bn, unroll=u,
                                           num_warps=wp)), ops, "TOP/s")
    del q, k; torch.cuda.empty_cache()

json.dump(dict(device=torch.cuda.get_device_name(0), torch=torch.__version__, triton=triton.__version__,
               date=time.strftime("%Y-%m-%d"), rows=rows), open(os.path.join(OUT, "bench.json"), "w"), indent=1)
with open(os.path.join(OUT, "bench.md"), "w") as f:
    f.write(f"# Benchmarks on {torch.cuda.get_device_name(0)}\n\n| kernel | shape | impl | ms | rate |\n|---|---|---|---|---|\n")
    for r in rows: f.write(f"| {r['kernel']} | {r['shape']} | {r['impl']} | {r['ms']} | {r['rate']} {r['unit']} |\n")
print("wrote results/bench.json and results/bench.md")
