"""Is the INT8 indexer still faster once the Hadamard rotation is paid for? python bench_indexer_e2e.py

Post 4 claimed INT8 + Hadamard is both more accurate and cheaper than V3.2's FP8 indexer on GPUs with fast INT8. The
accuracy used the rotation; the speed was the indexer kernel alone. Here every step is timed (V3.2 shapes: 64 heads x 128):
  fp32           eager fp32 indexer on float q and the float key cache (reference.index_torch_fp32, no rotation needed)
  int8 kernel    int8_index_v2 alone, inputs already quantized (what post 4 timed)
  int8 + rotate  rotate+quantize q (fused kernel), fold the q scales into the head weights, rotate+quantize the new
                 key(s) into the K-major cache, then int8_index_v2: a full indexer step from float inputs
  int8 + rotate (cuBLAS)  the rotation as a cuBLAS fp32 matmul followed by act_quant_kmajor (B=1 only). On this GPU
                 it beats both fused Triton kernels (matrix and butterfly): their in-register shuffles cost more than
                 the memory pass they save
Also checks the full INT8 + rotate path against fp32 scores: top-2048 overlap per query (prefill) / per sequence (decode).
"""
import json, os, time
import torch, triton
from hadamard_quant import rotate_quant_kmajor, hadamard_matrix
from kernels_v2 import int8_index_v2, act_quant_kmajor
import reference as R

dev = "cuda"; H, D = 64, 128; HAD = hadamard_matrix(device=dev); torch.manual_seed(0)
bench = lambda f: triton.testing.do_bench(f, warmup=25, rep=200)
rows = []
def rec(case, path, ms, extra=None):
    rows.append(dict(case=case, path=path, ms=round(ms, 4), **(extra or {}))); print(f"  {case:34s} {path:16s} {ms:9.3f} ms", flush=True)

def tune(qT, w, kT, ks, M, causal):
    cfgs = [(bq, bn, wp, u) for bq in (1, 2) for bn in (32, 64, 128) for wp in (1, 2, 4) for u in (4, 8)
            if 64 <= bq * H * bn // (32 * wp) <= 256 and not (M == 1 and bq > 1)]
    return min((bench(lambda: int8_index_v2(qT, w, kT, ks, H, causal=causal, block_q=bq, block_n=bn, unroll=u, num_warps=wp)), (bq, bn, wp, u))
               for bq, bn, wp, u in cfgs)[1]

def realistic(B, n, heads=None):
    """Float vectors with a few outlier channels, as real indexer keys have (the reason V3.2 rotates)."""
    shape = (B, n, heads, D) if heads else (B, n, D); x = torch.randn(shape, device=dev)
    x[..., :4] *= 20; return x

CASES = [("decode B=4 M=1 N=32768", 4, 1, 32768, False), ("decode B=1 M=1 N=131072", 1, 1, 131072, False),
         ("prefill B=1 M=N=4096 causal", 1, 4096, 4096, True)]
for name, B, M, N, causal in CASES:
    print(name, flush=True)
    q = realistic(B, M, H); k = realistic(B, N); w = torch.randn(B, M, H, device=dev).abs() / H ** 0.5
    # fp32 reference
    ones = torch.ones(B, N, device=dev)
    rec(name, "fp32", bench(lambda: R.index_torch_fp32(q, w, k, ones)))
    ref = R.index_torch_fp32(q, w, k, ones)
    # INT8: the key cache is quantized once, as tokens arrive; per step only the new key(s) are
    kT, ks = rotate_quant_kmajor(k)
    qT, qs = rotate_quant_kmajor(q.view(B, M * H, D)); wq = (w * qs.view(B, M, H)).contiguous()
    cfg = tune(qT, wq, kT, ks, M, causal); bq, bn, wp, u = cfg
    idx = lambda qT, wq: int8_index_v2(qT, wq, kT, ks, H, causal=causal, block_q=bq, block_n=bn, unroll=u, num_warps=wp)
    t_kernel = bench(lambda: idx(qT, wq)); rec(name, "int8 kernel", t_kernel, dict(cfg=cfg))
    new = M if causal else 1  # keys added this step: the whole prompt in prefill, one token per sequence in decode
    knew = k[:, -new:].contiguous()
    def step():
        qT2, qs2 = rotate_quant_kmajor(q.view(B, M * H, D)); w2 = w * qs2.view(B, M, H)
        kn, kns = rotate_quant_kmajor(knew); kT[:, :, N - new:] = kn; ks[:, N - new:] = kns
        return idx(qT2, w2)
    t_step = bench(step); rec(name, "int8 + rotate", t_step)
    if B == 1:  # rotation as a cuBLAS fp32 matmul + act_quant_kmajor (same K-major layout when B = 1)
        def step2():
            qT2, qs2 = act_quant_kmajor((q.view(M * H, D) @ HAD).contiguous()); w2 = w * qs2.view(B, M, H)
            kn, kns = act_quant_kmajor((knew.view(new, D) @ HAD).contiguous()); kT[0, :, N - new:] = kn; ks[0, N - new:] = kns[:, 0]
            return idx(qT2[None], w2)
        t2 = bench(step2); rec(name, "int8 + rotate (cuBLAS)", t2); t_step = min(t_step, t2)
    t_rq = bench(lambda: (rotate_quant_kmajor(q.view(B, M * H, D)), rotate_quant_kmajor(knew))); rec(name, "  rotate+quant only", t_rq)
    if B == 1:
        two = lambda: (act_quant_kmajor((q.view(M * H, D) @ HAD).contiguous()), act_quant_kmajor((knew.view(new, D) @ HAD).contiguous()))
        rec(name, "  two-kernel rotate+quant", bench(two))
    # accuracy of the full path vs fp32: top-k overlap of the selected keys
    out = step(); K = min(2048, N)
    if causal:
        sl = slice(M - 512, M)  # late queries have > 2048 visible keys
        a = ref[0, sl].topk(K, -1).indices; b = out[0, sl].topk(K, -1).indices
        ov = torch.stack([torch.isin(b[i], a[i]).float().mean() for i in range(len(a))]).mean().item()
    else:
        a = ref[:, 0].topk(K, -1).indices; b = out[:, 0].topk(K, -1).indices
        ov = torch.stack([torch.isin(b[i], a[i]).float().mean() for i in range(B)]).mean().item()
    t_fp32 = [r["ms"] for r in rows if r["case"] == name and r["path"] == "fp32"][0]
    rows.append(dict(case=name, path="summary", topk_overlap=round(ov, 4), speedup=round(t_fp32 / t_step, 2), rotation_share=round(t_rq / t_step, 3)))
    print(f"  top-{K} overlap with fp32: {100 * ov:.2f}%   speedup best int8+rotate vs fp32: {t_fp32 / t_step:.2f}x"
          f"   rotation share (fused timing): {100 * t_rq / t_step:.1f}%", flush=True)
    del q, k, kT, ref, out; torch.cuda.empty_cache()

os.makedirs("results", exist_ok=True)
json.dump(dict(device=torch.cuda.get_device_name(0), torch=torch.__version__, triton=triton.__version__, date=time.strftime("%Y-%m-%d"), rows=rows),
          open("results/indexer_e2e.json", "w"), indent=1)
print("wrote results/indexer_e2e.json")
