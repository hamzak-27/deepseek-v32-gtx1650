"""Why does Hadamard rotation help INT8 but hurt FP8 in the indexer, and why does centering the keys help everything?

Hypothesis: in some layers the keys share a large offset (a mean key) concentrated in a few channels (Qwen has a k_proj
bias; RoPE leaves the non-rotating low-frequency channels nearly constant).
  - INT8 has one uniform step per vector, set by the largest value. A big offset channel makes that step coarse for
    every channel, so rotation (which spreads the offset and lowers the peak) helps.
  - FP8's step is relative to each value. Without rotation the offset channel is coarse but the small, informative
    channels stay fine. Rotation mixes the offset into every coordinate, so every coordinate becomes large and coarse.
  - Subtracting the mean key removes the offset before either format sees it.

Part A tests this on the real layers (results/indexer_quant_<model>.json). Part B isolates it: random keys plus an
offset of controlled size, either in one channel or spread over all channels.

Run: python experiments/indexer_mechanism.py [0.5b|1.5b ...]   (writes results/indexer_mechanism.json)
"""
import json, os, sys
import torch
from proxy import hadamard, quant, index_score, dev

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = {}


def spearman(a, b):
    ra = torch.tensor(a).argsort().argsort().float(); rb = torch.tensor(b).argsort().argsort().float()
    return torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()


# ---------------- Part A: real layers ----------------
for tag in (sys.argv[1:] or ["0.5b"]):
    d = json.load(open(os.path.join(HERE, "results", f"indexer_quant_{tag}.json")))
    K = 2048
    rec = {(r["seq"], r["layer"], r["group"], r["fmt"], r["rot"], r["center"]): r["recall"]
           for r in d["rows"] if r["k"] == K}
    cells = [(s["seq"], s["layer"], s["group"]) for s in d["layer_stats"]]
    st = {(s["seq"], s["layer"], s["group"]): s for s in d["layer_stats"]}
    lost = lambda c, f, r, ce: 100 * (1 - rec[c + (f, r, ce)])
    # non-trivial cells only: some variant loses something
    nt = [c for c in cells if any(rec[c + v] < 1 for v in [("int4", "none", "none"), ("fp8", "hadamard", "none")])]
    fp8_pen = [lost(c, "fp8", "hadamard", "none") - lost(c, "fp8", "none", "none") for c in nt]
    int8_gain = [lost(c, "int8", "none", "none") - lost(c, "int8", "hadamard", "none") for c in nt]
    center_gain_fp8 = [lost(c, "fp8", "hadamard", "none") - lost(c, "fp8", "hadamard", "k") for c in nt]
    off = [st[c]["k_offset"] for c in nt]; conc = [st[c]["k_offset_peak"] for c in nt]
    pk = [st[c]["k_peak"] for c in nt]; pkh = [st[c]["k_peak_had"] for c in nt]
    res = dict(cells=len(nt),
               spearman_offset_vs_fp8_hadamard_penalty=spearman(off, fp8_pen),
               spearman_offset_vs_int8_hadamard_gain=spearman(off, int8_gain),
               spearman_offset_vs_centering_gain_fp8=spearman(off, center_gain_fp8),
               spearman_offset_x_concentration_vs_fp8_penalty=spearman([o * c for o, c in zip(off, conc)], fp8_pen),
               mean_k_peak_over_rms=sum(pk) / len(pk), mean_k_peak_over_rms_after_hadamard=sum(pkh) / len(pkh))
    per_layer = []
    for l in range(d["layers"]):
        cs = [c for c in nt if c[1] == l]
        if not cs: continue
        avg = lambda xs: sum(xs) / len(xs)
        per_layer.append(dict(layer=l, k_offset=avg([st[c]["k_offset"] for c in cs]),
                              k_offset_peak=avg([st[c]["k_offset_peak"] for c in cs]),
                              fp8_none=avg([lost(c, "fp8", "none", "none") for c in cs]),
                              fp8_had=avg([lost(c, "fp8", "hadamard", "none") for c in cs]),
                              int8_none=avg([lost(c, "int8", "none", "none") for c in cs]),
                              int8_had=avg([lost(c, "int8", "hadamard", "none") for c in cs]),
                              fp8_had_centered=avg([lost(c, "fp8", "hadamard", "k") for c in cs])))
    res["per_layer"] = per_layer; out[tag] = res
    print(f"\n== Part A, Qwen2.5-{tag}, top-{K}, {len(nt)} non-trivial (seq, layer, group) cells ==")
    for k_, v in res.items():
        if k_ != "per_layer": print(f"  {k_:52s} {v:.3f}" if isinstance(v, float) else f"  {k_:52s} {v}")
    print(f"  {'layer':>5s} {'offset':>7s} {'1-chan':>7s} | lost %: {'fp8':>6s} {'fp8+H':>6s} {'int8':>6s} {'int8+H':>6s} {'fp8+H+c':>7s}")
    for p in per_layer:
        print(f"  {p['layer']:5d} {p['k_offset']:7.2f} {p['k_offset_peak']:7.2f} |         {p['fp8_none']:6.2f} "
              f"{p['fp8_had']:6.2f} {p['int8_none']:6.2f} {p['int8_had']:6.2f} {p['fp8_had_centered']:7.2f}")

# ---------------- Part B: synthetic, offset size and shape controlled ----------------
torch.manual_seed(0)
D, H, NK, P, K = 128, 6, 8192, 256, 512
HAD = hadamard(D)
noise_k = torch.randn(NK, D, device=dev); q = torch.randn(H, P, D, device=dev)
dense = torch.randn(D, device=dev); dense /= dense.norm()
sparse = torch.zeros(D, device=dev); sparse[0] = 1.0
syn = []
print(f"\n== Part B: synthetic keys = N(0, I) + c * sqrt({D}) * offset direction; top-{K} of {NK}; lost % ==")
print(f"  {'offset':7s} {'c':>4s} | {'fp8':>6s} {'fp8+H':>6s} {'int8':>6s} {'int8+H':>6s} | centered: {'fp8+H':>6s} {'int8+H':>6s}")
for shape, u in (("1 chan", sparse), ("dense", dense)):
    for c in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0):
        k = noise_k + c * D ** 0.5 * u
        ref = index_score(q, k); tau = ref.topk(K, -1).values[:, -1:]
        row = dict(offset=shape, c=c)
        kmean = k[:1024].mean(0)
        for fmt in ("fp8", "int8"):
            for rot in ("none", "hadamard"):
                for center in ("none", "k"):
                    R = HAD if rot == "hadamard" else torch.eye(D, device=dev)
                    kin = k - kmean if center == "k" else k
                    kd = quant(kin @ R, fmt) @ R.T + (kmean if center == "k" else 0)
                    qd = quant(q @ R, fmt) @ R.T
                    ix = index_score(qd, kd).topk(K, -1).indices
                    row[f"{fmt}_{rot}_{center}"] = 100 * (1 - (ref.gather(1, ix) >= tau).float().mean().item())
        syn.append(row)
        print(f"  {shape:7s} {c:4.2f} | {row['fp8_none_none']:6.2f} {row['fp8_hadamard_none']:6.2f} "
              f"{row['int8_none_none']:6.2f} {row['int8_hadamard_none']:6.2f} | {row['fp8_hadamard_k']:15.2f} "
              f"{row['int8_hadamard_k']:6.2f}")
out["synthetic"] = syn
json.dump(out, open(os.path.join(HERE, "results", "indexer_mechanism.json"), "w"), indent=1)
