"""Charts for posts, from results/bench.json and the measurements in roofline.py / the dp4a prototypes.
Writes charts/*.png. Run: python make_charts.py"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "charts"); os.makedirs(OUT, exist_ok=True)
rows = json.load(open(os.path.join(HERE, "results", "bench.json")))["rows"]

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"  # reference categorical order, validated light
BASE = "#b9b8b2"  # baselines: neutral, so "ours" is the only colored series
FOOT = "GTX 1650 (Turing, 4 GB) · PyTorch 2.6 · Triton 3.2 · kernel time via triton.testing.do_bench"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def get(kernel, shape, impl_prefix):
    for r in rows:
        if r["kernel"] == kernel and r["shape"] == shape and r["impl"].startswith(impl_prefix):
            return r
    raise KeyError((kernel, shape, impl_prefix))


def frame(title, subtitle, size=(12, 6.75), foot=FOOT):
    fig = plt.figure(figsize=size, dpi=150)
    fig.text(0.04, 0.94, title, fontsize=21, weight="bold", color=INK, va="top")
    fig.text(0.04, 0.875, subtitle, fontsize=13.5, color=INK2, va="top")
    fig.text(0.04, 0.025, foot, fontsize=10, color=MUTED)
    return fig


def hbars(ax, labels, values, colors, fmt, xlabel):
    y = range(len(labels))[::-1]
    ax.barh(list(y), values, color=colors, height=0.62, edgecolor=SURFACE, linewidth=2)
    ax.set_yticks(list(y)); ax.set_yticklabels(labels, fontsize=13, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=1); ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    vmax = max(values)
    for yi, v in zip(y, values):
        ax.text(v + vmax * 0.012, yi, fmt(v), va="center", fontsize=12.5, color=INK)
    ax.set_xlim(0, vmax * 1.16); ax.set_xlabel(xlabel)


# 1. The GPU's real ceilings ---------------------------------------------------------------------------------
fig = frame("On a GTX 1650, FP16 is 5x slower than FP32. INT8 is 6.4x faster.",
            "Measured matrix-multiply throughput, 4096 x 4096 x 4096, cuBLAS through PyTorch")
ax = fig.add_axes([0.2, 0.14, 0.74, 0.66])
labels = ["INT8  (torch._int_mm)", "FP32", "BF16  (emulated)", "FP16"]
vals = [11.21, 1.74, 1.54, 0.33]  # roofline.py, same session as bench.json
hbars(ax, labels, vals, [S1, BASE, BASE, BASE], lambda v: f"{v:.2f}", "Tera-operations per second (higher is better)")
ax.text(0.33 + 11.21 * 0.09, 0, "← half precision: 0.19x of FP32 on this card", va="center", fontsize=12, color=INK2)
fig.savefig(os.path.join(OUT, "1_gpu_formats.png")); plt.close(fig)

# 2. Lightning indexer, prefill 4096 -------------------------------------------------------------------------
shp = "prefill B=1 M=4096 N=4096"
items = [("PyTorch FP32 (eager)", get("index", shp, "torch fp32"), BASE),
         ("PyTorch + cuBLAS INT8", get("index", shp, "torch int8"), BASE),
         ("Triton v0  (DeepSeek's tiling)", get("index", shp, "triton v0"), S1),
         ("v1  + skip masked tiles", get("index", shp, "triton v1 + causal"), S1),
         ("v2  dp4a, K-major", get("index", shp, "triton v2 dp4a q"), S1),
         ("v2  + skip masked tiles", get("index", shp, "triton v2 dp4a + causal"), S1)]
base = items[0][1]["ms"]
fig = frame(f"DeepSeek-V3.2's indexer: {base / items[-1][1]['ms']:.1f}x faster on a laptop GPU",
            f"4,096 tokens, 64 heads x 128 dims, causal. {items[1][1]['ms'] / items[-1][1]['ms']:.1f}x faster than "
            "PyTorch + cuBLAS INT8. Lower is better")
ax = fig.add_axes([0.27, 0.14, 0.67, 0.66])
hbars(ax, [i[0] for i in items], [i[1]["ms"] for i in items], [i[2] for i in items],
      lambda v: f"{v:.0f} ms  ({base / v:.1f}x)" if v < base else f"{v:.0f} ms", "Milliseconds")
fig.savefig(os.path.join(OUT, "2_indexer_prefill.png")); plt.close(fig)

# 3. The dp4a story: one plain INT8 GEMM, 2048^3 --------------------------------------------------------------
fig = frame("Why my INT8 kernel was stuck, and what fixed it",
            "INT8 matrix multiply, one change per step, best tested tile at each step. 2048³ except the first row")
ax = fig.add_axes([0.36, 0.14, 0.58, 0.66])
steps = ["Triton tl.dot on int8 (DeepSeek shapes)", "+ call dp4a directly (inline PTX)", "+ store operands K-major",
         "+ ~128 accumulators per thread", "cuBLAS INT8 (reference)"]
vals = [1.28, 2.00, 2.81, 6.70, 9.27]  # bench.json v0 ceiling; dp4a_proto.py / dp4a_proto2.py sweeps
hbars(ax, steps, vals, [S1, S1, S1, S1, BASE], lambda v: f"{v:.2f} TOP/s", "Tera-operations per second (higher is better)")
fig.savefig(os.path.join(OUT, "3_dp4a_story.png")); plt.close(fig)

# 4. Block-scaled GEMM across batch size, expert up-projection 7168 -> 2048 ------------------------------------
name = "expert up 7168->2048"; Ms = [32, 128, 1024, 4096]
series = [("cuBLAS INT8, no block scales (ceiling)", "torch _int_mm", S2, "s", "--"),
          ("Ours: v2 dp4a, block-scaled", "triton int8 v2", S1, "o", "-"),
          ("PyTorch FP32, unquantized", "torch fp32", S3, "^", "-"),
          ("Ours: v0 Triton tl.dot", "triton int8 v0", S4, "D", "-")]
fig = frame("Block-scaled INT8 matmul: 1.7x over FP32, 3.8x over v0",
            "DeepSeek-V3.2 MoE expert up-projection (7168 → 2048), 128 x 128 block scales. Throughput vs tokens per batch")
ax = fig.add_axes([0.08, 0.14, 0.6, 0.66])
for label, impl, c, mk, ls in series:
    ys = [get("gemm", f"{name} M={m}", impl)["rate"] for m in Ms]
    ax.plot(Ms, ys, color=c, marker=mk, markersize=9, linewidth=2, linestyle=ls,
            markeredgecolor=SURFACE, markeredgewidth=2, label=label)
    ax.text(Ms[-1] * 1.25, ys[-1], f"{label}  {ys[-1]:.1f}", va="center", fontsize=11.5, color=INK)
ax.set_xscale("log", base=2); ax.set_xticks(Ms); ax.set_xticklabels([str(m) for m in Ms])
ax.set_xlabel("Tokens per batch (M)"); ax.set_ylabel("Tera-operations per second")
ax.yaxis.grid(True, color=GRID, linewidth=1); ax.set_axisbelow(True); ax.set_ylim(0, 13)
ax.legend(loc="upper left", frameon=False, fontsize=11.5)
fig.savefig(os.path.join(OUT, "4_gemm_scaling.png")); plt.close(fig)
# 5-7. Indexer precision experiments (experiments/indexer_quant.py, indexer_mechanism.py) ------------------------
PROXY = ("Proxy, not DeepSeek's trained indexer: real post-RoPE q/k from Qwen2.5, uniform head weights, "
         "4 texts x 8,192 tokens, 256 queries each")
IQ = {t: json.load(open(os.path.join(HERE, "results", f"indexer_quant_{t}.json"))) for t in ("0.5b", "1.5b")
      if os.path.exists(os.path.join(HERE, "results", f"indexer_quant_{t}.json"))}


def lost(d, fmt, rot, center="none", K=2048):
    R = [r for r in d["rows"] if r["k"] == K]
    triv = {l for l in range(d["layers"]) if all(r["recall"] == 1.0 for r in R if r["layer"] == l and r["fmt"] != "window")}
    s = [r["recall"] for r in R if (r["fmt"], r["rot"], r["center"]) == (fmt, rot, center) and r["layer"] not in triv]
    return 100 * (1 - sum(s) / len(s))


def grouped(ax, rows, series, xmax, xlabel, legend="upper right"):
    """rows: [label]; series: [(legend, color, [values per row])]."""
    ys = list(range(len(rows)))[::-1]; hgt = 0.8 / len(series)
    for i, (lab, c, vals) in enumerate(series):
        off = (len(series) - 1) / 2 * hgt - i * hgt
        ax.barh([y + off for y in ys], vals, height=hgt, color=c, edgecolor=SURFACE, linewidth=2, label=lab)
        for y, v in zip(ys, vals):
            ax.text(v + xmax * 0.01, y + off, f"{v:.2f}%", va="center", fontsize=11.5, color=INK)
    ax.set_yticks(ys); ax.set_yticklabels(rows, fontsize=13, color=INK); ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False); ax.xaxis.grid(True, color=GRID, linewidth=1); ax.set_axisbelow(True)
    ax.set_xlim(0, xmax); ax.set_xlabel(xlabel); ax.legend(loc=legend, frameon=False, fontsize=12)


if len(IQ) == 2:
    # 5. Post 1: INT8 vs FP8, both models
    cfgs = [("FP8", "fp8", "none"), ("FP8 + Hadamard\n(DeepSeek-V3.2's setup)", "fp8", "hadamard"),
            ("INT8", "int8", "none"), ("INT8 + Hadamard", "int8", "hadamard")]
    series = [(f"Qwen2.5-1.5B (128-dim heads, as V3.2)", S1, [lost(IQ["1.5b"], f, r) for _, f, r in cfgs]),
              (f"Qwen2.5-0.5B (64-dim heads)", S2, [lost(IQ["0.5b"], f, r) for _, f, r in cfgs])]
    fig = frame("INT8 drops 4-5x fewer of the indexer's top keys than FP8",
                "DeepSeek-V3.2's indexer quantization on real attention vectors: share of the full-precision top-2048 "
                "keys lost", foot=PROXY)
    ax = fig.add_axes([0.25, 0.16, 0.69, 0.64])
    grouped(ax, [c[0] for c in cfgs], series, 5.0, "Top-2048 keys lost vs full precision, % (lower is better)",
            legend="lower right")
    fig.savefig(os.path.join(OUT, "5_indexer_precision.png")); plt.close(fig)

    # 6. Post 2: centering the keys
    d = IQ["1.5b"]
    cfgs = [("FP8 + Hadamard", "fp8"), ("INT8 + Hadamard", "int8"), ("INT6 + Hadamard", "int6"), ("INT4 + Hadamard", "int4")]
    series = [("As DeepSeek-V3.2 quantizes", BASE, [lost(d, f, "hadamard") for _, f in cfgs]),
              ("Mean key subtracted first", S1, [lost(d, f, "hadamard", "k") for _, f in cfgs])]
    fig = frame("Subtract the mean key before quantizing: 3-6x fewer indexer errors",
                "Qwen2.5-1.5B, top-2048. Most of the gain is in layer 0, whose keys share a huge offset; elsewhere ~2x",
                foot="Mean key from the first 1,024 tokens, kept in fp32; q . mean is added back, so the score is "
                     "unchanged. Same proxy setup as the INT8 vs FP8 chart")
    ax = fig.add_axes([0.2, 0.16, 0.74, 0.64])
    grouped(ax, [c[0] for c in cfgs], series, 8.0, "Top-2048 keys lost vs full precision, % (lower is better)")
    fig.savefig(os.path.join(OUT, "6_indexer_centering.png")); plt.close(fig)

mech = os.path.join(HERE, "results", "indexer_mechanism.json")
if os.path.exists(mech):
    # 7. Post 3: why rotation helps INT8 and can hurt FP8
    syn = [r for r in json.load(open(mech))["synthetic"] if r["offset"] == "1 chan"]
    cs = [r["c"] for r in syn]
    fig = frame("Why Hadamard rotation helps INT8 but can hurt FP8",
                "Random keys plus one large channel. INT8's step is set by the biggest value; FP8's step follows each "
                "value", foot="Synthetic: keys ~ N(0, I) + offset in one of 128 channels, 6 random query heads, "
                "top-512 of 8,192 keys")
    ax = fig.add_axes([0.08, 0.16, 0.6, 0.64])
    for key, lab, c, mk in (("fp8_hadamard_none", "FP8 + Hadamard", S2, "o"), ("int8_none_none", "INT8", S4, "D"),
                            ("fp8_none_none", "FP8", S3, "^"), ("int8_hadamard_none", "INT8 + Hadamard", S1, "s")):
        ys = [r[key] for r in syn]
        ax.plot(cs, ys, color=c, marker=mk, markersize=9, linewidth=2, markeredgecolor=SURFACE, markeredgewidth=2, label=lab)
        ax.text(cs[-1] + 0.12, ys[-1], f"{lab}  {ys[-1]:.1f}%", va="center", fontsize=11.5, color=INK)
    ax.set_xlabel("Size of the shared offset (x the norm of the random part)")
    ax.set_ylabel("Top-512 keys lost, %"); ax.set_ylim(0, 10.5)
    ax.yaxis.grid(True, color=GRID, linewidth=1); ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, fontsize=11.5)
    fig.savefig(os.path.join(OUT, "7_indexer_mechanism.png")); plt.close(fig)

ds0 = os.path.join(HERE, "results", "ds_layer0.json")
if os.path.exists(ds0):
    # 8. Real DeepSeek-V3.2-Exp indexer, layer 0 (experiments/ds_layer0.py)
    R0 = [r for r in json.load(open(ds0))["rows"] if r["k"] == 2048 and r["center"] == "none"]
    l0 = lambda fmt, rot: 100 * (1 - sum(r["recall"] for r in R0 if (r["fmt"], r["rot"]) == (fmt, rot)) / 4)
    cfgs = [("FP8 + Hadamard, power-of-2 scales\n(DeepSeek-V3.2's setup)", "fp8_ue8m0", "hadamard", BASE),
            ("FP8 + Hadamard", "fp8", "hadamard", BASE), ("INT8", "int8", "none", BASE),
            ("INT8 + Hadamard", "int8", "hadamard", S1), ("BF16, 16-bit (reference)", "bf16", "none", BASE)]
    vals = [l0(f, r) for _, f, r, _ in cfgs]
    fig = frame("DeepSeek-V3.2's real indexer: INT8 loses 4x fewer tokens than FP8",
                "Layer 0 of DeepSeek-V3.2-Exp, computed from its released weights: share of the full-precision "
                "top-2048 tokens lost",
                foot="Layer 0 only. fp32 reference per DeepSeek's inference code, real head weights. "
                     "4 texts x 8,192 tokens, 256 queries each; INT8 wins in all 4")
    ax = fig.add_axes([0.3, 0.16, 0.64, 0.64])
    hbars(ax, [c[0] for c in cfgs], vals, [c[3] for c in cfgs], lambda v: f"{v:.2f}%",
          "Top-2048 tokens lost vs full precision, % (lower is better)")
    fig.savefig(os.path.join(OUT, "8_deepseek_layer0.png")); plt.close(fig)
print("wrote", sorted(os.listdir(OUT)))
