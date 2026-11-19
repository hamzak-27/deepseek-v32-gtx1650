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
                     "2 texts x 2 windows x 8,192 tokens, 256 queries each; INT8 wins in all 4")
    ax = fig.add_axes([0.3, 0.16, 0.64, 0.64])
    hbars(ax, [c[0] for c in cfgs], vals, [c[3] for c in cfgs], lambda v: f"{v:.2f}%",
          "Top-2048 tokens lost vs full precision, % (lower is better)")
    fig.savefig(os.path.join(OUT, "8_deepseek_layer0.png")); plt.close(fig)
dsl = os.path.join(HERE, "results", "ds_layers.json")
if os.path.exists(dsl):
    # 9. Real DeepSeek-V3.2-Exp layers 0-3 with the main attention (experiments/ds_layers.py)
    RL = json.load(open(dsl))["rows"]
    m = lambda l, key, f, r: 100 * sum(x[key] for x in RL if x["layer"] == l and (x["fmt"], x["rot"], x["center"]) == (f, r, "none")) / \
        len([x for x in RL if x["layer"] == l and (x["fmt"], x["rot"], x["center"]) == (f, r, "none")])
    series = [("FP8 + Hadamard, power-of-2 scales (DeepSeek-V3.2's setup)", BASE, [m(l, "err", "fp8_ue8m0", "hadamard") for l in range(4)]),
              ("INT8 + Hadamard", S1, [m(l, "err", "int8", "hadamard") for l in range(4)])]
    fig = frame("INT8 halves the indexer's attention error in DeepSeek-V3.2",
                "Change in the real sparse-attention output caused by quantizing the indexer, layers 0-3 of DeepSeek-V3.2-Exp",
                foot="fp32 forward through dense layers 0-2 from the released weights; 2 texts x 2 windows x 8,192 tokens, "
                     "256 queries each. INT8 lower in 15 of 16 layer-windows")
    ax = fig.add_axes([0.16, 0.16, 0.78, 0.62])
    grouped(ax, [f"Layer {l}" for l in range(4)], series, 1.25, "Attention output error vs full-precision indexer, % (lower is better)",
            legend="upper right")
    fig.savefig(os.path.join(OUT, "9_deepseek_layers.png")); plt.close(fig)
syn = os.path.join(HERE, "results", "fmt_tails_synth.json"); dst = os.path.join(HERE, "results", "ds_tails.json")
if os.path.exists(syn):
    # 10. Replication of van Baalen et al. 2023 (Qualcomm), Fig. 5 (experiments/fmt_tails.py)
    S = json.load(open(syn)); f5 = S["fig5"]; fm = ["INT8", "E2M5", "E3M4", "E4M3", "E5M2"]
    col = {"INT8": S1, "E4M3": S2}
    fig = frame("Replicating Qualcomm's INT8 vs FP8 result: it's all about the outliers",
                "Bits of accuracy, log2(sigma / RMSE), one scale per sample. Dotted: the best possible 255-level quantizer",
                foot="van Baalen et al. 2023 (arXiv 2303.17951) Fig. 5. Uniform, Gaussian within 0.2 bits of the paper; Student-t INT8 / E2M5 / E3M4\n"
                     "depend on the sample's largest value (100,000 samples here). FP8 = sign + E exponent + M mantissa bits; INT8 = [-127, 127]")
    ax = fig.add_axes([0.07, 0.19, 0.9, 0.6]); w = 0.16
    for gi, (dist, lab) in enumerate((("uniform", "Uniform"), ("normal", "Gaussian"), ("t3", "Student-t, 3 d.o.f. (heavy tails)"))):
        for j, f in enumerate(fm):
            v = f5[dist][f]["absmax"]; x = gi + (j - 2) * w
            ax.bar(x, v, w * 0.92, color=col.get(f, BASE), edgecolor=SURFACE, linewidth=2)
            ax.text(x, v + 0.08, f"{v:.1f}", ha="center", fontsize=10.5, color=INK)
            ax.text(x, -0.35, f, ha="center", fontsize=9.5, color=INK2)
        lm = f5[dist]["lloyd_max"]; ax.plot([gi - 2.6 * w, gi + 2.6 * w], [lm, lm], ":", color=INK2, linewidth=1.5)
        ax.text(gi, 9.0, lab, ha="center", fontsize=13, color=INK, weight="bold")
    ax.set_xticks([]); ax.set_ylim(0, 9.4); ax.set_ylabel("Bits of accuracy (higher is better)")
    ax.yaxis.grid(True, color=GRID, linewidth=1); ax.set_axisbelow(True); ax.spines["bottom"].set_visible(False)
    fig.savefig(os.path.join(OUT, "10_fp8_int8_replication.png")); plt.close(fig)
if os.path.exists(syn) and os.path.exists(dst):
    # 11. Real DeepSeek-V3.2-Exp vectors on the same axis, before and after the Hadamard rotation (experiments/ds_tails.py)
    sw = S["sweep_per_vector"]; D = [r for r in json.load(open(dst))["rows"] if r["tensor"] != "residual"]
    adv = lambda r: r["INT8_per_vector"] - r["E4M3_per_vector"]
    fig = frame("DeepSeek's indexer vectors favour INT8 even before the rotation",
                "INT8's lead over FP8 (E4M3) in bits, one scale per 128 values, as V3.2 quantizes.\n"
                "Hollow: the vector as computed. Filled: after the Hadamard rotation",
                foot="DeepSeek-V3.2-Exp layers 0-3 from the released weights, 2 texts x 2,048 tokens, 512 positions each. Curve: Student-t\n"
                     "vectors, 1 to infinite degrees of freedom. V3.2's power-of-2 FP8 scales give within 0.06 bits of plain E4M3")
    ax = fig.add_axes([0.08, 0.2, 0.66, 0.58]); ax.set_xscale("log")
    ax.axhspan(-0.6, 0, color="#f3f2ee", zorder=0); ax.axhline(0, color=INK2, linewidth=1)
    ax.text(2.0, -0.38, "FP8 wins", fontsize=12, color=INK2)
    from matplotlib.ticker import NullFormatter; ax.xaxis.set_minor_formatter(NullFormatter())
    ax.plot([r["kurtosis"] for r in sw], [r["INT8"] - r["E4M3"] for r in sw], "--", color=MUTED, linewidth=2, zorder=2)
    ax.text(11, 0.78, "Student-t (synthetic)", fontsize=11, color=INK2, rotation=-24)
    l2 = next(r for r in D if r["tensor"] == "mlp_down_in" and r["layer"] == 2)["none"]
    ax.annotate(f"MLP input, layer 2: FP8 ahead by {-adv(l2):.2f} bits.\nRotated: INT8 ahead by "
                f"{adv(next(r for r in D if r['tensor'] == 'mlp_down_in' and r['layer'] == 2)['hadamard']):.1f}",
                (l2["kurtosis"], adv(l2)), (9, -0.47), fontsize=10.5, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
    ax.text(5.5, 2.12, "Indexer q, k: INT8 ahead by 1.4-1.9 bits\nbefore rotating, 1.9-2.0 after", fontsize=10.5, color=INK)
    groups = (("idx_q", "Indexer queries", S1, "o"), ("idx_k", "Indexer keys", S1, "s"),
              ("mlp_down_in", "MLP down_proj input", S2, "D"), ("attn_in", "Attention input", BASE, "^"), ("q_lat", "Query latent", BASE, "v"))
    for t, lab, c, mk in groups:
        for i, r in enumerate(x for x in D if x["tensor"] == t):
            a, b = r["none"], r["hadamard"]
            ax.annotate("", (b["kurtosis"], adv(b)), (a["kurtosis"], adv(a)), arrowprops=dict(arrowstyle="-|>", color=c, lw=1, alpha=0.55, shrinkA=5, shrinkB=5))
            ax.plot(a["kurtosis"], adv(a), mk, ms=9, mfc=SURFACE, mec=c, mew=2, zorder=3, label=lab if i == 0 else None)
            ax.plot(b["kurtosis"], adv(b), mk, ms=9, mfc=c, mec=SURFACE, mew=1.5, zorder=4)
    ax.set_xlim(1.8, 80); ax.set_ylim(-0.6, 2.6); ax.set_xticks([2, 3, 5, 10, 20, 50]); ax.set_xticklabels(["2", "3\nGaussian", "5", "10", "20", "50"])
    ax.set_xlabel("Tail weight: kurtosis of each 128-value vector, log scale"); ax.set_ylabel("INT8 bits minus FP8 bits")
    ax.yaxis.grid(True, color=GRID, linewidth=1); ax.set_axisbelow(True)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=11.5, title="Hollow -> filled = rotated", title_fontsize=10.5)
    fig.savefig(os.path.join(OUT, "11_deepseek_tails.png")); plt.close(fig)
print("wrote", sorted(os.listdir(OUT)))
