"""INT8 vs FP8 by tail weight: van Baalen et al. 2023 ("FP8 versus INT8 for efficient deep learning inference", Qualcomm),
Figure 5, replicated, then the same question asked of real DeepSeek-V3.2-Exp vectors before and after the Hadamard rotation.

Formats (paper, Appendix A): 8-bit sign + magnitude, E exponent bits, M = 7 - E mantissa bits, subnormals, the code for
negative zero reserved, so 255 values; clip to the largest. INT8 = symmetric [-127, 127]. The scale is free (it absorbs the
exponent bias), so a format is its grid shape. Accuracy = log2(sigma / RMSE), "bits of accuracy", as the paper.

Part A (python experiments/fmt_tails.py synth): uniform, Gaussian, Student-t (nu = 3); one absmax scale for a large sample
(this reproduces the paper's bars), plus the MSE-optimal scale, plus Lloyd-Max (the best 255-level quantizer).
Part B (shared with ds_tails.py): one absmax scale per 128-vector, as V3.2's act_quant and our INT8 indexer, over a
sweep of Student-t nu: where does FP8-E4M3 start to beat INT8?
"""
import json, math, os, sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(os.path.dirname(HERE), "results")


def fp_grid(e):
    """Non-negative values of the paper's 8-bit ExM(7-e) format, scaled so the largest is 1."""
    m = 7 - e
    if e == 0: v = torch.arange(128, dtype=torch.float64)  # INT8 (sign-magnitude = symmetric int8, 0..127)
    else:
        B = 2 ** (e - 1); vals = []
        for p in range(2 ** e):
            for d in range(2 ** m):
                vals.append(2.0 ** (1 - B) * d / 2 ** m if p == 0 else 2.0 ** (p - B) * (1 + d / 2 ** m))
        v = torch.tensor(sorted(set(vals)), dtype=torch.float64)
    return v / v.max()


FMTS = {"INT8": 0, "E2M5": 2, "E3M4": 3, "E4M3": 4, "E5M2": 5}
GRIDS = {}


def qgrid(x, grid):
    """Round |x| (already divided by its scale, so the grid max is 1) to the nearest grid value, keep the sign."""
    g = grid.to(x.device, x.dtype); a = x.abs().clamp(max=1.0)
    i = torch.bucketize(a, g).clamp(1, len(g) - 1); lo, hi = g[i - 1], g[i]
    return torch.where(a - lo <= hi - a, lo, hi) * x.sign()


def quant(x, fmt, amax):
    """Quantize-dequantize x with scale amax (broadcast), the format's largest value mapping to amax."""
    if fmt not in GRIDS: GRIDS[fmt] = fp_grid(FMTS[fmt])
    if fmt == "INT8":  # exact int8 rounding, as the kernels (round half to even differs only on exact ties)
        s = amax / 127.0; return torch.round(x / s).clamp(-127, 127) * s
    return qgrid(x / amax, GRIDS[fmt]) * amax


def bits(x, y, dim=None):
    """log2(sigma / RMSE). With dim, each vector is first scaled to unit RMS, so every vector counts once."""
    if dim is None: return 0.5 * math.log2(x.pow(2).mean().item() / (x - y).pow(2).mean().item())
    r = (x - y).pow(2).mean(dim) / x.pow(2).mean(dim)
    return -0.5 * math.log2(r.mean().item())


PDF = {"uniform": (lambda t: (t.abs() <= 1).double() / 2, 1 / 3, 1.0),
       "normal": (lambda t: torch.exp(-t * t / 2) / math.sqrt(2 * math.pi), 1.0, 12.0),
       "t3": (lambda t: 6 * math.sqrt(3) / (math.pi * (3 + t * t) ** 2), 3.0, 1e4)}


def lloyd_max(dist, levels=255, iters=3000, n=400_001):
    """The paper's dotted line: the best 255-level quantizer of the distribution itself (not of a sample), by Lloyd's
    iteration on a fine grid weighted by the density, started from the high-resolution optimum (levels ~ p^(1/3))."""
    p, var, R = PDF[dist]
    u = torch.linspace(-1, 1, n, dtype=torch.float64)
    t = torch.sinh(u * math.asinh(R)) if dist == "t3" else R * u  # t3: dense near 0, reaches +-1e4
    dt = torch.gradient(t)[0]; w = p(t) * dt; w /= w.sum()
    cdf = (p(t).pow(1 / 3) * dt).cumsum(0); cdf /= cdf[-1].clone()
    c = t[torch.bucketize((torch.arange(levels, dtype=torch.float64) + 0.5) / levels, cdf).clamp(max=n - 1)]
    for _ in range(iters):
        idx = torch.bucketize(t, (c[1:] + c[:-1]) / 2)
        sw = torch.zeros(levels, dtype=t.dtype).index_add_(0, idx, w); sx = torch.zeros(levels, dtype=t.dtype).index_add_(0, idx, w * t)
        c = torch.where(sw > 0, sx / sw.clamp_min(1e-300), c)
    idx = torch.bucketize(t, (c[1:] + c[:-1]) / 2); mse = (w * (t - c[idx]) ** 2).sum().item()
    return 0.5 * math.log2(var / mse)


def sample(dist, n, g, nu=3.0):
    if dist == "uniform": return torch.rand(n, generator=g, dtype=torch.float64) * 2 - 1
    if dist == "normal": return torch.randn(n, generator=g, dtype=torch.float64)
    z = torch.randn(n, generator=g, dtype=torch.float64)  # Student-t: z / sqrt(chi2_nu / nu)
    chi = torch.distributions.Chi2(torch.tensor(float(nu), dtype=torch.float64)).sample((n,))
    return z / (chi / nu).sqrt()


def synth(n=100_000, seed=0):
    """Figure 5. Paper's sample size is not given; absmax over 1e5 draws reproduces its bars (see README)."""
    g = torch.Generator().manual_seed(seed); torch.manual_seed(seed); res = {}
    for dist in ("uniform", "normal", "t3"):
        x = sample(dist, n, g); amax = x.abs().max(); r = {}
        for f in FMTS:
            r[f] = dict(absmax=bits(x, quant(x, f, amax)),
                        mse_opt=max(bits(x, quant(x, f, amax * c)) for c in torch.logspace(-2.5, 0, 400, dtype=torch.float64)))
        r["lloyd_max"] = lloyd_max(dist); res[dist] = r
        print(f"{dist:8s} " + "  ".join(f"{f} {r[f]['absmax']:.2f} ({r[f]['mse_opt']:.2f})" for f in FMTS) + f"  Lloyd-Max {r['lloyd_max']:.2f}", flush=True)
    return res


def kurtosis(x, dim=-1):
    """Pooled kurtosis of the values after each vector is scaled to unit RMS (Gaussian = 3)."""
    z = x / x.pow(2).mean(dim, keepdim=True).sqrt().clamp_min(1e-30); return z.pow(4).mean().item()


def per_vector(x, fmts=("INT8", "E4M3")):
    """x [n, 128]: one absmax scale per vector. Returns bits per format, kurtosis, crest (amax / rms)."""
    amax = x.abs().amax(-1, keepdim=True).clamp_min(1e-30)
    out = {f: bits(x, quant(x, f, amax), dim=-1) for f in fmts}
    out["kurtosis"] = kurtosis(x); out["crest"] = (amax[:, 0] / x.pow(2).mean(-1).sqrt()).mean().item()
    return out


def sweep(nvec=20_000, seed=0):
    """Student-t vectors of 128 values, one absmax scale per vector, nu from heavy to Gaussian."""
    g = torch.Generator().manual_seed(seed); torch.manual_seed(seed); rows = []
    for nu in (1.0, 1.1, 1.2, 1.3, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0, 20.0, 50.0, float("inf")):
        x = (sample("normal", nvec * 128, g) if nu == float("inf") else sample("t", nvec * 128, g, nu)).view(nvec, 128)
        r = per_vector(x, tuple(FMTS)); r["nu"] = nu; rows.append(r)
        print(f"t nu={nu:5}: kurt {r['kurtosis']:6.2f} crest {r['crest']:5.2f}  " + " ".join(f"{f} {r[f]:.2f}" for f in FMTS), flush=True)
    return rows


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); path = os.path.join(OUT, "fmt_tails_synth.json")
    print("== Figure 5: one absmax scale per sample (in brackets: MSE-optimal scale) =="); fig5 = synth()
    print("\n== per-128-vector absmax scale, Student-t sweep =="); sw = sweep()
    json.dump(dict(paper="arXiv 2303.17951 Fig. 5", fig5=fig5, sweep_per_vector=sw), open(path, "w"), indent=1)
    print("wrote", path)
