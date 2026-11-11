"""Real DeepSeek-V3.2-Exp indexer at layers 0-3, with the main attention: what does each indexer format cost the model?

Layers 0-2 of V3.2 are dense (first_k_dense_replace = 3), so the residual stream into layers 1, 2 and 3 can be computed
exactly from about 1.9 GB of fp8 weights, without any of the 256 experts. Per layer (as deepseek-ai/DeepSeek-V3.2-Exp
inference/model.py, in fp32, with its fp8 KV-cache simulation):
    x = RMSNorm(h);  qr = RMSNorm(q_a(x));  q = q_b(qr) -> 128 heads x (128 nope + 64 rope, interleaved YaRN RoPE)
    kv, k_pe = kv_a(x);  kv = fp8_ue8m0(RMSNorm(kv));  k_nope, v = kv_b(kv);  scale = 192^-0.5 * mscale^2
    attention over the indexer's top-2048 keys (fp32 indexer here, as the reference), then o_proj; residual
    h += mlp(RMSNorm(h)), SwiGLU 18432
The attention runs in MLA's absorbed form (scores against the 512-d latent), which is the same arithmetic, to fit 4 GB.

For the 256 metric queries of each window and each indexer format (as ds_layer0.py) it reports:
  lost        share of the fp32 indexer's top-2048 missing from the format's top-2048 (ds_layer0's metric)
  mass        share of the dense (full causal) main-attention weight that the selected 2048 keys hold, mean over heads
  out_err     ||o_fmt - o_fp32|| / ||o_fp32|| of the real (sparse) attention output, all heads, before o_proj
Baselines: oracle (the 2048 keys with the most attention weight, per head; an upper bound for mass) and window (the
last 2048 tokens). If the forward pass were wrong at some layer, the fp32 indexer's mass would fall toward the window's.

Each window starts with BOS (id 0), as real inputs do, so layer-0 numbers differ slightly from ds_layer0.py.
Run: python experiments/ds_layers.py [windows]   (default: all 4 = 2 texts x 2 windows)  -> results/ds_layers.json
"""
import json, math, os, sys, time
import torch, torch.nn.functional as Fn
from tokenizers import Tokenizer
import ds_fetch as F
from proxy import texts, hadamard, quant, FORMATS, dev
from ds_layer0 import embed_rows, freqs_cis, rope, rms

N, P, K, SEED = 8192, 256, 2048, 0
HI, DI, R = 64, 128, 64                       # indexer heads, head dim, rope dim
HM, DN, DV, LAT = 128, 128, 128, 512          # main heads, nope dim, value dim, kv latent
SCALE = (DN + R) ** -0.5 * (0.1 * math.log(40) + 1.0) ** 2
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(os.path.dirname(HERE), "results")
HAD = hadamard(DI); LAYERS = 4


def rope_i(x, fc):  # interleaved pairs (2i, 2i + 1), as apply_rotary_emb(interleaved=True); x [..., n, 64]
    a, b = x[..., 0::2], x[..., 1::2]; c, s = fc.real.to(x.device), fc.imag.to(x.device)
    return torch.stack([a * c - b * s, a * s + b * c], -1).flatten(-2)


class Layer:
    """One layer's weights: fp8 matrices stay fp8 on the GPU and are dequantized in row blocks when used."""
    ATT = ["q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj"]; IDX = ["wq_b", "wk"]

    def __init__(s, i, full):
        p = f"model.layers.{i}."; s.w8 = {}; s.v = {}
        names = [p + "self_attn." + n for n in s.ATT] + [p + "self_attn.indexer." + n for n in s.IDX]
        if full: names += [p + "self_attn.o_proj"] + [p + "mlp." + n for n in ("gate_proj", "up_proj", "down_proj")]
        for n in names:
            w, sc = F.load8(n + ".weight"); s.w8[n.split(".")[-1]] = (w.to(dev), sc.to(dev))
        for n in ("input_layernorm", "post_attention_layernorm", "self_attn.q_a_layernorm", "self_attn.kv_a_layernorm",
                  "self_attn.indexer.k_norm", "self_attn.indexer.weights_proj"):
            if n == "post_attention_layernorm" and not full: continue
            s.v[n.split(".")[-1]] = F.load(p + n + ".weight").to(dev)
        s.v["k_norm_b"] = F.load(p + "self_attn.indexer.k_norm.bias").to(dev)
        s.wkv_b = s.deq("kv_b_proj").view(HM, DN + DV, LAT)   # [128, 256, 512], small enough to keep in fp32

    def deq(s, n, a=0, b=None):  # rows a:b; a is a multiple of 128, b may not be (kv_a has 576 rows)
        w, sc = s.w8[n]; b = b or w.shape[0]
        return w[a:b].float() * sc[a // 128:-(-b // 128)].repeat_interleave(128, 0)[:b - a].repeat_interleave(128, 1)[:, :w.shape[1]]

    def lin(s, x, n, blk=2048):
        w = s.w8[n][0]; y = torch.empty(x.shape[0], w.shape[0], device=dev)
        for a in range(0, w.shape[0], blk): y[:, a:a + blk] = x @ s.deq(n, a, min(a + blk, w.shape[0])).T
        return y


@torch.no_grad()
def project(L, h, fc):
    """Everything attention needs, for all positions."""
    x = rms(h, L.v["input_layernorm"]); qr = rms(L.lin(x, "q_a_proj"), L.v["q_a_layernorm"])
    kv = L.lin(x, "kv_a_proj_with_mqa"); lat, kpe = kv[:, :LAT], kv[:, LAT:]
    lat = quant(rms(lat, L.v["kv_a_layernorm"]).view(-1, LAT // 128, 128), "fp8_ue8m0").view(-1, LAT)  # fp8 KV cache
    kpe = rope_i(kpe, fc)
    ki = Fn.layer_norm(L.lin(x, "wk"), (DI,), L.v["k_norm"], L.v["k_norm_b"], 1e-6); ki = torch.cat([rope(ki[:, :R], fc), ki[:, R:]], -1)
    wi = (x @ L.v["weights_proj"].T) * HI ** -0.5 * DI ** -0.5
    return dict(x=x, qr=qr, lat=lat, kpe=kpe, ki=ki, wi=wi)


def iq(L, A, pos, fc):  # indexer queries [HI, p, DI]
    q = L.lin(A["qr"][pos], "wq_b").view(len(pos), HI, DI).transpose(0, 1)
    return torch.cat([rope(q[..., :R], fc[pos]), q[..., R:]], -1)


def iscore(q, k, w): return torch.einsum("ph,hpn->pn", w, torch.relu(torch.einsum("hpd,nd->hpn", q, k)))


def mq(L, A, pos, fc):  # main queries in latent form: [p, HM, LAT] and rope part [p, HM, R]
    q = L.lin(A["qr"][pos], "q_b_proj").view(len(pos), HM, DN + R)
    qlat = torch.einsum("phd,hdc->phc", q[..., :DN], L.wkv_b[:, :DN]); return qlat, rope_i(q[..., DN:], fc[pos][:, None])


def logits(A, qlat, qpe, pos):  # [p, HM, N] main attention logits, causal
    s = (torch.einsum("phc,nc->phn", qlat, A["lat"]) + torch.einsum("phr,nr->phn", qpe, A["kpe"])) * SCALE
    return s.masked_fill(torch.arange(N, device=dev)[None, None] > torch.as_tensor(pos, device=dev)[:, None, None], float("-inf"))


def attend(L, A, lg, sel):  # sparse attention over sel [p, K] (same keys for every head, as V3.2); out [p, HM, DV]
    m = torch.full_like(lg[:, 0], float("-inf")).scatter_(-1, sel, 0.0)
    pr = (lg + m[:, None]).softmax(-1)
    return torch.einsum("phc,hdc->phd", torch.einsum("phn,nc->phc", pr, A["lat"]), L.wkv_b[:, DN:])


@torch.no_grad()
def forward(L, A, h, fc, blk=64):
    """Layer output for all positions, with the fp32 indexer's top-2048 (the reference path)."""
    o = torch.empty(N, HM * DV, device=dev)
    for a in range(0, N, blk):
        pos = list(range(a, a + blk)); lg = logits(A, *mq(L, A, pos, fc), pos)
        sc = iscore(iq(L, A, pos, fc), A["ki"], A["wi"][pos]).masked_fill(torch.arange(N, device=dev)[None] > torch.tensor(pos, device=dev)[:, None], float("-inf"))
        o[a:a + blk] = attend(L, A, lg, sc.topk(min(K, a + blk), -1).indices).flatten(1)
    h = h + L.lin(o, "o_proj"); del o
    for a in range(0, N, 1024):  # MLP in token blocks
        x = rms(h[a:a + 1024], L.v["post_attention_layernorm"])
        h[a:a + 1024] += L.lin(Fn.silu(L.lin(x, "gate_proj")) * L.lin(x, "up_proj"), "down_proj")
    return h


@torch.no_grad()
def evaluate(L, A, qpos, fc):
    """Formats on the metric queries: lost, mass, out_err."""
    q, k, w = iq(L, A, qpos, fc), A["ki"], A["wi"][qpos]; qd = torch.as_tensor(qpos, device=dev)
    fut = torch.arange(N, device=dev)[None] > qd[:, None]
    ref = iscore(q, k, w).masked_fill(fut, float("-inf")); tau = ref.topk(K, -1).values[:, -1:]
    sel = {("fp32", "none", "none"): ref.topk(K, -1).indices, ("window", "none", "none"): (qd[:, None] - torch.arange(K, device=dev)[None]).clamp_min(0)}
    kmean = k[1:1024].mean(0)
    for center in ("none", "k"):
        kin = k - kmean if center == "k" else k
        for rot in ("none", "hadamard"):
            if center == "k" and rot == "none": continue
            Rm = HAD if rot == "hadamard" else None
            for fmt in FORMATS:
                if center == "k" and fmt != "int8": continue
                a, b = quant(q @ Rm if Rm is not None else q, fmt), quant(kin @ Rm if Rm is not None else kin, fmt)
                if Rm is not None: a, b = a @ Rm.T, b @ Rm.T
                if center == "k": b = b + kmean
                sel[(fmt, rot, center)] = iscore(a, b, w).masked_fill(fut, float("-inf")).topk(K, -1).indices
    acc = {key: dict(lost=0.0, mass=0.0, err=0.0) for key in sel}; acc[("oracle", "none", "none")] = dict(lost=float("nan"), mass=0.0, err=float("nan"))
    for a in range(0, len(qpos), 32):  # main attention in query blocks
        pos = qpos[a:a + 32]; lg = logits(A, *mq(L, A, pos, fc), pos); dense = lg.softmax(-1)
        o_ref = attend(L, A, lg, sel[("fp32", "none", "none")][a:a + 32]).flatten(1)
        acc[("oracle", "none", "none")]["mass"] += dense.topk(K, -1).values.sum(-1).mean(-1).sum().item()
        for key, s in sel.items():
            s = s[a:a + 32]; acc[key]["mass"] += dense.gather(-1, s[:, None].expand(-1, HM, -1)).sum(-1).mean(-1).sum().item()
            acc[key]["lost"] += (1 - (ref[a:a + 32].gather(1, s) >= tau[a:a + 32]).float().mean(-1)).sum().item()
            o = attend(L, A, lg, s).flatten(1); acc[key]["err"] += ((o - o_ref).norm(dim=-1) / o_ref.norm(dim=-1)).sum().item()
        del lg, dense
    stats = dict(k_offset=(k.mean(0).norm() / (k - k.mean(0)).pow(2).sum(-1).mean().sqrt()).item(),
                 k_peak=(k.abs().amax(-1) / k.pow(2).mean(-1).sqrt()).mean().item(),
                 k_peak_had=((k @ HAD).abs().amax(-1) / k.pow(2).mean(-1).sqrt()).mean().item())
    return [dict(fmt=f, rot=r, center=c, **{m: v / len(qpos) for m, v in d.items()}) for (f, r, c), d in acc.items()], stats


if __name__ == "__main__":
    nwin = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    tok = Tokenizer.from_file(os.path.join(F.HERE, "tokenizer.json")); g = torch.Generator().manual_seed(SEED)
    qpos = sorted((torch.randperm(N // 2, generator=g)[:P] + N // 2).tolist()); fc = freqs_cis(N)
    seqs = {}
    for name, text in texts().items():
        ids = tok.encode(text).ids
        for wdw in range(2): seqs[f"{name}{wdw}"] = [0] + ids[wdw * (N - 1):(wdw + 1) * (N - 1)]
    seqs = dict(list(seqs.items())[:nwin]); E = embed_rows(sorted({i for s in seqs.values() for i in s}))
    hs = {n: torch.stack([E[i] for i in s]).float() for n, s in seqs.items()}; del E  # residual streams, kept on the CPU
    rows, stats = [], {}
    for li in range(LAYERS):
        t0 = time.time(); L = Layer(li, full=li < LAYERS - 1); print(f"layer {li}: weights ready {time.time() - t0:.0f}s, GPU {torch.cuda.memory_allocated() / 2**30:.2f} GB", flush=True)
        for n in seqs:
            t1 = time.time(); h = hs[n].to(dev); A = project(L, h, fc)
            r, st = evaluate(L, A, qpos, fc); rows += [dict(layer=li, seq=n, **x) for x in r]; stats[f"{li}/{n}"] = st
            st["h_rms_in"] = h[1:].pow(2).mean().sqrt().item()  # residual stream size into this layer (BOS excluded)
            if li < LAYERS - 1: hs[n] = forward(L, A, h, fc).cpu()
            f32 = next(x for x in r if x["fmt"] == "fp32"); win = next(x for x in r if x["fmt"] == "window"); orc = next(x for x in r if x["fmt"] == "oracle")
            print(f"  {n}: {time.time() - t1:.0f}s  mass fp32 {f32['mass']:.3f} window {win['mass']:.3f} oracle {orc['mass']:.3f}  k_offset {st['k_offset']:.2f}", flush=True)
            del A, h; torch.cuda.empty_cache()
        del L; torch.cuda.empty_cache()
        json.dump(dict(model="DeepSeek-V3.2-Exp indexer, layers 0-3, with main attention", N=N, queries=P, k=K, rows=rows, stats=stats),
                  open(os.path.join(OUT, "ds_layers.json"), "w"))
    print(f"\n== DeepSeek-V3.2-Exp, top-{K} of {N}, mean over {len(seqs)} windows ==")
    keys = [("fp32", "none", "none"), ("oracle", "none", "none"), ("window", "none", "none")] + \
           [(f, r, "none") for f in FORMATS for r in ("none", "hadamard")] + [("int8", "hadamard", "k")]
    for li in range(LAYERS):
        print(f"\nlayer {li}\n{'format':10s} {'rotation':9s} {'center':6s} {'lost %':>7s} {'mass %':>7s} {'out err %':>9s}")
        for f, r, c in keys:
            s = [x for x in rows if x["layer"] == li and (x["fmt"], x["rot"], x["center"]) == (f, r, c)]
            if s: print(f"{f:10s} {r:9s} {c:6s} " + " ".join(f"{100 * sum(x[m] for x in s) / len(s):{wd}.3f}" for m, wd in (("lost", 7), ("mass", 7), ("err", 9))))
