"""Does INT8 instead of FP8 change which tokens DeepSeek-V3.2's lightning indexer selects?
And does the Hadamard rotation that V3.2 applies before quantizing matter?

V3.2 indexer (inference/model.py, Indexer.forward):
    q, k  -> Hadamard rotation -> act_quant (one scale per 128-dim vector, FP8 e4m3)
    score[m, n] = sum_h w[m, h] * relu(q[m, h] . k[n]);  keep top-2048 keys per query

We cannot run V3.2 (671B) or load its indexer, so the proxy indexer uses REAL post-RoPE query/key vectors from
Qwen2.5-0.5B (24 layers, 14 query heads sharing 2 key heads, 64 dims), with the 7 query heads of each key group as
the indexer heads and uniform head weights. The question is purely numerical: given realistic activations (outlier
channels, attention sinks, RoPE), how far does each number format move the top-k set away from full precision?

Metrics per (sequence, layer, key group, format, rotation, k), averaged over 256 query positions in [N/2, N):
  recall   = fraction of the selected keys whose full-precision score is >= the k-th best full-precision score
             (tie-aware: relu makes many scores exactly 0)
  attn     = share of the model's REAL softmax attention mass (same heads, full precision) that the selected keys
             hold, i.e. how much of what attention actually uses survives the selection
Baselines for attn: full-precision selection (the ceiling) and a "last k tokens" sliding window.

Run: python experiments/indexer_quant.py   (writes results/indexer_quant.json)
"""
import glob, json, math, os, time
import torch
import torch.nn.functional as F
from safetensors import safe_open
from tokenizers import Tokenizer

torch.backends.cuda.matmul.allow_tf32 = False
dev = "cuda"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HUB = os.path.expanduser("~/.cache/huggingface/hub")
SNAP = glob.glob(os.path.join(HUB, "models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/*"))[0]
N, P, KS, SEED = 8192, 256, (512, 2048), 0

# ------------------------------------------------------------------ Qwen2 forward, written out from the weights
cfg = json.load(open(os.path.join(SNAP, "config.json")))
L, HQ, HK, DIM = cfg["num_hidden_layers"], cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["hidden_size"]
D = DIM // HQ; G = HQ // HK; EPS = cfg["rms_norm_eps"]
W = {}
with safe_open(os.path.join(SNAP, "model.safetensors"), "pt", device=dev) as f:
    for name in f.keys():
        W[name] = f.get_tensor(name).float()


def rms(x, w):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w


def rope(n):
    inv = 1.0 / (cfg["rope_theta"] ** (torch.arange(0, D, 2, device=dev).float() / D))
    fr = torch.outer(torch.arange(n, device=dev).float(), inv)
    emb = torch.cat([fr, fr], -1)
    return emb.cos(), emb.sin()


def rot_half(x):
    return torch.cat([-x[..., D // 2:], x[..., :D // 2]], -1)


@torch.no_grad()
def forward(ids, qpos):
    """Returns per-layer post-RoPE q at qpos [HQ, P, D], all k [HK, N, D], and the mean next-token loss."""
    x = W["model.embed_tokens.weight"][ids]; n = len(ids); cos, sin = rope(n); caps = []
    for l in range(L):
        p = f"model.layers.{l}."
        h = rms(x, W[p + "input_layernorm.weight"])
        lin = lambda t, m: F.linear(t, W[p + f"self_attn.{m}.weight"], W.get(p + f"self_attn.{m}.bias"))
        q = lin(h, "q_proj").view(n, HQ, D).transpose(0, 1)
        k = lin(h, "k_proj").view(n, HK, D).transpose(0, 1)
        v = lin(h, "v_proj").view(n, HK, D).transpose(0, 1)
        q = q * cos + rot_half(q) * sin; k = k * cos + rot_half(k) * sin
        caps.append((q[:, qpos].cpu(), k.cpu()))
        a = F.scaled_dot_product_attention(q[None], k.repeat_interleave(G, 0)[None], v.repeat_interleave(G, 0)[None],
                                           is_causal=True)[0]
        x = x + F.linear(a.transpose(0, 1).reshape(n, DIM), W[p + "self_attn.o_proj.weight"])
        h = rms(x, W[p + "post_attention_layernorm.weight"])
        x = x + F.linear(F.silu(F.linear(h, W[p + "mlp.gate_proj.weight"])) * F.linear(h, W[p + "mlp.up_proj.weight"]),
                         W[p + "mlp.down_proj.weight"])
    x = rms(x, W["model.norm.weight"]); loss = 0.0
    for i in range(0, n - 1, 512):  # tied embeddings; chunked so logits never exceed ~300 MB
        lg = x[i:min(i + 512, n - 1)] @ W["model.embed_tokens.weight"].T
        loss += F.cross_entropy(lg, ids[i + 1:i + 1 + len(lg)], reduction="sum").item()
    return caps, loss / (n - 1)


# ------------------------------------------------------------------ number formats, one scale per vector (as V3.2)
def hadamard(d):
    h = torch.ones(1, 1)
    while h.shape[0] < d:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return (h / math.sqrt(d)).to(dev)


HAD = hadamard(D)


def quant(x, fmt):
    if fmt == "fp32": return x
    if fmt == "bf16": return x.bfloat16().float()
    amax = x.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    if fmt.startswith("fp8"):
        s = amax / 448.0
        if fmt == "fp8_ue8m0": s = torch.exp2(torch.ceil(torch.log2(s)))
        return (x / s).clamp(-448, 448).to(torch.float8_e4m3fn).float() * s
    qmax = {"int8": 127, "int6": 31, "int4": 7}[fmt]
    s = amax * (1.0 / qmax)
    return torch.round(x / s).clamp(-qmax, qmax) * s


FORMATS = ["bf16", "fp8", "fp8_ue8m0", "int8", "int6", "int4"]


def index_score(q, k):  # q [G, P, D], k [N, D] -> [P, N], uniform head weights
    return torch.relu(torch.einsum("hpd,nd->hpn", q, k)).mean(0)


@torch.no_grad()
def evaluate(caps, qpos):
    qpos_d = torch.tensor(qpos, device=dev); keys = torch.arange(N, device=dev)
    future = keys[None, :] > qpos_d[:, None]; out = []
    for l, (qa, ka) in enumerate(caps):
        for g in range(HK):
            q = qa[g * G:(g + 1) * G].to(dev); k = ka[g].to(dev)
            ref = index_score(q, k).masked_fill(future, float("-inf"))
            attn = torch.softmax((torch.einsum("hpd,nd->hpn", q, k) / math.sqrt(D)).masked_fill(future, float("-inf")),
                                 -1).mean(0)
            for kk in KS:
                tau = ref.topk(kk, -1).values[:, -1:]
                def stats(sel):
                    return dict(recall=(ref.gather(1, sel) >= tau).float().mean().item(),
                                attn=attn.gather(1, sel).sum(-1).mean().item())
                win = (qpos_d[:, None] - torch.arange(kk, device=dev)[None]).clamp_min(0)
                base = dict(layer=l, group=g, k=kk)
                out.append(dict(base, fmt="fp32", rot="none", **stats(ref.topk(kk, -1).indices)))
                out.append(dict(base, fmt="window", rot="none", **stats(win)))
                for rot in ("none", "hadamard"):
                    qr, kr = (q @ HAD, k @ HAD) if rot == "hadamard" else (q, k)
                    for fmt in FORMATS:
                        s = index_score(quant(qr, fmt), quant(kr, fmt)).masked_fill(future, float("-inf"))
                        out.append(dict(base, fmt=fmt, rot=rot, **stats(s.topk(kk, -1).indices)))
    return out


# ------------------------------------------------------------------ data: one long code file, concatenated prose
tok = Tokenizer.from_file(os.path.join(SNAP, "tokenizer.json"))
import importlib.util  # noqa: E402  find transformers' source on disk without importing it (its import is broken here)
code = open(os.path.join(os.path.dirname(importlib.util.find_spec("transformers").origin), "modeling_utils.py"),
            encoding="utf-8").read()
import pyarrow.parquet as pq  # noqa: E402
yelp = glob.glob(os.path.join(HUB, "datasets--Yelp--yelp_review_full/snapshots/*/yelp_review_full/test-*.parquet"))[0]
prose = "\n\n".join(pq.read_table(yelp, columns=["text"]).column("text").to_pylist()[:6000])
seqs = {}
for name, text in (("code", code), ("prose", prose)):
    ids = tok.encode(text).ids
    for w in range(2):
        seqs[f"{name}{w}"] = torch.tensor(ids[w * N:(w + 1) * N], device=dev)
        assert len(seqs[f"{name}{w}"]) == N, (name, len(ids))

g = torch.Generator().manual_seed(SEED)
qpos = sorted((torch.randperm(N // 2, generator=g)[:P] + N // 2).tolist())
rows, meta = [], {}
for name, ids in seqs.items():
    t0 = time.time(); caps, loss = forward(ids, qpos)
    for r in evaluate(caps, qpos):
        rows.append(dict(seq=name, **r))
    meta[name] = dict(loss=round(loss, 4)); print(f"{name}: next-token loss {loss:.3f}, {time.time() - t0:.0f}s", flush=True)
    del caps; torch.cuda.empty_cache()

os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
json.dump(dict(model="Qwen2.5-0.5B-Instruct (proxy)", N=N, queries=P, ks=KS, seqs=meta, rows=rows),
          open(os.path.join(HERE, "results", "indexer_quant.json"), "w"))

# ------------------------------------------------------------------ summary
def mean(xs): return sum(xs) / len(xs)
for kk in KS:
    print(f"\n== top-{kk} of {N} keys (queries at positions {N // 2}-{N}) ==")
    print(f"{'format':12s} {'rotation':9s} {'recall':>8s} {'worst layer':>12s} {'attn mass':>10s}")
    for fmt, rot in [("fp32", "none"), ("window", "none")] + [(f, r) for f in FORMATS for r in ("none", "hadamard")]:
        sel = [r for r in rows if r["k"] == kk and r["fmt"] == fmt and r["rot"] == rot]
        per_layer = [mean([r["recall"] for r in sel if r["layer"] == l]) for l in range(L)]
        print(f"{fmt:12s} {rot:9s} {mean([r['recall'] for r in sel]):8.4f} {min(per_layer):12.4f} "
              f"{mean([r['attn'] for r in sel]):10.4f}")
