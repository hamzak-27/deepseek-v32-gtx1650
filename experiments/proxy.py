"""Shared pieces for the indexer-precision experiments.

- A Qwen2 forward pass written out from the safetensors weights. Layers are streamed to the GPU one at a time, so a
  1.5B model runs in fp32 on a 4 GB card. (transformers is not imported: its torchao dependency breaks on torch 2.6.)
- Number formats with one scale per vector, as V3.2's act_quant on the indexer (block 128 = one head vector).
- The indexer score: sum_h w_h * relu(q_h . k), uniform w.
"""
import glob, json, math, os
import torch
import torch.nn.functional as F
from safetensors import safe_open
from tokenizers import Tokenizer

torch.backends.cuda.matmul.allow_tf32 = False
dev = "cuda"
HUB = os.path.expanduser("~/.cache/huggingface/hub")


class Qwen2:
    def __init__(self, name):
        self.snap = glob.glob(os.path.join(HUB, f"models--Qwen--{name}/snapshots/*"))[0]
        c = json.load(open(os.path.join(self.snap, "config.json")))
        self.L, self.HQ, self.HK, self.DIM = (c["num_hidden_layers"], c["num_attention_heads"],
                                              c["num_key_value_heads"], c["hidden_size"])
        self.D = self.DIM // self.HQ; self.G = self.HQ // self.HK; self.eps = c["rms_norm_eps"]; self.theta = c["rope_theta"]
        self.files = sorted(glob.glob(os.path.join(self.snap, "*.safetensors")))
        self.where = {}
        for f in self.files:
            with safe_open(f, "pt") as h:
                for k in h.keys(): self.where[k] = f
        self.tok = Tokenizer.from_file(os.path.join(self.snap, "tokenizer.json"))
        self.embed = self.get("model.embed_tokens.weight").to(dev)  # bf16 on GPU, ~0.5 GB for 1.5B
        self.lm_head = self.get("lm_head.weight").to(dev) if "lm_head.weight" in self.where else self.embed

    def get(self, name):
        with safe_open(self.where[name], "pt") as h:
            return h.get_tensor(name)

    def layer(self, l):
        p = f"model.layers.{l}."
        return {k[len(p):]: self.get(k).to(dev).float() for k in self.where if k.startswith(p)}

    def rms(self, x, w):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * w

    @torch.no_grad()
    def forward(self, ids, qpos, prefix):
        """Per layer: post-RoPE q at qpos [HQ, P, D], mean q over positions < prefix [HQ, D], all k [HK, N, D].
        Also returns the mean next-token loss (a check that this forward pass is right)."""
        n, D = len(ids), self.D
        inv = 1.0 / (self.theta ** (torch.arange(0, D, 2, device=dev).float() / D))
        fr = torch.outer(torch.arange(n, device=dev).float(), inv); emb = torch.cat([fr, fr], -1)
        cos, sin = emb.cos(), emb.sin()
        rot = lambda t: t * cos + torch.cat([-t[..., D // 2:], t[..., :D // 2]], -1) * sin
        x = self.embed[ids].float(); caps = []
        for l in range(self.L):
            W = self.layer(l)
            h = self.rms(x, W["input_layernorm.weight"])
            lin = lambda m: F.linear(h, W[f"self_attn.{m}.weight"], W.get(f"self_attn.{m}.bias"))
            q = rot(lin("q_proj").view(n, self.HQ, D).transpose(0, 1))
            k = rot(lin("k_proj").view(n, self.HK, D).transpose(0, 1))
            v = lin("v_proj").view(n, self.HK, D).transpose(0, 1)
            caps.append((q[:, qpos].cpu(), q[:, :prefix].mean(1).cpu(), k.cpu()))
            a = F.scaled_dot_product_attention(q[None], k.repeat_interleave(self.G, 0)[None],
                                               v.repeat_interleave(self.G, 0)[None], is_causal=True)[0]
            del q, k, v
            x = x + F.linear(a.transpose(0, 1).reshape(n, self.DIM), W["self_attn.o_proj.weight"])
            h = self.rms(x, W["post_attention_layernorm.weight"])
            x = x + F.linear(F.silu(F.linear(h, W["mlp.gate_proj.weight"])) * F.linear(h, W["mlp.up_proj.weight"]),
                             W["mlp.down_proj.weight"])
            del W, h, a; torch.cuda.empty_cache()
        x = self.rms(x, self.get("model.norm.weight").to(dev).float()); loss = 0.0; E = self.lm_head.float()
        for i in range(0, n - 1, 256):  # logits chunked so they stay small
            lg = x[i:min(i + 256, n - 1)] @ E.T
            loss += F.cross_entropy(lg, ids[i + 1:i + 1 + len(lg)], reduction="sum").item()
        del E; torch.cuda.empty_cache()
        return caps, loss / (n - 1)


def texts():
    """Two long texts: one source file (code, one coherent document) and concatenated Yelp reviews (prose)."""
    import importlib.util
    import pyarrow.parquet as pq
    code = open(os.path.join(os.path.dirname(importlib.util.find_spec("transformers").origin), "modeling_utils.py"),
                encoding="utf-8").read()
    yelp = glob.glob(os.path.join(HUB, "datasets--Yelp--yelp_review_full/snapshots/*/yelp_review_full/test-*.parquet"))[0]
    prose = "\n\n".join(pq.read_table(yelp, columns=["text"]).column("text").to_pylist()[:6000])
    return {"code": code, "prose": prose}


def hadamard(d):
    h = torch.ones(1, 1)
    while h.shape[0] < d:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return (h / math.sqrt(d)).to(dev)


FORMATS = ["bf16", "fp8", "fp8_ue8m0", "int8", "int6", "int4"]


def quant(x, fmt):
    """Quantize then dequantize, one scale per vector (last dim), as V3.2's act_quant."""
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


def index_score(q, k):  # q [H, P, D], k [N, D] -> [P, N]
    return torch.relu(torch.einsum("hpd,nd->hpn", q, k)).mean(0)
