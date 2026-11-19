"""How heavy-tailed are DeepSeek-V3.2-Exp's real vectors, before and after the Hadamard rotation, and which 8-bit
format wins on them? The follow-up to fmt_tails.py (van Baalen et al. 2023, Fig. 5): their rule is "it's all about the
outliers", INT8 for well-behaved data and FP8-E4M3 for heavy tails.

Real activations of layers 0-3 (dense layers: exact forward from the fp8 weights, as ds_layers.py; 2 texts x one window
of 2048 tokens, 512 sampled positions per window), cut into 128-value vectors as V3.2 quantizes them:
  idx_q, idx_k     the lightning indexer's queries and keys after RoPE, the vectors V3.2 rotates before FP8
  attn_in, q_lat   inputs of the attention projections (RMSNorm(h), and RMSNorm(q_a(x)) into q_b / the indexer's wq_b)
  mlp_down_in      silu(gate) * up, the input of down_proj (layers 0-2)
  residual         the residual stream h itself (not quantized in V3.2; the classic heavy-tailed tensor, for reference)
Each with and without a 128-point Hadamard rotation per vector; quantized INT8 and E4M3 with
  per_vector   one absmax scale per 128 values (V3.2's act_quant, our INT8 indexer)
  per_tensor   one absmax scale for the whole tensor (the paper's setting)
plus V3.2's own FP8 path (float8_e4m3fn, power-of-two ue8m0 scale per vector). Accuracy = log2(rms / rms error), each
vector scaled to unit rms first. Tail weight = pooled kurtosis of the unit-rms vectors (Gaussian 3).
Run: python experiments/ds_tails.py -> results/ds_tails.json
"""
import json, os, time
import torch, torch.nn.functional as Fn
from tokenizers import Tokenizer
import ds_fetch as F
import ds_layers as D
from proxy import texts, hadamard, dev
from ds_layer0 import embed_rows, freqs_cis, rms
from fmt_tails import quant as gquant, kurtosis

N, P, SEED = 2048, 512, 0
D.N = N                                       # ds_layers reads its module-level N at call time
HAD = hadamard(128); OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def v32_fp8(x):  # V3.2 act_quant with scale_fmt ue8m0: power-of-two scale >= amax / 448, float8_e4m3fn
    s = torch.exp2(torch.ceil(torch.log2(x.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448.0)))
    return (x / s).clamp(-448, 448).to(torch.float8_e4m3fn).float() * s


def acc(x, y):  # bits, every vector counted once
    return (-0.5 * torch.log2(((x - y).pow(2).mean(-1) / x.pow(2).mean(-1).clamp_min(1e-30)).mean())).item()


@torch.no_grad()
def measure(x):
    """x [m, 128] fp32 on the GPU -> metrics for none / hadamard."""
    out = {}
    for rot in ("none", "hadamard"):
        z = x @ HAD if rot == "hadamard" else x
        am_v = z.abs().amax(-1, keepdim=True).clamp_min(1e-30); am_t = z.abs().amax()
        r = dict(kurtosis=kurtosis(z), crest=(am_v[:, 0] / z.pow(2).mean(-1).sqrt().clamp_min(1e-30)).mean().item(),
                 crest_tensor=(am_t / z.pow(2).mean().sqrt()).item(), v32_fp8=acc(z, v32_fp8(z)))
        for f in ("INT8", "E4M3", "E2M5", "E3M4"):
            r[f"{f}_per_vector"] = acc(z, gquant(z, f, am_v)); r[f"{f}_per_tensor"] = acc(z, gquant(z, f, am_t))
        out[rot] = r
    return out


def blocks(t): return t.reshape(-1, 128).float()


@torch.no_grad()
def layer_pass(L, h, fc, pos, full):
    A = D.project(L, h, fc); got = {}
    got["residual"] = blocks(h[pos]); got["attn_in"] = blocks(A["x"][pos]); got["q_lat"] = blocks(A["qr"][pos])
    got["idx_q"] = blocks(D.iq(L, A, pos, fc)); got["idx_k"] = blocks(A["ki"][pos])
    if not full: return got, None
    # D.forward, with the MLP input kept for the sampled positions
    o = torch.empty(N, D.HM * D.DV, device=dev)
    for a in range(0, N, 64):
        p = list(range(a, a + 64)); lg = D.logits(A, *D.mq(L, A, p, fc), p)
        sc = D.iscore(D.iq(L, A, p, fc), A["ki"], A["wi"][p]).masked_fill(torch.arange(N, device=dev)[None] > torch.tensor(p, device=dev)[:, None], float("-inf"))
        o[a:a + 64] = D.attend(L, A, lg, sc.topk(min(D.K, a + 64), -1).indices).flatten(1)
    h = h + L.lin(o, "o_proj"); del o, A
    x = rms(h[pos], L.v["post_attention_layernorm"]); got["mlp_down_in"] = blocks(Fn.silu(L.lin(x, "gate_proj")) * L.lin(x, "up_proj"))
    for a in range(0, N, 512):
        x = rms(h[a:a + 512], L.v["post_attention_layernorm"])
        h[a:a + 512] += L.lin(Fn.silu(L.lin(x, "gate_proj")) * L.lin(x, "up_proj"), "down_proj")
    return got, h


if __name__ == "__main__":
    tok = Tokenizer.from_file(os.path.join(F.HERE, "tokenizer.json")); g = torch.Generator().manual_seed(SEED)
    pos = sorted((torch.randperm(N - 256, generator=g)[:P] + 256).tolist()); fc = freqs_cis(N)
    seqs = {n: [0] + tok.encode(t).ids[:N - 1] for n, t in texts().items()}
    E = embed_rows(sorted({i for s in seqs.values() for i in s})); hs = {n: torch.stack([E[i] for i in s]).float() for n, s in seqs.items()}; del E
    rows = []
    for li in range(4):
        t0 = time.time(); full = li < 3; L = D.Layer(li, full=full); acc_t = {}
        for n in seqs:
            got, hn = layer_pass(L, hs[n].to(dev), fc, pos, full)
            for k, v in got.items(): acc_t.setdefault(k, []).append(v)
            if hn is not None: hs[n] = hn.cpu()
            del got, hn; torch.cuda.empty_cache()
        for k, v in acc_t.items():
            m = measure(torch.cat(v)); rows.append(dict(layer=li, tensor=k, vectors=sum(len(x) for x in v), **m))
            a, b = m["none"], m["hadamard"]
            print(f"L{li} {k:12s} kurt {a['kurtosis']:7.2f} -> {b['kurtosis']:5.2f}  crest {a['crest']:5.2f} -> {b['crest']:4.2f}   "
                  f"per-vector INT8 {a['INT8_per_vector']:.2f}->{b['INT8_per_vector']:.2f}  E4M3 {a['E4M3_per_vector']:.2f}->{b['E4M3_per_vector']:.2f}   "
                  f"per-tensor INT8 {a['INT8_per_tensor']:.2f}->{b['INT8_per_tensor']:.2f}  E4M3 {a['E4M3_per_tensor']:.2f}->{b['E4M3_per_tensor']:.2f}", flush=True)
        del L, acc_t; torch.cuda.empty_cache(); print(f"  layer {li} {time.time() - t0:.0f}s", flush=True)
        json.dump(dict(model="DeepSeek-V3.2-Exp layers 0-3", N=N, positions=P, texts=list(seqs), rows=rows), open(os.path.join(OUT, "ds_tails.json"), "w"), indent=1)
    print("wrote results/ds_tails.json")
