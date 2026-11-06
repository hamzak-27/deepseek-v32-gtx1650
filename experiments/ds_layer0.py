"""Real DeepSeek-V3.2-Exp indexer, layer 0: does the key offset exist there, and how do INT8 / FP8 compare?

Layer 0's input is RMSNorm(embedding(tokens)), so its lightning indexer can be computed exactly from a few tensors:
    x  = RMSNorm(embed[ids]) * attn_norm.w
    qr = RMSNorm(x @ q_a_proj.T) * q_a_layernorm.w
    q  = qr @ indexer.wq_b.T -> [64 heads, 128];  RoPE (non-interleaved, YaRN) on dims 0:64
    k  = LayerNorm(x @ indexer.wk.T; k_norm.w, k_norm.b);  RoPE on dims 0:64
    w  = x @ indexer.weights_proj.T * 64^-0.5 * 128^-0.5          (real per-head, per-query weights)
    score[m, n] = sum_h w[m, h] * relu(q[m, h] . k[n]);  top-2048
(all as Indexer.forward / MLA.forward in deepseek-ai/DeepSeek-V3.2-Exp inference/model.py; fp32 here as reference.)
Weights are fetched tensor by tensor (ds_fetch.py); only the embedding rows of the tokens used are downloaded.

Same texts, windows, query positions, formats, rotation and centering variants and lost-% metric as indexer_quant.py.
Attention mass is not reported: that would need layer 0's main attention too.
Run: python experiments/ds_layer0.py  -> results/ds_layer0.json
"""
import json, math, os, time
from concurrent.futures import ThreadPoolExecutor
import torch
from tokenizers import Tokenizer
import ds_fetch as F
from proxy import texts, hadamard, quant, FORMATS, dev

N, P, PREFIX, KS, SEED, H, D, R = 8192, 256, 1024, (512, 2048), 0, 64, 128, 64
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(os.path.dirname(HERE), "results")
HAD = hadamard(D)
L0 = "model.layers.0."


def embed_rows(ids):
    """Embedding rows for the given ids, fetched as merged runs of nearby rows in parallel; cached."""
    p = os.path.join(F.CACHE, "embed_rows.pt"); have = torch.load(p) if os.path.exists(p) else {}
    need = sorted(set(ids) - set(have)); runs = []
    for i in need:
        if runs and i - runs[-1][1] <= 4: runs[-1][1] = i + 1
        else: runs.append([i, i + 1])
    if runs:
        print(f"fetching {len(need)} embedding rows in {len(runs)} requests", flush=True)
        F.header(F.WMAP["model.embed_tokens.weight"])
        with ThreadPoolExecutor(16) as ex:
            for (a, b), t in zip(runs, ex.map(lambda r: F.raw("model.embed_tokens.weight", rows=tuple(r)), runs)):
                for j in range(a, b): have[j] = t[j - a].clone()
        torch.save(have, p)
    return have


def freqs_cis(n):  # YaRN as precompute_freqs_cis (inference max_seq_len > 4096, so the correction applies)
    dim, base, factor, orig, bf, bs = R, 10000.0, 40, 4096, 32, 1
    cdim = lambda rot: dim * math.log(orig / (rot * 2 * math.pi)) / (2 * math.log(base))
    low, high = max(math.floor(cdim(bf)), 0), min(math.ceil(cdim(bs)), dim - 1)
    f = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    smooth = 1 - ((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low)).clamp(0, 1)
    f = f / factor * (1 - smooth) + f * smooth
    return torch.polar(torch.ones(n, dim // 2), torch.outer(torch.arange(n, dtype=torch.float32), f))


def rope(x, fc):  # non-interleaved: pairs (i, i + 32); x [..., n, 64], fc [n, 32]
    a, b = x[..., :R // 2], x[..., R // 2:]
    c, s = fc.real.to(x.device), fc.imag.to(x.device)
    return torch.cat([a * c - b * s, a * s + b * c], -1)


def rms(x, w, eps=1e-6): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


@torch.no_grad()
def indexer(ids, qpos):
    E = embed_rows(ids); W = {n: F.load(L0 + n).to(dev) for n in (
        "input_layernorm.weight", "self_attn.q_a_proj.weight", "self_attn.q_a_layernorm.weight",
        "self_attn.indexer.wq_b.weight", "self_attn.indexer.wk.weight", "self_attn.indexer.k_norm.weight",
        "self_attn.indexer.k_norm.bias", "self_attn.indexer.weights_proj.weight")}
    x = rms(torch.stack([E[i] for i in ids]).float().to(dev), W["input_layernorm.weight"])
    fc = freqs_cis(len(ids))
    k = torch.nn.functional.layer_norm(x @ W["self_attn.indexer.wk.weight"].T, (D,),
                                       W["self_attn.indexer.k_norm.weight"], W["self_attn.indexer.k_norm.bias"], 1e-6)
    k = torch.cat([rope(k[:, :R], fc), k[:, R:]], -1)
    xq = x[qpos]
    qr = rms(xq @ W["self_attn.q_a_proj.weight"].T, W["self_attn.q_a_layernorm.weight"])
    q = (qr @ W["self_attn.indexer.wq_b.weight"].T).view(len(qpos), H, D).transpose(0, 1)  # [H, P, D]
    q = torch.cat([rope(q[..., :R], fc[qpos]), q[..., R:]], -1)
    w = (xq @ W["self_attn.indexer.weights_proj.weight"].T) * H ** -0.5 * D ** -0.5  # [P, H]
    return q, k, w


def score(q, k, w):  # q [H, P, D], k [N, D], w [P, H] -> [P, N]
    return torch.einsum("ph,hpn->pn", w, torch.relu(torch.einsum("hpd,nd->hpn", q, k)))


@torch.no_grad()
def evaluate(q, k, w, qpos):
    qpos_d = torch.tensor(qpos, device=dev); future = torch.arange(N, device=dev)[None, :] > qpos_d[:, None]
    ref = score(q, k, w).masked_fill(future, float("-inf"))
    kbar = k.mean(0); kc = k - kbar
    nope = k[:, R:]; nbar = nope.mean(0)
    stats = dict(k_offset=(kbar.norm() / kc.pow(2).sum(-1).mean().sqrt()).item(),
                 k_offset_nope=(nbar.norm() / (nope - nbar).pow(2).sum(-1).mean().sqrt()).item(),
                 k_offset_peak=(kbar.abs().max() / kbar.norm()).item(),
                 k_peak=(k.abs().amax(-1) / k.pow(2).mean(-1).sqrt()).mean().item(),
                 k_peak_had=((k @ HAD).abs().amax(-1) / k.pow(2).mean(-1).sqrt()).mean().item(),
                 w_neg_share=(w < 0).float().mean().item(),
                 positive_keys_per_query=(ref > 0).float().sum(-1).mean().item())
    sel = {}
    for kk in KS:
        sel[("fp32", "none", "none", kk)] = ref.topk(kk, -1).indices
        sel[("window", "none", "none", kk)] = (qpos_d[:, None] - torch.arange(kk, device=dev)[None]).clamp_min(0)
    kmean = k[:PREFIX].mean(0)
    for center in ("none", "k"):
        kin = k - kmean if center == "k" else k
        for rot in ("none", "hadamard"):
            Rm = HAD if rot == "hadamard" else None
            for fmt in FORMATS:
                qd = quant(q @ Rm if Rm is not None else q, fmt); kd = quant(kin @ Rm if Rm is not None else kin, fmt)
                if Rm is not None: qd, kd = qd @ Rm.T, kd @ Rm.T
                if center == "k": kd = kd + kmean
                s = score(qd, kd, w).masked_fill(future, float("-inf"))
                for kk in KS: sel[(fmt, rot, center, kk)] = s.topk(kk, -1).indices
    rows = []
    for (fmt, rot, center, kk), ix in sel.items():
        tau = ref.topk(kk, -1).values[:, -1:]
        rows.append(dict(k=kk, fmt=fmt, rot=rot, center=center, recall=(ref.gather(1, ix) >= tau).float().mean().item()))
    return rows, stats


if __name__ == "__main__":
    tok = Tokenizer.from_file(os.path.join(F.HERE, "tokenizer.json"))
    g = torch.Generator().manual_seed(SEED)
    qpos = sorted((torch.randperm(N // 2, generator=g)[:P] + N // 2).tolist())
    rows, stats = [], {}
    for name, text in texts().items():
        ids = tok.encode(text).ids
        for wdw in range(2):
            seq = f"{name}{wdw}"; t0 = time.time()
            q, k, w = indexer(ids[wdw * N:(wdw + 1) * N], qpos)
            r, st = evaluate(q, k, w, qpos)
            rows += [dict(seq=seq, **x) for x in r]; stats[seq] = st
            print(f"{seq}: {time.time() - t0:.0f}s  " + "  ".join(f"{a}={b:.3f}" for a, b in st.items()), flush=True)
    json.dump(dict(model="DeepSeek-V3.2-Exp layer 0 indexer", N=N, queries=P, prefix=PREFIX, ks=KS, rows=rows,
                   stats=stats), open(os.path.join(OUT, "ds_layer0.json"), "w"))
    for kk in KS:
        print(f"\n== DeepSeek-V3.2-Exp layer 0, top-{kk} of {N} ==\n{'format':10s} {'rotation':9s} {'center':7s} {'lost %':>7s}  per text")
        for fmt in ["fp32", "window"] + FORMATS:
            for center in ("none", "k"):
                for rot in ("none", "hadamard"):
                    s = [r for r in rows if r["k"] == kk and r["fmt"] == fmt and r["rot"] == rot and r["center"] == center]
                    if s:
                        print(f"{fmt:10s} {rot:9s} {center:7s} {100 * (1 - sum(x['recall'] for x in s) / len(s)):7.2f}  "
                              + " ".join(f"{100 * (1 - x['recall']):5.2f}" for x in s))
