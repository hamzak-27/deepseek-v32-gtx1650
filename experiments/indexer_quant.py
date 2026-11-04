"""Does INT8 instead of FP8 change which tokens DeepSeek-V3.2's lightning indexer selects?
Does the Hadamard rotation V3.2 applies before quantizing matter? And does centering the keys fix what is left?

V3.2 indexer (inference/model.py, Indexer.forward):
    q, k  -> Hadamard rotation -> act_quant (one scale per 128-dim vector, FP8 e4m3)
    score[m, n] = sum_h w[m, h] * relu(q[m, h] . k[n]);  keep top-2048 keys per query

V3.2 (671B) cannot run here, so the proxy indexer uses REAL post-RoPE q/k from a Qwen2.5 model, the query heads of each
key group as indexer heads, uniform head weights. The question is numerical: on realistic activations, how far does
each number format move the top-k set away from full precision?

Variants: format x rotation (none / Hadamard) x key centering (none / subtract the mean key of the first 1,024 tokens
before quantizing, keep that mean in fp32, add q . mean back exactly; the indexer function is unchanged).

Metrics, averaged over 256 queries in [N/2, N):
  recall = share of selected keys whose full-precision score is >= the k-th best full-precision score (tie-aware)
  attn   = share of the model's real softmax attention (same heads, full precision) on the selected keys

Run: python experiments/indexer_quant.py [Qwen2.5-0.5B-Instruct | Qwen2.5-1.5B-Instruct]
"""
import json, math, os, sys, time
import torch
from proxy import Qwen2, texts, hadamard, quant, index_score, FORMATS, dev

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen2.5-0.5B-Instruct"
N, P, PREFIX, KS, SEED = 8192, 256, 1024, (512, 2048), 0
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
m = Qwen2(MODEL); HAD = hadamard(m.D)


@torch.no_grad()
def evaluate(caps, qpos):
    qpos_d = torch.tensor(qpos, device=dev); future = torch.arange(N, device=dev)[None, :] > qpos_d[:, None]
    out, stats_rows = [], []
    for l, (qa, _, ka) in enumerate(caps):
        for g in range(m.HK):
            q = qa[g * m.G:(g + 1) * m.G].to(dev); k = ka[g].to(dev)
            ref = index_score(q, k).masked_fill(future, float("-inf"))
            attn = torch.softmax((torch.einsum("hpd,nd->hpn", q, k) / math.sqrt(m.D)).masked_fill(future, float("-inf")),
                                 -1).mean(0)
            # layer statistics for the mechanism analysis
            kbar = k.mean(0); kc = k - kbar; qbar = q.mean(1, keepdim=True)
            peak = lambda t: (t.abs().amax(-1) / t.pow(2).mean(-1).sqrt()).mean().item()
            stats_rows.append(dict(layer=l, group=g,
                                   k_offset=(kbar.norm() / kc.pow(2).sum(-1).mean().sqrt()).item(),
                                   q_offset=(qbar.norm(dim=-1) / (q - qbar).pow(2).sum(-1).mean(-1, keepdim=True).sqrt()).mean().item(),
                                   k_peak=peak(k), k_peak_had=peak(k @ HAD), q_peak=peak(q), q_peak_had=peak(q @ HAD),
                                   k_offset_peak=(kbar.abs().max() / kbar.norm()).item()))
            sel = {}
            for kk in KS:
                sel[("fp32", "none", "none", kk)] = ref.topk(kk, -1).indices
                sel[("window", "none", "none", kk)] = (qpos_d[:, None] - torch.arange(kk, device=dev)[None]).clamp_min(0)
            kmean = k[:PREFIX].mean(0)  # causal: known before any query we score
            for center in ("none", "k"):
                kin = k - kmean if center == "k" else k
                for rot in ("none", "hadamard"):
                    R = HAD if rot == "hadamard" else None
                    for fmt in FORMATS:
                        qd = quant(q @ R if R is not None else q, fmt)
                        kd = quant(kin @ R if R is not None else kin, fmt)
                        if R is not None: qd, kd = qd @ R.T, kd @ R.T  # back to the original basis (orthogonal)
                        if center == "k": kd = kd + kmean
                        s = index_score(qd, kd).masked_fill(future, float("-inf"))
                        for kk in KS: sel[(fmt, rot, center, kk)] = s.topk(kk, -1).indices
            for (fmt, rot, center, kk), ix in sel.items():
                tau = ref.topk(kk, -1).values[:, -1:]
                out.append(dict(layer=l, group=g, k=kk, fmt=fmt, rot=rot, center=center,
                                recall=(ref.gather(1, ix) >= tau).float().mean().item(),
                                attn=attn.gather(1, ix).sum(-1).mean().item()))
    return out, stats_rows


g = torch.Generator().manual_seed(SEED)
qpos = sorted((torch.randperm(N // 2, generator=g)[:P] + N // 2).tolist())
rows, lstats, meta = [], [], {}
for name, text in texts().items():
    ids = m.tok.encode(text).ids
    for w in range(2):
        seq = f"{name}{w}"; t0 = time.time()
        caps, loss = m.forward(torch.tensor(ids[w * N:(w + 1) * N], device=dev), qpos, PREFIX)
        r, st = evaluate(caps, qpos)
        rows += [dict(seq=seq, **x) for x in r]; lstats += [dict(seq=seq, **x) for x in st]
        meta[seq] = dict(loss=round(loss, 4)); print(f"{seq}: next-token loss {loss:.3f}, {time.time() - t0:.0f}s", flush=True)
        del caps; torch.cuda.empty_cache()

tag = MODEL.split("-")[1].lower()  # "0.5b" / "1.5b"
json.dump(dict(model=MODEL, layers=m.L, head_dim=m.D, heads_per_key=m.G, N=N, queries=P, prefix=PREFIX, ks=KS,
               seqs=meta, rows=rows, layer_stats=lstats),
          open(os.path.join(HERE, "results", f"indexer_quant_{tag}.json"), "w"))

# summary over layers where the top-k choice is not trivial (some format gets it wrong somewhere)
for kk in KS:
    triv = {l for l in range(m.L) if all(r["recall"] == 1.0 for r in rows if r["layer"] == l and r["k"] == kk
                                          and r["fmt"] != "window")}
    print(f"\n== {MODEL}: top-{kk} of {N}; {m.L - len(triv)} of {m.L} layers non-trivial ==")
    print(f"{'format':10s} {'rotation':9s} {'center':7s} {'lost %':>7s} {'attn':>7s}")
    for fmt in ["fp32", "window"] + FORMATS:
        for center in ("none", "k"):
            for rot in ("none", "hadamard"):
                s = [r for r in rows if r["k"] == kk and r["fmt"] == fmt and r["rot"] == rot and r["center"] == center
                     and r["layer"] not in triv]
                if s:
                    print(f"{fmt:10s} {rot:9s} {center:7s} {100 * (1 - sum(x['recall'] for x in s) / len(s)):7.2f} "
                          f"{100 * sum(x['attn'] for x in s) / len(s):7.2f}")
