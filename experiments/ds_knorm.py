"""DeepSeek-V3.2-Exp indexer keys: k = LayerNorm(wk x) = gamma * z + beta, then RoPE on dims 0:64. beta on dims 64:128
is one offset added to every key. Its size relative to the key spread (about |gamma|, since z has unit second moment per
channel on average) is comparable to indexer_quant.py's k_offset (|mean key| / RMS spread). All 61 layers + the MTP layer.
Run: python experiments/ds_knorm.py -> results/ds_knorm.json
"""
import json, os, statistics
import ds_fetch as F

rows = []
for L in range(62):
    g = F.load(f"model.layers.{L}.self_attn.indexer.k_norm.weight"); b = F.load(f"model.layers.{L}.self_attn.indexer.k_norm.bias")
    rows.append(dict(layer=L, bias_offset_nope=(b[64:].norm() / g[64:].norm()).item(), bias_offset_all=(b.norm() / g.norm()).item(),
                     gamma_mean_abs=g.abs().mean().item(), beta_max_abs=b.abs().max().item()))
    print("%2d  beta/gamma (no-RoPE dims) %.3f   all dims %.3f   mean|gamma| %.3f" % (L, rows[-1]["bias_offset_nope"], rows[-1]["bias_offset_all"], rows[-1]["gamma_mean_abs"]))
med = statistics.median(r["bias_offset_nope"] for r in rows); mx = max(r["bias_offset_nope"] for r in rows)
print(f"median {med:.4f}, max {mx:.4f}")
json.dump(dict(rows=rows, median_bias_offset_nope=med, max_bias_offset_nope=mx),
          open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "ds_knorm.json"), "w"), indent=1)
