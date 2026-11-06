"""Fetch single tensors from the DeepSeek-V3.2-Exp checkpoint without downloading whole shards.

A .safetensors file is: 8-byte header length, a JSON header with each tensor's dtype, shape and byte range, then the
raw bytes. So one small request reads the header, and one HTTP range request per tensor reads only that tensor.
Results are cached in experiments/ds_weights/cache/ as .pt files (fp32, fp8 weights dequantized with their 128x128
block scales, as weight_dequant in DeepSeek's inference/model.py does).
"""
import json, os, struct
import requests, torch

REPO = "https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/resolve/main/"
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ds_weights"); CACHE = os.path.join(HERE, "cache")
os.makedirs(CACHE, exist_ok=True)
WMAP = json.load(open(os.path.join(HERE, "index.json")))["weight_map"]
S = requests.Session(); _headers = {}
DT = {"BF16": torch.bfloat16, "F32": torch.float32, "F8_E4M3": torch.float8_e4m3fn, "F16": torch.float16}


def _get(shard, a, b):
    r = S.get(REPO + shard, headers={"Range": f"bytes={a}-{b - 1}"}, timeout=120)
    r.raise_for_status(); assert len(r.content) == b - a, (shard, a, b, len(r.content)); return r.content


def header(shard):
    if shard not in _headers:
        n = struct.unpack("<Q", _get(shard, 0, 8))[0]
        _headers[shard] = (8 + n, json.loads(_get(shard, 8, 8 + n)))
    return _headers[shard]


def raw(name, rows=None):
    """Tensor `name` as stored. rows=(r0, r1) fetches only rows r0:r1 of a 2-D tensor."""
    shard = WMAP[name]; base, h = header(shard); e = h[name]; a, b = e["data_offsets"]; dt = DT[e["dtype"]]; shape = list(e["shape"])
    if rows:
        rb = (b - a) // shape[0]; a, b = a + rows[0] * rb, a + rows[1] * rb; shape[0] = rows[1] - rows[0]
    return torch.frombuffer(bytearray(_get(shard, base + a, base + b)), dtype=dt).reshape(shape)


def dequant(w, s, bs=128):
    out, inn = w.shape
    w = w.float().reshape(out // bs, bs, inn // bs, bs) if out % bs == 0 and inn % bs == 0 else None
    if w is None: raise ValueError("shape not a multiple of the block size")
    return (w * s.float()[:, None, :, None]).reshape(out, inn)


def load(name):
    """fp32 tensor, dequantized if it is fp8; cached on disk."""
    p = os.path.join(CACHE, name + ".pt")
    if os.path.exists(p): return torch.load(p)
    t = raw(name)
    if t.dtype == torch.float8_e4m3fn: t = dequant(t, raw(name + "_scale_inv"))
    t = t.float(); torch.save(t, p); return t
