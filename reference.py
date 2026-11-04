"""Pure-PyTorch references (exact semantics, for tests) and baselines (what you would run without custom kernels)."""
import torch

GROUP = 128


# ---------------- references: float64 math on the same integers, so the only differences are fp32 rounding ----
def act_quant_ref(x, round_scale=False):
    *lead, N = x.shape
    g = x.float().view(*lead, N // GROUP, GROUP)
    s = torch.clamp(g.abs().amax(-1), min=1e-4) * (1.0 / 127.0)
    if round_scale:
        s = torch.exp2(torch.ceil(torch.log2(s)))
    y = torch.clamp(torch.round(g / s.unsqueeze(-1)), -127, 127).to(torch.int8)
    return y.view(*lead, N), s


def int8_gemm_ref(a, a_s, b, b_s):
    """Dequantize both sides block by block in float64, then matmul."""
    K = a.shape[-1]; N = b.shape[0]
    A = a.double().view(-1, K // GROUP, GROUP) * a_s.double().view(-1, K // GROUP, 1)
    bs = b_s.double().repeat_interleave(GROUP, 0)[:N]  # [N, K // 128]
    B = b.double().view(N, K // GROUP, GROUP) * bs.unsqueeze(-1)
    return (A.view(-1, K) @ B.view(N, K).T).view(*a.shape[:-1], N)


def int8_index_ref(q, w, k, k_s, causal=False, start=0):
    logits = torch.einsum("bmhd,bnd->bmhn", q.double(), k.double()).clamp_min(0)
    out = (logits * w.double().unsqueeze(-1)).sum(2) * k_s.double().unsqueeze(1)
    if causal:
        M, Nk = out.shape[1:]
        keep = torch.arange(Nk, device=q.device)[None, :] <= start + torch.arange(M, device=q.device)[:, None]
        out = out.masked_fill(~keep, float("-inf"))
    return out


# ---------------- baselines ------------------------------------------------------------------------------------
def act_quant_torch(x, round_scale=False):
    """Eager PyTorch, as you would write it without a kernel."""
    return act_quant_ref(x, round_scale)


def gemm_dequant_fp32(a, a_s, b_deq):
    """The float path: dequantize activations, then cuBLAS fp32 matmul against pre-dequantized weights.
    fp32 is the fastest float matmul on this GPU (fp16 measured 5x slower)."""
    K = a.shape[-1]
    A = (a.float().view(-1, K // GROUP, GROUP) * a_s.view(-1, K // GROUP, 1)).view(*a.shape[:-1], K)
    return A @ b_deq.T


def gemm_fp32(x, w):
    """Unquantized fp32 linear layer: the accuracy and speed target."""
    return x @ w.T


def int_mm_unscaled(a, bT):
    """torch._int_mm with no block scales: an upper bound on INT8 cuBLAS speed, not a usable layer."""
    return torch._int_mm(a, bT)


def index_torch_fp32(q, w, k, k_s, chunk=64):
    """Eager indexer on dequantized fp32 inputs, chunked over queries so the [m, h, n] logits fit in memory."""
    out = []
    kf = k.float()
    for i in range(0, q.shape[1], chunk):
        lg = torch.einsum("bmhd,bnd->bmhn", q[:, i:i + chunk].float(), kf).clamp_min_(0)
        out.append((lg * w[:, i:i + chunk].unsqueeze(-1)).sum(2) * k_s.unsqueeze(1))
    return torch.cat(out, 1)


def index_torch_int8(q, w, k, k_s, chunk=64):
    """Strong baseline: the same math through cuBLAS INT8 (torch._int_mm) on stacked heads, chunked over queries."""
    B, M, H, D = q.shape; out = []
    for b in range(B):
        kT = k[b].T.contiguous()
        rows = []
        for i in range(0, M, chunk):
            qi = q[b, i:i + chunk].reshape(-1, D)
            lg = torch._int_mm(qi, kT).float().clamp_min_(0).view(-1, H, k.shape[1])
            rows.append((lg * w[b, i:i + chunk].unsqueeze(-1)).sum(1) * k_s[b])
        out.append(torch.cat(rows, 0))
    return torch.stack(out)
