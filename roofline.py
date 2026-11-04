"""Measure the GPU's real ceilings: matmul throughput per dtype and memory bandwidth."""
import torch, triton
torch.backends.cuda.matmul.allow_tf32 = False
print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0))
N = 4096
for name, dt in [("fp32", torch.float32), ("fp16", torch.float16), ("bf16", torch.bfloat16)]:
    a = torch.randn(N, N, device="cuda", dtype=dt); b = torch.randn(N, N, device="cuda", dtype=dt)
    ms = triton.testing.do_bench(lambda: a @ b, warmup=50, rep=300)
    print(f"{name:5s} matmul {N}^3: {ms:7.2f} ms  {2*N**3/ms/1e9:6.2f} TFLOP/s")
a = torch.randint(-127, 127, (N, N), device="cuda", dtype=torch.int8); b = torch.randint(-127, 127, (N, N), device="cuda", dtype=torch.int8)
ms = triton.testing.do_bench(lambda: torch._int_mm(a, b), warmup=50, rep=300)
print(f"int8  _int_mm {N}^3: {ms:7.2f} ms  {2*N**3/ms/1e9:6.2f} TOP/s")
x = torch.empty(256 * 2**20 // 4, device="cuda"); y = torch.empty_like(x)
ms = triton.testing.do_bench(lambda: y.copy_(x), warmup=20, rep=200)
print(f"copy 256MB: {ms:.2f} ms  {2*x.numel()*4/ms/1e6:.1f} GB/s")
