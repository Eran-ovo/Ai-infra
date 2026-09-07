# 全量测试：5 个算子的正确性对拍 + 性能 benchmark（交错计时，同会话）
import time
import torch
import ai_infra_ops

torch.manual_seed(42)
device = "cuda"


def bench(fn, iters=200):
    for _ in range(20):
        fn()  # warmup
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000  # ms


def check(name, out, ref, tol=1e-3):
    err = (out - ref).abs().max().item()
    ok = err < tol
    print(f"[{name}] {'PASS' if ok else 'FAIL'}  maxErr={err:.3e}")
    return ok


# ---------------- RMSNorm ----------------
B, N = 1024, 1024
eps = 1e-5
x = torch.randn(B, N, device=device)
g = torch.rand(N, device=device) + 0.5
ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * g
out = ai_infra_ops.rmsnorm(x, g, eps)
check("RMSNorm", out, ref, 1e-4)
t_mine = bench(lambda: ai_infra_ops.rmsnorm(x, g, eps))
t_torch = bench(lambda: x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * g)
print(f"  custom {t_mine:.4f} ms | torch {t_torch:.4f} ms | {t_torch/t_mine:.2f}x")

# ---------------- Softmax ----------------
out = ai_infra_ops.softmax(x)
ref = torch.softmax(x, dim=-1)
check("Softmax", out, ref)
t_mine = bench(lambda: ai_infra_ops.softmax(x))
t_torch = bench(lambda: torch.softmax(x, dim=-1))
print(f"  custom {t_mine:.4f} ms | torch {t_torch:.4f} ms | {t_torch/t_mine:.2f}x")

# ---------------- LayerNorm（无 affine，对拍需去掉 gamma/beta） ----------------
out = ai_infra_ops.layernorm(x, eps)
mean = x.mean(-1, keepdim=True)
var = x.var(-1, unbiased=False, keepdim=True)
ref = (x - mean) / torch.sqrt(var + eps)
check("LayerNorm", out, ref)
t_mine = bench(lambda: ai_infra_ops.layernorm(x, eps))
t_torch = bench(lambda: (x - x.mean(-1, keepdim=True)) / torch.sqrt(x.var(-1, unbiased=False, keepdim=True) + eps))
print(f"  custom {t_mine:.4f} ms | torch {t_torch:.4f} ms | {t_torch/t_mine:.2f}x")

# ---------------- GEMM ----------------
M, K, Nn = 1024, 1024, 1024
a = torch.randn(M, K, device=device)
b = torch.randn(K, Nn, device=device)
out = ai_infra_ops.gemm(a, b)
ref = a @ b
check("GEMM", out, ref, 0.5)  # fp32 tiled 累加顺序不同，容差放宽
t_mine = bench(lambda: ai_infra_ops.gemm(a, b), iters=20)
t_torch = bench(lambda: a @ b, iters=20)
print(f"  custom {t_mine:.4f} ms | torch {t_torch:.4f} ms | {t_torch/t_mine:.2f}x")

# ---------------- FlashAttention（v3：N=8192 D=64） ----------------
N2 = 8192
q = torch.randn(N2, 64, device=device)
k = torch.randn(N2, 64, device=device)
v = torch.randn(N2, 64, device=device)
scale = 64 ** -0.5

def torch_attn(causal):
    m = torch.tril(torch.ones(N2, N2, device=device)) if causal else None
    s = (q @ k.T) * scale
    if causal:
        s = s.masked_fill(m == 0, float("-inf"))
    p = torch.softmax(s, dim=-1)
    return p @ v

out_full = ai_infra_ops.flashattention(q, k, v, False)
out_causal = ai_infra_ops.flashattention(q, k, v, True)
check("FlashAttention full", out_full, torch_attn(False), 1e-2)
check("FlashAttention causal", out_causal, torch_attn(True), 1e-2)
t_mine_f = bench(lambda: ai_infra_ops.flashattention(q, k, v, False), iters=20)
t_mine_c = bench(lambda: ai_infra_ops.flashattention(q, k, v, True), iters=20)
print(f"  custom full {t_mine_f:.4f} ms | causal {t_mine_c:.4f} ms | causal speedup {t_mine_f/t_mine_c:.2f}x")
