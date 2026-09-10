# 全量测试：所有算子的正确性对拍、FlashAttention 边界/BHD 回归与基础 benchmark
import time
import torch
import torch.nn.functional as F
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
    if not ok:
        raise AssertionError(f"{name} maxErr={err:.3e} exceeds tol={tol:.3e}")
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

# ---------------- FlashAttention v4（fp16 Tensor Core，含尾块 + causal） ----------------
# N=129 专门覆盖不是 64 倍数的 K/V 尾块；v4 的 Q/K/V 与输出均为 fp16。
N3 = 129
q16 = torch.randn(N3, 64, device=device, dtype=torch.float16)
k16 = torch.randn(N3, 64, device=device, dtype=torch.float16)
v16 = torch.randn(N3, 64, device=device, dtype=torch.float16)

def torch_attn_fp16(causal):
    # 用 fp16 量化后的输入做 fp32 reference，最后转回 fp16 对齐 v4 输出语义。
    s = (q16.float() @ k16.float().T) * (64 ** -0.5)
    if causal:
        future = torch.triu(torch.ones(N3, N3, device=device, dtype=torch.bool), diagonal=1)
        s = s.masked_fill(future, float("-inf"))
    return (torch.softmax(s, dim=-1) @ v16.float()).half()

out = ai_infra_ops.flashattention_fp16(q16, k16, v16, False)
check("FlashAttention v4 full", out.float(), torch_attn_fp16(False).float(), 2e-2)
out = ai_infra_ops.flashattention_fp16(q16, k16, v16, True)
check("FlashAttention v4 causal", out.float(), torch_attn_fp16(True).float(), 2e-2)
t_v4_f = bench(lambda: ai_infra_ops.flashattention_fp16(q16, k16, v16, False), iters=50)
t_v4_c = bench(lambda: ai_infra_ops.flashattention_fp16(q16, k16, v16, True), iters=50)
print(f"  v4 full {t_v4_f:.4f} ms | causal {t_v4_c:.4f} ms | causal speedup {t_v4_f/t_v4_c:.2f}x")

# ---------------- FlashAttention v5（P fragment 寄存器直连） ----------------
out = ai_infra_ops.flashattention_v5(q16, k16, v16, False)
check("FlashAttention v5 full", out.float(), torch_attn_fp16(False).float(), 2e-2)
out = ai_infra_ops.flashattention_v5(q16, k16, v16, True)
check("FlashAttention v5 causal", out.float(), torch_attn_fp16(True).float(), 2e-2)
t_v5_f = bench(lambda: ai_infra_ops.flashattention_v5(q16, k16, v16, False), iters=50)
t_v5_c = bench(lambda: ai_infra_ops.flashattention_v5(q16, k16, v16, True), iters=50)
print(f"  v5 full {t_v5_f:.4f} ms | causal {t_v5_c:.4f} ms | causal speedup {t_v5_f/t_v5_c:.2f}x")

# ---------------- FlashAttention v4/v5 边界回归 ----------------
# 同时覆盖 Bq/Bc=64 的边界前、边界上、边界后以及多个 KV tile。
edge_sizes = (1, 7, 63, 64, 65, 127, 128, 129, 257)
max_edge_err = 0.0
max_v4_v5_diff = 0.0
for n in edge_sizes:
    qe = torch.randn(n, 64, device=device, dtype=torch.float16)
    ke = torch.randn(n, 64, device=device, dtype=torch.float16)
    ve = torch.randn(n, 64, device=device, dtype=torch.float16)
    scores = (qe.float() @ ke.float().T) * (64 ** -0.5)
    for causal in (False, True):
        masked_scores = scores
        if causal:
            future = torch.triu(
                torch.ones(n, n, device=device, dtype=torch.bool), diagonal=1
            )
            masked_scores = scores.masked_fill(future, float("-inf"))
        ref = (torch.softmax(masked_scores, dim=-1) @ ve.float()).half()
        out4 = ai_infra_ops.flashattention_fp16(qe, ke, ve, causal)
        out5 = ai_infra_ops.flashattention_v5(qe, ke, ve, causal)
        max_edge_err = max(
            max_edge_err,
            (out4.float() - ref.float()).abs().max().item(),
            (out5.float() - ref.float()).abs().max().item(),
        )
        max_v4_v5_diff = max(
            max_v4_v5_diff,
            (out4.float() - out5.float()).abs().max().item(),
        )

edge_ok = max_edge_err < 2e-2 and max_v4_v5_diff == 0.0
print(
    f"[FlashAttention v4/v5 edge suite] {'PASS' if edge_ok else 'FAIL'}  "
    f"maxErr={max_edge_err:.3e}  v4-v5={max_v4_v5_diff:.3e}"
)
if not edge_ok:
    raise AssertionError("FlashAttention v4/v5 edge regression failed")

# ---------------- FlashAttention v5 批量多头 [B,H,N,D] ----------------
# N=65 同时覆盖多头寻址隔离和每个 head 内的 K/V tail。
B4, H4, N4 = 2, 3, 65
qb = torch.randn(B4, H4, N4, 64, device=device, dtype=torch.float16)
kb = torch.randn_like(qb)
vb = torch.randn_like(qb)
for causal in (False, True):
    out = ai_infra_ops.flashattention_v5(qb, kb, vb, causal)
    ref = F.scaled_dot_product_attention(qb, kb, vb, is_causal=causal)
    check(
        f"FlashAttention v5 BHD {'causal' if causal else 'full'}",
        out.float(),
        ref.float(),
        2e-2,
    )
