# 全量测试：只验证正确性、边界条件与 PyTorch CUDA 调用契约。
# 性能测试单独放在 bench_ops.py，避免测试与 benchmark 相互污染。
import torch
import torch.nn.functional as F
import ai_infra_ops

torch.manual_seed(42)
device = "cuda"

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

# ---------------- Softmax ----------------
out = ai_infra_ops.softmax(x)
ref = torch.softmax(x, dim=-1)
check("Softmax", out, ref)

# ---------------- LayerNorm（无 affine，对拍需去掉 gamma/beta） ----------------
out = ai_infra_ops.layernorm(x, eps)
mean = x.mean(-1, keepdim=True)
var = x.var(-1, unbiased=False, keepdim=True)
ref = (x - mean) / torch.sqrt(var + eps)
check("LayerNorm", out, ref)

# ---------------- GEMM ----------------
M, K, Nn = 1024, 1024, 1024
a = torch.randn(M, K, device=device)
b = torch.randn(K, Nn, device=device)
out = ai_infra_ops.gemm(a, b)
ref = a @ b
check("GEMM", out, ref, 0.5)  # fp32 tiled 累加顺序不同，容差放宽

# ---------------- GEMM MMA 边界回归 ----------------
# 覆盖小矩阵、奇数 N，以及 M/N/K 均非 Tensor Core tile 整数倍的情况。
gemm_mma_shapes = (
    (1, 1, 1),
    (3, 5, 7),
    (63, 15, 65),
    (65, 17, 33),
    (127, 129, 131),
    (128, 80, 192),  # vec/async 的 16-byte 对齐 fast path
)
max_gemm_mma_err = 0.0
for m, k_dim, n in gemm_mma_shapes:
    a16 = torch.randn(m, k_dim, device=device, dtype=torch.float16)
    b16 = torch.randn(k_dim, n, device=device, dtype=torch.float16)
    out = ai_infra_ops.gemm_mma(a16, b16)
    vec_out = ai_infra_ops.gemm_mma_vec(a16, b16)
    async_out = ai_infra_ops.gemm_mma_async(a16, b16)
    ldmatrix_out = ai_infra_ops.gemm_mma_ldmatrix(a16, b16)
    ldmatrix_padded_out = ai_infra_ops.gemm_mma_ldmatrix_padded(a16, b16)
    cublas_out = ai_infra_ops.gemm_cublas_fp32(a16, b16)
    ref = a16.float() @ b16.float()
    max_gemm_mma_err = max(
        max_gemm_mma_err,
        (out - ref).abs().max().item(),
    )
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(vec_out, ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(async_out, ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(ldmatrix_out, ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(ldmatrix_padded_out, ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(cublas_out, ref, rtol=2e-2, atol=2e-2)

print(f"[GEMM MMA edge suite] PASS  maxErr={max_gemm_mma_err:.3e}")

# contiguous 不代表 storage 起点仍为 16-byte 对齐；偏移 half tensor 必须回退通用路径。
offset_a_storage = torch.randn(64 * 16 + 1, device=device, dtype=torch.float16)
offset_b_storage = torch.randn(16 * 64 + 1, device=device, dtype=torch.float16)
offset_a = offset_a_storage[1:].view(64, 16)
offset_b = offset_b_storage[1:].view(16, 64)
assert offset_a.is_contiguous() and offset_b.is_contiguous()
assert offset_a.data_ptr() % 16 != 0 and offset_b.data_ptr() % 16 != 0
offset_ref = offset_a.float() @ offset_b.float()
torch.testing.assert_close(
    ai_infra_ops.gemm_mma_async(offset_a, offset_b),
    offset_ref,
    rtol=2e-2,
    atol=2e-2,
)
# v8 与 v5/v6 共用 16-byte fast-path dispatch 条件；未对齐时也必须安全回退 v4。
torch.testing.assert_close(
    ai_infra_ops.gemm_mma_ldmatrix_padded(offset_a, offset_b),
    offset_ref,
    rtol=2e-2,
    atol=2e-2,
)
print("[GEMM MMA misaligned-storage fallback] PASS")

# 确定性 fragment 映射测试：A 的每一行只选择 B 的一行。
# 如果 ldmatrix 的四个 A 子矩阵或 B 的 .trans 顺序错误，输出会呈现整行错位。
map_a = torch.zeros(64, 16, device=device, dtype=torch.float16)
map_rows = torch.arange(64, device=device)
map_a[map_rows, map_rows % 16] = 1
map_b = ((torch.arange(16 * 64, device=device) % 31) - 15).view(16, 64).half() / 16
map_ref = map_a.float() @ map_b.float()
for op in (ai_infra_ops.gemm_mma_ldmatrix, ai_infra_ops.gemm_mma_ldmatrix_padded):
    torch.testing.assert_close(op(map_a, map_b), map_ref, rtol=0, atol=0)
print("[GEMM MMA ldmatrix deterministic mapping] PASS")

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

# ---------------- FlashAttention v5（P fragment 寄存器直连） ----------------
out = ai_infra_ops.flashattention_v5(q16, k16, v16, False)
check("FlashAttention v5 full", out.float(), torch_attn_fp16(False).float(), 2e-2)
out = ai_infra_ops.flashattention_v5(q16, k16, v16, True)
check("FlashAttention v5 causal", out.float(), torch_attn_fp16(True).float(), 2e-2)

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

# ---------------- FlashAttention v6 D=128 ----------------
# D=128 会把 Q/K/V 的 shared-memory stride 和 PV 输出 tile 数都翻倍。
B6, H6, N6, D6 = 2, 2, 65, 128
q128 = torch.randn(B6, H6, N6, D6, device=device, dtype=torch.float16)
k128 = torch.randn_like(q128)
v128 = torch.randn_like(q128)
for causal in (False, True):
    out = ai_infra_ops.flashattention_v6(q128, k128, v128, causal)
    ref = F.scaled_dot_product_attention(q128, k128, v128, is_causal=causal)
    check(
        f"FlashAttention v6 D128 {'causal' if causal else 'full'}",
        out.float(),
        ref.float(),
        3e-2,
    )

# ---------------- PyTorch CUDA stream / 输入契约回归 ----------------
# 输入生产、手写算子和 reference 全部排入同一条非默认 stream。若 wrapper
# 错误地把 kernel 发往默认 stream，kernel 可能在输入尚未生成时就开始读取。
test_stream = torch.cuda.Stream()
with torch.cuda.stream(test_stream):
    if hasattr(torch.cuda, "_sleep"):
        torch.cuda._sleep(5_000_000)

    xs = torch.randn(17, 70, device=device)
    gs = torch.rand(70, device=device) + 0.5
    rms_out = ai_infra_ops.rmsnorm(xs, gs, eps)
    rms_ref = xs * torch.rsqrt(xs.pow(2).mean(-1, keepdim=True) + eps) * gs

    softmax_out = ai_infra_ops.softmax(xs)
    softmax_ref = torch.softmax(xs, dim=-1)

    layernorm_out = ai_infra_ops.layernorm(xs, eps)
    layernorm_ref = (xs - xs.mean(-1, keepdim=True)) / torch.sqrt(
        xs.var(-1, unbiased=False, keepdim=True) + eps
    )

    ga = torch.randn(35, 19, device=device)
    gb = torch.randn(19, 27, device=device)
    gemm_out = ai_infra_ops.gemm(ga, gb)
    gemm_ref = ga @ gb

    ga16 = ga.half()
    gb16 = gb.half()
    gemm_mma_out = ai_infra_ops.gemm_mma(ga16, gb16)
    gemm_cublas_out = ai_infra_ops.gemm_cublas_fp32(ga16, gb16)
    gemm_mma_ref = ga16.float() @ gb16.float()

    pipeline_a = torch.randn(64, 16, device=device, dtype=torch.float16)
    pipeline_b = torch.randn(16, 64, device=device, dtype=torch.float16)
    gemm_async_out = ai_infra_ops.gemm_mma_async(pipeline_a, pipeline_b)
    # ldmatrix 是 warp 同步指令；放在非默认 stream 中可同时检查当前 stream
    # 获取是否正确，以及扩展没有偷偷落到 legacy default stream。
    gemm_ldmatrix_padded_out = ai_infra_ops.gemm_mma_ldmatrix_padded(
        pipeline_a, pipeline_b
    )
    gemm_async_ref = pipeline_a.float() @ pipeline_b.float()

    sq = torch.randn(65, 64, device=device)
    sk = torch.randn_like(sq)
    sv = torch.randn_like(sq)
    flash_out = ai_infra_ops.flashattention(sq, sk, sv, False)
    flash_ref = torch.softmax((sq @ sk.T) * (64 ** -0.5), dim=-1) @ sv

test_stream.synchronize()
check("RMSNorm non-default stream", rms_out, rms_ref, 1e-4)
check("Softmax non-default stream", softmax_out, softmax_ref, 1e-4)
check("LayerNorm non-default stream", layernorm_out, layernorm_ref, 1e-4)
check("GEMM non-default stream", gemm_out, gemm_ref, 1e-4)
check("GEMM MMA non-default stream", gemm_mma_out, gemm_mma_ref, 2e-2)
check("GEMM MMA async non-default stream", gemm_async_out, gemm_async_ref, 2e-2)
check(
    "GEMM MMA ldmatrix+padded non-default stream",
    gemm_ldmatrix_padded_out,
    gemm_async_ref,
    2e-2,
)
check("cuBLAS baseline non-default stream", gemm_cublas_out, gemm_mma_ref, 2e-2)
check("FlashAttention v3 non-default stream", flash_out, flash_ref, 1e-2)

# 空 batch 可以直接返回空输出；K=0 的 GEMM 按数学语义返回全零。
empty_x = torch.empty(0, 17, device=device)
empty_g = torch.ones(17, device=device)
assert ai_infra_ops.rmsnorm(empty_x, empty_g, eps).shape == empty_x.shape
assert ai_infra_ops.softmax(empty_x).shape == empty_x.shape
assert ai_infra_ops.layernorm(empty_x, eps).shape == empty_x.shape
zero_k_out = ai_infra_ops.gemm(
    torch.empty(3, 0, device=device), torch.empty(0, 5, device=device)
)
assert zero_k_out.shape == (3, 5) and torch.count_nonzero(zero_k_out).item() == 0

# 防止 fp16 权重被 reinterpret_cast 成 float* 后静默产生错误结果。
try:
    ai_infra_ops.rmsnorm(xs, gs.half(), eps)
except RuntimeError as exc:
    assert "float32" in str(exc)
else:
    raise AssertionError("RMSNorm must reject a non-float32 weight")

print("[PyTorch CUDA contract suite] PASS")
