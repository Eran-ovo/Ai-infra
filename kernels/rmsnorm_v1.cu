#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#define BLOCK 256

// RMSNorm: y = x / sqrt(mean(x^2) + eps) * g
// 与 LayerNorm 的区别：不减均值（去 centering），只需一次归约（sum of x^2）
// LLaMA/Qwen 标配。面试考点：为什么比 LayerNorm 快？
//   1) 一次归约 vs 两次串行归约  2) 不用暂存 x-mean，少读写一次显存
__global__ void rmsnorm_fused(const float* __restrict__ x,
                              const float* __restrict__ g,   // 可学习的缩放参数 gamma
                              float* __restrict__ y, int N, float eps){
    int row = blockIdx.x;              // 该 block 处理的行号
    const float* rx = x + row * N;     // 输入行起始地址
    float* ry = y + row * N;           // 输出行起始地址
    __shared__ float s[BLOCK];
    int tid = threadIdx.x;

    // 1. 线程粗化：每线程串行扫 N/BLOCK 个元素，累加局部平方和
    float sum_sq = 0;
    for (int idx = tid; idx < N; idx += BLOCK)
        sum_sq += rx[idx] * rx[idx];
    s[tid] = sum_sq;
    __syncthreads();

    // 2. 块内树形归约：256 -> 128 -> 64 -> 32（注意 s2 >= 32，做到只剩 32 个！）
    for (int s2 = BLOCK/2; s2 >= 32; s2 >>= 1) {
        if (tid < s2)
            s[tid] += s[tid + s2];
        __syncthreads();
    }
    // 3. warp 内 shuffle 归约（32 -> 1），免 __syncthreads
    //    从 16 开始：32 已经是 warp 宽度，shfl_down(32) 会越界返回自身导致翻倍
    if (tid < 32) {
        float val = s[tid];
        for (int s2 = 16; s2 > 0; s2 >>= 1)
            val += __shfl_down_sync(0xffffffff, val, s2);
        if (tid == 0)
            s[0] = val;
    }
    __syncthreads();

    // 4. 归一化系数：1 / sqrt(mean(x^2) + eps)
    float inv = rsqrtf(s[0] / N + eps);

    // 5. 写回：y = x * inv * g（g 是可学习参数，推理时也要乘）
    for (int idx = tid; idx < N; idx += BLOCK)
        ry[idx] = rx[idx] * inv * g[idx];
}

// CPU 参考实现，用于对拍
static void rmsnorm_cpu(const float* x, const float* g, float* y, int B, int N, float eps){
    for (int b = 0; b < B; ++b) {
        const float* rx = x + b * N;
        float* ry = y + b * N;
        float sum_sq = 0;
        for (int i = 0; i < N; ++i) sum_sq += rx[i] * rx[i];
        float inv = 1.0f / sqrtf(sum_sq / N + eps);
        for (int i = 0; i < N; ++i) ry[i] = rx[i] * inv * g[i];
    }
}

int main(){
    int B = 1024, N = 1024;
    float eps = 1e-5;
    size_t bytes = B * N * sizeof(float);
    size_t gbytes = N * sizeof(float);

    float *hX = (float*)malloc(bytes);
    float *hG = (float*)malloc(gbytes);
    float *hY = (float*)malloc(bytes);
    float *hRef = (float*)malloc(bytes);
    srand(42);
    for (int i = 0; i < B * N; ++i) hX[i] = (rand() % 100) / 10.0f;
    for (int i = 0; i < N; ++i)     hG[i] = (rand() % 10) / 10.0f + 0.5f; // gamma 一般初始化在 1 附近

    // CPU 对拍参考
    rmsnorm_cpu(hX, hG, hRef, B, N, eps);

    float *dX, *dG, *dY;
    cudaMalloc(&dX, bytes);
    cudaMalloc(&dG, gbytes);
    cudaMalloc(&dY, bytes);
    cudaMemcpy(dX, hX, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dG, hG, gbytes, cudaMemcpyHostToDevice);

    rmsnorm_fused<<<B, BLOCK>>>(dX, dG, dY, N, eps);
    cudaDeviceSynchronize();

    // 计时 100 次取平均
    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);
    cudaEventRecord(s);
    for (int i = 0; i < 100; ++i)
        rmsnorm_fused<<<B, BLOCK>>>(dX, dG, dY, N, eps);
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms;
    cudaEventElapsedTime(&ms, s, e);
    ms /= 100;
    std::cout << "RMSNorm Fused 1024x1024: " << ms << " ms" << std::endl;

    // 对拍
    cudaMemcpy(hY, dY, bytes, cudaMemcpyDeviceToHost);
    float err = 0.0f;
    for (int i = 0; i < B * N; ++i) err = fmaxf(err, fabsf(hY[i] - hRef[i]));
    std::cout << (err < 1e-3 ? "PASS!" : "FAIL!") << " maxErr=" << err
              << " Sample y[0]=" << hY[0] << " Ref=" << hRef[0] << std::endl;

    cudaFree(dX); cudaFree(dG); cudaFree(dY);
    free(hX); free(hG); free(hY); free(hRef);
    return 0;
}
