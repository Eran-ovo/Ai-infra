#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cmath>

// v0: 最朴素的GEMM，一个线程算C的一个元素
// 完全照着PMPP第5章的思路，不做任何优化
__global__ void gemm_naive(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // 行号
    int col = blockIdx.x * blockDim.x + threadIdx.x; // 列号

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main() {
    const int M = 512, N = 512, K = 512;
    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);

    // 1. CPU上分配内存并随机初始化
    float *h_A = (float*)malloc(bytes_A);
    float *h_B = (float*)malloc(bytes_B);
    float *h_C = (float*)malloc(bytes_C);
    float *h_C_ref = (float*)malloc(bytes_C);
    for (int i = 0; i < M*K; ++i) h_A[i] = rand() % 5;
    for (int i = 0; i < K*N; ++i) h_B[i] = rand() % 5;

    // 2. CPU算一遍正确结果，用来对比
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float sum = 0;
            for (int k = 0; k < K; ++k) sum += h_A[i*K+k] * h_B[k*N+j];
            h_C_ref[i*N+j] = sum;
        }

    // 3. GPU上分配内存
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes_A);
    cudaMalloc(&d_B, bytes_B);
    cudaMalloc(&d_C, bytes_C);

    // 4. 拷贝到GPU -> 计算 -> 拷回来
    cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice);

    dim3 block(16, 16); // 一个block 16x16=256个线程
    dim3 grid((N+15)/16, (M+15)/16); // 需要多少个block

    gemm_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // 5. 验证结果
    bool ok = true;
    for (int i = 0; i < M*N; ++i) {
        if (fabs(h_C[i] - h_C_ref[i]) > 1e-3) { ok = false; break; }
    }
    if (ok) std::cout << "PASS! GEMM v0 Naive Correct. M=N=K=512" << std::endl;
    else std::cout << "FAIL! 结果不对" << std::endl;

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C); free(h_C_ref);
    return 0;
}