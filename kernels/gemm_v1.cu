#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#define TILE 32

// v1: Shared Memory Tiling分块
__global__ void gemm_tiled(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        // 协作加载一块Tile到共享内存
        if (row < M && t*TILE + threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + t*TILE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && t*TILE + threadIdx.y < K)
            Bs[threadIdx.y][threadIdx.x] = B[(t*TILE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads(); // 等所有线程搬完

        for (int k = 0; k < TILE; ++k)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads(); // 等所有线程算完，才能搬下一块
    }
    if (row < M && col < N)
        C[row * N + col] = sum;
}

int main() {
    const int M = 1024, N = 1024, K = 1024;
    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);

    srand(42); // 固定种子，和v0保持一致
    float *h_A = (float*)malloc(bytes_A);
    float *h_B = (float*)malloc(bytes_B);
    float *h_C = (float*)malloc(bytes_C);
    float *h_C_ref = (float*)malloc(bytes_C);
    for (int i = 0; i < M*K; ++i) h_A[i] = rand() % 3;
    for (int i = 0; i < K*N; ++i) h_B[i] = rand() % 3;

    // CPU参考结果
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0;
            for (int k = 0; k < K; ++k) sum += h_A[i*K+k] * h_B[k*N+j];
            h_C_ref[i*N+j] = sum;
        }
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes_A);
    cudaMalloc(&d_B, bytes_B);
    cudaMalloc(&d_C, bytes_C);
    cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice);

    dim3 block(TILE, TILE);
    dim3 grid((N+TILE-1)/TILE, (M+TILE-1)/TILE);

    // 预热
    gemm_tiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    // 计时 20次取平均
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for(int i=0;i<20;++i) {
        gemm_tiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= 20.0f;
    double gflops = 2.0 * M * N * K / (ms / 1000.0) / 1e9; // 共M*N个元素，每个元素需要K次乘法、K-1次加法，总共2*K-1次浮点运算，约等于2*K次浮点运算
    std::cout << "v1 Tiled GEMM 1024x1024: " << ms << " ms, " << gflops << " GFLOPS" << std::endl;

    cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // 验证结果
    bool ok = true;
    for (int i = 0; i < M*N; ++i) {
        if (fabs(h_C[i] - h_C_ref[i]) > 1e-3) { ok = false; break; }
    }
    if (ok) std::cout << "PASS! v1 Done. Sample C[0]=" << h_C[0] << std::endl;
    else    std::cout << "FAIL! v1 结果不对 Sample C[0]=" << h_C[0] << " Ref=" << h_C_ref[0] << std::endl;

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C); free(h_C_ref);
    return 0;
}