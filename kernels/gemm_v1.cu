#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#define TILE 32

// v1: Shared Memory Tiling分块
__global__ void gemm_tiled(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    // ===== Bank Conflict 演示（已注释，仅对比学习用）=====
    // 当前正确写法（无冲突）：
    //   sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    //   - As[ty][k]: 同一warp ty相同,k相同 -> 32线程读同一地址 -> 广播，无冲突
    //   - Bs[k][tx]: 同一warp tx 0-31连续 -> 32线程读连续地址 -> 32个不同Bank，无冲突
    //
    // 反例（若改成列优先访问则会产生32路冲突，不要取消注释）：
    //   // for (int k=0;k<TILE;++k) sum += As[k][threadIdx.y] * Bs[threadIdx.x][k];(As的列乘Bs的行，就会变成求B*A的结果)
    //   - As[ty][k]: 同一warp ty相同,k相同 -> 32线程读同一地址 -> 广播，无冲突
    //   - Bs[tx][k]: 同一warp k相同，tx 0-31连续 -> 32线程读非连续地址 -> 相同Bank，有冲突
    // 经典解法：__shared__ float Bs[TILE][TILE+1]; 
    // =====================================================

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        if (row < M && t*TILE + threadIdx.x < K)//防止行越界
            As[threadIdx.y][threadIdx.x] = A[row * K + t*TILE + threadIdx.x];//一个warp32线程地址连续4B递增 -> 1个128B事务(已合并)
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;
        
        if (col < N && t*TILE + threadIdx.y < K)
            Bs[threadIdx.y][threadIdx.x] = B[(t*TILE + threadIdx.y) * N + col];//一个warp32线程地址连续4B递增 -> 1个128B事务(已合并)
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();// 等待所有线程把As/Bs搬完
        for (int k = 0; k < TILE; ++k)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x]; // 无Bank Conflict
        __syncthreads();
    }
    if (row < M && col < N)
        C[row * N + col] = sum;
}

int main() {
    const int M = 1024, N = 1024, K = 1024;
    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);
    srand(42);
    float *h_A = (float*)malloc(bytes_A);
    float *h_B = (float*)malloc(bytes_B);
    float *h_C = (float*)malloc(bytes_C);
    float *h_C_ref = (float*)malloc(bytes_C);
    for (int i = 0; i < M*K; ++i) h_A[i] = rand() % 3;
    for (int i = 0; i < K*N; ++i) h_B[i] = rand() % 3;
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float sum = 0;
            for (int k = 0; k < K; ++k) sum += h_A[i*K+k] * h_B[k*N+j];
            h_C_ref[i*N+j] = sum;
        }
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes_A); 
    cudaMalloc(&d_B, bytes_B); 
    cudaMalloc(&d_C, bytes_C);
    cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice);
    dim3 block(TILE, TILE);
    dim3 grid((N+TILE-1)/TILE, (M+TILE-1)/TILE);
    gemm_tiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    cudaEvent_t start, stop;
    cudaEventCreate(&start); 
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for(int i=0;i<20;++i) gemm_tiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop); 
    cudaEventSynchronize(stop);
    float ms; cudaEventElapsedTime(&ms, start, stop); 
    ms /= 20.0f;
    double gflops = 2.0 * M * N * K / (ms / 1000.0) / 1e9;
    std::cout << "v1 Tiled GEMM 1024x1024: " << ms << " ms, " << gflops << " GFLOPS" << std::endl;
    cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost); cudaDeviceSynchronize();
    bool ok = true;
    for (int i = 0; i < M*N; ++i) 
        if (fabs(h_C[i] - h_C_ref[i]) > 1e-3){ 
            ok = false; 
            break; 
        }
    if (ok) 
        std::cout << "PASS! v1 Done. Sample C[0]=" << h_C[0] << std::endl;
    else    
        std::cout << "FAIL! v1 Sample C[0]=" << h_C[0] << " Ref=" << h_C_ref[0] << std::endl;
    cudaFree(d_A); 
    cudaFree(d_B); 
    cudaFree(d_C);
    free(h_A); 
    free(h_B); 
    free(h_C); 
    free(h_C_ref);
    return 0;
}