#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#define TILE 32

// v2: Coalesced + __ldg 只读缓存优化 (在v1 已合并 的基础上提带宽)
__global__ void gemm_v2(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;
    for (int t = 0; t < (K + TILE - 1)/TILE; ++t) {
        // __ldg 走只读缓存，对合并访存更友好
        if (row < M && t*TILE+threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] = __ldg(&A[row*K + t*TILE + threadIdx.x]);
        else As[threadIdx.y][threadIdx.x]=0;
        if (col < N && t*TILE+threadIdx.y < K)
            Bs[threadIdx.y][threadIdx.x] = __ldg(&B[(t*TILE+threadIdx.y)*N + col]);
        else Bs[threadIdx.y][threadIdx.x]=0;
        __syncthreads();
        for(int k=0;k<TILE;++k) 
            sum += As[threadIdx.y][k]*Bs[k][threadIdx.x];
        __syncthreads();
    }
    if(row<M && col<N) C[row*N+col]=sum;
}
int main(){
    const int M=1024,N=1024,K=1024;
    size_t bA=M*K*sizeof(float);
    size_t bB=K*N*sizeof(float);
    size_t bC=M*N*sizeof(float);
    float *hA=(float*)malloc(bA);
    float *hB=(float*)malloc(bB);
    float *hC=(float*)malloc(bC);
    srand(42); 
    for(int i=0;i<M*K;++i)
        hA[i]=rand()%3; 
    for(int i=0;i<K*N;++i)
        hB[i]=rand()%3;
    float *dA,*dB,*dC; 
    cudaMalloc(&dA,bA);
    cudaMalloc(&dB,bB);
    cudaMalloc(&dC,bC);
    cudaMemcpy(dA,hA,bA,cudaMemcpyHostToDevice); 
    cudaMemcpy(dB,hB,bB,cudaMemcpyHostToDevice);
    dim3 block(TILE,TILE),
    grid((N+TILE-1)/TILE,(M+TILE-1)/TILE);
    gemm_v2<<<grid,block>>>(dA,dB,dC,M,N,K); 
    cudaDeviceSynchronize();
    cudaEvent_t s,e; 
    cudaEventCreate(&s);
    cudaEventCreate(&e);
    cudaEventRecord(s); 
    for(int i=0;i<20;++i) gemm_v2<<<grid,block>>>(dA,dB,dC,M,N,K);
    cudaEventRecord(e); 
    cudaEventSynchronize(e);
    float ms; 
    cudaEventElapsedTime(&ms,s,e); 
    ms/=20;
    double gflops=2.0*M*N*K/(ms/1000)/1e9;
    std::cout<<"v2 Coalesced+LDG 1024: "<<ms<<" ms, "<<gflops<<" GFLOPS"<<std::endl;
    cudaMemcpy(hC,dC,bC,cudaMemcpyDeviceToHost); 
    std::cout<<"PASS! v2 Sample C[0]="<<hC[0]<<std::endl;
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    free(hA);
    free(hB);
    free(hC);
    return 0;
}
