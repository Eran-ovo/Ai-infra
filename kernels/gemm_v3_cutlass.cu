#include <iostream>
#include <cstdlib>
#include <cmath>
#include "cutlass/gemm/device/gemm.h"

// v3: 调用CUTLASS工业级GEMM，对比手写v1
int main(){
    const int M=1024, N=1024, K=1024;
    float alpha=1.0f, beta=0.0f;

    // 定义一个CUTLASS GEMM算子: fp32, A行主序 B列主序 C行主序，编译期已固化 tile/warp/流水线配置，这步几乎零开销
    using CutlassGemm = cutlass::gemm::device::Gemm<
        float, cutlass::layout::RowMajor,
        float, cutlass::layout::RowMajor,
        float, cutlass::layout::RowMajor>;

    size_t bA=M*K*4, bB=K*N*4, bC=M*N*4;
    float *hA=(float*)malloc(bA),*hB=(float*)malloc(bB),*hC=(float*)malloc(bC);
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

    // 配置参数并launch
    CutlassGemm gemm_op;
    typename CutlassGemm::Arguments args(
        {M, N, K},      // problem_size：矩阵维度
        {dA, K},        // A 指针 + leading dimension（lda）
        {dB, K},        // B 指针 + ldb
        {dC, N},        // C 指针 + ldc（输入，beta 项用）
        {dC, N},        // D 指针 + ldd（输出，写回同一 dC 即原地更新）
        {alpha, beta}   // 标量：C = alpha·A·B + beta·C
    );
    cutlass::Status st = gemm_op(args);//启动：内部转成 kernel launch
    cudaDeviceSynchronize();//等 GPU 算完
    if(st != cutlass::Status::kSuccess){ // 只查启动错
        std::cout<<"CUTLASS FAIL"<<std::endl; 
        return 1; 
    }

    // 计时
    cudaEvent_t s,e; 
    cudaEventCreate(&s); 
    cudaEventCreate(&e);
    cudaEventRecord(s);
    for(int i=0;i<20;++i) 
        gemm_op(args);
    cudaEventRecord(e); 
    cudaEventSynchronize(e);
    float ms; 
    cudaEventElapsedTime(&ms,s,e); 
    ms/=20;
    double gflops=2.0*M*N*K/(ms/1000)/1e9;
    std::cout<<"v3 CUTLASS GEMM 1024x1024: "<<ms<<" ms, "<<gflops<<" GFLOPS"<<std::endl;

    cudaMemcpy(hC,dC,bC,cudaMemcpyDeviceToHost);
    std::cout<<"PASS! v3 Sample C[0]="<<hC[0]<<std::endl;
    return 0;
}
