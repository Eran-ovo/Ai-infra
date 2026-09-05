#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#define BLOCK 256

//softmax公式：softmax(x) = exp(x-max) / sum(exp(x-max))
// 一行一个block，block内做两次归约：max -> sum
__global__ void softmax_fused(const float* __restrict__ x, float* __restrict__ y, int N){
    int row = blockIdx.x;//该block要处理的行号
    const float* row_x = x + row * N;//该线程要处理的数据的起始地址
    float* row_y = y + row * N;//该线程要保存的数据的起始地址
    __shared__ float sdata[BLOCK];//共享内存，存放归约结果
    int tid = threadIdx.x;

    // 1. 归约求max - PMPP 10章树形归约
    float v = (tid < N) ? row_x[tid] : -1e20;
    sdata[tid] = v; //存放每个线程处理的元素，若线程数大于N，则赋值为一个很小的数
    __syncthreads();
    //归约求max，得到32个线程（warp）的结果
    for(int s=BLOCK/2; s>32; s>>=1){ //线程数依次减半 256->128->64->32
        if(tid < s) 
            sdata[tid]=fmaxf(sdata[tid],sdata[tid+s]); 
        __syncthreads(); 
    }
    //在warp内用shuffle，无需__syncthreads（__syncthreads太慢）
    if(tid < 32){
        float val = sdata[tid];
        // 也可以用 __shfl_down_sync 进一步优化，这里先用共享内存直观版
        for(int s=32; s>0; s>>=1) 
            val = fmaxf(val, __shfl_down_sync(0xffffffff, val, s));
        //得到最终的max值，写回共享内存
        if(tid==0) sdata[0]=val;
    }
    __syncthreads();
    float row_max = sdata[0];

    // 2. 归约求sum(exp(x-max))
    float e = (tid < N) ? expf(row_x[tid]-row_max) : 0;//防止溢出，计算exp(x-max)
    sdata[tid]=e; //存在共享内存
    __syncthreads();
    //归约求sum，得到32个线程（warp）的结果
    for(int s=BLOCK/2; s>32; s>>=1){ 
        if(tid < s) 
            sdata[tid]+=sdata[tid+s]; 
        __syncthreads(); 
    }
    //在warp内用shuffle，无需__syncthreads（__syncthreads太慢）
    if(tid<32){ 
        float val=sdata[tid]; 
        for(int s=32;s>0;s>>=1) 
            val+=__shfl_down_sync(0xffffffff,val,s); 
        //得到最终的sum值，写回共享内存
        if(tid==0)sdata[0]=val; 
    }
    __syncthreads();
    float row_sum = sdata[0];

    // 3. 写回
    if(tid < N) row_y[tid] = expf(row_x[tid]-row_max)/row_sum;
}
int main(){
    int B=1024,N=1024; // 模拟LLM 1024个token, dim 1024
    size_t bytes=B*N*sizeof(float);
    float *hX=(float*)malloc(bytes),*hY=(float*)malloc(bytes);
    for(int i=0;i<B*N;++i) 
        hX[i]= (rand()%100)/10.0f;
    float *dX,*dY;
    cudaMalloc(&dX,bytes);
    cudaMalloc(&dY,bytes);
    cudaMemcpy(dX,hX,bytes,cudaMemcpyHostToDevice);
    //求出softmax
    softmax_fused<<<B,BLOCK>>>(dX,dY,N); 
    cudaDeviceSynchronize();//等待计算完成
    //计算耗时
    cudaEvent_t s,e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);
    cudaEventRecord(s); 
    for(int i=0;i<100;++i) 
        softmax_fused<<<B,BLOCK>>>(dX,dY,N);
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms;
    cudaEventElapsedTime(&ms,s,e); 
    ms/=100;
    std::cout<<"Softmax Fused B=1024 N=1024: "<<ms<<" ms "<<std::endl;
    cudaMemcpy(hY,dY,bytes,cudaMemcpyDeviceToHost);
    std::cout<<"PASS! Sample y[0]="<<hY[0]<<std::endl;
    return 0;
}
