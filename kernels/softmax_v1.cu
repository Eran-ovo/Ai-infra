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

    // 1. 每线程先串行扫 N/BLOCK=4 个元素，得线程局部 max
    float v = -1e20f;
    for (int idx = tid; idx < N; idx += BLOCK)   // 线程粗化grid-stride：tid, tid+256, tid+512, tid+768
        v = fmaxf(v, row_x[idx]);
    sdata[tid] = v; //存放每个线程处理的元素，若线程数大于N，则赋值为一个很小的数
    __syncthreads();//此时共享内存sdata存放了每个线程处理的元素的局部max
    //归约求max，得到32个线程（warp）的结果
    for(int s=BLOCK/2; s>=32; s>>=1){ //线程数依次减半 256->128->64->32
        if(tid < s) 
            sdata[tid]=fmaxf(sdata[tid],sdata[tid+s]); 
        __syncthreads(); 
    }
    //在warp内用shuffle，无需__syncthreads（__syncthreads太慢）
    if(tid < 32){
        float val = sdata[tid];
        // 也可以用 __shfl_down_sync 进一步优化，这里先用共享内存直观版
        for(int s=16; s>0; s>>=1) 
            val = fmaxf(val, __shfl_down_sync(0xffffffff, val, s));
        //得到最终的max值，写回共享内存
        if(tid==0) 
            sdata[0]=val;
    }
    __syncthreads();
    float row_max = sdata[0];

    // 2. 归约求sum(exp(x-max))
    float e = 0;
    for (int idx = tid; idx < N; idx += BLOCK) //线程粗化
        e += expf(row_x[idx]-row_max);
    sdata[tid]=e; //存在共享内存
    __syncthreads();
    //归约求sum，得到32个线程（warp）的结果
    for(int s=BLOCK/2; s>=32; s>>=1){ 
        if(tid < s) 
            sdata[tid]+=sdata[tid+s]; 
        __syncthreads(); 
    }
    //在warp内用shuffle，无需__syncthreads（__syncthreads太慢）
    if(tid<32){ 
        float val=sdata[tid]; 
        for(int s=16;s>0;s>>=1) 
            val+=__shfl_down_sync(0xffffffff,val,s); 
        //得到最终的sum值，写回共享内存
        if(tid==0)
            sdata[0]=val; 
    }
    __syncthreads();
    float row_sum = sdata[0];

    // 3. 写回
    for(int idx = tid; idx < N; idx += BLOCK) {
        if(idx < N) {
            row_y[idx] = expf(row_x[idx]-row_max)/row_sum;
        }
    }
}

static void softmax_cpu(const float* x, float* y, int B, int N){
    for(int b=0;b<B;++b){
        const float* row_x=x+b*N;
        float* row_y=y+b*N;
        //求max
        float row_max=-1e20f;
        for(int i=0;i<N;++i) 
            row_max=fmaxf(row_max,row_x[i]);
        //求sum(exp(x-max))
        float row_sum=0;
        for(int i=0;i<N;++i) 
            row_sum+=expf(row_x[i]-row_max);
        //写回
        for(int i=0;i<N;++i) 
            row_y[i]=expf(row_x[i]-row_max)/row_sum;
    }
}

int main(){
    int B=1024,N=1024; // 模拟LLM 1024个token, dim 1024
    size_t bytes=B*N*sizeof(float);
    float *hX=(float*)malloc(bytes),*hY=(float*)malloc(bytes),*hRef=(float*)malloc(bytes);
    for(int i=0;i<B*N;++i) 
        hX[i]= (rand()%100)/10.0f;
    softmax_cpu(hX,hRef,B,N); // CPU参考实现
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
    //对拍
    cudaMemcpy(hY,dY,bytes,cudaMemcpyDeviceToHost);
    float err=0.0f;
    for(int i=0;i<B*N;++i){
        err = fmaxf(err, fabs(hY[i] - hRef[i]));
    }
    std::cout << (err < 1e-3 ? "PASS!" : "FAIL!") << " maxErr=" << err
              << " Sample y[0]=" << hY[0] << " Ref=" << hRef[0] << std::endl;    
    cudaFree(dX);
    cudaFree(dY);
    free(hX);
    free(hY);
    free(hRef);          
    return 0;
}
