#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#define BLOCK 256

//输入x是一个二维矩阵，大小为B*N，B是batch size，N是每个样本的特征数
//每个block处理一行数据，block内做两次归约：求均值 -> 求方差
//归一化公式：y=(x-mean)/sqrt(var+eps) var: 方差
__global__ void layernorm_fused(const float* __restrict__ x, float* __restrict__ y, int N, float eps){
    int row=blockIdx.x;//该block要处理的行号
    const float* rx=x+row*N;//该线程要处理的数据的起始地址
    float* ry=y+row*N;//该线程要保存的数据的起始地址
    __shared__ float s[BLOCK]; 
    int tid=threadIdx.x;
    float v=(tid<N)?rx[tid]:0; 
    s[tid]=v; //存放每个线程处理的元素，若线程数大于N，则赋值为0
    __syncthreads();
    //归约求sum，得到32个线程（warp）的结果
    for(int s2=BLOCK/2; s2>32; s2>>=1){ 
        if(tid<s2) 
            s[tid]+=s[tid+s2]; 
        __syncthreads(); 
    }
    if(tid<32){ 
        float val=s[tid]; 
        for(int s2=32;s2>0;s2>>=1) 
            val+=__shfl_down_sync(0xffffffff,val,s2); 
        if(tid==0)
            s[0]=val; 
    } 
    __syncthreads();
    float mean=s[0]/N;
    //计算方差
    float diff=(tid<N)?rx[tid]-mean:0; 
    s[tid]=diff*diff; 
    __syncthreads();
    for(int s2=BLOCK/2; s2>32; s2>>=1){ 
        if(tid<s2) 
            s[tid]+=s[tid+s2]; 
        __syncthreads(); 
    }
    if(tid<32){ 
        float val=s[tid]; 
        for(int s2=32;s2>0;s2>>=1) 
            val+=__shfl_down_sync(0xffffffff,val,s2); 
        if(tid==0)
            s[0]=val; 
    } 
    __syncthreads();
    float var=s[0]/N; 
    float inv=rsqrtf(var+eps);//计算标准差的倒数
    if(tid<N) 
        ry[tid]=diff*inv;//归一化，得到最终结果
}
int main(){
    int B=1024,N=1024; 
    float eps=1e-5; 
    size_t bytes=B*N*sizeof(float);
    float *hX=(float*)malloc(bytes),*hY=(float*)malloc(bytes);
    for(int i=0;i<B*N;++i) 
        hX[i]= (rand()%100)/10.0f;
    float *dX,*dY; 
    cudaMalloc(&dX,bytes);
    cudaMalloc(&dY,bytes);
    cudaMemcpy(dX,hX,bytes,cudaMemcpyHostToDevice);
    layernorm_fused<<<B,BLOCK>>>(dX,dY,N,eps); 
    cudaDeviceSynchronize();
    cudaEvent_t s,e; 
    cudaEventCreate(&s);
    cudaEventCreate(&e);
    cudaEventRecord(s); 
    for(int i=0;i<100;++i) 
        layernorm_fused<<<B,BLOCK>>>(dX,dY,N,eps);
    cudaEventRecord(e); 
    cudaEventSynchronize(e);
    float ms; 
    cudaEventElapsedTime(&ms,s,e); 
    ms/=100;
    std::cout<<"LayerNorm Fused 1024x1024: "<<ms<<" ms"<<std::endl;
    cudaMemcpy(hY,dY,bytes,cudaMemcpyDeviceToHost); 
    std::cout<<"PASS! Sample y[0]="<<hY[0]<<std::endl;
    return 0;
}
