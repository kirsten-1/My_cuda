#include <cstdio>
#include <cuda.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cmath>

#define THREAD_PER_BLOCK 256

// 专门处理最后一个 Warp 的归约函数
// 使用 volatile 确保对 sdata 的读写直接作用于共享内存，避免编译器优化导致的数据不一致
// 专门处理最后一个 Warp 的归约函数
__device__ void warpReduce(volatile float* sdata, int tid) {
    // 这里的展开不需要判断 blockSize，因为进入这里的线程 tid < 32
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

// 使用模板参数 blockSize
template <unsigned int blockSize>
__global__ void reduce(float *d_in, float *d_out) {
    __shared__ float sdata[blockSize]; // 使用模板参数定义共享内存大小
    unsigned int tid = threadIdx.x;
    
    // 这里的 blockSize 替代了原来的 blockDim.x
    float *input_dim = d_in + blockIdx.x * blockSize * 2;

    sdata[tid] = input_dim[tid] + input_dim[tid + blockSize];
    __syncthreads();

    // 下面所有的 if 条件在编译时就会被评估
    // 如果调用 reduce<256>，那么 >= 512 的那个 if 块在生成的机器码里会被彻底删掉
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64)  { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }

    // 最后一个 Warp 展开
    if (tid < 32) warpReduce(sdata, tid);

    if (tid == 0) {
        d_out[blockIdx.x] = sdata[0];
    }
}

// --- 结果校验函数 ---
bool check(float *out, float *res, int n) {
    for (int i = 0; i < n; i++) {
        if (fabs(out[i] - res[i]) > 0.005) {
            return false;
        }
    }
    return true;
}

int main() {
    // 1. 定义数据规模
    const int N = 32 * 1024 * 1024;
    // 这里的配置必须与模板参数一致
    const int threads = THREAD_PER_BLOCK; 
    int block_num = N / threads / 2;

    // 2. 分配 Host 内存
    float *input = (float *)malloc(N * sizeof(float));
    float *output = (float *)malloc(block_num * sizeof(float));
    float *result = (float *)malloc(block_num * sizeof(float)); // 用于存储 CPU 计算结果

    // 3. 分配 Device 内存
    float *d_input, *d_output;
    cudaMalloc((void **)&d_input, N * sizeof(float));
    cudaMalloc((void **)&d_output, block_num * sizeof(float));

    // 4. 初始化输入数据
    for (int i = 0; i < N; i++) {
        input[i] = 2.0 * (float)drand48() - 1.0;
    }

    // 5. CPU 计算结果（用于对照验证）
    for (int i = 0; i < block_num; i++) {
        float cur = 0;
        for (int j = 0; j < 2 * THREAD_PER_BLOCK; j++) { // 每个 block 处理 2*THREAD_PER_BLOCK 个元素
            cur += input[i * 2 * THREAD_PER_BLOCK + j]; // 注意这里的偏移需要乘以 2*THREAD_PER_BLOCK
        }
        result[i] = cur;
    }

    // 6. 拷贝数据到 GPU
    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

    // 7. 配置 Grid 和 Block 维度
    dim3 Grid(block_num, 1);
    dim3 Block(THREAD_PER_BLOCK, 1);

    // 8. 调用 GPU 核函数
    reduce<THREAD_PER_BLOCK><<<Grid, Block>>>(d_input, d_output);

    // 9. 将结果拷贝回 Host
    cudaMemcpy(output, d_output, block_num * sizeof(float), cudaMemcpyDeviceToHost);

    // 10. 校验结果
    if (check(output, result, block_num)) {
        printf("the ans is right\n");
    } else {
        printf("the ans is wrong\n");
        // 如果错了，打印前 10 个结果看看对比
        for (int i = 0; i < 10; i++) {
            printf("GPU: %f, CPU: %f\n", output[i], result[i]);
        }
        printf("\n");
    }

    // 11. 释放资源
    cudaFree(d_input);
    cudaFree(d_output);
    free(input);
    free(output);
    free(result);

    return 0;
}