#include <cstdio>
#include <cuda.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cmath>

#define THREAD_PER_BLOCK 256

// 专门处理最后一个 Warp 的归约函数
// 使用 volatile 确保对 sdata 的读写直接作用于共享内存，避免编译器优化导致的数据不一致
__device__ void warpReduce(volatile float* sdata, int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}
// --- GPU 核函数：并行归约（v4 加载时归约，提升带宽利用率）---
// 每个 block 处理 2*blockDim.x 个元素，在加载到 shared memory 时完成第一轮归约
__global__ void reduce(float *d_in, float *d_out) {
    // 每个 block 处理的数据范围：d_in[blockIdx.x * 2*blockDim.x ... blockIdx.x * 2*blockDim.x + 2*blockDim.x - 1]
    // 相比 v3，block 数量减半，每个 block 处理的数据量翻倍（从 256 个元素增加到 512 个元素）
    __shared__ float sdata[THREAD_PER_BLOCK]; // 共享内存大小不变，仍为 256 个元素
    float *input_dim = d_in + blockIdx.x * blockDim.x * 2; // 每个 block 处理 2*blockDim.x 个元素，偏移量需乘以 2
    // 【核心优化】加载时归约：每个线程加载 2 个元素并立即求和，存入 shared memory
    // 这样 shared memory 初始化完成时，已经完成了第一轮归约（512→256）
    // 相比 v3 的优势：
    //   1. 所有 256 个线程都参与计算，无线程闲置（v3 第一轮只有 128 个线程工作）
    //   2. 充分利用内存带宽，每个线程加载 2 个数据而非 1 个
    //   3. 减少 block 数量，降低 block 启动和同步的固定开销
    sdata[threadIdx.x] = input_dim[threadIdx.x] + input_dim[threadIdx.x + blockDim.x];
    __syncthreads(); // 确保所有线程完成加载和第一轮归约后再进行后续归约
    // 树形归约：从 i=blockDim.x/2 开始（相比 v3 无变化，但此时 shared memory 已存储 256 个预归约结果）
    // 第 k 轮结束后，只有前 blockDim.x / 2^k 个线程的数据被更新
    // 例如 blockDim.x=8，过程如下：
    //   i=4: [0]+=[4], [1]+=[5], [2]+=[6], [3]+=[7]  (前 4 个线程参与)
    //   i=2: [0]+=[2], [1]+=[3]                       (前 2 个线程参与)
    //   i=1: [0]+=[1]                                 (前 1 个线程参与)
    for (int i = blockDim.x / 2; i > 32; i /= 2) {
        // 归约逻辑与 v3 相同：threadIdx.x < i 判断，活跃线程从 T0 开始连续分布
        // 当 i >= 32 时无 warp divergence，当 i < 32 时仅 Warp 0 内部有 divergence
        if (threadIdx.x < i) {
            sdata[threadIdx.x] += sdata[threadIdx.x + i];
        }
        // 块内同步：确保本轮所有写入完成后，下一轮才能读取
        __syncthreads();
    }
    if (threadIdx.x < 32) { // 最后一个 warp 内的线程继续归约
        warpReduce(sdata, threadIdx.x);
    }

    // 归约完成后 sdata[0] 存有整个 block 的总和（2*blockDim.x 个元素的和），由 thread 0 写出
    if (threadIdx.x == 0) {
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
    int block_num = N / THREAD_PER_BLOCK / 2; // 每个线程处理 2 个元素, 因此 block_num 需要除以 2

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
    reduce<<<Grid, Block>>>(d_input, d_output);

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