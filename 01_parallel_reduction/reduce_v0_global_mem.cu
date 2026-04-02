#include <cstdio>
#include <cuda.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cmath>

#define THREAD_PER_BLOCK 256

// --- GPU 核函数：并行归约（global memory 版，交错寻址）---
// 每个 block 负责对自己分到的 blockDim.x 个元素求和，结果写入 d_out[blockIdx.x]
__global__ void reduce(float *d_in, float *d_out) {
    // 让每个 block 的指针直接指向属于自己的那段输入数据
    // block i 处理 d_in[i*blockDim.x ... (i+1)*blockDim.x - 1]
    float *input_dim = d_in + blockIdx.x * blockDim.x;

    // 树形归约：每轮步长 i 翻倍（1, 2, 4, 8, ...）
    // 第 k 轮结束后，间距为 2^k 的位置上保存了 2^k 个元素的局部和
    // 例如 blockDim.x=8，过程如下：
    //   i=1: [0]+=[1], [2]+=[3], [4]+=[5], [6]+=[7]
    //   i=2: [0]+=[2], [4]+=[6]
    //   i=4: [0]+=[4]  → input_dim[0] 即为全部 8 个元素之和
    for (int i = 1; i < blockDim.x; i *= 2) {
        // 只有 threadIdx.x 是 2*i 倍数的线程参与本轮计算
        // 注意：这种取模判断会导致同一个 warp 内线程走不同分支（warp divergence），
        // 是 v0 版本性能较低的主要原因之一
        if (threadIdx.x % (2 * i) == 0) {
            input_dim[threadIdx.x] += input_dim[threadIdx.x + i];
        }
        // 块内同步：确保本轮所有写入完成后，下一轮才能读取
        __syncthreads();
    }

    // 归约完成后 input_dim[0] 存有整个 block 的总和，由 thread 0 写出
    if (threadIdx.x == 0) {
        d_out[blockIdx.x] = input_dim[0];
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
    int block_num = N / THREAD_PER_BLOCK;

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
        for (int j = 0; j < THREAD_PER_BLOCK; j++) {
            cur += input[i * THREAD_PER_BLOCK + j];
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