#include <cstdio>
#include <cuda.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cmath>

#define THREAD_PER_BLOCK 256

// 专门处理最后一个 Warp 的归约函数
__device__ void warpReduce(volatile float* sdata, int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

// 使用模板参数 + #pragma unroll 实现完全展开
template <unsigned int blockSize>
__global__ void reduce(float *d_in, float *d_out) {
    __shared__ float sdata[blockSize];
    unsigned int tid = threadIdx.x;

    float *input_dim = d_in + blockIdx.x * blockSize * 2;
    sdata[tid] = input_dim[tid] + input_dim[tid + blockSize];
    __syncthreads();

    // 用 for 循环 + #pragma unroll 替代手动 if 链
    // blockSize 是编译期常量，编译器可以完全展开循环并删除不可达的迭代
    #pragma unroll
    for (unsigned int i = blockSize / 2; i > 32; i /= 2) {
        if (tid < i) {
            sdata[tid] += sdata[tid + i];
        }
        __syncthreads();
    }

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
    const int N = 32 * 1024 * 1024;
    const int threads = THREAD_PER_BLOCK;
    int block_num = N / threads / 2;

    float *input = (float *)malloc(N * sizeof(float));
    float *output = (float *)malloc(block_num * sizeof(float));
    float *result = (float *)malloc(block_num * sizeof(float));

    float *d_input, *d_output;
    cudaMalloc((void **)&d_input, N * sizeof(float));
    cudaMalloc((void **)&d_output, block_num * sizeof(float));

    for (int i = 0; i < N; i++) {
        input[i] = 2.0 * (float)drand48() - 1.0;
    }

    for (int i = 0; i < block_num; i++) {
        float cur = 0;
        for (int j = 0; j < 2 * THREAD_PER_BLOCK; j++) {
            cur += input[i * 2 * THREAD_PER_BLOCK + j];
        }
        result[i] = cur;
    }

    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 Grid(block_num, 1);
    dim3 Block(THREAD_PER_BLOCK, 1);

    reduce<THREAD_PER_BLOCK><<<Grid, Block>>>(d_input, d_output);

    cudaMemcpy(output, d_output, block_num * sizeof(float), cudaMemcpyDeviceToHost);

    if (check(output, result, block_num)) {
        printf("the ans is right\n");
    } else {
        printf("the ans is wrong\n");
        for (int i = 0; i < 10; i++) {
            printf("GPU: %f, CPU: %f\n", output[i], result[i]);
        }
        printf("\n");
    }

    cudaFree(d_input);
    cudaFree(d_output);
    free(input);
    free(output);
    free(result);

    return 0;
}
