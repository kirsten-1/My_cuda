#include <cstdio>
#include <cuda.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cmath>

#define THREAD_PER_BLOCK 256

__device__ float warpReduceShuffle(float val) {
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    return val;
}

// v8: grid-stride loop + 完全展开 + 模板
// 核心变化：block 数量与数据规模解耦，每个 block 通过 while 循环处理多段数据
template <unsigned int blockSize>
__global__ void reduce(float *d_in, float *d_out, unsigned int n) {
    __shared__ float sdata[blockSize];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + tid;
    unsigned int gridSize = blockSize * 2 * gridDim.x; // 一轮 grid 处理的总元素数

    // grid-stride loop：每个线程累加多对元素
    sdata[tid] = 0;
    while (i < n) {
        sdata[tid] += d_in[i] + d_in[i + blockSize];
        i += gridSize;
    }
    __syncthreads();

    // 完全展开的归约（与 v7 相同）
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64)  { sdata[tid] += sdata[tid + 64];  } __syncthreads(); }

    if (tid < 32) {
        float val = sdata[tid] + sdata[tid + 32];  // 合并 stride=32 的数据
        val = warpReduceShuffle(val);
        if (tid == 0) d_out[blockIdx.x] = val;
    }
}

// --- 结果校验函数 ---
bool check(float *out, int num_blocks, float expected) {
    float gpu_sum = 0;
    for (int i = 0; i < num_blocks; i++) {
        gpu_sum += out[i];
    }
    return fabs(gpu_sum - expected) < 1.0f;
}

int main() {
    const int N = 32 * 1024 * 1024;
    const int NUM_RUNS = 20;

    float *input = (float *)malloc(N * sizeof(float));
    float *d_input, *d_output;
    cudaMalloc((void **)&d_input, N * sizeof(float));

    // 初始化输入数据
    for (int i = 0; i < N; i++) {
        input[i] = 2.0 * (float)drand48() - 1.0;
    }

    // CPU 参考结果（全局总和）
    float cpu_sum = 0;
    for (int i = 0; i < N; i++) {
        cpu_sum += input[i];
    }

    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

    // 扫描不同的 grid size（block 数量），寻找最优值
    int grid_sizes[] = {128, 256, 512, 1024, 2048, 4096, 8192, 16384};
    int num_configs = sizeof(grid_sizes) / sizeof(grid_sizes[0]);

    printf("Sweeping grid sizes (block_num) with %d threads/block, N=%dM, %d runs each:\n\n",
           THREAD_PER_BLOCK, N / 1024 / 1024, NUM_RUNS);
    printf("%-12s  %10s  %14s  %8s\n", "GridSize", "Time(ms)", "BW(GB/s)", "Correct");
    printf("%-12s  %10s  %14s  %8s\n", "--------", "--------", "--------", "-------");

    for (int g = 0; g < num_configs; g++) {
        int grid_size = grid_sizes[g];

        float *output = (float *)malloc(grid_size * sizeof(float));
        cudaMalloc((void **)&d_output, grid_size * sizeof(float));

        // 预热
        reduce<THREAD_PER_BLOCK><<<grid_size, THREAD_PER_BLOCK>>>(d_input, d_output, N);
        cudaDeviceSynchronize();

        // 计时
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        float total_ms = 0;

        for (int r = 0; r < NUM_RUNS; r++) {
            cudaEventRecord(start);
            reduce<THREAD_PER_BLOCK><<<grid_size, THREAD_PER_BLOCK>>>(d_input, d_output, N);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float ms = 0;
            cudaEventElapsedTime(&ms, start, stop);
            total_ms += ms;
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        cudaMemcpy(output, d_output, grid_size * sizeof(float), cudaMemcpyDeviceToHost);

        float avg_ms = total_ms / NUM_RUNS;
        double bytes = (double)N * sizeof(float) + (double)grid_size * sizeof(float);
        double bw = (bytes / 1e9) / (avg_ms / 1e3);
        bool correct = check(output, grid_size, cpu_sum);

        printf("%-12d  %10.3f  %14.2f  %8s\n", grid_size, avg_ms, bw, correct ? "PASS" : "FAIL");

        free(output);
        cudaFree(d_output);
    }

    cudaFree(d_input);
    free(input);

    return 0;
}
