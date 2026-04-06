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

// v12: 寄存器累加 + warp 分层归约
// 核心变化：
//   v11: grid-stride 累加到 shared memory → block 级 shared memory 树形归约 → warp shuffle
//   v12: grid-stride 累加到寄存器 → 每个 warp 内 shuffle 归约 → warpLevelSums[] → 一个 warp 最终 shuffle
// 优势：shared memory 从 blockSize 个 float 降到 warpCount 个，__syncthreads() 从 3~4 次降到 1 次
template <unsigned int blockSize>
__global__ void reduce(float *d_in, float *d_out, unsigned int n) {
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + tid;
    unsigned int gridSize = blockSize * 2 * gridDim.x;

    // ① grid-stride loop：累加到寄存器（而非 shared memory）
    float sum = 0.0f;
    while (i < n) {
        sum += d_in[i] + d_in[i + blockSize];
        i += gridSize;
    }

    // ② Warp 内 shuffle 归约：每个 warp 的 32 个线程归约为 1 个部分和
    sum = warpReduceShuffle(sum);

    // ③ 每个 warp 的 lane 0 把部分和写入 shared memory
    const unsigned int warpCount = blockSize / 32;
    __shared__ float warpLevelSums[warpCount];
    const int laneId = tid % 32;
    const int warpId = tid / 32;
    if (laneId == 0) {
        warpLevelSums[warpId] = sum;
    }
    __syncthreads();  // 整个 kernel 只需要这一次 __syncthreads()

    // ④ 用第一个 warp 对 warpLevelSums[] 做最终归约
    if (warpId == 0) {
        sum = (laneId < warpCount) ? warpLevelSums[laneId] : 0.0f;
        sum = warpReduceShuffle(sum);
        if (laneId == 0) {
            d_out[blockIdx.x] = sum;
        }
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
