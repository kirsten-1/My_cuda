#include <cstdio>
#include <cuda.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cmath>

#define THREAD_PER_BLOCK 256

__device__ void warpReduce(volatile float* sdata, int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

// v9: 与 v8 相同的 kernel，区别在于 launcher 使用 Occupancy API 自动计算 gridSize
template <unsigned int blockSize>
__global__ void reduce(float *d_in, float *d_out, unsigned int n) {
    __shared__ float sdata[blockSize];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + tid;
    unsigned int gridSize = blockSize * 2 * gridDim.x;

    sdata[tid] = 0;
    while (i < n) {
        sdata[tid] += d_in[i] + d_in[i + blockSize];
        i += gridSize;
    }
    __syncthreads();

    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64)  { sdata[tid] += sdata[tid + 64];  } __syncthreads(); }

    if (tid < 32) warpReduce(sdata, tid);

    if (tid == 0) {
        d_out[blockIdx.x] = sdata[0];
    }
}

bool check(float *out, int num_blocks, float expected) {
    float gpu_sum = 0;
    for (int i = 0; i < num_blocks; i++) gpu_sum += out[i];
    return fabs(gpu_sum - expected) < 1.0f;
}

int main() {
    const int N = 32 * 1024 * 1024;
    const int NUM_RUNS = 20;

    // 使用 Occupancy API 自动计算最优 gridSize
    int minGridSize, blockSize_api;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize_api,
                                       reduce<THREAD_PER_BLOCK>, 0, 0);

    // 对于 grid-stride loop，minGridSize 只是"填满 SM 的最小值"
    // 实际需要更多 block 来充分利用内存流水线，取 minGridSize 的倍数
    // 但不超过 N / (2 * blockSize)（即每个 block 至少处理一批数据）
    int gridSize = minGridSize * 8;
    int maxGrid = N / (2 * THREAD_PER_BLOCK);
    if (gridSize > maxGrid) gridSize = maxGrid;

    printf("=== V9: Occupancy API auto-tuned gridSize ===\n");
    printf("Occupancy API recommends: blockSize=%d, minGridSize=%d\n", blockSize_api, minGridSize);
    printf("Using: blockSize=%d (template), gridSize=%d\n\n", THREAD_PER_BLOCK, gridSize);

    float *h_in = (float *)malloc(N * sizeof(float));
    float *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, gridSize * sizeof(float));

    for (int i = 0; i < N; i++)
        h_in[i] = 2.0f * (float)drand48() - 1.0f;

    float cpu_sum = 0;
    for (int i = 0; i < N; i++) cpu_sum += h_in[i];

    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

    // 预热
    reduce<THREAD_PER_BLOCK><<<gridSize, THREAD_PER_BLOCK>>>(d_in, d_out, N);
    cudaDeviceSynchronize();

    // 计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float total_ms = 0;

    for (int r = 0; r < NUM_RUNS; r++) {
        cudaEventRecord(start);
        reduce<THREAD_PER_BLOCK><<<gridSize, THREAD_PER_BLOCK>>>(d_in, d_out, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        total_ms += ms;
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    float *h_out = (float *)malloc(gridSize * sizeof(float));
    cudaMemcpy(h_out, d_out, gridSize * sizeof(float), cudaMemcpyDeviceToHost);

    float avg_ms = total_ms / NUM_RUNS;
    double bytes = (double)N * sizeof(float) + (double)gridSize * sizeof(float);
    double bw = (bytes / 1e9) / (avg_ms / 1e3);
    bool correct = check(h_out, gridSize, cpu_sum);

    printf("Time: %.3f ms, BW: %.2f GB/s, Efficiency: %.2f%%, Correct: %s\n",
           avg_ms, bw, bw / 1008.0 * 100.0, correct ? "PASS" : "FAIL");

    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
