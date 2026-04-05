#include <cstdio>
#include <cuda.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ void warpReduce(volatile float* sdata, int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

// v10: 模板 kernel，通过 Occupancy API 自动选择最优 blockSize 和 gridSize
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

// ============================================================
// Occupancy 查询工具：获取每个 SM 能常驻的最大 block 数
// 对每个模板实例化调用 cudaOccupancyMaxActiveBlocksPerMultiprocessor
// 计算 totalActiveThreads = activeBlocksPerSM * blockSize * numSMs
// 选择 totalActiveThreads 最大的 blockSize
// ============================================================
struct BlockConfig {
    int blockSize;
    int activeBlocksPerSM;
    int totalActiveThreads;
    int minGridSize;
};

// 辅助：对指定模板实例化查询 occupancy
template <unsigned int BS>
BlockConfig queryOccupancy(int numSMs) {
    int activeBlocks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&activeBlocks, reduce<BS>, BS, 0);
    int minGridSize, blockSize_api;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize_api, reduce<BS>, 0, 0);
    return { (int)BS, activeBlocks, (int)(activeBlocks * BS * numSMs), minGridSize };
}

bool check(float *out, int num_blocks, double expected) {
    double gpu_sum = 0;
    for (int i = 0; i < num_blocks; i++) gpu_sum += (double)out[i];
    double diff = fabs(gpu_sum - expected);
    // GPU 用 float 累加，误差与元素数量成正比
    bool ok = diff < fmax(1.0, fabs(expected) * 1e-2);
    if (!ok) fprintf(stderr, "  check: gpu=%.4f cpu=%.4f diff=%.4f\n", gpu_sum, expected, diff);
    return ok;
}

// 运行指定 blockSize 的 benchmark
template <unsigned int BS>
void runBenchmark(float *d_in, float *d_out, float *h_out, double cpu_sum,
                  int N, int gridSize, int NUM_RUNS) {
    // 预热
    cudaMemset(d_out, 0, gridSize * sizeof(float));
    reduce<BS><<<gridSize, BS>>>(d_in, d_out, N);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float total_ms = 0;

    for (int r = 0; r < NUM_RUNS; r++) {
        cudaEventRecord(start);
        reduce<BS><<<gridSize, BS>>>(d_in, d_out, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        total_ms += ms;
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(h_out, d_out, gridSize * sizeof(float), cudaMemcpyDeviceToHost);
    float avg_ms = total_ms / NUM_RUNS;
    double bytes = (double)N * sizeof(float) + (double)gridSize * sizeof(float);
    double bw = (bytes / 1e9) / (avg_ms / 1e3);
    bool correct = check(h_out, gridSize, cpu_sum);

    printf("  blockSize=%-4d  gridSize=%-6d  Time=%.3f ms  BW=%.2f GB/s  Eff=%.2f%%  %s\n",
           BS, gridSize, avg_ms, bw, bw / 1008.0 * 100.0, correct ? "PASS" : "FAIL");
}

int main() {
    const int N = 32 * 1024 * 1024;
    const int NUM_RUNS = 20;

    // 获取 GPU SM 数量
    int deviceId;
    cudaGetDevice(&deviceId);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);
    int numSMs = prop.multiProcessorCount;

    printf("=== V10: Auto-tune blockSize + gridSize via Occupancy API ===\n");
    printf("GPU: %s, SMs: %d\n\n", prop.name, numSMs);

    // 对所有候选 blockSize 查询 occupancy
    BlockConfig configs[] = {
        queryOccupancy<32>(numSMs),
        queryOccupancy<64>(numSMs),
        queryOccupancy<128>(numSMs),
        queryOccupancy<256>(numSMs),
        queryOccupancy<512>(numSMs),
        queryOccupancy<1024>(numSMs),
    };
    int numConfigs = sizeof(configs) / sizeof(configs[0]);

    printf("--- Occupancy Analysis ---\n");
    printf("%-10s  %18s  %20s  %12s\n",
           "blockSize", "activeBlocks/SM", "totalActiveThreads", "minGridSize");
    printf("%-10s  %18s  %20s  %12s\n",
           "--------", "---------------", "------------------", "-----------");

    int bestIdx = 0;
    for (int i = 0; i < numConfigs; i++) {
        printf("%-10d  %18d  %20d  %12d\n",
               configs[i].blockSize, configs[i].activeBlocksPerSM,
               configs[i].totalActiveThreads, configs[i].minGridSize);
        if (configs[i].totalActiveThreads > configs[bestIdx].totalActiveThreads)
            bestIdx = i;
    }
    printf("\nBest: blockSize=%d (totalActiveThreads=%d)\n\n",
           configs[bestIdx].blockSize, configs[bestIdx].totalActiveThreads);

    // 准备数据
    float *h_in = (float *)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++)
        h_in[i] = 2.0f * (float)drand48() - 1.0f;

    double cpu_sum = 0;
    for (int i = 0; i < N; i++) cpu_sum += (double)h_in[i];

    float *d_in;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

    // 为最大可能的 gridSize 分配输出（使用 *8 后的实际值）
    int maxGridSize = 0;
    for (int i = 0; i < numConfigs; i++) {
        int gs = configs[i].minGridSize * 8;
        int maxGs = N / (2 * configs[i].blockSize);
        if (gs > maxGs) gs = maxGs;
        configs[i].minGridSize = gs;
        if (gs > maxGridSize) maxGridSize = gs;
    }

    float *d_out;
    cudaMalloc(&d_out, maxGridSize * sizeof(float));
    float *h_out = (float *)malloc(maxGridSize * sizeof(float));

    // 对所有 blockSize 实际跑 benchmark 比较性能
    // gridSize 使用 minGridSize * 8，充分利用内存流水线
    // 注意：blockSize < 64 会导致 warpReduce 中 sdata[tid+32] 越界，跳过
    printf("--- Performance Comparison ---\n");
    runBenchmark<64>(d_in, d_out, h_out, cpu_sum, N, configs[1].minGridSize, NUM_RUNS);
    runBenchmark<128>(d_in, d_out, h_out, cpu_sum, N, configs[2].minGridSize, NUM_RUNS);
    runBenchmark<256>(d_in, d_out, h_out, cpu_sum, N, configs[3].minGridSize, NUM_RUNS);
    runBenchmark<512>(d_in, d_out, h_out, cpu_sum, N, configs[4].minGridSize, NUM_RUNS);

    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
