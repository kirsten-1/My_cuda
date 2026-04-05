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

// 用一个大 kernel 刷 L2 缓存
__global__ void flush_l2(float *buf, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] += 1.0f;
}

int main() {
    const int NUM_RUNS = 20;
    const double IDEAL_BW = 1008.0; // GB/s

    // ============================================================
    // 测试 1：不同数据规模（验证 L2 缓存影响）
    // RTX 4090 L2 = 72 MB
    // 32M floats = 128 MB (> L2), 64M = 256 MB, 128M = 512 MB, 256M = 1 GB
    // ============================================================
    int data_sizes[] = {
        32  * 1024 * 1024,
        64  * 1024 * 1024,
        128 * 1024 * 1024,
        256 * 1024 * 1024,
    };
    int num_data_sizes = sizeof(data_sizes) / sizeof(data_sizes[0]);
    int grid_size = 2048;

    printf("=== Test 1: Data size sweep (grid_size=%d, threads=%d) ===\n", grid_size, THREAD_PER_BLOCK);
    printf("RTX 4090 L2 Cache = 72 MB\n\n");
    printf("%-10s  %10s  %10s  %14s  %10s  %8s\n",
           "N", "N(MB)", "Time(ms)", "BW(GB/s)", "Efficiency", "Correct");
    printf("%-10s  %10s  %10s  %14s  %10s  %8s\n",
           "--------", "--------", "--------", "--------", "----------", "-------");

    for (int d = 0; d < num_data_sizes; d++) {
        int N = data_sizes[d];
        float data_mb = (float)N * sizeof(float) / 1024.0f / 1024.0f;

        float *h_in = (float *)malloc((size_t)N * sizeof(float));
        float *d_in, *d_out;
        cudaMalloc(&d_in, (size_t)N * sizeof(float));
        cudaMalloc(&d_out, grid_size * sizeof(float));

        for (int i = 0; i < N; i++)
            h_in[i] = 2.0f * (float)drand48() - 1.0f;

        // CPU 参考
        double cpu_sum = 0;
        for (int i = 0; i < N; i++) cpu_sum += h_in[i];

        cudaMemcpy(d_in, h_in, (size_t)N * sizeof(float), cudaMemcpyHostToDevice);

        // 预热
        reduce<THREAD_PER_BLOCK><<<grid_size, THREAD_PER_BLOCK>>>(d_in, d_out, N);
        cudaDeviceSynchronize();

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        float total_ms = 0;

        for (int r = 0; r < NUM_RUNS; r++) {
            cudaEventRecord(start);
            reduce<THREAD_PER_BLOCK><<<grid_size, THREAD_PER_BLOCK>>>(d_in, d_out, N);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float ms = 0;
            cudaEventElapsedTime(&ms, start, stop);
            total_ms += ms;
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        float *h_out = (float *)malloc(grid_size * sizeof(float));
        cudaMemcpy(h_out, d_out, grid_size * sizeof(float), cudaMemcpyDeviceToHost);
        double gpu_sum = 0;
        for (int i = 0; i < grid_size; i++) gpu_sum += h_out[i];

        float avg_ms = total_ms / NUM_RUNS;
        double bytes = (double)N * sizeof(float) + (double)grid_size * sizeof(float);
        double bw = (bytes / 1e9) / (avg_ms / 1e3);
        double eff = bw / IDEAL_BW * 100.0;
        bool correct = fabs(gpu_sum - cpu_sum) < (N / 1024.0 / 1024.0); // 容差随规模增大

        printf("%-10d  %10.0f  %10.3f  %14.2f  %9.2f%%  %8s\n",
               N, data_mb, avg_ms, bw, eff, correct ? "PASS" : "FAIL");

        free(h_in);
        free(h_out);
        cudaFree(d_in);
        cudaFree(d_out);
    }

    // ============================================================
    // 测试 2：L2 缓存冷启动 vs 热启动对比（N=32M）
    // ============================================================
    printf("\n=== Test 2: Cold vs Hot L2 (N=32M, grid_size=%d) ===\n\n", grid_size);

    int N = 32 * 1024 * 1024;
    float *h_in = (float *)malloc((size_t)N * sizeof(float));
    for (int i = 0; i < N; i++) h_in[i] = 2.0f * (float)drand48() - 1.0f;

    float *d_in, *d_out;
    cudaMalloc(&d_in, (size_t)N * sizeof(float));
    cudaMalloc(&d_out, grid_size * sizeof(float));
    cudaMemcpy(d_in, h_in, (size_t)N * sizeof(float), cudaMemcpyHostToDevice);

    // 分配一个大 buffer 用来刷 L2（> 72 MB）
    int flush_n = 96 * 1024 * 1024 / sizeof(float); // 96 MB
    float *d_flush;
    cudaMalloc(&d_flush, (size_t)flush_n * sizeof(float));
    cudaMemset(d_flush, 0, (size_t)flush_n * sizeof(float));

    printf("%-12s  %10s  %14s  %10s\n", "Mode", "Time(ms)", "BW(GB/s)", "Efficiency");
    printf("%-12s  %10s  %14s  %10s\n", "--------", "--------", "--------", "----------");

    // Hot: 连续跑，L2 有缓存
    {
        // 先预热
        reduce<THREAD_PER_BLOCK><<<grid_size, THREAD_PER_BLOCK>>>(d_in, d_out, N);
        cudaDeviceSynchronize();

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        float total_ms = 0;

        for (int r = 0; r < NUM_RUNS; r++) {
            cudaEventRecord(start);
            reduce<THREAD_PER_BLOCK><<<grid_size, THREAD_PER_BLOCK>>>(d_in, d_out, N);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float ms = 0;
            cudaEventElapsedTime(&ms, start, stop);
            total_ms += ms;
        }
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        float avg_ms = total_ms / NUM_RUNS;
        double bytes = (double)N * sizeof(float) + (double)grid_size * sizeof(float);
        double bw = (bytes / 1e9) / (avg_ms / 1e3);
        printf("%-12s  %10.3f  %14.2f  %9.2f%%\n", "Hot L2", avg_ms, bw, bw / IDEAL_BW * 100.0);
    }

    // Cold: 每次运行前刷 L2
    {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        float total_ms = 0;

        for (int r = 0; r < NUM_RUNS; r++) {
            // 刷 L2：用大 buffer 写入，把之前的数据挤出缓存
            flush_l2<<<(flush_n + 255) / 256, 256>>>(d_flush, flush_n);
            cudaDeviceSynchronize();

            cudaEventRecord(start);
            reduce<THREAD_PER_BLOCK><<<grid_size, THREAD_PER_BLOCK>>>(d_in, d_out, N);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float ms = 0;
            cudaEventElapsedTime(&ms, start, stop);
            total_ms += ms;
        }
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        float avg_ms = total_ms / NUM_RUNS;
        double bytes = (double)N * sizeof(float) + (double)grid_size * sizeof(float);
        double bw = (bytes / 1e9) / (avg_ms / 1e3);
        printf("%-12s  %10.3f  %14.2f  %9.2f%%\n", "Cold L2", avg_ms, bw, bw / IDEAL_BW * 100.0);
    }

    cudaFree(d_flush);

    // ============================================================
    // 测试 3：不同 grid size 扫描（N=32M）
    // ============================================================
    printf("\n=== Test 3: Grid size sweep (N=32M, threads=%d) ===\n\n", THREAD_PER_BLOCK);
    printf("%-12s  %10s  %14s  %10s  %8s\n",
           "GridSize", "Time(ms)", "BW(GB/s)", "Efficiency", "Correct");
    printf("%-12s  %10s  %14s  %10s  %8s\n",
           "--------", "--------", "--------", "----------", "-------");

    double cpu_sum = 0;
    for (int i = 0; i < N; i++) cpu_sum += h_in[i];

    int grid_sizes[] = {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536};
    int num_gs = sizeof(grid_sizes) / sizeof(grid_sizes[0]);

    for (int g = 0; g < num_gs; g++) {
        int gs = grid_sizes[g];
        float *d_out2;
        cudaMalloc(&d_out2, gs * sizeof(float));

        // 预热
        reduce<THREAD_PER_BLOCK><<<gs, THREAD_PER_BLOCK>>>(d_in, d_out2, N);
        cudaDeviceSynchronize();

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        float total_ms = 0;

        for (int r = 0; r < NUM_RUNS; r++) {
            cudaEventRecord(start);
            reduce<THREAD_PER_BLOCK><<<gs, THREAD_PER_BLOCK>>>(d_in, d_out2, N);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float ms = 0;
            cudaEventElapsedTime(&ms, start, stop);
            total_ms += ms;
        }
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        float *h_out2 = (float *)malloc(gs * sizeof(float));
        cudaMemcpy(h_out2, d_out2, gs * sizeof(float), cudaMemcpyDeviceToHost);
        double gpu_sum = 0;
        for (int i = 0; i < gs; i++) gpu_sum += h_out2[i];

        float avg_ms = total_ms / NUM_RUNS;
        double bytes = (double)N * sizeof(float) + (double)gs * sizeof(float);
        double bw = (bytes / 1e9) / (avg_ms / 1e3);
        bool correct = fabs(gpu_sum - cpu_sum) < 1.0f;

        printf("%-12d  %10.3f  %14.2f  %9.2f%%  %8s\n",
               gs, avg_ms, bw, bw / IDEAL_BW * 100.0, correct ? "PASS" : "FAIL");

        free(h_out2);
        cudaFree(d_out2);
    }

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);

    return 0;
}
