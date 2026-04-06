#include <cstdio>
#include <cuda_runtime.h>
#include <cmath>
#include <stdlib.h>
#include <vector>
#include <string>

#define N (256 * 1024 * 1024)       // 256M floats = 1 GB，远超 L2 (72 MB)
#define THREAD_PER_BLOCK 256
#define IDEAL_BANDWIDTH 1008.0      // GB/s，当前设备理论峰值
#define NUM_RUNS 20                 // 每个版本运行次数，取平均
#define L2_FLUSH_SIZE (96 * 1024 * 1024)  // 96 MB，用于刷 L2 缓存

// ============================================================
// 每个版本在自己的 .cu 文件中实现核函数，并提供一个 launcher：
//   void launch_vN(float *d_in, float *d_out, int n, int block_num);
// 在这里声明即可，添加新版本只需加一行 extern + 一行注册。
// ============================================================
extern void launch_v0(float *d_in, float *d_out, int n, int block_num);
extern void launch_v1(float *d_in, float *d_out, int n, int block_num);
extern void launch_v2(float *d_in, float *d_out, int n, int block_num);
extern void launch_v3(float *d_in, float *d_out, int n, int block_num);
extern void launch_v4(float *d_in, float *d_out, int n, int block_num);
extern void launch_v5(float *d_in, float *d_out, int n, int block_num);
extern void launch_v6(float *d_in, float *d_out, int n, int block_num);
extern void launch_v7(float *d_in, float *d_out, int n, int block_num);
extern void launch_v7b(float *d_in, float *d_out, int n, int block_num);
extern void launch_v8(float *d_in, float *d_out, int n, int block_num);
extern void launch_v9(float *d_in, float *d_out, int n, int block_num);
extern void launch_v10(float *d_in, float *d_out, int n, int block_num);
extern void launch_v11(float *d_in, float *d_out, int n, int block_num);

// ============================================================
// L2 缓存刷新 kernel
// 读写一个大于 L2 的无关数组，将之前的数据挤出缓存
// 这是学术界验证 Cold Cache 性能的标准做法
// ============================================================
__global__ void flush_l2_kernel(float *buf, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] += 1.0f;
}

// ============================================================
// 版本注册表：添加新版本只需在这里加一行
// ============================================================
struct ReduceVersion {
    const char *name;
    void (*launcher)(float*, float*, int, int);
    int elements_per_block;  // 每个 block 处理多少个输入元素
    int output_blocks;       // 0 表示由 n/elements_per_block 决定，>0 表示固定输出 block 数（v8）
};

static ReduceVersion versions[] = {
    { "v0",  launch_v0,  THREAD_PER_BLOCK,     0 },
    { "v1",  launch_v1,  THREAD_PER_BLOCK,     0 },
    { "v2",  launch_v2,  THREAD_PER_BLOCK,     0 },
    { "v3",  launch_v3,  THREAD_PER_BLOCK,     0 },
    { "v4",  launch_v4,  THREAD_PER_BLOCK * 2, 0 },
    { "v5",  launch_v5,  128 * 2,              0 },
    { "v6",  launch_v6,  THREAD_PER_BLOCK * 2, 0 },
    { "v7",  launch_v7,  THREAD_PER_BLOCK * 2, 0 },
    { "v7b", launch_v7b, THREAD_PER_BLOCK * 2, 0 },
    { "v8",  launch_v8,  0,                    2048 },
    { "v9",  launch_v9,  0,                    1824 },  // v9: occupancy API auto gridSize
    { "v10", launch_v10, 0,                    1824 },  // v10: occupancy API auto blockSize + gridSize
    { "v11", launch_v11, 0,                    2048 },  // v11: warp shuffle replace shared memory warp reduce
};
static const int NUM_VERSIONS = sizeof(versions) / sizeof(versions[0]);

// ============================================================
// 工具函数
// ============================================================
bool check(float *out, float *ref, int n) {
    for (int i = 0; i < n; i++) {
        if (fabs(out[i] - ref[i]) > fmax(0.005f, fabs(ref[i]) * 1e-4f)) return false;
    }
    return true;
}

struct BenchmarkResult {
    const char *version;
    float  time_ms;
    double bandwidth_gb_s;
    double efficiency_pct;
    bool   correct;
};

BenchmarkResult run_benchmark(
    const ReduceVersion &ver,
    float *d_in, float *d_out,
    float *h_in, float *h_out,
    float *d_flush, int flush_n,
    int n, int block_num
) {
    // 计算该版本实际输出的 block 数量
    int actual_blocks = ver.output_blocks > 0 ? ver.output_blocks : n / ver.elements_per_block;

    // 预热
    cudaMemcpy(d_in, h_in, (size_t)n * sizeof(float), cudaMemcpyHostToDevice);
    ver.launcher(d_in, d_out, n, block_num);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float total_ms = 0.0f;
    for (int i = 0; i < NUM_RUNS; i++) {
        // v0 会原地修改 d_in，每次都需要重新拷贝
        cudaMemcpy(d_in, h_in, (size_t)n * sizeof(float), cudaMemcpyHostToDevice);

        // 刷 L2 缓存，确保 Cold Cache 公平测量
        flush_l2_kernel<<<(flush_n + 255) / 256, 256>>>(d_flush, flush_n);
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        ver.launcher(d_in, d_out, n, block_num);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        total_ms += ms;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(h_out, d_out, actual_blocks * sizeof(float), cudaMemcpyDeviceToHost);

    float  avg_ms   = total_ms / NUM_RUNS;
    // 读 N 个 float，写 actual_blocks 个 float
    double bytes    = (double)n * sizeof(float) + (double)actual_blocks * sizeof(float);
    double bw       = (bytes / 1e9) / (avg_ms / 1e3);
    double eff      = bw / IDEAL_BANDWIDTH * 100.0;

    bool correct;
    {
        double gpu_sum = 0;
        for (int i = 0; i < actual_blocks; i++) gpu_sum += (double)h_out[i];
        double cpu_sum = 0;
        for (int i = 0; i < n; i++) cpu_sum += (double)h_in[i];
        double tol = fmax(1.0, fabs(cpu_sum) * 1e-3);
        correct = fabs(gpu_sum - cpu_sum) < tol;
    }

    return { ver.name, avg_ms, bw, eff, correct };
}

// ============================================================
// main
// ============================================================
int main() {
    const int block_num = N / THREAD_PER_BLOCK;
    const int max_output_size = block_num;  // 最大输出大小（v0-v3）

    float *h_in  = (float *)malloc((size_t)N * sizeof(float));
    float *h_out = (float *)malloc(max_output_size * sizeof(float));

    float *d_in, *d_out;
    cudaMalloc(&d_in,  (size_t)N * sizeof(float));
    cudaMalloc(&d_out, max_output_size * sizeof(float));

    // 分配 L2 Flush 缓冲区（> L2 大小）
    int flush_n = L2_FLUSH_SIZE / sizeof(float);
    float *d_flush;
    cudaMalloc(&d_flush, L2_FLUSH_SIZE);
    cudaMemset(d_flush, 0, L2_FLUSH_SIZE);

    // 初始化输入
    for (int i = 0; i < N; i++)
        h_in[i] = 2.0f * (float)drand48() - 1.0f;

    // 运行所有版本
    std::vector<BenchmarkResult> results;
    printf("Running benchmarks (%d runs each, with L2 flush)...\n\n", NUM_RUNS);
    for (int i = 0; i < NUM_VERSIONS; i++)
        results.push_back(run_benchmark(versions[i], d_in, d_out, h_in, h_out,
                                        d_flush, flush_n, N, block_num));

    // 打印结果
    printf("%-8s  %10s  %14s  %14s  %s\n",
           "Version", "Time(ms)", "BW(GB/s)", "Efficiency", "Correct");
    printf("%-8s  %10s  %14s  %14s  %s\n",
           "-------", "--------", "--------", "----------", "-------");
    for (const auto &r : results) {
        printf("%-8s  %10.3f  %14.2f  %13.2f%%  %s\n",
               r.version, r.time_ms, r.bandwidth_gb_s, r.efficiency_pct,
               r.correct ? "PASS" : "FAIL");
    }

    printf("\nConfig: N=%dM, block=%d, ideal_bw=%.0f GB/s, L2_flush=%dMB\n",
           N / 1024 / 1024, THREAD_PER_BLOCK, IDEAL_BANDWIDTH,
           L2_FLUSH_SIZE / 1024 / 1024);

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_flush);
    free(h_in);
    free(h_out);
    return 0;
}
