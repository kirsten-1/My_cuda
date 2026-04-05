#include <cstdio>
#include <cuda_runtime.h>
#include <cmath>
#include <stdlib.h>
#include <vector>
#include <string>

#define N (32 * 1024 * 1024)
#define THREAD_PER_BLOCK 256
#define IDEAL_BANDWIDTH 1008.0  // GB/s，当前设备理论峰值
#define NUM_RUNS 20             // 每个版本运行次数，取平均

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

// ============================================================
// 版本注册表：添加新版本只需在这里加一行
// ============================================================
struct ReduceVersion {
    const char *name;
    void (*launcher)(float*, float*, int, int);
    int elements_per_block;  // 每个 block 处理多少个输入元素
};

static ReduceVersion versions[] = {
    { "v0", launch_v0, THREAD_PER_BLOCK },
    { "v1", launch_v1, THREAD_PER_BLOCK },
    { "v2", launch_v2, THREAD_PER_BLOCK },
    { "v3", launch_v3, THREAD_PER_BLOCK },
    { "v4", launch_v4, THREAD_PER_BLOCK * 2 },  // v4 每个 block 处理 2 倍数据
    { "v5", launch_v5, 128 * 2 },                  // v5: 128 threads, 每个 block 处理 256 个元素
    { "v6", launch_v6, THREAD_PER_BLOCK * 2 },    // v6: warp unrolling, 每个 block 处理 512 个元素
};
static const int NUM_VERSIONS = sizeof(versions) / sizeof(versions[0]);

// ============================================================
// 工具函数
// ============================================================
bool check(float *out, float *ref, int n) {
    for (int i = 0; i < n; i++) {
        if (fabs(out[i] - ref[i]) > 0.005f) return false;
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
    int n, int block_num
) {
    // 计算该版本实际输出的 block 数量
    int actual_blocks = n / ver.elements_per_block;

    // 为该版本生成 CPU 参考结果
    float *h_ref = (float *)malloc(actual_blocks * sizeof(float));
    for (int i = 0; i < actual_blocks; i++) {
        float s = 0;
        for (int j = 0; j < ver.elements_per_block; j++)
            s += h_in[i * ver.elements_per_block + j];
        h_ref[i] = s;
    }

    // 预热（部分版本会原地修改 d_in，每次运行前重新拷贝）
    cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice);
    ver.launcher(d_in, d_out, n, block_num);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float total_ms = 0.0f;
    for (int i = 0; i < NUM_RUNS; i++) {
        cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice);
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

    bool correct = check(h_out, h_ref, actual_blocks);
    free(h_ref);

    return { ver.name, avg_ms, bw, eff, correct };
}

// ============================================================
// main
// ============================================================
int main() {
    const int block_num = N / THREAD_PER_BLOCK;
    const int max_output_size = block_num;  // 最大输出大小（v0-v3）

    float *h_in  = (float *)malloc(N * sizeof(float));
    float *h_out = (float *)malloc(max_output_size * sizeof(float));

    float *d_in, *d_out;
    cudaMalloc(&d_in,  N * sizeof(float));
    cudaMalloc(&d_out, max_output_size * sizeof(float));

    // 初始化输入
    for (int i = 0; i < N; i++)
        h_in[i] = 2.0f * (float)drand48() - 1.0f;

    // 运行所有版本
    std::vector<BenchmarkResult> results;
    printf("Running benchmarks (%d runs each)...\n\n", NUM_RUNS);
    for (int i = 0; i < NUM_VERSIONS; i++)
        results.push_back(run_benchmark(versions[i], d_in, d_out, h_in, h_out, N, block_num));

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

    printf("\nConfig: N=%dM, block=%d, ideal_bw=%.0f GB/s\n",
           N / 1024 / 1024, THREAD_PER_BLOCK, IDEAL_BANDWIDTH);

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);
    return 0;
}
