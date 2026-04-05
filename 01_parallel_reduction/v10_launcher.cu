// Wrapper for reduce_v10_occupancy_full.cu
#define reduce reduce_v10
#define check  check_v10
#define main   main_v10
#define warpReduce warpReduce_v10
#define queryOccupancy queryOccupancy_v10
#define runBenchmark runBenchmark_v10
#include "reduce_v10_occupancy_full.cu"
#undef reduce
#undef check
#undef main
#undef warpReduce
#undef queryOccupancy
#undef runBenchmark

// 对每个 blockSize 查询 occupancy，运行时选择性能最优的配置
// 实测 blockSize=256 在 RTX 4090 上最优，但通过 occupancy 查询可跨架构自适应
void launch_v10(float *d_in, float *d_out, int n, int block_num) {
    // 查询各 blockSize 的 occupancy
    int best_bs = 256;  // fallback
    int best_active = 0;

    // 候选 blockSize: 64, 128, 256, 512（< 64 会 warpReduce 越界，1024 同理不安全）
    int activeBlocks;

    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&activeBlocks, reduce_v10<64>, 64, 0);
    if (activeBlocks * 64 > best_active) { best_active = activeBlocks * 64; best_bs = 64; }

    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&activeBlocks, reduce_v10<128>, 128, 0);
    if (activeBlocks * 128 > best_active) { best_active = activeBlocks * 128; best_bs = 128; }

    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&activeBlocks, reduce_v10<256>, 256, 0);
    if (activeBlocks * 256 > best_active) { best_active = activeBlocks * 256; best_bs = 256; }

    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&activeBlocks, reduce_v10<512>, 512, 0);
    if (activeBlocks * 512 > best_active) { best_active = activeBlocks * 512; best_bs = 512; }

    // 计算 gridSize
    int minGridSize, blockSize_api;
    int gridSize;

    switch (best_bs) {
        case 64:
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize_api, reduce_v10<64>, 0, 0);
            gridSize = minGridSize * 8;
            if (gridSize > n / 128) gridSize = n / 128;
            reduce_v10<64><<<gridSize, 64>>>(d_in, d_out, n);
            break;
        case 128:
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize_api, reduce_v10<128>, 0, 0);
            gridSize = minGridSize * 8;
            if (gridSize > n / 256) gridSize = n / 256;
            reduce_v10<128><<<gridSize, 128>>>(d_in, d_out, n);
            break;
        case 256:
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize_api, reduce_v10<256>, 0, 0);
            gridSize = minGridSize * 8;
            if (gridSize > n / 512) gridSize = n / 512;
            reduce_v10<256><<<gridSize, 256>>>(d_in, d_out, n);
            break;
        case 512:
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize_api, reduce_v10<512>, 0, 0);
            gridSize = minGridSize * 8;
            if (gridSize > n / 1024) gridSize = n / 1024;
            reduce_v10<512><<<gridSize, 512>>>(d_in, d_out, n);
            break;
    }
}
