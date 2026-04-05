// Wrapper for reduce_v9_occupancy_grid.cu
#define reduce reduce_v9
#define check  check_v9
#define main   main_v9
#define warpReduce warpReduce_v9
#include "reduce_v9_occupancy_grid.cu"
#undef reduce
#undef check
#undef main
#undef warpReduce

void launch_v9(float *d_in, float *d_out, int n, int block_num) {
    // 使用 Occupancy API 自动计算 gridSize
    int minGridSize, blockSize_api;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize_api,
                                       reduce_v9<256>, 0, 0);
    int gridSize = minGridSize * 8;
    int maxGrid = n / (2 * 256);
    if (gridSize > maxGrid) gridSize = maxGrid;
    reduce_v9<256><<<gridSize, 256>>>(d_in, d_out, n);
}
