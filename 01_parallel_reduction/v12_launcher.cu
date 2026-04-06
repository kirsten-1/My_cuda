// Wrapper for reduce_v12_warp_level_sums.cu
#define reduce reduce_v12
#define check  check_v12
#define main   main_v12
#define warpReduceShuffle warpReduceShuffle_v12
#include "reduce_v12_warp_level_sums.cu"
#undef reduce
#undef check
#undef main
#undef warpReduceShuffle

#define V12_GRID_SIZE 2048

void launch_v12(float *d_in, float *d_out, int n, int block_num) {
    reduce_v12<256><<<V12_GRID_SIZE, 256>>>(d_in, d_out, n);
}
