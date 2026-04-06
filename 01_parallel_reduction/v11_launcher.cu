// Wrapper for reduce_v11_shuffle.cu
#define reduce reduce_v11
#define check  check_v11
#define main   main_v11
#define warpReduceShuffle warpReduceShuffle_v11
#include "reduce_v11_shuffle.cu"
#undef reduce
#undef check
#undef main
#undef warpReduceShuffle

#define V11_GRID_SIZE 2048

void launch_v11(float *d_in, float *d_out, int n, int block_num) {
    reduce_v11<256><<<V11_GRID_SIZE, 256>>>(d_in, d_out, n);
}
