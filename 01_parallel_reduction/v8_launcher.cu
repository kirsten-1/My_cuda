// Wrapper for reduce_v8_grid_stride_loop.cu
#define reduce reduce_v8
#define check  check_v8
#define main   main_v8
#define warpReduce warpReduce_v8
#include "reduce_v8_grid_stride_loop.cu"
#undef reduce
#undef check
#undef main
#undef warpReduce

#define V8_GRID_SIZE 2048

void launch_v8(float *d_in, float *d_out, int n, int block_num) {
    reduce_v8<256><<<V8_GRID_SIZE, 256>>>(d_in, d_out, n);
}
