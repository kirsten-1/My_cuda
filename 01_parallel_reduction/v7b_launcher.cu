// Wrapper for reduce_v7_v2_pragma_unroll.cu
#define reduce reduce_v7b
#define check  check_v7b
#define main   main_v7b
#define warpReduce warpReduce_v7b
#include "reduce_v7_v2_pragma_unroll.cu"
#undef reduce
#undef check
#undef main
#undef warpReduce

void launch_v7b(float *d_in, float *d_out, int n, int block_num) {
    dim3 grid(block_num / 2);
    dim3 block(256);
    reduce_v7b<256><<<grid, block>>>(d_in, d_out);
}
