// Wrapper for reduce_v7_complete_unrolling_and_templates.cu
#define reduce reduce_v7
#define check  check_v7
#define main   main_v7
#define warpReduce warpReduce_v7
#include "reduce_v7_complete_unrolling_and_templates.cu"
#undef reduce
#undef check
#undef main
#undef warpReduce

void launch_v7(float *d_in, float *d_out, int n, int block_num) {
    dim3 grid(block_num / 2);
    dim3 block(256);
    reduce_v7<256><<<grid, block>>>(d_in, d_out);
}
