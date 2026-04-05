// Wrapper for reduce_v6_unrolling_the_last_warp.cu
#define reduce reduce_v6
#define check  check_v6
#define main   main_v6
#define warpReduce warpReduce_v6
#include "reduce_v6_unrolling_the_last_warp.cu"
#undef reduce
#undef check
#undef main
#undef warpReduce

void launch_v6(float *d_in, float *d_out, int n, int block_num) {
    dim3 grid(block_num / 2);
    dim3 block(256);
    reduce_v6<<<grid, block>>>(d_in, d_out);
}
