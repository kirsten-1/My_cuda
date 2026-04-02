// Wrapper for reduce_v2_warp_divergence.cu
#define reduce reduce_v2
#define check  check_v2
#define main   main_v2
#include "reduce_v2_warp_divergence.cu"
#undef reduce
#undef check
#undef main

void launch_v2(float *d_in, float *d_out, int n, int block_num) {
    dim3 Grid(block_num);
    dim3 Block(256);
    reduce_v2<<<Grid, Block>>>(d_in, d_out);
}
