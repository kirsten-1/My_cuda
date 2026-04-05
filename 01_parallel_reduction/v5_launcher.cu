// Wrapper for reduce_v4_add_during_load.cu
#define reduce reduce_v5
#define check  check_v5
#define main   main_v5
#include "reduce_v5_another_add_during_load.cu"
#undef reduce
#undef check
#undef main

void launch_v5(float *d_in, float *d_out, int n, int block_num) {
    dim3 grid(n / 128 / 2, 1);
    dim3 block(128, 1);
    reduce_v5<<<grid, block>>>(d_in, d_out);
}
