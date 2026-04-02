// Wrapper for reduce_v1_shm.cu
#define reduce reduce_v1
#define check  check_v1
#define main   main_v1
#include "reduce_v1_shm.cu"
#undef reduce
#undef check
#undef main

void launch_v1(float *d_in, float *d_out, int n, int block_num) {
    dim3 Grid(block_num);
    dim3 Block(256);
    reduce_v1<<<Grid, Block>>>(d_in, d_out);
}
