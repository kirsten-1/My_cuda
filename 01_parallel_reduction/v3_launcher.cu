// Wrapper for reduce_v3_bank_conflict.cu
#define reduce reduce_v3
#define check  check_v3
#define main   main_v3
#include "reduce_v3_bank_conflict.cu"
#undef reduce
#undef check
#undef main

void launch_v3(float *d_in, float *d_out, int n, int block_num) {
    dim3 Grid(block_num);
    dim3 Block(256);
    reduce_v3<<<Grid, Block>>>(d_in, d_out);
}
