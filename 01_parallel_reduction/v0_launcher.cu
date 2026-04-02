// Wrapper for reduce_v0_global_mem.cu
// Renames symbols to avoid conflicts when linking with other versions
#define reduce reduce_v0
#define check  check_v0
#define main   main_v0
#include "reduce_v0_global_mem.cu"
#undef reduce
#undef check
#undef main

void launch_v0(float *d_in, float *d_out, int n, int block_num) {
    dim3 Grid(block_num);
    dim3 Block(256);
    reduce_v0<<<Grid, Block>>>(d_in, d_out);
}
