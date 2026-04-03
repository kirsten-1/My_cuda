// Wrapper for reduce_v4_add_during_load.cu
#define reduce reduce_v4
#define check  check_v4
#define main   main_v4
#include "reduce_v4_add_during_load.cu"
#undef reduce
#undef check
#undef main

void launch_v4(float *d_in, float *d_out, int n, int block_num) {
    // v4 每个 block 处理 2*blockDim.x 个元素，所以实际需要的 block 数量是传入的一半
    dim3 Grid(block_num / 2);
    dim3 Block(256);
    reduce_v4<<<Grid, Block>>>(d_in, d_out);
}
