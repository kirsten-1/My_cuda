NVCC    = /usr/local/cuda/bin/nvcc
CFLAGS  = -O3 -std=c++14

# 单独编译各版本（保持不变）
reduce_v0: reduce_v0_global_mem.cu
	$(NVCC) $(CFLAGS) -o $@ $<

reduce_v1: reduce_v1_shm.cu
	$(NVCC) $(CFLAGS) -o $@ $<

reduce_v2: reduce_v2_warp_divergence.cu
	$(NVCC) $(CFLAGS) -o $@ $<

reduce_v3: reduce_v3_bank_conflict.cu
	$(NVCC) $(CFLAGS) -o $@ $<

reduce_v4: reduce_v4_add_during_load.cu
	$(NVCC) $(CFLAGS) -o $@ $<

# benchmark：链接所有版本的 launcher
benchmark: benchmark.cu v0_launcher.cu v1_launcher.cu v2_launcher.cu v3_launcher.cu v4_launcher.cu v5_launcher.cu v6_launcher.cu
	$(NVCC) $(CFLAGS) -o $@ $^

# 添加新版本 vN 的步骤：
#   1. 写 reduce_vN_xxx.cu（核函数命名为 reduce，有自己的 main，可独立运行）
#   2. 写 vN_launcher.cu（用宏重命名后 include，提供 launch_vN 函数）
#      —— 或者直接在 reduce_vN_xxx.cu 里提供 launch_vN，跳过 wrapper
#   3. 在 benchmark.cu 加一行 extern + 一行注册
#   4. 在这里的 benchmark 目标里加上 vN_launcher.cu

.PHONY: clean
clean:
	rm -f reduce_v0 reduce_v1 reduce_v2 reduce_v3 reduce_v4 reduce_v5 reduce_v6 benchmark
