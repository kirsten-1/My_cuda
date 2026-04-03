# My CUDA Learning Journey

> 记录学习 CUDA 并行编程的心路历程、资料收集与实践笔记。

## 环境

| 项目 | 版本 |
|------|------|
| GPU | NVIDIA GeForce RTX 4090 D (24 GB) |
| Driver | 580.105.08 |
| CUDA Toolkit | 12.8 (V12.8.93) |
| CMake | 3.22.1 |
| OS | Ubuntu 22.04.5 LTS (Jammy Jellyfish) |

## 学习路线

- [x] 01 - 并行规约 (Parallel Reduction)
- [ ] 02 - 矩阵乘法优化 (GEMM)
- [ ] 03 - 共享内存优化 (Shared Memory)
- [ ] 04 - Warp 级原语 (Warp Primitives)
- [ ] 05 - 流与并发 (Streams & Concurrency)

## 项目结构

```
My_cuda/
├── CMakeLists.txt
└── 01_parallel_reduction/          # 并行规约优化
    ├── CMakeLists.txt
    ├── Makefile
    ├── benchmark.cu                # 多版本性能对比
    ├── reduce_v0_global_mem.cu     # v0: 全局内存朴素实现 (baseline)
    ├── reduce_v1_shm.cu            # v1: 共享内存优化
    ├── reduce_v2_warp_divergence.cu # v2: 连续寻址，仍存在 warp divergence
    └── reduce_v3_bank_conflict.cu  # v3: 消除 warp divergence
```

## 学习笔记

### 01 - 并行规约 (Parallel Reduction)

**核心思想：** 将 N 个元素的求和（或其他满足结合律的操作）通过树形归约在 O(log N) 步内完成，充分利用 GPU 的大规模并行能力。

**版本迭代计划：**

| 版本 | 文件 | 策略 | 备注 |
|------|------|------|------|
| v0 | `reduce_v0_global_mem.cu` | 全局内存，交错寻址树形归约 | baseline ✅ |
| v1 | `reduce_v1_shm.cu` | 共享内存，交错寻址树形归约 | 避免修改原数据，减少 global memory 访问 ✅ |
| v2 | `reduce_v2_warp_divergence.cu` | 共享内存，连续寻址（仍有 warp divergence） | 改变索引方式，但分支问题未消除 ✅ |
| v3 | `reduce_v3_bank_conflict.cu` | 共享内存，步长减半，减少 warp divergence | `if (threadIdx.x < i)` 保证前期无 divergence，仅最后几轮（i<32）Warp 0 内有分支 ✅ |
| v4 | (待续) | 加载时归约，提升带宽利用率 | 每线程加载 2 个元素并预先求和，减少 block 数量，提高计算密度 |

**v0 实现要点：**
- 每个 block 处理 `THREAD_PER_BLOCK`(256) 个元素，结果写入 `d_out[blockIdx.x]`
- 树形归约：步长 `i` 每轮翻倍（1→2→4→…），`log2(blockDim.x)` 轮后 `input[0]` 即为 block 总和
- 已知瓶颈：`threadIdx.x % (2*i) == 0` 的取模判断导致同一 warp 内线程走不同分支（warp divergence），v1 将针对此优化
- 数据规模：N = 32M floats，CPU 结果对照校验误差 < 0.005

**v0 的 Global Memory 问题：**
- **高延迟**：归约循环的每一轮读写都直接访问 Global Memory，其延迟比寄存器或 Shared Memory 慢 100 倍以上，是性能瓶颈所在
- **破坏原数据**：`input_dim` 直接指向 `d_in`，归约过程中的加法会原地修改显存中的输入数据，归约完成后原始数据已被覆盖
- **改进方向（v1）**：先将数据从 Global Memory 加载到每个 Block 私有的 Shared Memory，所有加法在片上完成后再写回，可大幅降低访存延迟

**v2 实现要点：**
- 改用连续寻址：`index = threadIdx.x * (2*i)`，每轮只有前 `blockDim.x/(2*i)` 个线程参与
- 相比 v1 的取模判断，活跃线程更集中，但仍跨 warp 边界产生 divergence，性能与 v1 接近

**Warp Divergence 问题分析：**

问题出在条件判断 `if (threadIdx.x < blockDim.x / (2*i))`：

```cuda
for (int i = 1; i < blockDim.x; i *= 2) {
    if (threadIdx.x < blockDim.x / (2*i)) {  // ⚠️ divergence 源头
        int index = threadIdx.x * (2*i);
        sdata[index] += sdata[index + i];
    }
    __syncthreads();
}
```

迭代过程（blockDim.x=256）：

| 轮次 | i | 活跃线程范围 | Warp 0 (T0-T31) 状态 | 浪费比例 |
|------|---|-------------|---------------------|---------|
| 1-3 | 1,2,4 | T0-127/63/31 | 边界对齐，无 divergence | 0% |
| 4 | 8 | T0-15 | T0-15 执行，T16-31 idle | 50% |
| 5 | 16 | T0-7 | T0-7 执行，T8-31 idle | 75% |
| 6 | 32 | T0-3 | T0-3 执行，T4-31 idle | 87.5% |
| 7 | 64 | T0-1 | T0-1 执行，T2-31 idle | 93.75% |
| 8 | 128 | T0 | 仅 T0 执行，T1-31 idle | 96.875% |

**核心问题：** 后期迭代中，Warp 0 内大部分线程空转（masked off），硬件无法同时执行不同路径，必须串行化处理 if/else 分支，导致吞吐量骤降。v3 将通过改变归约策略彻底消除此问题。

**v3 实现要点（减少 warp divergence）：**
- 改用步长减半策略：`for (int i = blockDim.x/2; i > 0; i /= 2)`
- 条件判断改为 `if (threadIdx.x < i)`，活跃线程从 T0 开始连续分布
- **Divergence 分析：**
  - 当 `i >= 32` 时：活跃线程数是 warp 大小的整数倍，warp 边界对齐，**无 divergence**
    - i=128: Warp 0-3 全活跃，Warp 4-7 全 idle ✅
    - i=64: Warp 0-1 全活跃 ✅
    - i=32: Warp 0 全活跃 ✅
  - 当 `i < 32` 时：仅 Warp 0 内部产生 divergence（T0-Ti 执行，Ti+1-T31 idle）
    - i=16: 50% 浪费，i=8: 75%，i=4: 87.5%，i=2: 93.75%，i=1: 96.875%
- **相比 v2 的改进：** v2 从 i=8 开始就有多个 warp 受 divergence 影响，v3 仅最后 5 轮、仅 Warp 0 受影响，大幅减少浪费
- 性能提升显著：从 ~400 GB/s 跃升至 581 GB/s（+45%）

**v3 的瓶颈与 v4 优化方向：**
- **问题 1：线程利用率低**
  - 第一轮归约时 i=128，只有前 128 个线程工作，另外 128 个线程仅负责搬运数据后就闲置
  - 计算资源浪费：一半线程在起始阶段就在"围观"
- **问题 2：指令开销占比高**
  - 每个 block 处理的数据量少（仅 256 个元素），block 启动、同步、写回的固定开销占比大
  - 内存带宽利用率不足：每个线程只加载 1 个数据，未充分利用内存带宽
- **v4 优化策略：加载时归约（Load-Time Reduction）**
  - 减少 block 数量至原来的一半：`Grid(N / (2*THREAD_PER_BLOCK))`
  - 每个线程加载 2 个元素并预先求和：`sdata[tid] = input[tid] + input[tid + blockDim.x]`
  - 效果：Shared Memory 初始化完成时已完成第一轮归约，后续循环从 i=blockDim.x/2 开始，所有线程立即投入工作
  - 预期收益：提高计算密度，减少管理开销，进一步提升带宽利用率

**Benchmark 结果（RTX 4090 D，N=32M floats，block=256，理论带宽 1008 GB/s）：**

| 版本 | 耗时 (ms) | 带宽 (GB/s) | 效率 | 相比 v0 提升 |
|------|-----------|-------------|------|-------------|
| v0 | 0.383 | 351.55 | 34.88% | baseline |
| v1 | 0.337 | 400.08 | 39.69% | +13.8% |
| v2 | 0.342 | 393.78 | 39.07% | +12.0% |
| v3 | 0.232 | 581.19 | 57.66% | **+65.3%** |

v1→v2 性能几乎持平，说明连续寻址本身并未消除 warp divergence 的根本开销，v3 将从算法层面彻底解决。

**编译与运行：**

```bash
# 直接用 nvcc 编译
/usr/local/cuda/bin/nvcc -arch=sm_89 reduce_v0_global_mem.cu -o reduce_v0
./reduce_v0
# 输出: hello reduce
```

**参考：**
- [Optimizing Parallel Reduction in CUDA - Mark Harris (NVIDIA)](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)

---

## 资料收集

### 官方文档
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [CUDA Math API](https://docs.nvidia.com/cuda/cuda-math-api/)
- [PTX ISA Reference](https://docs.nvidia.com/cuda/parallel-thread-execution/)

### 经典资料
- Mark Harris — *Optimizing Parallel Reduction in CUDA* (NVIDIA, 2007)
- [NVIDIA Developer Blog](https://developer.nvidia.com/blog/)
- [CUTLASS](https://github.com/NVIDIA/cutlass) — NVIDIA 官方高性能 GEMM 模板库

### 性能分析工具
- **Nsight Systems** — 系统级时间线分析
- **Nsight Compute** — kernel 级性能剖析（roofline、内存带宽等）
- **compute-sanitizer** — 内存越界 / 竞争条件检测（替代旧版 cuda-memcheck）
