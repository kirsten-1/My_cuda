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
├── README.md
└── 01_parallel_reduction/          # 并行规约优化
    ├── CMakeLists.txt
    ├── Makefile
    ├── benchmark.cu                # 多版本性能对比
    ├── reduce_v0_global_mem.cu     # v0: 全局内存朴素实现 (baseline)
    ├── reduce_v1_shm.cu            # v1: 共享内存优化
    ├── reduce_v2_warp_divergence.cu # v2: 连续寻址，仍存在 warp divergence
    ├── reduce_v3_bank_conflict.cu  # v3: 消除 warp divergence
    ├── reduce_v4_add_during_load.cu          # v4: 加载时归约（减 block 数）
    ├── reduce_v5_another_add_during_load.cu  # v5: 加载时归约（减线程数）
    ├── reduce_v6_unrolling_the_last_warp.cu # v6: Warp 展开优化
    ├── v0_launcher.cu              # v0 单独编译运行入口
    ├── v1_launcher.cu              # v1 单独编译运行入口
    ├── v2_launcher.cu              # v2 单独编译运行入口
    ├── v3_launcher.cu              # v3 单独编译运行入口
    ├── v4_launcher.cu              # v4 单独编译运行入口
    ├── v5_launcher.cu              # v5 单独编译运行入口
    └── v6_launcher.cu              # v6 单独编译运行入口
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
| v4 | `reduce_v4_add_during_load.cu` | 加载时归约，减少 block 数量 | 每线程加载 2 个元素并预先求和，block 数减半，提高计算密度 ✅ |
| v5 | `reduce_v5_another_add_during_load.cu` | 加载时归约，减少线程数 | 每 block 线程数减半（128），每线程处理 2 个元素，block 数不变 ✅ |
| v6 | `reduce_v6_unrolling_the_last_warp.cu` | Warp 展开优化 | 最后一个 Warp 内手动展开归约，去除 `__syncthreads()` 和分支判断 ✅ |

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
- **问题 1：线程利用率低** — 第一轮归约 i=128，只有前 128 个线程工作，另外 128 个线程仅搬运数据后闲置
- **问题 2：指令开销占比高** — 每个 block 仅处理 256 个元素，block 启动、同步、写回的固定开销占比大，内存带宽利用不充分

**v4 实现要点（加载时归约 — 减少 block 数量）：**
- Grid 大小减半：`Grid(N / (2*THREAD_PER_BLOCK))`，每个 block 处理 512 个元素
- 每个线程加载 2 个元素并预先求和：`sdata[tid] = input[tid] + input[tid + blockDim.x]`
- 效果：Shared Memory 初始化完成时已完成第一轮归约，所有 256 个线程从一开始就参与计算
- 充分利用内存带宽，减少 block 管理开销，带宽效率达到 **88.83%**

**v5 实现要点（加载时归约 — 减少线程数）：**
- 与 v4 同为"加载时归约"优化，但分配策略相反：
  - **v4**：减少 block 数量，保持每 block 线程数（256），"少而壮的 block"
  - **v5**：减少每 block 线程数（128），每 block 处理数据量不变（256 个元素），"多而轻的 block"
- `THREAD_PER_BLOCK=128`，Grid 大小：`Grid(N / 128 / 2)`，Block 大小：`Block(128)`
- 每个线程加载 2 个元素并预先求和，shared memory 缩小到 128 floats
- 实际结果：887.93 GB/s（88.09%），与 v4 的 895.45 GB/s（88.83%）非常接近，仅慢约 **0.8%**

**v4 vs v5 对比分析：**

| | v4 | v5 |
|---|---|---|
| Block 数量 | N/512（65536） | N/256（131072） |
| 每 block 线程数 | 256（8 warp） | 128（4 warp） |
| 每 block 处理元素数 | 512 | 256 |
| 每线程处理元素数 | 2 | 2 |
| Shared memory 用量 | 256 floats (1 KB) | 128 floats (512 B) |
| 带宽 | **895.45 GB/s** | 887.93 GB/s |
| 效率 | **88.83%** | 88.09% |

**v5 略慢于 v4 的原因分析（差距仅 ~0.8%）：**
1. **Warp 级延迟隐藏能力下降** — v4 每 block 有 8 个 warp，v5 只有 4 个 warp。当某个 warp 等待内存时，warp 调度器在 v4 中有更多候选 warp 可切换，延迟隐藏更充分
2. **指令开销翻倍** — v5 的 block 数量是 v4 的 2 倍（131072 vs 65536），block 启动、__syncthreads() 同步、结果写回 `d_out` 的固定开销加倍
3. **归约轮数相同但参与线程更少** — 两者都需要 log2 轮归约，但 v5 每轮参与的线程更少，计算密度略低
4. **差距极小的原因** — RTX 4090 的 SM 数量（128）和调度能力足够强，v5 虽然每 block 只有 4 个 warp，但 block 数量翻倍弥补了 occupancy，两种策略在该架构上几乎等效

**v6 实现要点（Warp 展开优化 — Unrolling the Last Warp）：**
- **核心观察：** 随着归约层层递进，活跃线程数不断减少。当步长 s ≤ 32 时，所有活跃线程都属于同一个 Warp（线程束）
- **Warp 的同步特性（SIMD synchronous）：** 同一 Warp 内的线程隐式同步执行，步调完全一致（如线程 0 执行加法时，线程 31 也在执行同一条指令）
- **优化点：**
  1. **不再需要 `__syncthreads()`** — Warp 内部天生同步，不需要昂贵的块级别屏障指令来强制等待
  2. **不再需要 `if (tid < s)` 判断** — 同一 Warp 内即使加了 if 判断，硬件依然会串行处理分支（Warp Divergence），并不能节省时间，索性去掉判断让所有线程都运行
- **具体操作：** 手动展开循环的最后 6 次迭代（对应步长 32, 16, 8, 4, 2, 1），使用 `volatile` 修饰 shared memory 指针确保编译器不会优化掉中间读写
- **归约循环变为** `for (i = blockDim.x/2; i > 32; i /= 2)`，当 i ≤ 32 时调用 `warpReduce()` 函数完成最后 6 步
- 保留 v4 的 add-during-load 优化（256 线程，每 block 处理 512 个元素）

**Benchmark 结果（RTX 4090 D，N=32M floats，block=256，理论带宽 1008 GB/s）：**

| 版本 | 耗时 (ms) | 带宽 (GB/s) | 效率 | 相比 v0 提升 |
|------|-----------|-------------|------|-------------|
| v0 | 0.381 | 353.22 | 35.04% | baseline |
| v1 | 0.324 | 416.37 | 41.31% | +17.9% |
| v2 | 0.329 | 409.36 | 40.61% | +15.9% |
| v3 | 0.220 | 613.03 | 60.82% | **+73.6%** |
| v4 | 0.151 | 890.75 | 88.37% | **+152.2%** |
| v5 | 0.149 | 905.31 | 89.81% | **+156.3%** |
| v6 | 0.148 | 908.15 | 90.09% | **+157.1%** |

v1→v2 性能几乎持平，说明连续寻址本身并未消除 warp divergence 的根本开销。v3 从算法层面大幅减少 divergence，v4/v5 通过加载时归约将带宽效率推至接近理论峰值（~89%）。v6 在 v4 基础上展开最后一个 Warp 的归约循环，消除了 5 次 `__syncthreads()` 和分支判断，带宽效率进一步提升至 **90.09%**。

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
