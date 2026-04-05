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
    ├── reduce_v7_complete_unrolling_and_templates.cu # v7: 完全展开 + 模板
    ├── reduce_v7_v2_pragma_unroll.cu # v7b: #pragma unroll + 模板
    ├── reduce_v8_grid_stride_loop.cu # v8: Grid-stride loop
    ├── test_v8.cu                   # v8 专项测试（数据规模扫描/L2冷热对比/grid size扫描）
    ├── v0_launcher.cu ~ v8_launcher.cu  # 各版本 benchmark launcher
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
| v7 | `reduce_v7_complete_unrolling_and_templates.cu` | 完全展开 + 模板 | 模板参数替代 `blockDim.x`，所有归约轮次编译期展开，消除循环开销 ✅ |
| v7b | `reduce_v7_v2_pragma_unroll.cu` | `#pragma unroll` + 模板 | 用 `for` 循环 + `#pragma unroll` 替代手动 if 链，代码更简洁，效果等价 ✅ |
| v8 | `reduce_v8_grid_stride_loop.cu` | Grid-stride loop | block 数量与数据规模解耦，每个线程通过 while 循环处理多段数据 ✅ |

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

**v7 实现要点（完全展开 + 模板参数 — Complete Unrolling with Templates）：**
- **核心思想：** v6 只展开了最后一个 Warp（步长 ≤ 32），v7 更进一步，将整个归约循环完全展开，彻底消除循环控制开销
- **模板参数替代 `blockDim.x`：** 使用 `template <unsigned int blockSize>` 将 block 大小作为编译期常量，所有条件判断（`if (blockSize >= 512)` 等）在编译时求值，不满足的分支被编译器直接删除，不会生成任何机器码
- **完全展开的归约过程（以 blockSize=256 为例）：**
  - `blockSize >= 512` → 编译期判定为 false，整个 if 块被删除
  - `blockSize >= 256` → true，执行 `sdata[tid] += sdata[tid + 128]`，后接 `__syncthreads()`
  - `blockSize >= 128` → true，执行 `sdata[tid] += sdata[tid + 64]`，后接 `__syncthreads()`
  - 步长 ≤ 32 后调用 `warpReduce()` 完成最后 6 步（与 v6 相同）
- **优势：** 消除了 `for` 循环的迭代变量更新、条件判断和跳转指令，生成的 PTX/SASS 代码为纯线性序列
- **灵活性：** 同一份代码可通过不同模板实例化（`reduce<128>`、`reduce<256>`、`reduce<512>`）适配不同 block 大小，无需手动修改

**v7b 实现要点（`#pragma unroll` + 模板 — 编译器辅助完全展开）：**
- **与 v7 的区别：** v7 手动写 `if (blockSize >= N)` 链逐一列出每轮归约，v7b 用 `for` 循环 + `#pragma unroll` 让编译器自动展开
- **代码形式：**
  ```cuda
  #pragma unroll
  for (unsigned int i = blockSize / 2; i > 32; i /= 2) {
      if (tid < i) { sdata[tid] += sdata[tid + i]; }
      __syncthreads();
  }
  ```
- `blockSize` 是模板编译期常量，`#pragma unroll` 指示编译器将循环完全展开，生成的机器码与 v7 等价
- **优势：** 代码更简洁，无需为每种 block 大小手动写 if 分支，可维护性更好；添加新的 block 大小支持无需修改归约逻辑

**v8 实现要点（Grid-Stride Loop — 网格跨步循环）：**
- **核心变化：** v0-v7b 中 block 数量由数据规模决定（`block_num = N / elements_per_block`），block 数量与 N 绑死。v8 将 grid 大小（block 数量）与数据规模彻底解耦，block 数量可以自由设置（如 2048），每个 block 通过 `while` 循环源源不断地拉取数据
- **Grid-Stride Loop 代码：**
  ```cuda
  sdata[tid] = 0;
  while (i < n) {
      sdata[tid] += d_in[i] + d_in[i + blockSize];
      i += gridSize;  // gridSize = blockSize * 2 * gridDim.x
  }
  ```
- **为什么这个优化能让带宽起飞：**
  1. **保持 SM 始终满载（Keep SMs busy）：** 以前的版本启动成千上万个 block，GPU 需要不断进行上下文切换。v8 通过设置合理的 grid_size（SM 数量的倍数），让固定数量的线程"常驻"在 SM 上，通过 while 循环持续拉取数据
  2. **隐藏指令开销（Instruction Latency Hiding）：** 每个线程在 while 循环里处理多批数据，连续的访存请求能更好地填满 GPU 的内存流水线，极大提高吞吐量
  3. **减少 block 管理开销：** 只需启动和管理 2048 个 block（而非 65536+），block 启动、同步、结果写回的固定开销大幅降低
- **Grid Size 选择：** RTX 4090 有 128 个 SM，grid_size 取 SM 数量的倍数较为合理。实测 512 以上即可饱和，甜点区间为 512-65536

**v8 性能验证 — 对 ~100% 效率的怀疑与求证：**

在初版 benchmark（N=32M，无 L2 flush）中，v8 测得 **1006 GB/s（99.83%）**，几乎达到理论峰值。这个数字引起了怀疑，于是进行了三组专项测试（`test_v8.cu`）：

**测试 1 — 数据规模扫描（验证 L2 缓存影响）：**

RTX 4090 的 L2 缓存为 72 MB。32M floats = 128 MB 虽然超过 L2，但反复运行时部分数据仍可命中缓存。

| N | 数据量 | 耗时 (ms) | 带宽 (GB/s) | 效率 |
|---|--------|-----------|-------------|------|
| 32M | 128 MB | 0.146 | 918.35 | 91.11% |
| 64M | 256 MB | 0.286 | 937.77 | 93.03% |
| 128M | 512 MB | 0.567 | 947.25 | 93.97% |
| 256M | 1024 MB | 1.126 | 953.44 | 94.59% |

数据越大效率越高，因为大数据摊薄了 kernel 启动开销，且内存访问模式更连续。

**测试 2 — Cold vs Hot L2 对比（N=32M）：**

| 模式 | 耗时 (ms) | 带宽 (GB/s) | 效率 |
|------|-----------|-------------|------|
| Hot L2（连续跑） | 0.147 | 913.45 | 90.62% |
| Cold L2（每次刷缓存） | 0.202 | 665.75 | 66.05% |

关键发现：Hot L2 下约 913 GB/s，Cold L2 下降至 666 GB/s。初版 benchmark 中的 ~1006 GB/s 包含了 L2 缓存加成（benchmark 每轮执行 `cudaMemcpy` 重传数据，传输本身会将数据预热到 L2 中）。

**测试 3 — Grid Size 扫描（N=32M）：**

| Grid Size | 耗时 (ms) | 带宽 (GB/s) | 效率 |
|-----------|-----------|-------------|------|
| 64 | 0.306 | 439.19 | 43.57% |
| 128 | 0.182 | 735.72 | 72.99% |
| 256 | 0.149 | 902.11 | 89.50% |
| 512 | 0.146 | 917.60 | 91.03% |
| 1024 | 0.146 | 919.21 | 91.19% |
| 2048 | 0.146 | 919.89 | 91.26% |
| 4096 | 0.146 | 921.92 | 91.46% |
| 8192 | 0.146 | 921.02 | 91.37% |

64 blocks（仅 0.5× SM 数）性能断崖，256 blocks 起即可达到 ~90%，512 以上完全饱和。甜点在 512-8192，选 2048 作为默认值。

**Benchmark 方法论改进：**

基于 v8 测试中发现的 L2 缓存问题，对 benchmark 框架做了以下改进，确保所有版本在同等条件下公平对比：

1. **数据规模提升至 N=256M（1 GB）**：远超 L2 缓存（72 MB），L2 命中率被稀释到 ~7% 以下，测得的数据直接反映 DRAM 吞吐
2. **引入 L2 Flush Kernel**：每次计时运行前，执行一个 96 MB 的无关数组读写，将之前的数据挤出 L2 缓存，确保 Cold Cache 测量。这是学术界验证"冷缓存"性能的标准做法
3. **保留每轮 `cudaMemcpy`**：因为 v0 会原地修改 `d_in`（global memory 归约），需要每轮恢复数据。`cudaMemcpy` 在 L2 flush 之前执行，不影响 Cold Cache 效果
4. **统一使用全局求和校验**：大数据量下逐 block 校验的浮点误差过大，改为对所有 block 输出求和后与 CPU 双精度参考值对比

**Benchmark 结果（RTX 4090 D，N=256M floats，block=256，理论带宽 1008 GB/s，L2 flush=96MB）：**

| 版本 | 耗时 (ms) | 带宽 (GB/s) | 效率 | 相比 v0 提升 |
|------|-----------|-------------|------|-------------|
| v0 | 2.860 | 376.93 | 37.39% | baseline |
| v1 | 2.462 | 437.87 | 43.44% | +16.2% |
| v2 | 2.515 | 428.53 | 42.51% | +13.7% |
| v3 | 1.706 | 631.69 | 62.67% | **+67.6%** |
| v4 | 1.193 | 902.03 | 89.49% | **+139.3%** |
| v5 | 1.197 | 900.65 | 89.35% | **+138.9%** |
| v6 | 1.191 | 903.26 | 89.61% | **+139.6%** |
| v7 | 1.191 | 903.07 | 89.59% | **+139.6%** |
| v7b | 1.191 | 903.07 | 89.59% | **+139.6%** |
| v8 | 1.188 | 903.91 | 89.67% | **+139.8%** |

v4-v8 在 Cold Cache + 大数据量条件下性能非常接近（~903 GB/s，89.6%），说明在数据量远超 L2 缓存时，**内存带宽已成为绝对瓶颈**，计算侧的优化（warp 展开、模板展开、grid-stride loop）收益趋于饱和。v8 的 grid-stride loop 优势主要体现在 L2 可以部分命中的小数据场景（N=32M 时 v8 比 v4-v7b 高约 3%），以及代码灵活性（grid 大小可独立调优）。

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
