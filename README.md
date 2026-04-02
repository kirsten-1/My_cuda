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
    └── reduce_v0_global_mem.cu     # v0: 全局内存朴素实现 (baseline)
```

## 学习笔记

### 01 - 并行规约 (Parallel Reduction)

**核心思想：** 将 N 个元素的求和（或其他满足结合律的操作）通过树形归约在 O(log N) 步内完成，充分利用 GPU 的大规模并行能力。

**版本迭代计划：**

| 版本 | 文件 | 策略 | 备注 |
|------|------|------|------|
| v0 | `reduce_v0_global_mem.cu` | 全局内存，交错寻址树形归约 | baseline ✅ 已实现并验证 |
| v1 | (待续) | 共享内存 | 减少全局内存访问 |
| v2 | (待续) | 展开循环 + warp 原语 | 消除 warp 内同步开销 |

**v0 实现要点：**
- 每个 block 处理 `THREAD_PER_BLOCK`(256) 个元素，结果写入 `d_out[blockIdx.x]`
- 树形归约：步长 `i` 每轮翻倍（1→2→4→…），`log2(blockDim.x)` 轮后 `input[0]` 即为 block 总和
- 已知瓶颈：`threadIdx.x % (2*i) == 0` 的取模判断导致同一 warp 内线程走不同分支（warp divergence），v1 将针对此优化
- 数据规模：N = 32M floats，CPU 结果对照校验误差 < 0.005

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
