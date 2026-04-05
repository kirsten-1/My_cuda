# 性能优化总纲（CUDA 优化的内功心法）

## 一句话总结
> 懂硬件特性，拿峰值对标，找准瓶颈，先改算法再扣指令。

---

## 1. 深入理解 CUDA 硬件性能特征

在进行任何优化之前，必须在脑海中建立以下四座大山的概念：

### 1) Memory Coalescing（全局内存合并访问）
**核心法则：** 相邻的线程（如 T0, T1, T2）必须访问相邻的内存地址（如 addr, addr+4, addr+8）。
**违背后果：** 硬件无法将多个线程的小读取合并成一个大事务（Transaction），导致多次重复发起显存请求，带宽利用率暴跌。

### 2) Divergent Branching（分支散离 / 线程束分化）
**核心法则：** 同一个 Warp（32个线程）内，尽量避免走不同的 `if/else` 分支。
**违背后果：** 硬件没有分支预测，如果 T0 走 `if`，T1 走 `else`，那么整个 Warp 会**串行**把 `if` 和 `else` 两条路都走一遍（走其中一条时，另一部分线程处于 idle 掩码状态），性能减半甚至更糟。

### 3) Bank Conflicts（共享内存 Bank 冲突）
**核心法则：** 同一个 Warp 的线程访问 Shared Memory 时，尽量不要访问同一个 Bank 的不同地址。
**违背后果：** Shared Memory 被划分为 32 个 Bank（类似 32 个独立的小内存条）。如果多个线程同时访问同一个 Bank，硬件只能把这些请求排队处理（串行化），这叫 Bank Conflict。

### 4) Latency Hiding（延迟隐藏）
**核心法则：** 让尽量多的 Warp 同时驻留在 SM（流式多处理器）上。
**违背后果：** GPU 没有 CPU 那样庞大的 Cache 来掩盖内存延迟，它靠的是“人海战术”。当 Warp A 在等内存数据时，调度器会瞬间切换到 Warp B 执行计算。如果驻留的 Warp 太少（Occupancy 低），SM 就会彻底闲置。

---

## 2. 以“理论峰值”作为优化罗盘

> **"Use peak performance metrics to guide optimization"**

**盲目优化是大忌。** 在优化之前，你必须知道当前硬件的**理论极限**在哪里：
1. **显存带宽极限**：比如 RTX 4090 的显存带宽是 ~1008 GB/s。
2. **算力极限**：TFLOPS（每秒万亿次浮点运算）。

**实战意义**：
在我们的 Reduce 实验中，每次 benchmark 都会打印当前的 Bandwidth 效率。
- v0: 349 GB/s (34%) -> "说明还有巨大空间"
- v4: 895 GB/s (88%) -> "说明内存带宽已经被榨干，不再是瓶颈"
**当你达到理论峰值的 80%-90% 时，就该停止在这个维度的死磕了。**

---

## 3. 找准瓶颈 (Identify the Bottleneck)

不要头痛医脚，先弄清楚你的程序受限于什么：

| 瓶颈类型 | 表现 | 常见对策 |
|----------|------|----------|
| **Memory Bound (访存瓶颈)** | 频繁访问 Global Memory，计算指令少 | 改用 Shared Memory、确保内存合并、加载时归约 (v4) |
| **Compute Bound (计算瓶颈)** | 复杂的数学运算 (sin, exp) 或深层循环 | 算法降维、使用快速数学库 (`__fdividef` 等) |
| **Instruction Overhead (指令开销)** | 一大堆 `if`，一堆 `__syncthreads()`，一堆变量自增 | 循环展开 (v6)、完全展开 (v7) |

---

## 4. 优化的先后顺序（极度重要！）

> **"Optimize your algorithm, *then* unroll loops"**

**第一步：算法级优化（降维打击）**
比如树形归约的交错寻址到连续寻址（v2/v3 解决 Divergence）、加载时完成归约（v4 砍掉一半 block）。这些优化动辄带来 50% 甚至翻倍的提升。

**第二步：底层指令级优化（锦上添花）**
比如 Warp 展开（v6）、模板完全展开（v7）。这些属于压榨指令流水线，通常只能带来 1% ~ 5% 的微小提升。

**反面教材：** 拿着一版满是 Warp Divergence 的 v0 代码，花了一天时间去写模板做循环展开，最后发现性能还是卡在最外层的 if 判断上。

---

## 面试速通 Q&A

**Q1：在接手一个性能不佳的 CUDA Kernel 时，你的优化思路和步骤是什么？**
**答：** 我会遵循“测量定位 -> 算法优化 -> 硬件级微调”的步骤。
首先，利用 Nsight Compute 等工具测量 Kernel 的性能，重点对比当前带宽/算力与硬件理论峰值（Peak Performance）的比例，判断是 Memory Bound、Compute Bound 还是 Instruction Overhead。
其次，优先从算法层面入手解决致命问题：比如检查 Global Memory 是否合并访问（Coalescing）、消除同一个 Warp 内的分支分化（Divergence）、解决 Shared Memory 的 Bank Conflict。
最后，当算法层面的大头解决后，再进行指令级别的微调，例如通过 `#pragma unroll` 或 C++ 模板（Templates）做完全循环展开，消除循环控制和显式同步（`__syncthreads()`）的开销。

**Q2：什么是延迟隐藏（Latency Hiding），它和 CPU 的缓存机制有什么区别？**
**答：** 
CPU 掩盖内存高延迟的方法是使用庞大的 L1/L2/L3 Cache。如果数据在 Cache 里，就不用去慢速内存拿。
GPU 掩盖延迟的方法是**线程切换（Context Switch）**。GPU 上每个 SM（流式多处理器）会同时容纳非常多的 Warp（几十个甚至上百个）。当 Warp A 执行一条读取显存的指令（可能需要几百个时钟周期）时，硬件调度器会瞬间以零开销切换到 Warp B 去执行 ALU 计算。只要驻留的 Warp 足够多，SM 的计算单元就永远不会闲置。这种靠“人海战术”掩盖延迟的机制就叫 Latency Hiding。

**Q3：为什么说“先优化算法，再展开循环”？如果不按这个顺序会怎样？**
**答：** 因为这两者解决的瓶颈不在一个数量级。算法层面的问题（比如严重的 Warp Divergence 或非合并的内存访问）会极大浪费硬件吞吐量，是主要矛盾；而循环展开解决的是指令开销（Instruction Overhead），是次要矛盾。
如果在算法存在根本性缺陷时就去做循环展开，相当于在一条满是坑洼的泥泞小路上给汽车换上 F1 赛车的轮胎——不仅对速度提升微乎其微，还会因为循环展开导致代码体积（Register 和 Instruction Cache）膨胀，反而可能降低 Occupancy，得不偿失。