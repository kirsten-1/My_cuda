# Kernel 启动参数的艺术：`<<<Grid, Block>>>` 怎么配？

## 一句话总结
> 对于普通的 elementwise 任务，**Block 设 128 或 256** 是万金油，**Grid 设置为能产生足够多 Wave 的大小**，让所有 SM 都忙起来且避免长尾效应。

---

## 1. 参数全貌

Kernel 启动的标准语法：
```cpp
cuda_kernel<<<Dg, Db, Ns, S>>>(...)
```
- **Dg (Grid Size)**：`dim3` 类型，一共启动多少个 Block。
- **Db (Block Size)**：`dim3` 类型，每个 Block 里有多少个 Thread。
- **Ns (Shared Memory)**：`size_t`，动态分配的共享内存大小（字节），默认 0。
- **S (Stream)**：`cudaStream_t`，所在的流，默认 0（同步流）。

最简形式：`<<<grid_size, block_size>>>`，此时 Ns=0, S=0，且 Grid 和 Block 都是一维的。

---

## 2. Block Size 应该怎么定？（Db）

Block Size 是这门艺术的核心。它绝不能随便拍脑袋，受限于以下几个物理铁律：

### 铁律 1：最大 1024
硬件规定，每个 Block 的线程数上限是 1024。

### 铁律 2：必须是 32 的倍数（Warp 机制）
GPU 按 Warp（32个线程）为单位发射指令。如果 Block 大小是 100，硬件依然会分配 4 个 Warp（128个线程），剩下的 28 个线程纯属浪费。

### 铁律 3：满足 Occupancy（占用率）及 SM 资源限制
这是最考验内功的地方：
- **Block 是调度的最小原子**：一旦一个 Block 被扔到 SM 上，它的所有线程必须**同时**呆在那个 SM 里。
- **Occupancy 追求**：我们要让 SM 上的活跃线程数尽量接近它的硬件上限。
  - V100/A100 的上限：每 SM 最大 2048 线程，最多 32 个 Block。
  - RTX 3090 的上限：每 SM 最大 1536 线程，最多 16 个 Block。
- **推导最优解**：
  - 为了不触碰 Block 数量的上限而导致线程填不满（比如 3090 上，16 个 block × 32 线程 = 512 线程 < 1536 上限），Block 大小**不能小于 96**。
  - 为了完美整除最大线程数，Block Size 必须是 2048 和 1536 的公约数（512）。
  - 结合以上，通用解只剩下：**128、256、512**。
- **寄存器雷区**：每个 SM 的寄存器总量有限（一般 64K 个）。如果 Kernel 每个线程用的寄存器巨多（比如逼近 255 个上限），Block 设为 512 会导致“一个 Block 就把 SM 寄存器吸干”甚至启动失败。此时 **128 或 256** 是最安全的 fallback。

**结论：** 无脑首选 **128 或 256**，遇到极吃共享内存的情况再去单独推算。

---

## 3. Grid Size 应该怎么定？（Dg）

Grid Size 决定了总工作量。

### 常规逻辑：元素总数 / Block Size
最简单的：`grid_size = (N + block_size - 1) / block_size`。每个线程处理一个元素。

### 进阶优化：公共开销均摊（Grid-Stride Loop）
如果你的 Kernel 每次进门都要算一个复杂的公共值：
```cpp
__global__ void kernel(const float* x, const float* v, float* y) {
   const float sqrt_v = sqrt(*v);  // ← 公共开销，很贵！
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   // 如果用 Grid-Stride Loop，每个线程可以处理多个元素，均摊开销
   for (int i = idx; i < N; i += blockDim.x * gridDim.x) {
       y[i] = x[i] * sqrt_v;
   }
}
```
此时你不需要给每个元素开一个线程，你可以**减少 Grid Size**，让每个线程通过 `for` 循环多干点活。

### 必须懂的黑话：Wave（波）和 Tail Effect（长尾效应）
- **Wave**：假设你的 GPU 有 100 个 SM，每个 SM 能同时跑 2 个 Block，那么 GPU 一次性最多吃下 200 个 Block。这 200 个 Block 就是“一个 Wave”。
- **Tail Effect**：如果你的 `grid_size = 201`，那么前 200 个 Block 会在第一个 Wave 同时执行。大家执行完后，**只剩下最后 1 个 Block**。此时 GPU 为了等这最后 1 个独苗执行完，99 个 SM 都在挂机摸鱼！
- **怎么设**：
  1. Grid 决不能小于 SM 数量（否则有 SM 永远不干活）。
  2. Grid 最好足够大（几千上万）。当 Wave 数量极多（比如几十上百个）时，最后一个未满的 Wave 造成的占比极小，Tail Effect 就可以忽略不计了。

---

## 面试速通 Q&A

**Q1：启动 Kernel 时，<<<Grid, Block>>> 中的 Block 设置为 100 可以吗？为什么？**
**答：** 语法上可以，但性能上绝对不行。GPU 的调度和执行是以 Warp（32个线程）为基本单位的。如果 Block 设为 100，硬件会为它分配 4 个 Warp（即 128 个线程），其中 28 个线程的计算掩码被关闭（被浪费掉）。同时这 128 个线程依然会占用 SM 上的寄存器和共享内存资源，极大地降低了硬件利用率。所以 Block Size 必须是 32 的倍数。

**Q2：如果我有一个 1000 万长度的数组要处理，我是应该创建 1000 万个线程（巨大的 Grid Size），还是使用少量的线程让每个线程内部做 for 循环（Grid-Stride Loop）？**
**答：** 视情况而定。
如果每个线程只是执行极简的 `C[i] = A[i] + B[i]`，创建 1000 万个线程完全没问题，因为 GPU 创建和调度线程的开销极小。
但如果 Kernel 开头包含了昂贵的公共操作（比如从 Global Memory 读一个常数，或者执行一个耗时的 `sqrt` / `exp` 计算），那就更推荐 Grid-Stride Loop。通过限制总线程数（Grid Size），让每个线程在内部 `for` 循环处理多个元素，可以极大地均摊这个公共操作的开销。同时，设置合适的 Grid Size（产生足够多的完整 Wave）可以有效避免长尾效应（Tail Effect）。

**Q3：为什么各大开源库（如 CUTLASS、PyTorch）里，很多 Elementwise 算子的默认 Block Size 都喜欢设为 128 或 256，而不是最大的 1024？**
**答：** 为了追求极致的 SM 占用率（Occupancy）并避免资源溢出。
Block 被调度到 SM 是原子操作。每个 SM 的寄存器和最大活跃线程数有硬件上限（比如 1536 或 2048）。
如果设为 1024，在最大线程数为 1536 的架构上，SM 只能塞进 1 个 Block，剩下的 512 个线程容量就被浪费了（Occupancy 只有 66%）。
如果设为 128 或 256，不仅能完美整除主流架构的最大线程数，而且颗粒度更细。即使 Kernel 消耗了比较多的寄存器，SM 也能灵活塞进 3个、4个 或 5个 Block，极大地保证了多代架构下的兼容性和高利用率。