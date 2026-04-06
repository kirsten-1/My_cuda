# `__syncthreads()` — Block 级同步原语

## 一句话总结

> 所有线程停下手中的活，把刚才写的数据刷到内存里，等大家都到齐了，再一起往后走。

---

## 三个核心功能

### 1. 屏障同步（Barrier Synchronization）

同一 Block 内的所有线程必须都到达 `__syncthreads()`，任何线程才能继续执行。

- **作用域**：仅限当前 Block，不同 Block 之间互不感知
- **类比**：马拉松的检查站——跑得快的人必须等所有人到齐

**代码示例**（v1 `reduce_v1_shm.cu`）：

```cuda
// 每个线程将全局内存数据加载到共享内存
sdata[threadIdx.x] = input_dim[threadIdx.x];
__syncthreads();  // ← 确保 256 个线程全部加载完毕，才开始归约

for (int i = 1; i < blockDim.x; i *= 2) {
    if (threadIdx.x % (2 * i) == 0) {
        sdata[threadIdx.x] += sdata[threadIdx.x + i];  // 读取其他线程写入的数据
    }
    __syncthreads();  // ← 确保本轮所有写入完成，下一轮才能读取
}
```

如果去掉第一个 `__syncthreads()`：线程 T0 可能在 T255 还没加载完时就开始归约，读到的是垃圾数据。

### 2. 内存一致性（Memory Fence）

GPU 为了性能，线程对 Shared Memory 的写入**不会立即**对其他线程可见。`__syncthreads()` 强制将之前的所有写入刷新为全局可见。

**代码示例**（v3 `reduce_v3_bank_conflict.cu`）：

```cuda
for (int i = blockDim.x / 2; i > 0; i /= 2) {
    if (threadIdx.x < i) {
        sdata[threadIdx.x] += sdata[threadIdx.x + i];  // 写入 sdata
    }
    __syncthreads();  // ← 内存屏障：确保 sdata 的写入对所有线程可见
    // 下一轮循环中，其他线程会读取本轮写入的值
    // 没有这个屏障 → 可能读到旧值 → 结果错误
}
```

每轮归约都依赖上一轮的结果。没有内存屏障，线程 A 写入的 `sdata[0]` 可能还停留在寄存器/缓存中，线程 B 读到的仍是旧值。

### 3. 指令执行顺序保证

硬件层面，不同 Warp 的执行进度不同（有的快、有的慢）。`__syncthreads()` 保证：**所有线程完成同步点之前的最后一条指令，才会有任何线程执行同步点之后的第一条指令。**

**代码示例**（v4 `reduce_v4_add_during_load.cu`）：

```cuda
// 阶段1：加载时归约（所有 256 个线程参与）
sdata[threadIdx.x] = input_dim[threadIdx.x] + input_dim[threadIdx.x + blockDim.x];
__syncthreads();  // ← 阶段分界线：加载阶段 100% 完成后，才进入归约阶段

// 阶段2：树形归约
for (int i = blockDim.x / 2; i > 0; i /= 2) { ... }
```

没有这个同步点，Warp 0 可能已经开始归约，而 Warp 7 还在做加载时求和，归约读到的是未完成的中间状态。

---

## 两个致命禁忌

### 禁忌 1：在分歧的条件分支中使用 → 死锁

```cuda
// 错误示范！
if (threadIdx.x < 16) {
    __syncthreads();  // T0-T15 在这里等待
} else {
    // T16-T255 永远不会到达上面的同步点
}
// → 死锁：T0-T15 永远等不到 T16-T255
```

**正确做法**（v3 的写法）：

```cuda
for (int i = blockDim.x / 2; i > 0; i /= 2) {
    if (threadIdx.x < i) {
        sdata[threadIdx.x] += sdata[threadIdx.x + i];
    }
    __syncthreads();  // ✅ 所有线程都会执行到这里，只是部分线程跳过了 if 内的计算
}
```

关键区别：`__syncthreads()` 放在 `if` **外面**，所有线程都能到达。

### 禁忌 2：过度使用的性能损耗

`__syncthreads()` 是重量级指令：
- **排空指令流水线**：SM 的调度器失去灵活性，无法交错执行其他指令
- **全 Block 等待**：最慢的线程决定了所有线程的速度

**量化影响**（v3 vs 未来的 v6）：

```
v3 的归约循环（blockDim.x=256）：
  i=128: 计算 + __syncthreads()  ← 必要
  i=64:  计算 + __syncthreads()  ← 必要
  i=32:  计算 + __syncthreads()  ← 必要
  i=16:  计算 + __syncthreads()  ← 可省！同一 Warp 内天然同步
  i=8:   计算 + __syncthreads()  ← 可省！
  i=4:   计算 + __syncthreads()  ← 可省！
  i=2:   计算 + __syncthreads()  ← 可省！
  i=1:   计算 + __syncthreads()  ← 可省！
```

当 `i < 32` 时，所有活跃线程都在同一个 Warp 内。**Warp 内 32 个线程是 SIMT 锁步执行的**，天然同步，不需要显式屏障。

这就是 **v6（Warp Unrolling）** 的优化方向：展开最后 5 轮循环，省掉 5 次 `__syncthreads()` 调用。

---

## 面试速记

| 问题 | 答案 |
|------|------|
| `__syncthreads()` 的作用域？ | 仅当前 Block，跨 Block 无效 |
| 除了同步还做什么？ | 内存屏障，确保写入对 Block 内所有线程可见 |
| 能放在 if 里吗？ | 只有当所有线程都能到达时才行，否则死锁 |
| 为什么想减少它的使用？ | 排空流水线、阻塞调度，是性能热点 |
| 什么时候可以不用它？ | 同一 Warp 内的线程（SIMT 锁步），即 `i < 32` 时 |
| v6 怎么优化？ | 展开最后 5 轮（i=16→1），利用 Warp 内隐式同步省掉 `__syncthreads()` |
