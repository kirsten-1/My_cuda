# PyTorch 中的 CUDA Kernel 性能分析基础

## 一句话总结

> 在 PyTorch 中测量 GPU 代码性能，不能用 Python 的 `time` 模块，必须用 `torch.cuda.Event` 或 `torch.profiler`，因为 CUDA 是异步执行的。

---

## 核心概念一：为什么不能用 Python `time` 模块？

### CUDA 的异步执行特性

```python
import time
import torch

a = torch.randn(10000, 10000).cuda()

start = time.time()
result = torch.square(a)  # ← 这行代码立刻返回！
end = time.time()

print(f"Time: {end - start}")  # ← 测到的是 CPU 提交任务的时间，不是 GPU 执行时间
```

**问题所在：**
- PyTorch 的 CUDA 操作是**异步（Asynchronous）**的
- `torch.square(a)` 只是把任务提交到 GPU 的命令队列，CPU 立刻继续执行
- GPU 可能还在后台慢慢计算，但 Python 已经记录了 `end` 时间

**结果：** 测到的时间可能只有几微秒，完全不准确。

---

## 核心概念二：正确的计时方法 — `torch.cuda.Event`

### 使用 CUDA Event 同步计时

```python
import torch

def time_pytorch_function(func, input):
    # 创建 CUDA 事件对象
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Warmup：让 GPU 预热，避免首次执行的初始化开销
    for _ in range(5):
        func(input)

    # 在 GPU 命令流中插入开始标记
    start.record()
    func(input)
    # 在 GPU 命令流中插入结束标记
    end.record()
    
    # 等待 GPU 完成所有操作
    torch.cuda.synchronize()
    
    # 返回两个事件之间的实际 GPU 执行时间（毫秒）
    return start.elapsed_time(end)

b = torch.randn(10000, 10000).cuda()
print(f"torch.square 耗时: {time_pytorch_function(torch.square, b)} ms")
```

**关键点：**
1. **`start.record()` / `end.record()`**：在 GPU 的命令流中插入时间戳标记
2. **`torch.cuda.synchronize()`**：阻塞 CPU，等待 GPU 完成所有已提交的任务
3. **`start.elapsed_time(end)`**：返回两个标记之间 GPU 的实际执行时间

---

## 核心概念三：使用 `torch.profiler` 深度分析

### 基础用法

```python
import torch
from torch.profiler import profile, ProfilerActivity

b = torch.randn(10000, 10000).cuda()

with torch.profiler.profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
) as prof:
    torch.square(b)

# 打印性能报告，按 CUDA 时间排序
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

**输出示例：**
```
---------------------------------  ------------  ------------  ------------  
                             Name    Self CPU %      Self CPU   Self CUDA %  
---------------------------------  ------------  ------------  ------------  
                     aten::square        15.2%       1.234ms        85.3%
                      aten::mul_         8.1%       0.654ms        14.7%
...
```

### 高级用法：warmup + 导出 Chrome Trace

```python
def trace_handler(prof):
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
    prof.export_chrome_trace(f"/tmp/trace_{prof.step_num}.json")

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(
        wait=1,      # 跳过第 1 次迭代
        warmup=1,    # 第 2 次迭代预热
        active=2,    # 第 3-4 次迭代记录
        repeat=1     # 重复 1 次
    ),
    on_trace_ready=trace_handler
) as p:
    for iter in range(10):
        torch.square(torch.randn(10000, 10000).cuda())
        p.step()  # 通知 profiler 进入下一次迭代
```

**关键参数：**
- **`wait`**：跳过前 N 次迭代（避免初始化噪声）
- **`warmup`**：预热 N 次（让 GPU 进入稳定状态）
- **`active`**：实际记录 N 次迭代的性能数据
- **`export_chrome_trace`**：导出 JSON 文件，可在 Chrome 浏览器 `chrome://tracing` 中可视化

---

## 核心概念四：比较不同实现的性能

### 三种平方操作的性能对比

```python
b = torch.randn(10000, 10000).cuda()

def square_mul(a):
    return a * a

def square_pow(a):
    return a ** 2

# 测试三种方法
print(f"torch.square: {time_pytorch_function(torch.square, b)} ms")
print(f"a * a:        {time_pytorch_function(square_mul, b)} ms")
print(f"a ** 2:       {time_pytorch_function(square_pow, b)} ms")
```

**典型结果：**
- `torch.square` 和 `a * a` 性能接近（都调用优化的 CUDA kernel）
- `a ** 2` 可能稍慢（涉及更通用的幂运算路径）

---

## 面试速记

| 问题 | 答案 |
|------|------|
| 为什么不能用 `time.time()` 测 GPU 代码？ | CUDA 是异步的，CPU 提交任务后立刻返回，测到的是提交时间而非执行时间 |
| 如何正确测量 PyTorch GPU 操作的时间？ | 用 `torch.cuda.Event` 记录开始/结束标记，调用 `synchronize()` 等待完成 |
| `torch.cuda.synchronize()` 的作用？ | 阻塞 CPU 线程，等待 GPU 完成所有已提交的 CUDA 操作 |
| `torch.profiler` 相比手动计时的优势？ | 能看到每个算子的详细耗时、CPU/GPU 时间分布、内存占用，还能导出可视化 trace |
| 为什么需要 warmup？ | 首次执行有 CUDA 初始化、kernel 编译、缓存预热等开销，会严重干扰测量结果 |
| Chrome Trace 怎么查看？ | 在 Chrome 浏览器地址栏输入 `chrome://tracing`，加载导出的 JSON 文件 |

---

## 实战建议

1. **日常快速测试**：用 `torch.cuda.Event` 手动计时
2. **深度性能分析**：用 `torch.profiler` + Chrome Trace 可视化
3. **对比实验**：始终做 warmup，多次测量取中位数
4. **生产环境**：用 NVIDIA Nsight Systems / Nsight Compute 做更底层的分析
