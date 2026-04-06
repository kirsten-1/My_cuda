# CUDA Mode Lecture 1 总结：从 PyTorch 到自定义 CUDA Kernel

## 一句话总结

> 学会用 profiling 工具找到性能瓶颈，然后用 Triton/Numba/CUDA 写自定义 kernel 来优化，最后用 `load_inline` 无缝集成到 PyTorch。

---

## 核心主题：打破 CUDA Tutorial Hell

### 什么是 Tutorial Hell？

**症状：**
- 看了无数 CUDA 教程，还是不知道怎么在实际项目中用
- 理解了 GPU 架构，但不知道如何诊断性能问题
- 会写简单的 kernel，但不知道何时需要自定义 kernel

**本课程的解决方案：**
1. **先学会测量**：用 profiling 工具找到真正的瓶颈
2. **再学会优化**：掌握多种编写 GPU 代码的方式
3. **最后学会集成**：把自定义 kernel 无缝接入 PyTorch

---

## 学习路径：从易到难的 GPU 编程方式

### Level 1：PyTorch 原生操作

```python
import torch

a = torch.randn(10000, 10000).cuda()
result = torch.square(a)  # 使用 PyTorch 内置算子
```

**优点：**
- 简单易用，无需了解底层
- PyTorch 团队已经优化过

**缺点：**
- 无法实现自定义融合操作
- 某些特殊场景性能不是最优

**何时使用：**
- 90% 的情况下，PyTorch 原生操作已经足够

---

### Level 2：Triton — 高层 GPU 编程

```python
import triton
import triton.language as tl

@triton.jit
def square_kernel(output_ptr, input_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = input_ptr + row_idx * n_cols + col_offsets
    
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=0)
    square_output = row * row
    
    output_ptrs = output_ptr + row_idx * n_cols + col_offsets
    tl.store(output_ptrs, square_output, mask=col_offsets < n_cols)
```

**优点：**
- Python 语法，易于学习
- 编译器自动优化内存访问
- 适合快速实现融合算子

**缺点：**
- 性能上限不如手写 CUDA
- 某些高级特性不支持

**何时使用：**
- 需要融合多个操作（如 LayerNorm、FlashAttention）
- 研究新算法，需要快速原型开发

---

### Level 3：Numba CUDA — 学习 CUDA 编程模型

```python
from numba import cuda

@cuda.jit
def square_matrix_kernel(matrix, result):
    row, col = cuda.grid(2)
    if row < matrix.shape[0] and col < matrix.shape[1]:
        result[row, col] = matrix[row, col] ** 2
```

**优点：**
- 纯 Python，无需 C++ 编译
- 直接对应 CUDA 编程模型，适合教学
- 快速验证想法

**缺点：**
- 性能不如手写 CUDA
- 生态不如 CUDA 成熟

**何时使用：**
- 学习 CUDA 编程概念
- 快速原型开发
- 教学演示

---

### Level 4：CUDA C++ with `load_inline` — 最高性能

```python
from torch.utils.cpp_extension import load_inline

cuda_source = '''
__global__ void square_matrix_kernel(const float* matrix, float* result, 
                                     int width, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < height && col < width) {
        int idx = row * width + col;
        result[idx] = matrix[idx] * matrix[idx];
    }
}

torch::Tensor square_matrix(torch::Tensor matrix) {
    const auto height = matrix.size(0);
    const auto width = matrix.size(1);
    auto result = torch::empty_like(matrix);
    
    dim3 threads_per_block(16, 16);
    dim3 number_of_blocks(
        (width + threads_per_block.x - 1) / threads_per_block.x,
        (height + threads_per_block.y - 1) / threads_per_block.y
    );
    
    square_matrix_kernel<<<number_of_blocks, threads_per_block>>>(
        matrix.data_ptr<float>(), result.data_ptr<float>(), width, height
    );
    
    return result;
}
'''

cpp_source = "torch::Tensor square_matrix(torch::Tensor matrix);"

square_ext = load_inline(
    name='square_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['square_matrix'],
    with_cuda=True,
    extra_cuda_cflags=["-O2"],
    build_directory='./build',
)

result = square_ext.square_matrix(input_tensor)
```

**优点：**
- 性能最优，完全控制硬件
- 支持所有 CUDA 特性
- 在 Python 中直接编写，无需独立编译

**缺点：**
- 需要学习 C++/CUDA
- 开发周期较长

**何时使用：**
- 性能关键路径
- 需要使用高级 CUDA 特性（如 Tensor Core、Cooperative Groups）
- 生产环境部署

---

## 性能分析工具链

### 工具选择矩阵

| 工具 | 用途 | 适用场景 |
|------|------|----------|
| `torch.cuda.Event` | 快速测量单个操作耗时 | 日常开发，对比不同实现 |
| `torch.profiler` | 查看 PyTorch 算子级别性能 | 定位慢算子，优化 Python 代码 |
| Nsight Systems | 系统级时间线分析 | 发现 CPU-GPU 同步、数据传输瓶颈 |
| Nsight Compute | Kernel 级硬件指标分析 | 深度优化单个 kernel |

### 典型分析流程

```
1. 用 torch.profiler 找到最慢的算子
   ↓
2. 用 Nsight Systems 查看是否有 CPU-GPU 同步问题
   ↓
3. 用 Nsight Compute 分析 kernel 是计算受限还是内存受限
   ↓
4. 根据瓶颈类型选择优化策略
```

---

## 实战决策树：何时需要自定义 Kernel？

### 决策流程

```
问题：代码运行慢
  ↓
用 torch.profiler 分析
  ↓
是否有单个算子占用 > 30% 时间？
  ├─ 否 → 优化 Python 代码逻辑
  └─ 是 → 继续
       ↓
该算子是否可以与其他算子融合？
  ├─ 是 → 考虑用 Triton 实现融合 kernel
  └─ 否 → 继续
       ↓
PyTorch 是否有更高效的替代算子？
  ├─ 是 → 使用 PyTorch 原生实现
  └─ 否 → 继续
       ↓
用 Nsight Compute 分析瓶颈
  ├─ 内存受限 → 优化内存访问模式（合并访问、使用共享内存）
  ├─ 计算受限 → 增加并行度或使用 Tensor Core
  └─ 占用率低 → 减少寄存器/共享内存使用
```

---

## 关键概念速记

### CUDA 异步执行

```python
# ❌ 错误：用 time.time() 测量 GPU 代码
start = time.time()
result = torch.square(a)  # 立刻返回，GPU 还在后台执行
end = time.time()  # 测到的是 CPU 提交任务的时间

# ✅ 正确：用 CUDA Event
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
result = torch.square(a)
end.record()
torch.cuda.synchronize()  # 等待 GPU 完成
elapsed = start.elapsed_time(end)  # 真实的 GPU 执行时间
```

### 内存合并访问

```python
# ❌ 非合并访问（stride 访问）
for i in range(n):
    output[i] = input[i * 1000]  # 相邻线程访问相距很远的内存

# ✅ 合并访问（连续访问）
for i in range(n):
    output[i] = input[i]  # 相邻线程访问相邻内存
```

### 线程配置原则

```python
# 每个 block 的线程数应该是 32 的倍数（warp size）
threads_per_block = (16, 16)  # 256 个线程 ✅
threads_per_block = (15, 15)  # 225 个线程 ❌

# 总线程数应该远大于数据量，以隐藏延迟
total_threads = blocks * threads_per_block
# 理想情况：total_threads >= data_size * 2
```

---

## 面试高频问题

**Q1：什么时候应该写自定义 CUDA kernel？**
**答：** 三种情况：(1) 需要融合多个操作减少内存访问；(2) PyTorch 没有对应的高效实现；(3) 需要使用特殊硬件特性（如 Tensor Core）。但 90% 的情况下，PyTorch 原生操作已经足够。

**Q2：Triton 和 CUDA 的主要区别是什么？**
**答：** Triton 是更高层的抽象，以"程序块"为单位思考，编译器自动优化内存访问；CUDA 需要手动管理每个线程的行为。Triton 开发效率高但性能上限略低，CUDA 性能最优但开发周期长。

**Q3：如何判断一个 kernel 是计算受限还是内存受限？**
**答：** 用 Nsight Compute 查看 Speed of Light 指标。如果 Compute (SM) 利用率高（>80%）而 Memory 利用率低，则是计算受限；反之则是内存受限。

**Q4：为什么不能用 Python 的 `time` 模块测量 GPU 代码？**
**答：** 因为 CUDA 是异步执行的，PyTorch 的 GPU 操作只是把任务提交到命令队列就立刻返回，CPU 继续执行。必须用 `torch.cuda.Event` 或 `torch.cuda.synchronize()` 来正确测量 GPU 的实际执行时间。

**Q5：`load_inline` 相比传统 C++ 扩展的优势是什么？**
**答：** 可以在 Python 文件中直接编写 CUDA 代码，PyTorch 自动编译和缓存，修改代码后无需手动重新编译。开发效率高，适合快速迭代。

---

## 学习建议

### 推荐学习路径

1. **第 1-2 周：掌握 profiling**
   - 熟练使用 `torch.profiler` 和 `torch.cuda.Event`
   - 学会用 Nsight Systems 查看时间线
   - 理解 CUDA 异步执行模型

2. **第 3-4 周：Triton 入门**
   - 实现简单的 element-wise 操作
   - 尝试融合多个操作（如 GELU + Dropout）
   - 对比 Triton 和 PyTorch 的性能

3. **第 5-6 周：Numba CUDA**
   - 理解 CUDA 线程模型（grid, block, thread）
   - 实现矩阵乘法、归约等经典算法
   - 学习共享内存和同步

4. **第 7-8 周：CUDA C++ + load_inline**
   - 学习 CUDA 内存层次（全局、共享、寄存器）
   - 优化内存访问模式（合并访问、bank conflict）
   - 用 Nsight Compute 深度分析

### 推荐资源

- **书籍：** *Programming Massively Parallel Processors* (PMPP)
- **课程：** CUDA Mode 系列讲座
- **实践：** 实现 PyTorch 中的经典算子（如 LayerNorm、Softmax）
- **工具：** 熟练使用 NVIDIA Nsight 工具链

---

## 总结

本讲座的核心思想：**不要陷入无休止的教程学习，而是要建立"发现问题 → 分析问题 → 解决问题"的完整工作流。**

1. **发现问题**：用 profiling 工具找到真正的瓶颈
2. **分析问题**：用 Nsight 工具诊断是计算、内存还是同步问题
3. **解决问题**：根据场景选择合适的工具（Triton/Numba/CUDA）

记住：**过早优化是万恶之源。先让代码跑起来，再用数据驱动优化。**
