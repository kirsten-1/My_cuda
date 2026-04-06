# Triton 编写 GPU Kernel 入门

## 一句话总结

> Triton 让你用类似 NumPy 的 Python 语法写 GPU kernel，编译器自动处理内存管理、线程调度和性能优化。

---

## 核心概念一：为什么需要 Triton？

### CUDA vs Triton 对比

**CUDA 的痛点：**
```cuda
// 需要手动管理：
// 1. 线程索引计算 (blockIdx, threadIdx)
// 2. 共享内存分配和同步
// 3. 内存合并访问优化
// 4. Bank conflict 避免
__global__ void square_kernel(float* out, const float* in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx] * in[idx];
    }
}
```

**Triton 的优势：**
```python
@triton.jit
def square_kernel(output_ptr, input_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)  # 自动处理线程块
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = input_ptr + row_idx * n_cols + col_offsets
    
    # 向量化加载，自动处理边界
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
    square_output = row * row
    
    # 向量化存储
    output_ptrs = output_ptr + row_idx * n_cols + col_offsets
    tl.store(output_ptrs, square_output, mask=col_offsets < n_cols)
```

**关键区别：**
- **抽象层级更高**：以"程序块（program）"为单位思考，而非单个线程
- **自动优化**：编译器处理内存合并、共享内存使用、寄存器分配
- **Python 语法**：无需学习 C++，直接在 PyTorch 代码中编写

---

## 核心概念二：Triton 的编程模型

### Program ID vs Thread ID

**CUDA 思维：**
```
每个线程处理一个元素
线程 0 → 元素 0
线程 1 → 元素 1
...
```

**Triton 思维：**
```
每个 program 处理一块数据（BLOCK_SIZE 个元素）
Program 0 → 元素 [0:BLOCK_SIZE]
Program 1 → 元素 [BLOCK_SIZE:2*BLOCK_SIZE]
...
```

### 示例：矩阵平方运算

```python
import triton
import triton.language as tl
import torch

@triton.jit
def square_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, 
                  n_cols, BLOCK_SIZE: tl.constexpr):
    # 获取当前 program 负责的行索引
    row_idx = tl.program_id(0)
    
    # 计算输入行的起始指针
    row_start_ptr = input_ptr + row_idx * input_row_stride
    
    # 生成列偏移量 [0, 1, 2, ..., BLOCK_SIZE-1]
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    
    # 向量化加载一整行数据
    # mask: 处理 BLOCK_SIZE > n_cols 的情况
    # other: 超出边界的位置填充值
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
    
    # 逐元素平方
    square_output = row * row
    
    # 向量化存储结果
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, square_output, mask=col_offsets < n_cols)
```

---

## 核心概念三：关键 API 详解

### 1. `tl.program_id(axis)`

```python
row_idx = tl.program_id(0)  # 获取当前 program 在第 0 维的索引
```
- 类似 CUDA 的 `blockIdx.x`
- 但 Triton 的 program 是更高层的抽象，一个 program 可能对应多个 CUDA block

### 2. `tl.arange(start, end)`

```python
col_offsets = tl.arange(0, BLOCK_SIZE)  # 生成 [0, 1, 2, ..., BLOCK_SIZE-1]
```
- 生成连续的整数向量
- 用于计算内存地址偏移

### 3. `tl.load(ptr, mask, other)`

```python
row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
```
- **`ptr`**：内存地址向量
- **`mask`**：布尔向量，指示哪些位置有效
- **`other`**：无效位置的填充值

**为什么需要 mask？**
```
假设矩阵有 781 列，BLOCK_SIZE=1024
最后一个 program 会尝试加载 [0:1024]，但只有 [0:781] 有效
mask 确保只加载有效数据，避免越界
```

### 4. `tl.store(ptr, value, mask)`

```python
tl.store(output_ptrs, square_output, mask=col_offsets < n_cols)
```
- 向量化存储，只写入 mask 为 True 的位置

### 5. `tl.constexpr`

```python
def square_kernel(..., BLOCK_SIZE: tl.constexpr):
```
- 标记编译期常量
- 编译器可以基于此做激进优化（如循环展开）

---

## 核心概念四：启动 Kernel

### Python 包装函数

```python
def square(x):
    n_rows, n_cols = x.shape
    
    # BLOCK_SIZE 必须是 2 的幂次方
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    
    # 根据 BLOCK_SIZE 调整 warp 数量
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    
    # 分配输出张量
    y = torch.empty_like(x)
    
    # 启动 kernel：每行一个 program
    square_kernel[(n_rows,)](  # grid size: (n_rows,)
        y,
        x,
        x.stride(0),  # 行步长（跨越一行需要的元素数）
        y.stride(0),
        n_cols,
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y
```

**关键参数：**
- **`square_kernel[(n_rows,)]`**：启动 `n_rows` 个 program，每个处理一行
- **`x.stride(0)`**：PyTorch 张量的步长，用于计算行起始地址
- **`num_warps`**：每个 program 使用的 warp 数量（性能调优参数）

---

## 核心概念五：性能基准测试

### 使用 Triton 的 Benchmark 工具

```python
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # x 轴：列数
        x_vals=[128 * i for i in range(2, 100)],  # 测试不同的矩阵大小
        line_arg='provider',  # 不同的实现方式
        line_vals=['triton', 'torch-native', 'torch-compile'],
        line_names=["Triton", "Torch (native)", "Torch (compiled)"],
        styles=[('blue', '-'), ('green', '-'), ('green', '--')],
        ylabel="GB/s",  # y 轴：带宽
        plot_name="square() performance",
        args={'M': 4096},  # 固定行数
    ))
def benchmark(M, N, provider):
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]  # 中位数、20%、80% 分位数
    
    if provider == 'torch-native':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.square(x), quantiles=quantiles
        )
    elif provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: square(x), quantiles=quantiles
        )
    elif provider == 'torch-compile':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.compile(torch.square)(x), quantiles=quantiles
        )
    
    # 计算带宽：(读 + 写) * 元素数 * 字节数 / 时间
    gbps = lambda ms: 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)

benchmark.run(show_plots=True, print_data=True, save_path='.')
```

**输出：**
- 自动生成性能对比图表
- 显示不同实现在不同数据规模下的带宽

---

## 面试速记

| 问题 | 答案 |
|------|------|
| Triton 相比 CUDA 的主要优势？ | 更高层的抽象，自动优化内存访问和线程调度，Python 语法易于集成 |
| `tl.program_id(0)` 对应 CUDA 的什么？ | 类似 `blockIdx.x`，但 Triton 的 program 是更高层的抽象 |
| 为什么需要 `mask` 参数？ | BLOCK_SIZE 通常是 2 的幂次方，可能大于实际数据大小，mask 防止越界访问 |
| `BLOCK_SIZE: tl.constexpr` 的作用？ | 标记编译期常量，让编译器做循环展开等优化 |
| 如何选择 `num_warps`？ | 根据 BLOCK_SIZE 调整：越大的 block 需要更多 warp 来隐藏延迟 |
| Triton 适合什么场景？ | 自定义融合算子、研究新算法、快速原型开发 |

---

## 实战建议

1. **学习路径**：先掌握 CUDA 基础，再学 Triton（理解底层有助于调优）
2. **性能调优**：用 `@triton.testing.perf_report` 对比不同配置
3. **调试技巧**：设置 `os.environ["TRITON_INTERPRET"] = "1"` 启用解释模式
4. **生产使用**：Triton 已被 PyTorch 2.0 的 `torch.compile` 后端采用
5. **参考资料**：Triton 官方教程（https://triton-lang.org/main/getting-started/tutorials/）
