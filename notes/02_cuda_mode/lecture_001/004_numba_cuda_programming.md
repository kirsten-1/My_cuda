# Numba CUDA 编程入门

## 一句话总结

> Numba 让你用 Python 装饰器直接编写 CUDA kernel，无需 C++，适合快速原型开发和教学。

---

## 核心概念一：Numba CUDA 的定位

### 三种 GPU 编程方式对比

| 特性 | CUDA C++ | Numba CUDA | Triton |
|------|----------|------------|--------|
| 语言 | C++/CUDA | Python | Python |
| 抽象层级 | 低（手动管理线程） | 低（手动管理线程） | 高（自动优化） |
| 性能上限 | 最高 | 中等 | 高 |
| 开发速度 | 慢 | 快 | 快 |
| 适用场景 | 生产环境 | 原型开发/教学 | 研究/融合算子 |

**Numba CUDA 的优势：**
- 纯 Python 语法，无需编译步骤
- 直接操作 NumPy 数组
- 适合学习 CUDA 编程模型

**Numba CUDA 的局限：**
- 性能不如手写 CUDA（缺少深度优化）
- 不支持某些高级 CUDA 特性
- 调试工具不如 CUDA 成熟

---

## 核心概念二：基本语法

### 完整示例：矩阵平方运算

```python
from numba import cuda
import numpy as np

# 使用 @cuda.jit 装饰器定义 CUDA kernel
@cuda.jit
def square_matrix_kernel(matrix, result):
    # 计算当前线程的全局索引
    row, col = cuda.grid(2)  # 2D 网格
    
    # 边界检查
    if row < matrix.shape[0] and col < matrix.shape[1]:
        # 执行计算
        result[row, col] = matrix[row, col] ** 2

# 创建输入数据
matrix = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

# 将数据传输到 GPU
d_matrix = cuda.to_device(matrix)
d_result = cuda.device_array(matrix.shape, dtype=np.float32)

# 配置线程块和网格
threads_per_block = (16, 16)  # 每个 block 有 16x16=256 个线程
blocks_per_grid_x = int(np.ceil(matrix.shape[0] / threads_per_block[0]))
blocks_per_grid_y = int(np.ceil(matrix.shape[1] / threads_per_block[1]))
blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

# 启动 kernel
square_matrix_kernel[blocks_per_grid, threads_per_block](d_matrix, d_result)

# 将结果传回 CPU
result = d_result.copy_to_host()

print("输入:")
print(matrix)
print("输出:")
print(result)
```

**输出：**
```
输入:
[[1. 2. 3.]
 [4. 5. 6.]]
输出:
[[ 1.  4.  9.]
 [16. 25. 36.]]
```

---

## 核心概念三：关键 API 详解

### 1. `@cuda.jit` 装饰器

```python
@cuda.jit
def my_kernel(input_array, output_array):
    # kernel 代码
    pass
```

**作用：**
- 将 Python 函数编译为 CUDA kernel
- 支持 JIT（即时编译），首次调用时编译

**可选参数：**
```python
@cuda.jit(device=True)  # 定义设备函数（被其他 kernel 调用）
def helper_function(x):
    return x * x

@cuda.jit
def main_kernel(data):
    idx = cuda.grid(1)
    data[idx] = helper_function(data[idx])
```

### 2. `cuda.grid(ndim)` — 计算全局索引

```python
# 1D 网格
idx = cuda.grid(1)
# 等价于 CUDA C++:
# int idx = blockIdx.x * blockDim.x + threadIdx.x;

# 2D 网格
row, col = cuda.grid(2)
# 等价于:
# int row = blockIdx.y * blockDim.y + threadIdx.y;
# int col = blockIdx.x * blockDim.x + threadIdx.x;

# 3D 网格
x, y, z = cuda.grid(3)
```

**内部实现：**
```python
# cuda.grid(2) 等价于：
row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
```

### 3. 内存管理 API

#### 主机到设备传输

```python
# 方式 1：复制数据到 GPU
host_array = np.array([1, 2, 3, 4], dtype=np.float32)
device_array = cuda.to_device(host_array)

# 方式 2：在 GPU 上分配空间（未初始化）
device_array = cuda.device_array(shape=(100, 100), dtype=np.float32)

# 方式 3：在 GPU 上分配并初始化为 0
device_array = cuda.device_array_like(host_array)
```

#### 设备到主机传输

```python
# 方式 1：复制回 CPU
result = device_array.copy_to_host()

# 方式 2：原地复制到已有数组
host_array = np.empty((100, 100), dtype=np.float32)
device_array.copy_to_host(host_array)
```

### 4. 线程索引 API

```python
@cuda.jit
def kernel(data):
    # 线程在 block 内的索引
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    tz = cuda.threadIdx.z
    
    # block 在 grid 内的索引
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    bz = cuda.blockIdx.z
    
    # block 的维度
    bdx = cuda.blockDim.x
    bdy = cuda.blockDim.y
    bdz = cuda.blockDim.z
    
    # grid 的维度
    gdx = cuda.gridDim.x
    gdy = cuda.gridDim.y
    gdz = cuda.gridDim.z
```

---

## 核心概念四：共享内存和同步

### 使用共享内存加速

```python
from numba import cuda

@cuda.jit
def sum_reduce_kernel(input_array, output_array):
    # 分配共享内存
    shared = cuda.shared.array(shape=(256,), dtype=np.float32)
    
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    bdim = cuda.blockDim.x
    
    # 加载数据到共享内存
    idx = bid * bdim + tid
    if idx < input_array.size:
        shared[tid] = input_array[idx]
    else:
        shared[tid] = 0
    
    # 同步：确保所有线程都加载完毕
    cuda.syncthreads()
    
    # 归约求和
    stride = bdim // 2
    while stride > 0:
        if tid < stride:
            shared[tid] += shared[tid + stride]
        cuda.syncthreads()  # 每轮归约后同步
        stride //= 2
    
    # 线程 0 写入结果
    if tid == 0:
        output_array[bid] = shared[0]
```

**关键点：**
- **`cuda.shared.array(shape, dtype)`**：分配共享内存
- **`cuda.syncthreads()`**：block 内线程同步（等价于 CUDA 的 `__syncthreads()`）

---

## 核心概念五：网格和线程块配置

### 计算合适的配置

```python
def configure_grid(data_shape, threads_per_block):
    """
    计算网格配置
    
    Args:
        data_shape: 数据维度，如 (1000, 2000)
        threads_per_block: 每个 block 的线程数，如 (16, 16)
    
    Returns:
        blocks_per_grid: 网格中 block 的数量
    """
    blocks_per_grid = tuple(
        int(np.ceil(data_dim / block_dim))
        for data_dim, block_dim in zip(data_shape, threads_per_block)
    )
    return blocks_per_grid

# 示例
matrix_shape = (1000, 2000)
threads_per_block = (16, 16)
blocks_per_grid = configure_grid(matrix_shape, threads_per_block)

print(f"数据维度: {matrix_shape}")
print(f"线程块: {threads_per_block}")
print(f"网格: {blocks_per_grid}")
# 输出:
# 数据维度: (1000, 2000)
# 线程块: (16, 16)
# 网格: (63, 125)
```

**计算逻辑：**
```
行方向：ceil(1000 / 16) = 63 个 block
列方向：ceil(2000 / 16) = 125 个 block
总共启动：63 * 125 = 7875 个 block
每个 block：16 * 16 = 256 个线程
总线程数：7875 * 256 = 2,016,000 个线程
```

---

## 核心概念六：性能优化技巧

### 1. 避免主机-设备传输

```python
# ❌ 低效：每次迭代都传输数据
for i in range(100):
    d_data = cuda.to_device(data)
    kernel[blocks, threads](d_data)
    result = d_data.copy_to_host()

# ✅ 高效：只传输一次
d_data = cuda.to_device(data)
for i in range(100):
    kernel[blocks, threads](d_data)
result = d_data.copy_to_host()
```

### 2. 使用合适的线程块大小

```python
# ❌ 太小：无法充分利用 GPU
threads_per_block = (4, 4)  # 只有 16 个线程

# ❌ 太大：超过硬件限制
threads_per_block = (64, 64)  # 4096 个线程，超过 1024 的限制

# ✅ 合适：256-512 个线程
threads_per_block = (16, 16)  # 256 个线程
threads_per_block = (32, 16)  # 512 个线程
```

### 3. 内存合并访问

```python
# ❌ 非合并访问（列优先）
@cuda.jit
def bad_kernel(matrix):
    row, col = cuda.grid(2)
    if row < matrix.shape[0] and col < matrix.shape[1]:
        # 相邻线程访问不连续的内存
        matrix[col, row] = matrix[col, row] * 2

# ✅ 合并访问（行优先）
@cuda.jit
def good_kernel(matrix):
    row, col = cuda.grid(2)
    if row < matrix.shape[0] and col < matrix.shape[1]:
        # 相邻线程访问连续的内存
        matrix[row, col] = matrix[row, col] * 2
```

---

## 面试速记

| 问题 | 答案 |
|------|------|
| Numba CUDA 相比 CUDA C++ 的优势？ | 纯 Python 语法，无需编译，开发速度快，适合原型开发 |
| `cuda.grid(2)` 的作用？ | 计算当前线程的全局 2D 索引 (row, col) |
| 如何在 Numba 中使用共享内存？ | `cuda.shared.array(shape, dtype)` 分配，`cuda.syncthreads()` 同步 |
| 为什么需要边界检查？ | 网格大小是向上取整的，会有多余线程，必须防止越界访问 |
| 如何优化主机-设备传输？ | 尽量减少传输次数，在 GPU 上保留数据，批量处理 |
| 推荐的线程块大小？ | 256-512 个线程（如 16x16 或 32x16），不超过 1024 |

---

## 实战建议

1. **学习路径**：先用 Numba 理解 CUDA 编程模型，再转向 CUDA C++ 或 Triton
2. **调试技巧**：使用 `print()` 在 kernel 中输出（但会严重影响性能）
3. **性能分析**：用 `cuda.profile_start()` / `cuda.profile_stop()` 配合 Nsight
4. **类型限制**：Numba 只支持有限的 Python 特性，避免使用复杂数据结构
5. **生产环境**：Numba 适合原型，生产环境考虑 CUDA C++ 或 Triton
