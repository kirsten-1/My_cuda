# PyTorch 中内联加载 CUDA Kernel

## 一句话总结

> 用 `torch.utils.cpp_extension.load_inline` 可以在 Python 脚本里直接写 CUDA 代码，PyTorch 会自动编译并加载成可调用的扩展模块。

---

## 核心概念一：为什么需要 `load_inline`？

### 传统方式的痛点

**方式 1：纯 PyTorch 操作**
```python
result = input * input  # 简单但性能可能不是最优
```
- 优点：简单易用
- 缺点：无法实现自定义融合操作，某些场景性能受限

**方式 2：编写独立的 C++/CUDA 扩展**
```bash
# 需要写 setup.py，运行 python setup.py install
# 修改代码后需要重新编译安装
```
- 优点：性能最优
- 缺点：开发流程繁琐，调试困难

**方式 3：`load_inline` — 最佳实践**
```python
# 在 Python 文件里直接写 CUDA 代码
# 自动编译、缓存、加载
```
- 优点：开发效率高，修改即生效
- 缺点：首次编译需要时间

---

## 核心概念二：`load_inline` 的基本用法

### 完整示例：矩阵平方运算

```python
import torch
from torch.utils.cpp_extension import load_inline

# 定义 CUDA kernel
cuda_source = '''
__global__ void square_matrix_kernel(const float* matrix, float* result, int width, int height) {
    // 计算当前线程负责的行列索引
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // 边界检查
    if (row < height && col < width) {
        int idx = row * width + col;
        result[idx] = matrix[idx] * matrix[idx];
    }
}

// C++ 包装函数：从 PyTorch 调用 CUDA kernel
torch::Tensor square_matrix(torch::Tensor matrix) {
    const auto height = matrix.size(0);
    const auto width = matrix.size(1);

    auto result = torch::empty_like(matrix);

    // 配置线程块和网格
    dim3 threads_per_block(16, 16);  // 每个 block 有 16x16=256 个线程
    dim3 number_of_blocks(
        (width + threads_per_block.x - 1) / threads_per_block.x,
        (height + threads_per_block.y - 1) / threads_per_block.y
    );

    // 启动 kernel
    square_matrix_kernel<<<number_of_blocks, threads_per_block>>>(
        matrix.data_ptr<float>(), 
        result.data_ptr<float>(), 
        width, 
        height
    );

    return result;
}
'''

# C++ 函数声明
cpp_source = "torch::Tensor square_matrix(torch::Tensor matrix);"

# 编译并加载扩展
square_matrix_extension = load_inline(
    name='square_matrix_extension',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['square_matrix'],
    with_cuda=True,
    extra_cuda_cflags=["-O2"],
    build_directory='./load_inline_cuda',
)

# 使用自定义 kernel
a = torch.tensor([[1., 2., 3.], [4., 5., 6.]], device='cuda')
result = square_matrix_extension.square_matrix(a)
print(result)
# 输出: tensor([[ 1.,  4.,  9.],
#               [16., 25., 36.]], device='cuda:0')
```

---

## 核心概念三：关键参数详解

### `load_inline` 参数说明

```python
load_inline(
    name='module_name',              # 扩展模块的名称
    cpp_sources=cpp_code,            # C++ 代码（函数声明）
    cuda_sources=cuda_code,          # CUDA 代码（kernel 实现）
    functions=['func1', 'func2'],    # 要导出的函数列表
    with_cuda=True,                  # 启用 CUDA 支持
    extra_cuda_cflags=["-O2"],       # 额外的编译选项
    build_directory='./build',       # 编译缓存目录
)
```

**重要参数：**
- **`name`**：生成的 Python 模块名，后续通过 `module.function_name()` 调用
- **`cpp_sources`**：C++ 函数声明，定义 Python 可见的接口
- **`cuda_sources`**：CUDA kernel 实现 + C++ 包装函数
- **`functions`**：要暴露给 Python 的函数名列表
- **`build_directory`**：编译产物缓存目录，避免重复编译

---

## 核心概念四：CUDA Kernel 的线程配置

### 二维网格配置详解

```cuda
// 线程块配置：16x16 = 256 个线程
dim3 threads_per_block(16, 16);

// 网格配置：计算需要多少个 block 才能覆盖整个矩阵
dim3 number_of_blocks(
    (width + threads_per_block.x - 1) / threads_per_block.x,   // x 方向的 block 数
    (height + threads_per_block.y - 1) / threads_per_block.y   // y 方向的 block 数
);
```

**计算逻辑：**
- 假设矩阵是 100x100，线程块是 16x16
- x 方向需要：`(100 + 16 - 1) / 16 = 7` 个 block
- y 方向需要：`(100 + 16 - 1) / 16 = 7` 个 block
- 总共启动 7x7=49 个 block，每个 block 有 256 个线程

**线程索引计算：**
```cuda
int row = blockIdx.y * blockDim.y + threadIdx.y;  // 全局行索引
int col = blockIdx.x * blockDim.x + threadIdx.x;  // 全局列索引
```

**边界检查的必要性：**
```cuda
if (row < height && col < width) {
    // 因为 7*16=112 > 100，会有多余的线程
    // 必须检查边界，避免越界访问
}
```

---

## 核心概念五：编译缓存机制

### 首次编译 vs 后续加载

**首次运行：**
```bash
$ python load_inline.py
Emitting ninja build file /path/to/load_inline_cuda/build.ninja...
Building extension module square_matrix_extension...
[1/2] c++ -c main.cpp -o main.o
[2/2] nvcc -c cuda.cu -o cuda.cuda.o
Linking...
tensor([[ 1.,  4.,  9.],
        [16., 25., 36.]], device='cuda:0')
```
- 生成 `.ninja` 构建文件
- 编译 C++ 和 CUDA 代码
- 链接生成共享库

**后续运行：**
```bash
$ python load_inline.py
Loading extension module square_matrix_extension...
tensor([[ 1.,  4.,  9.],
        [16., 25., 36.]], device='cuda:0')
```
- 直接加载已编译的缓存
- 启动速度快

**触发重新编译的情况：**
- 修改了 `cuda_sources` 或 `cpp_sources`
- 删除了 `build_directory`
- 更改了编译选项（如 `extra_cuda_cflags`）

---

## 面试速记

| 问题 | 答案 |
|------|------|
| `load_inline` 的主要用途？ | 在 Python 脚本中直接编写和加载 CUDA kernel，无需单独的编译步骤 |
| 为什么需要 `cpp_sources` 和 `cuda_sources` 两个参数？ | `cpp_sources` 定义 Python 可见的接口，`cuda_sources` 包含 CUDA kernel 实现 |
| `dim3(16, 16)` 表示什么？ | 一个线程块包含 16x16=256 个线程，适合处理二维数据（如矩阵） |
| 为什么 kernel 里需要边界检查？ | 网格大小是向上取整的，会有多余线程，必须防止越界访问 |
| 修改 CUDA 代码后需要重新运行吗？ | 是的，但 `load_inline` 会自动检测变化并重新编译 |
| 如何优化编译速度？ | 指定 `build_directory` 启用缓存，避免每次都重新编译 |

---

## 实战建议

1. **开发阶段**：用 `load_inline` 快速迭代，修改代码立即生效
2. **生产部署**：考虑预编译成独立的扩展包，避免运行时编译开销
3. **调试技巧**：添加 `extra_cuda_cflags=["-G"]` 启用调试符号
4. **性能优化**：用 `extra_cuda_cflags=["-O3", "--use_fast_math"]` 开启激进优化
5. **兼容性**：确保 CUDA 版本与 PyTorch 编译时使用的版本一致
