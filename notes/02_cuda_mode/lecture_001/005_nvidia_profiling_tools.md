# NVIDIA Profiling 工具链：Nsight Systems 与 Nsight Compute

## 一句话总结

> Nsight Systems 看整体时间线和瓶颈，Nsight Compute 深挖单个 kernel 的硬件指标。

---

## 核心概念一：两个工具的定位

### Nsight Systems (nsys) — 系统级性能分析

**用途：**
- 查看整个应用的时间线（CPU + GPU）
- 识别 CPU-GPU 同步问题
- 发现 kernel 启动开销
- 分析数据传输瓶颈

**典型问题：**
- "为什么 GPU 利用率只有 30%？"
- "CPU 和 GPU 之间有没有不必要的同步？"
- "数据传输占用了多少时间？"

### Nsight Compute (ncu) — Kernel 级性能分析

**用途：**
- 深度分析单个 kernel 的执行
- 查看 SM 占用率、内存带宽、指令吞吐
- 识别性能瓶颈（计算受限 vs 内存受限）
- 获取硬件计数器数据

**典型问题：**
- "这个 kernel 的内存带宽利用率是多少？"
- "有没有 bank conflict？"
- "寄存器使用是否过多导致占用率下降？"

---

## 核心概念二：Nsight Systems 基础用法

### 1. 命令行采集数据

```bash
# 基础用法：分析 Python 脚本
nsys profile -o output_report python script.py

# 常用参数
nsys profile \
  -o my_report \              # 输出文件名
  --trace=cuda,nvtx \         # 追踪 CUDA API 和 NVTX 标记
  --stats=true \              # 生成统计摘要
  python pytorch_square.py
```

**输出文件：**
- `my_report.nsys-rep`：二进制报告文件
- `my_report.sqlite`：数据库文件（可选）

### 2. 在 PyTorch 中使用

```python
import torch

# 示例：分析 square 操作
a = torch.randn(10000, 10000).cuda()

# 预热
for _ in range(5):
    torch.square(a)

# 开始分析
torch.cuda.synchronize()
result = torch.square(a)
torch.cuda.synchronize()
```

**运行：**
```bash
nsys profile -o square_profile python nsys_square.py
```

### 3. 查看报告

```bash
# 方式 1：命令行查看统计信息
nsys stats square_profile.nsys-rep

# 方式 2：GUI 查看（推荐）
nsys-ui square_profile.nsys-rep
```

**GUI 界面关键信息：**
- **Timeline**：时间轴视图，显示 CPU/GPU 活动
- **CUDA API Calls**：CUDA API 调用耗时
- **Kernel Launches**：kernel 启动和执行时间
- **Memory Operations**：数据传输（H2D/D2H）

---

## 核心概念三：Nsight Compute 基础用法

### 1. 命令行采集数据

```bash
# 基础用法：分析所有 kernel
ncu -o output_report python script.py

# 只分析特定 kernel
ncu --kernel-name square_matrix_kernel -o report python script.py

# 采集完整指标集（慢但详细）
ncu --set full -o report python script.py

# 采集特定指标
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed \
    -o report python script.py
```

### 2. 常用指标说明

| 指标 | 含义 | 理想值 |
|------|------|--------|
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | SM 吞吐率 | > 80% |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | 内存带宽利用率 | > 60% |
| `smsp__sass_thread_inst_executed_op_*` | 各类指令执行数 | 取决于算法 |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` | 全局内存加载扇区数 | 越少越好 |
| `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct` | 内存合并效率 | 100% |

### 3. 查看报告

```bash
# 方式 1：命令行查看摘要
ncu --import report.ncu-rep

# 方式 2：GUI 查看（推荐）
ncu-ui report.ncu-rep
```

**GUI 界面关键部分：**
- **Details**：kernel 基本信息（网格大小、寄存器使用等）
- **Speed of Light**：各子系统利用率（计算、内存、L1/L2 缓存）
- **Memory Workload Analysis**：内存访问模式分析
- **Compute Workload Analysis**：计算指令分析

---

## 核心概念四：实战案例分析

### 案例 1：使用 Nsight Systems 发现 CPU-GPU 同步问题

**问题代码：**
```python
import torch

a = torch.randn(1000, 1000).cuda()

for i in range(100):
    result = torch.square(a)
    # ❌ 隐式同步：访问 result 会触发 CPU 等待 GPU
    if result[0, 0] > 0:
        print("Positive")
```

**Nsight Systems 分析：**
```bash
nsys profile -o sync_issue python bad_code.py
nsys-ui sync_issue.nsys-rep
```

**时间线显示：**
```
CPU: |====kernel_launch====|----wait----|====kernel_launch====|----wait----|
GPU: |----compute----|                   |----compute----|
     ^               ^                   ^               ^
     启动            同步                启动            同步
```

**优化后：**
```python
for i in range(100):
    result = torch.square(a)
# 只在最后同步一次
torch.cuda.synchronize()
final_result = result.cpu()
```

### 案例 2：使用 Nsight Compute 分析内存瓶颈

**示例 kernel：**
```python
@cuda.jit
def inefficient_kernel(input, output):
    idx = cuda.grid(1)
    if idx < input.size:
        # ❌ 非合并访问：stride 访问
        output[idx] = input[idx * 1000]
```

**Nsight Compute 分析：**
```bash
ncu --set full -o memory_issue python inefficient_kernel.py
ncu-ui memory_issue.ncu-rep
```

**关键指标：**
```
Memory Workload Analysis:
  Global Load Efficiency: 12.5%  ← 内存合并效率极低
  L1 Cache Hit Rate: 5%          ← 缓存命中率低
  DRAM Throughput: 85%           ← 内存带宽已饱和

Speed of Light:
  Compute (SM): 15%              ← 计算单元空闲
  Memory: 95%                    ← 内存瓶颈！
```

**结论：** 该 kernel 是内存受限（Memory-bound），需要优化内存访问模式。

---

## 核心概念五：PyTorch Profiler 与 NVIDIA 工具的集成

### 导出 Chrome Trace 供 Nsight Systems 分析

```python
import torch
from torch.profiler import profile, ProfilerActivity

with torch.profiler.profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    with_stack=True
) as prof:
    for _ in range(10):
        torch.square(torch.randn(10000, 10000).cuda())

# 导出为 Chrome Trace 格式
prof.export_chrome_trace("trace.json")
```

**在 Chrome 中查看：**
1. 打开 Chrome 浏览器
2. 地址栏输入 `chrome://tracing`
3. 点击 "Load" 加载 `trace.json`

**优势：**
- 可视化 PyTorch 算子级别的时间线
- 查看 Python 调用栈
- 识别 Python 层面的性能瓶颈

---

## 核心概念六：常见性能问题诊断

### 问题 1：GPU 利用率低

**Nsight Systems 诊断：**
```
Timeline 显示：
GPU: |--kernel--|_______|--kernel--|_______|--kernel--|_______|
     ^          ^       ^          ^       ^          ^
     执行       空闲    执行       空闲    执行       空闲
```

**可能原因：**
1. Kernel 太小，启动开销占比高
2. CPU-GPU 同步过于频繁
3. 数据传输占用时间过多

**解决方案：**
- 合并多个小 kernel
- 使用异步操作和 CUDA Stream
- 减少不必要的 `.cpu()` 或 `.item()` 调用

### 问题 2：Kernel 执行慢

**Nsight Compute 诊断流程：**

**步骤 1：查看 Speed of Light**
```
Compute (SM): 25%
Memory: 90%
```
→ 内存受限，优化内存访问

```
Compute (SM): 85%
Memory: 30%
```
→ 计算受限，优化算法或增加并行度

**步骤 2：查看 Occupancy**
```
Theoretical Occupancy: 100%
Achieved Occupancy: 35%
```
→ 寄存器或共享内存使用过多，限制了并发

**步骤 3：查看 Memory Workload**
```
Global Load Efficiency: 25%
```
→ 内存访问未合并，需要调整访问模式

---

## 面试速记

| 问题 | 答案 |
|------|------|
| Nsight Systems 和 Nsight Compute 的区别？ | nsys 看整体时间线和系统瓶颈，ncu 深挖单个 kernel 的硬件指标 |
| 如何判断 kernel 是计算受限还是内存受限？ | 用 ncu 查看 Speed of Light，Compute 高则计算受限，Memory 高则内存受限 |
| GPU 利用率低的常见原因？ | Kernel 太小、CPU-GPU 同步频繁、数据传输过多 |
| 如何查看内存合并效率？ | ncu 的 Memory Workload Analysis 中的 Global Load/Store Efficiency |
| PyTorch 代码如何与 nsys 集成？ | 直接用 `nsys profile python script.py`，或导出 Chrome Trace |
| 什么是 Occupancy？ | SM 上实际活跃 warp 数占理论最大值的比例，影响延迟隐藏能力 |

---

## 实战建议

1. **开发流程：**
   - 先用 PyTorch Profiler 快速定位慢算子
   - 用 Nsight Systems 查看整体时间线
   - 用 Nsight Compute 深挖慢 kernel 的根因

2. **性能优化顺序：**
   - 消除 CPU-GPU 同步（最大收益）
   - 优化内存访问模式（合并访问）
   - 调整 kernel 配置（block size, occupancy）
   - 算法级优化（减少计算量）

3. **工具选择：**
   - 日常开发：PyTorch Profiler + Chrome Trace
   - 系统级优化：Nsight Systems
   - Kernel 级优化：Nsight Compute
   - 生产监控：集成 NVTX 标记 + 自动化分析
