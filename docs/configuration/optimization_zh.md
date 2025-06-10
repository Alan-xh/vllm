# 优化与调优

本指南涵盖了 vLLM V1 的优化策略和性能调优。

## 抢占（Preemption）

由于变换器架构的自回归特性，有时 KV 缓存空间不足以处理所有批处理请求。在这种情况下，vLLM 可以抢占请求以释放 KV 缓存空间供其他请求使用。被抢占的请求会在足够的 KV 缓存空间可用时重新计算。当发生这种情况时，你可能会看到以下警告：

```text
WARNING 05-09 00:49:33 scheduler.py:1057 序列组 0 被 PreemptionMode.RECOMPUTE 模式抢占，因为 KV 缓存空间不足。这可能会影响端到端性能。增加 gpu_memory_utilization 或 tensor_parallel_size 以提供更多 KV 缓存内存。total_cumulative_preemption_cnt=1
```

虽然这种机制确保了系统鲁棒性，但抢占和重新计算可能会对端到端延迟产生不利影响。如果经常遇到抢占，可以考虑以下操作：

- 增加 `gpu_memory_utilization`。vLLM 使用此百分比预分配 GPU 缓存。通过增加利用率，可以提供更多 KV 缓存空间。
- 减少 `max_num_seqs` 或 `max_num_batched_tokens`。这会减少批处理中并发请求的数量，从而减少所需的 KV 缓存空间。
- 增加 `tensor_parallel_size`。这会将模型权重分片到多个 GPU 上，使每个 GPU 有更多内存可用于 KV 缓存。但增加此值可能会导致过多的同步开销。
- 增加 `pipeline_parallel_size`。这会将模型层分布到多个 GPU 上，减少每个 GPU 上模型权重所需的内存，间接为 KV 缓存留出更多内存。但增加此值可能会导致延迟惩罚。

你可以通过 vLLM 暴露的 Prometheus 指标监控抢占请求的数量。此外，可以通过设置 `disable_log_stats=False` 记录累计抢占请求数。

在 vLLM V1 中，默认抢占模式是 `RECOMPUTE` 而非 `SWAP`，因为在 V1 架构中重新计算的开销较低。

[](){ #chunked-prefill }

## 分块预填充（Chunked Prefill）

分块预填充允许 vLLM 将大型预填充请求分成较小的块，并与解码请求一起批处理。此功能通过更好地平衡计算密集型（预填充）和内存密集型（解码）操作，有助于提高吞吐量和降低延迟。

在 vLLM V1 中，**分块预填充默认始终启用**。这与 vLLM V0 不同，在 V0 中它是根据模型特性有条件启用的。

启用分块预填充后，调度策略会优先处理解码请求。它会先批处理所有待处理的解码请求，然后再调度预填充操作。当 `max_num_batched_tokens` 预算中有可用令牌时，它会调度待处理的预填充请求。如果待处理的预填充请求无法适应 `max_num_batched_tokens`，它会自动将其分块。

此策略有两个好处：

- 它提高了 ITL（inter-token latency）和生成解码速度，因为解码请求被优先处理。
- 通过将计算密集型（预填充）和内存密集型（解码）请求分配到同一批次中，有助于实现更好的 GPU 利用率。

### 使用分块预填充进行性能调优

你可以通过调整 `max_num_batched_tokens` 来调优性能：

- 较小的值（例如 2048）可以实现更好的 ITL，因为较少的预填充会减慢解码速度。
- 较高的值可以实现更好的首次令牌时间（TTFT），因为可以在一个批次中处理更多预填充令牌。
- 为获得最佳吞吐量，我们建议将 `max_num_batched_tokens > 8096`，特别是对于大型 GPU 上的较小模型。
- 如果 `max_num_batched_tokens` 与 `max_model_len` 相同，这几乎等同于 V0 的默认调度策略（除了仍优先处理解码）。

```python
from vllm import LLM

# 设置 max_num_batched_tokens 以调优性能
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", max_num_batched_tokens=16384)
```

有关更多详细信息，请参阅相关论文（<https://arxiv.org/pdf/2401.08671> 或 <https://arxiv.org/pdf/2308.16369>）。

## 并行策略

vLLM 支持多种并行策略，可以组合使用以优化不同硬件配置下的性能。

### 张量并行（Tensor Parallelism, TP）

张量并行将模型参数分片到每个模型层的多个 GPU 上。这是单节点内大型模型推理的最常见策略。

**使用场景：**

- 当模型太大无法适应单个 GPU 时
- 当需要减少每个 GPU 的内存压力以提供更多 KV 缓存空间以提高吞吐量时

```python
from vllm import LLM

# 将模型分片到 4 个 GPU 上
llm = LLM(model="meta-llama/Llama-3.3-70B-Instruct", tensor_parallel_size=4)
```

对于无法适应单个 GPU 的模型（例如 70B 参数模型），张量并行是必不可少的。

### 流水线并行（Pipeline Parallelism, PP）

流水线并行将模型层分布到多个 GPU 上。每个 GPU 按顺序处理模型的不同部分。

**使用场景：**

- 当你已经最大化了高效的张量并行但仍需进一步分布模型，或跨节点分布时
- 对于非常深且窄的模型，层分布比张量分片更有效

流水线并行可以与张量并行结合用于非常大的模型：

```python
from vllm import LLM

# 结合流水线并行和张量并行
llm = LLM(
    model="meta-llama/Llama-3.3-70B-Instruct",
    tensor_parallel_size=4,
    pipeline_parallel_size=2
)
```

### 专家并行（Expert Parallelism, EP）

专家并行是针对混合专家（MoE）模型的特殊并行形式，其中不同的专家网络分布在多个 GPU 上。

**使用场景：**

- 专门用于 MoE 模型（例如 DeepSeekV3、Qwen3MoE、Llama-4）
- 当你想在 GPU 之间平衡专家计算负载时

通过设置 `enable_expert_parallel=True` 启用专家并行，它将对 MoE 层使用专家并行而不是张量并行。它将使用与张量并行设置相同的并行度。

### 数据并行（Data Parallelism, DP）

数据并行将整个模型复制到多个 GPU 集上，并并行处理不同的请求批次。

**使用场景：**

- 当你有足够的 GPU 来复制整个模型时
- 当你需要扩展吞吐量而不是模型大小时
- 在多用户环境中，请求批次之间的隔离有益时

数据并行可以与其他并行策略结合使用，并通过 `data_parallel_size=N` 设置。注意，MoE 层将根据张量并行大小和数据并行大小的乘积进行分片。

## 减少内存使用

如果遇到内存不足问题，可以考虑以下策略：

### 上下文长度和批次大小

你可以通过限制上下文长度和批次大小来减少内存使用：

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    max_model_len=2048,  # 限制上下文窗口
    max_num_seqs=4       # 限制批次大小
)
```

### 调整 CUDA 图编译

V1 中的 CUDA 图编译比 V0 使用更多内存。你可以通过调整编译级别来减少内存使用：

```python
from vllm import LLM
from vllm.config import CompilationConfig, CompilationLevel

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    compilation_config=CompilationConfig(
        level=CompilationLevel.PIECEWISE,
        cudagraph_capture_sizes=[1, 2, 4, 8]  # 捕获更少的批次大小
    )
)
```

或者，如果不关心延迟或整体性能，可以通过设置 `enforce_eager=True` 完全禁用 CUDA 图编译：

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    enforce_eager=True  # 禁用 CUDA 图编译
)
```

### 多模态模型

对于多模态模型，你可以通过限制每个请求的图像/视频数量来减少内存使用：

```python
from vllm import LLM

# 每个提示最多接受 2 张图像
llm = LLM(
    model="Qwen/Qwen2.5-VL-3B-Instruct",
    limit_mm_per_prompt={"image": 2}
)
```