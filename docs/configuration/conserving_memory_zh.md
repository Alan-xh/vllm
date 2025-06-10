# 内存优化

大型模型可能会导致机器内存不足（OOM）。以下是一些有助于缓解此问题的选项。

## 张量并行（TP）

张量并行（`tensor_parallel_size` 选项）可用于将模型拆分到多个 GPU 上。

以下代码将模型拆分到 2 个 GPU 上。

```python
from vllm import LLM

llm = LLM(model="ibm-granite/granite-3.1-8b-instruct",
          tensor_parallel_size=2)
```

!!! warning
    为确保 vLLM 正确初始化 CUDA，请避免在初始化 vLLM 之前调用相关函数（例如 [torch.cuda.set_device][]）。
    否则，可能会遇到类似 `RuntimeError: Cannot re-initialize CUDA in forked subprocess` 的错误。

    要控制使用哪些设备，请设置 `CUDA_VISIBLE_DEVICES` 环境变量。

!!! note
    启用张量并行后，每个进程将读取整个模型并将其拆分成块，这会使磁盘读取时间更长（与张量并行的大小成正比）。

    您可以使用 <gh-file:examples/offline_inference/save_sharded_state.py> 将模型检查点转换为分片检查点。转换过程可能需要一些时间，但之后加载分片检查点的速度会更快。模型加载时间应与张量并行的大小无关，保持恒定。

## 量化

量化模型会以较低精度为代价占用更少的内存。

静态量化模型可以从 HF Hub 下载（一些热门模型可在 [Red Hat AI](https://huggingface.co/RedHatAI) 找到），
无需额外配置即可直接使用。

动态量化也通过 `quantization` 选项支持——详情请见 [此处][quantization-index]。

## 上下文长度和批处理大小

您可以通过限制模型的上下文长度（`max_model_len` 选项）和最大批处理大小（`max_num_seqs` 选项）进一步减少内存使用。

```python
from vllm import LLM

llm = LLM(model="adept/fuyu-8b",
          max_model_len=2048,
          max_num_seqs=2)
```

## 减少 CUDA 图

默认情况下，我们使用 CUDA 图优化模型推理，这会占用额外的 GPU 内存。

!!! warning
    CUDA 图捕获在 V1 版本中比 V0 版本占用更多内存。

您可以调整 `compilation_config` 以在推理速度和内存使用之间取得更好的平衡：

```python
from vllm import LLM
from vllm.config import CompilationConfig, CompilationLevel

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    compilation_config=CompilationConfig(
        level=CompilationLevel.PIECEWISE,
        # 默认情况下，它会达到 max_num_seqs
        cudagraph_capture_sizes=[1, 2, 4, 8, 16],
    ),
)
```

您可以通过 `enforce_eager` 标志完全禁用图捕获：

```python
from vllm import LLM

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct",
          enforce_eager=True)
```

## 调整缓存大小

如果您的 CPU RAM 不足，请尝试以下选项：

- （仅限多模态模型）您可以使用 `VLLM_MM_INPUT_CACHE_GIB` 环境变量设置多模态输入缓存的大小（默认 4 GiB）。
- （仅限 CPU 后端）您可以使用 `VLLM_CPU_KVCACHE_SPACE` 环境变量设置键值缓存的大小（默认 4 GiB）。

## 多模态输入限制

您可以减少每个提示允许的多模态项目数量，以降低模型的内存占用：

```python
from vllm import LLM

# 每个提示最多接受 3 张图片和 1 个视频
llm = LLM(model="Qwen/Qwen2.5-VL-3B-Instruct",
          limit_mm_per_prompt={"image": 3, "video": 1})
```

您可以更进一步，通过将限制设置为零来完全禁用未使用的模态。
例如，如果您的应用程序只接受图像输入，则无需为视频分配任何内存。

```python
from vllm import LLM

# 接受任意数量的图像，但不接受视频
llm = LLM(model="Qwen/Qwen2.5-VL-3B-Instruct",
          limit_mm_per_prompt={"video": 0})
```

您甚至可以将多模态模型用于纯文本推理：

```python
from vllm import LLM

# 不接受图像，仅限文本
llm = LLM(model="google/gemma-3-27b-it",
          limit_mm_per_prompt={"image": 0})
```

## 多模态处理器参数

对于某些模型，您可以调整多模态处理器参数以减少处理后的多模态输入大小，从而节省内存。

以下是一些示例：

```python
from vllm import LLM

# 适用于 Qwen2-VL 系列模型
llm = LLM(model="Qwen/Qwen2.5-VL-3B-Instruct",
          mm_processor_kwargs={
              "max_pixels": 768 * 768,  # 默认值为 1280 * 28 * 28
          })

# 适用于 InternVL 系列模型
llm = LLM(model="OpenGVLab/InternVL2-2B",
          mm_processor_kwargs={
              "max_dynamic_patch": 4,  # 默认值为 12
          })
```