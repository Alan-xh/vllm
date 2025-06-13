---
title: BitBLAS
---
[](){ #bitblas }

vLLM 现支持 [BitBLAS](https://github.com/microsoft/BitBLAS)，以实现更高效、更灵活的模型推理。与其他量化框架相比，BitBLAS 提供更多的精度组合。

!!! note
    请确保您的硬件支持所选的 `dtype`（`torch.bfloat16` 或 `torch.float16`）。
    大多数较新的 NVIDIA GPU 支持 `float16`，而 `bfloat16` 在较新的架构（如 Ampere 或 Hopper）上更常见。
    有关详情，请参阅 [支持的硬件](https://docs.vllm.ai/en/latest/features/quantization/supported_hardware.html)。

以下是使用 vLLM 和 BitBLAS 的步骤。

```console
pip install bitblas>=0.1.0
```

vLLM 会读取模型的配置文件，并支持预量化的检查点。

您可以在以下位置找到预量化的模型：

- [Hugging Face (BitBLAS)](https://huggingface.co/models?search=bitblas)
- [Hugging Face (GPTQ)](https://huggingface.co/models?search=gptq)

通常，这些仓库中会有一个包含 `quantization_config` 部分的 `quantize_config.json` 文件。

## 读取 bitblas 格式的检查点

```python
from vllm import LLM
import torch

# "hxbgsyxh/llama-13b-4bit-g-1-bitblas" 是一个预量化的检查点。
model_id = "hxbgsyxh/llama-13b-4bit-g-1-bitblas"
llm = LLM(
    model=model_id,
    dtype=torch.bfloat16,
    trust_remote_code=True,
    quantization="bitblas"
)
```

## 读取 gptq 格式的检查点

```python
from vllm import LLM
import torch

# "hxbgsyxh/llama-13b-4bit-g-1" 是一个预量化的检查点。
model_id = "hxbgsyxh/llama-13b-4bit-g-1"
llm = LLM(
    model=model_id,
    dtype=torch.float16,
    trust_remote_code=True,
    quantization="bitblas",
    max_model_len=1024
)
```