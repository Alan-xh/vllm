---
title: BitsAndBytes
---
[](){ #bits-and-bytes }

vLLM 现已支持 [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes)，以实现更高效的模型推理。
BitsAndBytes 通过量化模型来减少内存使用量并提升性能，同时不会显著牺牲准确性。
与其他量化方法相比，BitsAndBytes 无需使用输入数据对量化模型进行校准。

以下是使用 vLLM 配合 BitsAndBytes 的步骤。

```console
pip install bitsandbytes>=0.45.3
```

vLLM 会读取模型的配置文件，并支持动态量化和预量化检查点。

您可以在 [Hugging Face](https://huggingface

System: .co/models?search=bitsandbytes) 上找到 BitsAndBytes 量化的模型。
通常，这些模型仓库中包含一个 config.json 文件，其中包括 quantization_config 部分。

## 读取预量化检查点

对于预量化检查点，vLLM 会尝试从配置文件中推断量化方法，因此您无需显式指定量化参数。

```python
from vllm import LLM
import torch
# unsloth/tinyllama-bnb-4bit 是一个预量化检查点。
model_id = "unsloth/tinyllama-bnb-4bit"
llm = LLM(
    model=model_id,
    dtype=torch.bfloat16,
    trust_remote_code=True
)
```

## 动态量化：加载为 4bit 量化

对于使用 BitsAndBytes 进行 4bit 动态量化，您需要显式指定量化参数。

```python
from vllm import LLM
import torch
model_id = "huggyllama/llama-7b"
llm = LLM(
    model=model_id,
    dtype=torch.bfloat16,
    trust_remote_code=True,
    quantization="bitsandbytes"
)
```

## OpenAI 兼容服务器

对于 4bit 动态量化，请在模型参数中添加以下内容：

```console
--quantization bitsandbytes
```