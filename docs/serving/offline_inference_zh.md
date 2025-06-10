---
title：离线推理
---
[](){ #offline-inference }

您可以用自己的代码针对一系列提示运行 vLLM。

离线 API 基于 [LLM][vllm.LLM] 类。
要初始化 vLLM 引擎，请创建一个新的 `LLM` 实例并指定要运行的模型。

例如，以下代码从 HuggingFace 下载 [`facebook/opt-125m`](https://huggingface.co/facebook/opt-125m) 模型，
并使用默认配置在 vLLM 中运行该模型。

```python
from vllm import LLM

llm = LLM(model="facebook/opt-125m")
```

初始化 `LLM` 实例后，您可以使用各种 API 执行模型推理。
可用的 API 取决于正在运行的模型类型：

- [生成模型][generative-models] 输出对数概率，并从中采样以获得最终的输出文本。
- [池化模型][pooling-models] 直接输出其隐藏状态。

有关每个 API 的更多详细信息，请参阅以上页面。

!!! info
[API 参考][offline-inference-api]