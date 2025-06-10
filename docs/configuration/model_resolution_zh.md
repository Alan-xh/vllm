# 模型解析

vLLM 通过检查模型库 `config.json` 中的 `architectures` 字段，并找到已注册到 vLLM 的相应实现来加载与 HuggingFace 兼容的模型。
然而，我们的模型解析可能会由于以下原因失败：

- 模型库的 `config.json` 缺少 `architectures` 字段。
- 非官方库引用的模型使用了 vLLM 中未记录的其他名称。
- 多个模型使用相同的架构名称，导致无法确定应该加载哪个模型。

要解决此问题，请通过将 `config.json` 覆盖传递给 `hf_overrides` 选项来明确指定模型架构。
例如：

```python
from vllm import LLM

model = LLM(
model="cerebras/Cerebras-GPT-1.3B",
hf_overrides={"architectures": ["GPT2LMHeadModel"]}, # GPT-2
)
```

我们的[支持模型列表][supported-models] 显示了 vLLM 可以识别的模型架构。