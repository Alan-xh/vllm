---
title: 生成模型
---
[](){ #generative-models }

vLLM 为生成模型提供了一流的支持，涵盖了大多数大型语言模型（LLMs）。

在 vLLM 中，生成模型实现了 [VllmModelForTextGeneration][vllm.model_executor.models.VllmModelForTextGeneration] 接口。
基于输入的最终隐藏状态，这些模型输出生成令牌的对数概率，
随后通过 [Sampler][vllm.model_executor.layers.Sampler] 获得最终文本。

对于生成模型，唯一支持的 `--task` 选项是 `"generate"`。
通常，这会自动推断，因此您无需明确指定。

## 离线推理

[LLM][vllm.LLM] 类提供了多种离线推理方法。
有关初始化模型时选项的列表，请参见 [configuration][configuration]。

### `LLM.generate`

[generate][vllm.LLM.generate] 方法适用于 vLLM 中的所有生成模型。
它类似于 [HF Transformers 中的对应方法](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationMixin.generate)，
不同之处在于它还会自动执行分词和去分词。

```python
from vllm import LLM

llm = LLM(model="facebook/opt-125m")
outputs = llm.generate("Hello, my name is")

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"提示: {prompt!r}, 生成的文本: {generated_text!r}")
```

您可以通过传递 [SamplingParams][vllm.SamplingParams] 来选择性地控制语言生成。
例如，您可以通过设置 `temperature=0` 使用贪婪采样：

```python
from vllm import LLM, SamplingParams

llm = LLM(model="facebook/opt-125m")
params = SamplingParams(temperature=0)
outputs = llm.generate("Hello, my name is", params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"提示: {prompt!r}, 生成的文本: {generated_text!r}")
```

!!! warning
    默认情况下，如果存在，vLLM 将使用模型创建者推荐的采样参数，应用来自 HuggingFace 模型仓库的 `generation_config.json`。在大多数情况下，如果未指定 [SamplingParams][vllm.SamplingParams]，这将为您提供最佳结果。

    但是，如果需要 vLLM 的默认采样参数，请在创建 [LLM][vllm.LLM] 实例时传递 `generation_config="vllm"`。
代码示例可在此处找到：<gh-file:examples/offline_inference/basic/basic.py>

### `LLM.beam_search`

[beam_search][vllm.LLM.beam_search] 方法在 [generate][vllm.LLM.generate] 的基础上实现了 [束搜索](https://huggingface.co/docs/transformers/en/generation_strategies#beam-search)。
例如，使用 5 个束进行搜索并输出最多 50 个令牌：

```python
from vllm import LLM
from vllm.sampling_params import BeamSearchParams

llm = LLM(model="facebook/opt-125m")
params = BeamSearchParams(beam_width=5, max_tokens=50)
outputs = llm.beam_search([{"prompt": "Hello, my name is "}], params)

for output in outputs:
    generated_text = output.sequences[0].text
    print(f"生成的文本: {generated_text!r}")
```

### `LLM.chat`

[chat][vllm.LLM.chat] 方法在 [generate][vllm.LLM.generate] 的基础上实现了聊天功能。
特别是，它接受类似于 [OpenAI 聊天补全 API](https://platform.openai.com/docs/api-reference/chat) 的输入，
并自动应用模型的 [聊天模板](https://huggingface.co/docs/transformers/en/chat_templating) 来格式化提示。

!!! warning
    通常，只有经过指令调整的模型才具有聊天模板。
    基础模型可能表现不佳，因为它们未经过训练以响应聊天对话。

```python
from vllm import LLM

llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct")
conversation = [
    {
        "role": "system",
        "content": "你是一个有帮助的助手"
    },
    {
        "role": "user",
        "content": "你好"
    },
    {
        "role": "assistant",
        "content": "你好！今天我能帮你什么？"
    },
    {
        "role": "user",
        "content": "写一篇关于高等教育重要性的文章。",
    },
]
outputs = llm.chat(conversation)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"提示: {prompt!r}, 生成的文本: {generated_text!r}")
```

代码示例可在此处找到：<gh-file:examples/offline_inference/basic/chat.py>

如果模型没有聊天模板或您想指定另一个模板，
您可以明确传递聊天模板：

```python
from vllm.entrypoints.chat_utils import load_chat_template

# 您可以在 `examples/` 下找到现有聊天模板的列表
custom_template = load_chat_template(chat_template="<path_to_template>")
print("已加载的聊天模板:", custom_template)

outputs = llm.chat(conversation, chat_template=custom_template)
```

## 在线服务

我们的 [OpenAI 兼容服务器][openai-compatible-server] 提供了与离线 API 对应的端点：

- [补全 API][completions-api] 类似于 `LLM.generate`，但仅接受文本。
- [聊天 API][chat-api] 类似于 `LLM.chat`，接受文本和 [多模态输入][multimodal-inputs]，适用于具有聊天模板的模型。