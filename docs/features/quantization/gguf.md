
title: GGUF
---
[](){ #gguf }

!!! warning
    请注意，vLLM 中的 GGUF 支持目前处于高度实验阶段且尚未优化，可能与其他功能不兼容。目前，您可以使用 GGUF 来减少内存占用。如果遇到任何问题，请向 vLLM 团队报告。

!!! warning
    当前，vLLM 仅支持加载单文件 GGUF 模型。如果您有多个文件的 GGUF 模型，可以使用 [gguf-split](https://github.com/ggerganov/llama.cpp/pull/6135) 工具将其合并为单文件模型。

要使用 vLLM 运行 GGUF 模型，您可以从 [TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF) 下载并使用本地 GGUF 模型，命令如下：

```console
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
# 我们建议使用基础模型的分词器，以避免耗时且可能出错的分词器转换。
vllm serve ./tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
   --tokenizer TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

您还可以添加 `--tensor-parallel-size 2` 以启用 2 个 GPU 的张量并行推理：

```console
# 我们建议使用基础模型的分词器，以避免耗时且可能出错的分词器转换。
vllm serve ./tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
   --tokenizer TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
   --tensor-parallel-size 2
```

!!! warning
    我们建议使用基础模型的分词器而非 GGUF 模型的分词器。因为从 GGUF 进行分词器转换耗时且不稳定，尤其是对于一些词汇量较大的模型。

GGUF 假设 Hugging Face 可以将元数据转换为配置文件。如果 Hugging Face 不支持您的模型，您可以手动创建配置文件并将其作为 `hf-config-path` 传递：

```console
# 如果您的模型不受 Hugging Face 支持，您可以手动提供一个兼容 Hugging Face 的配置文件路径
vllm serve ./tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
   --tokenizer TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
   --hf-config-path TinyLlama/TInyLlama-1.1B-Chat-v1.0
```

您也可以通过 LLM 入口直接使用 GGUF 模型：

```python
from vllm import LLM, SamplingParams

# 在此脚本中，我们展示了如何将输入传递给 chat 方法：
conversation = [
   {
      "role": "system",
      "content": "您是一个有用的助手"
   },
   {
      "role": "user",
      "content": "您好"
   },
   {
      "role": "assistant",
      "content": "您好！今天我能帮您什么？"
   },
   {
      "role": "user",
      "content": "写一篇关于高等教育重要性的文章。",
   },
]

# 创建一个采样参数对象。
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# 创建一个 LLM。
llm = LLM(model="./tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
         tokenizer="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
# 从提示生成文本。输出是一个包含提示、生成文本和其他信息的 RequestOutput 对象列表。
outputs = llm.chat(conversation, sampling_params)

# 打印输出。
for output in outputs:
   prompt = output.prompt
   generated_text = output.outputs[0].text
   print(f"提示: {prompt!r}, 生成的文本: {generated_text!r}")
```
