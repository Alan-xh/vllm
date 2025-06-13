# 提示嵌入输入

本页面教您如何将提示嵌入输入传递给 vLLM。

## 什么是提示嵌入？

大型语言模型的传统文本数据流是从文本到令牌 ID（通过分词器），然后从令牌 ID 到提示嵌入。对于传统的仅解码器模型（例如 meta-llama/Llama-3.1-8B-Instruct），将令牌 ID 转换为提示嵌入的步骤通过从学习到的嵌入矩阵中查找完成，但模型不仅限于处理与其令牌词汇表对应的嵌入。

!!! note
    提示嵌入目前仅在 v0 引擎中支持。

## 离线推理

要输入多模态数据，请遵循 [vllm.inputs.EmbedsPrompt][] 中的以下模式：

- `prompt_embeds`：一个表示提示/令牌嵌入序列的 PyTorch 张量。其形状为 (sequence_length, hidden_size)，其中 sequence_length 是令牌嵌入的数量，hidden_size 是模型的隐藏大小（嵌入大小）。

### Hugging Face Transformers 输入

您可以将 Hugging Face Transformers 模型的提示嵌入传递到提示嵌入字典的 `'prompt_embeds'` 字段，如以下示例所示：

<gh-file:examples/offline_inference/prompt_embed_inference.py>

## 在线服务

我们的 OpenAI 兼容服务器通过 [Completions API](https://platform.openai.com/docs/api-reference/completions) 接受提示嵌入输入。提示嵌入输入通过 JSON 包中的一个新的 `'prompt_embeds'` 键添加。

当单个请求中同时提供 `'prompt_embeds'` 和 `'prompt'` 输入时，提示嵌入总是首先返回。

提示嵌入以 base64 编码的 PyTorch 张量形式传递。

### 通过 OpenAI 客户端的 Transformers 输入

首先，启动 OpenAI 兼容服务器：

```bash
vllm serve meta-llama/Llama-3.2-1B-Instruct --task generate \
  --max-model-len 4096 --enable-prompt-embeds
```

然后，您可以按以下方式使用 OpenAI 客户端：

<gh-file:examples/online_serving/prompt_embed_inference_with_openai_client.py>