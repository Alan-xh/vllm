---
title: OpenAI兼容服务器
---

[](){ #openai-compatible-server }

vLLM 提供了一个 HTTP 服务器，支持 OpenAI 的 [补全 API](https://platform.openai.com/docs/api-reference/completions)、[聊天 API](https://platform.openai.com/docs/api-reference/chat) 等功能！通过该功能，您可以部署模型并使用 HTTP 客户端与其交互。

在终端中，您可以 [安装](../getting_started/installation/README.md) vLLM，然后使用 [`vllm serve`][serve-args] 命令启动服务器。（您也可以使用我们的 [Docker][deployment-docker] 镜像。）

```bash
vllm serve NousResearch/Meta-Llama-3-8B-Instruct \
  --dtype auto \
  --api-key token-abc123
```

要调用服务器，请在您喜欢的文本编辑器中创建一个使用 HTTP 客户端的脚本，并包含您想发送给模型的消息。然后运行该脚本。以下是使用 [官方 OpenAI Python 客户端](https://github.com/openai/openai-python) 的示例脚本：

```python
from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123",
)

completion = client.chat.completions.create(
    model="NousResearch/Meta-Llama-3-8B-Instruct",
    messages=[
        {"role": "user", "content": "你好！"}
    ]
)

print(completion.choices[0].message)
```

!!! tip
    vLLM 支持一些 OpenAI 不支持的参数，例如 `top_k`。
    您可以通过 OpenAI 客户端的 `extra_body` 参数将这些参数传递给 vLLM，例如 `extra_body={"top_k": 50}`。

!!! warning
    默认情况下，服务器会应用 Hugging Face 模型库中的 `generation_config.json`（如果存在）。这意味着某些采样参数的默认值可能会被模型创建者推荐的值覆盖。

    要禁用此行为，请在启动服务器时传递 `--generation-config vllm`。

## 支持的 API

我们目前支持以下 OpenAI API：

- [补全 API][completions-api] (`/v1/completions`)
    - 仅适用于 [文本生成模型](../models/generative_models.md) (`--task generate`)。
    - *注意：不支持 `suffix` 参数。*
- [聊天补全 API][chat-api] (`/v1/chat/completions`)
    - 仅适用于具有 [聊天模板][chat-template] 的 [文本生成模型](../models/generative_models.md) (`--task generate`)。
    - *注意：`parallel_tool_calls` 和 `user` 参数将被忽略。*
- [嵌入 API][embeddings-api] (`/v1/embeddings`)
    - 仅适用于 [嵌入模型](../models/pooling_models.md) (`--task embed`)。
- [转录 API][transcriptions-api] (`/v1/audio/transcriptions`)
    - 仅适用于自动语音识别（ASR）模型（OpenAI Whisper）(`--task generate`)。

此外，我们还提供以下自定义 API：

- [分词器 API][tokenizer-api] (`/tokenize`, `/detokenize`)
    - 适用于任何具有分词器的模型。
- [池化 API][pooling-api] (`/pooling`)
    - 适用于所有 [池化模型](../models/pooling_models.md)。
- [分类 API][classification-api] (`/classify`)
    - 仅适用于 [分类模型](../models/pooling_models.md) (`--task classify`)。
- [评分 API][score-api] (`/score`)
    - 适用于嵌入模型和 [跨编码器模型](../models/pooling_models.md) (`--task score`)。
- [重新排序 API][rerank-api] (`/rerank`, `/v1/rerank`, `/v2/rerank`)
    - 实现 [Jina AI 的 v1 重新排序 API](https://jina.ai/reranker/)
    - 也兼容 [Cohere 的 v1 和 v2 重新排序 API](https://docs.cohere.com/v2/reference/rerank)
    - Jina 和 Cohere 的 API 非常相似；Jina 的重新排序端点响应中包含额外信息。
    - 仅适用于 [跨编码器模型](../models/pooling_models.md) (`--task score`)。

[](){ #chat-template }

## 聊天模板

为了使语言模型支持聊天协议，vLLM 要求模型在其分词器配置中包含一个聊天模板。聊天模板是一个 Jinja2 模板，指定了角色、消息和其他特定于聊天的令牌如何在输入中编码。

`NousResearch/Meta-Llama-3-8B-Instruct` 的聊天模板示例可在此处找到：[链接](https://github.com/meta-llama/llama3?tab=readme-ov-file#instruction-tuned-models)

某些模型即使经过了指令/聊天微调，也没有提供聊天模板。对于这些模型，您可以在 `--chat-template` 参数中手动指定聊天模板，指定文件路径或字符串形式的模板。如果没有聊天模板，服务器将无法处理聊天请求，所有聊天请求都会报错。

```bash
vllm serve <model> --chat-template ./path-to-chat-template.jinja
```

vLLM 社区为流行模型提供了一组聊天模板，您可以在 <gh-dir:examples> 目录下找到它们。

随着多模态聊天 API 的引入，OpenAI 规范现在接受一种新的聊天消息格式，指定了 `type` 和 `text` 字段。以下是一个示例：

```python
completion = client.chat.completions.create(
    model="NousResearch/Meta-Llama-3-8B-Instruct",
    messages=[
        {"role": "user", "content": [{"type": "text", "text": "分类此情感：vLLM 很棒！"}]}
    ]
)
```

大多数语言模型的聊天模板期望 `content` 字段是字符串，但一些较新的模型（如 `meta-llama/Llama-Guard-3-1B`）期望内容按照 OpenAI 架构在请求中格式化。vLLM 提供尽力而为的自动检测支持，记录为类似 *"检测到聊天模板内容格式为..."* 的日志，并内部将传入请求转换为匹配的检测格式，可能为以下之一：

- `"string"`：字符串。
    - 示例：`"你好，世界"`
- `"openai"`：类似 OpenAI 架构的字典列表。
    - 示例：`[{"type": "text", "text": "你好，世界！"}]`

如果结果不符合您的期望，您可以使用 `--chat-template-content-format` CLI 参数来覆盖使用的格式。

## 额外参数

vLLM 支持一组 OpenAI API 不包含的参数。您可以通过 OpenAI 客户端将它们作为额外参数传递，或者直接将它们合并到 JSON 负载中（如果直接使用 HTTP 调用）。

```python
completion = client.chat.completions.create(
    model="NousResearch/Meta-Llama-3-8B-Instruct",
    messages=[
        {"role": "user", "content": "分类此情感：vLLM 很棒！"}
    ],
    extra_body={
        "guided_choice": ["正面", "负面"]
    }
)
```

## 额外 HTTP 头

目前仅支持 `X-Request-Id` HTTP 请求头。可以通过 `--enable-request-id-headers` 启用。

> 注意：启用头可能会在高 QPS 速率下显著影响性能。我们建议在路由器级别（例如通过 Istio）实现 HTTP 头，而不是在 vLLM 层中。
> 更多详情请参见 [此 PR](https://github.com/vllm-project/vllm/pull/11529)。

```python
completion = client.chat.completions.create(
    model="NousResearch/Meta-Llama-3-8B-Instruct",
    messages=[
        {"role": "user", "content": "分类此情感：vLLM 很棒！"}
    ],
    extra_headers={
        "x-request-id": "情感分类-00001",
    }
)
print(completion._request_id)

completion = client.completions.create(
    model="NousResearch/Meta-Llama-3-8B-Instruct",
    prompt="机器人不得伤害人类",
    extra_headers={
        "x-request-id": "补全测试",
    }
)
print(completion._request_id)
```

## API 参考

[](){ #completions-api }

### 补全 API

我们的补全 API 与 [OpenAI 的补全 API](https://platform.openai.com/docs/api-reference/completions) 兼容；
您可以使用 [官方 OpenAI Python 客户端](https://github.com/openai/openai-python) 与其交互。

代码示例：<gh-file:examples/online_serving/openai_completion_client.py>

#### 额外参数

支持以下 [采样参数][sampling-params]。

```python
--8<-- "vllm/entrypoints/openai/protocol.py:completion-sampling-params"
```

支持以下额外参数：

```python
--8<-- "vllm/entrypoints/openai/protocol.py:completion-extra-params"
```

[](){ #chat-api }

### 聊天 API

我们的聊天 API 与 [OpenAI 的聊天补全 API](https://platform.openai.com/docs/api-reference/chat) 兼容；
您可以使用 [官方 OpenAI Python 客户端](https://github.com/openai/openai-python) 与其交互。

我们支持 [视觉](https://platform.openai.com/docs/guides/vision) 和
[音频](https://platform.openai.com/docs/guides/audio?audio-generation-quickstart-example=audio-in) 相关参数；
更多信息请参见我们的 [多模态输入][multimodal-inputs] 指南。
- *注意：不支持 `image_url.detail` 参数。*

代码示例：<gh-file:examples/online_serving/openai_chat_completion_client.py>

#### 额外参数

支持以下 [采样参数][sampling-params]。

```python
--8<-- "vllm/entrypoints/openai/protocol.py:chat-completion-sampling-params"
```

支持以下额外参数：

```python
--8<-- "vllm/entrypoints/openai/protocol.py:chat-completion-extra-params"
```

[](){ #embeddings-api }

### 嵌入 API

我们的嵌入 API 与 [OpenAI 的嵌入 API](https://platform.openai.com/docs/api-reference/embeddings) 兼容；
您可以使用 [官方 OpenAI Python 客户端](https://github.com/openai/openai-python) 与其交互。

如果模型具有 [聊天模板][chat-template]，您可以用一组 `messages`（与 [聊天 API][chat-api] 相同的架构）替换 `inputs`，这将被视为对模型的单一提示。

代码示例：<gh-file:examples/online_serving/openai_embedding_client.py>

#### 多模态输入

您可以通过为服务器定义自定义聊天模板并在请求中传递一组 `messages` 来将多模态输入传递给嵌入模型。以下是示例说明。

=== "VLM2Vec"

    部署模型：

    ```bash
    vllm serve TIGER-Lab/VLM2Vec-Full --task embed \
      --trust-remote-code \
      --max-model-len 4096 \
      --chat-template examples/template_vlm2vec.jinja
    ```

    !!! warning
        由于 VLM2Vec 与 Phi-3.5-Vision 具有相同的模型架构，我们必须明确传递 `--task embed`
        以在嵌入模式而非文本生成模式下运行此模型。

        自定义聊天模板与此模型的原始模板完全不同，
        可在此处找到：<gh-file:examples/template_vlm2vec.jinja>

    由于请求架构未由 OpenAI 客户端定义，我们使用底层的 `requests` 库向服务器发送请求：

    ```python
    import requests

    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"

    response = requests.post(
        "http://localhost:8000/v1/embeddings",
        json={
            "model": "TIGER-Lab/VLM2Vec-Full",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": "表示给定的图像。"},
                ],
            }],
            "encoding_format": "float",
        },
    )
    response.raise_for_status()
    response_json = response.json()
    print("嵌入输出：", response_json["data"][0]["embedding"])
    ```

=== "DSE-Qwen2-MRL"

    部署模型：

    ```bash
    vllm serve MrLight/dse-qwen2-2b-mrl-v1 --task embed \
      --trust-remote-code \
      --max-model-len 8192 \
      --chat-template examples/template_dse_qwen2_vl.jinja
    ```

    !!! warning
        与 VLM2Vec 类似，我们必须明确传递 `--task embed`。

        此外，`MrLight/dse-qwen2-2b-mrl-v1` 需要为嵌入设置 EOS 令牌，
        这通过自定义聊天模板处理：<gh-file:examples/template_dse_qwen2_vl.jinja>

    !!! warning
        `MrLight/dse-qwen2-2b-mrl-v1` 需要为文本查询嵌入提供最小图像尺寸的占位图像。完整代码示例请参见下文。

完整示例：<gh-file:examples/online_serving/openai_chat_embedding_client_for_multimodal.py>

#### 额外参数

支持以下 [池化参数][pooling-params]。

```python
--8<-- "vllm/entrypoints/openai/protocol.py:embedding-pooling-params"
```

默认支持以下额外参数：

```python
--8<-- "vllm/entrypoints/openai/protocol.py:embedding-extra-params"
```

对于类似聊天的输入（即传递了 `messages`），支持以下额外参数：

```python
--8<-- "vllm/entrypoints/openai/protocol.py:chat-embedding-extra-params"
```

[](){ #transcriptions-api }

### 转录 API

我们的转录 API 与 [OpenAI 的转录 API](https://platform.openai.com/docs/api-reference/audio/createTranscription) 兼容；
您可以使用 [官方 OpenAI Python 客户端](https://github.com/openai/openai-python) 与其交互。

!!! note
    要使用转录 API，请使用 `pip install vllm[audio]` 安装额外的音频依赖。

代码示例：<gh-file:examples/online_serving/openai_transcription_client.py>
<!-- TODO: api 强制限制 + 上传音频 -->

#### 额外参数

支持以下 [采样参数][sampling-params]。

```python
--8<-- "vllm/entrypoints/openai/protocol.py:transcription-sampling-params"
```

支持以下额外参数：

```python
--8<-- "vllm/entrypoints/openai/protocol.py:transcription-extra-params"
```

[](){ #tokenizer-api }

### 分词器 API

我们的分词器 API 是对 [HuggingFace 风格分词器](https://huggingface.co/docs/transformers/en/main_classes/tokenizer) 的简单封装。
它包含两个端点：

- `/tokenize` 对应于调用 `tokenizer.encode()`。
- `/detokenize` 对应于调用 `tokenizer.decode()`。

[](){ #pooling-api }

### 池化 API

我们的池化 API 使用 [池化模型](../models/pooling_models.md) 编码输入提示并返回相应的隐藏状态。

输入格式与 [嵌入 API][embeddings-api] 相同，但输出数据可以包含任意嵌套列表，而不仅仅是一维浮点数列表。

代码示例：<gh-file:examples/online_serving/openai_pooling_client.py>

[](){ #classification-api }

### 分类 API

我们的分类 API 直接支持 Hugging Face 序列分类模型，例如 [ai21labs/Jamba-tiny-reward-dev](https://huggingface.co/ai21labs/Jamba-tiny-reward-dev) 和 [jason9693/Qwen2.5-1.5B-apeach](https://huggingface.co/jason9693/Qwen2.5-1.5B-apeach)。

我们通过 `as_classification_model()` 自动包装任何其他转换器，在最后一个令牌上进行池化，附加一个 `RowParallelLinear` 头，并应用 softmax 以生成每个类别的概率。

代码示例：<gh-file:examples/online_serving/openai_classification_client.py>

#### 示例请求

您可以通过传递字符串数组来分类多个文本：

请求：

```bash
curl -v "http://127.0.0.1:8000/classify" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jason9693/Qwen2.5-1.5B-apeach",
    "input": [
      "喜欢这个新咖啡馆——咖啡很棒。",
      "这次更新搞砸了一切，太烦人了。"
    ]
  }'
```

响应：

```bash
{
  "id": "classify-7c87cac407b749a6935d8c7ce2a8fba2",
  "object": "list",
  "created": 1745383065,
  "model": "jason9693/Qwen2.5-1.5B-apeach",
  "data": [
    {
      "index": 0,
      "label": "默认",
      "probs": [
        0.565970778465271,
        0.4340292513370514
      ],
      "num_classes": 2
    },
    {
      "index": 1,
      "label": "负面",
      "probs": [
        0.26448777318000793,
        0.7355121970176697
      ],
      "num_classes": 2
    }
  ],
  "usage": {
    "prompt_tokens": 20,
    "total_tokens": 20,
    "completion_tokens": 0,
    "prompt_tokens_details": null
  }
}
```

您也可以直接将字符串传递到 `input` 字段：

请求：

```bash
curl -v "http://127.0.0.1:8000/classify" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jason9693/Qwen2.5-1.5B-apeach",
    "input": "喜欢这个新咖啡馆——咖啡很棒。"
  }'
```

响应：

```bash
{
  "id": "classify-9bf17f2847b046c7b2d5495f4b4f9682",
  "object": "list",
  "created": 1745383213,
  "model": "jason9693/Qwen2.5-1.5B-apeach",
  "data": [
    {
      "index": 0,
      "label": "默认",
      "probs": [
        0.565970778465271,
        0.4340292513370514
      ],
      "num_classes": 2
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "total_tokens": 10,
    "completion_tokens": 0,
    "prompt_tokens_details": null
  }
}
```

#### 额外参数

支持以下 [池化参数][pooling-params]。

```python
--8<-- "vllm/entrypoints/openai/protocol.py:classification-pooling-params"
```

支持以下额外参数：

```python
--8<-- "vllm/entrypoints/openai/protocol.py:classification-extra-params"
```

[](){ #score-api }

### 评分 API

我们的评分 API 可以使用跨编码器模型或嵌入模型来预测句子对的得分。当使用嵌入模型时，得分对应于每对嵌入之间的余弦相似度。
通常，句子对的得分表示两个句子之间的相似性，范围为 0 到 1。

您可以在 [sbert.net](https://www.sbert.net/docs/package_reference/cross_encoder/cross_encoder.html) 找到跨编码器模型的文档。

代码示例：<gh-file:examples/online_serving/openai_cross_encoder_score.py>

#### 单次推理

您可以将字符串传递给 `text_1` 和 `text_2`，形成一个句子对。

请求：

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/score' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "model": "BAAI/bge-reranker-v2-m3",
  "encoding_format": "float",
  "text_1": "法国的首都是什么？",
  "text_2": "法国的首都是巴黎。"
}'
```

响应：

```bash
{
  "id": "score-request-id",
  "object": "list",
  "created": 693447,
  "model": "BAAI/bge-reranker-v2-m3",
  "data": [
    {
      "index": 0,
      "object": "score",
      "score": 1
    }
  ],
  "usage": {}
}
```

#### 批量推理

您可以将字符串传递给 `text_1`，将列表传递给 `text_2`，形成多个句子对，
每个句子对由 `text_1` 和 `text_2` 中的一个字符串构成。
句子对的总数为 `len(text_2)`。

请求：

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/score' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "model": "BAAI/bge-reranker-v2-m3",
  "text_1": "法国的首都是什么？",
  "text_2": [
    "巴西的首都是巴西利亚。",
    "法国的首都是巴黎。"
  ]
}'
```

响应：

```bash
{
  "id": "score-request-id",
  "object": "list",
  "created": 693570,
  "model": "BAAI/bge-reranker-v2-m3",
  "data": [
    {
      "index": 0,
      "object": "score",
      "score": 0.001094818115234375
    },
    {
      "index": 1,
      "object": "score",
      "score": 1
    }
  ],
  "usage": {}
}
```

您也可以将列表传递给 `text_1` 和 `text_2`，形成多个句子对，
每个句子对由 `text_1` 和 `text_2` 中的对应字符串构成（类似于 `zip()`）。
句子对的总数为 `len(text_2)`。

请求：

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/score' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "model": "BAAI/bge-reranker-v2-m3",
  "encoding_format": "float",
  "text_1": [
    "巴西的首都是什么？",
    "法国的首都是什么？"
  ],
  "text_2": [
    "巴西的首都是巴西利亚。",
    "法国的首都是巴黎。"
  ]
}'
```

响应：

```bash
{
  "id": "score-request-id",
  "object": "list",
  "created": 693447,
  "model": "BAAI/bge-reranker-v2-m3",
  "data": [
    {
      "index": 0,
      "object": "score",
      "score": 1
    },
    {
      "index": 1,
      "object": "score",
      "score": 1
    }
  ],
  "usage": {}
}
```

#### 额外参数

支持以下 [池化参数][pooling-params]。

```python
--8<-- "vllm/entrypoints/openai/protocol.py:score-pooling-params"
```

支持以下额外参数：

```python
--8<-- "vllm/entrypoints/openai/protocol.py:score-extra-params"
```

[](){ #rerank-api }

### 重新排序 API

我们的重新排序 API 可以使用嵌入模型或跨编码器模型来预测单个查询与文档列表之间的相关性得分。通常，句子对的得分表示两个句子之间的相似性，范围为 0 到 1。

您可以在 [sbert.net](https://www.sbert.net/docs/package_reference/cross_encoder/cross_encoder.html) 找到跨编码器模型的文档。

重新排序端点支持流行的重新排序模型，例如 `BAAI/bge-reranker-base` 和其他支持 `score` 任务的模型。此外，`/rerank`、`/v1/rerank` 和 `/v2/rerank`
端点与 [Jina AI 的重新排序 API 接口](https://jina.ai/reranker/) 和
[Cohere 的重新排序 API 接口](https://docs.cohere.com/v2/reference/rerank) 兼容，以确保与
流行的开源工具兼容。

代码示例：<gh-file:examples/online_serving/jinaai_rerank_client.py>

#### 示例请求

请注意，`top_n` 请求参数是可选的，默认为 `documents` 字段的长度。
结果文档将按相关性排序，`index` 属性可用于确定原始顺序。

请求：

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/v1/rerank' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "model": "BAAI/bge-reranker-base",
  "query": "法国的首都是什么？",
  "documents": [
    "巴西的首都是巴西利亚。",
    "法国的首都是巴黎。",
    "马和牛都是动物"
  ]
}'
```

响应：

```bash
{
  "id": "rerank-fae51b2b664d4ed38f5969b612edff77",
  "model": "BAAI/bge-reranker-base",
  "usage": {
    "total_tokens": 56
  },
  "results": [
    {
      "index": 1,
      "document": {
        "text": "法国的首都是巴黎。"
      },
      "relevance_score": 0.99853515625
    },
    {
      "index": 0,
      "document": {
        "text": "巴西的首都是巴西利亚。"
      },
      "relevance_score": 0.0005860328674316406
    }
  ]
}
```

#### 额外参数

支持以下 [池化参数][pooling-params]。

```python
--8<-- "vllm/entrypoints/openai/protocol.py:rerank-pooling-params"
```

支持以下额外参数：

```python
--8<-- "vllm/entrypoints/openai/protocol.py:rerank-extra-params"
```