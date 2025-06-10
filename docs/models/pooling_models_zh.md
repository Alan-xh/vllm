---
title: 池化模型
---
[](){ #pooling-models }

vLLM 还支持池化模型，包括嵌入、重新排序和奖励模型。

在 vLLM 中，池化模型实现了 [VllmModelForPooling][vllm.model_executor.models.VllmModelForPooling] 接口。
这些模型使用 [Pooler][vllm.model_executor.layers.Pooler] 来提取输入的最终隐藏状态，
然后返回这些状态。

!!! note
    我们目前主要为了方便而支持池化模型。
    如 [Compatibility Matrix][compatibility-matrix] 所示，大多数 vLLM 功能不适用于池化模型，
    因为这些功能仅在生成或解码阶段起作用，因此性能可能不会有太大提升。

对于池化模型，我们支持以下 `--task` 选项。
所选选项会设置用于提取最终隐藏状态的默认池化器：

| 任务                            | 池化类型       | 归一化          | Softmax   |
|---------------------------------|----------------|-----------------|-----------|
| 嵌入 (`embed`)                 | `LAST`         | ✅︎              | ❌         |
| 分类 (`classify`)              | `LAST`         | ❌               | ✅︎        |
| 句子对评分 (`score`)           | *              | *               | *         |

*默认池化器始终由模型定义。

!!! note
    如果 vLLM 中模型的实现定义了其自己的池化器，则默认池化器将设置为该池化器，而不是本表中指定的池化器。

当加载 [Sentence Transformers](https://huggingface.co/sentence-transformers) 模型时，
我们会尝试根据其 Sentence Transformers 配置文件（`modules.json`）覆盖默认池化器。

!!! tip
    您可以通过 `--override-pooler-config` 选项自定义模型的池化方法，
    该选项优先于模型和 Sentence Transformers 的默认设置。

## 离线推理

[LLM][vllm.LLM] 类提供了多种用于离线推理的方法。
有关初始化模型时的选项列表，请参见 [configuration][configuration]。

### `LLM.encode`

[encode][vllm.LLM.encode] 方法适用于 vLLM 中的所有池化模型。
它直接返回提取的隐藏状态，这对奖励模型非常有用。

```python
from vllm import LLM

llm = LLM(model="Qwen/Qwen2.5-Math-RM-72B", task="reward")
(output,) = llm.encode("Hello, my name is")

data = output.outputs.data
print(f"Data: {data!r}")
```

### `LLM.embed`

[embed][vllm.LLM.embed] 方法为每个提示输出一个嵌入向量。
它主要为嵌入模型设计。

```python
from vllm import LLM

llm = LLM(model="intfloat/e5-mistral-7b-instruct", task="embed")
(output,) = llm.embed("Hello, my name is")

embeds = output.outputs.embedding
print(f"Embeddings: {embeds!r} (size={len(embeds)})")
```

代码示例可在此处找到：<gh-file:examples/offline_inference/basic/embed.py>

### `LLM.classify`

[classify][vllm.LLM.classify] 方法为每个提示输出一个概率向量。
它主要为分类模型设计。

```python
from vllm import LLM

llm = LLM(model="jason9693/Qwen2.5-1.5B-apeach", task="classify")
(output,) = llm.classify("Hello, my name is")

probs = output.outputs.probs
print(f"Class Probabilities: {probs!r} (size={len(probs)})")
```

代码示例可在此处找到：<gh-file:examples/offline_inference/basic/classify.py>

### `LLM.score`

[score][vllm.LLM.score] 方法输出句子对之间的相似度得分。
它为嵌入模型和交叉编码器模型设计。嵌入模型使用余弦相似度，而 [交叉编码器模型](https://www.sbert.net/examples/applications/cross-encoder/README.html) 在 RAG 系统中用作候选查询-文档对的重新排序器。

!!! note
    vLLM 只能执行 RAG 的模型推理部分（例如嵌入、重新排序）。
    要处理更高层次的 RAG，您应使用集成框架，例如 [LangChain](https://github.com/langchain-ai/langchain)。

```python
from vllm import LLM

llm = LLM(model="BAAI/bge-reranker-v2-m3", task="score")
(output,) = llm.score("What is the capital of France?",
                      "The capital of Brazil is Brasilia.")

score = output.outputs.score
print(f"Score: {score}")
```

代码示例可在此处找到：<gh-file:examples/offline_inference/basic/score.py>

## 在线服务

我们的 [OpenAI-Compatible Server][openai-compatible-server] 提供了与离线 API 相对应的端点：

- [Pooling API][pooling-api] 类似于 `LLM.encode`，适用于所有类型的池化模型。
- [Embeddings API][embeddings-api] 类似于 `LLM.embed`，接受文本和 [多模态输入][multimodal-inputs] 用于嵌入模型。
- [Classification API][classification-api] 类似于 `LLM.classify`，适用于序列分类模型。
- [Score API][score-api] 类似于 `LLM.score`，适用于交叉编码器模型。

## Matryoshka 嵌入

[Matryoshka 嵌入](https://sbert.net/examples/sentence_transformer/training/matryoshka/README.html#matryoshka-embeddings) 或 [Matryoshka 表示学习 (MRL)](https://arxiv.org/abs/2205.13147) 是一种用于训练嵌入模型的技术。它允许用户在性能和成本之间进行权衡。

!!! warning
    并非所有嵌入模型都使用 Matryoshka 表示学习进行训练。为了避免误用 `dimensions` 参数，vLLM 会对尝试更改不支持 Matryoshka 嵌入的模型输出维度的请求返回错误。

    例如，在使用 `BAAI/bge-m3` 模型时设置 `dimensions` 参数将导致以下错误：

    ```json
    {"object":"error","message":"Model \"BAAI/bge-m3\" does not support matryoshka representation, changing output dimensions will lead to poor results.","type":"BadRequestError","param":null,"code":400}
    ```

### 手动启用 Matryoshka 嵌入

目前没有官方接口用于指定支持 Matryoshka 嵌入。在 vLLM 中，如果 `config.json` 中的 `is_matryoshka` 为 `True`，则允许将输出更改为任意维度。使用 `matryoshka_dimensions` 可以控制允许的输出维度。

对于支持 Matryoshka 嵌入但未被 vLLM 识别的模型，请使用 `hf_overrides={"is_matryoshka": True}`、`hf_overrides={"matryoshka_dimensions": [<allowed output dimensions>]}`（离线）或 `--hf_overrides '{"is_matryoshka": true}'`、`--hf_overrides '{"matryoshka_dimensions": [<allowed output dimensions>]}'`（在线）手动覆盖配置。

以下是启用 Matryoshka 嵌入的服务模型示例：

```text
vllm serve Snowflake/snowflake-arctic-embed-m-v1.5 --hf_overrides '{"matryoshka_dimensions":[256]}'
```

### 离线推理

您可以通过 [PoolingParams][vllm.PoolingParams] 中的 `dimensions` 参数更改支持 Matryoshka 嵌入的嵌入模型的输出维度。

```python
from vllm import LLM, PoolingParams

model = LLM(model="jinaai/jina-embeddings-v3", 
            task="embed", 
            trust_remote_code=True)
outputs = model.embed(["Follow the white rabbit."], 
                      pooling_params=PoolingParams(dimensions=32))
print(outputs[0].outputs)
```

代码示例可在此处找到：<gh-file:examples/offline_inference/embed_matryoshka_fy.py>

### 在线推理

使用以下命令启动 vLLM 服务器：

```text
vllm serve jinaai/jina-embeddings-v3 --trust-remote-code
```

您可以通过 `dimensions` 参数更改支持 Matryoshka 嵌入的嵌入模型的输出维度。

```text
curl http://127.0.0.1:8000/v1/embeddings \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "Follow the white rabbit.",
    "model": "jinaai/jina-embeddings-v3",
    "encoding_format": "float",
    "dimensions": 32
  }'
```

预期输出：

```json
{"id":"embd-5c21fc9a5c9d4384a1b021daccaf9f64","object":"list","created":1745476417,"model":"jinaai/jina-embeddings-v3","data":[{"index":0,"object":"embedding","embedding":[-0.3828125,-0.1357421875,0.03759765625,0.125,0.21875,0.09521484375,-0.003662109375,0.1591796875,-0.130859375,-0.0869140625,-0.1982421875,0.1689453125,-0.220703125,0.1728515625,-0.2275390625,-0.0712890625,-0.162109375,-0.283203125,-0.055419921875,-0.0693359375,0.031982421875,-0.04052734375,-0.2734375,0.1826171875,-0.091796875,0.220703125,0.37890625,-0.0888671875,-0.12890625,-0.021484375,-0.0091552734375,0.23046875]}],"usage":{"prompt_tokens":8,"total_tokens":8,"completion_tokens":0,"prompt_tokens_details":null}}
```

OpenAI 客户端示例可在此处找到：<gh-file:examples/online_serving/openai_embedding_matryoshka_fy.py>