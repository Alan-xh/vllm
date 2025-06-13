---
title: 推测解码
---
[](){ #spec-decode }

!!! warning
    请注意，vLLM 中的推测解码尚未优化，通常无法为所有提示数据集或采样参数实现令牌间延迟的减少。
    优化工作正在进行中，可在此处跟踪进展：<gh-issue:4630>

!!! warning
    当前，vLLM 中的推测解码与流水线并行不兼容。

本文档展示如何在 vLLM 中使用[推测解码](https://x.com/karpathy/status/1697318534555336961)。
推测解码是一种技术，可改善内存受限的大型语言模型（LLM）推理中的令牌间延迟。

## 使用草稿模型进行推测

以下代码在离线模式下配置 vLLM 以使用推测解码，采用草稿模型，每次推测 5 个令牌。

```python
from vllm import LLM, SamplingParams

prompts = [
    "人工智能的未来是",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
    model="facebook/opt-6.7b",
    tensor_parallel_size=1,
    speculative_config={
        "model": "facebook/opt-125m",
        "num_speculative_tokens": 5,
    },
)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"提示: {prompt!r}, 生成的文本: {generated_text!r}")
```

要在在线模式下执行相同操作，请启动服务器：

```bash
python -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port 8000 \
    --model facebook/opt-6.7b \
    --seed 42 \
    -tp 1 \
    --gpu_memory_utilization 0.8 \
    --speculative_config '{"model": "facebook/opt-125m", "num_speculative_tokens": 5}'
```

!!! warning
    注意：请使用 `--speculative_config` 设置所有与推测解码相关的配置。之前通过 `--speculative_model` 指定模型并单独添加相关参数（例如 `--num_speculative_tokens`）的方法现已弃用。

然后使用客户端：

```python
from openai import OpenAI

# 修改 OpenAI 的 API 密钥和 API 基础地址以使用 vLLM 的 API 服务器。
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    # 默认为 os.environ.get("OPENAI_API_KEY")
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id

# 补全 API
stream = False
completion = client.completions.create(
    model=model,
    prompt="人工智能的未来是",
    echo=False,
    n=1,
    stream=stream,
)

print("补全结果:")
if stream:
    for c in completion:
        print(c)
else:
    print(completion)
```

## 通过匹配提示中的 n-gram 进行推测

以下代码配置 vLLM 使用推测解码，其中通过匹配提示中的 n-gram 生成预测。  
更多信息请阅读[此线程](https://x.com/joao_gante/status/1747322413006643259)。

```python
from vllm import LLM, SamplingParams

prompts = [
    "人工智能的未来是",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
    model="facebook/opt-6.7b",
    tensor_parallel_size=1,
    speculative_config={
        "method": "ngram",
        "num_speculative_tokens": 5,
        "prompt_lookup_max": 4,
    },
)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"提示: {prompt!r}, 生成的文本: {generated_text!r}")
```

## 使用 MLP 推测器进行推测

以下代码配置 vLLM 使用推测解码，其中预测由基于上下文向量和采样令牌的草稿模型生成。  
更多信息请参见[此博客](https://pytorch.org/blog/hitchhikers-guide-speculative-decoding/) 或 [此技术报告](https://arxiv.org/abs/2404.19124)。  

```python
from vLLM import LLM, SamplingParams

prompts = [
    "人工智能的未来是",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
    model="meta-llama/Meta-Llama-3.1-70B-Instruct",
    tensor_parallel_size=4,
    speculative_config={
        "model": "ibm-ai-platform/llama3-70b-accelerator",
        "draft_tensor_parallel_size": 1,
    },
)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"提示: {prompt!r}, 生成的文本: {generated_text!r}")
```

请注意，这些推测模型目前需要不使用张量并行运行，尽管主模型可以使用张量并行（参见上面的示例）。  
由于推测模型相对较小，我们仍然可以看到显著的加速。然而，这一限制将在未来版本中修复。  

HF hub 上提供了一系列的此类推测模型：  

- [llama-13b-accelerator](https://huggingface.co/ibm-ai-platform/llama-13b-accelerator)  
- [llama3-8b-accelerator](https://huggingface.co/ibm-ai-platform/llama3-8b-accelerator)  
- [codellama-34b-accelerator](https://huggingface.co/ibm-ai-platform/codellama-34b-accelerator)  
- [llama2-70b-accelerator](https://huggingface.co/ibm-ai-platform/llama2-70b-accelerator)  
- [llama3-70b-accelerator](https://huggingface.co/ibm-ai-platform/llama3-70b-accelerator)  
- [granite-3b-code-instruct-accelerator](https://huggingface.co/ibm-granite/granite-3b-code-instruct-accelerator)  
- [granite-8b-code-instruct-accelerator](https://huggingface.co/ibm-granite/granite-8b-code-instruct-accelerator)  
- [granite-7b-instruct-accelerator](https://huggingface.co/ibm-granite/granite-7b-instruct-accelerator)  
- [granite-20b-code-instruct-accelerator](https://huggingface.co/ibm-granite/granite-20b-code-instruct-accelerator)  

## 使用基于 EAGLE 的草稿模型进行预测  

以下代码配置 vLLM 使用推测解码，其中预测由基于 [EAGLE（用于更大语言模型效率的推测算法）](https://arxiv.org/pdf/2401.15077) 的草稿模型生成。  
离线模式的详细示例，包括如何提取请求级接受率，可参考 [这里](gh-file:examples/offline_inference/eagle.py)。  

```python
from vLLM import LLM, SamplingParams

prompts = [
    "人工智能的未来是",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    tensor_parallel_size=4,
    speculative_config={"        "model": "yuhuili/EAGLE-LLaMA3-Instruct-8B",
        "draft_tensor_parallel_size": 1,
    },
)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"提示: {prompt!r}, 生成的文本: {generated_text!r}")

```

使用基于 EAGLE 的草稿模型时需要考虑的几个重要事项：  

1. [EAGLE 模型的 HF 仓库](https://huggingface.co/yuhuili) 中可用的 EAGLE 草稿模型在 [PR 12304](https://github.com/vllm-project/vllm/pull/12304) 后应能直接由 vLLM 加载和使用。  
   如果您使用的是 [PR 12304](https://github.com/vllm-project/vllm/pull/12304) 之前的 vLLM 版本，请使用 [脚本](https://gist.github.com/abhigoyal1997/1a6b9ccb7704fbc67f625e86b8312d6d) 转换推测模型，  
   并在 `speculative_config` 中指定 `"model": "path/to/modified/eagle/model"`。如果使用最新版本的 vLLM 仍出现权重加载问题，请留言反馈或提出问题。  

2. 基于 EAGLE 的草稿模型需要在无张量并行模式下运行（即在 `speculative_config` 中设置 `draft_tensor_parallel_size` 为 1），  
   尽管如此，主模型可以使用张量并行（参见上述示例）。  

3. 在 vLLM 中使用基于 EAGLE 的推测器时，观察到的加速效果低于 [参考实现](https://github.com/SafeAILab/EAGLE) 中报告的性能。  
   该问题正在调查中，并在此处跟踪：[https://github.com/vllm-project/vllm/issues/9565](https://github.com/vllm-project/vllm/issues/9565)。  

HF hub 上提供了一些 EAGLE 草稿模型：  

| 基础模型                                                             |  
| EAGLE 在 HuggingFace 上的模型                     | # EAGLE 参数 |  
|---------------------------------------------------------------------|---------------------------------------------|--------------------|  
| Vicuna-7B-v1.3                                                       | yuhuili/EAGLE-Vicuna-7B-v1.3              | 0.24B         |  
| Vicuna-13B-v1.3                                                       | yuhuili/EAGLE-Vicuna-13B-v1.3            | 0.37B         |  
| Vicuna-33B-v1.3                                                      | yuhuili/EAGLE-Vicuna-33B-v1.3           | 0.56B         |  
| LLaMA2-Chat-7B                                                       | yuhuili/EAGLE-llama2-chat-7B             | 0.24B         |  
| LLaMA2-Chat-13B                                                     | yuhuili/EAGLE-llama2-chat-13B            | 0.37B         |  
| LLaMA2-Chat-70B                                                     | yuhuili/EAGLE-llama2-chat-70B            | 0.99B         |  
| Mixtral-8x7B-Instruct-v0.1                                           | yuhuili/EAGLE-mixtral-instruct-8x7B      | 0.28B         |  
| LLaMA3-Instruct-8B                                                   | yuhuili/EAGLE-LLaMA3-Instruct-8B         | 0.25B         |  
| LLaMA3-Instruct-70B                                          | yuhuili/EAGLE-LLaMA3-Instruct-70B        | 0.99B         |  
| Qwen2-7B-Instruct                                                    | yuhuili/EAGLE-Qwen2-7B-Instruct           | 0.26B         |  
| Qwen2-72B-Instruct                                                   | yuhuili/EAGLE-Qwen2-72B-Instruct      | 0.31.05B        |  

## 推测解码的无损保证  

在 vLLM 中，推测解码旨在提高推理效率，同时保持准确性。本节讨论推测解码的无损保证，将其分为三个关键领域：  

1. **理论无损性**  
   - 推测解码采样在理论上是无损的，直到达到硬件数字的精度限制。浮点误差可能导致输出分布的轻微变化，如在  
     [加速大规模语言模型解码与推测采样](https://arxiv.org/pdf/2302.01318) 中讨论。  

2. **算法无损性**  
   - vLLM 的推测解码实现经过算法验证是无损的。关键验证测试包括：  

   > - **拒绝采样器收敛性**：确保 vLLM 的拒绝采样器的样本与目标分布一致。[查看测试代码](https://github.com/vllm-project/vLLm/blob/47b65a550866c7ffbd076ecb74106714838ce7da/tests/samplers/test_rejection_sampler.py#L252)  
   > - **贪婪采样等价性**：确认使用推测解码的贪婪采样与不使用时的贪婪采样相匹配。这验证了 vLLM 的推测解码框架，当与 vLLM 前向传播和 vLLM 拒绝采样器集成时，提供了一个无损保证。  
   > - <几乎所有的测试在 <gh-dir:tests/spec-decode/e2e>> 中，使用了 [此断言实现](https://github.com/vllm-project/vLLm/blob/b67ae00cdbbe1a58ffc8b8f170f0c8d7b944a6842a/tests/spec_decode/e2e/conftest.py#L291) 验证了此属性。  

3. **vLLM 日志概率稳定性**  
   - vLLM 目前不保证稳定的令牌对数概率（logprobs）。这可能导致同一请求在多次运行中产生不同的输出。更多详情，请参见 [FAQ 部分](faq) 中标题为 *提示词的输出在 vLLM 中可能因运行而异吗？* 的常见问题解答。  

虽然 vLLM 努力确保推测解码的无损性，但由于以下因素，使用和不使用推测解码时生成的输出可能会有所不同：  

- **浮点精度**：硬件数字精度的差异可能导致输出分布的细微差异。  
- **批量大小和数值稳定性**：批量大小的变化可能导致对数概率和输出概率的变化，可能由于批处理操作中的非确定性行为或数值不稳定。  

有关缓解策略，请参阅 [FAQ 条目](https://github.com/vllm-project/vllm/blob/main/docs/faq.rst) *提示的输出在 vLLM 中可能因运行而异吗？* 在 [FAQs][faq] 中。  

## vLLM 贡献者的资源  

- [vLLM 中的推测解码黑客指南](https://www.youtube.com/watch?v=9wNApX6z_4)  
- [什么是 vLLM 中的前瞻调度？](https://docs.google.com/document/d/1Z9TvqzzBPnh5WnappX7qZzdqpyK2UEeFeq5zMZbMFE8jR0HCs/edit#heading=h.1fjd0donq5a)  
- [关于批量扩展的信息](https://docs.google.com/document/d/1T-JaS2T1NRdjwdqpykCXx8tORppiwx5asxA/edit#heading=h.kk7dq05lc6q8)  
- [动态推测解码](https://github.com/vllm-project/vllm/issues/4565)