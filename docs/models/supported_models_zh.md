---
title: 支持的模型
---
[](){ #supported-models }

vLLM 支持多种任务的 [生成模型](./generative_models.md) 和 [池化模型](./pooling_models.md)。
如果一个模型支持多个任务，可以通过 `--task` 参数设置任务类型。

对于每种任务，我们列出了 vLLM 中已实现的模型架构。
在每个架构旁边，我们还包括了一些使用该架构的热门模型。

## 模型实现

### vLLM

如果 vLLM 原生支持某个模型，其实现可以在 <gh-file:vllm/model_executor/models> 中找到。

这些模型是我们列在 [支持的文本模型][supported-text-models] 和 [支持的多模态模型][supported-mm-models] 中的模型。

[](){ #transformers-backend }

### Transformers

vLLM 还支持 Transformers 中可用的模型实现。目前并非所有模型都支持，但大多数解码器语言模型都受支持，视觉语言模型的支持也在计划中！

要检查模型后端是否为 Transformers，可以简单地执行以下代码：

```python
from vllm import LLM
llm = LLM(model=..., task="generate")  # 你的模型名称或路径
llm.apply_model(lambda model: print(type(model)))
```

如果输出为 `TransformersForCausalLM`，则表示该模型基于 Transformers！

!!! tip
    你可以通过设置 `model_impl="transformers"`（用于 [离线推理][offline-inference]）或 `--model-impl transformers`（用于 [OpenAI 兼容服务器][openai-compatible-server]）来强制使用 `TransformersForCausalLM`。

!!! note
    vLLM 可能无法完全优化 Transformers 实现，因此与 vLLM 原生模型相比，使用 Transformers 模型时性能可能会下降。

#### 自定义模型

如果一个模型既不被 vLLM 原生支持，也不被 Transformers 支持，它仍然可以在 vLLM 中使用！

要使模型与 vLLM 的 Transformers 后端兼容，模型必须：

- 是一个与 Transformers 兼容的自定义模型（参见 [Transformers - 自定义模型](https://huggingface.co/docs/transformers/en/custom_models)）：
    * 模型目录必须具有正确的结构（例如，包含 `config.json` 文件）。
    * `config.json` 必须包含 `auto_map.AutoModel`。
- 是一个与 vLLM 的 Transformers 后端兼容的模型（参见 [编写自定义模型][writing-custom-models]）：
    * 自定义应在基础模型中完成（例如，在 `MyModel` 中，而不是 `MyModelForCausalLM` 中）。

如果兼容模型：

- 在 Hugging Face 模型中心，只需为 [离线推理][offline-inference] 设置 `trust_remote_code=True`，或为 [OpenAI 兼容服务器][openai-compatible-server] 设置 `--trust-remote-code`。
- 在本地目录中，只需将目录路径传递给 `model=<MODEL_DIR>`（用于 [离线推理][offline-inference]）或 `vllm serve <MODEL_DIR>`（用于 [OpenAI 兼容服务器][openai-compatible-server]）。

这意味着，通过 vLLM 的 Transformers 后端，新模型可以在 Transformers 或 vLLM 正式支持之前使用！

[](){ #writing-custom-models }

#### 编写自定义模型

本节详细介绍了如何对与 Transformers 兼容的自定义模型进行必要修改，使其与 vLLM 的 Transformers 后端兼容。（我们假设已经创建了一个与 Transformers 兼容的自定义模型，参见 [Transformers - 自定义模型](https://huggingface.co/docs/transformers/en/custom_models)）。

要使模型与 Transformers 后端兼容，需要：

1. 从 `MyModel` 到 `MyAttention` 的所有模块都传递 `kwargs`。
2. `MyAttention` 必须使用 `ALL_ATTENTION_FUNCTIONS` 调用注意力机制。
3. `MyModel` 必须包含 `_supports_attention_backend = True`。

```python title="modeling_my_model.py"

from transformers import PreTrainedModel
from torch import nn

class MyAttention(nn.Module):

    def forward(self, hidden_states, **kwargs):
        ...
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            **kwargs,
        )
        ...

class MyModel(PreTrainedModel):
    _supports_attention_backend = True
```

以下是加载此模型时后台发生的事情：

1. 加载配置。
2. 从配置中的 `auto_map` 加载 `MyModel` Python 类，并检查模型是否 `is_backend_compatible()`。
3. 将 `MyModel` 加载到 `TransformersForCausalLM` 中（参见 <gh-file:vllm/model_executor/models/transformers.py>），它会设置 `self.config._attn_implementation = "vllm"`，以使用 vLLM 的注意力层。

就这样！

要使你的模型与 vLLM 的张量并行和/或流水线并行功能兼容，你必须在模型的配置类中添加 `base_model_tp_plan` 和/或 `base_model_pp_plan`：

```python title="configuration_my_model.py"

from transformers import PretrainedConfig

class MyConfig(PretrainedConfig):
    base_model_tp_plan = {
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }
```

- `base_model_tp_plan` 是一个 `dict`，将完全限定的层名称模式映射到张量并行样式（目前仅支持 `"colwise"` 和 `"rowwise"`）。
- `base_model_pp_plan` 是一个 `dict`，将直接子层名称映射到 `tuple` 的 `list` 的 `str`：
    * 仅需为不在所有流水线阶段的层执行此操作
    * vLLM 假设只有一个 `nn.ModuleList`，它分布在流水线阶段上
    * `tuple` 的第一个元素的 `list` 包含输入参数的名称
    * `tuple` 的最后一个元素的 `list` 包含模型代码中层输出的变量名称

## 加载模型

### Hugging Face 模型中心

默认情况下，vLLM 从 [Hugging Face (HF) 模型中心](https://huggingface.co/models) 加载模型。要更改模型的下载路径，可以设置 `HF_HOME` 环境变量；有关更多详细信息，请参阅 [官方文档](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hfhome)。

要确定给定的模型是否被原生支持，可以检查 HF 仓库中的 `config.json` 文件。
如果 `"architectures"` 字段包含以下列出的模型架构，则该模型应被原生支持。

模型不_需要_被原生支持即可在 vLLM 中使用。
[Transformers 后端][transformers-backend] 使你能够直接使用模型的 Transformers 实现（甚至包括 Hugging Face 模型中心上的远程代码！）。

!!! tip
    检查模型是否真正受支持的最简单方法是在运行时运行以下程序：

    ```python
    from vllm import LLM

    # 仅适用于生成模型（task=generate）
    llm = LLM(model=..., task="generate")  # 你的模型名称或路径
    output = llm.generate("你好，我的名字是")
    print(output)

    # 仅适用于池化模型（task={embed,classify,reward,score}）
    llm = LLM(model=..., task="embed")  # 你的模型名称或路径
    output = llm.encode("你好，我的名字是")
    print(output)
    ```

    如果 vLLM 成功返回文本（对于生成模型）或隐藏状态（对于池化模型），则表明你的模型受支持。

否则，请参阅 [添加新模型][new-model] 以获取在 vLLM 中实现模型的说明。
或者，你可以在 [GitHub 上提出问题](https://github.com/vllm-project/vllm/issues/new/choose) 请求 vLLM 支持。

#### 下载模型

如果你愿意，可以使用 Hugging Face CLI 来 [下载模型](https://huggingface.co/docs/huggingface_hub/guides/cli#huggingface-cli-download) 或模型仓库中的特定文件：

```console
# 下载模型
huggingface-cli download HuggingFaceH4/zephyr-7b-beta

# 指定自定义缓存目录
huggingface-cli download HuggingFaceH4/zephyr-7b-beta --cache-dir ./path/to/cache

# 从模型仓库下载特定文件
huggingface-cli download HuggingFaceH4/zephyr-7b-beta eval_results.json
```

#### 列出已下载的模型

使用 Hugging Face CLI 来 [管理本地缓存中的模型](https://huggingface.co/docs/huggingface_hub/guides/manage-cache#scan-your-cache)：

```console
# 列出缓存的模型
huggingface-cli scan-cache

# 显示详细（冗长）输出
huggingface-cli scan-cache -v

# 指定自定义缓存目录
huggingface-cli scan-cache --dir ~/.cache/huggingface/hub
```

#### 删除缓存的模型

使用 Hugging Face CLI 交互式地 [删除已下载的模型](https://huggingface.co/docs/huggingface_hub/guides/manage-cache#clean-your-cache) 从缓存中：

```console
# `delete-cache` 命令需要额外的依赖项才能使用 TUI。
# 请运行 `pip install huggingface_hub[cli]` 安装它们。

# 启动交互式 TUI 以选择要删除的模型
$ huggingface-cli delete-cache
? 选择要删除的修订版本：已选择 1 个修订版本，占用 438.9M。
  ○ 不选择以下任何内容（如果选择，则不会删除任何内容）。
模型 BAAI/bge-base-en-v1.5（438.9M，使用于 1 周前）
❯ ◉ a5beb1e3: main # 1 周前修改

模型 BAAI/bge-large-en-v1.5（1.3G，使用于 1 周前）
  ○ d4aa6901: main # 1 周前修改

模型 BAAI/bge-reranker-base（1.1G，使用于 4 周前）
  ○ 2cfc18c9: main # 4 周前修改

按 <space> 选择，按 <enter> 确认，按 <ctrl+c> 退出而不进行修改。

# 选择后需要确认
? 选择要删除的修订版本：已选择 1 个修订版本。
? 已选择 1 个修订版本，占用 438.9M。确认删除？是
开始删除。
完成。删除了 1 个仓库和 0 个修订版本，总计 438.9M。
```

#### 使用代理

以下是从 Hugging Face 加载/下载模型时使用代理的一些提示：

- 为你的会话全局设置代理（或在 profile 文件中设置）：

```shell
export http_proxy=http://your.proxy.server:port
export https_proxy=http://your.proxy.server:port
```

- 仅为当前命令设置代理：

```shell
https_proxy=http://your.proxy.server:port huggingface-cli download <model_name>

# 或直接使用 vllm 命令
https_proxy=http://your.proxy.server:port vllm serve <model_name> --disable-log-requests
```

- 在 Python 解释器中设置代理：

```python
import os

os.environ['http_proxy'] = 'http://your.proxy.server:port'
os.environ['https_proxy'] = 'http://your.proxy.server:port'
```

### ModelScope

要使用 [ModelScope](https://www.modelscope.cn) 的模型而不是 Hugging Face 模型中心，请设置环境变量：

```shell
export VLLM_USE_MODELSCOPE=True
```

并使用 `trust_remote_code=True`。

```python
from vllm import LLM

llm = LLM(model=..., revision=..., task=..., trust_remote_code=True)

# 仅适用于生成模型（task=generate）
output = llm.generate("你好，我的名字是")
print(output)

# 仅适用于池化模型（task={embed,classify,reward,score}）
output = llm.encode("你好，我的名字是")
print(output)
```

[](){ #feature-status-legend }

## 功能状态图例

- ✅︎ 表示该模型支持该功能。

- 🚧 表示该功能已计划但尚未支持。

- ⚠️ 表示该功能可用，但可能存在已知问题或限制。

[](){ #supported-text-models }

## 纯文本语言模型列表

### 生成模型

有关如何使用生成模型的更多信息，请参见 [此页面][generative-models]。

#### 文本生成

使用 `--task generate` 指定。

| 架构                                              | 模型                                                | 示例 HF 模型                                                                                                                                                                | [LoRA][lora-adapter]   | [PP][distributed-serving]   |
|---------------------------------------------------|-----------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------|-----------------------------|
| `AquilaForCausalLM`                               | Aquila, Aquila2                                     | `BAAI/Aquila-7B`, `BAAI/AquilaChat-7B` 等                                                                                                                                    | ✅︎                     | ✅︎                          |
| `ArcticForCausalLM`                               | Arctic                                              | `Snowflake/snowflake-arctic-base`, `Snowflake/snowflake-arctic-instruct` 等                                                                                                  |                        | ✅︎                          |
| `BaiChuanForCausalLM`                             | Baichuan2, Baichuan                                 | `baichuan-inc/Baichuan2-13B-Chat`, `baichuan-inc/Baichuan-7B` 等                                                                                                             | ✅︎                     | ✅︎                          |
| `BambaForCausalLM`                                | Bamba                                               | `ibm-ai-platform/Bamba-9B-fp8`, `ibm-ai-platform/Bamba-9B`                                                                                                                   | ✅︎                     | ✅︎                          |
| `BloomForCausalLM`                                | BLOOM, BLOOMZ, BLOOMChat                            | `bigscience/bloom`, `bigscience/bloomz` 等                                                                                                                                   |                        | ✅︎                          |
| `BartForConditionalGeneration`                    | BART                                                | `facebook/bart-base`, `facebook/bart-large-cnn` 等                                                                                                                           |                        |                             |
| `ChatGLMModel`, `ChatGLMForConditionalGeneration` | ChatGLM                                             | `THUDM/chatglm2-6b`, `THUDM/chatglm3-6b`, `ShieldLM-6B-chatglm3` 等                                                                                                          | ✅︎                     | ✅︎                          |
| `CohereForCausalLM`, `Cohere2ForCausalLM`         | Command-R                                           | `CohereForAI/c4ai-command-r-v01`, `CohereForAI/c4ai-command-r7b-12-2024` 等                                                                                                  | ✅︎                     | ✅︎                          |
| `DbrxForCausalLM`                                 | DBRX                                                | `databricks/dbrx-base`, `databricks/dbrx-instruct` 等                                                                                                                        |                        | ✅︎                          |
| `DeciLMForCausalLM`                               | DeciLM                                              | `nvidia/Llama-3_3-Nemotron-Super-49B-v1` 等                                                                                                                                  | ✅︎                     | ✅︎                          |
| `DeepseekForCausalLM`                             | DeepSeek                                            | `deepseek-ai/deepseek-llm-67b-base`, `deepseek-ai/deepseek-llm-7b-chat` 等                                                                                                   |                        | ✅︎                          |
| `DeepseekV2ForCausalLM`                           | DeepSeek-V2                                         | `deepseek-ai/DeepSeek-V2`, `deepseek-ai/DeepSeek-V2-Chat` 等                                                                                                                 |                        | ✅︎                          |
| `DeepseekV3ForCausalLM`                           | DeepSeek-V3                                         | `deepseek-ai/DeepSeek-V3-Base`, `deepseek-ai/DeepSeek-V3` 等                                                                                                                 |                        | ✅︎                          |
| `ExaoneForCausalLM`                               | EXAONE-3                                            | `LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct` 等                                                                                                                                    | ✅︎                     | ✅︎                          |
| `FalconForCausalLM`                               | Falcon                                              | `tiiuae/falcon-7b`, `tiiuae/falcon-40b`, `tiiuae/falcon-rw-7b` 等                                                                                                            |                        | ✅︎                          |
| `FalconMambaForCausalLM`                          | FalconMamba                                         | `tiiuae/falcon-mamba-7b`, `tiiuae/falcon-mamba-7b-instruct` 等                                                                                                               |                        | ✅︎                          |
| `FalconH1ForCausalLM`                             | Falcon-H1                                           | `tiiuae/Falcon-H1-34B-Base`, `tiiuae/Falcon-H1-34B-Instruct` 等                                                                                                              | ✅︎                     | ✅︎                          |
| `GemmaForCausalLM`                                | Gemma                                               | `google/gemma-2b`, `google/gemma-1.1-2b-it` 等                                                                                                                               | ✅︎                     | ✅︎                          |
| `Gemma2ForCausalLM`                               | Gemma 2                                             | `google/gemma-2-9b`, `google/gemma-2-27b` 等                                                                                                                                 | ✅︎                     | ✅︎                          |
| `Gemma3ForCausalLM`                               | Gemma 3                                             | `google/gemma-3-1b-it` 等                                                                                                                                                    | ✅︎                     | ✅︎                          |
| `GlmForCausalLM`                                  | GLM-4                                               | `THUDM/glm-4-9b-chat-hf` 等                                                                                                                                                  | ✅︎                     | ✅︎                          |
| `Glm4ForCausalLM`                                 | GLM-4-0414                                          | `THUDM/GLM-4-32B-0414` 等                                                                                                                                                    | ✅︎                     | ✅︎                          |
| `GPT2LMHeadModel`                                 | GPT-2                                               | `gpt2`, `gpt2-xl` 等                                                                                                                                                         |                        | ✅︎                          |
| `GPTBigCodeForCausalLM`                           | StarCoder, SantaCoder, WizardCoder                  | `bigcode/starcoder`, `bigcode/gpt_bigcode-santacoder`, `WizardLM/WizardCoder-15B-V1.0` 等                                                                                    | ✅︎                     | ✅︎                          |
| `GPTJForCausalLM`                                 | GPT-J                                               | `EleutherAI/gpt-j-6b`, `nomic-ai/gpt4all-j` 等                                                                                                                               |                        | ✅︎                          |
| `GPTNeoXForCausalLM`                              | GPT-NeoX, Pythia, OpenAssistant, Dolly V2, StableLM | `EleutherAI/gpt-neox-20b`, `EleutherAI/pythia-12b`, `OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5`, `databricks/dolly-v2-12b`, `stabilityai/stablelm-tuned-alpha-7b` 等   |                        | ✅︎                          |
| `GraniteForCausalLM`                              | Granite 3.0, Granite 3.1, PowerLM                   | `ibm-granite/granite-3.0-2b-base`, `ibm-granite/granite-3.1-8b-instruct`, `ibm/PowerLM-3b` 等                                                                               | ✅︎                     | ✅︎                          |
| `GraniteMoeForCausalLM`                           | Granite 3.0 MoE, PowerMoE                           | `ibm-granite/granite-3.0-1b-a400m-base`, `ibm-granite/granite-3.0-3b-a800m-instruct`, `ibm/PowerMoE-3b` 等                                                                 | ✅︎                     | ✅︎                          |
| `GraniteMoeHybridForCausalLM`                     | Granite 4.0 MoE Hybrid                              | `ibm-granite/granite-4.0-tiny-preview` 等                                                                                                                                    | ✅︎                     | ✅︎                          |
| `GraniteMoeSharedForCausalLM`                     | Granite MoE Shared                                  | `ibm-research/moe-7b-1b-active-shared-experts`（测试模型）                                                                                                                    | ✅︎                     | ✅︎                          |
| `GritLM`                                          | GritLM                                              | `parasail-ai/GritLM-7B-vllm`                                                                                                                                                 | ✅︎                     | ✅︎                          |
| `Grok1ModelForCausalLM`                           | Grok1                                               | `hpcai-tech/grok-1`                                                                                                                                                          | ✅︎                     | ✅︎                          |
| `InternLMForCausalLM`                             | InternLM                                            | `internlm/internlm-7b`, `internlm/internlm-chat-7b` 等                                                                                                                       | ✅︎                     | ✅︎                          |
| `InternLM2ForCausalLM`                            | InternLM2                                           | `internlm/internlm2-7b`, `internlm/internlm2-chat-7b` 等                                                                                                                     | ✅︎                     | ✅︎                          |
| `InternLM3ForCausalLM`                            | InternLM3                                           | `internlm/internlm3-8b-instruct` 等                                                                                                                                          | ✅︎                     | ✅︎                          |
| `JAISLMHeadModel`                                 | Jais                                                | `inceptionai/jais-13b`, `inceptionai/jais-13b-chat`, `inceptionai/jais-30b-v3`, `inceptionai/jais-30b-chat-v3` 等                                                           |                        | ✅︎                          |
| `JambaForCausalLM`                                | Jamba                                               | `ai21labs/AI21-Jamba-1.5-Large`, `ai21labs/AI21-Jamba-1.5-Mini`, `ai21labs/Jamba-v0.1` 等                                                                                   | ✅︎                     | ✅︎                          |
| `LlamaForCausalLM`                                | Llama 3.1, Llama 3, Llama 2, LLaMA, Yi              | `meta-llama/Meta-Llama-3.1-405B-Instruct`, `meta-llama/Meta-Llama-3.1-70B`, `meta-llama/Meta-Llama-3-70B-Instruct`, `meta-llama/Llama-2-70b-hf`, `01-ai/Yi-34B` 等         | ✅︎                     | ✅︎                          |
| `MambaForCausalLM`                                | Mamba                                               | `state-spaces/mamba-130m-hf`, `state-spaces/mamba-790m-hf`, `state-spaces/mamba-2.8b-hf` 等                                                                                  |                        | ✅︎                          |
| `MiniCPMForCausalLM`                              | MiniCPM                                             | `openbmb/MiniCPM-2B-sft-bf16`, `openbmb/MiniCPM-2B-dpo-bf16`, `openbmb/MiniCPM-S-1B-sft` 等                                                                                 | ✅︎                     | ✅︎                          |
| `MiniCPM3ForCausalLM`                             | MiniCPM3                                            | `openbmb/MiniCPM3-4B` 等                                                                                                                                                    | ✅︎                     | ✅︎                          |
| `MistralForCausalLM`                              | Mistral, Mistral-Instruct                           | `mistralai/Mistral-7B-v0.1`, `mistralai/Mistral-7B-Instruct-v0.1` 等                                                                                                        | ✅︎                     | ✅︎                          |
| `MixtralForCausalLM`                              | Mixtral-8x7B, Mixtral-8x7B-Instruct                 | `mistralai/Mixtral-8x7B-v0.1`, `mistralai/Mixtral-8x7B-Instruct-v0.1`, `mistral-community/Mixtral-8x22B-v0.1` 等                                                              | ✅︎                     | ✅︎                          |
| `MPTForCausalLM`                                  | MPT, MPT-Instruct, MPT-Chat, MPT-StoryWriter                                 | `mosaicml/mpt-7b`, `mosaicml/mpt-7b-storywriter`, `mosaicml/mpt-30b` 等                                                                                                     |                        | ✅︎                          |
| `NemotronForCausalLM`                             | Nemotron-3, Nemotron-4, Minitron                    | `nvidia/Minitron-8B-Base`, `mgoin/Nemotron-4-340B-Base-hf-FP8` 等                                                                                                            | ✅︎                     | ✅︎                          |
| `NemotronHForCausalLM`                            | Nemotron-H                                          | `nvidia/Nemotron-H-8B-Base-8K`, `nvidia/Nemotron-H-47B-Base-8K`, `nvidia/Nemotron-H-56B-Base-8K` 等                                                                         | ✅︎                     | ✅︎                          |
| `OLMoForCausalLM`                                 | OLMo                                                | `allenai/OLMo-1B-hf`, `allenai/OLMo-7B-hf` 等                                                                                                                                |                        | ✅︎                          |
| `OLMo2ForCausalLM`                                | OLMo2                                               | `allenai/OLMo-2-0425-1B` 等                                                                 |                        | ✅︎                          |
| `OLMoEForCausalLM`                                | OLMoE                                               | `allenai/OLMoE-1B-7B-0924`, `allenai/OLMoE-1B-7B-0924-Instruct` 等                                                                                                           |                        | ✅︎                          |
| `OPTForCausalLM`                                  | OPT, OPT-IML                                        | `facebook/opt-66b`, `facebook/opt-iml-max-30b` 等                                                                                                                           |                        | ✅︎                          |
| `OrionForCausalLM`                                | Orion                                               | `OrionStarAI/Orion-14B-Base`, `OrionStarAI/Orion-14B-Chat` 等                                                                                                                |                        | ✅︎                                                                                  |
| `PhiForCausalLM`                                 | Phi                                                 | `microsoft/phi-1_5`, `microsoft/phi-2` 等                                                                                                                                    | ✅︎                     | ✅︎                                                                                  |
| `Phi3ForCausalLM`                                 | Phi-4, Phi-3                                        | `microsoft/Phi-4-mini-instruct`, `microsoft/Phi-4`, `microsoft/Phi-3-mini-4k-instruct`, `microsoft/Phi-3-mini-128k-instruct`, `microsoft/Phi-3-medium-128k-instruct` 等   | ✅︎                     | ✅︎                                                                 |
| `Phi3SmallForCausalLM`                            | Phi-3-Small                                         | `microsoft/Phi-3-small-8k-instruct`, `microsoft/Phi-3-small-128k-instruct` 等                                                                                                |                        | ✅                                                                               |
| `PhiMoEForCausalLM`                               | Phi-3.5-MoE                                         | `microsoft/Phi-3.5-MoE-instruct` 等                                                                                                                                        | ✅︎                                                                                     | ✅︎                                                                 |
| `PersimmonForCausalLM`                            | Persimmon                                           | `adept/persimmon-8b-base`, `adept/persimmon-8b-chat` 等                                                                                                                      |                        | ✅                                                                               |
| `Plamo2ForCausalLM`                               | PLaMo2                                               | `pfnet/plamo-2-1b`, `pfnet/plamo-2-8b` 等                                                                                                                                       |                        |                                                                                             |
| `QWenLMHeadModel`                                 | Qwen                                                | `Qwen/Qwen-7B`, `Qwen/Qwen-7B-Chat` 等                                                                                                                                    | ✅︎                     | ✅︎                          |
| `Qwen2ForCausalLM`                                | QwQ, Qwen2                                          | `Qwen/QwQ-32B-Preview`, `Qwen/Qwen2-7B-Instruct`, `Qwen/Qwen2-7B` 等                                                                                                       | ✅︎                     | ✅︎                          |
| `Qwen2MoeForCausalLM`                             | Qwen2MoE                                            | `Qwen/Qwen1.5-MoE-A2.7B`, `Qwen/Qwen1.5-MoE-A2.7B-Chat` 等                                                                                                                 |                        | ✅︎                          |
| `Qwen3ForCausalLM`                                | Qwen3                                               | `Qwen/Qwen3-8B` 等                                                                                                                                                         | ✅︎                     | ✅︎                          |
| `Qwen3MoeForCausalLM`                             | Qwen3MoE                                            | `Qwen/Qwen3-30B-A3B` 等                                                                                                                                                    |                        | ✅︎                          |
| `StableLmForCausalLM`                             | StableLM                                            | `stabilityai/stablelm-3b-4e1t`, `stabilityai/stablelm-base-alpha-7b-v2` 等                                                                                                 |                        |                             |
| `Starcoder2ForCausalLM`                           | Starcoder2                                          | `bigcode/starcoder2-3b`, `bigcode/starcoder2-7b`, `bigcode/starcoder2-15b` 等                                                                                              |                        | ✅︎                          |
| `SolarForCausalLM`                                | Solar Pro                                           | `upstage/solar-pro-preview-instruct` 等                                                                                                                                    | ✅︎                     | ✅︎                          |
| `TeleChat2ForCausalLM`                            | TeleChat2                                           | `Tele-AI/TeleChat2-3B`, `Tele-AI/TeleChat2-7B`, `Tele-AI/TeleChat2-35B` 等                                                                                                 | ✅︎                     | ✅︎                          |
| `TeleFLMForCausalLM`                              | TeleFLM                                             | `CofeAI/FLM-2-52B-Instruct-2407`, `CofeAI/Tele-FLM` 等                                                                                                                     | ✅︎                     | ✅︎                          |
| `XverseForCausalLM`                               | XVERSE                                              | `xverse/XVERSE-7B-Chat`, `xverse/XVERSE-13B-Chat`, `xverse/XVERSE-65B-Chat` 等                                                                                             | ✅︎                     | ✅︎                          |
| `MiniMaxText01ForCausalLM`                        | MiniMax-Text                                        | `MiniMaxAI/MiniMax-Text-01` 等                                                                                                                                            |                        |                             |
| `Zamba2ForCausalLM`                               | Zamba2                                              | `Zyphra/Zamba2-7B-instruct`, `Zyphra/Zamba2-2.7B-instruct`, `Zyphra/Zamba2-1.2B-instruct` 等                                                                               |                        |                             |

!!! note
    当前，vLLM 的 ROCm 版本仅支持 Mistral 和 Mixtral，上下文长度最多为 4096。

### 池化模型

有关如何使用池化模型的更多信息，请参见 [此页面](./pooling_models.md)。

!!! warning
    由于一些模型架构同时支持生成和池化任务，
    你应明确指定任务类型，以确保模型以池化模式而不是生成模式使用。

#### 文本嵌入

使用 `--task embed` 指定。

| 架构                                           | 模型                | 示例 HF 模型                                                                                                   | [LoRA][lora-adapter]   | [PP][distributed-serving]   |
|--------------------------------------------------------|---------------------|---------------------------------------------------------------------------------------------------------------------|------------------------|-----------------------------|
| `BertModel`                                            | 基于 BERT           | `BAAI/bge-base-en-v1.5`, `Snowflake/snowflake-arctic-embed-xs` 等                                                   |                        |                             |
| `Gemma2Model`                                          | 基于 Gemma 2        | `BAAI/bge-multilingual-gemma2` 等                                                                                   | ✅︎                     |                             |
| `GritLM`                                               | GritLM              | `parasail-ai/GritLM-7B-vllm`                                                                                        | ✅︎                     | ✅︎                          |
| `GteModel`                                             | Arctic-Embed-2.0-M  | `Snowflake/snowflake-arctic-embed-m-v2.0`                                                                           | ︎                      |                             |
| `GteNewModel`                                          | mGTE-TRM（见注释）  | `Alibaba-NLP/gte-multilingual-base` 等                                                                              | ︎                      | ︎                           |
| `ModernBertModel`                                      | 基于 ModernBERT     | `Alibaba-NLP/gte-modernbert-base` 等                                                                                | ︎                      | ︎                           |
| `NomicBertModel`                                       | Nomic BERT          | `nomic-ai/nomic-embed-text-v1`, `nomic-ai/nomic-embed-text-v2-moe`, `Snowflake/snowflake-arctic-embed-m-long` 等  | ︎                      | ︎                           |
| `LlamaModel`, `LlamaForCausalLM`, `MistralModel` 等    | 基于 Llama          | `intfloat/e5-mistral-7b-instruct` 等                                                                                | ✅︎                     | ✅︎                          |
| `Qwen2Model`, `Qwen2ForCausalLM`                       | 基于 Qwen2          | `ssmits/Qwen2-7B-Instruct-embed-base`（见注释），`Alibaba-NLP/gte-Qwen2-7B-instruct`（见注释）等                   | ✅︎                     | ✅︎                          |
| `RobertaModel`, `RobertaForMaskedLM`                   | 基于 RoBERTa        | `sentence-transformers/all-roberta-large-v1` 等                                                                     |                        |                             |

!!! note
    `ssmits/Qwen2-7B-Instruct-embed-base` 的 Sentence Transformers 配置定义不正确。
    你需要通过传递 `--override-pooler-config '{"pooling_type": "MEAN"}'` 手动设置均值池化。

!!! note
    对于 `Alibaba-NLP/gte-Qwen2-*`，你需要启用 `--trust-remote-code` 以加载正确的分词器。
    参见 [HF Transformers 上的相关问题](https://github.com/huggingface/transformers/issues/34882)。

!!! note
    `jinaai/jina-embeddings-v3` 通过 LoRA 支持多种任务，而 vLLM 目前仅通过合并 LoRA 权重临时支持文本匹配任务。

!!! note
    第二代 GTE 模型 mGTE-TRM）被命名为 `NewModel`。名称 `NewModel` 过于通用，你应设置 `--hf-overrides '{"architecture": ["GteNewModel"]}'` 以指定使用 `GteNewModel` 架构。

如果你的模型不在上述列表中，我们将尝试使用 [as_embedding_model][vllm.model_executor.models.adapters.as_embedding_model] 自动转换模型。默认情况下，将从最后一个标记的归一化隐藏状态中提取整个提示的嵌入。

#### 奖励建模

使用 `--task reward` 指定。

| 架构                      | 模型                  | 示例 HF 模型                                                               | [LoRA][lora-adapter]   | [PP][distributed-serving]   |
|---------------------------|-----------------------|------------------------------------------------------------------------|------------------------|-----------------------------|
| `InternLM2ForRewardModel`   | 基于 InternLM2         | `internlm/internlm2-1_8b-reward`, `internlm/internlm2-7b-reward` 等   | ✅︎                     | ✅︎                          |
| `LlamaForCausalLM`         | 基于 Llama            | `peiyi9979/math-shepherd-mistral-7b-prm` 等                       | ✅︎                     | ✅︎                          |
| `Qwen2ForRewardModel`      | 基于 Qwen2            | `Qwen/Qwen2.5-Math-RM-72B` 等                                       | ✅︎                     | ✅︎                          |

如果你的模型不在上述列表中，我们将尝试使用 [as_reward_model][vllm.model_executor.models.adapters.as_reward_model] 自动转换模型。默认情况下，我们直接返回每个标记的隐藏状态。

!!! warning
    对于像 `peiyi9979/math-shepherd-mistral-7b-prm` 这样的过程监督奖励模型，应明确设置池化配置，
    例如：`--override-pooler-config '{"pooling_type": "STEP", "step_tag_id": 123, "returned_token_ids": [456, 789]}'`。

#### 分类

使用 `--task classify` 指定。

| 架构                              | 模型       | 示例 HF 模型                             | [LoRA][lora-adapter]   | [PP][distributed-serving]   |
|-----------------------------------|------------|------------------------------------------|------------------------|-----------------------------|
| `JambaForSequenceClassification`  | Jamba      | `ai21labs/Jamba-tiny-reward-dev` 等      | ✅︎                     | ✅︎                          |

如果你的模型不在上述列表中，我们将尝试使用 [as_classification_model][vllm.model_executor.models.adapters.as_classification_model] 自动转换模型。默认情况下，从最后一个标记的 softmax 隐藏状态中提取类概率。

#### 句子对评分

使用 `--task score` 指定。

| 架构                                  | 模型                | 示例 HF 模型                                |
|---------------------------------------|---------------------|----------------------------------------------|
| `BertForSequenceClassification`       | 基于 BERT           | `cross-encoder/ms-marco-MiniLM-L-6-v2` 等   |
| `RobertaForSequenceClassification`    | 基于 RoBERTa        | `cross-encoder/quora-roberta-base` 等       |
| `XLMRobertaForSequenceClassification` | 基于 XLM-RoBERTa    | `BAAI/bge-reranker-v2-m3` 等                |

[](){ #supported-mm-models }

## 多模态语言模型列表

以下模态根据模型支持：

- **T** 文本
- **I** 图像
- **V** 视频
- **A** 音频

任何通过 `+` 连接的模态组合都受支持。

- 例如：`T + I` 表示模型支持纯文本、纯图像以及文本与图像的输入。

另一方面，用 `/` 分隔的模态是互斥的。

- 例如：`T / I` 表示模型支持纯文本和纯图像输入，但不支持文本与图像的输入。

有关如何向模型传递多模态输入的信息，请参见 [此页面][multimodal-inputs]。

!!! warning
    **要在 vLLM V0 中启用每个文本提示的多个多模态输入，你需要设置** `limit_mm_per_prompt`（离线推理）
    或 `--mlimit-mm-per-prompt`（在线服务）。例如，要启用每个文本提示最多传递 4 张图像：

    离线推理：

    ```python
    from vllm import LLM

    llm = LLM(
        model="Qwen/Qwen2-VL-7B-Instruct",
        limit_mm_per_prompt={"image": 4},
    )
    ```

    在线服务：

    ```bash
    vllm serve Qwen/Qwen2-VL-7B-Instruct --limit-mm-per-prompt '{"image":4}'
    ```

    **如果你使用的是 vLLM V1，则不再需要此设置。**

!!! note
    vLLM 当前仅支持对多模态模型的语言骨干添加 LoRA。

### 生成模型

有关如何使用生成模型的更多信息，请参见 [此页面][generative-models]。

#### 文本生成

使用 `--task generate` 指定。

| 架构                                          | 模型                                                                     | 输入                                                                | 示例 HF 模型                                                                                                   | [LoRA][lora-adapter]   | [PP][distributed-serving] | [V1](gh-issue:8779)   |
|-----------------------------------------------|--------------------------------------------------------------------------|---------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|------------------------|---------------------------|-----------------------|
| `AriaForConditionalGeneration`                | Aria                                                                     | T + I<sup>+</sup>                                                   | `rhymes-ai/Aria`                                                                                               |                        |                           | ✅           |
| `AyaVisionForConditionalGeneration`           | Aya Vision                                                               | T + I<sup>+</sup>                                                   | `CohereForAI/aya-vision-8b`, `CohereForAI/aya-vision-32b` 等                                                    |                        | ✅                          | ✅           |
| `Blip2ForConditionalGeneration`               | BLIP-2                                                                   | T + I<sup>E</sup>                                                   | `Salesforce/blip2-opt-2.7b`, `Salesforce/blip2-opt-6.7b` 等                                                   |                        | ✅                          | ✅          |
| `ChameleonForConditionalGeneration`           | Chameleon                                                                | T + I                                                               | `facebook/chameleon-7b` 等                                                                                     |                        | ✅                          | ✅           |
| `DeepseekVLV2ForCausalLM`<sup>^</sup>         | DeepSeek-VL2                                                             | T + I<sup>+</sup>                                                   | `deepseek-ai/deepseek-vl2-tiny`, `deepseek-ai/deepseek-vl2-small`, `deepseek-ai/deepseek-vl2` 等             |                        | ✅                           | ✅           |
| `Florence2ForConditionalGeneration`           | Florence-2                                                                | T + I                                                               | `microsoft/Florence-2-base`, `microsoft/Florence-2-large` 等                                    |                        |                             |                       |
| `FuyuForCausalLM`                             | Fuyu                                                                     | T + I                                                               | `adept/fuyu-8b` 等                                                                                             |                        | ✅                           | ✅          |
| `Gemma3ForConditionalGeneration`              | Gemma 3                                                                  | T + I<sup>+</sup>                                                   | `google/gemma-3-4b-it`, `google/gemma-3-27b-it` 等                                                             | ✅                      | ✅                          | ⚠️          |
| `GLM4VForCausalLM`<sup>^</sup>                | GLM-4V                                                                   | T + I                                                               | `THUDM/glm-4v-9b`, `THUDM/cogagent-9b-20241220` 等                                                             | ✅                      | ✅                          | ✅           |
| `GraniteSpeechForConditionalGeneration`       | Granite Speech                                                           | T + A                                                               | `ibm-granite/granite-speech-3.3-8b`                                                                            | ✅                      | ✅                          | ✅           |
| `H2OVLChatModel`                              | H2OVL                                                                    | T + I<sup>E+</sup>                                                  | `h2oai/h2ovl-mississippi-800m`, `h2oai/h2ovl-mississippi-2b` 等                                               |                        | ✅                          | ✅\*         |
| `Idefics3ForConditionalGeneration`            | Idefics3                                                                 | T + I                                                               | `HuggingFaceM4/Idefics3-8B-Llama3` 等                                                                          | ✅                      |                           | ✅           |
| `InternVLChatModel`                           | InternVL 3.0, InternVideo 2.5, InternVL 2.5, Mono-InternVL, InternVL 2.0 | T + I<sup>E+</sup> + (V<sup>E+</sup>)                               | `OpenGVLab/InternVL3-9B`, `OpenGVLab/InternVideo2_5_Chat_8B`, `OpenGVLab/InternVL2_5-4B`, `OpenGVLab/Mono-InternVL-2B`, `OpenGVLab/InternVL2-4B` 等 | ✅                      | ✅                          | ✅           |
| `KimiVLForConditionalGeneration`              | Kimi-VL-A3B-Instruct, Kimi-VL-A3B-Thinking                               | T + I<sup>+</sup>                                                   | `moonshotai/Kimi-VL-A3B-Instruct`, `moonshotai/Kimi-VL-A3B-Thinking`                                           |                        |                           | ✅           |
| `Llama4ForConditionalGeneration`              | Llama 4                                                                  | T + I<sup>+</sup>                                                   | `meta-llama/Llama-4-Scout-17B-16E-Instruct`, `meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8`, `meta-llama/Llama-4-Maverick-17B-128E-Instruct` 等 |                        | ✅                          | ✅           |
| `LlavaForConditionalGeneration`               | LLaVA-1.5                                                                | T + I<sup>E+</sup>                                                  | `llava-hf/llava-1.5-7b-hf`, `TIGER-Lab/Mantis-8B-siglip-llama3`（见注释）等                                  |                        | ✅                          | ✅           |
| `LlavaNextForConditionalGeneration`           | LLaVA-NeXT                                                               | T + I<sup>E+</sup>                                                  | `llava-hf/llava-v1.6-mistral-7b-hf`, `llava-hf/llava-v1.6-vicuna-7b-hf` 等                                     |                        | ✅                          | ✅           |
| `LlavaNextVideoForConditionalGeneration`      | LLaVA-NeXT-Video                                                         | T + V                                                               | `llava-hf/LLaVA-NeXT-Video-7B-hf` 等                                                                           |                        | ✅                          | ✅           |
| `LlavaOnevisionForConditionalGeneration`      | LLaVA-Onevision                                                          | T + I<sup>+</sup> + V<sup>+</sup>                                   | `llava-hf/llava-onevision-qwen2-7b-ov-hf`, `llava-hf/llava-onevision-qwen2-0.5b-ov-hf` 等                     |                        | ✅                          | ✅           |
| `MiniCPMO`                                    | MiniCPM-O                                                                | T + I<sup>E+</sup> + V<sup>E+</sup> + A<sup>E+</sup>                | `openbmb/MiniCPM-o-2_6` 等                                                                                     | ✅                      | ✅                          | ✅           |
| `MiniCPMV`                                    | MiniCPM-V                                                                | T + I<sup>E+</sup> + V<sup>E+</sup>                                 | `openbmb/MiniCPM-V-2`（见注释），`openbmb/MiniCPM-Llama3-V-2_5`, `openbmb/MiniCPM-V-2_6` 等                   | ✅                      |                           | ✅           |
| `MiniMaxVL01ForConditionalGeneration`         | MiniMax-VL                                                               | T + I<sup>E+</sup>                                                  | `MiniMaxAI/MiniMax-VL-01` 等                                                                                   |                        | ✅                          |                       |
| `Mistral3ForConditionalGeneration`            | Mistral3                                                                 | T + I<sup>+</sup>                                                   | `mistralai/Mistral-Small-3.1-24B-Instruct-2503` 等                                                             | ✅                      | ✅                          | ✅           |
| `MllamaForConditionalGeneration`              | Llama 3.2                                                                | T + I<sup>+</sup>                                                   | `meta-llama/Llama-3.2-90B-Vision-Instruct`, `meta-llama/Llama-3.2-11B-Vision` 等                               |                        |                           |                       |
| `MolmoForCausalLM`                            | Molmo                                                                    | T + I<sup>+</sup>                                                   | `allenai/Molmo-7B-D-0924`, `allenai/Molmo-7B-O-0924` 等                                                       | ✅                      | ✅                          | ✅           |
| `NVLM_D_Model`                                | NVLM-D 1.0                                                               | T + I<sup>+</sup>                                                   | `nvidia/NVLM-D-72B` 等                                                                                        |                        | ✅                          | ✅           |
| `Ovis`                                        | Ovis2, Ovis1.6                                                           | T + I<sup>+</sup>                                                   | `AIDC-AI/Ovis2-1B`, `AIDC-AI/Ovis1.6-Llama3.2-3B` 等                                                          |                        | ✅                          | ✅           |
| `PaliGemmaForConditionalGeneration`           | PaliGemma, PaliGemma 2                                                   | T + I<sup>E</sup>                                                   | `google/paligemma-3b-pt-224`, `google/paligemma-3b-mix-224`, `google/paligemma2-3b-ft-docci-448` 等           |                        | ✅                          | ⚠️           |
| `Phi3VForCausalLM`                            | Phi-3-Vision, Phi-3.5-Vision                                             | T + I<sup>E+</sup>                                                  | `microsoft/Phi-3-vision-128k-instruct`, `microsoft/Phi-3.5-vision-instruct` 等                                 |                        | ✅                          | ✅           |
| `Phi4MMForCausalLM`                           | Phi-4-multimodal                                                              | T + I<sup>+</sup> / T + A<sup>+</sup> / I<sup>+</sup> + A<sup>+</sup> | `microsoft/Phi-4-multimmodal-instruct` 等                                                          | ✅                      | ✅                                                                                  | ✅  |
| `PixtralForConditionalGeneration`             | Pixtral                                                                  | T + I<sup>+</sup>                                                   | `mistralai/Mistral-pixtral-12b` 等                                                                |                        | ✅                          | ✅           |
| `QwenVLForConditionalGeneration`<sup>^</sup>  | Qwen-VL                                                             | T + I<sup>E+</sup>                                                  | `Qwen/Qwen-VL`, `Qwen/Qwen-VL-Chat` 等                                                                           | ✅                      | ✅                          | ✅            |
| `Qwen2AudioForConditionalGeneration`          | Qwen2-Audio                                                              | T + A<sup>+</sup>                                                   | `Qwen/Qwen2-Audio-7B-Instruct`                                                                                   |                        | ✅                                                                         | ✅   |
| `Qwen2VLForConditionalGeneration`             | QVQ, Qwen2-VL                                                              | T + I<sup>E+</sup> + V<sup>E+</sup>           | `Qwen/QVQ-72B-Preview`, `Qwen/Qwen2-VL-7B-Instruct`, `Qwen/Qwen2-VL-72B-Instruct` 等                             | ✅                      | ✅                                 | ✅           |
| `Qwen2_5_VLForConditionalGeneration`          | Qwen2.5-VL                                                               |                                                             | `Qwen/Qwen2.5-VL-3B-Instruct`, `Qwen/Qwen2.5-VL-72B-Instruct` 等                                         | ✅                      | ✅                                                                         | ✅           |
| `Qwen2_5OmniThinkerForConditionalGeneration` | Qwen2.5-Omni                                                                     | T + I<sup>E+</sup> + V<sup>E+</sup> + A<sup>+</sup> | `Qwen/Qwen2.5-Omni-7B`                                                                                   |                        | ✅                                                                         | ✅\*         |
| `SkyworkR1VChatModel`                         | Skywork-R1V-38B                                                          | T + I                                                               | `Skywork/Skywork-R1V-38B`                                                                                        |                        | ✅                          | ✅           |
| `SmolVLMForConditionalGeneration`             | SmolVLM2                                                                 | T + I                                                               | `SmolVLM2-2.2B-Instruct`                                                                                         | ✅                      |                           | ✅           |
| `TarsierForConditionalGeneration`                | Tarsier                                                                  | T + I<sup>E+</sup>                                                  | `omni-search/Tarsier-7b`,`omni-search/Tarsier-34b`                                                               |                        | ✅                          | ✅           |

<sup>^</sup> 你需要通过 `--hf-overrides` 设置架构名称以匹配 vLLM 中的名称。  
    • 例如，要使用 DeepSeek-VL2 系列模型：  
      `--hf-overrides '{"architectures": ["DeepseekVLV2ForCausalLM"]}'`  
<sup>E</sup> 此模态可以输入预计算的嵌入。  
<sup>+</sup> 每个文本提示可以输入多个项目。

!!! warning
    V0 和 V1 都支持 `Gemma3ForConditionalGeneration` 的纯文本输入。
    然而，它们在处理文本 + 图像输入时存在差异：

    V0 正确实现了模型的注意力模式：
    - 对对应同一图像的图像标记使用双向注意力
    - 对其他标记使用因果注意力
    - 通过（朴素的）PyTorch SDPA 配合掩码张量实现
    - 注意：对于包含图像的长提示可能会使用大量内存

    V1 目前使用简化的注意力模式：
    - 对所有标记（包括图像标记）使用因果注意力
    - 生成合理的输出，但与原始模型的文本 + 图像输入的注意力模式不匹配，特别是当 `{"do_pan_and_scan": true}` 时
    - 未来将更新以支持正确的行为

    这种限制存在是因为模型的混合注意力模式（图像双向，其他因果）尚未被 vLLM 的注意力后端支持。

!!! note
    目前只有使用 Qwen2.5 文本骨干的 `InternVLChatModel`（如 `OpenGVLab/InternVL3-2B`，`OpenGVLab/InternVL2.5-1B` 等）支持视频输入。

!!! note
    `h2oai/h2ovl-mississippi-2b` 将在 V1 中可用，一旦我们支持头部大小为 80。

!!! note
    要使用 `TIGER-Lab/Mantis-8B-siglip-llama3`，你需要在运行 vLLM 时传递 `--hf_overrides '{"architectures": ["MantisForConditionalGeneration"]}'`。

!!! warning
    `AllenAI/Molmo-7B-D-0924` 的输出质量（特别是在对象定位任务中）在最近的更新中有所下降。

    为获得最佳结果，我们建议使用以下依赖版本（在 A10 和 L40 上测试）：

    ```text
    # 核心 vLLM 兼容依赖项，适用于 Molmo 精度设置（在 L40 上测试）
    torch==2.5.1
    torchvision==0.20.1
    transformers==4.48.1
    tokenizers==0.21.0
    tiktoken==0.7.0
    vllm==0.7.0

    # 可选但推荐用于提高性能和稳定性
    triton==3.1.0
    xformers==0.0.28.post3
    uvloop==0.21.0
    protobuf==5.29.3
    openai==1.60.2
    opencv-python-headless==4.11.0.86
    pillow==10.4.0

    # 已安装 FlashAttention（仅用于 float16）
    flash-attn>=2.5.6  # 在 float32 中未使用，但应记录
    ```

    **注意：** 确保你了解使用过时包的安全隐患。

!!! note
    官方的 `openbmb/MiniCPM-V-2` 尚不可用，因此我们需要暂时使用一个分支（`HwwwH/MiniCPM-V-2`）。
    有关更多详细信息，请参见：<gh-pr:4087#issuecomment-2250397630>

!!! warning
    我们的 PaliGemma 实现与 Gemma 3（见上文）在 V0 和 V1 上存在相同的问题。

!!! note
    要使用 Qwen2.5-Omni，你必须通过以下方式从源代码安装 Hugging Face Transformers 库：
    `pip install git+https://github.com/huggingface/transformers.git`。

    从视频预处理中读取音频目前在 V0 上受支持（但在 V1 上不受支持），因为 V1 尚未支持重叠模态。
    `--mm-processor-kwargs '{"use_audio_in_video": true}'`。

### 池化模型

有关如何使用池化模型的更多信息，请参见 [此页面](./pooling_models.md)。

!!! warning
    由于一些模型架构同时支持生成和池化任务，
    你应明确指定任务类型，以确保模型以池化模式而不是生成模式使用。

#### 文本嵌入

使用 `--task embed` 指定。

任何文本生成模型都可以通过传递 `--task embed` 转换为嵌入模型。

!!! note
    为获得最佳结果，你应使用专门训练为池化模型的模型。

下表列出了在 vLLM 中测试过的模型。

| 架构                                 | 模型                   | 输入     | 示例 HF 模型             | [LoRA][lora-adapter]   | [PP][distributed-serving]   |
|--------------------------------------|------------------------|----------|--------------------------|------------------------|-----------------------------|
| `LlavaNextForConditionalGeneration`  | 基于 LLaVA-NeXT        | T / I    | `royokong/e5-v`          |                        |                             |
| `Phi3VForCausalLM`                   | 基于 Phi-3-Vision      | T + I    | `TIGER-Lab/VLM2Vec-Full` | 🚧                      | ✅                           |

#### 转录

使用 `--task transcription` 指定。

专门为自动语音识别训练的 Speech2Text 模型。

| 架构           | 模型     | 示例 HF 模型       | [LoRA][lora-adapter]   | [PP][distributed-serving]   |
|----------------|----------|--------------------|------------------------|-----------------------------|

---

## 模型支持政策

在 vLLM，我们致力于促进第三方模型在我们生态系统中的集成和支持。我们的方法旨在平衡鲁棒性需求与支持广泛模型的实际限制。以下是我们管理第三方模型支持的方式：

1. **社区驱动支持**：我们鼓励社区为添加新模型做出贡献。当用户请求支持新模型时，我们欢迎社区的拉取请求（PR）。这些贡献主要基于其生成输出的合理性进行评估，而不是与现有实现（如 transformers）的严格一致性。**贡献号召：** 直接来自模型供应商的 PR 非常受欢迎！

2. **尽力保持一致性**：虽然我们旨在保持 vLLM 中实现的模型与其他框架（如 transformers）的一定一致性，但完全对齐并不总是可行的。加速技术和低精度计算等因素可能导致差异。我们承诺确保实现的模型功能正常并产生合理的结果。

    !!! tip
        比较 Hugging Face Transformers 的 `model.generate` 输出与 vLLM 的 `llm.generate` 输出时，请注意，前者会读取模型的生成配置文件（即 [generation_config.json](https://github.com/huggingface/transformers/blob/19dabe96362803fb0a9ae7073d03533966598b17/src/transformers/generation/utils.py#L1945)）并应用生成默认参数，而后者仅使用传递给函数的参数。比较输出时，确保所有采样参数相同。

3. **问题解决和模型更新**：我们鼓励用户报告他们在第三方模型中遇到的任何错误或问题。建议的修复应通过 PR 提交，并清楚说明问题及建议解决方案的理由。如果一个模型的修复影响另一个模型，我们依赖社区来强调和解决这些跨模型依赖。注意：对于错误修复 PR，通知原始作者以征求他们的反馈是良好的礼节。

4. **监控和更新**：对特定模型感兴趣的用户应监控这些模型的提交历史（例如，通过跟踪 main/vllm/model_executor/models 目录中的更改）。这种主动方法帮助用户了解可能影响他们使用的模型的更新和变化。

5. **选择性关注**：我们的资源主要用于具有重大用户兴趣和影响的模型。使用频率较低的模型可能获得的关注较少，我们依赖社区在这些模型的维护和改进中发挥更积极的作用。

通过这种方法，vLLM 营造了一个协作环境，核心开发团队和更广泛的社区共同为我们生态系统中支持的第三方模型的鲁棒性和多样性做出贡献。

请注意，作为推理引擎，vLLM 不会引入新模型。因此，在这方面，vLLM 支持的所有模型都是第三方模型。

我们对模型有以下测试级别：

1. **严格一致性**：我们比较模型在贪婪解码下的输出与 HuggingFace Transformers 库中模型的输出。这是最高级别的测试。请参阅 [模型测试](https://github.com/vllm-project/vllm/blob/main/tests/models) 以了解通过此测试的模型。
2. **输出合理性**：我们检查模型的输出是否合理和连贯，通过测量输出的困惑度和检查任何明显错误。这是一个较低级别的测试。
3. **运行时功能**：我们检查模型是否可以加载并无错误运行。这是最低级别的测试。请参阅 [功能测试](gh-dir:tests) 和 [示例](gh-dir:examples) 以了解通过此测试的模型。
4. **社区反馈**：我们依靠社区提供模型的反馈。如果模型出现故障或未按预期工作，我们鼓励用户提出问题报告或提交拉取请求修复。其余模型属于此类别。