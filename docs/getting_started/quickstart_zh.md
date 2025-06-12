---
title: 快速入门
---
[](){ #quickstart }

本指南将帮助您快速开始使用 vLLM 进行：

- [离线批量推理][quickstart-offline]
- [使用 OpenAI 兼容服务器进行在线服务][quickstart-online]

## 前提条件

- 操作系统：Linux
- Python：3.9 -- 3.12

## 安装

如果您使用的是 NVIDIA GPU，可以直接使用 [pip](https://pypi.org/project/vllm/) 安装 vLLM。

建议使用 [uv](https://docs.astral.sh/uv/)，一个非常快速的 Python 环境管理器，来创建和管理 Python 环境。请按照 [文档](https://docs.astral.sh/uv/#getting-started) 安装 `uv`。安装 `uv` 后，您可以使用以下命令创建一个新的 Python 环境并安装 vLLM：

```console
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install vllm --torch-backend=auto
```

`uv` 可以通过 `--torch-backend=auto`（或 `UV_TORCH_BACKEND=auto`）在运行时通过检查已安装的 CUDA 驱动版本 [自动选择适当的 PyTorch 索引](https://docs.astral.sh/uv/guides/integration/pytorch/#automatic-backend-selection)。要选择特定的后端（例如 `cu126`），请设置 `--torch-backend=cu126`（或 `UV_TORCH_BACKEND=cu126`）。

另一种便捷的方式是使用 `uv run` 配合 `--with [dependency]` 选项，这允许您运行诸如 `vllm serve` 的命令，而无需创建任何永久环境：

```console
uv run --with vllm vllm --help
```

您还可以使用 [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html) 来创建和管理 Python 环境。如果您希望在 conda 环境中管理 `uv`，可以通过 `pip` 安装 `uv`。

```console
conda create -n myenv python=3.12 -y
conda activate myenv
pip install --upgrade uv
uv pip install vllm --torch-backend=auto
```

!!! note
    对于更多细节和非 CUDA 平台，请参考 [此处][installation-index] 获取有关如何安装 vLLM 的具体说明。

[](){ #quickstart-offline }

## 离线批量推理

安装 vLLM 后，您可以开始为输入提示列表生成文本（即离线批量推理）。请参见示例脚本：<gh-file:examples/offline_inference/basic/basic.py>

该示例的第一行导入了 [LLM][vllm.LLM] 和 [SamplingParams][vllm.SamplingParams] 类：

- [LLM][vllm.LLM] 是使用 vLLM 引擎进行离线推理的主要类。
- [SamplingParams][vllm.SamplingParams] 指定了采样过程的参数。

```python
from vllm import LLM, SamplingParams
```

下一部分定义了输入提示列表和文本生成的采样参数。[采样温度](https://arxiv.org/html/2402.05201v1) 设置为 `0.8`，[核采样概率](https://en.wikipedia.org/wiki/Top-p_sampling) 设置为 `0.95`。您可以 [此处][sampling-params] 找到有关采样参数的更多信息。

!!! warning
    默认情况下，如果 Hugging Face 模型库中存在 `generation_config.json`，vLLM 将使用模型创建者推荐的采样参数。在大多数情况下，如果未指定 [SamplingParams][vllm.SamplingParams]，这将为您提供最佳的默认结果。

    但是，如果您更喜欢 vLLM 的默认采样参数，请在创建 [LLM][vllm.LLM] 实例时设置 `generation_config="vllm"`。

```python
prompts = [
    "你好，我的名字是",
    "美国总统是",
    "法国的首都是",
    "人工智能的未来是",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
```

[LLM][vllm.LLM] 类初始化 vLLM 引擎和 [OPT-125M 模型](https://arxiv.org/abs/2205.01068) 用于离线推理。支持的模型列表可以在 [此处][supported-models] 找到。

```python
llm = LLM(model="facebook/opt-125m")
```

!!! note
    默认情况下，vLLM 从 [Hugging Face](https://huggingface.co/) 下载模型。如果您想使用 [ModelScope](https://www.modelscope.cn) 的模型，请在初始化引擎之前设置环境变量 `VLLM_USE_MODELSCOPE`。

    ```shell
    export VLLM_USE_MODELSCOPE=True
    ```

现在是激动人心的部分！使用 `llm.generate` 生成输出。它将输入提示添加到 vLLM 引擎的等待队列，并执行 vLLM 引擎以高吞吐量生成输出。输出将作为 `RequestOutput` 对象列表返回，其中包括所有输出 token。

```python
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"提示: {prompt!r}, 生成的文本: {generated_text!r}")
```

[](){ #quickstart-online }

## OpenAI 兼容服务器

vLLM 可以部署为实现 OpenAI API 协议的服务器。这允许 vLLM 作为使用 OpenAI API 的应用程序的直接替换。默认情况下，服务器启动在 `http://localhost:8000`。您可以使用 `--host` 和 `--port` 参数指定地址。服务器目前一次托管一个模型，并实现了 [列出模型](https://platform.openai.com/docs/api-reference/models/list)、[创建聊天完成](https://platform.openai.com/docs/api-reference/chat/completions/create) 和 [创建完成](https://platform.openai.com/docs/api-reference/completions/create) 等端点。

运行以下命令以使用 [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) 模型启动 vLLM 服务器：

```console
vllm serve Qwen/Qwen2.5-1.5B-Instruct
```

!!! note
    默认情况下，服务器使用存储在分词器中的预定义聊天模板。您可以 [here][chat-template] 了解如何覆盖它。

!!! warning
    默认情况下，如果 Hugging Face 模型库中存在 `generation_config.json`，服务器将应用它。这意味着某些采样参数的默认值可能会被模型创建者推荐的值覆盖。

    要禁用此行为，请在启动服务器时传递 `--generation-config vllm`。

该服务器可以按照 OpenAI API 的格式进行查询。例如，列出模型：

```console
curl http://localhost:8000/v1/models
```

您可以通过传递 `--api-key` 参数或环境变量 `VLLM_API_KEY` 启用服务器检查请求头中的 API 密钥。

### 使用 vLLM 的 OpenAI 完成 API

一旦服务器启动，您可以使用输入提示查询模型：

```console
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "prompt": "旧金山是一个",
        "max_tokens": 7,
        "temperature": 0
    }'
```

由于此服务器与 OpenAI API 兼容，您可以将其作为使用 OpenAI API 的任何应用程序的直接替换。例如，另一种查询服务器的方式是通过 `openai` Python 包：

```python
from openai import OpenAI

# 修改 OpenAI 的 API 密钥和 API 基础地址以使用 vLLM 的 API 服务器。
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
completion = client.completions.create(model="Qwen/Qwen2.5-1.5B-Instruct",
                                      prompt="旧金山是一个")
print("完成结果:", completion)
```

更详细的客户端示例可以在此处找到：<gh-file:examples/online_serving/openai_completion_client.py>

### 使用 vLLM 的 OpenAI 聊天完成 API

vLLM 还设计支持 OpenAI 聊天完成 API。聊天界面是一种更动态、交互式的与模型通信方式，允许来回交流并存储在聊天历史中。这对于需要上下文或更详细解释的任务非常有用。

您可以使用 [创建聊天完成](https://platform.openai.com/docs/api-reference/chat/completions/create) 端点与模型交互：

```console
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "messages": [
            {"role": "system", "content": "你是一个有用的助手。"},
            {"role": "user", "content": "2020 年世界系列赛的冠军是谁？"}
        ]
    }'
```

或者，您可以使用 `openai` Python 包：

```python
from openai import OpenAI
# 设置 OpenAI 的 API 密钥和 API 基础地址以使用 vLLM 的 API 服务器。
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

chat_response = client.chat.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    messages=[
        {"role": "system", "content": "你是一个有用的助手。"},
        {"role": "user", "content": "告诉我一个笑话。"},
    ]
)
print("聊天响应:", chat_response)
```

## 关于注意力后端

目前，vLLM 支持多种后端以在不同平台和加速器架构上进行高效的注意力计算。它会自动选择与您的系统和模型规格兼容的最优后端。

如果需要，您也可以通过配置环境变量 `VLLM_ATTENTION_BACKEND` 手动设置您选择的后端，可选项包括：`FLASH_ATTN`、`FLASHINFER` 或 `XFORMERS`。

!!! warning
    目前没有包含 Flash Infer 的预构建 vLLM 轮子，因此您必须先在环境中安装它。请参阅 [Flash Infer 官方文档](https://docs.flashinfer.ai/) 或查看 <gh-file:docker/Dockerfile> 获取安装说明。